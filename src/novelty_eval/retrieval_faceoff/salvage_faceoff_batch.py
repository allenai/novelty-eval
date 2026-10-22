#!/usr/bin/env python3
"""salvage_faceoff_batch.py — rescue a cancelled/failed face-off batch and score it.

Why this exists: when an OpenAI batch is cancelled mid-flight it lands in a
terminal status of `cancelled`/`failed`, and both `retrieve_raw_batch_content`
and `web_search_novelty_judge --mode collect` refuse to touch it — they only
download on status == "completed". The partial output file is still there and
still paid for. This module downloads it anyway.

Scoring then has to be restricted, not just recomputed. `_recompute_pointwise_metrics`
injects any instance absent from a run as a *wrong* prediction (a deliberate choice,
so genuine batch drops can't silently inflate accuracy). On a cancelled batch that
turns every un-run idea into a fake error and drives the web-search judge's accuracy
to the floor. So the salvaged ideas become the keep set and everything else is
excluded outright — the same exclude-set mechanism subset_reports.py uses, just
with a narrower manifest.

Usage:
    python -m novelty_eval.retrieval_faceoff.salvage_faceoff_batch ROOT
    python -m ...retrieval_faceoff.salvage_faceoff_batch ROOT --batch-id batch_abc123
    python -m ...retrieval_faceoff.salvage_faceoff_batch ROOT --no-download --out salvage.md

Arguments:
    ROOT              run dir from `run_retrieval_faceoff.sh sample` (holds
                      manifest.json + web_search_batch_map.json)
    --config          face-off config, for the baseline sweep paths
                      (default: config.yaml next to this module)
    --batch-id        batch to download (default: from the newest batch_info_*.json)
    --no-download     score what's already on disk; skips the OpenAI call entirely
    --force-download  re-download even if batch_output_*.jsonl already exists
    --balance         downsample the majority class so POS == NEG (see below)
    --balance-seed    seed for that draw (default: the manifest's seed)
    --divergence      also write a per-instance divergence report over the same subset
    --divergence-out  dir for it (default: ROOT/salvaged)
    --out             write Markdown here instead of stdout

An idea counts as salvaged when its response parses to a self-judgement verdict —
the same bar `run_collect` uses before writing a cache entry. A pair counts as
salvaged only when *both* of its pointwise ideas are. All batch_output_*.jsonl files
in ROOT are read and unioned, so earlier partial collects and `resubmit` rounds add up.

A cancelled batch returns whatever happened to finish first, so the salvaged ideas
carry none of the 50/50 balance the sampler built in — accuracy on them is then not
comparable to a full run's. `--balance` restores it by randomly dropping majority-class
ideas until the two classes match. Note this throws data away and the result moves with
the seed; the F1 macro column is balance-insensitive and uses every idea, so prefer it
when you just want one number. Pairs are one POS + one NEG by construction, so the
pairwise table is balanced already and `--balance` leaves it untouched.

`--divergence` writes the keep set out as a narrowed manifest and hands it to
divergence_report.py, giving the per-instance disagreement table for exactly the ideas
being scored here. It lands in ROOT/salvaged/ rather than ROOT so the full-subset
subset_divergence.md written by `poll`/`recompute` isn't clobbered. The reference judge
has pointwise verdicts only, so the report comes out pointwise-only.

This reads batch outputs directly and never writes into the artifact dirs, so it is
safe to run against sweeps you don't want to disturb. It does not re-derive the
web-search judge's own predictions: that row comes from the self-judge artifact dir
under ROOT, which `run_retrieval_faceoff.sh recompute` produces.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils import LOGGER as LOGGER  # reassigned to a file logger in main() once ROOT is known
from logging_utils import setup_logger
from novelty_eval.analysis.artifacts import (
    _collect_artifact_dirs,
    _extract_instances_path,
    _extract_model_from_artifact_dir,
    _extract_mode_from_artifact_dir,
)
from novelty_eval.analysis.filtering import (
    _recompute_pairwise_metrics,
    _recompute_pointwise_metrics,
)
from novelty_eval.retrieval_faceoff.divergence_report import _pointwise_labels

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


# ── download ──────────────────────────────────────────────────────────────────

def download_batch(root: Path, batch_id: str | None, force: bool) -> None:
    """Download a batch's output + error files regardless of terminal status.

    Unlike retrieve_raw_batch_content, a `cancelled`/`failed`/`expired` batch is
    still downloaded as long as it has an output_file_id — that partial output is
    the whole point here.
    """
    from utils import OPENAI_CLIENT

    if not batch_id:
        infos = list(root.glob("batch_info_*.json"))
        if not infos:
            raise SystemExit(f"no batch_info_*.json under {root} — pass --batch-id explicitly.")
        info = json.loads(max(infos, key=lambda p: p.stat().st_mtime).read_text())
        if info.get("provider", "openai") != "openai":
            raise SystemExit(f"provider {info.get('provider')!r} not supported — this is an OpenAI salvage tool.")
        batch_id = info["batch_id"]

    out_path = root / f"batch_output_{batch_id}.jsonl"
    if out_path.exists() and not force:
        LOGGER.info(f"[download] {out_path.name} already present — skipping "
                    f"(use --force-download to refresh).")
        return

    batch = OPENAI_CLIENT.batches.retrieve(batch_id)
    LOGGER.info(f"[download] batch {batch_id} status={batch.status}")

    for file_id, path in ((batch.output_file_id, out_path),
                          (getattr(batch, "error_file_id", None), root / f"batch_errors_{batch_id}.jsonl")):
        if not file_id:
            continue
        text = OPENAI_CLIENT.files.content(file_id).text
        path.write_text(text)
        n = sum(1 for ln in text.splitlines() if ln.strip())
        LOGGER.info(f"[download] wrote {n} line(s) to {path.name}")

    if not batch.output_file_id:
        raise SystemExit(f"batch {batch_id} has no output file — nothing was salvageable.")


# ── salvage set ───────────────────────────────────────────────────────────────

def salvaged_problem_ids(root: Path, batch_map: dict) -> set[str]:
    """Problem ids whose batch response parses to a verdict, unioned over all outputs.

    Mirrors run_collect's acceptance test (a paper list without a verdict is not a
    usable entry) but skips grounding, so it needs no network access.
    """
    from novelty_eval.retrieval.web_search_novelty_judge import (
        _responses_body_text,
        parse_self_judgement,
    )

    outputs = sorted(root.glob("batch_output_*.jsonl"))
    if not outputs:
        raise SystemExit(f"no batch_output_*.jsonl under {root} — run without --no-download first.")

    ids: set[str] = set()
    n_lines = n_err = n_noverdict = 0
    for path in outputs:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            n_lines += 1
            obj = json.loads(line)
            rec = batch_map.get(obj.get("custom_id"))
            if obj.get("error") or rec is None:
                n_err += 1
                continue
            if parse_self_judgement(_responses_body_text((obj.get("response") or {}).get("body") or {})) is None:
                n_noverdict += 1
                continue
            ids.add(str(rec["problem_id"]))

    LOGGER.info(f"[salvage] {len(outputs)} output file(s), {n_lines} response(s): "
                f"{len(ids)} with a verdict, {n_noverdict} unparsable, {n_err} errored.")
    return ids


def balance_keep_set(kept: set[str], labels: dict[str, str], seed: int) -> set[str]:
    """Drop majority-class ideas at random until POS and NEG counts match.

    Sorted before sampling so the draw depends only on the seed, not on set order.
    """
    pos = sorted(i for i in kept if labels.get(i) == "POSITIVE")
    neg = sorted(i for i in kept if labels.get(i) == "NEGATIVE")
    n = min(len(pos), len(neg))
    if n == 0:
        raise SystemExit(f"cannot balance: salvaged set has {len(pos)} POS and {len(neg)} NEG idea(s).")
    rng = random.Random(seed)
    balanced = set(rng.sample(pos, n)) | set(rng.sample(neg, n))
    LOGGER.info(f"[balance] seed {seed}: dropped {len(kept) - len(balanced)} "
                f"{'POSITIVE' if len(pos) > len(neg) else 'NEGATIVE'} idea(s) → {n} POS / {n} NEG.")
    return balanced


def salvaged_pair_ids(manifest: dict, batch_map: dict, kept: set[str]) -> set[str]:
    """Pairs where both the POS and the NEG pointwise idea survived."""
    by_pid = {str(e["problem_id"]): e for e in batch_map.values()}
    return {
        str(p["pair_id"]) for p in manifest["pairs"]
        if str(p["pos_pw_id"]) in by_pid and str(p["neg_pw_id"]) in by_pid
        and str(p["pos_pw_id"]) in kept and str(p["neg_pw_id"]) in kept
    }


# ── scoring ───────────────────────────────────────────────────────────────────

def _all_ids(instances_path: str) -> set[str]:
    if not instances_path:
        return set()
    p = Path(instances_path)
    p = p if p.is_absolute() else _PROJECT_ROOT / p
    if not p.exists():
        return set()
    return {str(k) for k in (yaml.safe_load(p.read_text()) or {})}


def _selfjudge_dir(root: Path) -> Path | None:
    """Newest self-judge artifact dir under ROOT, or None if `recompute` hasn't run."""
    base = root / "web_search_self_judge" / "accuracy_test_artifacts"
    runs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    return runs[-1] if runs else None


def confusion_counts(artifact_dir: Path, keep: set[str]) -> dict | None:
    """TP/FN/TN/FP over `keep` from this dir's pointwise scores.json.

    prediction 1 means "novel", so POSITIVE is the positive class. Instances in `keep`
    that the run never produced are counted as wrong — the same treatment
    _recompute_pointwise_metrics applies — so these counts reconcile with the
    accuracy column rather than quietly disagreeing with it.
    """
    scores_paths = sorted(artifact_dir.glob("run_pointwise_*/scores.json"))
    if not scores_paths:
        return None
    scores = json.loads(scores_paths[0].read_text())
    labels = _pointwise_labels(artifact_dir)

    counts = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "missing": 0}
    for iid in keep:
        label = labels.get(iid) or (scores.get(iid) or {}).get("label")
        if label is None:
            continue
        entry = scores.get(iid)
        if not isinstance(entry, dict) or entry.get("prediction") is None:
            counts["missing"] += 1
            counts["fn" if label == "POSITIVE" else "fp"] += 1  # absent → wrong
            continue
        pred = int(entry["prediction"])
        if label == "POSITIVE":
            counts["tp" if pred == 1 else "fn"] += 1
        else:
            counts["tn" if pred == 0 else "fp"] += 1
    return counts


def score_all(artifact_dirs: list[Path], keep_pointwise: set[str], keep_pairwise: set[str]) -> list[dict]:
    """Recompute metrics for each artifact dir restricted to the keep sets."""
    rows = []
    for a in artifact_dirs:
        mode = _extract_mode_from_artifact_dir(a)
        if mode not in ("pointwise", "pairwise"):
            continue
        keep = keep_pointwise if mode == "pointwise" else keep_pairwise
        exclude = _all_ids(_extract_instances_path(a)) - keep
        recompute = _recompute_pointwise_metrics if mode == "pointwise" else _recompute_pairwise_metrics
        metrics = recompute(a, exclude)
        if metrics:
            row = {"model": _extract_model_from_artifact_dir(a), "mode": mode, "metrics": metrics}
            if mode == "pointwise":
                row["confusion"] = confusion_counts(a, keep)
            rows.append(row)
        else:
            LOGGER.warning(f"{a.name}: could not recompute {mode} metrics (no scores.json?).")
    return rows


# ── divergence ────────────────────────────────────────────────────────────────

def _sorted_ids(ids: set[str]) -> list[str]:
    """Numeric-aware sort, so the manifest reads like the sampler's."""
    return sorted(ids, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))


def write_salvaged_manifest(manifest: dict, kept: set[str], kept_pairs: set[str],
                            pos: int, neg: int, n_salvaged: int,
                            balance_seed: int | None, out_path: Path) -> Path:
    """Write a manifest whose keep ids are the salvaged (optionally balanced) subset.

    Everything else is carried over from the sampler's manifest — `pairs` in particular,
    since divergence_report reads idea titles out of it.
    """
    narrowed = dict(manifest)
    narrowed["keep_pointwise_ids"] = _sorted_ids(kept)
    narrowed["keep_pairwise_ids"] = _sorted_ids(kept_pairs)
    narrowed["n_pairs_sampled"] = len(kept_pairs)
    narrowed["balance"] = {"positives": pos, "negatives": neg}
    narrowed["salvage"] = {
        "n_salvaged_ideas": n_salvaged,
        "n_sampled_ideas": len(manifest.get("keep_pointwise_ids", [])),
        "balanced": balance_seed is not None,
        "balance_seed": balance_seed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(narrowed, indent=2))
    LOGGER.info(f"Wrote narrowed manifest: {out_path}")
    return out_path


def run_divergence(cfg: dict, root: Path, manifest_path: Path, reference: Path, out_dir: Path) -> None:
    """Invoke divergence_report against the narrowed manifest.

    Written to its own out_dir so ROOT's full-subset subset_divergence.md survives.
    """
    sweeps = [Path(cfg["pointwise_sweep"]), Path(cfg["pairwise_sweep"])]
    sweeps = [s if s.is_absolute() else _PROJECT_ROOT / s for s in sweeps]
    cmd = [sys.executable, "-m", "novelty_eval.retrieval_faceoff.divergence_report",
           *[str(s) for s in sweeps],
           "--reference", str(reference), "--manifest", str(manifest_path),
           "--root", str(root), "--out", str(out_dir)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]))
    LOGGER.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env, check=True)


# ── rendering ─────────────────────────────────────────────────────────────────

def render(salvaged: list[dict], full: list[dict], manifest: dict, n_salvaged: int,
           n_ideas: int, n_pairs: int, pos: int, neg: int, balance_seed: int | None) -> str:
    full_by_key = {(r["mode"], r["model"]): r["metrics"] for r in full}
    scope = (f"salvaged {n_salvaged}/{len(manifest.get('keep_pointwise_ids', []))} ideas, "
             f"balanced down to {n_ideas} ({pos} POS / {neg} NEG) with seed `{balance_seed}`"
             if balance_seed is not None else
             f"salvaged {n_ideas}/{len(manifest.get('keep_pointwise_ids', []))} ideas "
             f"({pos} POS / {neg} NEG)")
    lines = [
        "# Salvaged-Subset Comparison",
        "",
        f"Seed `{manifest.get('seed')}` · cutoff `{manifest.get('cutoff_date')}` · {scope} · "
        f"{n_pairs}/{len(manifest.get('keep_pairwise_ids', []))} pairs.",
        "",
        "The *full-subset* column counts un-run ideas as wrong predictions, so it is "
        "meaningless for any judge that did not finish; it is shown only to make the gap visible.",
        "",
    ]
    if balance_seed is not None:
        lines += ["Pointwise rows are scored on the balanced draw; the pairwise table is "
                  "unaffected (each pair is one POS + one NEG already).", ""]

    pw = [r for r in salvaged if r["mode"] == "pointwise"]
    if pw:
        lines += ["## Pointwise (binary novelty classification)", "",
                  "| Judge / model | Support | Accuracy | F1 macro | F1 POS | F1 NEG | Acc (full subset) |",
                  "|---|---|---|---|---|---|---|"]
        for r in sorted(pw, key=lambda x: -x["metrics"]["accuracy"]):
            m = r["metrics"]
            f = full_by_key.get(("pointwise", r["model"]))
            full_acc = f"{f['accuracy']:.4f}" if f else "—"
            lines.append(
                f"| {r['model']} | {m['support']:.0f} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | "
                f"{m['f1_pos']:.4f} | {m['f1_neg']:.4f} | {full_acc} |")
        lines.append("")

    conf = [r for r in pw if r.get("confusion")]
    if conf:
        lines += [
            "## Pointwise confusion matrix",
            "",
            "Positive class is POSITIVE (`prediction = 1` means the judge called the idea novel). "
            "FN = a novel idea called derivative; FP = a derivative idea called novel.",
            "",
            "| Judge / model | TP | FN | TN | FP | POS recall | NEG recall | Says-novel rate |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(conf, key=lambda x: -x["metrics"]["accuracy"]):
            c = r["confusion"]
            n = c["tp"] + c["fn"] + c["tn"] + c["fp"]
            pos_rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
            neg_rec = c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else float("nan")
            says = (c["tp"] + c["fp"]) / n if n else float("nan")
            lines.append(f"| {r['model']} | {c['tp']} | {c['fn']} | {c['tn']} | {c['fp']} | "
                         f"{pos_rec:.3f} | {neg_rec:.3f} | {says:.3f} |")
        base = next((c["confusion"] for c in conf), None)
        if base:
            n = sum(base[k] for k in ("tp", "fn", "tn", "fp"))
            true_rate = (base["tp"] + base["fn"]) / n if n else float("nan")
            lines += ["", f"True novel rate on this subset: {true_rate:.3f}. A says-novel rate far "
                          f"below it means the judge is systematically calling novel ideas derivative.", ""]
        missing = {r["model"]: r["confusion"]["missing"] for r in conf if r["confusion"]["missing"]}
        if missing:
            lines += [f"*Counted as wrong because the run never produced them: "
                      f"{', '.join(f'{k} ({v})' for k, v in missing.items())}.*", ""]

    pr = [r for r in salvaged if r["mode"] == "pairwise"]
    if pr:
        lines += ["## Pairwise (preference / winner selection)", "",
                  "| Judge / model | Support | Acc (with ties) | Acc (w/o ties) | Ties |",
                  "|---|---|---|---|---|"]
        for r in sorted(pr, key=lambda x: -x["metrics"]["pairwise_accuracy"]):
            m = r["metrics"]
            lines.append(f"| {r['model']} | {m['support']:.0f} | {m['pairwise_accuracy']:.4f} | "
                         f"{m['pairwise_accuracy_no_ties']:.4f} | {m['n_ties']:.0f} |")
        lines.append("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Face-off run dir (holds manifest.json + web_search_batch_map.json)")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG),
                        help="Face-off config, for the baseline sweep paths")
    parser.add_argument("--batch-id", default=None, help="Batch to download (default: newest batch_info_*.json)")
    parser.add_argument("--no-download", action="store_true", help="Score what's already on disk")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if the output file exists")
    parser.add_argument("--balance", action="store_true",
                        help="Downsample the majority class so POS == NEG (pointwise only)")
    parser.add_argument("--balance-seed", type=int, default=None,
                        help="Seed for the balancing draw (default: the manifest's seed)")
    parser.add_argument("--divergence", action="store_true",
                        help="Also write a per-instance divergence report over the same subset")
    parser.add_argument("--divergence-out", default=None,
                        help="Dir for the divergence report (default: ROOT/salvaged)")
    parser.add_argument("--out", default=None, help="Write Markdown here instead of stdout")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"ROOT not found: {root}")

    global LOGGER
    LOGGER = setup_logger(output_dir=str(root), console_level="INFO")

    manifest = json.loads((root / "manifest.json").read_text())
    batch_map = json.loads((root / "web_search_batch_map.json").read_text())["map"]

    if not args.no_download:
        download_batch(root, args.batch_id, args.force_download)

    kept = salvaged_problem_ids(root, batch_map) & {str(i) for i in manifest["keep_pointwise_ids"]}
    if not kept:
        raise SystemExit("nothing salvageable — no response carried a parsable verdict.")

    # Pairs are drawn before balancing: each is one POS + one NEG already, so
    # downsampling pointwise ideas would only shrink the pairwise table for nothing.
    kept_pairs = salvaged_pair_ids(manifest, batch_map, kept)

    labels = {str(e["problem_id"]): e.get("gold_label") for e in batch_map.values()}
    n_salvaged = len(kept)
    balance_seed = None
    if args.balance:
        balance_seed = args.balance_seed if args.balance_seed is not None else manifest.get("seed", 0)
        kept = balance_keep_set(kept, labels, balance_seed)
    pos = sum(1 for i in kept if labels.get(i) == "POSITIVE")
    neg = sum(1 for i in kept if labels.get(i) == "NEGATIVE")
    LOGGER.info(f"[salvage] keeping {len(kept)} idea(s) ({pos} POS / {neg} NEG) and {len(kept_pairs)} pair(s) "
                f"of the {len(manifest['keep_pointwise_ids'])} / {len(manifest['keep_pairwise_ids'])} sampled.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    sources = [Path(cfg["pointwise_sweep"]), Path(cfg["pairwise_sweep"])]
    sources = [s if s.is_absolute() else _PROJECT_ROOT / s for s in sources]
    sj = _selfjudge_dir(root)
    if sj:
        sources.append(sj)
    else:
        LOGGER.warning("No self-judge run under ROOT — scoring baselines only. Run "
                       "`./scripts/run_retrieval_faceoff.sh recompute ROOT` to add the web-search judge.")

    artifact_dirs: list[Path] = []
    seen: set[str] = set()
    for s in sources:
        for a in _collect_artifact_dirs(s.resolve()):
            if a not in seen:
                seen.add(a)
                artifact_dirs.append(Path(a))
    LOGGER.info(f"Scoring {len(artifact_dirs)} artifact dir(s) on the salvaged subset.")

    salvaged = score_all(artifact_dirs, kept, kept_pairs)
    full = score_all(artifact_dirs,
                     {str(i) for i in manifest["keep_pointwise_ids"]},
                     {str(i) for i in manifest["keep_pairwise_ids"]})

    if args.divergence:
        if sj is None:
            LOGGER.warning("--divergence needs the self-judge run as its reference — skipping.")
        else:
            div_out = Path(args.divergence_out).resolve() if args.divergence_out else root / "salvaged"
            manifest_path = write_salvaged_manifest(manifest, kept, kept_pairs, pos, neg,
                                                    n_salvaged, balance_seed,
                                                    div_out / "manifest_salvaged.json")
            run_divergence(cfg, root, manifest_path, sj, div_out)

    output = render(salvaged, full, manifest, n_salvaged, len(kept), len(kept_pairs),
                    pos, neg, balance_seed)
    if args.out:
        Path(args.out).write_text(output)
        LOGGER.info(f"Table written to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
