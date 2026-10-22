#!/usr/bin/env python3
"""subset_reports.py — score artifact dirs on the matched subset, the merge way.

The ablation pipeline never re-runs judges to score a subset: it expresses the
subset as an *exclude set* of instance ids and recomputes metrics straight from
each run's ``scores.json`` via ``_recompute_{pointwise,pairwise}_metrics`` (which
delegate to the same ``metrics.compute_*`` functions every judge uses), then writes
a ``filtered_accuracy_report.txt`` per artifact dir.

This mirrors that exactly, deriving the exclude set from the sampled manifest:

    exclude_set = {all ids in the artifact's instances file} − {sampled keep ids}

so every collected artifact dir (the paper-finder baseline sweeps AND the
web-search self-judge run) is scored on the identical matched subset. Results are
written as ``subset_accuracy_report.txt`` alongside each ``accuracy_report.txt`` —
feed that filename to ``merge_ablation_runs.py --report-file`` for the full unified
comparison (paired bootstrap significance, helped/hurt, heatmaps).

A consolidated ``subset_comparison.md`` table is also written to --out (default:
the manifest's directory) for an immediate at-a-glance comparison.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils import LOGGER as LOGGER  # reassigned to a file logger in main() once out_dir is known
from logging_utils import setup_logger
from novelty_eval.analysis.artifacts import (
    _collect_artifact_dirs,
    _extract_ablation_from_artifact_dir,
    _extract_instances_path,
    _extract_mode_from_artifact_dir,
    _extract_model_from_artifact_dir,
)
from novelty_eval.analysis.filtering import (
    _recompute_pointwise_metrics,
    _recompute_pairwise_metrics,
    _write_filtered_accuracy_report,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _all_ids(instances_path: str) -> Set[str]:
    """All problem ids (as strings) in a test-instances YAML."""
    if not instances_path:
        return set()
    p = Path(instances_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        return set()
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return {str(k) for k in data}


def _score_dir(artifact_dir: Path, keep_pointwise: Set[str], keep_pairwise: Set[str],
               report_name: str) -> Optional[dict]:
    """Recompute subset metrics for one artifact dir and write its subset report."""
    original_report = artifact_dir / "accuracy_report.txt"
    if not original_report.exists():
        LOGGER.warning(f"No accuracy_report.txt in {artifact_dir} — skipping.")
        return None

    mode = _extract_mode_from_artifact_dir(artifact_dir)
    if mode not in ("pointwise", "pairwise"):
        LOGGER.warning(f"{artifact_dir.name}: mode={mode} not scorable (no scores.json) — skipping.")
        return None

    instances_path = _extract_instances_path(artifact_dir)
    all_ids = _all_ids(instances_path)
    keep = keep_pointwise if mode == "pointwise" else keep_pairwise
    # Exclude everything outside the sampled subset. If the artifact was already run
    # on the subset file (all_ids ⊆ keep) the exclude set is empty — same result.
    exclude_set = all_ids - keep
    kept_here = all_ids & keep
    if not kept_here:
        LOGGER.warning(f"{artifact_dir.name}: none of its {len(all_ids)} ids are in the "
                       f"sampled {mode} subset — skipping.")
        return None

    if mode == "pointwise":
        metrics = _recompute_pointwise_metrics(artifact_dir, exclude_set, logger_inst=LOGGER)
    else:
        metrics = _recompute_pairwise_metrics(artifact_dir, exclude_set, logger_inst=LOGGER)
    if metrics is None:
        LOGGER.warning(f"{artifact_dir.name}: could not recompute metrics (no scores.json?).")
        return None

    out_path = _write_filtered_accuracy_report(
        artifact_dir, mode, metrics, original_report, out_filename=report_name)
    model = _extract_model_from_artifact_dir(artifact_dir)
    config = _extract_ablation_from_artifact_dir(artifact_dir)
    LOGGER.info(f"[{mode}] {config} / {model}: support={metrics['support']:.0f} → {out_path}")
    return {"artifact_dir": str(artifact_dir), "mode": mode, "model": model,
            "config": config, "metrics": metrics}


def _write_comparison_md(rows: List[dict], out_path: Path, manifest: dict) -> None:
    pw = [r for r in rows if r["mode"] == "pointwise"]
    pr = [r for r in rows if r["mode"] == "pairwise"]
    lines: List[str] = [
        "# Matched-Subset Comparison",
        "",
        f"Seed `{manifest.get('seed')}` · {manifest.get('n_pairs_sampled')} pairs "
        f"({manifest['balance']['positives']} POS / {manifest['balance']['negatives']} NEG pointwise ideas) · "
        f"cutoff `{manifest.get('cutoff_date')}` · "
        f"{manifest['blocklist']['n_pairs_dropped']} pair(s) dropped by blocklist.",
        "",
    ]
    # Group by config, ranked within each: the same model can appear under several
    # configurations (e.g. a k=3 sweep and the k=1 sub-runs derived from it), so the
    # model name alone no longer identifies a row.
    def _by_config(rows: List[dict], metric: str):
        return sorted(rows, key=lambda x: (x.get("config", ""), -x["metrics"][metric]))

    if pw:
        lines += [
            "## Pointwise (binary novelty classification)",
            "",
            "| Config | Judge / model | Support | Accuracy | F1 macro | F1 POS | F1 NEG |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in _by_config(pw, "accuracy"):
            m = r["metrics"]
            lines.append(
                f"| {r.get('config', '')} | {r['model']} | {m['support']:.0f} | {m['accuracy']:.4f} | "
                f"{m['f1_macro']:.4f} | {m['f1_pos']:.4f} | {m['f1_neg']:.4f} |")
        lines.append("")
    if pr:
        lines += [
            "## Pairwise (preference / winner selection)",
            "",
            "| Config | Judge / model | Support | Acc (with ties) | Acc (w/o ties) | Ties |",
            "|---|---|---|---|---|---|",
        ]
        for r in _by_config(pr, "pairwise_accuracy"):
            m = r["metrics"]
            lines.append(
                f"| {r.get('config', '')} | {r['model']} | {m['support']:.0f} | {m['pairwise_accuracy']:.4f} | "
                f"{m['pairwise_accuracy_no_ties']:.4f} | {m['n_ties']:.1f} |")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info(f"Wrote consolidated comparison: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+",
                        help="Sweep/ablation output dirs, experiment roots, or single artifact dirs.")
    parser.add_argument("--manifest", required=True, help="manifest.json from sample_matched_subset.py.")
    parser.add_argument("--report-name", default="subset_accuracy_report.txt",
                        help="Per-artifact report filename (feed to merge --report-file).")
    parser.add_argument("--out", default=None,
                        help="Where to write subset_comparison.md (default: manifest's directory).")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.manifest).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    global LOGGER
    LOGGER = setup_logger(output_dir=str(out_dir), console_level="INFO")

    with open(args.manifest) as f:
        manifest = json.load(f)
    keep_pointwise = set(str(x) for x in manifest.get("keep_pointwise_ids", []))
    keep_pairwise = set(str(x) for x in manifest.get("keep_pairwise_ids", []))
    if not keep_pointwise and not keep_pairwise:
        LOGGER.error("Manifest has no keep ids — nothing to score.")
        return

    # Collect artifact dirs from every input path (dedup, preserve order).
    artifact_dirs: List[str] = []
    seen: Set[str] = set()
    for d in args.dirs:
        for a in _collect_artifact_dirs(Path(d).resolve()):
            if a not in seen:
                seen.add(a)
                artifact_dirs.append(a)
    LOGGER.info(f"Scoring {len(artifact_dirs)} artifact dir(s) on the matched subset "
                f"({len(keep_pointwise)} pointwise ids, {len(keep_pairwise)} pairwise ids).")

    rows: List[dict] = []
    for a in artifact_dirs:
        row = _score_dir(Path(a), keep_pointwise, keep_pairwise, args.report_name)
        if row:
            rows.append(row)

    if not rows:
        LOGGER.warning("No artifact dirs scored.")
        return

    _write_comparison_md(rows, out_dir / "subset_comparison.md", manifest)
    with open(out_dir / "subset_scores.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "metrics"} | {"metrics": r["metrics"]}
                   for r in rows], f, indent=2)


if __name__ == "__main__":
    main()
