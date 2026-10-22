#!/usr/bin/env python3
"""divergence_report.py — per-instance divergence table for the retrieval face-off.

subset_reports.py answers "which method is better on average" on the matched subset;
this answers "which specific ideas does it disagree on" — one row per instance where
correctness differs between the web-search judge (the --reference artifact dir) and
a baseline paper-finder sweep, with a link to the existing per-idea retrieval debug
report (``retrieval_debug/{id}/*retrieval.md``) on whichever side has one.

Every artifact dir in this pipeline runs with n=1 (one scores.json per instance, no
repeated sampling), so correctness is just right/wrong — no averaging across runs.

Written as ``subset_divergence.md`` alongside ``subset_comparison.md`` by the same
run.py scoring step (submit/poll/recompute) — no separate CLI stage needed.

    python -m novelty_eval.retrieval_faceoff.divergence_report \\
        <baseline_dir> [<baseline_dir> ...] \\
        --reference <web_search_self_judge_artifact_dir> \\
        --manifest ROOT/manifest.json --root ROOT
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils import LOGGER as LOGGER  # reassigned to a file logger in main() once out_dir is known
from logging_utils import setup_logger
from novelty_eval.analysis.artifacts import (
    _collect_artifact_dirs,
    _extract_ablation_from_artifact_dir,
    _extract_instance_id_from_key,
    _extract_instances_path,
    _extract_mode_from_artifact_dir,
    _extract_model_from_artifact_dir,
    _is_pairwise_failure_tie,
)
from novelty_eval.analysis.filtering import _load_yaml_cached

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# --- Instances-file lookups (labels / ground truth) -----------------------------

def _resolve_instances_path(artifact_dir: Path) -> Optional[Path]:
    raw = _extract_instances_path(artifact_dir)
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _pointwise_labels(artifact_dir: Path) -> Dict[str, str]:
    p = _resolve_instances_path(artifact_dir)
    if not p or not p.exists():
        return {}
    data = _load_yaml_cached(p)
    return {str(k): v["label"] for k, v in data.items() if isinstance(v, dict) and "label" in v}


def _pairwise_gt(artifact_dir: Path) -> Dict[str, Set[str]]:
    p = _resolve_instances_path(artifact_dir)
    if not p or not p.exists():
        return {}
    data = _load_yaml_cached(p)
    return {
        str(k): {str(w) for w in v["expected_winners"]}
        for k, v in data.items() if isinstance(v, dict) and "expected_winners" in v
    }


def _one_scores_file(artifact_dir: Path, glob_pat: str) -> Optional[Path]:
    """The single scores.json for this artifact dir.

    Divergence is a per-instance right/wrong comparison, so it needs exactly one
    run.  Most artifact dirs here have one; derived k=1 sub-run dirs carry several
    independent single-sample runs, and we compare against the first.
    """
    matches = sorted(artifact_dir.glob(glob_pat)) or sorted(artifact_dir.glob("scores.json"))
    if not matches:
        return None
    if len(matches) > 1:
        LOGGER.warning(f"{artifact_dir.name}: found {len(matches)} runs (derived sub-runs, or "
                       f"n>1?) — comparing against the first only: {matches[0]}.")
    return matches[0]


# --- Per-instance outcome extraction ---------------------------------------------

def _pointwise_outcomes(artifact_dir: Path, keep_ids: Set[str]) -> Dict[str, dict]:
    """{id: {'label', 'correct', 'pred'}} for ids in keep_ids with a resolvable label."""
    scores_path = _one_scores_file(artifact_dir, "run_pointwise_*/scores.json")
    if scores_path is None:
        return {}
    try:
        with open(scores_path) as f:
            scores = json.load(f)
    except Exception:
        return {}

    label_lookup = _pointwise_labels(artifact_dir)
    outcomes: Dict[str, dict] = {}
    seen: Set[str] = set()
    for k, v in scores.items():
        iid = _extract_instance_id_from_key(k)
        if iid is None or iid not in keep_ids or not isinstance(v, dict):
            continue
        pred = v.get("prediction")
        label = v.get("label") or label_lookup.get(iid)
        if pred is None or label is None:
            continue
        target = 1 if label == "POSITIVE" else 0
        outcomes[iid] = {"label": label, "correct": int(pred) == target, "pred": int(pred)}
        seen.add(iid)

    for iid in keep_ids - seen:
        label = label_lookup.get(iid)
        if label is None:
            continue
        outcomes[iid] = {"label": label, "correct": False, "pred": None}  # absent — treated as wrong

    return outcomes


def _pairwise_outcomes(artifact_dir: Path, keep_ids: Set[str]) -> Dict[str, dict]:
    """{pair_id: {'gt_winner', 'correct', 'winner'}} for ids in keep_ids with ground truth."""
    scores_path = _one_scores_file(artifact_dir, "run_pairwise_*/scores.json")
    if scores_path is None:
        return {}
    try:
        with open(scores_path) as f:
            scores = json.load(f)
    except Exception:
        return {}

    gt_lookup = _pairwise_gt(artifact_dir)
    outcomes: Dict[str, dict] = {}
    seen: Set[str] = set()
    for k, v in scores.items():
        iid = _extract_instance_id_from_key(k)
        if iid is None or iid not in keep_ids or not isinstance(v, dict):
            continue
        comps = v.get("comparisons") or []
        if not comps:
            continue
        comp = comps[0]
        if comp.get("gt_winner") is None and iid in gt_lookup:
            from novelty_eval.run_benchmark import _annotate_gt_winner
            _annotate_gt_winner(comps, gt_lookup[iid])
            comp = comps[0]
        gt = comp.get("gt_winner")
        if gt is None:
            continue
        is_tie = comp.get("winner") == 2 and not _is_pairwise_failure_tie(comp)
        outcomes[iid] = {
            "gt_winner": int(gt),
            "correct": (not is_tie) and (comp.get("winner") == gt),
            "winner": comp.get("winner"),
        }
        seen.add(iid)

    for iid in keep_ids - seen:
        if iid not in gt_lookup or not gt_lookup[iid]:
            continue
        outcomes[iid] = {  # absent — treated as wrong
            "gt_winner": int(next(iter(gt_lookup[iid]))), "correct": False, "winner": None,
        }

    return outcomes


# --- Retrieval-debug link resolution ---------------------------------------------

def _debug_md(base_dir: Path, pw_id: str) -> Optional[Path]:
    """Path to retrieval_debug/{pw_id}/*retrieval.md under base_dir, if it exists.

    This is how the web-search judge (the reference) saves debug info — one file
    per idea. The paper-finder baseline sweeps don't produce this; they dump every
    instance into one big debug_accuracy_report.txt.md instead (see below).
    """
    d = base_dir / "retrieval_debug" / pw_id
    if not d.is_dir():
        return None
    matches = sorted(d.glob("*retrieval.md"))
    return matches[0] if matches else None


# Baseline sweeps (md_report_writer.py) write one monolithic debug_accuracy_report.txt.md
# per artifact dir, with every instance as a bullet/heading inside it:
#   pointwise: "- **Instance {id}**"          (flat bullet, no anchor)
#   pairwise:  "### Problem {id}" or "... ✓"  (heading, but real ids aren't reliably
#                                               anchor-safe across markdown renderers)
# so instead of linking into that huge file, slice out just the id's block and cache
# it as its own small file — mirrors the reference's one-file-per-idea convention.
_INSTANCE_HEADER_RE = re.compile(r"^- \*\*Instance ([^\*]+)\*\*\s*$", re.MULTILINE)
_INSTANCE_BOUNDARY_RE = re.compile(r"^(?:- \*\*Instance |#)", re.MULTILINE)
_PROBLEM_HEADER_RE = re.compile(r"^### Problem (\S+)(?:\s*✓)?\s*$", re.MULTILINE)
_PROBLEM_BOUNDARY_RE = re.compile(r"^#", re.MULTILINE)


def _extract_block(text: str, header_re: re.Pattern, boundary_re: re.Pattern, pid: str) -> Optional[str]:
    for m in header_re.finditer(text):
        if m.group(1) == pid:
            nxt = boundary_re.search(text, m.end())
            return text[m.start():(nxt.start() if nxt else len(text))].rstrip() + "\n"
    return None


def _baseline_debug_excerpt(artifact_dir: Path, mode: str, pid: str, cache_dir: Path) -> Optional[Path]:
    """Slice pid's block out of artifact_dir's debug_accuracy_report.txt.md, cached to a file."""
    report_path = artifact_dir / "debug_accuracy_report.txt.md"
    if not report_path.exists():
        return None
    # artifact_dir is .../<model-dir>/accuracy_test_artifacts/<timestamp> — use the model dir
    # (accuracy_test_artifacts/<timestamp> alone collides across every model in a sweep).
    slug = artifact_dir.parent.parent.name if artifact_dir.parent.name == "accuracy_test_artifacts" \
        else artifact_dir.name
    excerpt_path = cache_dir / slug / f"{pid}.md"
    if excerpt_path.exists():
        return excerpt_path

    text = report_path.read_text(encoding="utf-8", errors="ignore")
    if mode == "pointwise":
        block = _extract_block(text, _INSTANCE_HEADER_RE, _INSTANCE_BOUNDARY_RE, pid)
    else:
        block = _extract_block(text, _PROBLEM_HEADER_RE, _PROBLEM_BOUNDARY_RE, pid)
    if block is None:
        return None

    excerpt_path.parent.mkdir(parents=True, exist_ok=True)
    excerpt_path.write_text(
        f"# Baseline debug excerpt — {slug}, id {pid}\n\nSliced from `{report_path}`.\n\n---\n\n{block}",
        encoding="utf-8",
    )
    return excerpt_path


def _resolve_debug(artifact_dir: Path, mode: str, pid: str, cache_dir: Path) -> Optional[Path]:
    return _debug_md(artifact_dir, pid) or _baseline_debug_excerpt(artifact_dir, mode, pid, cache_dir)


def _link(path: Optional[Path], out_dir: Path, label: str = "debug") -> str:
    if path is None:
        return "—"
    try:
        rel = os.path.relpath(path, out_dir)
    except ValueError:
        rel = str(path)
    return f"[{label}]({rel})"


def _check(correct: bool) -> str:
    return "✅" if correct else "❌"


def _confusion(ref: Dict[str, dict], base: Dict[str, dict], keep_ids: Set[str]) -> Dict[str, int]:
    counts = {"both_correct": 0, "reference_only": 0, "baseline_only": 0, "both_wrong": 0}
    for iid in keep_ids:
        r, b = ref.get(iid), base.get(iid)
        if r is None or b is None:
            continue
        if r["correct"] and b["correct"]:
            counts["both_correct"] += 1
        elif r["correct"]:
            counts["reference_only"] += 1
        elif b["correct"]:
            counts["baseline_only"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def _diverging_ids(ref: Dict[str, dict], base: Dict[str, dict], keep_ids: Set[str]) -> List[Tuple[str, dict, dict]]:
    rows = [(iid, ref[iid], base[iid]) for iid in keep_ids
            if iid in ref and iid in base and ref[iid]["correct"] != base[iid]["correct"]]
    rows.sort(key=lambda x: (not x[1]["correct"], x[0]))  # reference-correct rows first
    return rows


# --- Section rendering -------------------------------------------------------------

def _pointwise_section(ref: Dict[str, dict], ref_model: str, ref_dir: Path,
                       base: Dict[str, dict], base_model: str, base_dir: Path,
                       keep_ids: Set[str], id_to_title: Dict[str, str], out_dir: Path,
                       cache_dir: Path) -> List[str]:
    counts = _confusion(ref, base, keep_ids)
    n_compared = sum(counts.values())
    if n_compared == 0:
        return []

    rows = _diverging_ids(ref, base, keep_ids)
    lines = [
        f"### Pointwise — {ref_model} (reference) vs. {base_model}",
        "",
        f"Both correct: {counts['both_correct']} · reference only: {counts['reference_only']} · "
        f"baseline only: {counts['baseline_only']} · both wrong: {counts['both_wrong']} "
        f"(support {n_compared}).",
        "",
    ]
    if not rows:
        lines += ["*No diverging ideas.*", ""]
        return lines

    lines += [
        f"{len(rows)} diverging idea(s):",
        "",
        "| id | gold | title | reference | baseline | ref debug | baseline debug |",
        "|---|---|---|---|---|---|---|",
    ]
    for iid, r, b in rows:
        title = id_to_title.get(iid, "")
        lines.append(
            f"| {iid} | {r['label']} | {title} | {_check(r['correct'])} | {_check(b['correct'])} | "
            f"{_link(_debug_md(ref_dir, iid), out_dir)} | "
            f"{_link(_resolve_debug(base_dir, 'pointwise', iid, cache_dir), out_dir)} |"
        )
    lines.append("")
    return lines


def _pairwise_section(ref: Dict[str, dict], ref_model: str, ref_dir: Path,
                      base: Dict[str, dict], base_model: str, base_dir: Path,
                      keep_ids: Set[str], pair_info: Dict[str, dict], out_dir: Path,
                      cache_dir: Path) -> List[str]:
    counts = _confusion(ref, base, keep_ids)
    n_compared = sum(counts.values())
    if n_compared == 0:
        return []

    rows = _diverging_ids(ref, base, keep_ids)
    lines = [
        f"### Pairwise — {ref_model} (reference) vs. {base_model}",
        "",
        f"Both correct: {counts['both_correct']} · reference only: {counts['reference_only']} · "
        f"baseline only: {counts['baseline_only']} · both wrong: {counts['both_wrong']} "
        f"(support {n_compared}).",
        "",
    ]
    if not rows:
        lines += ["*No diverging pairs.*", ""]
        return lines

    lines += [
        f"{len(rows)} diverging pair(s) — gold winner is always the POSITIVE idea:",
        "",
        "| pair id | reference | baseline | positive idea | negative idea | baseline debug |",
        "|---|---|---|---|---|---|",
    ]
    for iid, r, b in rows:
        info = pair_info.get(iid, {})
        pos_id, neg_id = info.get("pos_pw_id"), info.get("neg_pw_id")
        pos_cell = f"{info.get('pos_title', '')} {_link(_debug_md(ref_dir, pos_id), out_dir, 'ref') if pos_id else ''}"
        neg_cell = f"{info.get('neg_title', '')} {_link(_debug_md(ref_dir, neg_id), out_dir, 'ref') if neg_id else ''}"
        base_link = _link(_resolve_debug(base_dir, "pairwise", iid, cache_dir), out_dir)
        lines.append(f"| {iid} | {_check(r['correct'])} | {_check(b['correct'])} | {pos_cell} | {neg_cell} | {base_link} |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+",
                        help="Baseline sweep/ablation output dirs (or single artifact dirs) to "
                             "compare against --reference.")
    parser.add_argument("--reference", required=True,
                        help="Web-search self-judge artifact dir (single artifact dir, not a sweep root).")
    parser.add_argument("--manifest", required=True, help="manifest.json from sample_matched_subset.py.")
    parser.add_argument("--root", default=None,
                        help="Face-off ROOT holding the reference judge's retrieval_debug/ "
                             "(default: manifest's directory).")
    parser.add_argument("--out", default=None,
                        help="Where to write subset_divergence.md (default: manifest's directory).")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.manifest).resolve().parent
    root = Path(args.root).resolve() if args.root else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    global LOGGER
    LOGGER = setup_logger(output_dir=str(out_dir), console_level="INFO")

    with open(args.manifest) as f:
        manifest = json.load(f)
    keep_pointwise = {str(x) for x in manifest.get("keep_pointwise_ids", [])}
    keep_pairwise = {str(x) for x in manifest.get("keep_pairwise_ids", [])}
    if not keep_pointwise and not keep_pairwise:
        LOGGER.error("Manifest has no keep ids — nothing to compare.")
        return

    id_to_title: Dict[str, str] = {}
    pair_info: Dict[str, dict] = {}
    for p in manifest.get("pairs", []):
        pos_id, neg_id = str(p["pos_pw_id"]), str(p["neg_pw_id"])
        id_to_title[pos_id] = p.get("pos_title", "")
        id_to_title[neg_id] = p.get("neg_title", "")
        pair_info[str(p["pair_id"])] = {
            "pos_title": p.get("pos_title", ""), "neg_title": p.get("neg_title", ""),
            "pos_pw_id": pos_id, "neg_pw_id": neg_id,
        }

    reference_dir = Path(args.reference).resolve()
    ref_model = _extract_model_from_artifact_dir(reference_dir)
    ref_pw = _pointwise_outcomes(reference_dir, keep_pointwise)
    ref_pr = _pairwise_outcomes(reference_dir, keep_pairwise)

    baseline_dirs: List[str] = []
    seen: Set[str] = set()
    for d in args.dirs:
        for a in _collect_artifact_dirs(Path(d).resolve()):
            if a not in seen:
                seen.add(a)
                baseline_dirs.append(a)
    LOGGER.info(f"Comparing reference ({ref_model}) against {len(baseline_dirs)} baseline artifact dir(s).")

    lines: List[str] = [
        "# Matched-Subset Divergence",
        "",
        f"Reference: **{ref_model}** (web-search judge) · seed `{manifest.get('seed')}` · "
        f"{manifest.get('n_pairs_sampled')} pairs.",
        "",
        "Rows are instances where the reference and a baseline disagree on correctness.",
        "",
    ]

    cache_dir = out_dir / "baseline_debug"
    any_section = False
    for a in baseline_dirs:
        artifact_dir = Path(a)
        mode = _extract_mode_from_artifact_dir(artifact_dir)
        # Qualify by ablation: the same model can appear under several configurations
        # (e.g. a k=3 sweep and the k=1 sub-runs derived from it), which would otherwise
        # produce two identically titled sections.
        model = f"{_extract_model_from_artifact_dir(artifact_dir)} " \
                f"({_extract_ablation_from_artifact_dir(artifact_dir)})"
        if mode == "pointwise":
            base = _pointwise_outcomes(artifact_dir, keep_pointwise)
            section = _pointwise_section(ref_pw, ref_model, root, base, model, artifact_dir,
                                         keep_pointwise, id_to_title, out_dir, cache_dir)
        elif mode == "pairwise":
            base = _pairwise_outcomes(artifact_dir, keep_pairwise)
            section = _pairwise_section(ref_pr, ref_model, root, base, model, artifact_dir,
                                        keep_pairwise, pair_info, out_dir, cache_dir)
        else:
            continue
        if section:
            lines += section
            any_section = True

    if not any_section:
        lines.append("*No comparable baseline runs found on the matched subset.*")
        lines.append("")

    out_path = out_dir / "subset_divergence.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info(f"Wrote divergence report: {out_path}")


if __name__ == "__main__":
    main()
