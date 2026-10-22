#!/usr/bin/env python3
"""
Derive sub-run artifacts from a 'current' ablation sweep.

Given a sweep directory that ran 'current' with mec_k=3 + bidirectional=True,
carves 3 independent sub-runs for:

  mec_k_1        (bidirectional=True,  mec_k=1): call index k for both fwd+bwd
  unidirectional (bidirectional=False, mec_k=1): forward call index k only

Each derived ablation gets run_{mode}_{0,1,2}/scores.json under a new
timestamped artifact dir, matching the structure of a real n_runs=3 sweep.
The output directory can be passed directly to merge_ablation_runs.py
alongside the original sweep directory.

By default the source rows are the sweep's ``*_current`` ablations, and the
``_current`` marker is stripped so the derived name reads e.g.
``pairwise-vanilla-ai_mec_k_1``.  Any other ablation can be the source via
--source-suffix; pair it with --keep-source-suffix when the suffix names a real
property of the judge rather than "this is the baseline", so that deriving from
``pointwise-vanilla-ai_retrieval`` yields ``pointwise-vanilla-ai_retrieval_mec_k_1``
rather than dropping the retrieval part (and colliding with the _current-derived run).

Usage:
    python derive_subrun_artifacts.py <sweep_dir1> [<sweep_dir2> ...] \\
        [--output-dir PATH] \\
        [--ablations mec_k_1 unidirectional] \\
        [--n-subsamples 3] \\
        [--source-suffix _current] [--keep-source-suffix]

Example:
    python derive_subrun_artifacts.py \\
        output/ablation_sweeps/20260518_110534 \\
        output/ablation_sweeps/20260528_171320 \\
        --output-dir output/ablation_sweeps/derived_mec_k1_uni
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader
import yaml

from utils import extract_json_choice
from novelty_eval.scoring import aggregate_unidirectional_comparisons
from novelty_eval.metrics import calculate_pairwise_accuracy, compute_pointwise_metrics
from novelty_eval.experiment_stats import ExperimentStats
from novelty_eval.report_writer import save_report


# ---------------------------------------------------------------------------
# Per-ablation config
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: dict[str, dict] = {
    "mec_k_1": {
        "bidirectional": True,
        "mode": "both",  # applies to pairwise and pointwise
        "description": "Bidirectional evaluation with mec_k=1 (single sample per direction)",
        "before_description": "Bidirectional evaluation with mec_k=3 (three samples per direction) to reduce position bias.",
    },
    "unidirectional": {
        "bidirectional": False,
        "mode": "pairwise",  # pairwise only
        "description": "Unidirectional (fwd-only, no position debiasing)",
        "before_description": "Bidirectional evaluation (+ three samples per direction) to reduce position bias.",
    },
}


# ---------------------------------------------------------------------------
# Batch JSONL parsing
# ---------------------------------------------------------------------------

def _get_message_content(record: dict) -> str:
    """Extract text content from an OpenAI or Anthropic batch response record."""
    # OpenAI format: response.body.choices[0].message.content
    choices = record.get("response", {}).get("body", {}).get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    # Anthropic format: result.message.content
    result = record.get("result", {})
    if result.get("type") == "succeeded" and "message" in result:
        content = result["message"].get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
        elif isinstance(content, str):
            return content
    return ""


def _parse_pairwise_batch_jsonl(jsonl_path: Path) -> dict:
    """Parse a pairwise batch_output JSONL.

    Returns: {(inst_id, idea_i, idea_j, call_index) -> {"novelty": int}}
    Custom-id format: cmp__{inst}__{idea_i}__{idea_j}__{call_index}
    """
    records: dict = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = record.get("custom_id", "")
            # Support both new (__) and old (-) separator formats
            if "__" in custom_id:
                parts = custom_id.split("__")
            else:
                parts = custom_id.split("-")
            if len(parts) < 5 or parts[0] != "cmp":
                continue
            inst_id, idea_i, idea_j = parts[1], parts[2], parts[3]
            try:
                call_index = int(parts[4])
            except ValueError:
                continue
            content = _get_message_content(record)
            if not content:
                continue
            choice = extract_json_choice(content)
            if choice:
                records[(inst_id, idea_i, idea_j, call_index)] = choice
    return records


def _parse_pointwise_batch_jsonl(jsonl_path: Path) -> dict:
    """Parse a pointwise batch_output JSONL.

    Returns: {(inst_id, call_index) -> {"novelty": int}}
    Custom-id format: ptw__{inst_id}__{call_index}
    """
    records: dict = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = record.get("custom_id", "")
            parts = custom_id.split("__")
            if len(parts) < 3 or parts[0] != "ptw":
                continue
            inst_id = parts[1]
            try:
                call_index = int(parts[2])
            except ValueError:
                continue
            content = _get_message_content(record)
            if not content:
                continue
            choice = extract_json_choice(content)
            if choice:
                records[(inst_id, call_index)] = choice
    return records


def _collect_pairwise_records(artifact_dir: Path) -> dict:
    """Collect all pairwise vote records from run_pairwise_*/batch_output_*.jsonl."""
    records: dict = {}
    for bf in sorted(artifact_dir.glob("run_pairwise_*/batch_output_*.jsonl")):
        records.update(_parse_pairwise_batch_jsonl(bf))
    return records


def _collect_pointwise_records(artifact_dir: Path) -> dict:
    """Collect all pointwise vote records from run_pointwise_*/batch_output_*.jsonl."""
    records: dict = {}
    for bf in sorted(artifact_dir.glob("run_pointwise_*/batch_output_*.jsonl")):
        records.update(_parse_pointwise_batch_jsonl(bf))
    return records


# ---------------------------------------------------------------------------
# Sub-run construction
# ---------------------------------------------------------------------------

def _is_forward_comparison(idea_i: str, idea_j: str) -> bool:
    """Forward = idea_i has a smaller index than idea_j."""
    try:
        return int(idea_i) < int(idea_j)
    except ValueError:
        return idea_i < idea_j


def _build_pairwise_subrun(records: dict, call_index: int, bidirectional: bool) -> dict:
    """Build comparisons_by_problem for a single pairwise sub-run.

    bidirectional=True  (mec_k_1):      both directions at call_index
    bidirectional=False (unidirectional): forward direction only at call_index
    """
    by_problem: dict = defaultdict(list)
    for (inst_id, idea_i, idea_j, k), choice in records.items():
        if k != call_index:
            continue
        if not bidirectional and not _is_forward_comparison(idea_i, idea_j):
            continue
        by_problem[inst_id].append({"idea_i": idea_i, "idea_j": idea_j, "scores": choice})
    return dict(by_problem)


def _build_pointwise_subrun(records: dict, call_index: int) -> dict:
    """Build per-instance predictions for a single pointwise sub-run.

    Returns: {inst_id: {"prediction": 0|1}}
    """
    results: dict = {}
    for (inst_id, k), choice in records.items():
        if k != call_index:
            continue
        votes = [v for v in choice.values() if isinstance(v, int) and v in (0, 1)]
        if votes:
            results[inst_id] = {"prediction": 1 if sum(votes) > len(votes) / 2 else 0}
    return results


# ---------------------------------------------------------------------------
# Ground-truth annotation
# ---------------------------------------------------------------------------

def _annotate_gt_winner(comparisons: list, expected_winners: set) -> None:
    """Add gt_winner to each pairwise comparison dict in-place."""
    for comp in comparisons:
        idea_0 = str(comp.get("idea_0", ""))
        idea_1 = str(comp.get("idea_1", ""))
        if idea_0 in expected_winners and idea_1 not in expected_winners:
            comp["gt_winner"] = 0
        elif idea_1 in expected_winners and idea_0 not in expected_winners:
            comp["gt_winner"] = 1
        else:
            comp["gt_winner"] = None


def _enrich_pairwise_with_gt(results: dict, inputs: dict) -> None:
    """Annotate all pairwise comparisons with gt_winner from the instances YAML."""
    for pid, data in results.items():
        inp = inputs.get(pid) or inputs.get(str(pid)) or {}
        ew = {str(w) for w in inp.get("expected_winners", [])}
        _annotate_gt_winner(data.get("comparisons", []), ew)


def _enrich_pointwise_with_labels(results: dict, inputs: dict) -> None:
    """Back-fill 'label' into each pointwise result entry from the instances YAML."""
    for inst_id, data in results.items():
        inp = inputs.get(inst_id) or inputs.get(str(inst_id)) or {}
        label = inp.get("label")
        if label and "label" not in data:
            data["label"] = label


# ---------------------------------------------------------------------------
# Stats helpers (avoid importing heavy run_benchmark.py)
# ---------------------------------------------------------------------------

def _compute_pairwise_stats(run_results: list[dict], inputs: dict) -> ExperimentStats:
    stats = ExperimentStats()
    for results in run_results:
        m = calculate_pairwise_accuracy(results, inputs)
        stats.pairwise_accuracies.append(m["accuracy_with_ties"])
        stats.pairwise_accuracies_no_ties.append(m["accuracy_without_ties"])
        stats.pairwise_n_ties.append(m["n_ties"])
        stats.pairwise_support.append(m["support"])
        stats.pairwise_support_no_ties.append(m["support_without_ties"])
    if stats.pairwise_accuracies:
        stats.mean_pairwise_accuracy = statistics.mean(stats.pairwise_accuracies)
    if stats.pairwise_accuracies_no_ties:
        stats.mean_pairwise_accuracy_no_ties = statistics.mean(stats.pairwise_accuracies_no_ties)
    if stats.pairwise_n_ties:
        stats.mean_pairwise_n_ties = statistics.mean(stats.pairwise_n_ties)
    if stats.pairwise_support:
        stats.mean_pairwise_support = statistics.mean(stats.pairwise_support)
    if stats.pairwise_support_no_ties:
        stats.mean_pairwise_support_no_ties = statistics.mean(stats.pairwise_support_no_ties)
    return stats


def _compute_pointwise_stats(run_results: list[dict], inputs: dict) -> ExperimentStats:
    stats = ExperimentStats()
    for results in run_results:
        m = compute_pointwise_metrics(results, inputs)
        stats.pointwise_accuracies.append(m["accuracy"])
        stats.pointwise_precision_pos.append(m["precision_pos"])
        stats.pointwise_precision_neg.append(m["precision_neg"])
        stats.pointwise_recall_pos.append(m["recall_pos"])
        stats.pointwise_recall_neg.append(m["recall_neg"])
        stats.pointwise_f1_pos.append(m["f1_pos"])
        stats.pointwise_f1_neg.append(m["f1_neg"])
        stats.pointwise_f1_macro.append(m["f1_macro"])
        stats.pointwise_support_pos.append(m["support_pos"])
        stats.pointwise_support_neg.append(m["support_neg"])
    if stats.pointwise_accuracies:
        stats.mean_pointwise_accuracy = statistics.mean(stats.pointwise_accuracies)
    if stats.pointwise_precision_pos:
        stats.mean_pointwise_precision_pos = statistics.mean(stats.pointwise_precision_pos)
    if stats.pointwise_precision_neg:
        stats.mean_pointwise_precision_neg = statistics.mean(stats.pointwise_precision_neg)
    if stats.pointwise_recall_pos:
        stats.mean_pointwise_recall_pos = statistics.mean(stats.pointwise_recall_pos)
    if stats.pointwise_recall_neg:
        stats.mean_pointwise_recall_neg = statistics.mean(stats.pointwise_recall_neg)
    if stats.pointwise_f1_pos:
        stats.mean_pointwise_f1_pos = statistics.mean(stats.pointwise_f1_pos)
    if stats.pointwise_f1_neg:
        stats.mean_pointwise_f1_neg = statistics.mean(stats.pointwise_f1_neg)
    if stats.pointwise_f1_macro:
        stats.mean_pointwise_f1_macro = statistics.mean(stats.pointwise_f1_macro)
    return stats


# ---------------------------------------------------------------------------
# Artifact dir helpers
# ---------------------------------------------------------------------------

def _extract_model(artifact_dir: Path) -> str:
    """Extract llm_engine from debug_*.md in an artifact dir."""
    for f in artifact_dir.glob("debug_*.md"):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if m:
                return json.loads(m.group(1)).get("llm_engine", "Unknown")
        except Exception:
            pass
    return "Unknown"


def _extract_instances_path(artifact_dir: Path) -> str:
    """Extract Test Instances Path from accuracy_report.txt."""
    report = artifact_dir / "accuracy_report.txt"
    if not report.exists():
        return ""
    try:
        content = report.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Test Instances Path:\s*(.+)", content)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""


def _load_yaml_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoader) or {}


def _load_instances(instances_path_str: str) -> dict:
    p = Path(instances_path_str)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        return {}
    data = _load_yaml_file(p)
    return {str(k): v for k, v in data.items()}


def _detect_mode(artifact_dir: Path) -> str:
    """Return 'pairwise' or 'pointwise' by checking which run_* dirs exist."""
    if any(artifact_dir.glob("run_pairwise_*/")):
        return "pairwise"
    if any(artifact_dir.glob("run_pointwise_*/")):
        return "pointwise"
    # Fall back to accuracy_report.txt
    report = artifact_dir / "accuracy_report.txt"
    if report.exists():
        try:
            content = report.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"Test Mode:\s*(\w+)", content)
            if m:
                return m.group(1).lower()
        except Exception:
            pass
    return "pairwise"


# ---------------------------------------------------------------------------
# Core derivation
# ---------------------------------------------------------------------------

def _derive_pairwise_artifact(
    source_art_dir: Path,
    new_ablation_name: str,
    model: str,
    instances_path_str: str,
    inputs: dict,
    target_ablation: str,
    output_root: Path,
    n_subsamples: int,
) -> Path | None:
    records = _collect_pairwise_records(source_art_dir)
    if not records:
        print(f"    No pairwise batch records in {source_art_dir.name}")
        return None

    bidirectional = ABLATION_CONFIGS[target_ablation]["bidirectional"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_art_dir = (
        output_root / new_ablation_name / f"{new_ablation_name}-{model}"
        / "accuracy_test_artifacts" / timestamp
    )
    new_art_dir.mkdir(parents=True, exist_ok=True)
    (new_art_dir / "debug_args.md").write_text(
        f'```json\n{{"llm_engine": "{model}"}}\n```\n', encoding="utf-8"
    )

    run_results: list[dict] = []
    for k in range(n_subsamples):
        comps = _build_pairwise_subrun(records, k, bidirectional)
        if not comps:
            print(f"    [!] No comparisons at call_index={k} for {new_ablation_name}/{model}")
            continue
        results = aggregate_unidirectional_comparisons(comps)
        if inputs:
            _enrich_pairwise_with_gt(results, inputs)

        run_dir = new_art_dir / f"run_pairwise_{k}"
        run_dir.mkdir(exist_ok=True)
        with open(run_dir / "scores.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        run_results.append(results)

    if not run_results:
        return None

    stats = _compute_pairwise_stats(run_results, inputs) if inputs else ExperimentStats()
    empty = ExperimentStats.empty()
    save_report(
        output_file=str(new_art_dir / "accuracy_report.txt"),
        debug_file=str(new_art_dir / "debug_accuracy_report.txt"),
        rr_stats=empty, swiss_stats=empty, random_stats=empty,
        n_runs=len(run_results),
        test_inputs_path=instances_path_str,
        num_instances=len(inputs),
        modes=["pairwise"],
        test_mode="pairwise",
        pairwise_stats=stats,
    )
    print(
        f"    {new_ablation_name}/{model} [pairwise]: "
        f"acc={stats.mean_pairwise_accuracy:.4f}  runs={len(run_results)}"
    )
    return new_art_dir


def _derive_pointwise_artifact(
    source_art_dir: Path,
    new_ablation_name: str,
    model: str,
    instances_path_str: str,
    inputs: dict,
    output_root: Path,
    n_subsamples: int,
) -> Path | None:
    records = _collect_pointwise_records(source_art_dir)
    if not records:
        print(f"    No pointwise batch records in {source_art_dir.name}")
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_art_dir = (
        output_root / new_ablation_name / f"{new_ablation_name}-{model}"
        / "accuracy_test_artifacts" / timestamp
    )
    new_art_dir.mkdir(parents=True, exist_ok=True)
    (new_art_dir / "debug_args.md").write_text(
        f'```json\n{{"llm_engine": "{model}"}}\n```\n', encoding="utf-8"
    )

    run_results: list[dict] = []
    for k in range(n_subsamples):
        results = _build_pointwise_subrun(records, k)
        if not results:
            print(f"    [!] No pointwise predictions at call_index={k} for {new_ablation_name}/{model}")
            continue
        if inputs:
            _enrich_pointwise_with_labels(results, inputs)

        run_dir = new_art_dir / f"run_pointwise_{k}"
        run_dir.mkdir(exist_ok=True)
        with open(run_dir / "scores.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        run_results.append(results)

    if not run_results:
        return None

    stats = _compute_pointwise_stats(run_results, inputs) if inputs else ExperimentStats()
    empty = ExperimentStats.empty()
    save_report(
        output_file=str(new_art_dir / "accuracy_report.txt"),
        debug_file=str(new_art_dir / "debug_accuracy_report.txt"),
        rr_stats=empty, swiss_stats=empty, random_stats=empty,
        n_runs=len(run_results),
        test_inputs_path=instances_path_str,
        num_instances=len(inputs),
        modes=["pointwise"],
        test_mode="pointwise",
        pointwise_stats=stats,
    )
    print(
        f"    {new_ablation_name}/{model} [pointwise]: "
        f"acc={stats.mean_pointwise_accuracy:.4f}  runs={len(run_results)}"
    )
    return new_art_dir


def _track_name(source_ablation_name: str, source_suffix: str, keep_source_suffix: bool) -> str:
    """The derived ablation's prefix: the source name, minus its suffix unless kept.

    Stripping keeps ``X_current`` → ``X_mec_k_1`` (the suffix only marked the
    baseline).  Keeping is right when the suffix names a real judge property, so
    ``X_retrieval`` → ``X_retrieval_mec_k_1`` stays distinguishable from the run
    derived off ``X_current``.
    """
    if keep_source_suffix or not source_suffix or not source_ablation_name.endswith(source_suffix):
        return source_ablation_name
    return source_ablation_name[: -len(source_suffix)]


def derive_for_artifact_dir(
    source_art_dir: Path,
    source_ablation_name: str,
    target_ablation: str,
    output_root: Path,
    n_subsamples: int = 3,
    source_suffix: str = "_current",
    keep_source_suffix: bool = False,
) -> Path | None:
    """Derive sub-run artifact dir from a single source artifact dir."""
    cfg = ABLATION_CONFIGS[target_ablation]
    mode = _detect_mode(source_art_dir)

    # unidirectional only applies to pairwise
    if target_ablation == "unidirectional" and mode != "pairwise":
        return None

    instances_path_str = _extract_instances_path(source_art_dir)
    inputs = _load_instances(instances_path_str) if instances_path_str else {}
    model = _extract_model(source_art_dir)

    # e.g. "pairwise-vanilla-ai_current" → "pairwise-vanilla-ai_mec_k_1"
    track = _track_name(source_ablation_name, source_suffix, keep_source_suffix)
    new_ablation_name = f"{track}_{target_ablation}"

    if mode == "pairwise":
        return _derive_pairwise_artifact(
            source_art_dir, new_ablation_name, model, instances_path_str, inputs,
            target_ablation, output_root, n_subsamples,
        )
    elif mode == "pointwise":
        return _derive_pointwise_artifact(
            source_art_dir, new_ablation_name, model, instances_path_str, inputs,
            output_root, n_subsamples,
        )
    else:
        print(f"    Skipping unsupported mode '{mode}' for {source_art_dir.name}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Derive mec_k_1 / unidirectional sub-runs from an existing 'current' ablation sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("source_dirs", nargs="+", help="Source ablation sweep directories")
    ap.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: output/ablation_sweeps/derived_{timestamp})",
    )
    ap.add_argument(
        "--ablations", nargs="+",
        default=list(ABLATION_CONFIGS.keys()),
        choices=list(ABLATION_CONFIGS.keys()),
        help="Which ablations to derive (default: all)",
    )
    ap.add_argument(
        "--n-subsamples", type=int, default=3,
        help="Number of independent sub-runs to extract per ablation (default: 3)",
    )
    ap.add_argument(
        "--source-suffix", default="_current",
        help="Derive from the sweep rows whose ablation name ends with this "
             "(default: _current; pass '' to use every row)",
    )
    ap.add_argument(
        "--keep-source-suffix", action="store_true",
        help="Keep --source-suffix in the derived ablation name instead of stripping it. "
             "Use it when the suffix names a judge property (e.g. _retrieval) rather than "
             "marking the baseline, so the derived run stays distinguishable.",
    )
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (_PROJECT_ROOT / "output" / "ablation_sweeps" / f"derived_{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # (source_ablation_name, target_ablation) -> list of new artifact dir paths
    collected: dict[tuple[str, str], list[str]] = defaultdict(list)

    for source_path in args.source_dirs:
        source_dir = Path(source_path).resolve()
        if not source_dir.is_dir():
            print(f"Warning: {source_dir} is not a directory — skipping")
            continue

        summary_path = source_dir / "ablation_summary.json"
        if not summary_path.exists():
            print(f"Warning: No ablation_summary.json in {source_dir} — skipping")
            continue

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        current_rows = [
            r for r in summary.get("runs", [])
            if str(r.get("ablation", "")).endswith(args.source_suffix)
        ]
        if not current_rows:
            print(f"Warning: No '{args.source_suffix}' rows in {source_dir} — skipping")
            continue

        print(f"\n{source_dir.name}")

        for row in current_rows:
            source_ablation_name = row["ablation"]
            for target_ablation in args.ablations:
                print(f"  {source_ablation_name} → {target_ablation}")
                for art_dir_str in row.get("artifact_dirs", []):
                    art_dir = Path(art_dir_str).resolve()
                    if not art_dir.exists():
                        print(f"    Skipping missing: {art_dir}")
                        continue
                    new_dir = derive_for_artifact_dir(
                        source_art_dir=art_dir,
                        source_ablation_name=source_ablation_name,
                        target_ablation=target_ablation,
                        output_root=output_dir,
                        n_subsamples=args.n_subsamples,
                        source_suffix=args.source_suffix,
                        keep_source_suffix=args.keep_source_suffix,
                    )
                    if new_dir:
                        collected[(source_ablation_name, target_ablation)].append(str(new_dir))

    # Build and write ablation_summary.json
    runs: list[dict] = []
    for (source_ablation_name, target_ablation), art_dirs in sorted(collected.items()):
        track = _track_name(source_ablation_name, args.source_suffix, args.keep_source_suffix)
        new_ablation_name = f"{track}_{target_ablation}"
        runs.append({
            "ablation": new_ablation_name,
            "description": ABLATION_CONFIGS[target_ablation]["description"],
            "status": "OK",
            "artifact_dirs": art_dirs,
        })

    summary_out = output_dir / "ablation_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "runs": runs}, f, indent=2)

    print(f"\nOutput: {output_dir}")
    print(f"Pass to merge_ablation_runs.py alongside the original sweep directories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
