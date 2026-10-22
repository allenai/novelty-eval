"""
Paper blocklist filtering and metric recomputation.

Loads paper_blocklist.yaml, derives which test-instance indices to exclude,
and recomputes pairwise / pointwise metrics on the filtered subset.
"""
import json
import re
import statistics
import sys
import yaml
import threading
from pathlib import Path

# --- Global caches ---
_YAML_CACHE: dict[Path, tuple[float, dict]] = {}
_YAML_CACHE_LOCK = threading.Lock()

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


def _load_yaml_cached(path: Path | str) -> dict:
    """Read and parse a YAML file with CSafeLoader, caching the result with mtime validation."""
    p = Path(path).resolve()
    if not p.exists():
        return {}
    
    mtime = p.stat().st_mtime
    with _YAML_CACHE_LOCK:
        if p in _YAML_CACHE:
            cached_mtime, data = _YAML_CACHE[p]
            if cached_mtime == mtime:
                return data

    try:
        with open(p, encoding="utf-8") as f:
            data = yaml.load(f, Loader=SafeLoader) or {}
    except Exception:
        data = {}

    with _YAML_CACHE_LOCK:
        _YAML_CACHE[p] = (mtime, data)
    return data


sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from novelty_eval.metrics import (
    compute_pairwise_metrics as _compute_pairwise_metrics,
    compute_pointwise_metrics as _compute_pointwise_metrics,
)
from novelty_eval.analysis.artifacts import (
    _extract_instances_path,
    _extract_instance_id_from_key,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BLOCKLIST_FILE = (
    _PROJECT_ROOT / "src" / "novelty_eval" / "benchmark_data" / "paper_blocklist.yaml"
)


def _load_paper_blocklist() -> set[str]:
    """Return lower-cased paper titles from paper_blocklist.yaml, or empty set."""
    if not _BLOCKLIST_FILE.exists():
        return set()
    data = _load_yaml_cached(_BLOCKLIST_FILE)
    return {t.lower() for t in data.get("blocked_titles", [])}


def _derive_turn_exclude_indices(instances_path: str, turn_n: int) -> dict[str, str]:
    """Return {idx: title} for pairwise instances where any NEGATIVE has turn_number == turn_n."""
    if not instances_path:
        return {}
    p = Path(instances_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    data = _load_yaml_cached(p)
    excluded: dict[str, str] = {}
    for idx_raw, entry in data.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata")
        if not meta or not isinstance(meta, dict):
            continue
        first_val = next(iter(meta.values()), None)
        if not isinstance(first_val, dict):
            continue  # pointwise flat format — not applicable
        idx = str(idx_raw)
        for idea_meta in meta.values():
            if not isinstance(idea_meta, dict):
                continue
            if idea_meta.get("type") == "NEGATIVE" and idea_meta.get("turn_number") == turn_n:
                excluded[idx] = idea_meta.get("title", "")
                break
    return excluded


def _instances_have_turn_numbers(instances_path: str) -> bool:
    """Return True if any idea in the pairwise instances YAML has a turn_number field."""
    if not instances_path:
        return False
    p = Path(instances_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    data = _load_yaml_cached(p)
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        for idea_meta in meta.values():
            if isinstance(idea_meta, dict) and "turn_number" in idea_meta:
                return True
    return False


def _derive_exclude_indices(instances_path: str, blocked_titles: set[str]) -> dict[int, str]:
    """
    Load a test instances YAML and return a mapping of index -> title for all
    instances where any participating paper title appears in blocked_titles.

    Handles both formats:
      - Pairwise/ranking: metadata is dict[idea_idx -> {title, ...}]
      - Pointwise:        metadata is a flat dict with a single 'title' key
    """
    if not instances_path or not blocked_titles:
        return {}
    p = Path(instances_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    data = _load_yaml_cached(p)

    excluded: dict[str, str] = {}
    for idx_raw, entry in data.items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata")
        if not meta or not isinstance(meta, dict):
            continue

        idx = str(idx_raw)

        first_val = next(iter(meta.values()), None)

        if isinstance(first_val, dict):
            # Pairwise/ranking: metadata keyed by idea index
            for idea_meta in meta.values():
                title = idea_meta.get("title", "")
                if title.lower() in blocked_titles:
                    excluded[idx] = title
                    break
        elif "title" in meta:
            # Pointwise: flat metadata with a single title
            title = meta["title"]
            if title.lower() in blocked_titles:
                excluded[idx] = title
    return excluded


def _recompute_pairwise_metrics(artifact_dir: Path, exclude_set: set[str], logger_inst=None) -> dict | None:
    """Load pairwise scores.json files, filter excluded indices, recompute metrics per run.

    Delegates to metrics.compute_pairwise_metrics so the computation logic stays
    in a single place.  calculate_pairwise_accuracy reads gt_winner from each
    comparison directly, so we backfill missing labels from test instances.
    """
    run_scores_paths = sorted(artifact_dir.glob("run_pairwise_*/scores.json"))
    if not run_scores_paths:
        return None

    # Load labels from the test instances YAML (batch scores.json may omit gt_winner).
    label_lookup: dict[str, list[int]] = {}
    instances_path = _extract_instances_path(artifact_dir)
    if instances_path:
        p = Path(instances_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        inst_data = _load_yaml_cached(p)
        for k, v in inst_data.items():
            if isinstance(v, dict) and 'expected_winners' in v:
                label_lookup[str(k)] = v['expected_winners']

    per_run_accs_wt: list[float] = []
    per_run_accs_strict: list[float] = []
    per_run_accs_nt: list[float] = []
    per_run_ties: list[int] = []
    per_run_support: list[int] = []
    per_run_support_nt: list[int] = []

    for scores_path in run_scores_paths:
        try:
            with open(scores_path) as f:
                scores = json.load(f)
        except Exception:
            continue

        # Back-fill gt_winner missing from batch scores.json.
        if label_lookup:
            from novelty_eval.run_benchmark import _annotate_gt_winner
            for k, v in scores.items():
                iid = _extract_instance_id_from_key(k)
                if isinstance(v, dict) and iid is not None and iid in label_lookup:
                    comps = v.get('comparisons', [])
                    if comps and 'gt_winner' not in comps[0]:
                        expected_winners = set(str(w) for w in label_lookup[iid])
                        _annotate_gt_winner(comps, expected_winners)

        # Reclassify failure-ties (winner=2 + empty mec_details) → wrong answer
        _reclassified_pids: list[str] = []
        for k, v in scores.items():
            if not isinstance(v, dict):
                continue
            _iid = _extract_instance_id_from_key(k) or str(k)
            for comp in v.get("comparisons", []):
                if comp.get("winner") == 2:
                    mec = comp.get("mec_details") or {}
                    if mec and all(
                        isinstance(dv, dict) and len(dv.get("fwd_votes", [1])) == 0
                        for dv in mec.values()
                    ):
                        gt = comp.get("gt_winner")
                        if gt is not None:
                            comp["winner"] = 1 - int(gt)  # flip to wrong answer
                            _reclassified_pids.append(_iid)
        if _reclassified_pids and logger_inst:
            logger_inst.warning(
                "[recompute / pairwise] %d failure-tie(s) reclassified as wrong "
                "(all MEC calls failed) — artifact=%s, run=%s, pids=%s",
                len(_reclassified_pids), artifact_dir.name, scores_path.name, _reclassified_pids,
            )

        # Filter scores: exclude indices in exclude_set
        filtered = {}
        for k, v in scores.items():
            iid = _extract_instance_id_from_key(k)
            if iid is None or iid not in exclude_set:
                filtered[k] = v

        # Inject pids from instances YAML that are absent from this run's scores.
        # These are genuine failures (batch drops, all-call failures) — count as wrong.
        _injected_pids: list[str] = []
        if label_lookup:
            present_pids = {_extract_instance_id_from_key(k) or str(k) for k in filtered}
            for pid, expected_winners in label_lookup.items():
                if pid not in present_pids and pid not in exclude_set:
                    gt_winner = int(expected_winners[0]) if expected_winners else 0
                    filtered[pid] = {
                        "comparisons": [{
                            "idea_0": "0", "idea_1": "1",
                            "winner": 1 - gt_winner,
                            "gt_winner": gt_winner,
                        }]
                    }
                    _injected_pids.append(pid)
        if _injected_pids and logger_inst:
            logger_inst.warning(
                "[recompute / pairwise] %d pid(s) absent from run → injected as wrong — "
                "artifact=%s, run=%s, instances=%s, pids=%s",
                len(_injected_pids), artifact_dir.name, scores_path.name,
                _extract_instances_path(artifact_dir) or str(artifact_dir), _injected_pids,
            )

        if not filtered:
            continue

        # _iter_labeled_results requires expected_winners in inputs; the actual
        # value is unused because pairwise accuracy reads gt_winner from scores.
        synthetic_inputs = {k: {"expected_winners": [0]} for k in filtered}
        m = _compute_pairwise_metrics(filtered, synthetic_inputs)
        if m["support"] == 0:
            continue

        per_run_accs_wt.append(m["accuracy_with_ties"])
        per_run_accs_strict.append(m["accuracy_strict"])
        per_run_accs_nt.append(m["accuracy_without_ties"])
        per_run_ties.append(m["n_ties"])
        per_run_support.append(m["support"])
        per_run_support_nt.append(m["support_without_ties"])

    if not per_run_accs_wt:
        return None

    return {
        "pairwise_accuracy": statistics.mean(per_run_accs_wt),
        "pairwise_accuracy_strict": statistics.mean(per_run_accs_strict),
        "pairwise_accuracy_no_ties": statistics.mean(per_run_accs_nt),
        "n_ties": statistics.mean(per_run_ties),
        "support": statistics.mean(per_run_support),
        "support_without_ties": statistics.mean(per_run_support_nt),
        "per_run_accuracies_with_ties": per_run_accs_wt,
        "per_run_accuracies_strict": per_run_accs_strict,
        "per_run_accuracies_no_ties": per_run_accs_nt,
        "per_run_ties": per_run_ties,
        "per_run_support": per_run_support,
        "per_run_support_no_ties": per_run_support_nt,
        "excluded": sorted(exclude_set),
    }


def _recompute_pointwise_metrics(artifact_dir: Path, exclude_set: set[str], logger_inst=None) -> dict | None:
    """Load pointwise scores.json files, filter excluded indices, recompute metrics per run.

    Delegates to metrics.compute_pointwise_metrics so the computation logic
    stays in a single place.  compute_pointwise_metrics ignores inputs entirely.
    """
    run_scores_paths = sorted(artifact_dir.glob("run_pointwise_*/scores.json"))
    if not run_scores_paths:
        return None

    # Load labels from the test instances YAML (batch scores.json omits 'label').
    label_lookup: dict[str, str] = {}
    instances_path = _extract_instances_path(artifact_dir)
    if instances_path:
        p = Path(instances_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        inst_data = _load_yaml_cached(p)
        for k, v in inst_data.items():
            if isinstance(v, dict) and 'label' in v:
                label_lookup[str(k)] = v['label']

    per_run_accs: list[float] = []
    per_run_f1_macro: list[float] = []
    per_run_f1_pos: list[float] = []
    per_run_f1_neg: list[float] = []
    per_run_prec_pos: list[float] = []
    per_run_prec_neg: list[float] = []
    per_run_rec_pos: list[float] = []
    per_run_rec_neg: list[float] = []
    per_run_support: list[int] = []

    for scores_path in run_scores_paths:
        try:
            with open(scores_path) as f:
                scores = json.load(f)
        except Exception:
            continue

        # Back-fill labels missing from batch scores.json.
        if label_lookup:
            for k, v in scores.items():
                iid = _extract_instance_id_from_key(k)
                if isinstance(v, dict) and 'label' not in v and iid is not None and iid in label_lookup:
                    v['label'] = label_lookup[iid]

        # Filter scores: exclude indices in exclude_set
        filtered = {}
        for k, v in scores.items():
            iid = _extract_instance_id_from_key(k)
            if iid is None or iid not in exclude_set:
                filtered[k] = v

        # Inject pids absent from this run as wrong predictions.
        # These are genuine failures (batch drops, all-call failures) — count as wrong.
        _injected_pids_ptw: list[str] = []
        if label_lookup:
            present_pids = {_extract_instance_id_from_key(k) or str(k) for k in filtered}
            for pid, label in label_lookup.items():
                if pid not in present_pids and pid not in exclude_set:
                    filtered[pid] = {
                        "prediction": 0 if label == "POSITIVE" else 1,
                        "label": label,
                    }
                    _injected_pids_ptw.append(pid)
        if _injected_pids_ptw and logger_inst:
            logger_inst.warning(
                "[recompute / pointwise] %d pid(s) absent from run → injected as wrong — "
                "artifact=%s, run=%s, instances=%s, pids=%s",
                len(_injected_pids_ptw), artifact_dir.name, scores_path.name,
                _extract_instances_path(artifact_dir) or str(artifact_dir), _injected_pids_ptw,
            )

        if not filtered:
            continue

        m = _compute_pointwise_metrics(filtered, {})
        support = m["support_pos"] + m["support_neg"]
        if support == 0:
            continue

        per_run_accs.append(m["accuracy"])
        per_run_f1_macro.append(m["f1_macro"])
        per_run_f1_pos.append(m["f1_pos"])
        per_run_f1_neg.append(m["f1_neg"])
        per_run_prec_pos.append(m["precision_pos"])
        per_run_prec_neg.append(m["precision_neg"])
        per_run_rec_pos.append(m["recall_pos"])
        per_run_rec_neg.append(m["recall_neg"])
        per_run_support.append(support)

    if not per_run_accs:
        return None

    return {
        "accuracy": statistics.mean(per_run_accs),
        "f1_macro": statistics.mean(per_run_f1_macro),
        "f1_pos": statistics.mean(per_run_f1_pos),
        "f1_neg": statistics.mean(per_run_f1_neg),
        "precision_pos": statistics.mean(per_run_prec_pos),
        "precision_neg": statistics.mean(per_run_prec_neg),
        "recall_pos": statistics.mean(per_run_rec_pos),
        "recall_neg": statistics.mean(per_run_rec_neg),
        "support": statistics.mean(per_run_support),
        "per_run_accuracies": per_run_accs,
        "per_run_f1_macro": per_run_f1_macro,
        "per_run_f1_pos": per_run_f1_pos,
        "per_run_f1_neg": per_run_f1_neg,
        "excluded": sorted(exclude_set),
    }


def _metrics_to_experiment_stats(mode: str, m: dict):
    from novelty_eval.experiment_stats import ExperimentStats as _ExperimentStats
    stats = _ExperimentStats()
    if mode == "pairwise":
        stats.mean_pairwise_accuracy = m["pairwise_accuracy"]
        stats.mean_pairwise_accuracy_strict = m["pairwise_accuracy_strict"]
        stats.mean_pairwise_accuracy_no_ties = m["pairwise_accuracy_no_ties"]
        stats.mean_pairwise_n_ties = m["n_ties"]
        stats.mean_pairwise_support = m["support"]
        stats.mean_pairwise_support_no_ties = m["support_without_ties"]
        stats.pairwise_accuracies = m["per_run_accuracies_with_ties"]
        stats.pairwise_accuracies_strict = m["per_run_accuracies_strict"]
        stats.pairwise_accuracies_no_ties = m["per_run_accuracies_no_ties"]
        stats.pairwise_n_ties = m["per_run_ties"]
        stats.pairwise_support = m["per_run_support"]
        stats.pairwise_support_no_ties = m["per_run_support_no_ties"]
    elif mode == "pointwise":
        stats.mean_pointwise_accuracy = m["accuracy"]
        stats.mean_pointwise_f1_macro = m["f1_macro"]
        stats.mean_pointwise_f1_pos = m["f1_pos"]
        stats.mean_pointwise_f1_neg = m["f1_neg"]
        stats.mean_pointwise_precision_pos = m["precision_pos"]
        stats.mean_pointwise_precision_neg = m["precision_neg"]
        stats.mean_pointwise_recall_pos = m["recall_pos"]
        stats.mean_pointwise_recall_neg = m["recall_neg"]
        stats.pointwise_accuracies = m["per_run_accuracies"]
        stats.pointwise_f1_macro = m["per_run_f1_macro"]
        stats.pointwise_f1_pos = m["per_run_f1_pos"]
        stats.pointwise_f1_neg = m["per_run_f1_neg"]
    return stats


def _write_filtered_accuracy_report(
    artifact_dir: Path,
    mode: str,
    metrics: dict,
    original_report: Path,
    out_filename: str = "filtered_accuracy_report.txt",
) -> Path:
    """Write filtered_accuracy_report.txt with recomputed metrics, preserving the header."""
    from novelty_eval.report_writer import (
        _write_pairwise_stats,
        _write_pointwise_stats,
        _write_pairwise_summary,
        _write_pointwise_summary,
    )

    try:
        original = original_report.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        original = ""

    # Keep everything before the mode section header
    if mode == "pairwise":
        split_pat = re.compile(r'\nPairwise\n')
    elif mode == "pointwise":
        split_pat = re.compile(r'\nPointwise\n')
    else:
        split_pat = None

    header = original
    if split_pat:
        match = split_pat.search(original)
        if match:
            header = original[:match.start()]

    # Update the instance count in the header to reflect excluded items.
    if mode == "pointwise" and "support" in metrics:
        header = re.sub(
            r'Number of Test Instances Processed: \d+',
            f'Number of Test Instances Processed: {metrics["support"]}',
            header,
        )

    stats = _metrics_to_experiment_stats(mode, metrics)
    excluded_str = str(metrics["excluded"])

    out_path = artifact_dir / out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header.rstrip())
        f.write(f"\n\n# Filtered: excluded abstract indices {excluded_str}\n\n")
        if mode == "pairwise":
            _write_pairwise_stats(f, "Pairwise", stats)
            f.write("Summary\n-------\n")
            _write_pairwise_summary(f, stats)
        elif mode == "pointwise":
            _write_pointwise_stats(f, "Pointwise", stats)
            f.write("Summary\n-------\n")
            _write_pointwise_summary(f, stats)
    return out_path
