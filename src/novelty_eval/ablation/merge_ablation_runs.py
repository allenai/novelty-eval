#!/usr/bin/env python3
"""
Merge Ablation Runs
===================
Collect artifact directories from one or more previous ablation (or sweep)
output directories and generate a single unified comparison report.

Usage:
    python merge_ablation_runs.py <dir1> <dir2> ... [--output-dir PATH] [--report-name NAME]

Examples:
    # Merge two ablation runs
    python merge_ablation_runs.py \
        output/ablation_sweeps/20260319_161416 \
        output/ablation_sweeps/20260319_170000

    # Merge and write report to a specific location
    python merge_ablation_runs.py \
        output/ablation_sweeps/20260319_161416 \
        output/ablation_sweeps/20260319_170000 \
        --output-dir output/ablation_sweeps/merged \
        --report-name my_comparison.md
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# --- Global caches ---
_METRICS_CACHE: dict[tuple[Path, str], dict | None] = {}
_YAML_CACHE: dict[Path, tuple[float, dict]] = {}
_YAML_CACHE_LOCK = threading.Lock()
# Keyed by (setup, baseline_name, lookup_abl, model, report_file).
# _process_single_model populates this so repeated callers (per-ablation reports
# and the overview heatmap) share the same bootstrap results.
_BOOTSTRAP_CACHE: dict[tuple, dict] = {}
_BOOTSTRAP_CACHE_LOCK = threading.Lock()
# Optional canonical ablation name to use as the comparison baseline instead of
# the default "current". Set once in main() from the merge config's `baseline:`
# key (e.g. "ai_researcher_base"). Consulted by _resolve_baseline_name and
# _baseline_canonical_names so the override is both used as the reference and
# skipped as a comparison subject.
_BASELINE_OVERRIDE: str | None = None

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

from logging_utils import setup_logger
from novelty_eval.ablation.config import (
    _load_ablations_yaml_data,
    _expand_ablation_variants,
    _load_descriptions_from_yaml,
    _load_before_descriptions_from_yaml,
    _load_mismatch_suppressed_ablations,
    _load_significance_alternatives_from_yaml,
    _build_canonical_key_map,
    _load_legacy_name_aliases,
    _is_baseline_ablation,
    _get_test_inputs_path,
    _load_test_inputs,
    _flatten_dict,
    _build_config_diff_table,
)
from novelty_eval.analysis.artifacts import (
    _relative_path_label,
    _extract_instance_id_from_key,
    _collect_artifact_dirs,
    _read_ablation_rows,
    _extract_refusal_indices,
    _extract_instances_path,
    _instances_content_matches,
    _extract_model_from_artifact_dir,
    _extract_mode_from_artifact_dir,
    _extract_exclude_set_from_filtered_report,
    _extract_raw_outcomes_for_dir,
)
from novelty_eval.analysis.filtering import (
    _load_paper_blocklist,
    _derive_exclude_indices,
    _derive_turn_exclude_indices,
    _instances_have_turn_numbers,
    _recompute_pairwise_metrics,
    _recompute_pointwise_metrics,
    _metrics_to_experiment_stats,
    _write_filtered_accuracy_report,
)
from novelty_eval.figures.viz import (
    _generate_aggregated_delta_heatmap,
    _generate_overview_heatmap,
    _generate_unified_heatmap,
)
from novelty_eval.analysis.stats_one_pass import (
    compute_bootstrap_results_pointwise_all,
    compute_bootstrap_results_pairwise_all,
)
from novelty_eval.figures import unified_chart_data

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RECHART_SCRIPT = (
    _PROJECT_ROOT / "src" / "novelty_eval" / "figures" / "rechart_unified.py"
)
_GENERATE_REPORT_SCRIPT = (
    _PROJECT_ROOT / "src" / "novelty_eval" / "analysis" / "generate_report.py"
)


def _backfill_labels_inplace(run_scores_list: list[dict], instances_path: str, setup: str) -> None:
    """Mutate scores dicts in place to back-fill missing labels from instances YAML."""
    if not instances_path:
        return
    p = Path(instances_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        return
    inst = _load_yaml_cached(p)
    label_lkp: dict[str, str] = {}
    gt_lkp: dict[str, list] = {}
    for k, v in inst.items():
        if isinstance(v, dict):
            if setup == "pointwise" and "label" in v:
                label_lkp[str(k)] = v["label"]
            if setup == "pairwise" and "expected_winners" in v:
                gt_lkp[str(k)] = v["expected_winners"]
    for scores in run_scores_list:
        for k, v in scores.items():
            if not isinstance(v, dict):
                continue
            pid = _extract_instance_id_from_key(k)
            if pid is None:
                continue
            if setup == "pointwise" and "label" not in v and pid in label_lkp:
                v["label"] = label_lkp[pid]
            if setup == "pairwise":
                comps = v.get("comparisons", [])
                if comps and "gt_winner" not in comps[0] and pid in gt_lkp:
                    from novelty_eval.run_benchmark import _annotate_gt_winner
                    _annotate_gt_winner(comps, {str(w) for w in gt_lkp[pid]})


def _generate_refusal_report(
    retrieval_index: dict,
    yaml_descriptions: dict,
    output_dir: Path,
    logger_inst=None,
) -> Path | None:
    """
    Generate a markdown report showing, per data file, which abstract indices
    were refused/blocked due to safety/refusal errors, and which ablation runs
    triggered those refusals.

    Structure:
        data_file → abstract_index → [(mode, ablation_description, model), ...]
    """
    # Collect: for each artifact dir, the instances path + refused indices
    # retrieval_index maps (mode, ablation_name, model) -> artifact_dir path
    seen_artifact_dirs: set[str] = set()

    # data_file_path -> {abstract_idx -> [(mode, ablation_desc, model)]}
    from collections import defaultdict
    data_file_refusals: dict[str, dict[int, list[tuple[str, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for (mode, ablation_name, model), artifact_dir_str in retrieval_index.items():
        if artifact_dir_str in seen_artifact_dirs:
            continue
        seen_artifact_dirs.add(artifact_dir_str)

        artifact_dir = Path(artifact_dir_str)
        refused = _extract_refusal_indices(artifact_dir)
        if not refused:
            continue

        instances_path = _extract_instances_path(artifact_dir)
        if not instances_path:
            instances_path = str(artifact_dir)

        # Resolve a short, readable label for the ablation
        desc = yaml_descriptions.get(ablation_name, "")
        if not desc:
            parts = ablation_name.split("_", 1)
            stripped = parts[1] if len(parts) == 2 and parts[0] in ("pairwise", "ranking", "pointwise") else ablation_name
            desc = yaml_descriptions.get(stripped, ablation_name)

        for idx in refused:
            data_file_refusals[instances_path][idx].append((mode, desc, model))

    if not data_file_refusals:
        if logger_inst:
            logger_inst.info("No safety refusals found across any artifact dir — skipping refusal report.")
        return None

    total_events = sum(
        len(runs)
        for per_file in data_file_refusals.values()
        for runs in per_file.values()
    )
    total_abstracts = sum(len(per_file) for per_file in data_file_refusals.values())

    md_lines: list[str] = [
        "# Safety Refusal Report",
        "",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Abstract indices that were skipped in at least one ablation run due to a "
        "safety / content-filter refusal (`invalid_prompt`, `refusal`, "
        "`content_filter`, `content_policy_violation`).",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Data files with refusals | {len(data_file_refusals)} |",
        f"| Unique refused abstract indices (across all files) | {total_abstracts} |",
        f"| Total refusal events | {total_events} |",
        "",
    ]

    for instances_path in sorted(data_file_refusals):
        per_file = data_file_refusals[instances_path]
        short = _relative_path_label(instances_path)
        md_lines.append(f"## `{short}`")
        md_lines.append("")
        md_lines.append("| Abstract Index | Mode | Ablation | Judge Model |")
        md_lines.append("| --- | --- | --- | --- |")
        for idx in sorted(per_file):
            runs = per_file[idx]
            for i, (mode, desc, model) in enumerate(runs):
                idx_cell = str(idx) if i == 0 else ""
                md_lines.append(f"| {idx_cell} | {mode} | {desc} | {model} |")
        md_lines.append("")

    report_path = output_dir / "safety_refusal_report.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if logger_inst:
        logger_inst.info(f"Safety refusal report written to: {report_path}")
    return report_path


def _generate_blocklist_report(
    filtered_info: dict[str, dict[int, str]],
    output_dir: Path,
    logger_inst=None,
) -> Path | None:
    """
    Generate a markdown report showing which abstracts were filtered out
    due to being on the paper blocklist.
    """
    if not filtered_info:
        if logger_inst:
            logger_inst.info("No abstracts were filtered by blocklist.")
        return None

    md_lines: list[str] = [
        "# Blocklist Filtering Report",
        "",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Abstracts that were excluded from the filtered metrics because their "
        "titles were found in `paper_blocklist.yaml`.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Data files with filtered abstracts | {len(filtered_info)} |",
        f"| Total filtered abstracts | {sum(len(v) for v in filtered_info.values())} |",
        "",
    ]

    for instances_path in sorted(filtered_info):
        per_file = filtered_info[instances_path]
        short = _relative_path_label(instances_path)
        md_lines.append(f"## `{short}`")
        md_lines.append("")
        md_lines.append("| Abstract Index | Paper Title |")
        md_lines.append("| --- | --- |")
        for idx in sorted(per_file):
            title = per_file[idx]
            md_lines.append(f"| {idx} | {title} |")
        md_lines.append("")

    report_path = output_dir / "blocklist_filter_report.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if logger_inst:
        logger_inst.info(f"Blocklist filtering report written to: {report_path}")
    return report_path


def _derive_pointwise_partner_excluded(
    pointwise_instances_path: str,
    blocked_titles: set[str],
    logger_inst=None,
) -> dict[str, str]:
    """
    For a pointwise instances file, find instances corresponding to the pairwise
    partners of blocked papers and return {index -> title}, mirroring the
    return type of _derive_exclude_indices.

    Looks up partners from the canonical source pairwise YAML recorded in
    _instance_creation_config.yaml. Matches by idea TEXT primarily, but
    falls back to title+context+proximity matching to handle cases where
    idea texts were transformed (e.g. into plans).
    """
    if not pointwise_instances_path:
        return {}

    p = Path(pointwise_instances_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    # _instance_creation_config.yaml sits one level above the timestamped subdir,
    # or inside the timestamped dir, or in the ablation base dir.
    config_candidates = [
        p.parent / "_instance_creation_config.yaml",
        p.parent.parent / "_instance_creation_config.yaml",
        _PROJECT_ROOT / "output" / "benchmark_instances" / "ablation" / p.parent.parent.name / "_instance_creation_config.yaml",
        _PROJECT_ROOT / "output" / "iclr_test_instances" / "ablation" / p.parent.parent.name / "_instance_creation_config.yaml",
    ]
    config_path = None
    for cand in config_candidates:
        if cand.exists():
            config_path = cand
            break

    if not config_path:
        return {}

    cfg = _load_yaml_cached(config_path)

    source_path_str = cfg.get("derive_pointwise_from_pairwise")
    if not source_path_str:
        return {}

    source_path = Path(source_path_str)
    if not source_path.is_absolute():
        source_path = _PROJECT_ROOT / source_path
        
    if not source_path.exists():
        return {}

    excluded_pair_indices = set(_derive_exclude_indices(str(source_path), blocked_titles).keys())
    if not excluded_pair_indices:
        return {}

    src_data = _load_yaml_cached(source_path)

    # Collect partner idea texts and metadata from excluded pairwise instances.
    partner_texts: dict[str, str] = {}  # idea_text -> title
    # Mapping of (blocked_title, context) -> list of (partner_title, partner_context)
    partner_metadata: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    # Track which partners we are looking for to warn if they are missing
    all_expected_partners: set[str] = set()

    for idx in excluded_pair_indices:
        # Try both string and int lookup for idx
        entry = src_data.get(idx)
        if entry is None and str(idx).isdigit():
            entry = src_data.get(int(idx))
            
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata", {})
        ideas = entry.get("ideas", {})
        context = entry.get("context", "")

        blocked_in_pair = []
        partners_in_pair = []
        for k, idea_meta in meta.items():
            if not isinstance(idea_meta, dict):
                continue
            title = idea_meta.get("title", "")
            if not title:
                continue
            
            if title.lower() in blocked_titles:
                blocked_in_pair.append(title)
            else:
                partners_in_pair.append((title, context))
                all_expected_partners.add(title)
                # Try both k and its string/int counterpart for idea_text
                idea_text = ideas.get(k)
                if idea_text is None:
                    alt_k = int(k) if isinstance(k, str) and k.isdigit() else str(k)
                    idea_text = ideas.get(alt_k)
                if idea_text:
                    partner_texts[str(idea_text).strip()] = title
        
        for bt in blocked_in_pair:
            for pt_info in partners_in_pair:
                partner_metadata[(bt.lower(), context)].append(pt_info)

    # Scan the pointwise YAML and match by idea text (exact match) 
    # OR by title+context proximity (fallback for transformed ideas).
    ptw_data = _load_yaml_cached(p)

    excluded: dict[str, str] = {}
    
    # 1. Primary pass: Exact text matching
    if partner_texts:
        for idx_raw, entry in ptw_data.items():
            if not isinstance(entry, dict):
                continue
            idx = str(idx_raw)
            idea_text = str(entry.get("idea", "")).strip()
            if idea_text in partner_texts:
                excluded[idx] = partner_texts[idea_text]

    # 2. Fallback pass: Title/Context matching with neighbor heuristic
    if partner_metadata:
        for idx_raw, entry in ptw_data.items():
            if not isinstance(entry, dict):
                continue
            idx_int = int(idx_raw)
            meta = entry.get("metadata", {})
            if not isinstance(meta, dict):
                continue
            title = meta.get("title", "")
            context = entry.get("context", "")
            
            lookup_key = (title.lower(), context)
            if lookup_key in partner_metadata:
                potential_partners = partner_metadata[lookup_key]
                for adj_idx in [idx_int - 1, idx_int + 1]:
                    adj_str = str(adj_idx)
                    if adj_str in ptw_data and adj_str not in excluded:
                        adj_entry = ptw_data[adj_str]
                        adj_title = adj_entry.get("metadata", {}).get("title", "")
                        adj_context = adj_entry.get("context", "")
                        for pt, pc in potential_partners:
                            if adj_title == pt and adj_context == pc:
                                excluded[adj_str] = adj_title

    # 3. Validation: Warn if any partners identified in source were not found in pointwise
    found_titles = set(excluded.values())
    missing = all_expected_partners - found_titles
    if missing and logger_inst:
        logger_inst.warning(
            f"Pointwise partner matching failed for {len(missing)} paper(s). "
            f"Blocked papers were found in source, but their partners were not "
            f"identified in {pointwise_instances_path}. Missing: {missing}"
        )

    return excluded


def _derive_turn_pointwise_direct(instances_path: str, turn_n: int) -> dict[str, str]:
    """Return {idx: title} for pointwise NEGATIVE instances with turn_number == turn_n."""
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
        if entry.get("label") != "NEGATIVE":
            continue
        meta = entry.get("metadata", {})
        if isinstance(meta, dict) and meta.get("turn_number") == turn_n:
            excluded[str(idx_raw)] = meta.get("title", "")
    return excluded


def _derive_turn_pointwise_partner_excluded(
    pointwise_instances_path: str,
    turn_n: int,
    logger_inst=None,
) -> dict[str, str]:
    """
    For a pointwise instances file, return {idx: title} for POSITIVE instances
    whose pairwise partner was excluded by turn_n filtering.

    Uses the same _instance_creation_config.yaml lookup as
    _derive_pointwise_partner_excluded to find the source pairwise YAML.
    Matches by idea text first (exact), then falls back to title+context metadata
    matching to handle cases where idea texts were transformed (e.g. into plan form).
    """
    if not pointwise_instances_path:
        return {}
    p = Path(pointwise_instances_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    config_candidates = [
        p.parent / "_instance_creation_config.yaml",
        p.parent.parent / "_instance_creation_config.yaml",
        _PROJECT_ROOT / "output" / "benchmark_instances" / "ablation"
            / p.parent.parent.name / "_instance_creation_config.yaml",
        _PROJECT_ROOT / "output" / "iclr_test_instances" / "ablation"
            / p.parent.parent.name / "_instance_creation_config.yaml",
    ]
    config_path = next((c for c in config_candidates if c.exists()), None)
    if not config_path:
        return {}

    cfg = _load_yaml_cached(config_path)
    source_path_str = cfg.get("derive_pointwise_from_pairwise")
    if not source_path_str:
        return {}
    source_path = Path(source_path_str)
    if not source_path.is_absolute():
        source_path = _PROJECT_ROOT / source_path
    if not source_path.exists():
        return {}

    excluded_pair_indices = set(_derive_turn_exclude_indices(str(source_path), turn_n).keys())
    if not excluded_pair_indices:
        return {}

    src_data = _load_yaml_cached(source_path)

    partner_texts: dict[str, str] = {}  # idea_text -> title
    partner_metadata: dict[tuple[str, str], str] = {}  # (title, context) -> title
    for idx in excluded_pair_indices:
        entry = src_data.get(idx)
        if entry is None and str(idx).isdigit():
            entry = src_data.get(int(idx))
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata", {})
        ideas = entry.get("ideas", {})
        context = entry.get("context", "")
        for k, idea_meta in meta.items():
            if not isinstance(idea_meta, dict):
                continue
            if idea_meta.get("type") == "POSITIVE":
                title = idea_meta.get("title", "")
                idea_text = ideas.get(k)
                if idea_text is None:
                    alt_k = int(k) if isinstance(k, str) and k.isdigit() else str(k)
                    idea_text = ideas.get(alt_k)
                if idea_text:
                    partner_texts[str(idea_text).strip()] = title
                if title:
                    partner_metadata[(title, context)] = title

    if not partner_texts and not partner_metadata:
        return {}

    ptw_data = _load_yaml_cached(p)
    excluded: dict[str, str] = {}

    # Primary pass: exact idea text matching
    for idx_raw, entry in ptw_data.items():
        if not isinstance(entry, dict):
            continue
        idea_text = str(entry.get("idea", "")).strip()
        if idea_text in partner_texts:
            excluded[str(idx_raw)] = partner_texts[idea_text]

    # Fallback pass: title+context metadata matching for transformed ideas
    # (e.g. when ideas were converted to plan form and text no longer matches)
    if partner_metadata:
        for idx_raw, entry in ptw_data.items():
            if not isinstance(entry, dict):
                continue
            if str(idx_raw) in excluded:
                continue
            if entry.get("label") != "POSITIVE":
                continue
            meta = entry.get("metadata", {})
            if not isinstance(meta, dict):
                continue
            title = meta.get("title", "")
            context = entry.get("context", "")
            key = (title, context)
            if key in partner_metadata:
                excluded[str(idx_raw)] = partner_metadata[key]

    return excluded


def _any_instances_have_turn_numbers(all_artifact_dirs: list[str]) -> bool:
    """Return True if any artifact dir's instances YAML contains turn_number on any idea."""
    seen: set[str] = set()
    for raw_path in all_artifact_dirs:
        instances_path = _extract_instances_path(Path(raw_path).resolve())
        if not instances_path or instances_path in seen:
            continue
        seen.add(instances_path)
        if _instances_have_turn_numbers(instances_path):
            return True
    return False


def _generate_filtered_reports(
    all_artifact_dirs: list[str],
    blocked_titles: set[str],
    logger_inst=None,
) -> dict[str, dict[int, str]]:
    """
    For each artifact dir, auto-derive excluded abstract indices from the paper
    blocklist (by matching titles in the test instances YAML), then recompute
    metrics and write filtered_accuracy_report.txt alongside accuracy_report.txt.

    Returns mapping of instances_path -> {index -> title} for all filtered items.
    Ranking mode is skipped (no scores.json available).
    """
    written = 0
    all_filtered: dict[str, dict[int, str]] = defaultdict(dict)
    _lock = threading.Lock()

    def _process_dir(raw_path: str):
        nonlocal written
        artifact_dir = Path(raw_path).resolve()
        original_report = artifact_dir / "accuracy_report.txt"
        if not original_report.exists():
            return

        instances_path = _extract_instances_path(artifact_dir)
        mode = _extract_mode_from_artifact_dir(artifact_dir)
        excluded_map = _derive_exclude_indices(instances_path, blocked_titles)
        # For pointwise, also exclude the partner idea that is collaterally removed
        # from pairwise (because its only pairwise comparison is gone). Uses idea
        # text matching to avoid false positives from reused generated-idea titles.
        if mode == "pointwise":
            partner_excluded = _derive_pointwise_partner_excluded(instances_path, blocked_titles)
            excluded_map = {**excluded_map, **partner_excluded}
        if not excluded_map:
            return

        with _lock:
            if instances_path:
                all_filtered[instances_path].update(excluded_map)

        exclude_set = set(excluded_map.keys())
        if mode == "ranking":
            if logger_inst:
                logger_inst.warning(
                    f"Skipping filtered recomputation for ranking artifact "
                    f"(no scores.json): {artifact_dir.name}"
                )
            return

        if mode == "pairwise":
            metrics = _recompute_pairwise_metrics(artifact_dir, exclude_set, logger_inst=logger_inst)
        elif mode == "pointwise":
            metrics = _recompute_pointwise_metrics(artifact_dir, exclude_set, logger_inst=logger_inst)
        else:
            metrics = None

        if metrics is None:
            if logger_inst:
                logger_inst.warning(
                    f"Could not recompute metrics for {artifact_dir.name} "
                    f"(mode={mode}) — no scores.json found or all runs failed."
                )
            return

        out_path = _write_filtered_accuracy_report(artifact_dir, mode, metrics, original_report)
        with _lock:
            written += 1
        if logger_inst:
            logger_inst.info(
                f"Filtered report ({mode}, excluded={metrics['excluded']}): "
                f"{out_path.relative_to(_PROJECT_ROOT) if out_path.is_relative_to(_PROJECT_ROOT) else out_path}"
            )

    if all_artifact_dirs:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_dir, d) for d in all_artifact_dirs]
            list(tqdm(as_completed(futures), total=len(futures), desc="Filtering reports"))

    return all_filtered


def _generate_turn_reports(
    all_artifact_dirs: list[str],
    turn_n: int,
    logger_inst=None,
) -> bool:
    """
    For each artifact dir, exclude instances where a NEGATIVE has turn_number == turn_n
    (plus any blocklisted papers), recompute metrics, and write
    turn{N}_accuracy_report.txt alongside accuracy_report.txt.

    Returns True if at least one report was written.
    Ranking mode is skipped (no scores.json available).
    """
    out_filename = f"turn{turn_n}_accuracy_report.txt"
    blocked_titles = _load_paper_blocklist()
    written = 0
    _lock = threading.Lock()

    def _process_dir(raw_path: str):
        nonlocal written
        artifact_dir = Path(raw_path).resolve()
        original_report = artifact_dir / "accuracy_report.txt"
        if not original_report.exists():
            return

        instances_path = _extract_instances_path(artifact_dir)
        mode = _extract_mode_from_artifact_dir(artifact_dir)

        if mode == "pairwise":
            excluded_map = _derive_turn_exclude_indices(instances_path, turn_n)
            if blocked_titles:
                excluded_map = {**excluded_map, **_derive_exclude_indices(instances_path, blocked_titles)}
        elif mode == "pointwise":
            excluded_map = _derive_turn_pointwise_direct(instances_path, turn_n)
            excluded_map = {**excluded_map, **_derive_turn_pointwise_partner_excluded(
                instances_path, turn_n, logger_inst
            )}
            if blocked_titles:
                excluded_map = {**excluded_map, **_derive_exclude_indices(instances_path, blocked_titles)}
                excluded_map = {**excluded_map, **_derive_pointwise_partner_excluded(
                    instances_path, blocked_titles, logger_inst
                )}
        else:
            return  # ranking: skip

        if not excluded_map:
            return

        exclude_set = set(excluded_map.keys())
        if mode == "pairwise":
            metrics = _recompute_pairwise_metrics(artifact_dir, exclude_set, logger_inst=logger_inst)
        else:
            metrics = _recompute_pointwise_metrics(artifact_dir, exclude_set, logger_inst=logger_inst)

        if metrics is None:
            if logger_inst:
                logger_inst.warning(
                    f"Could not recompute metrics for {artifact_dir.name} "
                    f"(mode={mode}, turn{turn_n}) — no scores.json found."
                )
            return

        out_path = _write_filtered_accuracy_report(
            artifact_dir, mode, metrics, original_report,
            out_filename=out_filename,
        )
        with _lock:
            written += 1
        if logger_inst:
            logger_inst.info(
                f"Turn{turn_n} report ({mode}, excluded={metrics['excluded']}): "
                f"{out_path.relative_to(_PROJECT_ROOT) if out_path.is_relative_to(_PROJECT_ROOT) else out_path}"
            )

    if all_artifact_dirs:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_dir, d) for d in all_artifact_dirs]
            list(tqdm(as_completed(futures), total=len(futures), desc=f"Turn{turn_n} reports"))

    return written > 0


# ============================================================
# Per-ablation report generation
# ============================================================

_SETUP_METRIC_LABELS: dict[str, dict[str, str]] = {
    "pairwise": {
        "pairwise_accuracy": "Acc (with ties)",
        "pairwise_accuracy_strict": "Acc (with ties, strict)",
        "pairwise_accuracy_no_ties": "Acc (w/o ties)",
        "n_ties": "Ties",
        "support": "Samples",
        "cost_usd": "Cost ($)",
    },
    "pointwise": {
        "accuracy": "Accuracy",
        "f1_macro": "F1 macro",
        "f1_pos": "F1 POS",
        "f1_neg": "F1 NEG",
        "precision_pos": "Prec POS",
        "precision_neg": "Prec NEG",
        "recall_pos": "Rec POS",
        "recall_neg": "Rec NEG",
        "support": "Samples",
        "cost_usd": "Cost ($)",
    },
    "ranking": {
        "ndcg": "NDCG",
        "mrr": "MRR",
        "hits_1": "Hits@1",
        "hits_2": "Hits@2",
        "hits_3": "Hits@3",
        "support": "Samples",
        "cost_usd": "Cost ($)",
    },
}

_GENERATE_REPORT_MOD = None


def _get_generate_report_mod(logger_inst=None):
    """Lazy-load generate_report.py as a module and inject our logger."""
    global _GENERATE_REPORT_MOD
    if _GENERATE_REPORT_MOD is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate_report", _GENERATE_REPORT_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GENERATE_REPORT_MOD = mod
    if logger_inst is not None:
        _GENERATE_REPORT_MOD.logger = logger_inst
    return _GENERATE_REPORT_MOD


def _get_metrics_for_setup(metrics_dict: dict, setup: str) -> dict | None:
    """Pick the right sub-mode key from extract_metrics_from_report output."""
    if setup in ("pairwise", "pointwise"):
        return metrics_dict.get(setup)
    # ranking artifacts report under 'swiss' or 'bi-swiss'
    return metrics_dict.get("swiss") or metrics_dict.get("bi-swiss")


def _extract_metrics_for_dir(
    artifact_dir: Path,
    setup: str,
    logger_inst=None,
    report_file: str = "accuracy_report.txt",
) -> dict | None:
    """Extract metrics dict for a given setup from an artifact directory.

    Falls back to accuracy_report.txt when report_file doesn't exist.
    Uses a global cache to avoid redundant parsing.
    """
    cache_key = (artifact_dir, report_file)
    if cache_key not in _METRICS_CACHE:
        report_path = artifact_dir / report_file
        if not report_path.exists() and report_file != "accuracy_report.txt":
            report_path = artifact_dir / "accuracy_report.txt"
        
        if not report_path.exists():
            _METRICS_CACHE[cache_key] = None
        else:
            mod = _get_generate_report_mod(logger_inst)
            report_data = mod.extract_metrics_from_report(str(report_path))

            # Inject ablation cost into each mode if available
            cost_path = artifact_dir / "cost_report.json"
            if cost_path.exists() and report_data:
                try:
                    with open(cost_path) as f:
                        cost_data = json.load(f)
                        cost_usd = cost_data.get("total_cost_usd", 0.0)
                        for mode_metrics in report_data.values():
                            if isinstance(mode_metrics, dict):
                                mode_metrics["cost_usd"] = cost_usd
                except Exception:
                    pass

            _METRICS_CACHE[cache_key] = report_data
    
    metrics_by_mode = _METRICS_CACHE[cache_key]
    if not metrics_by_mode:
        return None
    result = _get_metrics_for_setup(metrics_by_mode, setup)

    # Backfill accuracy_strict for reports generated before this metric was added,
    # or when it is 0.0 due to a bug where accuracy_strict was never stored per run.
    # Reads from scores.json (same source as filtered recomputation), result is
    # stored in the cached dict so it only runs once per artifact_dir.
    if setup == "pairwise" and result is not None and result.get("pairwise_accuracy_strict", 0.0) == 0.0:
        fallback = _recompute_pairwise_metrics(artifact_dir, set(), logger_inst=logger_inst)
        if fallback and fallback.get("pairwise_accuracy_strict", 0.0) != 0.0:
            result["pairwise_accuracy_strict"] = fallback["pairwise_accuracy_strict"]

    return result


def _baseline_canonical_names() -> set[str]:
    """Canonical + legacy names of all baseline-group ablations, plus any
    merge-config `baseline:` override.

    Used to skip baseline runs as the *subject* of a comparison everywhere. The
    override is included so a config-designated baseline (e.g. ai_researcher_base)
    is treated like the built-in `current` baseline rather than compared to itself.
    """
    names: set[str] = set()
    for name, entry in _load_ablations_yaml_data().get("ablations", {}).items():
        if entry.get("group") == "baseline":
            names.add(name)
            names.update(entry.get("legacy_names", []))
    if _BASELINE_OVERRIDE:
        names.add(_BASELINE_OVERRIDE)
    return names


def _resolve_baseline_name(
    setup: str,
    ablation_name: str,
    retrieval_index: dict,
    yaml_aliases: dict,
) -> str | None:
    """Find the baseline ablation key for a given ablation in retrieval_index."""
    track_prefix = ablation_name.split("_")[0] if "_" in ablation_name else ""
    candidates: list[str] = []
    # An optional merge-config `baseline:` override takes precedence over the
    # default "current" baseline (e.g. comparing ai_researcher_* variants against
    # ai_researcher_base instead of the unrelated novelty-judge `current` run).
    if _BASELINE_OVERRIDE:
        if track_prefix:
            candidates.append(f"{track_prefix}_{_BASELINE_OVERRIDE}")
        candidates.append(_BASELINE_OVERRIDE)
    if track_prefix:
        candidates.append(f"{track_prefix}_current")
    candidates.append("current")

    seen: set[str] = set()
    expanded: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            expanded.append(c)
        for alias in yaml_aliases.get(c, []):
            if alias not in seen:
                seen.add(alias)
                expanded.append(alias)

    for candidate in expanded:
        for (m, abl, _) in retrieval_index:
            if m == setup and abl == candidate:
                return candidate
    return None


def _generate_debug_examples_section(
    setup: str,
    model: str,
    base_outcomes: dict[str, float],
    abl_outcomes: dict[str, float],
    instance_data: dict,
    abl_artifact_dir: Path | None = None,
    base_artifact_dir: Path | None = None,
) -> list[str]:
    """
    Generate markdown lines for the per-example debug section of a per-ablation report.

    Classifies problems by (baseline_score, ablation_score) transition and shows
    helped/hurt examples with full idea texts in collapsible details blocks.
    Scores: 1.0 = correct, 0.5 = tie, 0.0 = wrong.
    """
    try:
        from novelty_eval.md_report_writer import (
            extract_all_comparison_reasonings,
            extract_pointwise_reasonings,
            format_reasoning_for_md,
        )
        _has_reasoning = True
    except Exception:
        _has_reasoning = False

    _pairwise_run_dirs: list[Path] = []
    _pointwise_run_dirs: list[Path] = []
    _base_pairwise_run_dirs: list[Path] = []
    _base_pointwise_run_dirs: list[Path] = []
    if _has_reasoning:
        if abl_artifact_dir:
            _pairwise_run_dirs = sorted(abl_artifact_dir.glob("run_pairwise_*/"))
            _pointwise_run_dirs = sorted(abl_artifact_dir.glob("run_pointwise_*/"))
        if base_artifact_dir:
            _base_pairwise_run_dirs = sorted(base_artifact_dir.glob("run_pairwise_*/"))
            _base_pointwise_run_dirs = sorted(base_artifact_dir.glob("run_pointwise_*/"))

    def _score_label(s: float) -> str:
        if s == 1.0:
            return "✅ correct"
        if s == 0.5:
            return "🔘 tie"
        return "❌ wrong"

    def _lookup_instance(pid: str) -> dict:
        entry = instance_data.get(pid)
        if entry is None and pid.lstrip("-").isdigit():
            entry = instance_data.get(int(pid))
        return entry or {}

    def _wrap_blockquote(text: str) -> str:
        lines = str(text).strip().splitlines()
        if not lines:
            return "> *(empty)*"
        return "\n".join(f"> {line}" if line else ">" for line in lines)

    def _render_example(pid: str, base_s: float, abl_s: float) -> list[str]:
        entry = _lookup_instance(pid)
        lines: list[str] = []
        lines.append(
            f"**Problem {pid}** — Baseline: {_score_label(base_s)} → Ablation: {_score_label(abl_s)}"
        )
        lines.append("")
        lines.append("<details>")
        if setup == "pairwise":
            gt_list = entry.get("expected_winners", [])
            gt_str = ", ".join(f"Idea `{w}`" for w in gt_list) if gt_list else "N/A"
            lines.append(f"<summary>Idea texts · GT winner: {gt_str} (click to expand)</summary>")
            lines.append("")
            ideas = entry.get("ideas", {})
            meta = entry.get("metadata", {})
            for k in sorted(ideas.keys(), key=lambda x: str(x)):
                k_str = str(k)
                title = ""
                if isinstance(meta, dict):
                    idea_meta = meta.get(k) or meta.get(k_str) or {}
                    if isinstance(idea_meta, dict):
                        title = idea_meta.get("title", "")
                idea_text = ideas.get(k) or ideas.get(k_str) or ""
                lines.append(f"**Idea {k_str}**" + (f" — {title}" if title else ""))
                lines.append("")
                lines.append(_wrap_blockquote(idea_text))
                lines.append("")
        elif setup == "pointwise":
            label = entry.get("label", "N/A")
            lines.append(f"<summary>Idea text · GT label: `{label}` (click to expand)</summary>")
            lines.append("")
            meta = entry.get("metadata", {})
            title = meta.get("title", "") if isinstance(meta, dict) else ""
            idea_text = entry.get("idea", "")
            lines.append(f"**Idea**" + (f" — {title}" if title else ""))
            lines.append("")
            lines.append(_wrap_blockquote(idea_text))
            lines.append("")

        # --- Comparison / Evaluation Reasoning (baseline then ablation) ---
        def _render_pairwise_reasoning_block(run_dirs: list[Path], label_prefix: str) -> list[str]:
            all_entries = []
            for run_dir in run_dirs:
                all_entries.extend(extract_all_comparison_reasonings(str(run_dir), pid, "0", "1"))
            out: list[str] = []
            if not all_entries:
                return out
            out.append("<details>")
            out.append(f"<summary><strong>{label_prefix} Reasoning</strong></summary>")
            out.append("")
            for r_entry in all_entries:
                winner_str = str(r_entry["winner"]) if r_entry["winner"] is not None else "?"
                sub_label = f"Idea {r_entry['idea0']} vs Idea {r_entry['idea1']} (winner = {winner_str})"
                out.append("<details>")
                out.append(f"<summary>{sub_label}</summary>")
                out.append("")
                for line in format_reasoning_for_md(r_entry["text"]).split("\n"):
                    out.append(f"> {line}")
                out.append("")
                out.append("</details>")
                out.append("")
            out.append("</details>")
            out.append("")
            return out

        def _render_pointwise_reasoning_block(run_dirs: list[Path], label_prefix: str) -> list[str]:
            all_entries = []
            for run_dir in run_dirs:
                all_entries.extend(extract_pointwise_reasonings(str(run_dir), pid))
            out: list[str] = []
            if not all_entries:
                return out
            out.append("<details>")
            out.append(f"<summary><strong>{label_prefix} Reasoning</strong></summary>")
            out.append("")
            for r_entry in all_entries:
                choice = r_entry.get("choice") or {}
                pred_str = str(next(iter(choice.values()), "?")) if choice else "?"
                out.append("<details>")
                out.append(f"<summary>Call {r_entry.get('call_index', '?')} (prediction = {pred_str})</summary>")
                out.append("")
                for line in format_reasoning_for_md(r_entry["text"]).split("\n"):
                    out.append(f"> {line}")
                out.append("")
                out.append("</details>")
                out.append("")
            out.append("</details>")
            out.append("")
            return out

        if _has_reasoning and setup == "pairwise":
            lines.extend(_render_pairwise_reasoning_block(_base_pairwise_run_dirs, "Baseline"))
            lines.extend(_render_pairwise_reasoning_block(_pairwise_run_dirs, "Ablation"))
        elif _has_reasoning and setup == "pointwise":
            lines.extend(_render_pointwise_reasoning_block(_base_pointwise_run_dirs, "Baseline"))
            lines.extend(_render_pointwise_reasoning_block(_pointwise_run_dirs, "Ablation"))

        lines.append("</details>")
        lines.append("")
        return lines

    # --- Classify ---
    common_pids = sorted(
        set(base_outcomes.keys()) & set(abl_outcomes.keys()),
        key=lambda x: (len(x), x),
    )
    helped_wrong_to_correct: list[str] = []
    helped_tie_to_correct: list[str] = []
    helped_wrong_to_tie: list[str] = []
    hurt_correct_to_wrong: list[str] = []
    hurt_correct_to_tie: list[str] = []
    hurt_tie_to_wrong: list[str] = []
    unchanged_correct: list[str] = []
    unchanged_wrong: list[str] = []
    unchanged_tie: list[str] = []

    for pid in common_pids:
        b = base_outcomes[pid]
        a = abl_outcomes[pid]
        if b == 0.0 and a == 1.0:
            helped_wrong_to_correct.append(pid)
        elif b == 0.5 and a == 1.0:
            helped_tie_to_correct.append(pid)
        elif b == 0.0 and a == 0.5:
            helped_wrong_to_tie.append(pid)
        elif b == 1.0 and a == 0.0:
            hurt_correct_to_wrong.append(pid)
        elif b == 1.0 and a == 0.5:
            hurt_correct_to_tie.append(pid)
        elif b == 0.5 and a == 0.0:
            hurt_tie_to_wrong.append(pid)
        elif b == 1.0 and a == 1.0:
            unchanged_correct.append(pid)
        elif b == 0.0 and a == 0.0:
            unchanged_wrong.append(pid)
        elif b == 0.5 and a == 0.5:
            unchanged_tie.append(pid)

    n_helped = len(helped_wrong_to_correct) + len(helped_tie_to_correct) + len(helped_wrong_to_tie)
    n_hurt = len(hurt_correct_to_wrong) + len(hurt_correct_to_tie) + len(hurt_tie_to_wrong)

    md: list[str] = []
    md.append(f"### Per-Example Analysis — {model}")
    md.append("")

    # Summary table
    helped_parts = []
    if helped_wrong_to_correct:
        helped_parts.append(f"wrong→correct: {len(helped_wrong_to_correct)}")
    if helped_tie_to_correct:
        helped_parts.append(f"tie→correct: {len(helped_tie_to_correct)}")
    if helped_wrong_to_tie:
        helped_parts.append(f"wrong→tie: {len(helped_wrong_to_tie)}")
    helped_detail = f" ({', '.join(helped_parts)})" if helped_parts else ""

    hurt_parts = []
    if hurt_correct_to_wrong:
        hurt_parts.append(f"correct→wrong: {len(hurt_correct_to_wrong)}")
    if hurt_correct_to_tie:
        hurt_parts.append(f"correct→tie: {len(hurt_correct_to_tie)}")
    if hurt_tie_to_wrong:
        hurt_parts.append(f"tie→wrong: {len(hurt_tie_to_wrong)}")
    hurt_detail = f" ({', '.join(hurt_parts)})" if hurt_parts else ""

    md.append("**Confusion Matrix** *(rows = baseline, cols = ablation)*")
    md.append("")
    if setup == "pointwise":
        # Derive actual predictions from correctness + ground truth label.
        # pred_is_positive = True iff (label=POSITIVE and correct) or (label=NEGATIVE and wrong).
        base_tp = base_fp = base_fn = base_tn = 0
        abl_tp = abl_fp = abl_fn = abl_tn = 0
        for pid in common_pids:
            entry = _lookup_instance(pid)
            label = entry.get("label", "")
            if label not in ("POSITIVE", "NEGATIVE"):
                continue
            gt_pos = label == "POSITIVE"
            b_pos = gt_pos == (base_outcomes[pid] >= 0.5)
            a_pos = gt_pos == (abl_outcomes[pid] >= 0.5)
            if gt_pos and b_pos:
                base_tp += 1
            elif not gt_pos and b_pos:
                base_fp += 1
            elif gt_pos and not b_pos:
                base_fn += 1
            else:
                base_tn += 1
            if gt_pos and a_pos:
                abl_tp += 1
            elif not gt_pos and a_pos:
                abl_fp += 1
            elif gt_pos and not a_pos:
                abl_fn += 1
            else:
                abl_tn += 1
        md.append("**Baseline** *(rows = GT, cols = predicted)*")
        md.append("")
        md.append("| GT \\ Pred | Positive | Negative |")
        md.append("|---|---|---|")
        md.append(f"| Positive | {base_tp} | {base_fn} |")
        md.append(f"| Negative | {base_fp} | {base_tn} |")
        md.append("")
        md.append("**Ablation** *(rows = GT, cols = predicted)*")
        md.append("")
        md.append("| GT \\ Pred | Positive | Negative |")
        md.append("|---|---|---|")
        md.append(f"| Positive | {abl_tp} | {abl_fn} |")
        md.append(f"| Negative | {abl_fp} | {abl_tn} |")
    else:
        md.append("| Baseline \\ Ablation | ✅ Correct | 🔘 Tie | ❌ Wrong |")
        md.append("|---|---|---|---|")
        md.append(f"| ✅ Correct | {len(unchanged_correct)} | {len(hurt_correct_to_tie)} | {len(hurt_correct_to_wrong)} |")
        md.append(f"| 🔘 Tie | {len(helped_tie_to_correct)} | {len(unchanged_tie)} | {len(hurt_tie_to_wrong)} |")
        md.append(f"| ❌ Wrong | {len(helped_wrong_to_correct)} | {len(helped_wrong_to_tie)} | {len(unchanged_wrong)} |")
    md.append("")

    # For pointwise, break down helped/hurt by GT label (POSITIVE / NEGATIVE).
    helped_label_detail = ""
    hurt_label_detail = ""
    if setup == "pointwise":
        def _count_by_label(pids: list[str]) -> tuple[int, int]:
            pos = neg = 0
            for pid in pids:
                lbl = _lookup_instance(pid).get("label", "")
                if lbl == "POSITIVE":
                    pos += 1
                elif lbl == "NEGATIVE":
                    neg += 1
            return pos, neg

        all_helped = helped_wrong_to_correct + helped_tie_to_correct + helped_wrong_to_tie
        all_hurt = hurt_correct_to_wrong + hurt_correct_to_tie + hurt_tie_to_wrong
        h_pos, h_neg = _count_by_label(all_helped)
        r_pos, r_neg = _count_by_label(all_hurt)
        if h_pos or h_neg:
            helped_label_detail = f", POS: {h_pos}, NEG: {h_neg}"
        if r_pos or r_neg:
            hurt_label_detail = f", POS: {r_pos}, NEG: {r_neg}"

    md.append("| Category | Count |")
    md.append("|---|---|")
    md.append(f"| 🟢 Helped | {n_helped}{helped_detail}{helped_label_detail} |")
    md.append(f"| 🔴 Hurt | {n_hurt}{hurt_detail}{hurt_label_detail} |")
    md.append(f"| Unchanged (both correct) | {len(unchanged_correct)} |")
    md.append(f"| Unchanged (both wrong) | {len(unchanged_wrong)} |")
    if unchanged_tie:
        md.append(f"| Unchanged (both tie) | {len(unchanged_tie)} |")
    md.append(f"| **Total shared** | **{len(common_pids)}** |")
    md.append("")

    # Helped examples
    all_helped = helped_wrong_to_correct + helped_tie_to_correct + helped_wrong_to_tie
    if all_helped:
        md.append(f"<details>")
        md.append(f"<summary>🟢 Examples Helped ({n_helped})</summary>")
        md.append("")
        for pid in all_helped:
            md.extend(_render_example(pid, base_outcomes[pid], abl_outcomes[pid]))
        md.append("</details>")
        md.append("")

    # Hurt examples
    all_hurt = hurt_correct_to_wrong + hurt_correct_to_tie + hurt_tie_to_wrong
    if all_hurt:
        md.append(f"<details>")
        md.append(f"<summary>🔴 Examples Hurt ({n_hurt})</summary>")
        md.append("")
        for pid in all_hurt:
            md.extend(_render_example(pid, base_outcomes[pid], abl_outcomes[pid]))
        md.append("</details>")
        md.append("")

    return md


def _run_bootstrap_for_setup(
    setup: str,
    all_models: list,
    baseline_name: str,
    lookup_abl: str,
    retrieval_index: dict,
    report_file: str,
    mismatch_suppressed: set,
    logger_inst,
) -> dict:
    """Run corpus-level paired bootstrap for one setup; return bootstrap_results dict."""
    bootstrap_results: dict[tuple[str, str], dict] = {}
    if setup not in ("pointwise", "pairwise"):
        return bootstrap_results

    _sig_alts = _load_significance_alternatives_from_yaml()
    alternative = _sig_alts.get(lookup_abl, "two-sided")

    def _process_single_model(model: str) -> dict[tuple[str, str], dict]:
        _cache_key = (setup, baseline_name, lookup_abl, model, report_file)
        with _BOOTSTRAP_CACHE_LOCK:
            if _cache_key in _BOOTSTRAP_CACHE:
                return _BOOTSTRAP_CACHE[_cache_key]

        model_results: dict[tuple[str, str], dict] = {}
        base_dir = retrieval_index.get((setup, baseline_name, model))
        abl_dir = retrieval_index.get((setup, lookup_abl, model))
        if not base_dir or not abl_dir:
            return model_results

        base_instances = _extract_instances_path(Path(base_dir))
        abl_instances = _extract_instances_path(Path(abl_dir))
        if base_instances and abl_instances:
            try:
                same_path = Path(base_instances).resolve() == Path(abl_instances).resolve()
            except Exception:
                same_path = base_instances == abl_instances
            if not same_path:
                all_match, mismatched = _instances_content_matches(
                    base_instances, abl_instances, setup
                )
                if not all_match and logger_inst:
                    if lookup_abl in mismatch_suppressed:
                        logger_inst.debug(
                            f"[{setup}/{model}] Instance mismatch warning suppressed for '{lookup_abl}' "
                            f"(suppress_instance_mismatch_warning=true in ablations.yaml)."
                        )
                    else:
                        preview = mismatched[:10]
                        suffix = "..." if len(mismatched) > 10 else ""
                        def _abs(p: str) -> str:
                            pp = Path(p)
                            return str(pp if pp.is_absolute() else (_PROJECT_ROOT / pp).resolve())
                        logger_inst.warning(
                            f"[{setup}/{model}] {len(mismatched)} pids have mismatched content "
                            f"between base and ablation instances — bootstrap pairing is unreliable.\n"
                            f"  base artifact dir:      {base_dir}\n"
                            f"  base instances:         {_abs(base_instances)}\n"
                            f"  ablation artifact dir:  {abl_dir}\n"
                            f"  ablation instances:     {_abs(abl_instances)}\n"
                            f"  mismatched pids:        {preview}{suffix}"
                        )

        base_outcomes_dict = _extract_raw_outcomes_for_dir(Path(base_dir), setup, baseline_name, logger_inst)
        abl_outcomes_dict = _extract_raw_outcomes_for_dir(Path(abl_dir), setup, lookup_abl, logger_inst)
        if not base_outcomes_dict or not abl_outcomes_dict:
            return model_results

        if report_file != "accuracy_report.txt":
            _excl = (
                _extract_exclude_set_from_filtered_report(Path(base_dir), report_filename=report_file)
                | _extract_exclude_set_from_filtered_report(Path(abl_dir), report_filename=report_file)
            )
            if _excl:
                base_outcomes_dict = {
                    mk: {pid: s for pid, s in outcomes.items() if pid not in _excl}
                    for mk, outcomes in base_outcomes_dict.items()
                }
                abl_outcomes_dict = {
                    mk: {pid: s for pid, s in outcomes.items() if pid not in _excl}
                    for mk, outcomes in abl_outcomes_dict.items()
                }

        def _get_bootstrap_for_model(all_results: dict) -> None:
            for metric_key, res in all_results.items():
                model_results[(model, metric_key)] = res

        if setup == "pointwise":
            base_pred = base_outcomes_dict.get("_pointwise_pred", {})
            base_target = base_outcomes_dict.get("_pointwise_target", {})
            abl_pred = abl_outcomes_dict.get("_pointwise_pred", {})
            abl_target = abl_outcomes_dict.get("_pointwise_target", {})
            corpus_pids = sorted(set(base_pred) & set(abl_pred) & set(base_target) & set(abl_target))
            if corpus_pids:
                base_pl = [(int(base_pred[p] >= 0.5), int(base_target[p] >= 0.5)) for p in corpus_pids]
                abl_pl = [(int(abl_pred[p] >= 0.5), int(abl_target[p] >= 0.5)) for p in corpus_pids]
                _get_bootstrap_for_model(compute_bootstrap_results_pointwise_all(base_pl, abl_pl, context_label=f"{model} / {lookup_abl} (pointwise)", _logger=logger_inst, alternative=alternative))

        elif setup == "pairwise":
            base_ic = base_outcomes_dict.get("_pairwise_is_correct", {})
            base_it = base_outcomes_dict.get("_pairwise_is_tie", {})
            base_gt = base_outcomes_dict.get("_pairwise_gt_winner", {})
            abl_ic = abl_outcomes_dict.get("_pairwise_is_correct", {})
            abl_it = abl_outcomes_dict.get("_pairwise_is_tie", {})
            abl_gt = abl_outcomes_dict.get("_pairwise_gt_winner", {})
            corpus_pids = sorted(
                set(base_ic) & set(base_it) & set(base_gt)
                & set(abl_ic) & set(abl_it) & set(abl_gt)
            )
            if corpus_pids:
                base_pw = [(base_ic[p], base_it[p], base_gt[p]) for p in corpus_pids]
                abl_pw = [(abl_ic[p], abl_it[p], abl_gt[p]) for p in corpus_pids]
                for _role, _pw in (("base", base_pw), ("abl", abl_pw)):
                    _bad = [p for p, (ic, it, _) in zip(corpus_pids, _pw) if ic >= 0.5 and it >= 0.5]
                    if _bad and logger_inst:
                        logger_inst.warning(
                            "[%s/%s/%s] %d pairwise instances are marked both correct and tie (invalid): %s",
                            model, lookup_abl, _role, len(_bad), _bad[:5],
                        )
                _get_bootstrap_for_model(compute_bootstrap_results_pairwise_all(base_pw, abl_pw, context_label=f"{model} / {lookup_abl} (pairwise)", _logger=logger_inst, alternative=alternative))

        with _BOOTSTRAP_CACHE_LOCK:
            _BOOTSTRAP_CACHE[_cache_key] = model_results
        return model_results

    with ThreadPoolExecutor() as executor:
        results_list = list(executor.map(_process_single_model, all_models))
        for res in results_list:
            bootstrap_results.update(res)

    return bootstrap_results

def _render_setup_tables(
    setup: str,
    all_models: list,
    baseline_metrics: dict,
    ablation_metrics: dict,
    metric_labels: dict,
    bootstrap_results: dict,
) -> list[str]:
    """Render baseline + ablation markdown tables with inline deltas and significance stars."""
    metric_keys = list(metric_labels.keys())
    metric_header_labels = [metric_labels[k] for k in metric_keys]
    md: list[str] = []

    md.append("### Baseline Results")
    md.append("")
    md.append("| Model | " + " | ".join(metric_header_labels) + " |")
    md.append("| --- | " + " | ".join("---" for _ in metric_keys) + " |")
    for model in all_models:
        bm = baseline_metrics.get(model, {})
        row = [f"**{model}**"]
        for k in metric_keys:
            val = bm.get(k)
            if val is not None:
                fmt = ".1f" if "support" in k or "n_ties" in k else ".4f"
                row.append(f"{val:{fmt}}")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("### Ablation Results")
    md.append("")
    md.append("| Model | " + " | ".join(metric_header_labels) + " |")
    md.append("| --- | " + " | ".join("---" for _ in metric_keys) + " |")
    for model in all_models:
        bm = baseline_metrics.get(model, {})
        am = ablation_metrics.get(model, {})
        cells = [f"**{model}**"]
        for mk in metric_keys:
            bv = bm.get(mk)
            av = am.get(mk)
            if av is not None:
                fmt = ".1f" if "support" in mk or "n_ties" in mk else ".4f"
                sig_stars = ""
                if (model, mk) in bootstrap_results:
                    res = bootstrap_results[(model, mk)]
                    if not res.get("reliable", True):
                        sig_stars = "?"
                    elif res.get("significant"):
                        sig_stars = "*"
                if bv is not None:
                    delta = av - bv
                    if "n_ties" in mk or "support" in mk:
                        sign = "+" if delta > 0 else ""
                        cells.append(f"{av:{fmt}} ({sign}{delta:.1f}){sig_stars}")
                    elif "cost" in mk:
                        cells.append(f"{av:{fmt}}")
                    else:
                        pct = delta * 100
                        sign = "+" if delta > 0 else ""
                        cells.append(f"{av:{fmt}} ({sign}{pct:.1f}%){sig_stars}")
                else:
                    cells.append(f"{av:{fmt}}")
            else:
                cells.append("—")
        md.append("| " + " | ".join(cells) + " |")

    if bootstrap_results:
        md.append("")
        md.append("*\\* significant (95% BCa CI); ? = CI unreliable (degenerate bootstrap distribution) — corpus-level paired bootstrap, n=100,000*")
    md.append("")
    return md


def _compute_confusion_counts(
    pred_dict: dict[str, float],
    target_dict: dict[str, float],
    exclude_set: set[str] | None = None,
) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for pid, pred in pred_dict.items():
        if exclude_set and pid in exclude_set:
            continue
        target = target_dict.get(pid)
        if target is None:
            continue
        pred_class = int(pred >= 0.5)
        tgt_class = int(target >= 0.5)
        if pred_class == 1 and tgt_class == 1:
            tp += 1
        elif pred_class == 0 and tgt_class == 0:
            tn += 1
        elif pred_class == 1 and tgt_class == 0:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _compute_pairwise_breakdown(
    is_correct_dict: dict[str, float],
    is_tie_dict: dict[str, float],
    exclude_set: set[str] | None = None,
) -> dict[str, int]:
    correct = wrong = tie = 0
    for pid, is_tie_val in is_tie_dict.items():
        if exclude_set and pid in exclude_set:
            continue
        if is_tie_val >= 0.5:
            tie += 1
        elif is_correct_dict.get(pid, 0.0) >= 0.5:
            correct += 1
        else:
            wrong += 1
    return {"correct": correct, "wrong": wrong, "tie": tie}


def _render_confusion_table(
    setup: str,
    all_models: list,
    baseline_name: str,
    lookup_abl: str,
    retrieval_index: dict,
    report_file: str,
    logger_inst,
) -> list[str]:
    """Render per-model confusion matrix (pointwise) or correct/wrong/tie table (pairwise)."""
    if setup not in ("pairwise", "pointwise"):
        return []

    rows: list[tuple] = []  # (model, role, counts_dict)
    for model in all_models:
        base_dir_str = retrieval_index.get((setup, baseline_name, model))
        abl_dir_str = retrieval_index.get((setup, lookup_abl, model))

        # Build shared exclusion set (union, same logic as per-example section)
        exclude_set: set[str] = set()
        if report_file != "accuracy_report.txt":
            if base_dir_str:
                exclude_set |= _extract_exclude_set_from_filtered_report(Path(base_dir_str), report_filename=report_file)
            if abl_dir_str:
                exclude_set |= _extract_exclude_set_from_filtered_report(Path(abl_dir_str), report_filename=report_file)

        for role, dir_str, abl_key in (
            ("Baseline", base_dir_str, baseline_name),
            ("Ablation", abl_dir_str, lookup_abl),
        ):
            if not dir_str:
                continue
            outcomes = _extract_raw_outcomes_for_dir(Path(dir_str), setup, abl_key, logger_inst)
            if not outcomes:
                continue
            if setup == "pointwise":
                pred_dict = outcomes.get("_pointwise_pred", {})
                target_dict = outcomes.get("_pointwise_target", {})
                counts = _compute_confusion_counts(pred_dict, target_dict, exclude_set or None)
            else:
                counts = _compute_pairwise_breakdown(
                    outcomes.get("_pairwise_is_correct", {}),
                    outcomes.get("_pairwise_is_tie", {}),
                    exclude_set or None,
                )
            rows.append((model, role, counts))

    if not rows:
        return []

    md: list[str] = []
    if setup == "pointwise":
        md += [
            "### Confusion Matrix",
            "",
            "| Model | Role | TP | FP | FN | TN | Total |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for model, role, c in rows:
            total = c["tp"] + c["fp"] + c["fn"] + c["tn"]
            md.append(f"| **{model}** | {role} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | {total} |")
    else:
        md += [
            "### Prediction Breakdown",
            "",
            "| Model | Role | Correct | Wrong | Tie | Total |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for model, role, c in rows:
            total = c["correct"] + c["wrong"] + c["tie"]
            md.append(f"| **{model}** | {role} | {c['correct']} | {c['wrong']} | {c['tie']} | {total} |")
    md.append("")
    return md


def _extract_batch_drop_stats(artifact_dir: Path) -> dict | None:
    """Parse all batch_output_*.jsonl files under an artifact dir and return drop statistics.

    A call is "dropped" when finish_reason == 'length' and content is empty — the model
    exhausted its token budget on reasoning with nothing left for output.

    Returns a dict with keys:
        total_calls, dropped_calls, drop_rate,
        total_examples, examples_with_drops,
        avg_successful_per_example, avg_total_per_example
    Returns None if no batch output files are found.
    """
    batch_files = sorted(artifact_dir.glob("run_*/batch_output_*.jsonl"))
    if not batch_files:
        return None

    from collections import defaultdict
    calls_per_example: dict[str, dict] = defaultdict(lambda: {"total": 0, "dropped": 0})

    for bf in batch_files:
        try:
            with open(bf, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    cid = record.get("custom_id", "")
                    parts = cid.split("__")
                    example_id = parts[1] if len(parts) >= 2 else cid
                    choices = (
                        record.get("response", {})
                        .get("body", {})
                        .get("choices", [])
                    )
                    is_dropped = any(
                        c.get("finish_reason") == "length"
                        or not c.get("message", {}).get("content")
                        for c in choices
                    )
                    calls_per_example[example_id]["total"] += 1
                    if is_dropped:
                        calls_per_example[example_id]["dropped"] += 1
        except Exception:
            continue

    if not calls_per_example:
        return None

    total_calls = sum(v["total"] for v in calls_per_example.values())
    dropped_calls = sum(v["dropped"] for v in calls_per_example.values())
    total_examples = len(calls_per_example)
    examples_with_drops = sum(1 for v in calls_per_example.values() if v["dropped"] > 0)
    avg_successful = (
        sum(v["total"] - v["dropped"] for v in calls_per_example.values()) / total_examples
        if total_examples > 0 else 0.0
    )
    avg_total = total_calls / total_examples if total_examples > 0 else 0.0

    return {
        "total_calls": total_calls,
        "dropped_calls": dropped_calls,
        "drop_rate": dropped_calls / total_calls if total_calls > 0 else 0.0,
        "total_examples": total_examples,
        "examples_with_drops": examples_with_drops,
        "avg_successful_per_example": avg_successful,
        "avg_total_per_example": avg_total,
    }


def _render_call_quality_section(
    setup: str,
    all_models: list,
    baseline_name: str,
    lookup_abl: str,
    retrieval_index: dict,
) -> list[str]:
    """Render a markdown section with LLM call drop statistics for each model."""
    rows: list[tuple] = []  # (model, role, stats_dict)
    for model in all_models:
        for role, abl_key in (("Baseline", baseline_name), ("Ablation", lookup_abl)):
            dir_str = retrieval_index.get((setup, abl_key, model))
            if not dir_str:
                continue
            stats = _extract_batch_drop_stats(Path(dir_str))
            if stats is not None:
                rows.append((model, role, stats))

    if not rows:
        return []

    md: list[str] = [
        "### LLM Call Quality",
        "",
        "| Model | Role | Total calls | Dropped | Drop rate | Examples w/ drops | Avg successful / example |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model, role, s in rows:
        md.append(
            f"| **{model}** | {role} "
            f"| {s['total_calls']} "
            f"| {s['dropped_calls']} "
            f"| {s['drop_rate']:.1%} "
            f"| {s['examples_with_drops']}/{s['total_examples']} "
            f"| {s['avg_successful_per_example']:.2f} (of {s['avg_total_per_example']:.2f} returned) |"
        )
    md.append("")
    return md


def _render_setup_heatmap(
    setup: str,
    all_models: list,
    baseline_metrics: dict,
    ablation_metrics: dict,
    metric_labels: dict,
    bootstrap_results: dict,
    figures_dir: Path,
    safe_name: str,
    name_suffix: str,
    desc: str,
    before_desc: str,
) -> list[str]:
    """Render the aggregated delta heatmap section and return markdown lines."""
    if setup == "pointwise":
        heatmap_metric_labels = {
            k: v for k, v in metric_labels.items()
            if k in ("f1_macro", "f1_pos", "f1_neg", "accuracy")
        }
    else:
        heatmap_metric_labels = {
            k: v for k, v in metric_labels.items()
            if k not in ("n_ties", "support", "support_without_ties", "cost_usd")
        }

    png_filename = f"{safe_name}_{setup}_impact{name_suffix}.png"
    out_path = figures_dir / png_filename
    result = _generate_aggregated_delta_heatmap(
        setup=setup,
        metric_labels=heatmap_metric_labels,
        models=all_models,
        baseline_metrics=baseline_metrics,
        ablation_metrics=ablation_metrics,
        out_path=out_path,
        after_desc=desc,
        before_desc=before_desc,
        ablation_name=desc,
        bootstrap_results=bootstrap_results,
    )
    md: list[str] = ["### Delta vs Baseline Heatmap", ""]
    if result is not None:
        md.append(f"![{setup.capitalize()} Impact](figures/{png_filename})")
    else:
        md.append("_(heatmap unavailable — matplotlib not installed)_")
    md.append("")
    return md


def _render_per_example_section(
    setup: str,
    all_models: list,
    baseline_name: str,
    lookup_abl: str,
    retrieval_index: dict,
    report_file: str,
    logger_inst,
) -> list[str]:
    """Render per-example debug analysis for pairwise/pointwise setups."""
    if setup not in ("pairwise", "pointwise"):
        return []
    md: list[str] = []
    for model in all_models:
        base_dir_str = retrieval_index.get((setup, baseline_name, model))
        abl_dir_str = retrieval_index.get((setup, lookup_abl, model))
        if not base_dir_str or not abl_dir_str:
            continue
        base_outcomes_by_metric = _extract_raw_outcomes_for_dir(
            Path(base_dir_str), setup, baseline_name, logger_inst
        )
        abl_outcomes_by_metric = _extract_raw_outcomes_for_dir(
            Path(abl_dir_str), setup, lookup_abl, logger_inst
        )
        if not base_outcomes_by_metric or not abl_outcomes_by_metric:
            continue
        if setup == "pairwise":
            base_ic = base_outcomes_by_metric.get("_pairwise_is_correct", {})
            base_it = base_outcomes_by_metric.get("_pairwise_is_tie", {})
            abl_ic = abl_outcomes_by_metric.get("_pairwise_is_correct", {})
            abl_it = abl_outcomes_by_metric.get("_pairwise_is_tie", {})
            base_metric = {pid: 0.5 if base_it.get(pid, 0.0) else v for pid, v in base_ic.items()}
            abl_metric = {pid: 0.5 if abl_it.get(pid, 0.0) else v for pid, v in abl_ic.items()}
        else:
            base_pred = base_outcomes_by_metric.get("_pointwise_pred", {})
            base_target = base_outcomes_by_metric.get("_pointwise_target", {})
            abl_pred = abl_outcomes_by_metric.get("_pointwise_pred", {})
            abl_target = abl_outcomes_by_metric.get("_pointwise_target", {})
            base_metric = {
                pid: 1.0 if int(pred >= 0.5) == int(base_target[pid] >= 0.5) else 0.0
                for pid, pred in base_pred.items() if pid in base_target
            }
            abl_metric = {
                pid: 1.0 if int(pred >= 0.5) == int(abl_target[pid] >= 0.5) else 0.0
                for pid, pred in abl_pred.items() if pid in abl_target
            }
        if not base_metric or not abl_metric:
            continue

        if report_file != "accuracy_report.txt":
            _excl = (
                _extract_exclude_set_from_filtered_report(Path(base_dir_str), report_filename=report_file)
                | _extract_exclude_set_from_filtered_report(Path(abl_dir_str), report_filename=report_file)
            )
            if _excl:
                base_metric = {pid: s for pid, s in base_metric.items() if pid not in _excl}
                abl_metric = {pid: s for pid, s in abl_metric.items() if pid not in _excl}

        instances_path_str = _extract_instances_path(Path(abl_dir_str))
        inst_data: dict = {}
        if instances_path_str:
            inst_p = Path(instances_path_str)
            if not inst_p.is_absolute():
                inst_p = _PROJECT_ROOT / inst_p
            if inst_p.exists():
                inst_data = _load_test_inputs(inst_p)

        md.extend(_generate_debug_examples_section(
            setup=setup,
            model=model,
            base_outcomes=base_metric,
            abl_outcomes=abl_metric,
            instance_data=inst_data,
            abl_artifact_dir=Path(abl_dir_str),
            base_artifact_dir=Path(base_dir_str),
        ))
    return md


def _generate_per_ablation_report(
    ablation_name: str,
    setup_to_base_name: dict[str, str | None],
    setup_to_index_key: dict[str, str | None],
    retrieval_index: dict,
    reports_dir: Path,
    desc: str,
    before_desc: str,
    logger_inst=None,
    report_file: str = "accuracy_report.txt",
    name_suffix: str = "",
    metric_labels_override: dict[str, dict[str, str]] | None = None,
) -> Path | None:
    """Write a per-ablation .md report with comparison tables and aggregated delta heatmaps."""
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', ablation_name)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    mismatch_suppressed = _load_mismatch_suppressed_ablations()
    config_table = _build_config_diff_table(ablation_name)

    md_lines: list[str] = [
        f"# Ablation Report: {desc}",
        "",
        f"**Before:** {before_desc or '(not specified)'}",
        f"**After (this ablation):** {desc}",
        "",
    ]
    if config_table:
        md_lines += [config_table, ""]

    any_section_written = False

    for setup in ["pointwise", "pairwise", "ranking"]:
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(f"## {setup.capitalize()} Results")
        md_lines.append("")

        baseline_name = setup_to_base_name.get(setup)
        if baseline_name is None:
            md_lines.append("_No baseline found for this setup._")
            md_lines.append("")
            continue

        # Use the pre-resolved index key for this setup (handles legacy names)
        lookup_abl = setup_to_index_key.get(setup)
        if lookup_abl is None:
            md_lines.append("_No data for this setup._")
            md_lines.append("")
            continue

        abl_models: set[str] = {
            model for (m, abl, model) in retrieval_index if m == setup and abl == lookup_abl
        }
        base_models: set[str] = {
            model for (m, abl, model) in retrieval_index if m == setup and abl == baseline_name
        }
        all_models = sorted(abl_models | base_models)

        if not all_models:
            md_lines.append("_No data for this setup._")
            md_lines.append("")
            continue

        # Extract metrics per model
        baseline_metrics: dict[str, dict] = {}
        ablation_metrics: dict[str, dict] = {}
        for model in all_models:
            base_dir = retrieval_index.get((setup, baseline_name, model))
            if base_dir:
                m_dict = _extract_metrics_for_dir(Path(base_dir), setup, logger_inst, report_file)
                if m_dict:
                    baseline_metrics[model] = m_dict
            abl_dir = retrieval_index.get((setup, lookup_abl, model))
            if abl_dir:
                m_dict = _extract_metrics_for_dir(Path(abl_dir), setup, logger_inst, report_file)
                if m_dict:
                    ablation_metrics[model] = m_dict

        if not baseline_metrics and not ablation_metrics:
            md_lines.append("_Could not extract metrics for this setup._")
            md_lines.append("")
            continue

        any_section_written = True
        metric_labels = (
            (metric_labels_override or {}).get(setup)
            or _SETUP_METRIC_LABELS.get(setup, {})
        )

        bootstrap_results = _run_bootstrap_for_setup(
            setup=setup,
            all_models=all_models,
            baseline_name=baseline_name,
            lookup_abl=lookup_abl,
            retrieval_index=retrieval_index,
            report_file=report_file,
            mismatch_suppressed=mismatch_suppressed,
            logger_inst=logger_inst,
        )
        md_lines.extend(_render_setup_tables(
            setup=setup,
            all_models=all_models,
            baseline_metrics=baseline_metrics,
            ablation_metrics=ablation_metrics,
            metric_labels=metric_labels,
            bootstrap_results=bootstrap_results,
        ))
        md_lines.extend(_render_call_quality_section(
            setup=setup,
            all_models=all_models,
            baseline_name=baseline_name,
            lookup_abl=lookup_abl,
            retrieval_index=retrieval_index,
        ))
        md_lines.extend(_render_confusion_table(
            setup=setup,
            all_models=all_models,
            baseline_name=baseline_name,
            lookup_abl=lookup_abl,
            retrieval_index=retrieval_index,
            report_file=report_file,
            logger_inst=logger_inst,
        ))
        md_lines.extend(_render_setup_heatmap(
            setup=setup,
            all_models=all_models,
            baseline_metrics=baseline_metrics,
            ablation_metrics=ablation_metrics,
            metric_labels=metric_labels,
            bootstrap_results=bootstrap_results,
            figures_dir=figures_dir,
            safe_name=safe_name,
            name_suffix=name_suffix,
            desc=desc,
            before_desc=before_desc,
        ))
        md_lines.extend(_render_per_example_section(
            setup=setup,
            all_models=all_models,
            baseline_name=baseline_name,
            lookup_abl=lookup_abl,
            retrieval_index=retrieval_index,
            report_file=report_file,
            logger_inst=logger_inst,
        ))

    if not any_section_written and logger_inst:
        logger_inst.warning(f"No data found for ablation '{ablation_name}' — report will be empty.")

    report_path = reports_dir / f"{safe_name}{name_suffix}.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    return report_path


def _generate_all_per_ablation_reports(
    retrieval_index: dict,
    output_dir: Path,
    yaml_descriptions: dict,
    before_descriptions: dict,
    yaml_aliases: dict,
    logger_inst=None,
    report_file: str = "accuracy_report.txt",
    name_suffix: str = "",
    metric_labels_override: dict[str, dict[str, str]] | None = None,
) -> list[Path]:
    """Generate per-ablation markdown reports for all non-baseline ablations."""
    # Load group info to identify and skip baselines
    baseline_canonical_names = _baseline_canonical_names()

    # Build reverse map: any variant name → canonical YAML key
    canonical_key_map = _build_canonical_key_map()

    # De-duplicate all index names → canonical YAML key
    seen_canonical: dict[str, list[str]] = {}  # canonical_yaml_key → [raw index names]
    for (mode, abl_name, _model) in retrieval_index:
        # Strip track prefix first, then resolve legacy names via canonical_key_map
        parts = abl_name.split("_", 1)
        stripped = parts[1] if len(parts) == 2 and parts[0] in ("pairwise", "ranking", "pointwise") else abl_name
        canonical = canonical_key_map.get(stripped, canonical_key_map.get(abl_name, stripped))
        if canonical not in seen_canonical:
            seen_canonical[canonical] = []
        if abl_name not in seen_canonical[canonical]:
            seen_canonical[canonical].append(abl_name)

    reports_dir = output_dir / "per_ablation_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # setup → the actual key in retrieval_index for this ablation (could be legacy/track-prefixed)
    tasks = []
    for canonical in sorted(seen_canonical):
        # Skip baselines
        if canonical in baseline_canonical_names:
            continue
        if any(canonical in yaml_aliases.get(b, []) for b in baseline_canonical_names):
            continue

        # Description and before_description always from canonical YAML key
        desc = yaml_descriptions.get(canonical, canonical)
        before_desc = before_descriptions.get(canonical, "")

        # All raw names in the index that map to this canonical ablation
        raw_names_in_index = seen_canonical[canonical]

        setup_to_base_name: dict[str, str | None] = {}
        setup_to_index_key: dict[str, str | None] = {}
        for setup in ["pointwise", "pairwise", "ranking"]:
            # Find whichever raw name from the index belongs to this setup
            index_key = next(
                (n for n in raw_names_in_index
                 if any(m == setup and abl == n for (m, abl, _) in retrieval_index)),
                None,
            )
            setup_to_index_key[setup] = index_key
            lookup_name = index_key or f"{setup}_{canonical}"
            setup_to_base_name[setup] = _resolve_baseline_name(
                setup, lookup_name, retrieval_index, yaml_aliases
            )

        tasks.append((canonical, setup_to_base_name, setup_to_index_key, desc, before_desc))

    def process_task(task):
        canonical, setup_to_base_name, setup_to_index_key, desc, before_desc = task
        return _generate_per_ablation_report(
            ablation_name=canonical,
            setup_to_base_name=setup_to_base_name,
            setup_to_index_key=setup_to_index_key,
            retrieval_index=retrieval_index,
            reports_dir=reports_dir,
            desc=desc,
            before_desc=before_desc,
            logger_inst=logger_inst,
            report_file=report_file,
            name_suffix=name_suffix,
            metric_labels_override=metric_labels_override,
        )

    generated: list[Path] = []
    if tasks:
        logger_inst.info(f"  Generating {len(tasks)} per-ablation reports in parallel...")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating reports"):
                res = future.result()
                if res:
                    generated.append(res)
                    try:
                        rel = res.relative_to(output_dir)
                    except ValueError:
                        rel = res
                    logger_inst.info(f"    Report: {rel}")

    return generated


def _generate_bootstrap_debug_csvs(
    retrieval_index: dict,
    yaml_aliases: dict,
    output_dir: Path,
    logger_inst=None,
    report_file: str = "accuracy_report.txt",
) -> Path | None:
    """
    For each (ablation, setup, model) triple, write a CSV with one row per example:

        pid | gt | base_pred | abl_pred | base_<metric> | abl_<metric> | delta_<metric> ...

    Scores are computed and averaged across runs in exactly the same way as the
    paired bootstrap inputs, so these CSVs let you inspect the raw data behind
    every significance test.

    Files land in output_dir/bootstrap_debug_csvs/{ablation}__{setup}__{model}.csv
    """
    import csv as _csv

    debug_dir = output_dir / "bootstrap_debug_csvs"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Identify baselines so we can skip them as the *subject* of comparison
    baseline_canonical_names = _baseline_canonical_names()

    canonical_key_map = _build_canonical_key_map()

    def _is_baseline(abl_name: str) -> bool:
        return _is_baseline_ablation(abl_name, canonical_key_map, baseline_canonical_names, yaml_aliases)

    def _load_run_scores(artifact_dir: Path, setup: str) -> list[dict]:
        """Return a list of scores dicts, one per run_* subdir."""
        glob = f"run_{setup}_*/scores.json"
        result = []
        for sf in sorted(artifact_dir.glob(glob)):
            try:
                with open(sf) as fh:
                    result.append(json.load(fh))
            except Exception:
                pass
        return result

    # -----------------------------------------------------------------------
    # Pairwise helpers
    # -----------------------------------------------------------------------
    def _pairwise_row_scores(runs: list[dict]) -> dict[str, dict]:
        """
        pid -> {pairwise_accuracy, pairwise_accuracy_no_ties, gt_winner, pred_winner}
        averages accuracy scores across runs (same as bootstrap).
        """
        per_pid: dict[str, list] = {}
        for scores in runs:
            for k, v in scores.items():
                if not isinstance(v, dict):
                    continue
                pid = _extract_instance_id_from_key(k)
                if pid is None:
                    pid = str(k)
                comps = v.get("comparisons", [])
                if not comps:
                    continue
                gt = comps[0].get("gt_winner")
                pred = comps[0].get("winner")
                if gt is None or pred is None:
                    continue
                is_tie = pred == 2
                if isinstance(gt, list):
                    correct = pred in gt
                else:
                    correct = pred == gt
                acc_wt = 0.5 if is_tie else (1.0 if correct else 0.0)
                acc_nt = None if is_tie else (1.0 if correct else 0.0)
                if pid not in per_pid:
                    per_pid[pid] = []
                per_pid[pid].append((gt, pred, acc_wt, acc_nt))

        result: dict[str, dict] = {}
        for pid, entries in per_pid.items():
            gts = [e[0] for e in entries]
            preds = [e[1] for e in entries]
            acc_wt_vals = [e[2] for e in entries]
            acc_nt_vals = [e[3] for e in entries if e[3] is not None]
            result[pid] = {
                "gt_winner": gts[0],
                "pred_winner": preds[0] if len(set(preds)) == 1 else f"mixed({','.join(str(p) for p in preds)})",
                "pairwise_accuracy": sum(acc_wt_vals) / len(acc_wt_vals),
                "pairwise_accuracy_no_ties": (sum(acc_nt_vals) / len(acc_nt_vals)) if acc_nt_vals else None,
            }
        return result

    # -----------------------------------------------------------------------
    # Pointwise helpers
    # -----------------------------------------------------------------------
    _PT_METRICS = ["accuracy"]

    def _pointwise_row_scores(runs: list[dict]) -> dict[str, dict]:
        """pid -> {gt_label, pred, accuracy} averaged across runs."""
        per_pid: dict[str, list] = {}
        for scores in runs:
            for k, v in scores.items():
                if not isinstance(v, dict):
                    continue
                pid = _extract_instance_id_from_key(k)
                if pid is None:
                    pid = str(k)
                pred = v.get("prediction")
                label = v.get("label")
                if pred is None or not label:
                    continue
                target = 1 if label == "POSITIVE" else 0
                correct = 1.0 if pred == target else 0.0

                row: dict[str, float | None] = {"accuracy": correct}

                if pid not in per_pid:
                    per_pid[pid] = []
                per_pid[pid].append((label, pred, row))

        result: dict[str, dict] = {}
        for pid, entries in per_pid.items():
            labels = [e[0] for e in entries]
            preds = [e[1] for e in entries]
            avg: dict[str, float | None] = {}
            for m in _PT_METRICS:
                vals = [e[2][m] for e in entries if e[2].get(m) is not None]
                avg[m] = sum(vals) / len(vals) if vals else None
            result[pid] = {
                "gt_label": labels[0],
                "pred": preds[0] if len(set(preds)) == 1 else f"mixed({','.join(str(p) for p in preds)})",
                **avg,
            }
        return result

    # -----------------------------------------------------------------------
    # Main loop: one CSV per (ablation, setup, model) triple
    # -----------------------------------------------------------------------
    seen_triples: set[tuple] = set()
    files_written: list[Path] = []
    _lock = threading.Lock()

    def _process_triple(item):
        (setup, abl_name, model), abl_dir_str = item
        if setup not in ("pairwise", "pointwise"):
            return
        if _is_baseline(abl_name):
            return

        triple = (abl_name, setup, model)
        with _lock:
            if triple in seen_triples:
                return
            seen_triples.add(triple)

        baseline_name = _resolve_baseline_name(setup, abl_name, retrieval_index, yaml_aliases)
        if not baseline_name:
            return
        base_dir_str = retrieval_index.get((setup, baseline_name, model))
        if not base_dir_str:
            return

        base_dir = Path(base_dir_str)
        abl_dir = Path(abl_dir_str)

        # Load and back-fill scores
        base_runs = _load_run_scores(base_dir, setup)
        abl_runs = _load_run_scores(abl_dir, setup)
        if not base_runs or not abl_runs:
            return

        inst_path = _extract_instances_path(abl_dir) or _extract_instances_path(base_dir)
        _backfill_labels_inplace(base_runs, inst_path, setup)
        _backfill_labels_inplace(abl_runs, inst_path, setup)

        if setup == "pairwise":
            base_data = _pairwise_row_scores(base_runs)
            abl_data = _pairwise_row_scores(abl_runs)
            metrics = ["pairwise_accuracy", "pairwise_accuracy_no_ties"]
            gt_col = "gt_winner"
            pred_col = "pred_winner"
        else:
            base_data = _pointwise_row_scores(base_runs)
            abl_data = _pointwise_row_scores(abl_runs)
            metrics = _PT_METRICS
            gt_col = "gt_label"
            pred_col = "pred"

        exclude_set: set[str] = set()
        if report_file != "accuracy_report.txt":
            exclude_set = (
                _extract_exclude_set_from_filtered_report(base_dir, report_filename=report_file)
                | _extract_exclude_set_from_filtered_report(abl_dir, report_filename=report_file)
            )

        def _pid_key(p: str):
            try:
                return (0, int(p))
            except ValueError:
                return (1, p)

        common_pids = sorted(
            {p for p in set(base_data) & set(abl_data) if p not in exclude_set},
            key=_pid_key,
        )
        if not common_pids:
            return

        # Build fieldnames: pid, gt, base_pred, abl_pred, then per-metric triples
        fieldnames = ["pid", "gt", "base_pred", "abl_pred"]
        for m in metrics:
            fieldnames += [f"base_{m}", f"abl_{m}", f"delta_{m}"]

        rows = []
        for pid in common_pids:
            bd = base_data[pid]
            ad = abl_data[pid]
            row: dict = {
                "pid": pid,
                "gt": bd.get(gt_col),
                "base_pred": bd.get(pred_col),
                "abl_pred": ad.get(pred_col),
            }
            for m in metrics:
                bv = bd.get(m)
                av = ad.get(m)
                row[f"base_{m}"] = "" if bv is None else bv
                row[f"abl_{m}"] = "" if av is None else av
                if bv is not None and av is not None:
                    row[f"delta_{m}"] = av - bv
                else:
                    row[f"delta_{m}"] = ""
            rows.append(row)

        safe_abl = re.sub(r"[^a-zA-Z0-9_-]", "_", abl_name)
        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        csv_path = debug_dir / f"{safe_abl}__{setup}__{safe_model}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        with _lock:
            files_written.append(csv_path)
        if logger_inst:
            logger_inst.info(f"Bootstrap debug CSV: {csv_path.relative_to(output_dir)}")

    items = list(retrieval_index.items())
    if items:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_triple, item) for item in items]
            list(tqdm(as_completed(futures), total=len(futures), desc="Generating debug CSVs"))

    if not files_written:
        if logger_inst:
            logger_inst.info("No bootstrap debug CSVs written (no paired ablation/baseline data found).")
        return None

    return debug_dir


def _generate_test_sets_consistency_report(
    retrieval_index: dict[tuple[str, str, str], str],
    output_dir: Path,
    blocked_titles: set[str],
    logger_inst=None,
) -> Path | None:
    """
    Generate a markdown report verifying that pointwise and pairwise test instances
    contain the exact same ideas (text and titles). Repeats the check for filtered data.
    """
    from collections import defaultdict, Counter

    # 1. Gather all instances paths per ablation_name and mode, normalized by canonical key
    canonical_key_map = _build_canonical_key_map()
    ablation_instances = defaultdict(lambda: {"pointwise": set(), "pairwise": set()})
    
    def normalize_ablation_name(name: str) -> str:
        # Resolve to canonical name if possible
        if name in canonical_key_map:
            return canonical_key_map[name]
        
        # Strip common mode prefixes (with either _ or -)
        temp = name
        for prefix in ("pairwise", "pointwise", "ranking"):
            if temp.startswith(f"{prefix}_"):
                temp = temp[len(prefix)+1:]
                break
            if temp.startswith(f"{prefix}-"):
                temp = temp[len(prefix)+1:]
                break
        
        if temp in canonical_key_map:
            return canonical_key_map[temp]
        return temp

    for (mode, abl_name, _model), artifact_dir_str in retrieval_index.items():
        if mode not in ("pointwise", "pairwise"):
            continue
            
        canonical = normalize_ablation_name(abl_name)
        instances_path = _extract_instances_path(Path(artifact_dir_str))
        if instances_path:
            ablation_instances[canonical][mode].add(instances_path)

    if not ablation_instances:
        if logger_inst:
            logger_inst.info("No pointwise/pairwise ablation instances found; skipping consistency report.")
        return None

    report_lines = [
        "# Test Sets Consistency Report",
        "",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report checks whether the pointwise and pairwise test instances for each ablation contain the exact same ideas (exact same text and titles).",
        "",
    ]

    def extract_ideas(
        instances_path: str,
        mode: str,
        exclude_indices: set[str] = None,
        only_positives: bool = False
    ) -> list[tuple[str, str]]:
        if not exclude_indices:
            exclude_indices = set()
        # Ensure all exclude_indices are strings
        exclude_indices = {str(i) for i in exclude_indices}

        ideas = []
        p = Path(instances_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.exists():
            return ideas
        data = _load_yaml_cached(p)

        for idx_str, entry in data.items():
            if not isinstance(entry, dict):
                continue
            if str(idx_str) in exclude_indices:
                continue

            meta = entry.get("metadata", {})
            if mode == "pairwise":
                expected_winners = entry.get("expected_winners", [])
                expected_winners = [str(w) for w in expected_winners]
                ideas_dict = entry.get("ideas", {})
                if not isinstance(meta, dict):
                    continue
                for k, idea_meta in meta.items():
                    if not isinstance(idea_meta, dict):
                        continue
                    if only_positives and str(k) not in expected_winners:
                        continue
                    title = idea_meta.get("title", "")
                    idea_text = ideas_dict.get(k) or ideas_dict.get(int(k) if isinstance(k, str) else str(k)) or ""
                    ideas.append((str(idea_text).strip(), str(title).strip()))
            elif mode == "pointwise":
                label = str(entry.get("label", "")).upper()
                if only_positives and label not in ("POS", "POSITIVE"):
                    continue
                if not isinstance(meta, dict):
                    continue
                title = meta.get("title", "")
                idea_text = entry.get("idea", "")
                ideas.append((str(idea_text).strip(), str(title).strip()))
        return ideas

    global_positives_unfiltered: dict[tuple[str, str], Counter] = {}
    global_positives_filtered: dict[tuple[str, str], Counter] = {}
    _lock = threading.Lock()

    def _process_ablation(ablation_name):
        paths = ablation_instances[ablation_name]
        pw_paths = sorted(list(paths["pointwise"]))
        pair_paths = sorted(list(paths["pairwise"]))

        if pw_paths:
            pw_path = pw_paths[0]
            unfilt = Counter(extract_ideas(pw_path, "pointwise", only_positives=True))
            
            pw_exclude_map = _derive_exclude_indices(pw_path, blocked_titles)
            partner_excluded = _derive_pointwise_partner_excluded(pw_path, blocked_titles)
            pw_exclude = set(pw_exclude_map.keys()) | set(partner_excluded.keys())
            filt = Counter(extract_ideas(pw_path, "pointwise", pw_exclude, only_positives=True))
            with _lock:
                global_positives_unfiltered[(ablation_name, "pointwise")] = unfilt
                global_positives_filtered[(ablation_name, "pointwise")] = filt

        if pair_paths:
            pair_path = pair_paths[0]
            unfilt = Counter(extract_ideas(pair_path, "pairwise", only_positives=True))
            
            pair_exclude_map = _derive_exclude_indices(pair_path, blocked_titles)
            pair_exclude = set(pair_exclude_map.keys())
            filt = Counter(extract_ideas(pair_path, "pairwise", pair_exclude, only_positives=True))
            with _lock:
                global_positives_unfiltered[(ablation_name, "pairwise")] = unfilt
                global_positives_filtered[(ablation_name, "pairwise")] = filt

    ablation_names = sorted(ablation_instances.keys())
    if ablation_names:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_ablation, name) for name in ablation_names]
            list(tqdm(as_completed(futures), total=len(futures), desc="Checking consistency"))

    def generate_global_check_lines(positives_dict: dict[tuple[str, str], Counter], label: str) -> list[str]:
        if not positives_dict:
            return []
        
        lines = [f"## Global Consistency: {label} Positive Ideas", ""]
        
        # We compare everyone against the first available entry
        first_key = next(iter(sorted(positives_dict.keys())))
        baseline_positives = positives_dict[first_key]
        
        mismatches = []
        for key in sorted(positives_dict.keys()):
            if positives_dict[key] != baseline_positives:
                mismatches.append(key)
        
        if not mismatches:
            count = sum(baseline_positives.values())
            lines.append(f"✅ **Global Consistency Verified**: All {len(positives_dict)} test sets ({label}) contain the exact same {count} positive ideas.")
        else:
            lines.append(f"❌ **Global Mismatch Detected**: The set of positive ideas ({label}) is not identical across all test sets.")
            lines.append(f"- Reference set: `{first_key[0]}` ({first_key[1]})")
            for key in mismatches:
                abl, mode = key
                lines.append(f"- Mismatch in `{abl}` ({mode})")
        
        lines.append("")
        return lines

    report_lines.extend(generate_global_check_lines(global_positives_unfiltered, "Unfiltered"))
    report_lines.extend(generate_global_check_lines(global_positives_filtered, "Filtered"))

    for ablation_name in sorted(ablation_instances.keys()):
        paths = ablation_instances[ablation_name]
        pw_paths = sorted(list(paths["pointwise"]))
        pair_paths = sorted(list(paths["pairwise"]))
        
        report_lines.append(f"## Per-Ablation Breakdown: `{ablation_name}`")
        report_lines.append("")
        
        if not pw_paths:
            report_lines.append("⚠️ **Missing Pointwise data** for this ablation in the current merge.")
            report_lines.append(f"- Found {len(pair_paths)} pairwise instance path(s).")
            report_lines.append("")
            continue
        if not pair_paths:
            report_lines.append("⚠️ **Missing Pairwise data** for this ablation in the current merge.")
            report_lines.append(f"- Found {len(pw_paths)} pointwise instance path(s).")
            report_lines.append("")
            continue
            
        pw_path = pw_paths[0]
        pair_path = pair_paths[0]
        
        # Unfiltered check
        pw_ideas = extract_ideas(pw_path, "pointwise")
        pair_ideas = extract_ideas(pair_path, "pairwise")
        
        pw_counter = Counter(pw_ideas)
        pair_counter = Counter(pair_ideas)
        
        report_lines.append("### Pointwise vs Pairwise (Unfiltered)")
        if pw_counter == pair_counter:
            report_lines.append("✅ Pointwise and pairwise instances contain the **exact same ideas**.")
        else:
            report_lines.append("❌ Mismatch detected between pointwise and pairwise ideas.")
            
            in_pw_not_pair = pw_counter - pair_counter
            in_pair_not_pw = pair_counter - pw_counter
            
            if in_pw_not_pair:
                report_lines.append("")
                report_lines.append("**Ideas in Pointwise but not in Pairwise:**")
                for (text, title), count in in_pw_not_pair.items():
                    report_lines.append(f"- Title: `{title}` (Count: {count})")
                    text_snippet = text[:100].replace('\n', ' ') + ('...' if len(text) > 100 else '')
                    report_lines.append(f"  - Text: {text_snippet}")
                
            if in_pair_not_pw:
                report_lines.append("")
                report_lines.append("**Ideas in Pairwise but not in Pointwise:**")
                for (text, title), count in in_pair_not_pw.items():
                    report_lines.append(f"- Title: `{title}` (Count: {count})")
                    text_snippet = text[:100].replace('\n', ' ') + ('...' if len(text) > 100 else '')
                    report_lines.append(f"  - Text: {text_snippet}")
        report_lines.append("")
                
        # Filtered check
        pair_exclude_map = _derive_exclude_indices(pair_path, blocked_titles)
        pair_exclude = set(pair_exclude_map.keys())
        
        pw_exclude_map = _derive_exclude_indices(pw_path, blocked_titles)
        partner_excluded = _derive_pointwise_partner_excluded(pw_path, blocked_titles)
        pw_exclude = set(pw_exclude_map.keys()) | set(partner_excluded.keys())
        
        filt_pw_ideas = extract_ideas(pw_path, "pointwise", pw_exclude)
        filt_pair_ideas = extract_ideas(pair_path, "pairwise", pair_exclude)
        
        filt_pw_counter = Counter(filt_pw_ideas)
        filt_pair_counter = Counter(filt_pair_ideas)
        
        report_lines.append("### Pointwise vs Pairwise (Filtered)")
        if filt_pw_counter == filt_pair_counter:
            report_lines.append("✅ After filtering blocklisted papers, pointwise and pairwise instances contain the **exact same ideas**.")
        else:
            report_lines.append("❌ Mismatch detected between pointwise and pairwise ideas after filtering.")
            
            in_pw_not_pair = filt_pw_counter - filt_pair_counter
            in_pair_not_pw = filt_pair_counter - filt_pw_counter
            
            if in_pw_not_pair:
                report_lines.append("")
                report_lines.append("**Ideas in Pointwise but not in Pairwise (Filtered):**")
                for (text, title), count in in_pw_not_pair.items():
                    report_lines.append(f"- Title: `{title}` (Count: {count})")
                    text_snippet = text[:100].replace('\n', ' ') + ('...' if len(text) > 100 else '')
                    report_lines.append(f"  - Text: {text_snippet}")
                
            if in_pair_not_pw:
                report_lines.append("")

                report_lines.append("**Ideas in Pairwise but not in Pointwise (Filtered):**")
                for (text, title), count in in_pair_not_pw.items():
                    report_lines.append(f"- Title: `{title}` (Count: {count})")
                    text_snippet = text[:100].replace('\n', ' ') + ('...' if len(text) > 100 else '')
                    report_lines.append(f"  - Text: {text_snippet}")
        report_lines.append("")

    report_path = output_dir / "test_sets_consistency_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    if logger_inst:
        logger_inst.info(f"Test sets consistency report written to: {report_path}")
    return report_path


def _extract_usage_from_record(record: dict) -> dict | None:
    """Normalize one batch output JSONL record into {input, output, reasoning} token counts.

    Supports OpenAI (response.body.usage) and Anthropic (result.message.usage) formats.
    Returns None for failed or unrecognised records.
    """
    # OpenAI format
    if "response" in record:
        resp = record["response"]
        if resp.get("status_code") != 200:
            return None
        usage = resp.get("body", {}).get("usage", {})
        if usage.get("prompt_tokens") is not None:
            return {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
                "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
            }

    # Anthropic format
    if "result" in record:
        result_obj = record["result"]
        if result_obj.get("type") != "succeeded":
            return None
        usage = result_obj.get("message", {}).get("usage", {})
        if usage.get("input_tokens") is not None:
            # Claude thinking tokens (extended/adaptive thinking) are a subset of
            # output_tokens, surfaced as a read-only breakdown — same semantics as
            # OpenAI's completion_tokens_details.reasoning_tokens above. Absent on
            # records produced without thinking or by older API versions.
            return {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "reasoning": (usage.get("output_tokens_details") or {}).get("thinking_tokens", 0) or 0,
            }

    return None


def _collect_token_stats_for_dir(artifact_dir: Path) -> dict | None:
    """Aggregate token usage across all batch_output_*.jsonl files under an artifact dir.

    Returns a stats dict or None if no batch files are found.
    """
    import statistics as _stats

    batch_files = sorted(artifact_dir.glob("run_*/batch_output_*.jsonl"))
    if not batch_files:
        return None

    all_inputs: list[int] = []
    all_outputs: list[int] = []
    all_reasonings: list[int] = []
    all_non_reasonings: list[int] = []

    for bf in batch_files:
        try:
            with open(bf, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    usage = _extract_usage_from_record(record)
                    if usage is None:
                        continue
                    all_inputs.append(usage["input"])
                    all_outputs.append(usage["output"])
                    all_reasonings.append(usage["reasoning"])
                    # Reasoning tokens are a subset of output; the remainder is the
                    # visible/answer output. max(0, ...) guards against any record
                    # where the reported breakdown exceeds output.
                    all_non_reasonings.append(max(0, usage["output"] - usage["reasoning"]))
        except Exception:
            continue

    if not all_inputs:
        return None

    n_calls = len(all_inputs)

    return {
        "n_calls": n_calls,
        "total_input": sum(all_inputs),
        "total_output": sum(all_outputs),
        "total_reasoning": sum(all_reasonings),
        "total_non_reasoning": sum(all_non_reasonings),
        "mean_input": sum(all_inputs) / n_calls,
        "median_input": _stats.median(all_inputs),
        "mean_output": sum(all_outputs) / n_calls,
        "median_output": _stats.median(all_outputs),
        "mean_reasoning": sum(all_reasonings) / n_calls,
        "median_reasoning": _stats.median(all_reasonings),
        "mean_non_reasoning": sum(all_non_reasonings) / n_calls,
        "median_non_reasoning": _stats.median(all_non_reasonings),
        "has_reasoning": any(r > 0 for r in all_reasonings),
    }


def _generate_token_usage_report(
    retrieval_index: dict,
    yaml_descriptions: dict,
    output_dir: Path,
    logger_inst=None,
) -> Path | None:
    """Generate token_usage_report.md with per-ablation/model token stats from batch outputs.

    Columns: calls, total/avg/median input & output tokens, and reasoning when available.
    """
    dir_stats: dict[str, dict | None] = {}
    _lock = threading.Lock()

    def _fetch(item):
        _, artifact_dir_str = item
        stats = _collect_token_stats_for_dir(Path(artifact_dir_str))
        with _lock:
            dir_stats[artifact_dir_str] = stats

    items = list(retrieval_index.items())
    if items:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_fetch, it) for it in items]
            list(tqdm(as_completed(futures), total=len(futures), desc="Collecting token stats"))

    if not any(v is not None for v in dir_stats.values()):
        if logger_inst:
            logger_inst.info("No batch output files found — skipping token usage report.")
        return None

    def _lookup_desc(abl_name: str) -> str:
        if abl_name in yaml_descriptions:
            return yaml_descriptions[abl_name]
        parts = abl_name.split("_", 1)
        stripped = (
            parts[1]
            if len(parts) == 2 and parts[0] in ("pairwise", "ranking", "pointwise")
            else abl_name
        )
        return yaml_descriptions.get(stripped, abl_name)

    def _is_baseline_key(abl_name: str) -> bool:
        bare = abl_name.split("_", 1)[-1] if "_" in abl_name else abl_name
        return bare in ("current", "baseline") or bare.startswith("current_") or bare.startswith("baseline_")

    md: list[str] = [
        "# Token Usage Report",
        "",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Token statistics from batch output JSONL files. "
        "The Reasoning column counts tokens spent on internal reasoning/thinking and "
        "is a subset of Output (already included in the Output totals). It is sourced "
        "from `completion_tokens_details.reasoning_tokens` for OpenAI models and "
        "`output_tokens_details.thinking_tokens` for Anthropic/Claude extended or "
        "adaptive thinking. Non-Reasoning is the remainder (Output − Reasoning), i.e. "
        "the answer/visible output tokens; it equals Output for calls made without "
        "thinking. The Reasoning/Non-Reasoning columns appear whenever any model in "
        "the section used thinking.",
        "",
    ]

    for setup in ("pairwise", "pointwise", "ranking"):
        # Build abl_name -> model -> stats mapping for this setup
        setup_entries: dict[str, dict[str, dict | None]] = {}
        for (s, abl_name, model), artifact_dir_str in retrieval_index.items():
            if s != setup:
                continue
            stats = dir_stats.get(artifact_dir_str)
            if abl_name not in setup_entries:
                setup_entries[abl_name] = {}
            setup_entries[abl_name][model] = stats

        if not setup_entries:
            continue

        has_reasoning = any(
            (s or {}).get("has_reasoning", False)
            for model_stats in setup_entries.values()
            for s in model_stats.values()
        )

        md.append(f"## {setup.capitalize()}")
        md.append("")

        def _sort_key(abl_name: str) -> tuple:
            return (0 if _is_baseline_key(abl_name) else 1, _lookup_desc(abl_name).lower())

        for abl_name in sorted(setup_entries, key=_sort_key):
            model_stats = setup_entries[abl_name]
            desc = _lookup_desc(abl_name)
            md.append(f"### {desc}")
            md.append("")

            header = [
                "Model", "Calls",
                "Input Total", "Input Avg", "Input Median",
                "Output Total", "Output Avg", "Output Median",
            ]
            if has_reasoning:
                header += [
                    "Reasoning Total", "Reasoning Avg", "Reasoning Median",
                    "Non-Reasoning Total", "Non-Reasoning Avg", "Non-Reasoning Median",
                ]

            md.append("| " + " | ".join(header) + " |")
            md.append("| " + " | ".join("---" for _ in header) + " |")

            def _i(n: float) -> str:
                return f"{int(n):,}"

            def _f(n: float) -> str:
                return f"{n:,.1f}"

            for model in sorted(model_stats):
                s = model_stats[model]
                if s is None:
                    row = [f"**{model}**"] + ["—"] * (len(header) - 1)
                else:
                    row = [
                        f"**{model}**",
                        _i(s["n_calls"]),
                        _i(s["total_input"]),
                        _f(s["mean_input"]),
                        _f(s["median_input"]),
                        _i(s["total_output"]),
                        _f(s["mean_output"]),
                        _f(s["median_output"]),
                    ]
                    if has_reasoning:
                        row += [
                            _i(s["total_reasoning"]),
                            _f(s["mean_reasoning"]),
                            _f(s["median_reasoning"]),
                            _i(s["total_non_reasoning"]),
                            _f(s["mean_non_reasoning"]),
                            _f(s["median_non_reasoning"]),
                        ]
                md.append("| " + " | ".join(row) + " |")

            md.append("")

    report_path = output_dir / "token_usage_report.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    if logger_inst:
        logger_inst.info(f"Token usage report written to: {report_path}")
    return report_path


def _generate_ablation_overview_figures(
    retrieval_index: dict,
    yaml_descriptions: dict,
    yaml_aliases: dict,
    output_dir: Path,
    logger_inst=None,
    report_file: str = "accuracy_report.txt",
    name_suffix: str = "",
    metric_labels_override: dict[str, dict[str, str]] | None = None,
) -> list[Path]:
    """
    Generate one overview delta heatmap PNG per judge model.

    Rows = ablations, columns = metrics split into two panels:
      left  — Pointwise (F1 macro | F1 POS | F1 NEG | Accuracy)
      right — Pairwise  (Acc w/ ties | Acc strict)

    Cells show the delta vs baseline; * marks a statistically significant delta
    (95% BCa paired bootstrap, n=100,000).  Figures land in
    output_dir/overview_heatmaps/.
    """
    try:
        import numpy as np
    except ImportError:
        if logger_inst:
            logger_inst.warning("numpy not available — skipping overview heatmaps.")
        return []

    canonical_key_map = _build_canonical_key_map()
    baseline_canonical_names = _baseline_canonical_names()

    def _is_bl(abl_name: str) -> bool:
        return _is_baseline_ablation(abl_name, canonical_key_map, baseline_canonical_names, yaml_aliases)

    mismatch_suppressed = _load_mismatch_suppressed_ablations()

    all_models: set[str] = set()
    seen_canonical: dict[str, list[str]] = {}

    for (mode, abl_name, model) in retrieval_index:
        all_models.add(model)
        if _is_bl(abl_name):
            continue
        parts = abl_name.split("_", 1)
        stripped = parts[1] if len(parts) == 2 and parts[0] in ("pairwise", "ranking", "pointwise") else abl_name
        canonical = canonical_key_map.get(stripped, canonical_key_map.get(abl_name, stripped))
        # Also try stripping a mode suffix (foo_pairwise / foo_pointwise → foo)
        # so that mode-specific sibling ablations collapse onto one row.
        if canonical == stripped:
            for _sfx in ("_pairwise", "_pointwise", "_ranking"):
                if stripped.endswith(_sfx):
                    base = stripped[: -len(_sfx)]
                    canonical = canonical_key_map.get(base, base)
                    break
        if canonical not in seen_canonical:
            seen_canonical[canonical] = []
        if abl_name not in seen_canonical[canonical]:
            seen_canonical[canonical].append(abl_name)

    if not seen_canonical:
        if logger_inst:
            logger_inst.info("No non-baseline ablations found — skipping overview heatmaps.")
        return []

    all_models_sorted = sorted(all_models)
    all_canonical_sorted = sorted(seen_canonical)

    POINTWISE_METRICS = [
        ("f1_macro", "F1 macro"),
        ("f1_pos", "F1 POS"),
        ("f1_neg", "F1 NEG"),
        ("accuracy", "Accuracy"),
        ("precision_pos", "Prec POS"),
        ("precision_neg", "Prec NEG"),
        ("recall_pos", "Rec POS"),
        ("recall_neg", "Rec NEG"),
    ]
    PAIRWISE_METRICS = [
        ("pairwise_accuracy", "Acc (w/ ties)"),
        ("pairwise_accuracy_strict", "Acc (strict)"),
        ("pairwise_accuracy_no_ties", "Acc (w/o ties)"),
    ]
    # Apply metric filter from config — preserve the default ordering above but
    # drop any key the user didn't ask for.
    if metric_labels_override:
        if "pointwise" in metric_labels_override:
            _ptw_keys = set(metric_labels_override["pointwise"])
            POINTWISE_METRICS = [(k, l) for k, l in POINTWISE_METRICS if k in _ptw_keys]
        if "pairwise" in metric_labels_override:
            _pair_keys = set(metric_labels_override["pairwise"])
            PAIRWISE_METRICS = [(k, l) for k, l in PAIRWISE_METRICS if k in _pair_keys]
    else:
        # Default: use the original curated subset (compact overview)
        POINTWISE_METRICS = [("f1_macro", "F1 macro"), ("f1_pos", "F1 POS"), ("f1_neg", "F1 NEG")]
        PAIRWISE_METRICS = [("pairwise_accuracy", "Acc (w/ ties)"), ("pairwise_accuracy_strict", "Acc (strict)")]
    all_metric_specs = (
        [(k, l, "pointwise") for k, l in POINTWISE_METRICS]
        + [(k, l, "pairwise") for k, l in PAIRWISE_METRICS]
    )
    n_pointwise = len(POINTWISE_METRICS)

    # Resolve (index_key, baseline_name) per canonical × setup
    canonical_setup_info: dict[str, dict[str, tuple[str, str]]] = {}
    for canonical in all_canonical_sorted:
        raw_names = seen_canonical[canonical]
        canonical_setup_info[canonical] = {}
        for setup in ("pointwise", "pairwise"):
            index_key = next(
                (n for n in raw_names
                 if any(m == setup and abl == n for (m, abl, _) in retrieval_index)),
                None,
            )
            if not index_key:
                continue
            baseline_name = _resolve_baseline_name(setup, index_key, retrieval_index, yaml_aliases)
            if baseline_name:
                canonical_setup_info[canonical][setup] = (index_key, baseline_name)

    # Parallel precompute: metrics + bootstrap for every (canonical, setup) pair
    setup_base_metrics: dict[tuple[str, str], dict[str, dict]] = {}
    setup_abl_metrics: dict[tuple[str, str], dict[str, dict]] = {}
    bootstrap_results_all: dict[tuple[str, str], dict] = {}

    def _compute_for_pair(canonical: str, setup: str, index_key: str, baseline_name: str):
        base_m: dict[str, dict] = {}
        abl_m: dict[str, dict] = {}
        for model in all_models_sorted:
            base_dir_str = retrieval_index.get((setup, baseline_name, model))
            abl_dir_str = retrieval_index.get((setup, index_key, model))
            if base_dir_str:
                m = _extract_metrics_for_dir(Path(base_dir_str), setup, logger_inst, report_file)
                if m:
                    base_m[model] = m
            if abl_dir_str:
                m = _extract_metrics_for_dir(Path(abl_dir_str), setup, logger_inst, report_file)
                if m:
                    abl_m[model] = m
        bs = _run_bootstrap_for_setup(
            setup=setup,
            all_models=all_models_sorted,
            baseline_name=baseline_name,
            lookup_abl=index_key,
            retrieval_index=retrieval_index,
            report_file=report_file,
            mismatch_suppressed=mismatch_suppressed,
            logger_inst=logger_inst,
        )
        return canonical, setup, base_m, abl_m, bs

    tasks_cs = [
        (canonical, setup, index_key, baseline_name)
        for canonical, setup_info in canonical_setup_info.items()
        for setup, (index_key, baseline_name) in setup_info.items()
    ]
    if tasks_cs:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_compute_for_pair, *t) for t in tasks_cs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Overview heatmap data"):
                canonical, setup, base_m, abl_m, bs = future.result()
                setup_base_metrics[(canonical, setup)] = base_m
                setup_abl_metrics[(canonical, setup)] = abl_m
                bootstrap_results_all[(canonical, setup)] = bs

    # Build numpy matrices and render one figure per model
    figures_dir = output_dir / "overview_heatmaps"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ablation_labels = list(all_canonical_sorted)
    n_abl = len(all_canonical_sorted)
    n_metrics = len(all_metric_specs)
    generated: list[Path] = []

    for model in all_models_sorted:
        delta_mat    = np.full((n_abl, n_metrics), np.nan)
        base_mat     = np.full((n_abl, n_metrics), np.nan)
        abs_mat      = np.full((n_abl, n_metrics), np.nan)
        sig_mat      = np.zeros((n_abl, n_metrics), dtype=bool)
        reliable_mat = np.ones((n_abl, n_metrics), dtype=bool)

        for i, canonical in enumerate(all_canonical_sorted):
            for j, (metric_key, _, setup) in enumerate(all_metric_specs):
                bm = setup_base_metrics.get((canonical, setup), {}).get(model, {})
                am = setup_abl_metrics.get((canonical, setup), {}).get(model, {})
                bs = bootstrap_results_all.get((canonical, setup), {})
                bv = bm.get(metric_key)
                av = am.get(metric_key)
                if bv is not None and av is not None:
                    delta_mat[i, j] = av - bv
                    base_mat[i, j]  = bv
                    abs_mat[i, j]   = av
                    bs_entry = bs.get((model, metric_key), {})
                    sig_mat[i, j]      = bool(bs_entry.get("significant", False))
                    reliable_mat[i, j] = bool(bs_entry.get("reliable", True))

        # Drop rows that have no data in either the pointwise or pairwise section.
        has_ptw  = ~np.all(np.isnan(delta_mat[:, :n_pointwise]),  axis=1)
        has_pair = ~np.all(np.isnan(delta_mat[:, n_pointwise:]), axis=1)
        keep = has_ptw & has_pair
        if not keep.any():
            continue
        kept_labels = [ablation_labels[i] for i in range(n_abl) if keep[i]]
        delta_mat    = delta_mat[keep]
        base_mat     = base_mat[keep]
        abs_mat      = abs_mat[keep]
        sig_mat      = sig_mat[keep]
        reliable_mat = reliable_mat[keep]

        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        out_path = figures_dir / f"overview_{safe_model}{name_suffix}.png"
        metric_specs_for_viz = [(l, s) for _, l, s in all_metric_specs]
        result = _generate_overview_heatmap(
            ablation_labels=kept_labels,
            metric_specs=metric_specs_for_viz,
            delta_matrix=delta_mat,
            sig_matrix=sig_mat,
            reliable_matrix=reliable_mat,
            out_path=out_path,
            model=model,
            n_pointwise=n_pointwise,
            base_matrix=base_mat,
            abs_matrix=abs_mat,
        )
        if result:
            generated.append(result)
            if logger_inst:
                logger_inst.info(f"Overview heatmap: {result.relative_to(output_dir)}")

    return generated


# ============================================================
# Unified side-by-side chart
# ============================================================

def _generate_unified_charts(
    retrieval_index: dict,
    yaml_descriptions: dict,
    yaml_aliases: dict,
    unified_ablations: list[str],
    output_dir: Path,
    metric_labels_override: dict[str, dict[str, str]] | None = None,
    unified_chart_titles: dict[str, str] | None = None,
    report_file: str = "accuracy_report.txt",
    name_suffix: str = "",
    logger_inst=None,
) -> list[Path]:
    """
    For each setup (pointwise, pairwise), generate one PNG that places the
    per-ablation delta heatmaps for every entry in *unified_ablations* side by
    side in a single figure.  Panels share a colour scale so deltas are
    visually comparable.

    Alongside each PNG, write unified_{setup}{name_suffix}.json holding the data
    behind the figure — for *every* non-baseline ablation, not just the rendered
    ones — so rechart_unified.py can redraw it in any format without repeating
    the metric extraction and bootstrap.

    unified_ablations: canonical ablation names (same format as skip_ablations).
    """
    if not unified_ablations:
        return []

    canonical_key_map = _build_canonical_key_map()
    mismatch_suppressed = _load_mismatch_suppressed_ablations()
    baseline_canonical_names = _baseline_canonical_names()

    # ── Map every retrieval_index entry to its canonical ablation name ────────
    seen_canonical: dict[str, list[str]] = {}
    all_models: set[str] = set()
    for (mode, abl_name, model) in retrieval_index:
        all_models.add(model)
        parts   = abl_name.split("_", 1)
        stripped = (parts[1]
                    if len(parts) == 2 and parts[0] in ("pairwise", "ranking", "pointwise")
                    else abl_name)
        canonical = canonical_key_map.get(stripped, canonical_key_map.get(abl_name, stripped))
        if canonical not in seen_canonical:
            seen_canonical[canonical] = []
        if abl_name not in seen_canonical[canonical]:
            seen_canonical[canonical].append(abl_name)

    all_models_sorted = sorted(all_models)

    unified_dir = output_dir / "unified_charts"
    unified_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for setup in ("pointwise", "pairwise"):
        # Derive metric labels (apply user filter then heatmap-appropriate subset)
        metric_labels: dict[str, str] = dict(
            (metric_labels_override or {}).get(setup)
            or _SETUP_METRIC_LABELS.get(setup, {})
        )
        if setup == "pointwise":
            _heatmap_keys = {"f1_macro", "f1_pos", "f1_neg", "accuracy",
                             "precision_pos", "precision_neg", "recall_pos", "recall_neg"}
            metric_labels = {k: v for k, v in metric_labels.items() if k in _heatmap_keys}
        else:
            _drop = {"n_ties", "support", "support_without_ties", "cost_usd"}
            metric_labels = {k: v for k, v in metric_labels.items() if k not in _drop}

        if not metric_labels:
            continue

        def _build_panel(req_canonical: str, resolved: str, raw_names: list[str],
                         warn: bool) -> dict | None:
            """Assemble one panel's metrics + bootstrap, or None if it has no data."""
            index_key = next(
                (n for n in raw_names
                 if any(m == setup and abl == n for (m, abl, _) in retrieval_index)),
                None,
            )
            if not index_key:
                if warn and logger_inst:
                    logger_inst.debug(
                        f"Unified chart: '{req_canonical}' has no {setup} data — panel skipped."
                    )
                return None

            baseline_name = _resolve_baseline_name(setup, index_key, retrieval_index, yaml_aliases)
            if not baseline_name:
                if warn and logger_inst:
                    logger_inst.warning(
                        f"Unified chart ({setup}): no baseline for '{req_canonical}' — panel skipped."
                    )
                return None

            # Collect per-model metrics
            baseline_metrics: dict[str, dict] = {}
            ablation_metrics: dict[str, dict] = {}
            for model in all_models_sorted:
                base_dir = retrieval_index.get((setup, baseline_name, model))
                if base_dir:
                    m = _extract_metrics_for_dir(Path(base_dir), setup, logger_inst, report_file)
                    if m:
                        baseline_metrics[model] = m
                abl_dir = retrieval_index.get((setup, index_key, model))
                if abl_dir:
                    m = _extract_metrics_for_dir(Path(abl_dir), setup, logger_inst, report_file)
                    if m:
                        ablation_metrics[model] = m

            if not ablation_metrics:
                return None

            bootstrap_results = _run_bootstrap_for_setup(
                setup=setup,
                all_models=all_models_sorted,
                baseline_name=baseline_name,
                lookup_abl=index_key,
                retrieval_index=retrieval_index,
                report_file=report_file,
                mismatch_suppressed=mismatch_suppressed,
                logger_inst=logger_inst,
            )

            _titles = unified_chart_titles or {}
            desc = (
                _titles.get(req_canonical)
                or _titles.get(resolved)
                or yaml_descriptions.get(resolved)
                or yaml_descriptions.get(req_canonical, req_canonical)
            )
            return {
                "canonical": resolved,
                "index_key": index_key,
                "baseline_name": baseline_name,
                "desc": desc,
                "baseline_metrics": baseline_metrics,
                "ablation_metrics": ablation_metrics,
                "bootstrap_results": bootstrap_results,
            }

        # ── Requested panels — rendered, in the order given in the config ─────
        panels_by_canonical: dict[str, dict] = {}
        rendered: list[str] = []
        for req_canonical in unified_ablations:
            # Allow the user to write the name with or without track prefix
            resolved = canonical_key_map.get(req_canonical, req_canonical)
            raw_names = seen_canonical.get(resolved) or seen_canonical.get(req_canonical, [])
            if not raw_names:
                if logger_inst:
                    logger_inst.warning(
                        f"Unified chart ({setup}): ablation '{req_canonical}' not found "
                        f"in retrieval index — skipping panel."
                    )
                continue
            if resolved in panels_by_canonical:
                rendered.append(resolved)
                continue
            panel = _build_panel(req_canonical, resolved, raw_names, warn=True)
            if panel:
                panels_by_canonical[resolved] = panel
                rendered.append(resolved)

        # ── Everything else — dumped but not drawn, so the set of panels stays
        #    editable offline without re-running the bootstrap.
        for canonical, raw_names in seen_canonical.items():
            if canonical in panels_by_canonical:
                continue
            if _is_baseline_ablation(raw_names[0], canonical_key_map,
                                     baseline_canonical_names, yaml_aliases):
                continue
            panel = _build_panel(canonical, canonical, raw_names, warn=False)
            if panel:
                panels_by_canonical[canonical] = panel

        if not rendered:
            continue

        out_path = unified_dir / f"unified_{setup}{name_suffix}.png"
        unified_chart_data.dump(
            out_path.with_suffix(".json"),
            setup=setup,
            name_suffix=name_suffix,
            report_file=report_file,
            all_models=all_models_sorted,
            metric_labels=metric_labels,
            # Full label map, not the config-filtered one, so columns dropped by
            # `metrics:` can still be added back when re-charting.
            available_metrics=dict(_SETUP_METRIC_LABELS.get(setup, {})),
            rendered=rendered,
            panels=panels_by_canonical,
        )

        result = _generate_unified_heatmap(
            setup=setup,
            panels=[panels_by_canonical[c] for c in rendered],
            metric_labels=metric_labels,
            all_models=all_models_sorted,
            out_path=out_path,
        )
        if result:
            generated.append(result)
            if logger_inst:
                try:
                    rel = result.relative_to(output_dir)
                except ValueError:
                    rel = result
                logger_inst.info(f"Unified chart ({setup}): {rel} (+ .json for recharting)")

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Merge artifact directories from multiple ablation runs into one report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        help="One or more ablation/sweep output directories to merge.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a YAML file listing directories to merge. "
            "Expected format:\n"
            "  dirs:\n"
            "    - output/ablation_sweeps/20260319_161416\n"
            "    - output/ablation_sweeps/20260319_170000\n"
            "  output_dir: output/ablation_sweeps/merged   # optional\n"
            "  report_name: my_report.md                   # optional\n"
            "  baseline: ai_researcher_base                # optional; default 'current'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the merged report into. "
             "Defaults to a timestamped folder next to the first input dir.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="merged_ablation_report.md",
        help="Filename for the merged report (default: merged_ablation_report.md).",
    )
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        default=True,
        help="Copy artifact directories into the output directory for Git tracking (default: True).",
    )
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help=(
            "Only (re)build the unified charts and their .json data files. Skips "
            "artifact copying, report generation, filtered/turn recomputation, "
            "per-ablation reports, and overview heatmaps. Reuses the "
            "filtered_accuracy_report.txt files a previous full run left in the "
            "artifact dirs, so the filtered variant is produced too when they exist."
        ),
    )
    args = parser.parse_args()
    if args.charts_only:
        args.copy_artifacts = False

    # Load YAML config and merge with CLI args (CLI takes precedence)
    cfg_dirs: list[str] = []
    skip_ablations: set[str] = set()
    exclude_models: set[str] = set()
    include_tracks: set[str] = set()
    metrics_filter: dict[str, list[str]] = {}
    unified_chart: list[str] = []
    unified_chart_titles: dict[str, str] = {}  # canonical_name -> display title override
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        cfg_dirs = [str(p) for p in cfg.get("dirs", [])]
        if not args.output_dir and cfg.get("output_dir"):
            args.output_dir = str(cfg["output_dir"])
        if args.report_name == "merged_ablation_report.md" and cfg.get("report_name"):
            args.report_name = str(cfg["report_name"])
        skip_ablations = {str(s) for s in cfg.get("skip_ablations", [])}
        exclude_models = {str(s) for s in cfg.get("exclude_models", [])}
        include_tracks = {str(s) for s in cfg.get("tracks", [])}
        metrics_filter = {k: list(v) for k, v in cfg.get("metrics", {}).items()}
        for _item in cfg.get("unified_chart", []):
            if isinstance(_item, dict):
                _name = str(_item.get("name", "")).strip()
                if _name:
                    unified_chart.append(_name)
                    if "title" in _item:
                        unified_chart_titles[_name] = str(_item["title"])
            elif _item is not None:
                unified_chart.append(str(_item))
        if cfg.get("baseline"):
            global _BASELINE_OVERRIDE
            _BASELINE_OVERRIDE = str(cfg["baseline"]).strip()

    all_dirs = cfg_dirs + list(args.dirs)
    if not all_dirs:
        parser.error("Provide directories via positional args or --config.")

    # Resolve output directory
    first_dir = Path(all_dirs[0]).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else first_dir.parent / f"merged_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = output_dir / "artifacts"
    if args.copy_artifacts:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(output_dir), console_level="INFO")

    if _BASELINE_OVERRIDE:
        logger.info(
            f"Baseline override active: comparing ablations against "
            f"'{_BASELINE_OVERRIDE}' instead of the default 'current'."
        )

    # Build effective metric label dicts, applying the user-specified metric filter.
    # Keys not listed in metrics_filter are kept; setups absent from metrics_filter keep
    # all their default metrics.
    effective_metric_labels: dict[str, dict[str, str]] = {}
    for _setup, _all_labels in _SETUP_METRIC_LABELS.items():
        _include = metrics_filter.get(_setup)
        if _include:
            effective_metric_labels[_setup] = {k: v for k, v in _all_labels.items() if k in _include}
            logger.info(
                f"Metrics filter ({_setup}): keeping {list(effective_metric_labels[_setup])} "
                f"(dropped {[k for k in _all_labels if k not in _include]})"
            )
        else:
            effective_metric_labels[_setup] = dict(_all_labels)

    # Load descriptions from ablations.yaml — single source of truth.
    yaml_descriptions = _load_descriptions_from_yaml()
    # Build legacy-name alias map for baseline resolution.
    yaml_aliases = _load_legacy_name_aliases()

    # Collect artifact dirs and map them by mode and model
    all_artifact_dirs: list[str] = []
    # category -> model -> list of (ablation_description, og_path, tracked_path)
    category_mapping: dict[str, dict[str, list[tuple[str, str, str]]]] = {
        "pointwise": defaultdict(list),
        "pairwise": defaultdict(list),
        "ranking": defaultdict(list)
    }
    # (mode, ablation_name, model) -> most-recent artifact_dir path
    # Used to find retrieval vs current pairs after collection.
    retrieval_index: dict[tuple[str, str, str], str] = {}

    def _get_yaml_description(name: str) -> str | None:
        """Helper to find the description in YAML using various name variants."""
        if not name:
            return None
        # 1. Exact match
        if name in yaml_descriptions:
            return yaml_descriptions[name]
        # 2. Strip model suffix (e.g., "current-gpt-4o" -> "current")
        prefix = name.split("-")[0]
        if prefix in yaml_descriptions:
            return yaml_descriptions[prefix]
        # 3. Strip track prefix (e.g., "pairwise_current" -> "current")
        if "_" in prefix:
            short = prefix.split("_", 1)[1]
            if short in yaml_descriptions:
                return yaml_descriptions[short]
        return None

    copy_tasks = []
    for raw_path in all_dirs:
        run_dir = Path(raw_path).resolve()
        if not run_dir.is_dir():
            logger.warning(f"Skipping non-existent directory: {run_dir}")
            continue
        dirs = _collect_artifact_dirs(run_dir)
        logger.info(f"{run_dir.name}: found {len(dirs)} artifact dir(s)")
        all_artifact_dirs.extend(dirs)

        # Map per-artifact model and ablation
        ablation_rows = _read_ablation_rows(run_dir, dirs)
        for row in ablation_rows:
            # Try to resolve description from YAML using ablation name,
            # then source dir name as fallback for lookup key.
            ablation_name = row.get("ablation")
            description = _get_yaml_description(ablation_name)

            if not description:
                description = _get_yaml_description(run_dir.name)

            if not description:
                # Absolute fallback: whatever was stored at run time
                description = row.get("description", ablation_name or run_dir.name)

            for d_path in row.get("artifact_dirs", []):
                d_path_abs = Path(d_path).resolve()
                model = _extract_model_from_artifact_dir(d_path_abs)
                mode = _extract_mode_from_artifact_dir(d_path_abs)

                tracked_path = ""
                if args.copy_artifacts:
                    # Create a unique name for the copied directory using the parent experiment's name
                    unique_name = f"{d_path_abs.parent.parent.name}_{d_path_abs.name}"
                    dest_path = artifacts_dir / unique_name
                    if not dest_path.exists():
                        copy_tasks.append((d_path_abs, dest_path))
                    tracked_path = str(dest_path)

                category_mapping[mode][model].append((description, str(d_path_abs), tracked_path))

                # Index by (mode, ablation_name, model) for retrieval pair matching.
                if ablation_name:
                    retrieval_index[(mode, ablation_name, model)] = str(d_path_abs)

    if copy_tasks:
        logger.info(f"Copying {len(copy_tasks)} artifact directories in parallel...")
        def do_copy(task):
            src, dst = task
            shutil.copytree(src, dst)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(do_copy, task) for task in copy_tasks]
            list(tqdm(as_completed(futures), total=len(futures), desc="Copying artifacts"))

    if skip_ablations:
        def _bare_name(abl: str) -> str:
            """Strip track prefix (pairwise_, pointwise_, ranking_) for matching."""
            for prefix in ("pairwise_", "pointwise_", "ranking_"):
                if abl.startswith(prefix):
                    return abl[len(prefix):]
            return abl

        before = len(retrieval_index)
        retrieval_index = {
            k: v for k, v in retrieval_index.items()
            if k[1] not in skip_ablations and _bare_name(k[1]) not in skip_ablations
        }
        removed = before - len(retrieval_index)
        if removed:
            logger.info(f"Skipped {removed} retrieval index entries for ablations: {sorted(skip_ablations)}")

    if exclude_models:
        before = len(retrieval_index)
        retrieval_index = {
            k: v for k, v in retrieval_index.items()
            if k[2] not in exclude_models
        }
        removed = before - len(retrieval_index)
        if removed:
            logger.info(f"Skipped {removed} retrieval index entries for excluded models: {sorted(exclude_models)}")

        # Also prune all_artifact_dirs (used by generate_report.py subprocess and
        # filtered/turn report helpers which don't go through retrieval_index).
        before_dirs = len(all_artifact_dirs)
        all_artifact_dirs = [
            d for d in all_artifact_dirs
            if _extract_model_from_artifact_dir(Path(d).resolve()) not in exclude_models
        ]
        removed_dirs = before_dirs - len(all_artifact_dirs)
        if removed_dirs:
            logger.info(f"Removed {removed_dirs} artifact directories for excluded models.")

        # Prune category_mapping so the summary table reflects the exclusion.
        for _mode in list(category_mapping.keys()):
            for _model in list(category_mapping[_mode].keys()):
                if _model in exclude_models:
                    del category_mapping[_mode][_model]

    if include_tracks:
        # Keep only artifacts whose ablation belongs to an allowed track (the
        # ablation-name prefix, e.g. 'pairwise-vanilla-ai' from
        # 'pairwise-vanilla-ai_ai_researcher_base'). Used to scope a merge to one
        # data-type when a single run dir holds several tracks (e.g. hvh + vanilla-ai),
        # which otherwise collide under their shared canonical name.
        def _track_of(abl: str) -> str:
            return abl.split("_", 1)[0]

        # Resolved artifact paths in an allowed track — computed before pruning
        # retrieval_index so we can also prune all_artifact_dirs and
        # category_mapping, which are keyed by path rather than ablation name.
        allowed_paths = {
            v for (m, abl, model), v in retrieval_index.items()
            if _track_of(abl) in include_tracks
        }

        before = len(retrieval_index)
        retrieval_index = {
            k: v for k, v in retrieval_index.items()
            if _track_of(k[1]) in include_tracks
        }
        removed = before - len(retrieval_index)
        if removed:
            logger.info(
                f"Kept only tracks {sorted(include_tracks)}: dropped {removed} "
                f"retrieval index entries from other tracks."
            )

        before_dirs = len(all_artifact_dirs)
        all_artifact_dirs = [
            d for d in all_artifact_dirs
            if str(Path(d).resolve()) in allowed_paths
        ]
        removed_dirs = before_dirs - len(all_artifact_dirs)
        if removed_dirs:
            logger.info(
                f"Removed {removed_dirs} artifact directories outside tracks {sorted(include_tracks)}."
            )

        # Prune category_mapping so the summary table reflects the track filter.
        # Each entry is (description, og_path, tracked_path); og_path is the
        # resolved artifact path, matching allowed_paths.
        for _mode in list(category_mapping.keys()):
            for _model in list(category_mapping[_mode].keys()):
                kept = [e for e in category_mapping[_mode][_model] if e[1] in allowed_paths]
                if kept:
                    category_mapping[_mode][_model] = kept
                else:
                    del category_mapping[_mode][_model]

    if not all_artifact_dirs:
        logger.error("No artifact directories found. Nothing to merge.")
        sys.exit(1)

    logger.info(f"Total artifact directories to merge: {len(all_artifact_dirs)}")

    # -----------------------------------------------------------------------
    # Charts-only fast path — rebuild just the unified charts and their .json
    # data files, skipping every other (slow) stage of the merge.
    # -----------------------------------------------------------------------
    if args.charts_only:
        if not unified_chart:
            logger.error("--charts-only needs a `unified_chart:` list in the config. Nothing to do.")
            sys.exit(1)

        # A previous full run wrote filtered_accuracy_report.txt into the source
        # artifact dirs; reuse those instead of recomputing the filtered metrics.
        has_filtered = any(
            (Path(d) / "filtered_accuracy_report.txt").exists() for d in all_artifact_dirs
        )
        variants = [("accuracy_report.txt", "")]
        if has_filtered:
            variants.append(("filtered_accuracy_report.txt", "_filtered"))
        else:
            logger.warning(
                "No filtered_accuracy_report.txt found in the artifact dirs — only the "
                "unfiltered variant will be built. Run the full merge once to create them."
            )

        for variant_report_file, variant_suffix in variants:
            paths = _generate_unified_charts(
                retrieval_index=retrieval_index,
                yaml_descriptions=yaml_descriptions,
                yaml_aliases=yaml_aliases,
                unified_ablations=unified_chart,
                output_dir=output_dir,
                metric_labels_override=effective_metric_labels,
                unified_chart_titles=unified_chart_titles,
                report_file=variant_report_file,
                name_suffix=variant_suffix,
                logger_inst=logger,
            )
            if paths:
                label = variant_suffix.lstrip("_") or "unfiltered"
                print(f"\nUnified charts ({label}) written to: {output_dir / 'unified_charts'}")

        print(
            "\nRe-render them in a different format without recomputation:\n"
            f"  python {_RECHART_SCRIPT} {output_dir / 'unified_charts'}"
        )
        return

    # Write a temporary dirs config for generate_report.py
    tmp_dirs_config = output_dir / "_tmp_dirs_config.yaml"
    with open(tmp_dirs_config, "w") as f:
        yaml.dump({"dirs": all_artifact_dirs}, f)

    report_path = (output_dir / args.report_name).resolve()
    cmd = [
        sys.executable,
        str(_GENERATE_REPORT_SCRIPT),
        "--config", str(tmp_dirs_config),
        "--output", str(report_path),
    ]
    logger.info(f"Generating report: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=os.environ.copy(), cwd=str(_PROJECT_ROOT))

    if result.returncode != 0:
        logger.error("Report generation failed.")
        if tmp_dirs_config.exists():
            tmp_dirs_config.unlink()
        sys.exit(1)

    logger.info(f"Merged report written to: {report_path}")

    # -----------------------------------------------------------------------
    # Filtered report — recompute metrics excluding blocklisted paper abstracts
    # -----------------------------------------------------------------------
    blocked_titles = _load_paper_blocklist()
    if blocked_titles:
        logger.info(f"Paper blocklist loaded: {len(blocked_titles)} title(s) will be excluded from filtered metrics.")
    
    filtered_info = _generate_filtered_reports(all_artifact_dirs, blocked_titles, logger)
    had_filtered = bool(filtered_info)

    if had_filtered:
        # Generate the specific blocklist filtering report
        blocklist_report_path = _generate_blocklist_report(filtered_info, output_dir, logger)
        if blocklist_report_path:
            print(f"\nBlocklist filtering report written to: {blocklist_report_path}")

        filtered_report_path = output_dir / (
            Path(args.report_name).stem + "_filtered" + Path(args.report_name).suffix
        )
        tmp_dirs_config.write_text(yaml.dump({"dirs": all_artifact_dirs}))
        cmd_filtered = [
            sys.executable,
            str(_GENERATE_REPORT_SCRIPT),
            "--config", str(tmp_dirs_config),
            "--report-file", "filtered_accuracy_report.txt",
            "--output", str(filtered_report_path),
        ]
        logger.info(f"Generating merged filtered report: {' '.join(cmd_filtered)}")
        subprocess.run(cmd_filtered, env=os.environ.copy(), cwd=str(_PROJECT_ROOT))
        print(f"Merged filtered report written to: {filtered_report_path}")
    elif blocked_titles:
        logger.info("No artifact dirs matched any blocklisted titles — filtered report skipped.")

    # -----------------------------------------------------------------------
    # Test Sets Consistency Report
    # -----------------------------------------------------------------------
    _generate_test_sets_consistency_report(retrieval_index, output_dir, blocked_titles, logger)

    # -----------------------------------------------------------------------
    # Token usage report
    # Per-ablation/model summary of input, output, and reasoning tokens
    # derived from batch output JSONL files.
    # -----------------------------------------------------------------------
    token_report_path = _generate_token_usage_report(
        retrieval_index=retrieval_index,
        yaml_descriptions=yaml_descriptions,
        output_dir=output_dir,
        logger_inst=logger,
    )
    if token_report_path:
        print(f"\nToken usage report written to: {token_report_path}")

    # -----------------------------------------------------------------------
    # Retrieval impact comparison reports
    # For each (mode, model) where we have both a *_current and *_retrieval
    # artifact, run compare_retrieval_impact.py in parallel to produce logs.
    # -----------------------------------------------------------------------
    _compare_retrieval_script = _THIS_DIR / "compare_retrieval_impact.py"
    retrieval_impact_dir = output_dir / "retrieval_impact"
    
    # Collect all tasks to run
    impact_tasks = []

    # Find all retrieval ablation keys in the index
    for (mode, abl_name, model), ret_dir in retrieval_index.items():
        if "retrieval" not in abl_name:
            continue
        # Derive the matching baseline name: replace 'retrieval' with 'current'
        baseline_name = abl_name.replace("retrieval", "current")
        baseline_dir = retrieval_index.get((mode, baseline_name, model))
        if not baseline_dir:
            # Try legacy baseline names
            track_prefix = abl_name.split("_")[0] if "_" in abl_name else ""
            candidates: list[str] = []
            if track_prefix:
                candidates += [f"{track_prefix}_current", f"{track_prefix}_all"]
            candidates += ["current", "all"]
            
            seen_cands: set[str] = set()
            expanded: list[str] = []
            for c in candidates:
                if c not in seen_cands:
                    seen_cands.add(c)
                    expanded.append(c)
                for alias in yaml_aliases.get(c, []):
                    if alias not in seen_cands:
                        seen_cands.add(alias)
                        expanded.append(alias)
            for candidate in expanded:
                baseline_dir = retrieval_index.get((mode, candidate, model))
                if baseline_dir:
                    break
        
        if baseline_dir:
            out_file = retrieval_impact_dir / mode / f"{mode}_{abl_name.replace('_retrieval', '')}_{model}_retrieval_impact.md"
            if not out_file.exists():
                impact_tasks.append((mode, abl_name, model, baseline_dir, ret_dir, out_file))

    if impact_tasks:
        logger.info(f"Generating {len(impact_tasks)} retrieval impact reports in parallel...")
        def run_impact(task):
            mode, abl_name, model, base_dir, ret_dir, out_f = task
            out_f.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(_compare_retrieval_script),
                "--baseline-dir", base_dir,
                "--retrieval-dir", ret_dir,
                "--output", str(out_f),
                "--baseline-label", "baseline",
                "--retrieval-label", abl_name,
            ]
            return subprocess.run(cmd, env=os.environ.copy(), cwd=str(_PROJECT_ROOT))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_impact, task) for task in impact_tasks]
            list(tqdm(as_completed(futures), total=len(futures), desc="Retrieval impact reports"))
        logger.info(f"Retrieval impact reports written to: {retrieval_impact_dir}")

    # Write a Markdown mapping of models, ablations, and paths
    try:
        rel_report_path = report_path.relative_to(_PROJECT_ROOT)
    except ValueError:
        rel_report_path = report_path

    md_lines = [
        "# Merged Ablation Artifact Mapping\n",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Merged Report:** `{rel_report_path}`\n"
    ]

    for mode in ["pointwise", "pairwise", "ranking"]:
        models_dict = category_mapping[mode]
        if not models_dict:
            continue

        md_lines.append(f"# {mode.capitalize()}\n")

        for model in sorted(models_dict.keys()):
            md_lines.append(f"## {model}\n")
            md_lines.append("| Ablation Description | Original Path | Git-Tracked Path |")
            md_lines.append("| --- | --- | --- |")

            # Sort items: baseline (current) first, then alphabetical
            items = models_dict[model]
            items.sort(key=lambda x: (0 if "current" in x[0].lower() else 1, x[0]))

            for desc, og_path, tracked_path in items:
                og_path_obj = Path(og_path).resolve()
                try:
                    rel_og_path = og_path_obj.relative_to(_PROJECT_ROOT)
                except ValueError:
                    rel_og_path = og_path_obj

                rel_tracked_path = "N/A"
                if tracked_path:
                    tracked_path_obj = Path(tracked_path).resolve()
                    try:
                        rel_tracked_path = tracked_path_obj.relative_to(_PROJECT_ROOT)
                    except ValueError:
                        rel_tracked_path = tracked_path_obj

                md_lines.append(f"| {desc} | `{rel_og_path}` | `{rel_tracked_path}` |")
            md_lines.append("")

    summary_md_path = output_dir / "merged_ablation_summary.md"
    with open(summary_md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nSummary mapping written to: {summary_md_path}")
    logger.info(f"Summary mapping written to: {summary_md_path}")

    # -----------------------------------------------------------------------
    # Per-ablation effect reports
    # One .md file per ablation showing before/after description and
    # per-setup comparison tables + delta heatmaps.
    # -----------------------------------------------------------------------
    before_descriptions = _load_before_descriptions_from_yaml()
    per_ablation_paths = _generate_all_per_ablation_reports(
        retrieval_index=retrieval_index,
        output_dir=output_dir,
        yaml_descriptions=yaml_descriptions,
        before_descriptions=before_descriptions,
        yaml_aliases=yaml_aliases,
        logger_inst=logger,
        metric_labels_override=effective_metric_labels,
    )
    if per_ablation_paths:
        logger.info(f"Per-ablation reports written to: {output_dir / 'per_ablation_reports'}")

    # -----------------------------------------------------------------------
    # Ablation overview heatmaps — one PNG per judge model showing deltas
    # across all ablations for both pointwise and pairwise setups.
    # -----------------------------------------------------------------------
    overview_figures = _generate_ablation_overview_figures(
        retrieval_index=retrieval_index,
        yaml_descriptions=yaml_descriptions,
        yaml_aliases=yaml_aliases,
        output_dir=output_dir,
        logger_inst=logger,
        metric_labels_override=effective_metric_labels,
    )
    if overview_figures:
        print(f"\nAblation overview heatmaps written to: {output_dir / 'overview_heatmaps'}")

    # -----------------------------------------------------------------------
    # Unified side-by-side charts (one PNG per setup)
    # -----------------------------------------------------------------------
    if unified_chart:
        unified_paths = _generate_unified_charts(
            retrieval_index=retrieval_index,
            yaml_descriptions=yaml_descriptions,
            yaml_aliases=yaml_aliases,
            unified_ablations=unified_chart,
            output_dir=output_dir,
            metric_labels_override=effective_metric_labels,
            unified_chart_titles=unified_chart_titles,
            logger_inst=logger,
        )
        if unified_paths:
            print(f"\nUnified charts written to: {output_dir / 'unified_charts'}")

    if had_filtered:
        filtered_per_ablation_paths = _generate_all_per_ablation_reports(
            retrieval_index=retrieval_index,
            output_dir=output_dir,
            yaml_descriptions=yaml_descriptions,
            before_descriptions=before_descriptions,
            yaml_aliases=yaml_aliases,
            logger_inst=logger,
            report_file="filtered_accuracy_report.txt",
            name_suffix="_filtered",
            metric_labels_override=effective_metric_labels,
        )
        if filtered_per_ablation_paths:
            logger.info(f"Filtered per-ablation reports written to: {output_dir / 'per_ablation_reports'}")

        filtered_overview_figures = _generate_ablation_overview_figures(
            retrieval_index=retrieval_index,
            yaml_descriptions=yaml_descriptions,
            yaml_aliases=yaml_aliases,
            output_dir=output_dir,
            logger_inst=logger,
            report_file="filtered_accuracy_report.txt",
            name_suffix="_filtered",
            metric_labels_override=effective_metric_labels,
        )
        if filtered_overview_figures:
            print(f"\nFiltered ablation overview heatmaps written to: {output_dir / 'overview_heatmaps'}")

        if unified_chart:
            filtered_unified_paths = _generate_unified_charts(
                retrieval_index=retrieval_index,
                yaml_descriptions=yaml_descriptions,
                yaml_aliases=yaml_aliases,
                unified_ablations=unified_chart,
                output_dir=output_dir,
                metric_labels_override=effective_metric_labels,
                unified_chart_titles=unified_chart_titles,
                report_file="filtered_accuracy_report.txt",
                name_suffix="_filtered",
                logger_inst=logger,
            )
            if filtered_unified_paths:
                print(f"\nFiltered unified charts written to: {output_dir / 'unified_charts'}")

        # -----------------------------------------------------------------------
        # Bootstrap debug CSVs — filtered only
        # One CSV per (ablation, setup, model) with per-example scores matching
        # the paired bootstrap inputs: pid | gt | base_pred | abl_pred | metric deltas
        # -----------------------------------------------------------------------
        debug_csvs_dir = _generate_bootstrap_debug_csvs(
            retrieval_index=retrieval_index,
            yaml_aliases=yaml_aliases,
            output_dir=output_dir,
            logger_inst=logger,
            report_file="filtered_accuracy_report.txt",
        )
        if debug_csvs_dir:
            logger.info(f"Bootstrap debug CSVs written to: {debug_csvs_dir.relative_to(output_dir)}")

    # -----------------------------------------------------------------------
    # Turn-number filtered reports (turn1 / turn2)
    # Exclude instances where the generated NEGATIVE has turn_number == N.
    # Mirrors the filtered (blocklist) pipeline: per-ablation reports,
    # overview heatmaps, and bootstrap debug CSVs per turn subset.
    # -----------------------------------------------------------------------
    has_turn_numbers = _any_instances_have_turn_numbers(all_artifact_dirs)
    if has_turn_numbers:
        for turn_n in (1, 2):
            turn_report_file = f"turn{turn_n}_accuracy_report.txt"
            name_suffix_t = f"_turn{turn_n}"
            had_turn = _generate_turn_reports(all_artifact_dirs, turn_n, logger)
            if had_turn:
                turn_merged_path = output_dir / (
                    Path(args.report_name).stem + name_suffix_t + Path(args.report_name).suffix
                )
                tmp_dirs_config.write_text(yaml.dump({"dirs": all_artifact_dirs}))
                cmd_turn = [
                    sys.executable,
                    str(_GENERATE_REPORT_SCRIPT),
                    "--config", str(tmp_dirs_config),
                    "--report-file", turn_report_file,
                    "--output", str(turn_merged_path),
                ]
                logger.info(f"Generating merged turn{turn_n} report: {' '.join(cmd_turn)}")
                subprocess.run(cmd_turn, env=os.environ.copy(), cwd=str(_PROJECT_ROOT))
                print(f"Merged turn{turn_n} report written to: {turn_merged_path}")

                turn_per_ablation_paths = _generate_all_per_ablation_reports(
                    retrieval_index=retrieval_index,
                    output_dir=output_dir,
                    yaml_descriptions=yaml_descriptions,
                    before_descriptions=before_descriptions,
                    yaml_aliases=yaml_aliases,
                    logger_inst=logger,
                    report_file=turn_report_file,
                    name_suffix=name_suffix_t,
                    metric_labels_override=effective_metric_labels,
                )
                if turn_per_ablation_paths:
                    logger.info(
                        f"Turn{turn_n} per-ablation reports written to: "
                        f"{output_dir / 'per_ablation_reports'}"
                    )

                turn_overview_figures = _generate_ablation_overview_figures(
                    retrieval_index=retrieval_index,
                    yaml_descriptions=yaml_descriptions,
                    yaml_aliases=yaml_aliases,
                    output_dir=output_dir,
                    logger_inst=logger,
                    report_file=turn_report_file,
                    name_suffix=name_suffix_t,
                    metric_labels_override=effective_metric_labels,
                )
                if turn_overview_figures:
                    print(
                        f"\nTurn{turn_n} ablation overview heatmaps written to: "
                        f"{output_dir / 'overview_heatmaps'}"
                    )

                turn_debug_csvs_dir = _generate_bootstrap_debug_csvs(
                    retrieval_index=retrieval_index,
                    yaml_aliases=yaml_aliases,
                    output_dir=output_dir,
                    logger_inst=logger,
                    report_file=turn_report_file,
                )
                if turn_debug_csvs_dir:
                    logger.info(
                        f"Turn{turn_n} bootstrap debug CSVs written to: "
                        f"{turn_debug_csvs_dir.relative_to(output_dir)}"
                    )
            else:
                logger.info(
                    f"No turn{turn_n} negatives found — turn{turn_n} reports skipped."
                )

    if tmp_dirs_config.exists():
        tmp_dirs_config.unlink()

    # -----------------------------------------------------------------------
    # Safety refusal report
    # Shows which abstract indices were refused per data file across all runs.
    # -----------------------------------------------------------------------
    refusal_report_path = _generate_refusal_report(
        retrieval_index=retrieval_index,
        yaml_descriptions=yaml_descriptions,
        output_dir=output_dir,
        logger_inst=logger,
    )
    if refusal_report_path:
        print(f"\nSafety refusal report written to: {refusal_report_path}")

    if args.copy_artifacts:
        logger.info("\n" + "="*60)
        logger.info("IMPORTANT: Artifacts have been copied for Git tracking.")
        try:
            rel_output_dir = output_dir.relative_to(_PROJECT_ROOT)
        except ValueError:
            rel_output_dir = output_dir
        logger.info(f"To track them, run: git add -f {rel_output_dir}")
        logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
