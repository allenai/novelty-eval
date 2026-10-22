"""
generate_report.py — Ablation comparison report generator.

Aggregates accuracy_report.txt files produced by accuracy_test.py / run_ablations.py
across multiple experiment directories and renders a single Markdown summary suitable
for comparing ablation configurations side-by-side.

Inputs
------
One or more accuracy_test_artifacts directories, each containing:
  - accuracy_report.txt     (primary metrics: pairwise accuracy or ranking NDCG/MRR/Hits)
  - debug_accuracy_report.txt.md  (experiment settings: model, reasoning effort, etc.)

Outputs
-------
  <output>.md               Markdown table(s) summarising all experiments grouped by
                            model and track (pairwise / ranking).
  <output>_pairwise_heatmap_no_ties.png   (if matplotlib is available)
  <output>_pairwise_heatmap_with_ties.png (if matplotlib is available)

Usage
-----
  # From a YAML config listing directories (see merge_config.yaml for format):
  python generate_report.py --config ablation/configs/merge/merge_config.yaml --output report.md

  # Directly from a list of directories:
  python generate_report.py --dirs output/ablation_sweeps/run1 output/ablation_sweeps/run2 \\
                            --output report.md

Config YAML format
------------------
  dirs:
    - path/to/experiment_dir_1
    - path/to/experiment_dir_2
  output_dir: optional/output/directory
  report_name: optional_report_name.md
"""
import os
import json
import argparse
import statistics
import math
import re
import yaml
import sys
from collections import defaultdict
from pathlib import Path

# Add src to python path to import logging_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.logging_utils import setup_logger

logger = None

# The ablation registry stays with the ablation package; this module only reads it.
_ABLATIONS_FILE = Path(__file__).resolve().parent.parent / "ablation" / "ablations.yaml"

# Descriptions to exclude from the report. Add/remove entries to toggle.
_SKIP_DESCRIPTIONS: set[str] = {
    "Verbatim↔plan format",
}

def _load_ablation_descriptions() -> dict[str, str]:
    _ablations_file = _ABLATIONS_FILE
    with open(_ablations_file) as f:
        data = yaml.safe_load(f)
    result: dict[str, str] = {}
    for name, entry in data["ablations"].items():
        desc = entry["description"]
        result[name] = desc
        tracks = entry.get("tracks", [])
        if "track" in entry:
            tracks = list(tracks) + [entry["track"]]
        for legacy in entry.get("legacy_names", []):
            result[legacy] = desc
            for track in tracks:
                result[f"{track}_{legacy}"] = desc
    return result


def _build_ablation_order_map() -> dict[str, int]:
    """
    Build a canonical description→sort-index mapping from ablations.yaml.

    The order follows the YAML definition order (Python dicts preserve insertion
    order since 3.7).  Using this everywhere ensures tables and every heatmap
    show ablations in the same sequence.

    Special pins:
      - 'current' (baseline) is always index 0.
      - Descriptions not in the YAML get index 998 (before All-bad catch-all).
      - 'All-bad' ablations are detected by suffix and get index 999.
    """
    _ablations_file = _ABLATIONS_FILE
    with open(_ablations_file) as f:
        data = yaml.safe_load(f)
    order: dict[str, int] = {}
    for i, (name, entry) in enumerate(data.get("ablations", {}).items()):
        desc = entry.get("description", "")
        if desc and desc not in order:
            order[desc] = i
    return order


# Ablation descriptions keyed by the short ablation name (prefix of the config name).
# Edit src/novelty_eval/ablation/ablations.yaml to update these.
_ABLATION_DESCRIPTIONS: dict[str, str] = _load_ablation_descriptions()

# Canonical sort order for ablation descriptions, derived from ablations.yaml order.
# Used by tables and all heatmaps so ablations always appear in the same sequence.
_ABLATION_ORDER_MAP: dict[str, int] = _build_ablation_order_map()


def _load_ablation_groups() -> dict[str, str]:
    """Return {description: group_name} from ablations.yaml (keyed by raw description)."""
    _ablations_file = _ABLATIONS_FILE
    if not _ablations_file.exists():
        return {}
    with open(_ablations_file) as f:
        data = yaml.safe_load(f) or {}
    result: dict[str, str] = {}
    for name, entry in data.get("ablations", {}).items():
        group = entry.get("group", "")
        desc = entry.get("description", "")
        if desc:
            result[desc] = group
        for legacy in entry.get("legacy_names", []):
            result[legacy] = group
    return result


# Maps raw ablation description → group name (e.g. "baseline", "evaluation", "backbone", …)
_ABLATION_GROUPS: dict[str, str] = _load_ablation_groups()


def _group_separator_rows(ablations: list[str], groups: dict[str, str]) -> list[int]:
    """Return row indices where the ablation group changes (used to draw separator lines)."""
    separators = []
    prev_group = None
    for i, abl in enumerate(ablations):
        grp = groups.get(abl, "")
        if prev_group is not None and grp != prev_group:
            separators.append(i)
        prev_group = grp
    return separators


def _compute_group_spans(
    ablations: list[str], groups: dict[str, str]
) -> list[tuple[str, int, int]]:
    """Return [(group_name, row_start, row_end), ...] for each contiguous group region."""
    spans: list[tuple[str, int, int]] = []
    prev_grp: str | None = None
    span_start = 0
    for i, abl in enumerate(ablations):
        grp = groups.get(abl, "")
        if grp != prev_grp:
            if prev_grp is not None:
                spans.append((prev_grp, span_start, i - 1))
            span_start = i
            prev_grp = grp
    if prev_grp is not None:
        spans.append((prev_grp, span_start, len(ablations) - 1))
    return spans


# Colors and display names for the group label bar
_GROUP_COLORS: dict[str, str] = {
    "baseline":    "#4a5568",   # dark slate        — neutral anchor
    "evaluation":  "#c97b52",   # muted burnt sienna — warm, near the red endpoint
    "idea_format": "#52886b",   # muted sage green   — earthy, near the green endpoint
    "retrieval":   "#4a82a8",   # muted slate blue
    "backbone":    "#8a6b9a",   # muted dusty violet
    "combined":    "#8a7250",   # muted warm tan/ochre
}
_GROUP_DISPLAY_NAMES: dict[str, str] = {
    "baseline":   "Baseline",
    "evaluation": "Evaluation",
    "idea_format": "Idea Format",
    "retrieval":  "Retrieval",
    "backbone":   "Backbone",
    "combined":   "Combined",
}

def load_dirs_from_yaml(yaml_path):
    """Load directories list from a YAML config file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'dirs' not in config:
        raise ValueError("YAML config must contain a 'dirs' key with a list of directories")

    return config['dirs']

def _extract_section_metrics(section_text):
    """Extract mean metrics from a report section's text."""
    metrics = {}
    ndcg_match = re.search(r'Mean NDCG: ([\d\.]+)', section_text)
    mrr_match = re.search(r'Mean MRR: ([\d\.]+)', section_text)
    hits1_match = re.search(r'Mean Hits@1: ([\d\.]+)', section_text)
    hits2_match = re.search(r'Mean Hits@2: ([\d\.]+)', section_text)
    hits3_match = re.search(r'Mean Hits@3: ([\d\.]+)', section_text)
    pairwise_match = re.search(r'Mean LLM Pairwise Accuracy: ([\d\.]+)', section_text)
    if ndcg_match: metrics['ndcg'] = float(ndcg_match.group(1))
    if mrr_match: metrics['mrr'] = float(mrr_match.group(1))
    if hits1_match: metrics['hits_1'] = float(hits1_match.group(1))
    if hits2_match: metrics['hits_2'] = float(hits2_match.group(1))
    if hits3_match: metrics['hits_3'] = float(hits3_match.group(1))
    if pairwise_match: metrics['pairwise_accuracy'] = float(pairwise_match.group(1))
    return metrics


def extract_metrics_from_report(report_path):
    """Return a dict keyed by mode name ('pairwise', 'swiss', 'bi-swiss', …) with metric dicts."""
    abs_report_path = os.path.abspath(report_path)
    if not os.path.exists(report_path):
        logger.warning(f"Report file not found at {report_path}")
        logger.warning(f"Absolute path: {abs_report_path}")
        return None

    with open(report_path, 'r') as f:
        content = f.read()

    results = {}

    # Detect test mode from report header
    test_mode_match = re.search(r'Test Mode:\s*(\w+)', content)
    test_mode = test_mode_match.group(1).lower() if test_mode_match else 'ranking'

    # Extract support (Number of Test Instances Processed) if available at the top
    top_support_match = re.search(r'Number of Test Instances Processed: (\d+)', content)
    top_support = float(top_support_match.group(1)) if top_support_match else None

    if test_mode == 'pointwise':
        # Extract pointwise classification metrics from the "Pointwise" section
        pw_section = re.search(r'Pointwise\n-+\n(.*?)(?=\nSummary|\Z)', content, re.DOTALL)
        if pw_section:
            section_text = pw_section.group(1)
            metrics = {}
            for key, pattern in [
                ('accuracy',         r'Mean Accuracy:\s+([\d\.]+)'),
                ('f1_macro',         r'Mean F1 \(macro\):\s+([\d\.]+)'),
                ('f1_pos',           r'Mean F1\s+\(POSITIVE\):\s+([\d\.]+)'),
                ('f1_neg',           r'Mean F1\s+\(NEGATIVE\):\s+([\d\.]+)'),
                ('precision_pos',    r'Mean Precision \(POSITIVE\):\s+([\d\.]+)'),
                ('precision_neg',    r'Mean Precision \(NEGATIVE\):\s+([\d\.]+)'),
                ('recall_pos',       r'Mean Recall\s+\(POSITIVE\):\s+([\d\.]+)'),
                ('recall_neg',       r'Mean Recall\s+\(NEGATIVE\):\s+([\d\.]+)'),
            ]:
                m = re.search(pattern, section_text)
                if m:
                    metrics[key] = float(m.group(1))
            if metrics:
                # Fall back to the top-level processed instance count for pointwise support
                metrics['support'] = top_support
                results['pointwise'] = metrics
        else:
            logger.warning(f"'Pointwise' section not found in {report_path}")
    elif test_mode == 'pairwise':
        # Extract pairwise metrics from the "Pairwise" section
        pairwise_section = re.search(r'Pairwise\n-+\n(.*?)(?=\nSummary|\Z)', content, re.DOTALL)
        if pairwise_section:
            section_text = pairwise_section.group(1)
            metrics = {}
            acc_with_ties = re.search(r'Mean LLM Pairwise Accuracy \(with ties\):\s+([\d\.]+)(?:\s+\(support=([\d\.]+)\))?', section_text)
            acc_strict = re.search(r'Mean LLM Pairwise Accuracy \(strict\):\s+([\d\.]+)', section_text)
            acc_no_ties = re.search(r'Mean LLM Pairwise Accuracy \(without ties\):\s+([\d\.]+)(?:\s+\(support=([\d\.]+)\))?', section_text)
            n_ties = re.search(r'Mean Number of Ties: ([\d\.]+)', section_text)
            if acc_with_ties:
                metrics['pairwise_accuracy'] = float(acc_with_ties.group(1))
                if acc_with_ties.group(2):
                    metrics['support'] = float(acc_with_ties.group(2))
            if acc_strict:
                metrics['pairwise_accuracy_strict'] = float(acc_strict.group(1))
            if acc_no_ties:
                metrics['pairwise_accuracy_no_ties'] = float(acc_no_ties.group(1))
                if acc_no_ties.group(2):
                    metrics['support_without_ties'] = float(acc_no_ties.group(2))
            if n_ties:
                metrics['n_ties'] = float(n_ties.group(1))
            
            # If pairwise-specific support wasn't found, fall back to top-level count
            if 'support' not in metrics and top_support is not None:
                metrics['support'] = top_support

            if metrics:
                results['pairwise'] = metrics
        else:
            logger.warning(f"'Pairwise' section not found in {report_path}")
    else:
        # Extract metrics for Swiss Tournament
        swiss_section = re.search(r'3\. Swiss Tournament\n-+\n(.*?)(?=\n\d+\.|\nSummary|\Z)', content, re.DOTALL)
        if swiss_section:
            metrics = _extract_section_metrics(swiss_section.group(1))
            if metrics:
                if top_support is not None:
                    metrics['support'] = top_support
                results['swiss'] = metrics
        else:
            logger.warning(f"'3. Swiss Tournament' section not found in {report_path}")

        # Extract metrics for Bi-Swiss Tournament
        bi_swiss_section = re.search(r'4\. Bi-Swiss Tournament\n-+\n(.*?)(?=\n\d+\.|\nSummary|\Z)', content, re.DOTALL)
        if bi_swiss_section:
            metrics = _extract_section_metrics(bi_swiss_section.group(1))
            if metrics:
                if top_support is not None:
                    metrics['support'] = top_support
                results['bi-swiss'] = metrics
        else:
            logger.debug(f"'4. Bi-Swiss Tournament' section not found in {report_path} (mode may not have been run)")


    if not results:
        logger.warning(f"No metrics extracted from {report_path}")
        return None

    return results

def extract_settings(base_dir):
    # Look for debug_accuracy_report.txt.md or similar
    settings = {'model': 'Unknown', 'retrieval': 'Unknown', 'reasoning_effort': 'Unknown'}
    
    debug_file = None
    if os.path.exists(base_dir):
        for f in os.listdir(base_dir):
            if f.startswith("debug_") and f.endswith(".md"):
                debug_file = os.path.join(base_dir, f)
                break
    
    if debug_file:
        try:
            with open(debug_file, 'r') as f:
                content = f.read()
                # Find the JSON block under "Command Line Arguments"
                match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    try:
                        args_dict = json.loads(json_str)
                        settings['model'] = args_dict.get('llm_engine', 'Unknown')
                        settings['retrieval'] = str(args_dict.get('retrieve_related_work', 'Unknown'))
                        settings['reasoning_effort'] = args_dict.get('reasoning_effort', 'none')
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON in {debug_file}")
                        pass
                else:
                    logger.warning(f"JSON block not found in {debug_file}")
        except Exception as e:
            logger.error(f"Error reading settings from {debug_file}: {e}")
    else:
        logger.warning(f"Debug file directory not found: {base_dir}")
            
    return settings

def extract_data_dir(base_dir):
    report_path = os.path.join(base_dir, "accuracy_report.txt")
    if not os.path.exists(report_path):
        return "Unknown"
        
    with open(report_path, 'r') as f:
        content = f.read()
        
    match = re.search(r'Test Instances Path: (.*)', content)
    if match:
        full_path = match.group(1).strip()
        # Extract the directory name containing the instances file
        # e.g., benchmark_data/benchmark_instances/20260218_121559/benchmark_instances.yaml -> 20260218_121559
        parts = full_path.split('/')
        if len(parts) >= 2:
            return parts[-2]
        return os.path.dirname(full_path)
    return "Unknown"

def extract_ablation_description(base_dir):
    """
    Infer ablation description from the artifact path.

    For ablation runs the path looks like:
      .../ablation_sweeps/<ts>/<ablation>/<ablation>-<model>/accuracy_test_artifacts/<ts>

    New-style (track-namespaced) runs look like:
      .../ablation_sweeps/<ts>/<track>_<ablation>/<track>_<ablation>-<model>/accuracy_test_artifacts/<ts>

    For regular sweep runs the path looks like:
      .../accuracy_sweep/<ts>/<config-name>/accuracy_test_artifacts/<ts>

    Strategy:
      1. Walk path parts looking for a known ablation-name prefix → return its description.
         Also strips a leading track prefix (e.g. "pairwise_current" → "current") to support
         the new {track}_{name} naming convention from ablations.yaml.
      2. Fall back to sweep_summary.json description if present.
      3. Return the raw prefix so old/unknown runs show a meaningful label instead of "Current".
    """
    from pathlib import Path
    norm = Path(os.path.normpath(base_dir))
    parts = norm.parts

    # Pass 0: metadata file written by run_ablations.py.
    # Use only the *ablation name* (key) stored there to look up the current
    # description from ablations.yaml.  The stored "description" field is
    # intentionally ignored — it was baked in at run time and may be stale.
    for parent in norm.parents:
        meta_path = parent / "_ablation_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as _f:
                    ablation_name = json.load(_f).get("ablation", "")
                if ablation_name:
                    if ablation_name in _ABLATION_DESCRIPTIONS:
                        return _ABLATION_DESCRIPTIONS[ablation_name]
                    # Handle track-expanded names: "pairwise_current" → look up "current"
                    underscore_idx = ablation_name.find("_")
                    if underscore_idx != -1:
                        short_name = ablation_name[underscore_idx + 1:]
                        if short_name in _ABLATION_DESCRIPTIONS:
                            return _ABLATION_DESCRIPTIONS[short_name]
            except Exception:
                pass
            break

    # Pass 1: known ablation name from prefix (direct or with track prefix stripped)
    for part in reversed(parts):
        prefix = part.split("-")[0]
        if prefix in _ABLATION_DESCRIPTIONS:
            return _ABLATION_DESCRIPTIONS[prefix]
        # New naming: "{track}_{ablation_name}-{model}" — strip the track prefix
        underscore_idx = prefix.find("_")
        if underscore_idx != -1:
            short_name = prefix[underscore_idx + 1:]
            if short_name in _ABLATION_DESCRIPTIONS:
                return _ABLATION_DESCRIPTIONS[short_name]

    # Pass 2: Check for sweep_summary.json in the directory containing accuracy_test_artifacts
    # The structure is usually: .../<config-name>/accuracy_test_artifacts/<ts>
    config_dir = norm.parent.parent
    sweep_summary_path = config_dir / "sweep_summary.json"
    if sweep_summary_path.exists():
        try:
            with open(sweep_summary_path) as f:
                data = json.load(f)
                if data.get("description"):
                    return data["description"]
        except Exception:
            pass

    # Pass 3: unknown ablation — return the raw config-dir prefix as a readable label
    # (better than silently collapsing everything to "Current")
    config_dir_name = norm.parent.parent.name  # e.g. "c10_gpt_backbone-gpt-5.4"
    return config_dir_name.split("-")[0] if config_dir_name else _ABLATION_DESCRIPTIONS.get("current", "Current")


def process_directory(base_dir, report_file="accuracy_report.txt"):
    """Return a list of stat dicts (one per mode) for a given artifact directory."""
    logger.info(f"Processing directory: {base_dir}")
    if not os.path.exists(base_dir):
        logger.error(f"Directory does not exist: {base_dir}")
        logger.error(f"Absolute path: {os.path.abspath(base_dir)}")
        return []

    report_path = os.path.join(base_dir, report_file)
    if not os.path.exists(report_path) and report_file != "accuracy_report.txt":
        report_path = os.path.join(base_dir, "accuracy_report.txt")
    all_metrics = extract_metrics_from_report(report_path)

    if not all_metrics:
        logger.warning(f"No metrics extracted from {report_path}")
        return []

    settings = extract_settings(base_dir)
    description = extract_ablation_description(base_dir)

    # Inject ablation cost if available
    cost_usd = None
    cost_path = os.path.join(base_dir, "cost_report.json")
    if os.path.exists(cost_path):
        try:
            with open(cost_path) as f:
                cost_data = json.load(f)
                cost_usd = cost_data.get("total_cost_usd")
        except Exception:
            pass

    rows = []
    for mode, metrics in all_metrics.items():
        rows.append({
            'ndcg': metrics.get('ndcg', None),
            'mrr': metrics.get('mrr', None),
            'hits_1': metrics.get('hits_1', None),
            'hits_2': metrics.get('hits_2', None),
            'hits_3': metrics.get('hits_3', None),
            'pairwise_accuracy': metrics.get('pairwise_accuracy', None),
            'pairwise_accuracy_no_ties': metrics.get('pairwise_accuracy_no_ties', None),
            'n_ties': metrics.get('n_ties', None),
            'support': metrics.get('support', None),
            'support_without_ties': metrics.get('support_without_ties', None),
            'accuracy': metrics.get('accuracy', None),
            'f1_macro': metrics.get('f1_macro', None),
            'f1_pos': metrics.get('f1_pos', None),
            'f1_neg': metrics.get('f1_neg', None),
            'precision_pos': metrics.get('precision_pos', None),
            'precision_neg': metrics.get('precision_neg', None),
            'recall_pos': metrics.get('recall_pos', None),
            'recall_neg': metrics.get('recall_neg', None),
            'cost_usd': cost_usd,
            'description': description,
            'model': settings['model'],
            'retrieval': settings['retrieval'],
            'reasoning_effort': settings['reasoning_effort'],
            'mode': mode,
        })
    return rows

def extract_random_baseline(dirs):
    all_pairwise = True  # innocent until proven guilty
    for base_dir in dirs:
        report_path = os.path.join(base_dir, "accuracy_report.txt")
        if not os.path.exists(report_path):
            continue

        with open(report_path, 'r') as f:
            content = f.read()

        # Check if this report is ranking mode (pairwise/pointwise reports have no random section)
        test_mode_match = re.search(r'Test Mode:\s*(\w+)', content)
        test_mode = test_mode_match.group(1).lower() if test_mode_match else 'ranking'
        if test_mode not in ('pairwise', 'pointwise'):
            all_pairwise = False

        # Extract metrics for Random Ranking
        random_section = re.search(r'1\. Random Ranking\n-+\n(.*?)(?=\n\n|\Z)', content, re.DOTALL)
        if random_section:
            random_text = random_section.group(1)
            metrics = {}

            ndcg_match = re.search(r'Mean NDCG: ([\d\.]+)', random_text)
            mrr_match = re.search(r'Mean MRR: ([\d\.]+)', random_text)
            hits1_match = re.search(r'Mean Hits@1: ([\d\.]+)', random_text)
            hits2_match = re.search(r'Mean Hits@2: ([\d\.]+)', random_text)
            hits3_match = re.search(r'Mean Hits@3: ([\d\.]+)', random_text)

            metrics['ndcg'] = float(ndcg_match.group(1)) if ndcg_match else 0.0
            metrics['mrr'] = float(mrr_match.group(1)) if mrr_match else 0.0
            metrics['hits_1'] = float(hits1_match.group(1)) if hits1_match else 0.0
            metrics['hits_2'] = float(hits2_match.group(1)) if hits2_match else 0.0
            metrics['hits_3'] = float(hits3_match.group(1)) if hits3_match else 0.0

            metrics['description'] = 'Random Baseline'
            metrics['model'] = 'Random'
            metrics['retrieval'] = 'N/A'
            metrics['reasoning_effort'] = 'N/A'
            metrics['pairwise_accuracy'] = None

            logger.info(f"Extracted random baseline from {base_dir}")
            return metrics

    if not all_pairwise:
        logger.warning("Could not find '1. Random Ranking' section in any provided directory.")
    return None

def _display_description(desc: str) -> str:
    """Strip leading 'Cx ablation: ' prefix for cleaner table display."""
    return re.sub(r'^C\d+[a-z]? ablation:\s*', '', desc)


def format_value(val, max_val, random_val=None):
    bold_str = f"**{val:.4f}**" if val == max_val else f"{val:.4f}"
    
    if random_val is not None:
        diff = val - random_val
        sign = "+" if diff >= 0 else ""
        return f"{bold_str} ({sign}{diff:.4f})"
    
    return bold_str

def _display_ranking_description(desc: str) -> str:
    """Strip leading 'Ranking Cx: ' or 'Ranking: ' prefix for heatmap row labels."""
    return re.sub(r'^Ranking\s*(?:C\d+[a-z]?:)?\s*', '', desc).strip()


def generate_ranking_heatmaps(
    ranking_rows: list[dict],
    output_path: str,
    figures_dir: str | None = None,
    group_filter: str | None = None,
) -> list[str]:
    """
    Generate one heatmap per ranking metric (NDCG, MRR, Hits@1-3) with
    ablations as rows and models as columns.  Saves PNGs next to *output_path*
    and returns their paths.  Returns [] if matplotlib is unavailable.

    group_filter: if set, only include ablations from that group (plus baseline).
                  The full heatmap (group_filter=None) draws separator lines
                  between groups.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping ranking heatmap generation.")
        return []

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    BASELINE_DESC = _ABLATION_DESCRIPTIONS.get("ranking_current", _ABLATION_DESCRIPTIONS.get("current", "Current"))

    # Filter to the requested group (always keep baseline rows for reference)
    if group_filter is not None:
        ranking_rows = [
            s for s in ranking_rows
            if _ABLATION_GROUPS.get(s["description"], "") in (group_filter, "baseline")
        ]

    seen_ablations: dict[str, int] = {}
    seen_models: dict[str, int] = {}
    seen_groups: dict[str, str] = {}  # display_desc -> group name (for separator lines)
    for s in ranking_rows:
        desc = _display_ranking_description(_display_description(s["description"]))
        if desc.lower() == "current":
            desc = _display_ranking_description(BASELINE_DESC)
        if desc not in seen_ablations:
            seen_ablations[desc] = _ABLATION_ORDER_MAP.get(desc, 999 if "all-bad" in desc.lower() else 998)
            seen_groups[desc] = _ABLATION_GROUPS.get(s["description"], "")
        model = s["model"]
        if model not in seen_models:
            seen_models[model] = len(seen_models)

    ablations = sorted(seen_ablations, key=lambda d: seen_ablations[d])
    models = sorted(seen_models, key=lambda m: seen_models[m])
    n_rows, n_cols = len(ablations), len(models)
    baseline_display = _display_ranking_description(_display_description(BASELINE_DESC))

    metrics = [
        ("ndcg",   "Mean NDCG", "ranking_heatmap_ndcg"),
        ("mrr",    "Mean MRR",  "ranking_heatmap_mrr"),
        ("hits_1", "Hits@1",    "ranking_heatmap_hits1"),
        ("hits_2", "Hits@2",    "ranking_heatmap_hits2"),
        ("hits_3", "Hits@3",    "ranking_heatmap_hits3"),
    ]

    import textwrap
    _wrap = lambda s: "\n".join(textwrap.wrap(s, width=32))

    png_paths = []
    for field, title, suffix in metrics:
        data = np.full((n_rows, n_cols), np.nan)
        for s in ranking_rows:
            desc = _display_ranking_description(_display_description(s["description"]))
            if desc.lower() == "current":
                desc = _display_ranking_description(BASELINE_DESC)
            val = s.get(field)
            if val is None or desc not in ablations:
                continue
            r = ablations.index(desc)
            c = models.index(s["model"])
            data[r, c] = val

        data_aug = data
        n_cols_aug = n_cols

        vmin = max(0.0, float(np.nanmin(data)) - 0.03) if not np.all(np.isnan(data)) else 0.0
        vmax = min(1.0, float(np.nanmax(data)) + 0.03) if not np.all(np.isnan(data)) else 1.0

        fig, ax = plt.subplots(
            figsize=(max(9, n_cols_aug * 1.8), max(4, n_rows * 1.3 + 2)),
            facecolor="white",
        )
        ax.set_facecolor("white")

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rg_muted", ["#d64045", "#f7f4ef", "#3d9970"], N=256
        )
        cmap.set_bad(color="#e2e8f0")
        im = ax.imshow(data_aug, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                       interpolation="nearest")

        _sep_ys = {sep - 0.5 for sep in _group_separator_rows(ablations, seen_groups)} if group_filter is None else set()
        for x in np.arange(-0.5, n_cols_aug + 0.5, 1):
            ax.axvline(x, color="white", linewidth=1.0)
        for y in np.arange(-0.5, n_rows, 1):
            if y not in _sep_ys:
                ax.axhline(y, color="white", linewidth=1.0)

        # Group separator lines (only in full heatmap, not per-group views)
        if group_filter is None:
            for sep_row in _group_separator_rows(ablations, seen_groups):
                ax.axhline(sep_row - 0.5, color="#64748b", linewidth=2.0, zorder=3)

        for r in range(n_rows):
            for c in range(n_cols_aug):
                v = data_aug[r, c]
                if np.isnan(v):
                    ax.text(c, r, "—", ha="center", va="center",
                            fontsize=13, color="#bbbbbb")
                else:
                    norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    text_color = "white" if norm_v < 0.3 or norm_v > 0.82 else "#1a1a1a"
                    ax.text(c, r, f"{v:.4f}", ha="center", va="center",
                            fontsize=13, color=text_color, fontweight="bold")

        baseline_row_idx = ablations.index(baseline_display) if baseline_display in ablations else None
        if baseline_row_idx is not None:
            br = baseline_row_idx
            ax.add_patch(mpatches.FancyBboxPatch(
                (-0.5, br - 0.5), n_cols_aug, 1,
                boxstyle="square,pad=0", linewidth=2.0,
                edgecolor="#1e293b", facecolor="none", zorder=4,
            ))

        # Group label bar on the right (full heatmap only)
        if group_filter is None:
            _BAR_W = 0.8
            ax.set_xlim(-0.5, n_cols_aug - 0.5 + _BAR_W)
            for grp, r_start, r_end in _compute_group_spans(ablations, seen_groups):
                color = _GROUP_COLORS.get(grp, "#aaaaaa")
                ax.add_patch(mpatches.Rectangle(
                    (n_cols_aug - 0.5, r_start - 0.5), _BAR_W, r_end - r_start + 1.0,
                    linewidth=0, facecolor=color, alpha=0.9, zorder=2,
                ))
                label = _GROUP_DISPLAY_NAMES.get(grp, grp.replace("_", " ").title())
                ax.text(
                    n_cols_aug - 0.5 + _BAR_W / 2, (r_start + r_end) / 2.0, label,
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white", rotation=90, zorder=3, clip_on=False,
                )
            ax.axvline(n_cols_aug - 0.5, color="white", linewidth=1.5, zorder=5)

        ax.set_xticks(range(n_cols_aug))
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=13)
        wrapped_ablations = [_wrap(a) for a in ablations]
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(wrapped_ablations, fontsize=13, linespacing=1.3)
        for i, abl in enumerate(ablations):
            weight = "bold" if abl == baseline_display else "normal"
            color  = "#0f172a" if abl == baseline_display else "#475569"
            ax.get_yticklabels()[i].set_fontweight(weight)
            ax.get_yticklabels()[i].set_color(color)
        ax.tick_params(length=0)
        group_label = f" [{group_filter}]" if group_filter else ""
        ax.set_title(f"{title}{group_label}", fontsize=16, pad=18, loc="left",
                     color="#0f172a")

        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, shrink=0.8)
        cb.ax.tick_params(labelsize=11, length=3)
        cb.set_label(title, fontsize=11, labelpad=8)
        cb.outline.set_visible(False)

        fig.patch.set_facecolor("white")
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        out_dir = figures_dir if figures_dir else os.path.dirname(output_path)
        group_suffix = f"_{group_filter}" if group_filter else ""
        png_path = os.path.join(out_dir, f"{base_name}_{suffix}{group_suffix}.png")
        fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Ranking heatmap ({title}) saved to: {png_path}")
        png_paths.append(png_path)

    return png_paths


def generate_pointwise_heatmaps(
    pointwise_rows: list[dict],
    output_path: str,
    figures_dir: str | None = None,
    group_filter: str | None = None,
) -> list[str]:
    """
    Generate one heatmap per pointwise metric (accuracy, f1_macro, f1_pos, f1_neg,
    precision_pos, precision_neg, recall_pos, recall_neg) with ablations as rows
    and models as columns.  Saves PNGs into *figures_dir* and returns their paths.
    Returns [] if matplotlib is unavailable.

    group_filter: if set, only include ablations from that group (plus baseline).
                  The full heatmap (group_filter=None) draws separator lines
                  between groups.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping heatmap generation.")
        return []

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    BASELINE_DESC = _ABLATION_DESCRIPTIONS.get("current", "Current")

    # Filter to the requested group (always keep baseline rows for reference)
    if group_filter is not None:
        pointwise_rows = [
            s for s in pointwise_rows
            if _ABLATION_GROUPS.get(s["description"], "") in (group_filter, "baseline")
        ]

    seen_ablations: dict[str, int] = {}
    seen_models: dict[str, int] = {}
    seen_groups: dict[str, str] = {}  # display_desc -> group name (for separator lines)
    for s in pointwise_rows:
        desc = _display_description(s["description"])
        if desc.lower() == "current":
            desc = BASELINE_DESC
        if desc not in seen_ablations:
            seen_ablations[desc] = _ABLATION_ORDER_MAP.get(desc, 999 if "all-bad" in desc.lower() else 998)
            seen_groups[desc] = _ABLATION_GROUPS.get(s["description"], "")
        model = s["model"]
        if model not in seen_models:
            seen_models[model] = len(seen_models)

    ablations = sorted(seen_ablations, key=lambda d: seen_ablations[d])
    models = sorted(seen_models, key=lambda m: seen_models[m])
    n_rows, n_cols = len(ablations), len(models)

    baseline_display = _display_description(BASELINE_DESC)
    baseline_row_idx = ablations.index(baseline_display) if baseline_display in ablations else None

    metrics = [
        ("f1_macro",     "F1 (macro)"),
        ("f1_pos",       "F1 (POSITIVE)"),
        ("f1_neg",       "F1 (NEGATIVE)"),
    ]

    import textwrap
    _wrap = lambda s: "\n".join(textwrap.wrap(s, width=28))
    wrapped_ablations = [_wrap(a) for a in ablations]

    out_dir = figures_dir if figures_dir else os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    png_paths = []

    for metric_key, metric_label in metrics:
        data = np.full((n_rows, n_cols), np.nan)
        for s in pointwise_rows:
            desc = _display_description(s["description"])
            if desc.lower() == "current":
                desc = BASELINE_DESC
            val = s.get(metric_key)
            if val is None:
                continue
            r = ablations.index(desc)
            c = models.index(s["model"])
            data[r, c] = val

        if np.all(np.isnan(data)):
            continue

        data_aug = data
        n_cols_aug = n_cols

        fig, ax = plt.subplots(
            figsize=(max(9, n_cols_aug * 1.8), max(4, n_rows * 1.3 + 2)),
            facecolor="white",
        )
        ax.set_facecolor("white")

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rg_muted", ["#d64045", "#f7f4ef", "#3d9970"], N=256
        )
        cmap.set_bad(color="#e2e8f0")
        vmin = max(0.0, np.nanmin(data) - 0.03)
        vmax = min(1.0, np.nanmax(data) + 0.03)
        im = ax.imshow(data_aug, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                       interpolation="nearest")

        _sep_ys = {sep - 0.5 for sep in _group_separator_rows(ablations, seen_groups)} if group_filter is None else set()
        for x in np.arange(-0.5, n_cols_aug + 0.5, 1):
            ax.axvline(x, color="white", linewidth=1.0)
        for y in np.arange(-0.5, n_rows, 1):
            if y not in _sep_ys:
                ax.axhline(y, color="white", linewidth=1.0)

        # Group separator lines (only in full heatmap, not per-group views)
        if group_filter is None:
            for sep_row in _group_separator_rows(ablations, seen_groups):
                ax.axhline(sep_row - 0.5, color="#64748b", linewidth=2.0, zorder=3)

        for r in range(n_rows):
            for c in range(n_cols_aug):
                v = data_aug[r, c]
                if np.isnan(v):
                    ax.text(c, r, "—", ha="center", va="center",
                            fontsize=13, color="#bbbbbb")
                else:
                    norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    text_color = "white" if norm_v < 0.3 or norm_v > 0.82 else "#1a1a1a"
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=13, color=text_color, fontweight="bold")

        if baseline_row_idx is not None:
            ax.add_patch(mpatches.FancyBboxPatch(
                (-0.5, baseline_row_idx - 0.5), n_cols_aug, 1,
                boxstyle="square,pad=0", linewidth=2.0,
                edgecolor="#1e293b", facecolor="none", zorder=4,
            ))

        # Group label bar on the right (full heatmap only)
        if group_filter is None:
            _BAR_W = 0.8
            ax.set_xlim(-0.5, n_cols_aug - 0.5 + _BAR_W)
            for grp, r_start, r_end in _compute_group_spans(ablations, seen_groups):
                color = _GROUP_COLORS.get(grp, "#aaaaaa")
                ax.add_patch(mpatches.Rectangle(
                    (n_cols_aug - 0.5, r_start - 0.5), _BAR_W, r_end - r_start + 1.0,
                    linewidth=0, facecolor=color, alpha=0.9, zorder=2,
                ))
                label = _GROUP_DISPLAY_NAMES.get(grp, grp.replace("_", " ").title())
                ax.text(
                    n_cols_aug - 0.5 + _BAR_W / 2, (r_start + r_end) / 2.0, label,
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white", rotation=90, zorder=3, clip_on=False,
                )
            ax.axvline(n_cols_aug - 0.5, color="white", linewidth=1.5, zorder=5)

        ax.set_xticks(range(n_cols_aug))
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=13)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(wrapped_ablations, fontsize=13, linespacing=1.3)
        for i, abl in enumerate(ablations):
            weight = "bold" if abl == baseline_display else "normal"
            color  = "#0f172a" if abl == baseline_display else "#475569"
            ax.get_yticklabels()[i].set_fontweight(weight)
            ax.get_yticklabels()[i].set_color(color)
        ax.tick_params(length=0)
        group_label = f" [{group_filter}]" if group_filter else ""
        ax.set_title(f"Pointwise {metric_label}{group_label}", fontsize=16,
                     pad=18, loc="left", color="#0f172a")

        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, shrink=0.8)
        cb.ax.tick_params(labelsize=11, length=3)
        cb.set_label(metric_label, fontsize=11, labelpad=8)
        cb.outline.set_visible(False)

        fig.patch.set_facecolor("white")

        suffix = metric_key.replace("_", "-")
        group_suffix = f"_{group_filter}" if group_filter else ""
        png_path = os.path.join(out_dir, f"{base_name}_pointwise_heatmap_{suffix}{group_suffix}.png")
        fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Pointwise heatmap saved to: {png_path}")
        png_paths.append(png_path)

    return png_paths


def generate_pairwise_heatmap(
    pairwise_rows: list[dict],
    output_path: str,
    metric: str = "pairwise_accuracy_no_ties",
    figures_dir: str | None = None,
    group_filter: str | None = None,
) -> str | None:
    """
    Generate a heatmap of pairwise accuracy with ablations as rows and models
    as columns.  *metric* selects which accuracy column to plot:
      - "pairwise_accuracy_no_ties"  → Acc (w/o ties)
      - "pairwise_accuracy"          → Acc (with ties)
    Saves a PNG into *figures_dir* (or next to *output_path* if not given)
    and returns its path, or None if matplotlib is unavailable.

    group_filter: if set, only include ablations from that group (plus baseline).
                  The full heatmap (group_filter=None) draws separator lines
                  between groups.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping heatmap generation.")
        return None

    # ── Style ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    # ── Build pivot: ablation × model ─────────────────────────────────────
    BASELINE_DESC = _ABLATION_DESCRIPTIONS.get("current", "Current")

    # Filter to the requested group (always keep baseline rows for reference)
    if group_filter is not None:
        pairwise_rows = [
            s for s in pairwise_rows
            if _ABLATION_GROUPS.get(s["description"], "") in (group_filter, "baseline")
        ]

    seen_ablations: dict[str, int] = {}
    seen_models: dict[str, int] = {}
    seen_groups: dict[str, str] = {}  # display_desc -> group name (for separator lines)
    for s in pairwise_rows:
        desc = _display_description(s["description"])
        if desc.lower() == "current":
            desc = BASELINE_DESC
        if desc not in seen_ablations:
            seen_ablations[desc] = _ABLATION_ORDER_MAP.get(desc, 999 if "all-bad" in desc.lower() else 998)
            seen_groups[desc] = _ABLATION_GROUPS.get(s["description"], "")
        model = s["model"]
        if model not in seen_models:
            seen_models[model] = len(seen_models)

    ablations = sorted(seen_ablations, key=lambda d: seen_ablations[d])
    models = sorted(seen_models, key=lambda m: seen_models[m])
    n_rows, n_cols = len(ablations), len(models)

    # Fallback order so we always get a value even if the requested metric is missing
    _fallback = (
        ["pairwise_accuracy_no_ties", "pairwise_accuracy"]
        if metric == "pairwise_accuracy_no_ties"
        else ["pairwise_accuracy", "pairwise_accuracy_no_ties"]
    )

    data = np.full((n_rows, n_cols), np.nan)
    for s in pairwise_rows:
        desc = _display_description(s["description"])
        if desc.lower() == "current":
            desc = BASELINE_DESC
        val = next((s.get(k) for k in _fallback if s.get(k) is not None), None)
        if val is None:
            continue
        r = ablations.index(desc)
        c = models.index(s["model"])
        data[r, c] = val

    data_aug = data
    n_cols_aug = n_cols

    baseline_display = _display_description(BASELINE_DESC)
    _metric_label = "w/o ties" if metric == "pairwise_accuracy_no_ties" else "with ties"
    _metric_suffix = "no_ties" if metric == "pairwise_accuracy_no_ties" else "with_ties"

    baseline_row_idx = ablations.index(baseline_display) if baseline_display in ablations else None

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax_heat = plt.subplots(
        figsize=(max(9, n_cols_aug * 1.8), max(4, n_rows * 1.3 + 2)),
        facecolor="white",
    )
    ax_heat.set_facecolor("white")

    # ── Heatmap ────────────────────────────────────────────────────────────
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg_muted", ["#d64045", "#f7f4ef", "#3d9970"], N=256
    )
    cmap.set_bad(color="#e2e8f0")
    vmin = max(0.0, np.nanmin(data) - 0.03)
    vmax = min(1.0, np.nanmax(data) + 0.03)
    im = ax_heat.imshow(data_aug, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                        interpolation="nearest")

    # Grid lines between cells
    _sep_ys = {sep - 0.5 for sep in _group_separator_rows(ablations, seen_groups)} if group_filter is None else set()
    for x in np.arange(-0.5, n_cols_aug + 0.5, 1):
        ax_heat.axvline(x, color="white", linewidth=1.0)
    for y in np.arange(-0.5, n_rows, 1):
        if y not in _sep_ys:
            ax_heat.axhline(y, color="white", linewidth=1.0)

    # Group separator lines (only in full heatmap, not per-group views)
    if group_filter is None:
        for sep_row in _group_separator_rows(ablations, seen_groups):
            ax_heat.axhline(sep_row - 0.5, color="#64748b", linewidth=2.0, zorder=3)

    # Cell annotations
    for r in range(n_rows):
        for c in range(n_cols_aug):
            v = data_aug[r, c]
            if np.isnan(v):
                ax_heat.text(c, r, "—", ha="center", va="center",
                             fontsize=13, color="#bbbbbb")
            else:
                norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = "white" if norm_v < 0.3 or norm_v > 0.82 else "#1a1a1a"
                ax_heat.text(c, r, f"{v:.3f}", ha="center", va="center",
                             fontsize=13, color=text_color, fontweight="bold")

    # Baseline row highlight: subtle shaded band + thick border
    if baseline_row_idx is not None:
        br = baseline_row_idx
        ax_heat.add_patch(mpatches.FancyBboxPatch(
            (-0.5, br - 0.5), n_cols_aug, 1,
            boxstyle="square,pad=0", linewidth=2.0,
            edgecolor="#1e293b", facecolor="none", zorder=4,
        ))

    # Group label bar on the right (full heatmap only)
    if group_filter is None:
        _BAR_W = 0.8
        ax_heat.set_xlim(-0.5, n_cols_aug - 0.5 + _BAR_W)
        for grp, r_start, r_end in _compute_group_spans(ablations, seen_groups):
            color = _GROUP_COLORS.get(grp, "#aaaaaa")
            ax_heat.add_patch(mpatches.Rectangle(
                (n_cols_aug - 0.5, r_start - 0.5), _BAR_W, r_end - r_start + 1.0,
                linewidth=0, facecolor=color, alpha=0.9, zorder=2,
            ))
            label = _GROUP_DISPLAY_NAMES.get(grp, grp.replace("_", " ").title())
            ax_heat.text(
                n_cols_aug - 0.5 + _BAR_W / 2, (r_start + r_end) / 2.0, label,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white", rotation=90, zorder=3, clip_on=False,
            )
        ax_heat.axvline(n_cols_aug - 0.5, color="white", linewidth=1.5, zorder=5)

    # Axes
    ax_heat.set_xticks(range(n_cols_aug))
    ax_heat.set_xticklabels(models, rotation=40, ha="right", fontsize=13)
    import textwrap
    _wrap = lambda s: "\n".join(textwrap.wrap(s, width=28))
    wrapped_ablations = [_wrap(a) for a in ablations]
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(wrapped_ablations, fontsize=13, linespacing=1.3)
    for i, abl in enumerate(ablations):
        weight = "bold" if abl == baseline_display else "normal"
        color  = "#0f172a" if abl == baseline_display else "#475569"
        ax_heat.get_yticklabels()[i].set_fontweight(weight)
        ax_heat.get_yticklabels()[i].set_color(color)
    ax_heat.tick_params(length=0)
    group_label = f" [{group_filter}]" if group_filter else ""
    ax_heat.set_title(f"Pairwise Accuracy ({_metric_label}){group_label}", fontsize=16,
                      pad=18, loc="left", color="#0f172a")

    cb = fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02, shrink=0.8)
    cb.ax.tick_params(labelsize=11, length=3)
    cb.set_label("Accuracy", fontsize=11, labelpad=8)
    cb.outline.set_visible(False)

    fig.patch.set_facecolor("white")

    base_name = os.path.splitext(os.path.basename(output_path))[0]
    out_dir = figures_dir if figures_dir else os.path.dirname(output_path)
    group_suffix = f"_{group_filter}" if group_filter else ""
    png_path = os.path.join(out_dir, f"{base_name}_pairwise_heatmap_{_metric_suffix}{group_suffix}.png")
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Pairwise heatmap saved to: {png_path}")
    return png_path


def main():
    global logger
    parser = argparse.ArgumentParser(description="Generate comparison report")
    parser.add_argument('--config', type=str, help="Path to YAML config file containing directories")
    parser.add_argument('--dirs', nargs='+', help="List of directories to compare (alternative to --config)")
    parser.add_argument('--output', type=str, default="comparison_report.md", help="Output markdown file")
    parser.add_argument(
        '--report-file',
        type=str,
        default="accuracy_report.txt",
        help=(
            "Name of the report file to read from each artifact directory "
            "(default: accuracy_report.txt). Falls back to accuracy_report.txt "
            "when the specified file does not exist in a directory."
        ),
    )

    args = parser.parse_args()
    
    # Load directories from config file or command line
    if args.config:
        dirs = load_dirs_from_yaml(args.config)
    elif args.dirs:
        dirs = args.dirs
    else:
        parser.error("Either --config or --dirs must be provided")

    # Setup logger
    # Use the first directory as the base for logs, or current directory if not available
    log_dir = dirs[0] if dirs else "."
    if not os.path.exists(log_dir):
        log_dir = "."
    logger = setup_logger(log_dir)
    
    logger.info(f"Current Working Directory: {os.getcwd()}")
    
    stats_list = []
    
    # Try to get random baseline from any of the directories
    random_stats = None
    if dirs:
        random_stats = extract_random_baseline(dirs)
        if random_stats:
            # We don't append to stats_list yet, we want to print it first
            logger.info(f"Extracted random baseline stats: {random_stats}")
    
    for d in dirs:
        rows = process_directory(d, report_file=args.report_file)
        if rows:
            stats_list.extend(rows)
            logger.info(f"Added {len(rows)} row(s) for directory {d}")
        else:
            logger.warning(f"Could not process directory {d}")
            
    if not stats_list and not random_stats:
        logger.error("No stats collected. Report will be empty.")

    # Sort all rows using the canonical ablations.yaml order so tables and
    # heatmaps always show ablations in the same sequence.
    stats_list.sort(key=lambda x: (
        x['model'],
        _ABLATION_ORDER_MAP.get(x.get('description', ''), 999 if "all-bad" in x.get('description', '').lower() else 998),
        x.get('description', ''),
        x['retrieval'],
        x['reasoning_effort'],
        x.get('mode', ''),
    ))
    stats_list = [s for s in stats_list if s.get('description') not in _SKIP_DESCRIPTIONS]

    pairwise_rows  = [s for s in stats_list if s.get('mode') == 'pairwise']
    pointwise_rows = [s for s in stats_list if s.get('mode') == 'pointwise']
    ranking_rows   = [s for s in stats_list if s.get('mode') not in ('pairwise', 'pointwise')]

    def _fmt(val, max_val, rand_val=None):
        if val is None or max_val is None:
            return "N/A"
        return format_value(val, max_val, rand_val)

    with open(args.output, 'w') as f:

        # ── Pointwise table ───────────────────────────────────────────────
        if pointwise_rows:
            f.write("## Pointwise Results\n\n")
            from itertools import groupby as _groupby_pw
            for model, group in _groupby_pw(pointwise_rows, key=lambda s: s['model']):
                group = list(group)
                local_acc    = [s['accuracy']       for s in group if s.get('accuracy')       is not None]
                local_f1_mac = [s['f1_macro']       for s in group if s.get('f1_macro')       is not None]
                local_f1_pos = [s['f1_pos']         for s in group if s.get('f1_pos')         is not None]
                local_f1_neg = [s['f1_neg']         for s in group if s.get('f1_neg')         is not None]
                local_prec_pos = [s['precision_pos'] for s in group if s.get('precision_pos') is not None]
                local_prec_neg = [s['precision_neg'] for s in group if s.get('precision_neg') is not None]
                local_rec_pos  = [s['recall_pos']    for s in group if s.get('recall_pos')    is not None]
                local_rec_neg  = [s['recall_neg']    for s in group if s.get('recall_neg')    is not None]
                max_acc      = max(local_acc)      if local_acc      else None
                max_f1_mac   = max(local_f1_mac)   if local_f1_mac   else None
                max_f1_pos   = max(local_f1_pos)   if local_f1_pos   else None
                max_f1_neg   = max(local_f1_neg)   if local_f1_neg   else None
                max_prec_pos = max(local_prec_pos) if local_prec_pos else None
                max_prec_neg = max(local_prec_neg) if local_prec_neg else None
                max_rec_pos  = max(local_rec_pos)  if local_rec_pos  else None
                max_rec_neg  = max(local_rec_neg)  if local_rec_neg  else None

                f.write(f"### {model}\n\n")
                f.write("| Description | Effort | Accuracy | F1 (macro) | F1 (POS) | F1 (NEG) | Samples | Cost ($) |\n")
                f.write("|---|---|---|---|---|---|---|---|\n")
                for s in group:
                    cost_str = f"${s['cost_usd']:.4f}" if s.get('cost_usd') is not None else "N/A"
                    f.write(
                        f"| {_display_description(s['description'])} | {s['reasoning_effort']}"
                        f" | {_fmt(s.get('accuracy'),     max_acc)}"
                        f" | {_fmt(s.get('f1_macro'),     max_f1_mac)}"
                        f" | {_fmt(s.get('f1_pos'),       max_f1_pos)}"
                        f" | {_fmt(s.get('f1_neg'),       max_f1_neg)}"
                        f" | {s.get('support', 'N/A')}"
                        f" | {cost_str} |\n"
                    )
                f.write("\n")
                logger.info(f"Wrote pointwise table for model {model} to report: {args.output}")

            # ── Pointwise heatmaps (one per metric) ───────────────────────
            pointwise_figures_dir = os.path.join(os.path.dirname(args.output), "pointwise_figures")
            os.makedirs(pointwise_figures_dir, exist_ok=True)
            # Full heatmap with group separators
            png_paths = generate_pointwise_heatmaps(
                pointwise_rows, args.output, figures_dir=pointwise_figures_dir,
            )
            for png_path in png_paths:
                logger.info(f"Pointwise heatmap written to: {os.path.abspath(png_path)}")

        # ── Pairwise table ────────────────────────────────────────────────
        if pairwise_rows:
            pa_vals            = [s['pairwise_accuracy']        for s in pairwise_rows if s.get('pairwise_accuracy')        is not None]
            pa_no_tie_v        = [s['pairwise_accuracy_no_ties'] for s in pairwise_rows if s.get('pairwise_accuracy_no_ties') is not None]
            ties_vals          = [s['n_ties']                    for s in pairwise_rows if s.get('n_ties')                    is not None]
            support_vals       = [s['support']                   for s in pairwise_rows if s.get('support')                   is not None]
            support_no_tie_v   = [s['support_without_ties']      for s in pairwise_rows if s.get('support_without_ties')      is not None]

            max_pa             = max(pa_vals)          if pa_vals          else None
            max_pa_no_tie      = max(pa_no_tie_v)      if pa_no_tie_v      else None
            max_ties           = max(ties_vals)         if ties_vals        else None
            max_support        = max(support_vals)      if support_vals     else None
            max_support_no_tie = max(support_no_tie_v)  if support_no_tie_v else None

            f.write("## Pairwise Results\n\n")

            # Group rows by model (order preserved since stats_list is sorted by model)
            from itertools import groupby
            for model, group in groupby(pairwise_rows, key=lambda s: s['model']):
                group = list(group)

                # Local extremes for this model's sub-table
                local_pa       = [s['pairwise_accuracy']        for s in group if s.get('pairwise_accuracy')        is not None]
                local_no_tie   = [s['pairwise_accuracy_no_ties'] for s in group if s.get('pairwise_accuracy_no_ties') is not None]
                local_ties     = [s['n_ties']                    for s in group if s.get('n_ties')                    is not None]
                local_max_pa      = max(local_pa)     if local_pa     else None
                local_max_no_tie  = max(local_no_tie) if local_no_tie else None
                local_min_ties    = min(local_ties)   if local_ties   else None

                f.write(f"### {model}\n\n")
                f.write("| Description | Effort | Acc (with ties) | Ties | Acc (w/o ties) | Samples | Cost ($) |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for s in group:
                    pa_str     = _fmt(s.get('pairwise_accuracy'),        local_max_pa)
                    no_tie_str = _fmt(s.get('pairwise_accuracy_no_ties'), local_max_no_tie)
                    # Ties: bold the minimum
                    ties_val = s.get('n_ties')
                    if ties_val is not None and local_min_ties is not None:
                        ties_str = f"**{ties_val:.1f}**" if ties_val == local_min_ties else f"{ties_val:.1f}"
                    else:
                        ties_str = "N/A"
                    cost_str = f"${s['cost_usd']:.4f}" if s.get('cost_usd') is not None else "N/A"
                    f.write(f"| {_display_description(s['description'])} | {s['reasoning_effort']} | {pa_str} | {ties_str} | {no_tie_str} | {s.get('support', 'N/A')} | {cost_str} |\n")
                f.write("\n")
                logger.info(f"Wrote pairwise table for model {model} to report: {args.output}")

            # ── Heatmaps (w/o ties and with ties) ────────────────────────
            pairwise_figures_dir = os.path.join(os.path.dirname(args.output), "pairwise_figures")
            os.makedirs(pairwise_figures_dir, exist_ok=True)
            for metric, label in [
                ("pairwise_accuracy_no_ties", "Pairwise Heatmap (w/o ties)"),
                ("pairwise_accuracy",         "Pairwise Heatmap (with ties)"),
            ]:
                # Full heatmap with group separators
                png_path = generate_pairwise_heatmap(
                    pairwise_rows, args.output, metric=metric, figures_dir=pairwise_figures_dir,
                )
                if png_path:
                    logger.info(f"Pairwise heatmap written to: {os.path.abspath(png_path)}")

        # ── Ranking table ─────────────────────────────────────────────────
        if ranking_rows or random_stats:
            all_ranking = ranking_rows + ([random_stats] if random_stats else [])
            ndcg_vals  = [s['ndcg']     for s in all_ranking if s.get('ndcg')     is not None]
            mrr_vals   = [s['mrr']      for s in all_ranking if s.get('mrr')      is not None]
            hits1_vals = [s['hits_1']   for s in all_ranking if s.get('hits_1')   is not None]
            hits2_vals = [s['hits_2']   for s in all_ranking if s.get('hits_2')   is not None]
            hits3_vals = [s['hits_3']   for s in all_ranking if s.get('hits_3')   is not None]

            rand_ndcg = random_stats['ndcg']     if random_stats else None
            rand_mrr  = random_stats['mrr']      if random_stats else None
            rand_h1   = random_stats['hits_1']   if random_stats else None
            rand_h2   = random_stats['hits_2']   if random_stats else None
            rand_h3   = random_stats['hits_3']   if random_stats else None

            if pairwise_rows:
                f.write("\n## Ranking Results\n\n")

            # Random baseline — shown once as a reference row
            if random_stats:
                max_ndcg   = max(ndcg_vals)  if ndcg_vals  else None
                max_mrr    = max(mrr_vals)   if mrr_vals   else None
                max_hits_1 = max(hits1_vals) if hits1_vals else None
                max_hits_2 = max(hits2_vals) if hits2_vals else None
                max_hits_3 = max(hits3_vals) if hits3_vals else None
                f.write("### Random Baseline\n\n")
                f.write("| Description | Mean NDCG | Mean MRR | Hits@1 | Hits@2 | Hits@3 |\n")
                f.write("|---|---|---|---|---|---|\n")
                f.write(
                    f"| *{random_stats['description']}*"
                    f" | {_fmt(random_stats['ndcg'],   max_ndcg)}"
                    f" | {_fmt(random_stats['mrr'],    max_mrr)}"
                    f" | {_fmt(random_stats['hits_1'], max_hits_1)}"
                    f" | {_fmt(random_stats['hits_2'], max_hits_2)}"
                    f" | {_fmt(random_stats['hits_3'], max_hits_3)}"
                    f" |\n\n"
                )

            # Group rows by model — mirrors the pairwise section layout
            from itertools import groupby as _groupby
            for model, group in _groupby(ranking_rows, key=lambda s: s['model']):
                group = list(group)

                # Local extremes for bold-max within this model's sub-table
                local_ndcg  = [s['ndcg']   for s in group if s.get('ndcg')   is not None]
                local_mrr   = [s['mrr']    for s in group if s.get('mrr')    is not None]
                local_hits1 = [s['hits_1'] for s in group if s.get('hits_1') is not None]
                local_hits2 = [s['hits_2'] for s in group if s.get('hits_2') is not None]
                local_hits3 = [s['hits_3'] for s in group if s.get('hits_3') is not None]
                local_max_ndcg  = max(local_ndcg)  if local_ndcg  else None
                local_max_mrr   = max(local_mrr)   if local_mrr   else None
                local_max_hits1 = max(local_hits1) if local_hits1 else None
                local_max_hits2 = max(local_hits2) if local_hits2 else None
                local_max_hits3 = max(local_hits3) if local_hits3 else None

                f.write(f"### {model}\n\n")
                f.write("| Description | Effort | Mean NDCG | Mean MRR | Hits@1 | Hits@2 | Hits@3 | Samples | Cost ($) |\n")
                f.write("|---|---|---|---|---|---|---|---|---|\n")
                for s in group:
                    cost_str = f"${s['cost_usd']:.4f}" if s.get('cost_usd') is not None else "N/A"
                    f.write(
                        f"| {_display_description(s['description'])} | {s['reasoning_effort']}"
                        f" | {_fmt(s['ndcg'],   local_max_ndcg,  rand_ndcg)}"
                        f" | {_fmt(s['mrr'],    local_max_mrr,   rand_mrr)}"
                        f" | {_fmt(s['hits_1'], local_max_hits1, rand_h1)}"
                        f" | {_fmt(s['hits_2'], local_max_hits2, rand_h2)}"
                        f" | {_fmt(s['hits_3'], local_max_hits3, rand_h3)}"
                        f" | {s.get('support', 'N/A')}"
                        f" | {cost_str} |\n"
                    )
                f.write("\n")
                logger.info(f"Wrote ranking table for model {model} to report: {args.output}")

            # ── Ranking heatmaps (one per metric) ─────────────────────────
            if ranking_rows:
                ranking_figures_dir = os.path.join(os.path.dirname(args.output), "ranking_figures")
                os.makedirs(ranking_figures_dir, exist_ok=True)
                # Full heatmap with group separators
                png_paths = generate_ranking_heatmaps(
                    ranking_rows, args.output, figures_dir=ranking_figures_dir,
                )
                for p in png_paths:
                    logger.info(f"Ranking heatmap written to: {os.path.abspath(p)}")

if __name__ == "__main__":
    main()
