#!/usr/bin/env python3
"""
Ablation Study Orchestrator
============================
Orchestrates the execution of ablation experiments defined in `ablations.yaml`.
Handles instance creation, accuracy sweeps, and unified report generation.

Usage:
    python run_ablations.py [--ablations name1,name2]
                            [--skip-create]
                            [--instances-dir output/benchmark_instances/ablation]
                            [--output-dir output/ablation_sweeps]
                            [--models gpt-5.2,gpt-5.4]
                            [--n-runs 1]

Experiment Definitions:
    All ablation logic, tracks, and descriptions are defined in `ablations.yaml`.
    The orchestrator automatically expands multi-track ablations and handles
    complex inheritance/composition.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import subprocess
import yaml
import json
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from logging_utils import setup_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_ABLATION_DIR = _THIS_DIR
_ABLATIONS_FILE = _THIS_DIR / "ablations.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merges overrides into base dictionary."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_ablation_config() -> dict:
    with open(_ABLATIONS_FILE) as f:
        data = yaml.safe_load(f)

    raw_ablations = data["ablations"]

    def resolve_inheritance(name, entry, visited):
        if name in visited:
            raise ValueError(f"Circular inheritance detected: {name}")
        visited.add(name)

        if "inherit_from" not in entry:
            return entry

        bases = entry["inherit_from"]
        if isinstance(bases, str):
            bases = [bases]

        merged_result = {}
        for base_name in bases:
            if base_name not in raw_ablations:
                raise ValueError(f"Ablation '{name}' inherits from unknown base: '{base_name}'")
            
            base_resolved = resolve_inheritance(base_name, raw_ablations[base_name], visited.copy())
            
            # Inherit all keys except metadata fields that belong only to the child.
            # Using a blacklist (not a whitelist) means new config sections inherit automatically.
            _METADATA_KEYS = {"description", "tracks", "track", "inherit_from", "legacy_names"}
            for key, val in base_resolved.items():
                if key in _METADATA_KEYS:
                    continue
                val = deepcopy(val)
                if key in merged_result and isinstance(merged_result[key], dict) and isinstance(val, dict):
                    merged_result[key] = _deep_merge(merged_result[key], val)
                else:
                    merged_result[key] = val
        
        # Finally apply the inheriting ablation's own overrides
        overrides = {k: v for k, v in entry.items() if k != "inherit_from"}
        return _deep_merge(merged_result, overrides)

    # 1. Resolve inheritance for all raw definitions
    resolved_raw = {}
    for name, entry in raw_ablations.items():
        resolved_raw[name] = resolve_inheritance(name, entry, set())

    # 2. Expand ablations with multiple tracks
    expanded_ablations = {}
    for name, entry in resolved_raw.items():
        tracks = entry.get("tracks", [])
        if not tracks and "track" in entry:
            tracks = [entry["track"]]

        for track in tracks:
            # Always namespace the experiment name by the track, avoiding double prefixes
            if name.startswith(f"{track}_"):
                exp_name = name
            else:
                exp_name = f"{track}_{name}"

            new_entry = deepcopy(entry)
            new_entry["track"] = track

            # If the entry reuses instances from another ablation by name,
            # resolve it to the correct experiment name within the same track.
            if isinstance(new_entry.get("instance"), str):
                src = new_entry["instance"]
                if not src.startswith(f"{track}_"):
                    new_entry["instance"] = f"{track}_{src}"

            expanded_ablations[exp_name] = new_entry

    data["ablations"] = expanded_ablations
    return data


_CONFIG_DATA = _load_ablation_config()
_ABLATION_DESCRIPTIONS = {k: v["description"] for k, v in _CONFIG_DATA["ablations"].items()}
_ABLATION_REGISTRY = _CONFIG_DATA["ablations"]

# Two levels up: src/novelty_eval → src → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EVAL_DIR = _PROJECT_ROOT / "src" / "novelty_eval"
_CREATE_INSTANCES_SCRIPT = _EVAL_DIR / "benchmark_data" / "create_benchmark_instances.py"
_RETRIEVE_CANDIDATES_SCRIPT = _EVAL_DIR / "retrieval" / "retrieve_candidates.py"
_RUN_SWEEP_SCRIPT = _EVAL_DIR / "run_benchmark_sweep.py"
_GENERATE_REPORT_SCRIPT = _EVAL_DIR / "analysis" / "generate_report.py"

# Module-level logger — re-initialized in main() once output_base is known.
LOGGER = setup_logger(
    str(_PROJECT_ROOT / "output" / "ablation_sweeps"),
    console_level="INFO",
)

def _run_subprocess(cmd: list[str], label: str) -> int:
    LOGGER.info(f"Running: {label}")
    LOGGER.info(f"Command: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd], env=os.environ.copy(), cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        LOGGER.error(f"'{label}' exited with code {result.returncode}")
    return result.returncode


def _create_instances(inst_cfg: dict, output_base: Path, ablation: str) -> Path | None:
    """
    Invoke create_benchmark_instances.py.
    Returns the timestamped output directory containing the created instances,
    or None on failure.
    """
    # Allow output_base override (--instances-dir flag)
    if output_base:
        inst_cfg["output_dir"] = str(output_base / ablation)

    # Write a patched copy so the script picks up our output_dir override
    patched_cfg_path = output_base / ablation / "_instance_creation_config.yaml"
    patched_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(patched_cfg_path, "w") as f:
        yaml.dump(inst_cfg, f, default_flow_style=False)

    rc = _run_subprocess(
        [sys.executable, _CREATE_INSTANCES_SCRIPT, "--config", patched_cfg_path],
        f"create_instances:{ablation}",
    )
    if rc != 0:
        return None

    # Discover the timestamped sub-directory created by the script
    base_out = Path(inst_cfg["output_dir"])
    if not base_out.is_dir():
        LOGGER.warning(f"Expected output dir not found: {base_out}")
        return None

    sub_dirs = sorted(base_out.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    sub_dirs = [p for p in sub_dirs if p.is_dir() and not p.name.startswith("_")]
    if not sub_dirs:
        LOGGER.warning(f"No sub-directories found in {base_out}")
        return None

    inst_dir = sub_dirs[0]
    meta_path = inst_dir / "_instance_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"ablation": ablation, "created_at": datetime.now().isoformat()}, f, indent=2)

    return inst_dir


def _find_instances_dir_by_meta(instances_base: Path, ablation_name: str) -> Path | None:
    """
    Find the most-recently-modified timestamped instance directory whose
    `_instance_meta.json` identifies it as `ablation_name`.

    Falls back to matching by parent folder name for directories that were
    created before meta files were introduced (backward compatibility).
    """
    if not instances_base.is_dir():
        return None

    candidates = []
    for parent in instances_base.iterdir():
        if not parent.is_dir():
            continue
        for ts_dir in parent.iterdir():
            if not ts_dir.is_dir() or ts_dir.name.startswith("_"):
                continue
            meta_path = ts_dir / "_instance_meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("ablation") == ablation_name:
                    candidates.append(ts_dir)
            except (json.JSONDecodeError, OSError):
                continue

    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # Backward-compat fallback: match by parent folder name
    legacy_base = instances_base / ablation_name
    if legacy_base.is_dir():
        sub_dirs = sorted(legacy_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        sub_dirs = [p for p in sub_dirs if p.is_dir() and not p.name.startswith("_")]
        if sub_dirs:
            return sub_dirs[0]

    return None


def _find_instances_yaml(instance_dir: Path) -> Path | None:
    """
    Given the timestamped output directory from create_benchmark_instances.py,
    find the benchmark_instances.yaml (may be under a 'manipulated/' sub-dir).
    """
    candidates = list(instance_dir.rglob("benchmark_instances.yaml"))
    if not candidates:
        candidates = list(instance_dir.rglob("iclr_test_instances.yaml"))
    if not candidates:
        candidates = list(instance_dir.rglob("iclr_pointwise_instances.yaml"))
    if not candidates:
        candidates = list(instance_dir.rglob("test_instances.yaml"))
    if not candidates:
        LOGGER.warning(f"No instances YAML found under {instance_dir}")
        return None
    # Prefer manipulated/ over raw_abstract/ when both exist
    for c in candidates:
        if "manipulated" in str(c):
            return c
    return candidates[0]


def _strip_eval_from_instances_yaml(source_yaml: Path, dest_dir: Path) -> Path:
    """Strip the **evaluation** section from idea text in an instances YAML.

    Parses the YAML, removes '**evaluation**:' and everything after it from
    each instance's idea string, then re-serializes.  This correctly handles
    multi-line evaluation values that a single-line regex would leave behind.
    """
    import yaml as _yaml

    data = _yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    for entry in data.values():
        idea = entry.get("idea", "")
        if isinstance(idea, str) and "**evaluation**" in idea.lower():
            stripped = re.sub(
                r'\n\s*\*\*evaluation\*\*:.*', '', idea,
                flags=re.DOTALL | re.IGNORECASE,
            )
            entry["idea"] = stripped.rstrip()

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "benchmark_instances.yaml"
    with open(dest_path, "w", encoding="utf-8") as fh:
        _yaml.dump(data, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return dest_path


def _run_retrieval(instances_yaml: Path, ablation: str, retrieval_cfg: dict, num_instances: int | None = None) -> Path | None:
    """
    Invoke retrieve_candidates.py for the given instances.
    Returns the path to the created retrieval_cache.json, or None on failure.
    retrieve_candidates.py handles incremental updates internally, so we always
    invoke it — it will skip ideas that are already cached.
    """
    import tempfile
    import yaml as _yaml

    cache_path = instances_yaml.parent / "retrieval_cache.json"
    if cache_path.exists():
        LOGGER.info(f"Retrieval cache already exists for {ablation}, will update missing entries: {cache_path}")

    LOGGER.info(f"Starting retrieval for '{ablation}' (instances: {instances_yaml})")

    config = {
        "test_inputs": str(instances_yaml),
        "output_file": str(cache_path),
        **retrieval_cfg,
    }
    if num_instances is not None:
        config["nr_examples"] = num_instances

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        _yaml.dump(config, tmp, allow_unicode=True)
        tmp_config_path = tmp.name

    cmd = [sys.executable, _RETRIEVE_CANDIDATES_SCRIPT, "--config", tmp_config_path]
    rc = _run_subprocess(cmd, f"run_retrieval:{ablation}")
    os.unlink(tmp_config_path)

    if rc != 0:
        return None

    return cache_path


def _patch_and_write_sweep_config(
    sweep_cfg: dict,
    instances_yaml: Path | None,
    output_dir: Path,
    models: list[str],
    ablation: str,
    n_runs: int | None,
    num_instances: int | None,
    retrieval_cache_file: Path | None = None,
    force_batch: bool = False,
) -> Path:
    """
    Load the sweep config template, substitute __PLACEHOLDER__ with the real
    instances path, and replace the configs list with one entry per model named
    '<ablation>-<model>'. Optionally overrides n_runs/num_instances and injections
    retrieval cache.
    Returns the path to the patched config.
    """
    base = sweep_cfg.setdefault("base_config", {})

    # Promote top-level sweep keys (from defaults/track merge) into base_config so
    # run_benchmark_sweep.py can read them — it only looks at base_config, not top-level keys.
    _META_KEYS = {"base_config", "configs", "description"}
    for key, value in list(sweep_cfg.items()):
        if key not in _META_KEYS and key not in base:
            base[key] = value

    # Substitute placeholder
    if instances_yaml and base.get("test_inputs") == "__PLACEHOLDER__":
        base["test_inputs"] = str(instances_yaml)

    if retrieval_cache_file:
        base["retrieval_cache_file"] = str(retrieval_cache_file)
        base["retrieve_related_work"] = True

    # Replace configs list: one entry per model, named <ablation>-<model>
    sweep_cfg["configs"] = [
        {"name": f"{ablation}-{model}", "llm_engine": model}
        for model in models
    ]

    # CLI --n-runs overrides the per-ablation YAML value when explicitly provided.
    # When absent (None), the value already promoted from sweep_cfg is kept, falling
    # back to 1 if the YAML didn't specify one either.
    if n_runs is not None:
        base["n_runs"] = n_runs
    else:
        base.setdefault("n_runs", 1)

    if num_instances is not None:
        base["num_instances"] = num_instances

    if force_batch:
        base["use_batch_api"] = True

    patched_path = output_dir / "_sweep_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(patched_path, "w") as f:
        yaml.dump(sweep_cfg, f, default_flow_style=False)

    return patched_path


def _run_sweep(patched_sweep_config: Path, output_dir: Path, ablation: str,
               parallel_models: bool = False, max_parallel_models: int | None = None) -> list[str]:
    """
    Invoke run_benchmark_sweep.py with the patched config.
    Returns a list of artifact directories produced (may be empty on failure).
    """
    cmd = [sys.executable, _RUN_SWEEP_SCRIPT, "--config", patched_sweep_config,
           "--output-dir", output_dir, "--skip-on-error"]
    if parallel_models:
        cmd.append("--parallel")
    if max_parallel_models is not None:
        cmd += ["--max-parallel-models", str(max_parallel_models)]
    rc = _run_subprocess(cmd, f"run_sweep:{ablation}")
    if rc != 0:
        return []

    # Collect artifact dirs from sweep_summary.json
    summary_path = None
    for p in sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        candidate = p / "sweep_summary.json" if p.is_dir() else output_dir / "sweep_summary.json"
        if candidate.exists():
            summary_path = candidate
            break
    if summary_path is None:
        summary_path = output_dir / "sweep_summary.json"

    if not summary_path.exists():
        LOGGER.warning(f"sweep_summary.json not found under {output_dir}")
        return []

    with open(summary_path) as f:
        summary = json.load(f)

    return [
        r["artifact_dir"]
        for r in summary.get("runs", [])
        if r.get("success") and r.get("artifact_dir")
    ]


def _collect_artifact_dirs_from_run(run_dir: Path) -> list[str]:
    """
    Given a previous ablation output directory, collect all successful artifact
    dirs from its ablation_summary.json (preferred) or any nested
    sweep_summary.json files as fallback.
    """
    summary_path = run_dir / "ablation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        dirs = [
            d
            for row in summary.get("runs", [])
            for d in row.get("artifact_dirs", [])
            if d
        ]
        if dirs:
            return dirs

    # Fallback: walk nested sweep_summary.json files
    dirs = []
    for sweep_summary in run_dir.rglob("sweep_summary.json"):
        with open(sweep_summary) as f:
            data = json.load(f)
        dirs += [
            r["artifact_dir"]
            for r in data.get("runs", [])
            if r.get("success") and r.get("artifact_dir")
        ]
    return dirs


_COMPARE_RETRIEVAL_SCRIPT = _THIS_DIR / "compare_retrieval_impact.py"


def _generate_retrieval_impact_reports(output_base: Path, summary_rows: list[dict]) -> None:
    """
    For every successful retrieval ablation in this run, generate a retrieval
    impact comparison report against the corresponding baseline (current) ablation.

    Uses compare_retrieval_impact.py in --sweep-dir mode so it auto-discovers
    all model-level pairs within output_base.
    """
    retrieval_rows = [
        r for r in summary_rows
        if r.get("status") == "OK" and "retrieval" in r.get("ablation", "")
    ]
    if not retrieval_rows:
        return

    # Find which tracks had a successful retrieval run
    tracks_with_retrieval = {r["ablation"].replace("_retrieval", "") for r in retrieval_rows}
    # Verify the baseline also succeeded in this run
    successful = {r["ablation"] for r in summary_rows if r.get("status") == "OK"}
    tracks_with_pair = {
        t for t in tracks_with_retrieval if f"{t}_current" in successful
    }

    if not tracks_with_pair:
        LOGGER.info("Retrieval ablations found but no matching current baseline in this run — "
                    "skipping impact comparison. "
                    "Run with --sweep-dir to compare across runs.")
        return

    for track in sorted(tracks_with_pair):
        LOGGER.info(f"Generating retrieval impact report for track: {track}")
        cmd = [
            sys.executable, str(_COMPARE_RETRIEVAL_SCRIPT),
            "--sweep-dir", str(output_base),
            "--track", track,
        ]
        rc = _run_subprocess(cmd, f"compare_retrieval_impact:{track}")
        if rc != 0:
            LOGGER.warning(f"Retrieval impact comparison failed for track '{track}'.")


def _empty_cost_bucket() -> dict:
    return {
        "total_cost_usd": 0.0,
        "total_calls": 0,
        "total_input_tokens": 0,
        "total_cached_input_tokens": 0,
        "total_cache_creation_tokens": 0,
        "total_output_tokens": 0,
        "by_stage": {},  # stage_name → same fields (without nested by_stage)
        "by_model": {},  # model_name → {cost_usd, call_count, input_tokens, ...}
        "instances_processed": None,
        "instances_total": None,
    }


def _add_report_to_bucket(bucket: dict, report: dict) -> None:
    bucket["total_cost_usd"]              += report.get("total_cost_usd", 0.0)
    bucket["total_calls"]                 += report.get("total_calls", 0)
    bucket["total_input_tokens"]          += report.get("total_input_tokens", 0)
    bucket["total_cached_input_tokens"]   += report.get("total_cached_input_tokens", 0)
    bucket["total_cache_creation_tokens"] += report.get("total_cache_creation_tokens", 0)
    bucket["total_output_tokens"]         += report.get("total_output_tokens", 0)
    # Accumulate instance counts (None means "not reported")
    for field in ("instances_processed", "instances_total"):
        val = report.get(field)
        if val is not None:
            bucket[field] = (bucket[field] or 0) + val
    # Merge by_model
    for model_name, md in report.get("models", {}).items():
        if model_name not in bucket["by_model"]:
            bucket["by_model"][model_name] = {
                "cost_usd": 0.0,
                "call_count": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "cache_creation_tokens": 0,
                "output_tokens": 0,
            }
        mb = bucket["by_model"][model_name]
        mb["cost_usd"]              += md.get("cost_usd", 0.0)
        mb["call_count"]            += md.get("call_count", 0)
        mb["input_tokens"]          += md.get("input_tokens", 0)
        mb["cached_input_tokens"]   += md.get("cached_input_tokens", 0)
        mb["cache_creation_tokens"] += md.get("cache_creation_tokens", 0)
        mb["output_tokens"]         += md.get("output_tokens", 0)
    # Merge by_stage
    for stage, stage_data in report.get("by_stage", {}).items():
        if stage not in bucket["by_stage"]:
            bucket["by_stage"][stage] = {
                "total_cost_usd": 0.0,
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_cached_input_tokens": 0,
                "total_cache_creation_tokens": 0,
                "total_output_tokens": 0,
            }
        sb = bucket["by_stage"][stage]
        sb["total_cost_usd"]              += stage_data.get("total_cost_usd", 0.0)
        sb["total_calls"]                 += stage_data.get("total_calls", 0)
        sb["total_input_tokens"]          += stage_data.get("total_input_tokens", 0)
        sb["total_cached_input_tokens"]   += stage_data.get("total_cached_input_tokens", 0)
        sb["total_cache_creation_tokens"] += stage_data.get("total_cache_creation_tokens", 0)
        sb["total_output_tokens"]         += stage_data.get("total_output_tokens", 0)


def _aggregate_cost_reports(summary_rows: list[dict], output_base: Path) -> None:
    """
    Collect cost_report.json files from each ablation's artifact directories and
    (when present) from the instance creation output directory.

    Groups results by ablation *short name* (track prefix stripped), with one
    subsection per mode (instance_creation / pairwise / ranking / pointwise), and
    within each subsection a per-stage breakdown table.
    Writes ablation_cost_summary.json and ablation_cost_summary.md.
    """
    PRICING_NOTE = (
        "Prices are USD/1M tokens. Entries marked PLACEHOLDER in cost_tracker.py "
        "are estimates — update MODEL_PRICING with actual rates."
    )

    # short_name → {mode → cost_bucket}
    # Also preserve insertion order for consistent section order in the report.
    by_short_name: dict[str, dict[str, dict]] = {}
    descriptions: dict[str, str] = {}
    # Instance dirs already charged in this summary — instance creation happens once,
    # so a dir shared by several ablations must not be counted more than once.
    counted_inst_dirs: set[str] = set()

    for row in summary_rows:
        full_ablation = row["ablation"]
        artifact_dirs = row.get("artifact_dirs", [])

        # Derive track (mode) and short name from the registry
        track = _ABLATION_REGISTRY.get(full_ablation, {}).get("track", "")
        if track and full_ablation.startswith(f"{track}_"):
            short_name = full_ablation[len(track) + 1:]
        else:
            short_name = full_ablation

        descriptions.setdefault(short_name, row.get("description", ""))

        if short_name not in by_short_name:
            by_short_name[short_name] = {}

        # --- Instance creation cost (separate subprocess) ---
        # Only charge it when THIS run actually created the instances. Reused or
        # pre-existing instance dirs still hold a cost_report.json from whenever they
        # were built, and counting it here inflates every later sweep's total.
        inst_dir = row.get("inst_dir")
        if inst_dir and row.get("inst_created"):
            inst_key = str(Path(inst_dir).resolve())
            if inst_key in counted_inst_dirs:
                inst_dir = None
            else:
                counted_inst_dirs.add(inst_key)
        else:
            inst_dir = None
        if inst_dir:
            inst_cost_path = Path(inst_dir) / "cost_report.json"
            if inst_cost_path.exists():
                with open(inst_cost_path) as f:
                    inst_report = json.load(f)
                if inst_report.get("total_calls", 0) > 0:
                    if "instance_creation" not in by_short_name[short_name]:
                        by_short_name[short_name]["instance_creation"] = _empty_cost_bucket()
                    _add_report_to_bucket(by_short_name[short_name]["instance_creation"], inst_report)

        # --- Eval cost (run_benchmark subprocess) ---
        if not artifact_dirs:
            continue

        mode_bucket = _empty_cost_bucket()
        for artifact_dir in artifact_dirs:
            cost_path = Path(artifact_dir) / "cost_report.json"
            if not cost_path.exists():
                continue
            with open(cost_path) as f:
                report = json.load(f)
            _add_report_to_bucket(mode_bucket, report)
            # Prefer the mode stored in the report; fall back to track from registry
            if not track:
                track = report.get("test_mode", "unknown")

        if mode_bucket["total_calls"] == 0:
            continue

        mode_key = track or "unknown"
        if mode_key in by_short_name[short_name]:
            # Multiple artifact dirs for the same mode (e.g. several model runs) — merge
            _add_report_to_bucket(by_short_name[short_name][mode_key], mode_bucket)
        else:
            by_short_name[short_name][mode_key] = mode_bucket

    # Remove empty short names (no cost data at all)
    by_short_name = {sn: modes for sn, modes in by_short_name.items() if modes}

    if not by_short_name:
        LOGGER.info("No cost_report.json files found in artifact directories — skipping cost summary.")
        return

    # -----------------------------------------------------------------------
    # Build JSON summary
    # -----------------------------------------------------------------------
    grand_total = sum(
        bucket["total_cost_usd"]
        for modes in by_short_name.values()
        for bucket in modes.values()
    )

    def _bucket_for_json(b: dict) -> dict:
        out = {k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in b.items() if k != "by_stage"}
        out["by_stage"] = {
            stage: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sd.items()}
            for stage, sd in b.get("by_stage", {}).items()
        }
        return out

    summary = {
        "ablations": {
            sn: {
                "description": descriptions.get(sn, ""),
                "modes": {mode: _bucket_for_json(bucket) for mode, bucket in modes.items()},
            }
            for sn, modes in by_short_name.items()
        },
        "grand_total_cost_usd": round(grand_total, 6),
        "pricing_note": PRICING_NOTE,
    }
    summary_json_path = output_base / "ablation_cost_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # -----------------------------------------------------------------------
    # Build Markdown summary — one section per ablation type, one subsection
    # per mode, with per-stage breakdown table within each subsection.
    # -----------------------------------------------------------------------
    # Modes rendered in this canonical order; "instance_creation" always first.
    EVAL_MODE_ORDER = ["pairwise", "ranking", "pointwise"]
    STAGE_TABLE_HEADER = (
        "| Stage | Calls | Input | Cached input | Cache creation | Output | Cost (USD) |\n"
        "|-------|------:|------:|-------------:|---------------:|-------:|-----------:|"
    )

    def _stage_row(label: str, d: dict, bold: bool = False) -> str:
        wrap = "**" if bold else ""
        cost = d.get("total_cost_usd", 0.0)
        return (
            f"| {wrap}{label}{wrap} "
            f"| {d.get('total_calls', 0):,} "
            f"| {d.get('total_input_tokens', 0):,} "
            f"| {d.get('total_cached_input_tokens', 0):,} "
            f"| {d.get('total_cache_creation_tokens', 0):,} "
            f"| {d.get('total_output_tokens', 0):,} "
            f"| {wrap}${cost:.4f}{wrap} |"
        )

    def _write_mode_subsection(lines: list, mode_label: str, bucket: dict) -> None:
        """Append a ### subsection for one mode with a per-stage table."""
        lines.append(f"### {mode_label}")
        lines.append("")
        lines.append(STAGE_TABLE_HEADER)

        by_stage = bucket.get("by_stage", {})
        visible_stages = {
            s: d for s, d in by_stage.items()
            if s != "unknown" or d.get("total_cost_usd", 0) > 0
        }
        if visible_stages:
            # Sort stages by cost descending
            for stage, sd in sorted(visible_stages.items(),
                                    key=lambda kv: kv[1].get("total_cost_usd", 0),
                                    reverse=True):
                lines.append(_stage_row(f"`{stage}`", sd))
        else:
            # Old-format report (no by_stage) — show a single summary row
            lines.append(_stage_row("*(all stages)*", bucket))

        # Total row
        lines.append(_stage_row("Total", bucket, bold=True))

        # Per-model breakdown
        by_model = bucket.get("by_model", {})
        if by_model:
            inst_proc  = bucket.get("instances_processed")
            inst_total = bucket.get("instances_total")
            show_est   = (inst_proc is not None and inst_total is not None
                          and inst_total > inst_proc and inst_proc > 0)
            scale      = inst_total / inst_proc if show_est else None

            header = (
                "| Model | Calls | Input tok. | Cached tok. | Output tok. "
                "| Run cost | Est. full set |"
            )
            divider = (
                "|-------|------:|-----------:|------------:|-----------:"
                "|---------:|--------------:|"
            )
            lines += ["", header, divider]
            total_calls = total_input = total_cached = total_output = 0
            total_run_cost = 0.0
            for model_name, mb in sorted(by_model.items(),
                                         key=lambda kv: kv[1].get("cost_usd", 0),
                                         reverse=True):
                run_cost = mb.get("cost_usd", 0.0)
                est_str  = f"${run_cost * scale:.4f}" if scale else "—"
                lines.append(
                    f"| `{model_name}` "
                    f"| {mb.get('call_count', 0):,} "
                    f"| {mb.get('input_tokens', 0):,} "
                    f"| {mb.get('cached_input_tokens', 0):,} "
                    f"| {mb.get('output_tokens', 0):,} "
                    f"| ${run_cost:.4f} "
                    f"| {est_str} |"
                )
                total_calls    += mb.get("call_count", 0)
                total_input    += mb.get("input_tokens", 0)
                total_cached   += mb.get("cached_input_tokens", 0)
                total_output   += mb.get("output_tokens", 0)
                total_run_cost += run_cost
            total_est_str = f"${total_run_cost * scale:.4f}" if scale else "—"
            lines.append(
                f"| **Total** "
                f"| **{total_calls:,}** "
                f"| **{total_input:,}** "
                f"| **{total_cached:,}** "
                f"| **{total_output:,}** "
                f"| **${total_run_cost:.4f}** "
                f"| **{total_est_str}** |"
            )

        # Compact instance-count footer
        inst_proc  = bucket.get("instances_processed")
        inst_total = bucket.get("instances_total")
        if inst_proc is not None:
            if inst_total is not None:
                lines.append(f"_Ran on {inst_proc:,} / {inst_total:,} examples — "
                             f"est. full set costs shown per model above._")
            else:
                lines.append(f"_Ran on {inst_proc:,} examples._")
        lines.append("")

    md_lines = [
        "# Ablation Cost Summary",
        "",
        f"> {PRICING_NOTE}",
        "",
    ]

    # Sort sections by total cost descending
    sorted_sections = sorted(
        by_short_name.items(),
        key=lambda kv: sum(b["total_cost_usd"] for b in kv[1].values()),
        reverse=True,
    )

    for short_name, modes in sorted_sections:
        section_total_cost = sum(b["total_cost_usd"] for b in modes.values())

        desc = descriptions.get(short_name, "")
        md_lines += [
            f"## `{short_name}`",
            f"_{desc}_" if desc else "",
            "",
        ]

        # instance_creation first
        if "instance_creation" in modes:
            _write_mode_subsection(md_lines, "instance_creation", modes["instance_creation"])

        # canonical eval modes
        for mode in EVAL_MODE_ORDER:
            if mode in modes:
                _write_mode_subsection(md_lines, mode, modes[mode])

        # any other modes not in the canonical lists
        for mode, bucket in modes.items():
            if mode not in EVAL_MODE_ORDER and mode != "instance_creation":
                _write_mode_subsection(md_lines, mode, bucket)

        md_lines.append(f"**Ablation Total: ${section_total_cost:.4f}**")
        md_lines.append("")

    md_lines += [
        "---",
        f"**Grand Total: ${grand_total:.4f}**",
        "",
    ]

    summary_md_path = output_base / "ablation_cost_summary.md"
    with open(summary_md_path, "w") as f:
        f.write("\n".join(md_lines))

    LOGGER.info(f"Cost summary saved to: {summary_md_path}")
    LOGGER.info(f"Grand total estimated cost: ${grand_total:.4f}")


def _generate_cross_ablation_report(artifact_dirs: list[str], report_path: Path) -> None:
    """Call generate_report.py to produce a single cross-ablation Markdown report."""
    tmp_dirs_config = report_path.parent / "_tmp_dirs_config.yaml"
    with open(tmp_dirs_config, "w") as f:
        yaml.dump({"dirs": artifact_dirs}, f)

    rc = _run_subprocess(
        [sys.executable, _GENERATE_REPORT_SCRIPT, "--config", tmp_dirs_config,
         "--output", report_path],
        "generate_cross_ablation_report",
    )
    if tmp_dirs_config.exists():
        tmp_dirs_config.unlink()

    if rc == 0:
        LOGGER.info(f"Cross-ablation report saved to: {report_path}")
    else:
        LOGGER.warning("Report generation failed.")


# ---------------------------------------------------------------------------
# Batch mode helpers
# ---------------------------------------------------------------------------

def _build_batch_state(
    summary_rows: list[dict],
    output_base: Path,
    instances_base: Path,
) -> dict:
    """Build the batch state dict from completed (submitted) summary rows."""
    from novelty_eval.ablation.batch_manager import scan_artifact_dirs_for_batches

    ablations_state: dict = {}
    for row in summary_rows:
        ablation_name = row["ablation"]
        artifact_dirs = row.get("artifact_dirs", [])

        # Group batches by artifact_dir → one "run" per artifact_dir
        batches_by_artifact: dict[str, list[dict]] = {}
        for b in scan_artifact_dirs_for_batches(artifact_dirs):
            batches_by_artifact.setdefault(b["artifact_dir"], []).append(b)

        runs = []
        for artifact_dir, batches in batches_by_artifact.items():
            run_dir = Path(artifact_dir)
            # Infer run_name from the model subdir (parent.parent of artifact_dir)
            run_name = run_dir.parent.parent.name if run_dir.parent.name == "accuracy_test_artifacts" else run_dir.parent.name
            runs.append({
                "run_name": run_name,
                "artifact_dir": artifact_dir,
                "status": "pending",
                "batches": [
                    {
                        "batch_id": b["batch_id"],
                        "provider": b["provider"],
                        "run_dir": b["run_dir"],
                        "batch_info_file": b["batch_info_file"],
                        "status": "pending",
                    }
                    for b in batches
                ],
            })

        ablations_state[ablation_name] = {
            "description": row.get("description", ""),
            "status": "pending" if runs else row.get("status", "failed"),
            "inst_dir": row.get("inst_dir"),
            "inst_created": row.get("inst_created", False),
            "runs": runs,
        }

    return {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "output_base": str(output_base),
        "instances_base": str(instances_base),
        "ablations": ablations_state,
    }


def _run_reprocess_report(run: dict, ablation_output_dir: Path) -> bool:
    """Re-generate accuracy_report.txt for a completed run using run_benchmark.py's update_report mode.

    Returns True on success.
    """
    import tempfile
    import yaml as _yaml

    artifact_dir = run["artifact_dir"]
    # _run_config.yaml lives two levels above the artifact dir:
    # {ablation_output_dir}/{run_name}/accuracy_test_artifacts/{timestamp}
    run_dir = Path(artifact_dir).parent.parent
    run_config_path = run_dir / "_run_config.yaml"

    if not run_config_path.exists():
        LOGGER.warning(f"_run_config.yaml not found at {run_config_path}; skipping reprocess for {run['run_name']}")
        return False

    with open(run_config_path) as f:
        cfg = _yaml.safe_load(f) or {}

    cfg["update_report"] = artifact_dir
    cfg["use_batch_api"] = False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        _yaml.dump(cfg, tmp, allow_unicode=True)
        tmp_path = tmp.name

    _ACCURACY_TEST_SCRIPT = _PROJECT_ROOT / "src" / "novelty_eval" / "run_benchmark.py"
    rc = _run_subprocess(
        [sys.executable, _ACCURACY_TEST_SCRIPT, "--config", tmp_path],
        f"reprocess:{run['run_name']}",
    )
    os.unlink(tmp_path)
    return rc == 0


def _print_poll_status_table(state: dict) -> None:
    """Print a markdown-style status table for all batches in the state."""
    lines = [
        "\n## Batch Poll Status\n",
        "| Ablation | Run | Batch ID | Provider | Status |",
        "|----------|-----|----------|----------|--------|",
    ]
    for ablation_name, entry in state["ablations"].items():
        for run in entry.get("runs", []):
            for batch in run.get("batches", []):
                lines.append(
                    f"| {ablation_name} | {run['run_name']} "
                    f"| `{batch['batch_id'][:16]}…` | {batch['provider']} | {batch['status']} |"
                )
    LOGGER.info("\n".join(lines))

    # Summary counts
    all_batches = [
        b for e in state["ablations"].values()
        for r in e.get("runs", [])
        for b in r.get("batches", [])
    ]
    pending = sum(1 for b in all_batches if b["status"] == "pending")
    completed = sum(1 for b in all_batches if b["status"] == "completed")
    failed = sum(1 for b in all_batches if b["status"] == "failed")
    LOGGER.info(f"Batches: {completed} completed, {pending} pending, {failed} failed (total {len(all_batches)})")


def _rebuild_summary_rows_from_state(state: dict) -> list[dict]:
    """Reconstruct summary_rows format for _aggregate_cost_reports()."""
    rows = []
    for ablation_name, entry in state["ablations"].items():
        artifact_dirs = [
            run["artifact_dir"]
            for run in entry.get("runs", [])
            if run.get("status") in ("completed", "partial")
        ]
        rows.append({
            "ablation": ablation_name,
            "description": entry.get("description", ""),
            "status": entry.get("status", "unknown"),
            "artifact_dirs": artifact_dirs,
            "inst_dir": entry.get("inst_dir"),
            "inst_created": entry.get("inst_created", False),
        })
    return rows


def _run_poll_mode(state_path: Path) -> None:
    """Poll pending batches, retrieve completed results, and generate reports."""
    from novelty_eval.ablation.batch_manager import (
        check_batch_status,
        load_state,
        run_has_any_results,
        run_is_reportable,
        write_state,
        _compute_run_status,
        _compute_ablation_status,
    )
    from novelty_eval.retrieve_batch_results import retrieve_batch_results

    state = load_state(state_path)
    output_base = Path(state["output_base"])

    LOGGER.info(f"Polling state file: {state_path}")

    for ablation_name, entry in state["ablations"].items():
        for run in entry.get("runs", []):
            run_dir_for_cost = None
            newly_retrieved = False
            from cost_tracker import GLOBAL_COST_TRACKER
            if GLOBAL_COST_TRACKER:
                GLOBAL_COST_TRACKER.reset()

            for batch in run.get("batches", []):
                if batch["status"] in ("completed", "failed"):
                    continue

                new_status = check_batch_status(batch["batch_id"], batch["provider"])
                LOGGER.info(f"  {batch['batch_id'][:20]} ({batch['provider']}): {batch['status']} → {new_status}")
                batch["status"] = new_status

                if new_status == "completed":
                    # Retrieve results → writes scores.json to run_dir
                    run_dir = batch["run_dir"]
                    run_dir_for_cost = run_dir
                    scores_path = Path(run_dir) / "scores.json"
                    if scores_path.exists():
                        LOGGER.info(f"  scores.json already present at {run_dir}, skipping retrieval")
                    else:
                        results = retrieve_batch_results(
                            batch["batch_id"], run_dir, provider=batch["provider"]
                        )
                        if results:
                            newly_retrieved = True
                            with open(scores_path, "w") as f:
                                json.dump(results, f, indent=2)
                            LOGGER.info(f"  Wrote scores.json to {run_dir}")
                        else:
                            LOGGER.warning(f"  Retrieval returned no results for {batch['batch_id']}")
                            batch["status"] = "failed"

            # Save cost report if we retrieved anything for this run
            if newly_retrieved and GLOBAL_COST_TRACKER and run_dir_for_cost:
                cost_report = GLOBAL_COST_TRACKER.get_report()
                # cost_report.json belongs in the artifact_dir, which is parent of run_dir
                cost_path = Path(run_dir_for_cost).parent / "cost_report.json"
                with open(cost_path, "w") as f:
                    json.dump(cost_report, f, indent=2)
                LOGGER.info(f"  Wrote cost_report.json to {cost_path.parent}")

            # Update run status and trigger reprocess when all batches settled
            run["status"] = _compute_run_status(run)
            if run_is_reportable(run) and run_has_any_results(run):
                if not run.get("report_generated"):
                    LOGGER.info(f"Generating accuracy report for run: {run['run_name']}")
                    ablation_output_dir = Path(run["artifact_dir"]).parent.parent.parent
                    success = _run_reprocess_report(run, ablation_output_dir)
                    if success:
                        run["report_generated"] = True
                else:
                    LOGGER.info(f"Accuracy report already generated for {run['run_name']}, skipping")

        entry["status"] = _compute_ablation_status(entry)
        if entry["status"] in ("completed", "partial") and not entry.get("sweep_report_generated"):
            ablation_artifact_dirs = [
                run["artifact_dir"]
                for run in entry.get("runs", [])
                if run.get("status") in ("completed", "partial")
            ]
            if ablation_artifact_dirs:
                ablation_output_dir = Path(ablation_artifact_dirs[0]).parent.parent.parent
                sweep_report_path = ablation_output_dir / "sweep_comparison_report.md"
                LOGGER.info(f"Regenerating sweep_comparison_report.md for {ablation_name}")
                _generate_cross_ablation_report(ablation_artifact_dirs, sweep_report_path)
                entry["sweep_report_generated"] = True

    write_state(state, state_path)
    _print_poll_status_table(state)

    # When all ablations have reached a terminal state, generate the full reports
    all_terminal = all(
        e["status"] in ("completed", "partial", "failed")
        for e in state["ablations"].values()
    )
    if all_terminal:
        LOGGER.info("All ablations reached terminal state — generating cross-ablation reports.")
        all_artifact_dirs = [
            run["artifact_dir"]
            for e in state["ablations"].values()
            for run in e.get("runs", [])
            if run.get("status") in ("completed", "partial")
        ]
        if all_artifact_dirs:
            report_path = output_base / "ablation_comparison_report.md"
            _generate_cross_ablation_report(all_artifact_dirs, report_path)
            summary_rows = _rebuild_summary_rows_from_state(state)
            _aggregate_cost_reports(summary_rows, output_base)
        else:
            LOGGER.warning("No completed runs found; skipping report generation.")
    else:
        pending_count = sum(
            1 for e in state["ablations"].values()
            for r in e.get("runs", [])
            for b in r.get("batches", [])
            if b["status"] == "pending"
        )
        LOGGER.info(f"{pending_count} batch(es) still pending. Re-run --poll to check again.")
        LOGGER.info(f"State file: {state_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global LOGGER

    parser = argparse.ArgumentParser(
        description="Run the ablation study for comparative idea quality evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ALL_ABLATIONS = list(_ABLATION_REGISTRY.keys())
    parser.add_argument(
        "--ablations",
        type=str,
        default=",".join(ALL_ABLATIONS),
        help=f"Comma-separated ablations to run. Options: {', '.join(ALL_ABLATIONS)}. Default: all.",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default=None,
        help="Comma-separated tracks to filter by (e.g. pairwise,ranking).",
    )
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="Skip instance creation and reuse existing instances (requires --instances-dir).",
    )
    parser.add_argument(
        "--instances-dir",
        type=str,
        default=str(_PROJECT_ROOT / "output" / "benchmark_instances" / "ablation"),
        help="Base directory where created instances are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for sweep outputs. Defaults to output/ablation_sweeps/<timestamp>.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gpt-5.4",
        help="Comma-separated LLM engines to evaluate (default: gpt-5.4). E.g. --models gpt-5.2,gpt-5.4",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Number of evaluation runs per ablation. Overrides the per-ablation n_runs in ablations.yaml. Defaults to the YAML value, or 1 if not set there.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Limit evaluation to the first N instances per ablation. Useful for smoke tests.",
    )
    parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="Run all model configs within each sweep in parallel.",
    )
    parser.add_argument(
        "--max-parallel-models",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap the number of models running concurrently when --parallel-models is set. "
            "Useful to avoid Anthropic's concurrent-connection rate limit. "
            "E.g. --max-parallel-models 3 runs at most 3 models at a time."
        ),
    )
    parser.add_argument(
        "--parallel-ablations",
        action="store_true",
        help="Run all ablations in parallel (each sweep as its own subprocess).",
    )
    parser.add_argument(
        "--max-parallel-retrievals",
        type=int,
        default=1,
        help="Max number of retrieval subprocesses running concurrently (default: 1).",
    )
    parser.add_argument(
        "--merge-dirs",
        type=str,
        default=None,
        help=(
            "Comma-separated list of previous ablation output directories to merge into a "
            "single unified report. Skips all instance creation and sweep execution."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Submit LLM evaluation calls as async batch jobs (OpenAI or Anthropic batch APIs). "
            "Writes ablation_batch_state.json; use --poll to retrieve results later."
        ),
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll pending batches from a state file and retrieve completed results.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Path to ablation_batch_state.json. Required for --poll; auto-named for --batch.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Poll mode: check/retrieve pending batches from an existing state file
    # -----------------------------------------------------------------------
    if args.poll:
        state_path = Path(args.state) if args.state else None
        if state_path is None:
            parser.error("--poll requires --state <path/to/ablation_batch_state.json>")
        if not state_path.exists():
            parser.error(f"State file not found: {state_path}")
        output_base_for_logger = state_path.parent
        os.environ.setdefault("OUTPUT_DIR", str(output_base_for_logger))
        LOGGER = setup_logger(str(output_base_for_logger), console_level="INFO")
        _run_poll_mode(state_path)
        return

    # -----------------------------------------------------------------------
    # Merge mode: collect results from existing runs and generate a report
    # -----------------------------------------------------------------------
    if args.merge_dirs:
        merge_paths = [Path(p.strip()) for p in args.merge_dirs.split(",") if p.strip()]
        merge_output = merge_paths[0].parent / f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        merge_output.mkdir(parents=True, exist_ok=True)
        LOGGER = setup_logger(str(merge_output), console_level="INFO")

        all_artifact_dirs: list[str] = []
        for p in merge_paths:
            if not p.is_dir():
                LOGGER.warning(f"Skipping non-existent directory: {p}")
                continue
            dirs = _collect_artifact_dirs_from_run(p)
            LOGGER.info(f"{p.name}: found {len(dirs)} artifact dir(s)")
            all_artifact_dirs.extend(dirs)

        if not all_artifact_dirs:
            LOGGER.error("No artifact directories found across provided runs.")
            sys.exit(1)

        report_path = merge_output / "merged_ablation_report.md"
        _generate_cross_ablation_report(all_artifact_dirs, report_path)
        LOGGER.info(f"Merged report written to: {report_path}")
        return

    # Parse models and ablation lists
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    raw_requested = [a.strip() for a in args.ablations.split(",") if a.strip()]
    allowed_tracks = [t.strip() for t in args.tracks.split(",") if t.strip()] if args.tracks else None
    
    requested = []
    unknown = []
    for r in raw_requested:
        if r in _ABLATION_REGISTRY:
            # Full name provided (e.g. 'pairwise_current'). Filter by track if requested.
            if allowed_tracks:
                if _ABLATION_REGISTRY[r].get("track") in allowed_tracks:
                    requested.append(r)
            else:
                requested.append(r)
        else:
            # Check if it's a short name that needs track expansion (e.g. 'current' -> 'pairwise_current')
            matches = [full_name for full_name in _ABLATION_REGISTRY if full_name.endswith(f"_{r}")]
            if allowed_tracks:
                matches = [m for m in matches if _ABLATION_REGISTRY[m].get("track") in allowed_tracks]
            
            if matches:
                requested.extend(matches)
            else:
                unknown.append(r)

    if unknown:
        parser.error(f"Unknown ablations: {unknown}. Valid options (full or short): {list(_ABLATION_REGISTRY.keys())}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) if args.output_dir else (
        _PROJECT_ROOT / "output" / "ablation_sweeps" / timestamp
    )
    instances_base = Path(args.instances_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Re-initialize logger now that we have the real output directory.
    # Also propagate OUTPUT_DIR to the environment so all child subprocesses
    # (retrieve_candidates.py, run_benchmark.py, etc.) log into the same dir.
    os.environ["OUTPUT_DIR"] = str(output_base)
    LOGGER = setup_logger(str(output_base), console_level="INFO")

    LOGGER.info(f"Output directory: {output_base}")
    LOGGER.info(f"Running ablations: {requested}")

    retrieval_semaphore = threading.Semaphore(args.max_parallel_retrievals)

    # Dictionary to coordinate track-to-track dependencies (e.g. pointwise deriving from pairwise)
    ablation_events = {a: threading.Event() for a in requested}
    ablation_inst_dirs: dict[str, str] = {}

    # -----------------------------------------------------------------------
    # Run each ablation
    # -----------------------------------------------------------------------
    summary_rows = []
    all_artifact_dirs: list[str] = []

    def _run_one_ablation(ablation: str) -> dict:
        """Run a single ablation end-to-end; returns a summary row dict."""
        entry = _ABLATION_REGISTRY[ablation]
        track_name = entry.get("track")
        track = _CONFIG_DATA["tracks"].get(track_name, {})

        # Resolve configurations via deep merge: defaults -> track -> ablation
        instance_cfg = _deep_merge(_CONFIG_DATA["defaults"].get("instance", {}), track.get("instance", {}))
        instance_cfg = _deep_merge(instance_cfg, entry.get("instance", {})) if isinstance(entry.get("instance"), dict) else instance_cfg

        sweep_cfg = _deep_merge(_CONFIG_DATA["defaults"].get("sweep", {}), track.get("sweep", {}))
        sweep_cfg = _deep_merge(sweep_cfg, {"base_config": entry.get("sweep", {})})

        LOGGER.info(f"{'='*60}")
        LOGGER.info(f"Starting: {ablation} — {_ABLATION_DESCRIPTIONS.get(ablation, ablation)}")
        LOGGER.info(f"{'='*60}")

        ablation_output_dir = output_base / ablation

        # Filter models for OpenAI-only ablations
        effective_models = models
        if entry.get("openai_only"):
            effective_models = [
                m for m in models
                if m.startswith("gpt-") or (len(m) >= 2 and m[0] == "o" and m[1].isdigit())
            ]
            skipped = [m for m in models if m not in effective_models]
            for m in skipped:
                LOGGER.warning(f"Skipping model '{m}' for '{ablation}': ablation is OpenAI-only.")
            if not effective_models:
                LOGGER.error(f"No OpenAI models available for '{ablation}'. Skipping ablation.")
                return {"ablation": ablation, "description": _ABLATION_DESCRIPTIONS.get(ablation, ablation),
                        "status": "SKIPPED (no OpenAI models)", "artifact_dirs": []}

        # Step 1: create instances (if needed)
        instances_yaml: Path | None = None
        inst_dir: Path | None = None  # tracks where instances (and cost_report.json) were written
        inst_created = False  # True only if THIS run created instances (reused dirs carry stale cost_report.json)

        # Coordinate instance extraction if requested to derive from another track
        derive_track = instance_cfg.get("derive_pointwise_from_track") if isinstance(instance_cfg, dict) else None
        if derive_track:
            if args.skip_create:
                inst_dir = _find_instances_dir_by_meta(instances_base, ablation)
                if inst_dir:
                    instances_yaml = _find_instances_yaml(inst_dir)
                    if instances_yaml:
                        LOGGER.info(f"Reusing existing instances for '{ablation}': {instances_yaml}")

            if instances_yaml is None and not args.skip_create:
                # Resolve source ablation name by swapping the track prefix
                source_ablation = ablation.replace(f"{track_name}_", f"{derive_track}_")
                if not source_ablation.startswith(f"{derive_track}_"):
                    source_ablation = f"{derive_track}_{ablation}"

                if source_ablation in ablation_events:
                    LOGGER.info(f"Waiting for instances from source ablation '{source_ablation}'...")
                    ablation_events[source_ablation].wait()
                    source_inst_dir_str = ablation_inst_dirs.get(source_ablation)
                    source_inst_dir = Path(source_inst_dir_str) if source_inst_dir_str else None
                else:
                    source_inst_dir = _find_instances_dir_by_meta(instances_base, source_ablation)

                source_yaml = None
                if source_inst_dir:
                    source_yaml = _find_instances_yaml(source_inst_dir)

                if source_yaml is None:
                    # Fallback: use the source track's configured test_inputs directly
                    source_track_cfg = _CONFIG_DATA["tracks"].get(derive_track, {})
                    fallback_str = source_track_cfg.get("sweep", {}).get("test_inputs")
                    if fallback_str and fallback_str != "__PLACEHOLDER__":
                        candidate = Path(fallback_str)
                        if not candidate.is_absolute():
                            candidate = _PROJECT_ROOT / candidate
                        if candidate.exists():
                            source_yaml = candidate
                            LOGGER.info(
                                f"No instance dir found for '{source_ablation}'; "
                                f"using track fallback: {source_yaml}"
                            )

                if source_yaml:
                    LOGGER.info(f"Deriving pointwise instances from: {source_yaml}")
                    # Patch config to use the derivation mode
                    derive_cfg = deepcopy(instance_cfg)
                    derive_cfg["derive_pointwise_from_pairwise"] = str(source_yaml)
                    derive_cfg["output_dir"] = str(instances_base / ablation)

                    inst_dir = _create_instances(derive_cfg, instances_base, ablation)
                    if inst_dir:
                        inst_created = True
                        instances_yaml = _find_instances_yaml(inst_dir)

                if instances_yaml is None:
                    LOGGER.error(f"FAILED: could not derive instances from '{source_ablation}'")
                    return {"ablation": ablation, "description": _ABLATION_DESCRIPTIONS.get(ablation, ablation),
                            "status": "FAILED (instance derivation)", "artifact_dirs": []}

        elif isinstance(entry.get("instance"), str):
            # String value = name of another ablation whose instances to reuse
            source_ablation = entry["instance"]
            inst_dir = _find_instances_dir_by_meta(instances_base, source_ablation)
            if inst_dir:
                instances_yaml = _find_instances_yaml(inst_dir)
            if instances_yaml is None:
                LOGGER.warning(
                    f"No existing instances found for source ablation '{source_ablation}' "
                    f"(needed by '{ablation}'). Run '{source_ablation}' first, or use --skip-create=False."
                )

        elif entry.get("instance") is not None and not args.skip_create:
            inst_dir = _create_instances(instance_cfg, instances_base, ablation)
            inst_created = inst_dir is not None
            if inst_dir is None:
                LOGGER.error(f"FAILED: instance creation for '{ablation}'")
                return {"ablation": ablation, "description": _ABLATION_DESCRIPTIONS.get(ablation, ablation),
                        "status": "FAILED (instance creation)", "artifact_dirs": []}
            instances_yaml = _find_instances_yaml(inst_dir)
            if instances_yaml is None:
                LOGGER.error(f"FAILED: could not locate instances YAML for '{ablation}'")
                return {"ablation": ablation, "description": _ABLATION_DESCRIPTIONS.get(ablation, ablation),
                        "status": "FAILED (instances YAML not found)", "artifact_dirs": []}
            LOGGER.info(f"Instances YAML: {instances_yaml}")

        elif entry.get("instance") is not None and args.skip_create:
            inst_dir = _find_instances_dir_by_meta(instances_base, ablation)
            if inst_dir:
                instances_yaml = _find_instances_yaml(inst_dir)
            if instances_yaml is None:
                LOGGER.warning(f"--skip-create set but no existing instances found for '{ablation}'")

        # Fallback: if instances_yaml is still None, try to read it from the resolved sweep config.
        # test_inputs from the track baseline lives at the top level of sweep_cfg; the ablation's
        # own sweep overrides are nested under base_config.
        if instances_yaml is None:
            tmp_path = sweep_cfg.get("test_inputs") or sweep_cfg.get("base_config", {}).get("test_inputs")
            if tmp_path and tmp_path != "__PLACEHOLDER__":
                instances_yaml = Path(tmp_path)
                if inst_dir is None:
                    # Record the directory containing the fallback instances
                    inst_dir = instances_yaml.parent
                    if inst_dir.name == "manipulated" or inst_dir.name == "raw_abstract":
                        inst_dir = inst_dir.parent

        # Signal that instances are ready for any dependent tracks (e.g. pointwise)
        if inst_dir:
            ablation_inst_dirs[ablation] = str(inst_dir)
        if ablation in ablation_events:
            ablation_events[ablation].set()

        # Step 2: run retrieval (if requested)
        retrieval_cache_file: Path | None = None
        if entry.get("retrieve") and instances_yaml is not None:
            retrieval_cfg = _deep_merge(_CONFIG_DATA["defaults"].get("retrieval", {}), entry.get("retrieval", {}))
            LOGGER.info(f"Waiting for retrieval slot (max_parallel_retrievals={args.max_parallel_retrievals})...")
            with retrieval_semaphore:
                retrieval_cache_file = _run_retrieval(instances_yaml, ablation, retrieval_cfg, num_instances=args.num_instances)
            if retrieval_cache_file is None:
                LOGGER.error(f"FAILED: retrieval for '{ablation}'")
                return {"ablation": ablation, "description": _ABLATION_DESCRIPTIONS.get(ablation, ablation),
                        "status": "FAILED (retrieval)", "artifact_dirs": []}

        # Optional preprocessing step (e.g. strip fields before sweep)
        if entry.get("preprocess") == "strip_eval" and instances_yaml is not None:
            instances_yaml = _strip_eval_from_instances_yaml(
                instances_yaml, ablation_output_dir / "stripped_instances"
            )
            LOGGER.info(f"Stripped **evaluation** field from instances → {instances_yaml}")

        # Write metadata so generate_report.py can identify this ablation without
        # fragile directory-name parsing.  Only the stable key fields are stored;
        # descriptions are always resolved live from ablations.yaml.
        meta_path = ablation_output_dir / "_ablation_meta.json"
        ablation_output_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as _f:
            json.dump({
                "ablation": ablation,
                "track": track_name,
            }, _f, indent=2)

        # Step 3: patch sweep config and run
        patched_cfg = _patch_and_write_sweep_config(
            sweep_cfg,
            instances_yaml,
            ablation_output_dir,
            effective_models,
            ablation,
            args.n_runs,
            args.num_instances,
            retrieval_cache_file=retrieval_cache_file,
            force_batch=getattr(args, "batch", False),
        )

        artifact_dirs = _run_sweep(patched_cfg, ablation_output_dir, ablation,
                                   parallel_models=args.parallel_models,
                                   max_parallel_models=getattr(args, "max_parallel_models", None))

        if artifact_dirs:
            return {"ablation": ablation, "description": _ABLATION_DESCRIPTIONS.get(ablation, ablation),
                    "status": "OK", "artifact_dirs": artifact_dirs,
                    "inst_dir": str(inst_dir) if inst_dir else None,
                    "inst_created": inst_created}
        return {"ablation": ablation, "description": _ABLATION_DESCRIPTIONS.get(ablation, ablation),
                "status": "FAILED (sweep)", "artifact_dirs": [],
                "inst_dir": str(inst_dir) if inst_dir else None,
                "inst_created": inst_created}

    if args.parallel_ablations:
        LOGGER.info(f"Running {len(requested)} ablation(s) in PARALLEL.")
        with ThreadPoolExecutor(max_workers=len(requested)) as executor:
            future_to_ablation = {executor.submit(_run_one_ablation, a): a for a in requested}
            rows_by_ablation = {}
            for future in as_completed(future_to_ablation):
                rows_by_ablation[future_to_ablation[future]] = future.result()
        # Preserve original order
        summary_rows = [rows_by_ablation[a] for a in requested]
    else:
        LOGGER.info(f"Running {len(requested)} ablation(s) sequentially.")
        summary_rows = [_run_one_ablation(a) for a in requested]

    for row in summary_rows:
        all_artifact_dirs.extend(row["artifact_dirs"])

    # -----------------------------------------------------------------------
    # Write run summary JSON
    # -----------------------------------------------------------------------
    summary_path = output_base / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"timestamp": timestamp, "runs": summary_rows}, f, indent=2)
    LOGGER.info(f"Run summary: {summary_path}")

    # -----------------------------------------------------------------------
    # Batch mode: write state file and exit (no sync reports yet)
    # -----------------------------------------------------------------------
    if args.batch:
        from novelty_eval.ablation.batch_manager import write_state
        batch_state = _build_batch_state(summary_rows, output_base, instances_base)
        state_path = Path(args.state) if args.state else output_base / "ablation_batch_state.json"
        write_state(batch_state, state_path)
        LOGGER.info(f"Batch state file: {state_path}")
        total_batches = sum(
            len(b)
            for e in batch_state["ablations"].values()
            for r in e.get("runs", [])
            for b in [r.get("batches", [])]
        )
        LOGGER.info(f"Submitted batches for {len(summary_rows)} ablation(s). Total batch job(s): {total_batches}")
        LOGGER.info(f"Poll when ready: python src/novelty_eval/ablation/run_ablations.py --poll --state {state_path}")
        return

    # -----------------------------------------------------------------------
    # Generate cross-ablation report
    # -----------------------------------------------------------------------
    if all_artifact_dirs:
        report_path = output_base / "ablation_comparison_report.md"
        _generate_cross_ablation_report(all_artifact_dirs, report_path)
    else:
        LOGGER.warning("No successful runs; skipping report generation.")

    # -----------------------------------------------------------------------
    # Generate retrieval impact comparison reports
    # For every retrieval ablation that succeeded, compare it against the
    # corresponding baseline (current) ablation within this run.
    # -----------------------------------------------------------------------
    _generate_retrieval_impact_reports(output_base, summary_rows)

    # -----------------------------------------------------------------------
    # Aggregate cost reports from all ablation artifact directories
    # -----------------------------------------------------------------------
    _aggregate_cost_reports(summary_rows, output_base)

    # -----------------------------------------------------------------------
    # Print final Markdown table to stdout and write to file
    # -----------------------------------------------------------------------
    md_lines = [
        "## Ablation Study Summary\n",
        f"| Ablation | Description | Status | Artifacts |",
        f"|----------|-------------|--------|-----------|",
    ]
    for row in summary_rows:
        dirs_str = "<br>".join(row["artifact_dirs"]) if row["artifact_dirs"] else "—"
        md_lines.append(f"| {row['ablation']} | {row['description']} | {row['status']} | {dirs_str} |")

    md_table = "\n".join(md_lines) + "\n"
    LOGGER.info(f"\n{md_table}")

    summary_md_path = output_base / "ablation_study_summary.md"
    with open(summary_md_path, "w") as f:
        f.write(md_table)
    LOGGER.info(f"Markdown summary written to: {summary_md_path}")


if __name__ == "__main__":
    main()
