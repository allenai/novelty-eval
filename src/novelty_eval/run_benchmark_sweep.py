#!/usr/bin/env python3
from __future__ import annotations
"""
Benchmark Sweep Script
======================
Reads a YAML sweep config, runs run_benchmark.py for each named configuration,
then calls generate_report.py to produce a cross-config comparison report.

Usage:
    python src/novelty_eval/run_benchmark_sweep.py --config src/novelty_eval/config/accuracy_sweep_config.yaml

Or with a custom output directory / report name:
    python src/novelty_eval/run_benchmark_sweep.py \
        --config src/novelty_eval/config/accuracy_sweep_config.yaml \
        --output-dir output/sweeps \
        --report-output sweep_comparison.md
"""

import argparse
import os
import sys
import subprocess
import yaml
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from copy import deepcopy

# Absolute path of this script's directory (src/novelty_eval/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is two levels up (src/novelty_eval -> src -> project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sweep_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _normalize_keys(d: dict) -> dict:
    """Convert all hyphenated keys to underscores for consistent lookups."""
    return {k.replace("-", "_"): v for k, v in d.items()}


def merge_config(base: dict, override: dict) -> dict:
    """Shallow-merge *override* on top of *base*; override wins.
    Keys are normalized (hyphens → underscores) so that YAML keys like
    ``retrieve-related-work`` match the Python lookups that use underscores."""
    merged = deepcopy(_normalize_keys(base))
    merged.update(_normalize_keys(override))
    return merged



def run_benchmark(run_cfg: dict, run_name: str, sweep_output_dir: str) -> str | None:
    """
    Invoke run_benchmark.py as a subprocess.
    Returns the path to the artifact directory on success, None on failure.
    """
    # Each run gets its own output_dir so artifacts don't clash
    run_output_dir = os.path.join(sweep_output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    cfg = deepcopy(run_cfg)
    cfg["output_dir"] = run_output_dir

    # Normalize sweep-specific key names to match run_benchmark.py's config schema:
    #   n_runs -> n
    if "n_runs" in cfg and "n" not in cfg:
        cfg["n"] = cfg.pop("n_runs")
    elif "n_runs" in cfg:
        cfg.pop("n_runs")

    # Drop keys that run_benchmark.py doesn't understand
    for unsupported in ("use_semantic_scholar",):
        cfg.pop(unsupported, None)

    # Write a temporary YAML config file for this run, since run_benchmark.py
    # only accepts --config <yaml_file> and not individual CLI flags.
    tmp_config_path = os.path.join(run_output_dir, "_run_config.yaml")
    with open(tmp_config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    benchmark_path = os.path.join(_THIS_DIR, "run_benchmark.py")
    cmd = [sys.executable, benchmark_path, "--config", tmp_config_path]
    print(f"\n[sweep] Running config '{run_name}'")
    print(f"[sweep] Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=os.environ.copy(), cwd=_PROJECT_ROOT)

    if result.returncode != 0:
        print(f"[sweep] ERROR: run_benchmark.py exited with code {result.returncode} for config '{run_name}'")
        return None

    # Discover the timestamped artifact sub-directory created by run_benchmark.py
    artifacts_base = os.path.join(run_output_dir, "accuracy_test_artifacts")
    if not os.path.isdir(artifacts_base):
        print(f"[sweep] WARNING: expected artifact directory not found at {artifacts_base}")
        return None

    # Pick the most-recently modified sub-directory
    sub_dirs = [
        os.path.join(artifacts_base, d)
        for d in os.listdir(artifacts_base)
        if os.path.isdir(os.path.join(artifacts_base, d))
    ]
    if not sub_dirs:
        print(f"[sweep] WARNING: no sub-directories found in {artifacts_base}")
        return None

    latest = max(sub_dirs, key=os.path.getmtime)
    print(f"[sweep] Artifacts for '{run_name}' saved in: {latest}")
    return latest


def run_generate_report(artifact_dirs: list[str], report_output_path: str) -> None:
    """
    Call generate_report.py to produce a cross-config comparison markdown report.
    Writes a temporary dirs_config.yaml so generate_report.py can consume it.
    """
    # Write a temporary YAML listing the artifact directories
    tmp_dirs_config = os.path.join(os.path.dirname(report_output_path), "_tmp_dirs_config.yaml")
    with open(tmp_dirs_config, "w") as f:
        yaml.dump({"dirs": artifact_dirs}, f)

    generate_report_path = os.path.join(_THIS_DIR, "analysis", "generate_report.py")
    cmd = [
        sys.executable,
        generate_report_path,
        "--config", tmp_dirs_config,
        "--output", report_output_path,
    ]
    print(f"\n[sweep] Generating comparison report: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=os.environ.copy(), cwd=_PROJECT_ROOT)

    # Clean up temp file
    if os.path.exists(tmp_dirs_config):
        os.remove(tmp_dirs_config)

    if result.returncode != 0:
        print(f"[sweep] ERROR: generate_report.py exited with code {result.returncode}")
    else:
        print(f"[sweep] Comparison report saved to: {report_output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a sweep of run_benchmark.py configurations and compare results."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the sweep YAML config file (e.g. scripts/accuracy_sweep_config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Base directory for all sweep outputs. "
            "Defaults to 'output/sweeps/<timestamp>' relative to the current working directory."
        ),
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default=None,
        help=(
            "Path for the final comparison report markdown file. "
            "Defaults to '<output-dir>/sweep_comparison_report.md'."
        ),
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Continue running remaining configs even if one fails.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run all configs in parallel (each as its own subprocess).",
    )
    parser.add_argument(
        "--max-parallel-models",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap the number of model configs that run concurrently when --parallel is set. "
            "Useful to avoid provider rate limits (e.g. Anthropic concurrent-connection cap). "
            "Defaults to len(tasks) (unlimited parallelism) when not specified."
        ),
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    sweep_cfg = load_sweep_config(args.config)
    base_config = sweep_cfg.get("base_config", {})
    run_configs = sweep_cfg.get("configs", [])
    description = sweep_cfg.get("description", "")

    if not run_configs:
        print("[sweep] No 'configs' entries found in sweep config. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "output", "sweeps", timestamp)
    os.makedirs(sweep_output_dir, exist_ok=True)
    print(f"[sweep] Sweep output directory: {sweep_output_dir}")

    if description:
        print(f"[sweep] Description: {description}")

    # Save the sweep config into the output dir for reproducibility
    with open(os.path.join(sweep_output_dir, "sweep_config.yaml"), "w") as f:
        yaml.dump(sweep_cfg, f, default_flow_style=False)

    # -----------------------------------------------------------------------
    # Build the list of (run_name, merged_cfg) tasks
    # -----------------------------------------------------------------------
    tasks = []
    for run_cfg_raw in run_configs:
        run_name = run_cfg_raw.get("name")
        if not run_name:
            print("[sweep] WARNING: a config entry is missing a 'name' field; skipping.")
            continue
        merged = merge_config(base_config, run_cfg_raw)
        merged.pop("name", None)
        tasks.append((run_name, merged))

    # -----------------------------------------------------------------------
    # Execute — parallel or sequential
    # -----------------------------------------------------------------------
    # results keyed by run_name to preserve insertion order later
    results: dict[str, str | None] = {}

    if not tasks:
        print("[sweep] No runnable configs found after parsing. Exiting.")
        sys.exit(1)

    if args.parallel:
        pool_size = args.max_parallel_models if args.max_parallel_models else len(tasks)
        print(f"[sweep] Running {len(tasks)} config(s) in PARALLEL (max {pool_size} at a time).\n")
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            future_to_name = {
                executor.submit(run_benchmark, cfg, name, sweep_output_dir): name
                for name, cfg in tasks
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    print(f"[sweep] ERROR: config '{name}' raised an exception: {exc}")
                    results[name] = None
    else:
        print(f"[sweep] Running {len(tasks)} config(s) sequentially.\n")
        for run_name, merged in tasks:
            artifact_dir = run_benchmark(merged, run_name, sweep_output_dir)
            results[run_name] = artifact_dir
            if artifact_dir is None and not args.skip_on_error:
                print(f"[sweep] Stopping sweep due to failure in config '{run_name}'. "
                      "Use --skip-on-error to continue despite failures.")
                # Fill remaining tasks as None
                for remaining_name, _ in tasks[tasks.index((run_name, merged)) + 1:]:
                    results[remaining_name] = None
                break

    # -----------------------------------------------------------------------
    # Collect ordered results
    # -----------------------------------------------------------------------
    artifact_dirs = []
    run_summary = []
    for run_name, _ in tasks:
        artifact_dir = results.get(run_name)
        run_summary.append({"name": run_name, "artifact_dir": artifact_dir, "success": artifact_dir is not None})
        if artifact_dir:
            artifact_dirs.append(artifact_dir)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary_path = os.path.join(sweep_output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "description": description,
                "sweep_config": str(args.config),
                "runs": run_summary,
            },
            f,
            indent=2,
        )
    print(f"\n[sweep] Run summary saved to: {summary_path}")

    # -----------------------------------------------------------------------
    # Generate comparison report
    # -----------------------------------------------------------------------
    if not artifact_dirs:
        print("[sweep] No successful runs to compare. Skipping report generation.")
        sys.exit(1)

    report_output = args.report_output or os.path.join(sweep_output_dir, "sweep_comparison_report.md")
    run_generate_report(artifact_dirs, report_output)

    print("\n[sweep] Done.")


if __name__ == "__main__":
    main()

