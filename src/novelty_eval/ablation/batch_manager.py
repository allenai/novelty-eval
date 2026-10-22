"""
batch_manager.py — State file I/O and batch status polling for ablation batch mode.

State file schema (ablation_batch_state.json):
{
  "version": 1,
  "created_at": "ISO timestamp",
  "output_base": "/abs/path",
  "instances_base": "/abs/path",
  "ablations": {
    "pairwise_current": {
      "description": "...",
      "status": "pending|completed|partial|failed",
      "inst_dir": null,
      "runs": [
        {
          "run_name": "pairwise_current-gpt-5.4",
          "artifact_dir": "/abs/path/accuracy_test_artifacts/2026-04-25_10-00-00",
          "status": "pending|completed|partial|failed",
          "batches": [
            {
              "batch_id": "batch_abc123",
              "provider": "openai|anthropic",
              "run_dir": "/abs/path/run_pairwise_0",
              "batch_info_file": "/abs/path/run_pairwise_0/batch_info_batch_abc123.json",
              "status": "pending|completed|failed"
            }
          ]
        }
      ]
    }
  }
}
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from logging_utils import setup_logger
from batch_api import check_batch_status  # noqa: F401 — re-exported for run_ablations.py

LOGGER = setup_logger(output_dir=os.getenv("OUTPUT_DIR", "."))


# ---------------------------------------------------------------------------
# State file I/O
# ---------------------------------------------------------------------------

def write_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scan artifact dirs for batch_info files
# ---------------------------------------------------------------------------

def scan_artifact_dirs_for_batches(artifact_dirs: list[str]) -> list[dict]:
    """Scan each artifact dir for run_*/batch_info_*.json files.

    Returns a list of dicts with keys: batch_id, provider, run_dir,
    batch_info_file, artifact_dir.
    """
    found = []
    for artifact_dir in artifact_dirs:
        p = Path(artifact_dir)
        if not p.is_dir():
            continue
        for batch_file in sorted(p.rglob("batch_info_*.json")):
            try:
                with open(batch_file) as f:
                    info = json.load(f)
            except Exception:
                continue
            batch_id = info.get("batch_id")
            if not batch_id:
                continue
            found.append({
                "batch_id": batch_id,
                "provider": info.get("provider", "openai"),
                "run_dir": str(batch_file.parent),
                "batch_info_file": str(batch_file),
                "artifact_dir": artifact_dir,
            })
    return found


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _compute_run_status(run: dict) -> str:
    statuses = {b["status"] for b in run.get("batches", [])}
    if not statuses:
        return "pending"
    if statuses <= {"completed"}:
        return "completed"
    if "pending" in statuses or "error" in statuses:
        return "pending"
    # Mix of completed + failed
    if "completed" in statuses:
        return "partial"
    return "failed"


def _compute_ablation_status(ablation_entry: dict) -> str:
    run_statuses = {_compute_run_status(r) for r in ablation_entry.get("runs", [])}
    if not run_statuses:
        return "pending"
    if run_statuses <= {"completed"}:
        return "completed"
    if "pending" in run_statuses:
        return "pending"
    if "completed" in run_statuses or "partial" in run_statuses:
        return "partial"
    return "failed"


def run_is_reportable(run: dict) -> bool:
    """True when all batches in a run have a terminal status (completed or failed)."""
    return all(b["status"] in ("completed", "failed") for b in run.get("batches", []))


def run_has_any_results(run: dict) -> bool:
    """True when at least one batch in the run completed successfully."""
    return any(b["status"] == "completed" for b in run.get("batches", []))
