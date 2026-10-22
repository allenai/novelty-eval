import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from novelty_eval.ablation.run_ablations import (
    _build_batch_state,
    _run_poll_mode
)
from novelty_eval.ablation.batch_manager import (
    _compute_run_status,
    _compute_ablation_status,
    write_state
)
from tests.utils import create_mock_artifact_dir

class TestBatchAblationMulti(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("temp_test_multi_batch").resolve()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        self.output_base = self.test_dir / "output"
        self.output_base.mkdir()
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def create_mock_run(self, ablation_name, model, n_batches=1):
        """Create a mock run with n batches."""
        run_name = f"{ablation_name}-{model}"
        artifact_dir = self.output_base / run_name / "accuracy_test_artifacts" / "2026-04-25_12-00-00"
        artifact_dir.mkdir(parents=True)
        
        batches = []
        for i in range(n_batches):
            run_dir = artifact_dir / f"run_pairwise_{i}"
            run_dir.mkdir()
            batch_id = f"batch_{ablation_name}_{model}_{i}"
            batch_info = {
                "batch_id": batch_id,
                "provider": "openai",
                "run_dir": str(run_dir)
            }
            batch_info_file = run_dir / f"batch_info_{batch_id}.json"
            with open(batch_info_file, "w") as f:
                json.dump(batch_info, f)
            
            batches.append({
                "batch_id": batch_id,
                "provider": "openai",
                "run_dir": str(run_dir),
                "batch_info_file": str(batch_info_file),
                "status": "pending"
            })
            
        return {
            "run_name": run_name,
            "artifact_dir": str(artifact_dir),
            "status": "pending",
            "batches": batches
        }

    @patch("novelty_eval.ablation.batch_manager.check_batch_status")
    @patch("novelty_eval.retrieve_batch_results.retrieve_batch_results")
    @patch("novelty_eval.ablation.run_ablations._run_reprocess_report")
    def test_multi_ablation_multi_model_polling(self, mock_reprocess, mock_retrieve, mock_status):
        """
        Challenging test:
        - 2 ablations, 2 models.
        - Mix of single-batch and multi-batch runs.
        - Mix of Success, Failure, and Partial results.
        """
        # 1. Setup the Grid
        # Ablation A: Current
        # Ablation B: Vague Criterion
        
        run_a_m1 = self.create_mock_run("current", "gpt-4o", n_batches=1)
        run_a_m2 = self.create_mock_run("current", "claude-3-5", n_batches=2) # Multi-batch
        run_b_m1 = self.create_mock_run("vague", "gpt-4o", n_batches=1)
        run_b_m2 = self.create_mock_run("vague", "claude-3-5", n_batches=1)
        
        summary_rows = [
            {"ablation": "current", "description": "Current", "status": "OK", "artifact_dirs": [run_a_m1["artifact_dir"], run_a_m2["artifact_dir"]], "runs": [run_a_m1, run_a_m2]},
            {"ablation": "vague", "description": "Vague", "status": "OK", "artifact_dirs": [run_b_m1["artifact_dir"], run_b_m2["artifact_dir"]], "runs": [run_b_m1, run_b_m2]}
        ]
        
        # Manually fix up summary_rows to match what _build_batch_state expects if needed
        # Actually _build_batch_state scans directories, so let's use it.
        state = _build_batch_state(summary_rows, self.output_base, self.test_dir / "instances")
        state_path = self.output_base / "ablation_batch_state.json"
        write_state(state, state_path)
        
        # 2. Define Mock Behaviors
        # A-M1: batch_current_gpt-4o_0 -> completed
        # A-M2: batch_current_claude-3-5_0 -> completed, batch_current_claude-3-5_1 -> failed
        # B-M1: batch_vague_gpt-4o_0 -> pending
        # B-M2: batch_vague_claude-3-5_0 -> completed
        
        def status_side_effect(batch_id, provider):
            if "current_gpt-4o_0" in batch_id: return "completed"
            if "current_claude-3-5_0" in batch_id: return "completed"
            if "current_claude-3-5_1" in batch_id: return "failed"
            if "vague_gpt-4o_0" in batch_id: return "pending"
            if "vague_claude-3-5_0" in batch_id: return "completed"
            return "pending"
        
        mock_status.side_effect = status_side_effect
        mock_retrieve.return_value = {"1": {"prediction": 1}}
        
        # 3. Poll
        _run_poll_mode(state_path)
        
        # 4. Assertions
        with open(state_path) as f:
            updated_state = json.load(f)
            
        ablations = updated_state["ablations"]
        
        # A (Current) status: should be "partial" because one run (A-M2) is partial
        # Wait, A-M1 is completed, A-M2 is partial. Aggregation logic:
        # _compute_run_status for A-M2: {completed, failed} -> "partial"
        # _compute_ablation_status for A: {completed, partial} -> "partial"
        self.assertEqual(ablations["current"]["status"], "partial")
        
        # B (Vague) status: should be "pending" because B-M1 is pending
        self.assertEqual(ablations["vague"]["status"], "pending")
        
        # Verify individual runs
        runs_a = {r["run_name"]: r for r in ablations["current"]["runs"]}
        self.assertEqual(runs_a["current-gpt-4o"]["status"], "completed")
        self.assertEqual(runs_a["current-claude-3-5"]["status"], "partial")
        
        runs_b = {r["run_name"]: r for r in ablations["vague"]["runs"]}
        self.assertEqual(runs_b["vague-gpt-4o"]["status"], "pending")
        self.assertEqual(runs_b["vague-claude-3-5"]["status"], "completed")

        # Verify that reprocess was called for completed/partial runs but NOT pending ones
        # Reportable runs: A-M1 (completed), A-M2 (partial), B-M2 (completed)
        # Non-reportable: B-M1 (pending)
        # Wait, run_is_reportable returns True if all batches are completed/failed.
        # B-M1 has 1 pending batch -> False.
        self.assertEqual(mock_reprocess.call_count, 3)

    def test_aggregate_cost_reports(self):
        """Test that _aggregate_cost_reports correctly sums up costs from artifact directories."""
        from novelty_eval.ablation.run_ablations import _aggregate_cost_reports
        
        # Setup mock artifact dirs with cost_report.json
        run_a = self.output_base / "run_a" / "accuracy_test_artifacts" / "2026-04-25_00-00-00"
        run_a.mkdir(parents=True)
        with open(run_a / "cost_report.json", "w") as f:
            json.dump({
                "total_cost_usd": 1.25,
                "total_calls": 10,
                "test_mode": "pairwise",
                "models": {"gpt-4o": {"cost_usd": 1.25}}
            }, f)
            
        run_b = self.output_base / "run_b" / "accuracy_test_artifacts" / "2026-04-25_00-00-00"
        run_b.mkdir(parents=True)
        with open(run_b / "cost_report.json", "w") as f:
            json.dump({
                "total_cost_usd": 0.75,
                "total_calls": 5,
                "test_mode": "pointwise",
                "models": {"gpt-4o": {"cost_usd": 0.75}}
            }, f)
            
        summary_rows = [
            {"ablation": "current", "artifact_dirs": [str(run_a)]},
            {"ablation": "vague", "artifact_dirs": [str(run_b)]}
        ]
        
        _aggregate_cost_reports(summary_rows, self.output_base)
        
        summary_json = self.output_base / "ablation_cost_summary.json"
        self.assertTrue(summary_json.exists())
        with open(summary_json) as f:
            data = json.load(f)
            
        # Grand total should be 1.25 + 0.75 = 2.0
        self.assertEqual(data["grand_total_cost_usd"], 2.0)
        self.assertIn("current", data["ablations"])
        self.assertIn("vague", data["ablations"])

if __name__ == "__main__":
    unittest.main()
