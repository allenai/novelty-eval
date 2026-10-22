import json
import os
import shutil
import subprocess
import sys
import unittest
import yaml
from pathlib import Path
from tests.utils import create_mock_artifact_dir

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MERGE_SCRIPT = _PROJECT_ROOT / "src" / "novelty_eval" / "ablation" / "merge_ablation_runs.py"

class TestMergeAblationEdgeCases(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("temp_test_merge").resolve()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)
        self.output_dir = self.test_dir / "merged_output"
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def run_merge(self, dirs, report_name="report.md"):
        cmd = [
            sys.executable, str(_MERGE_SCRIPT),
            *dirs,
            "--output-dir", str(self.output_dir),
            "--report-name", report_name
        ]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))

    def test_merge_partial_runs(self):
        """Test merging when one ablation run is missing a mode."""
        run1 = self.test_dir / "partial_run1"
        run1.mkdir()
        # Create successful pairwise and pointwise in run1
        create_mock_artifact_dir(run1 / "pairwise_current", "pairwise", scores={"1": {"gt_winner": 0}}, metrics={"Accuracy": 1.0})
        create_mock_artifact_dir(run1 / "pointwise_current", "pointwise", scores={"1": {"label": 1, "prediction": 1}}, metrics={"Accuracy": 1.0})
        
        # run2 missing pointwise
        run2 = self.test_dir / "partial_run2"
        run2.mkdir()
        create_mock_artifact_dir(run2 / "pairwise_vague_criterion", "pairwise", scores={"1": {"gt_winner": 0}}, metrics={"Accuracy": 1.0})
        
        # Write ablation_summary.json for run1
        with open(run1 / "ablation_summary.json", "w") as f:
            json.dump({"runs": [
                {"ablation": "current", "artifact_dirs": [str(run1 / "pairwise_current"), str(run1 / "pointwise_current")]}
            ]}, f)
        # Write ablation_summary.json for run2
        with open(run2 / "ablation_summary.json", "w") as f:
            json.dump({"runs": [
                {"ablation": "vague_criterion", "artifact_dirs": [str(run2 / "pairwise_vague_criterion")]}
            ]}, f)

        result = self.run_merge([str(run1), str(run2)])
        self.assertEqual(result.returncode, 0, f"Merge failed: {result.stderr}")
        
        report_path = self.output_dir / "report.md"
        self.assertTrue(report_path.exists())
        content = report_path.read_text()
        self.assertIn("Pairwise Results", content)
        self.assertIn("Pointwise Results", content)
        self.assertIn("Current", content)
        self.assertIn("Vague novelty criterion", content)

    def test_merge_with_failures(self):
        """Test merging when some runs failed (no artifact dirs)."""
        run1 = self.test_dir / "fail_run1"
        run1.mkdir()
        create_mock_artifact_dir(run1 / "pairwise_current", "pairwise", scores={"1": {"gt_winner": 0}}, metrics={"Accuracy": 1.0})
        
        run2 = self.test_dir / "fail_run2"
        run2.mkdir()
        # run2 has a failed ablation entry
        with open(run2 / "ablation_summary.json", "w") as f:
            json.dump({"runs": [
                {"ablation": "failed_abl", "status": "FAILED", "artifact_dirs": []}
            ]}, f)

        # Write summary for run1 too
        with open(run1 / "ablation_summary.json", "w") as f:
            json.dump({"runs": [
                {"ablation": "current", "artifact_dirs": [str(run1 / "pairwise_current")]}
            ]}, f)

        result = self.run_merge([str(run1), str(run2)])
        self.assertEqual(result.returncode, 0, f"Merge failed: {result.stderr}")
        
        report_path = self.output_dir / "report.md"
        self.assertTrue(report_path.exists())
        content = report_path.read_text()
        self.assertIn("Current", content)

    def test_merge_with_safety_refusals(self):
        """Test merging when some runs triggered safety refusals."""
        from tests.utils import create_mock_llm_failures
        run1 = self.test_dir / "safety_run1"
        run1.mkdir()
        create_mock_artifact_dir(run1 / "pairwise_current", "pairwise", scores={"1": {"gt_winner": 0}}, metrics={"Accuracy": 1.0})
        
        # Add a refusal to run1
        create_mock_llm_failures(run1 / "pairwise_current", [
            {"index": 123, "problem": "123", "error_code": "content_filter", "error_message": "Safety refusal"}
        ])
        
        with open(run1 / "ablation_summary.json", "w") as f:
            json.dump({"runs": [
                {"ablation": "current", "artifact_dirs": [str(run1 / "pairwise_current")]}
            ]}, f)

        result = self.run_merge([str(run1)])
        self.assertEqual(result.returncode, 0, f"Merge failed: {result.stderr}")
        
        refusal_report = self.output_dir / "safety_refusal_report.md"
        self.assertTrue(refusal_report.exists(), "Safety refusal report should have been generated")
        content = refusal_report.read_text()
        self.assertIn("content_filter", content)
        self.assertIn("123", content)

if __name__ == "__main__":
    unittest.main()
