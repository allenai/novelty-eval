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

# Import internal functions for unit testing
sys.path.append(str(_PROJECT_ROOT / "src" / "novelty_eval" / "ablation"))
import merge_ablation_runs as merge_utils

class TestBadAbstractFiltering(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("temp_test_filtering").resolve()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(exist_ok=True)
        self.output_dir = self.test_dir / "merged_output"
        
        # Mock paper_blocklist.yaml
        self.blocklist_file = _PROJECT_ROOT / "src" / "novelty_eval" / "benchmark_data" / "paper_blocklist.yaml"
        self.blocklist_backup = None
        if self.blocklist_file.exists():
            self.blocklist_backup = self.blocklist_file.read_text()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if self.blocklist_backup is not None:
            self.blocklist_file.write_text(self.blocklist_backup)
        elif self.blocklist_file.exists():
            self.blocklist_file.unlink()

    def set_blocklist(self, titles):
        self.blocklist_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.blocklist_file, "w") as f:
            yaml.dump({"blocked_titles": titles}, f)

    def run_merge(self, dirs, report_name="report.md"):
        cmd = [
            sys.executable, str(_MERGE_SCRIPT),
            *dirs,
            "--output-dir", str(self.output_dir),
            "--report-name", report_name
        ]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))

    def test_derive_exclude_indices_pointwise(self):
        """Test title extraction from pointwise instances (flat metadata)."""
        instances_path = self.test_dir / "pointwise_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({
                "1": {"metadata": {"title": "Good Paper"}},
                "2": {"metadata": {"title": "Bad Paper"}},
                "3": {"metadata": {"title": "Mixed Casing Paper"}}
            }, f)
        
        blocked = {"bad paper", "mixed casing paper"}
        excluded = merge_utils._derive_exclude_indices(str(instances_path), blocked)
        
        self.assertEqual(len(excluded), 2)
        self.assertIn("2", excluded)
        self.assertIn("3", excluded)
        self.assertEqual(excluded["2"], "Bad Paper")
        self.assertEqual(excluded["3"], "Mixed Casing Paper")

    def test_derive_exclude_indices_pairwise(self):
        """Test title extraction from pairwise instances (nested idea metadata)."""
        instances_path = self.test_dir / "pairwise_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({
                "10": {
                    "metadata": {
                        "0": {"title": "Winner Paper"},
                        "1": {"title": "Loser Paper"}
                    }
                },
                "20": {
                    "metadata": {
                        "0": {"title": "Safe Paper"},
                        "1": {"title": "Blocked Paper"}
                    }
                }
            }, f)
        
        blocked = {"blocked paper"}
        excluded = merge_utils._derive_exclude_indices(str(instances_path), blocked)
        
        self.assertEqual(len(excluded), 1)
        self.assertIn("20", excluded)
        self.assertEqual(excluded["20"], "Blocked Paper")

    def test_baseline_equivalency_empty_blocklist(self):
        """Phase 2: Verify recompute with empty blocklist matches original exactly."""
        run_dir = self.test_dir / "baseline_run"
        run_dir.mkdir()
        
        # Pointwise artifact
        # We need realistic scores so the stats aren't just 0/1
        scores = {
            "1": {"label": 1, "prediction": 1},
            "2": {"label": 0, "prediction": 0},
            "3": {"label": 1, "prediction": 0}
        }
        # Original accuracy = 2/3 = 0.6667
        metrics = {"Accuracy": 0.6667}
        artifact_path = run_dir / "pointwise_current"
        create_mock_artifact_dir(artifact_path, "pointwise", scores=scores, metrics=metrics)
        
        # Pointwise instances
        instances_path = artifact_path / "test_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({
                "1": {"metadata": {"title": "A"}},
                "2": {"metadata": {"title": "B"}},
                "3": {"metadata": {"title": "C"}}
            }, f)
            
        # Update report with correct path
        report_content = (artifact_path / "accuracy_report.txt").read_text()
        report_content = report_content.replace("test_instances.yaml", str(instances_path))
        (artifact_path / "accuracy_report.txt").write_text(report_content)

        # Write ablation_summary.json
        with open(run_dir / "ablation_summary.json", "w") as f:
            json.dump({"runs": [
                {"ablation": "current", "artifact_dirs": [str(artifact_path)]}
            ]}, f)

        self.set_blocklist([]) # Empty blocklist
        
        result = self.run_merge([str(run_dir)])
        self.assertEqual(result.returncode, 0)
        
        filtered_report = artifact_path / "filtered_accuracy_report.txt"
        self.assertFalse(filtered_report.exists(), "Filtered report should NOT be written if nothing is filtered")

        # Test internal recompute function directly for equivalency
        recomputed = merge_utils._recompute_pointwise_metrics(artifact_path, set())
        self.assertIsNotNone(recomputed)
        self.assertAlmostEqual(recomputed["accuracy"], 0.6667, places=4)
        self.assertEqual(recomputed["support"], 3)

    def test_surgical_removal_pointwise(self):
        """Phase 3: Verify filtering exactly N abstracts matches manual sub-calculation."""
        run_dir = self.test_dir / "surgical_run"
        run_dir.mkdir()
        
        # 4 instances: 3 correct, 1 wrong. Accuracy = 0.75
        scores = {
            "1": {"label": 1, "prediction": 1},
            "2": {"label": 0, "prediction": 0},
            "3": {"label": 1, "prediction": 1},
            "4": {"label": 1, "prediction": 0} 
        }
        metrics = {"Accuracy": 0.75}
        artifact_path = run_dir / "pointwise_current"
        create_mock_artifact_dir(artifact_path, "pointwise", scores=scores, metrics=metrics)
        
        instances_path = artifact_path / "test_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({
                "1": {"metadata": {"title": "Keep 1"}},
                "2": {"metadata": {"title": "Keep 2"}},
                "3": {"metadata": {"title": "Remove Me"}},
                "4": {"metadata": {"title": "Keep 3"}}
            }, f)
        
        # Update report with correct path
        report_content = (artifact_path / "accuracy_report.txt").read_text()
        report_content = report_content.replace("test_instances.yaml", str(instances_path))
        (artifact_path / "accuracy_report.txt").write_text(report_content)

        # Write ablation_summary.json
        with open(run_dir / "ablation_summary.json", "w") as f:
            json.dump({"runs": [
                {"ablation": "current", "artifact_dirs": [str(artifact_path)]}
            ]}, f)

        self.set_blocklist(["remove me"])
        
        result = self.run_merge([str(run_dir)])
        self.assertEqual(result.returncode, 0)
        
        filt_report = (artifact_path / "filtered_accuracy_report.txt").read_text()
        
        # After removing "Remove Me" (index 3), we have:
        # 1: Correct
        # 2: Correct
        # 4: Wrong
        # New Accuracy = 2/3 = 0.6667
        self.assertIn("0.6667", filt_report)
        self.assertIn("Filtered: excluded abstract indices ['3']", filt_report)

    def test_surgical_removal_pairwise(self):
        """Phase 3: Verify filtering exactly N abstracts matches manual sub-calculation for pairwise."""
        run_dir = self.test_dir / "surgical_run_pw"
        run_dir.mkdir()
        
        # 3 comparisons: 2 correct, 1 wrong. Accuracy = 0.6667
        scores = {
            "1": {"comparisons": [{"gt_winner": 0, "winner": 0}]},
            "2": {"comparisons": [{"gt_winner": 1, "winner": 1}]},
            "3": {"comparisons": [{"gt_winner": 0, "winner": 1}]}
        }
        metrics = {"Accuracy": 0.6667}
        artifact_path = run_dir / "pairwise_current"
        create_mock_artifact_dir(artifact_path, "pairwise", scores=scores, metrics=metrics)
        
        instances_path = artifact_path / "test_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({
                "1": {"metadata": {"0": {"title": "Keep A"}, "1": {"title": "Keep B"}}},
                "2": {"metadata": {"0": {"title": "Keep C"}, "1": {"title": "Remove Me"}}},
                "3": {"metadata": {"0": {"title": "Keep D"}, "1": {"title": "Keep E"}}}
            }, f)
        
        # Update report with correct path
        report_content = (artifact_path / "accuracy_report.txt").read_text()
        report_content = report_content.replace("test_instances.yaml", str(instances_path))
        (artifact_path / "accuracy_report.txt").write_text(report_content)

        # Write ablation_summary.json
        with open(run_dir / "ablation_summary.json", "w") as f:
            json.dump({"runs": [
                {"ablation": "current", "artifact_dirs": [str(artifact_path)]}
            ]}, f)

        self.set_blocklist(["remove me"])
        
        result = self.run_merge([str(run_dir)])
        self.assertEqual(result.returncode, 0)
        
        filt_report = (artifact_path / "filtered_accuracy_report.txt").read_text()
        
        # After removing "Remove Me" (index 2), we have:
        # 1: Correct
        # 3: Wrong
        # New Accuracy = 1/2 = 0.5000
        self.assertIn("0.5000", filt_report)
        self.assertIn("Filtered: excluded abstract indices ['2']", filt_report)

    def test_complete_blocklist_support_zero(self):
        """Phase 4: Verify complete blocklist handles support=0 gracefully."""
        run_dir = self.test_dir / "empty_run"
        run_dir.mkdir()
        
        scores = {"1": {"label": 1, "prediction": 1}}
        metrics = {"Accuracy": 1.0}
        artifact_path = run_dir / "pointwise_current"
        create_mock_artifact_dir(artifact_path, "pointwise", scores=scores, metrics=metrics)
        
        instances_path = artifact_path / "test_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({"1": {"metadata": {"title": "Delete Me"}}}, f)
            
        # Update report with correct path
        report_content = (artifact_path / "accuracy_report.txt").read_text()
        report_content = report_content.replace("test_instances.yaml", str(instances_path))
        (artifact_path / "accuracy_report.txt").write_text(report_content)

        # Write ablation_summary.json
        with open(run_dir / "ablation_summary.json", "w") as f:
            json.dump({"runs": [
                {"ablation": "current", "artifact_dirs": [str(artifact_path)]}
            ]}, f)

        self.set_blocklist(["delete me"])
        
        result = self.run_merge([str(run_dir)])
        self.assertEqual(result.returncode, 0)
        
        # In the current implementation, if support is 0, it logs a warning and metrics is None.
        # _generate_filtered_reports will then not write the filtered report.
        filtered_report = artifact_path / "filtered_accuracy_report.txt"
        self.assertFalse(filtered_report.exists(), "Filtered report should NOT be written if support is 0")
        
        # Check stderr for the warning
        self.assertIn("Could not recompute metrics", result.stderr)

    # ------------------------------------------------------------------
    # Failure-tie reclassification and missing-pid injection tests
    # ------------------------------------------------------------------

    def test_pairwise_failure_tie_reclassified(self):
        """winner=2 + empty mec_details.fwd_votes must be treated as wrong, not a tie."""
        run_dir = self.test_dir / "failure_tie_run"
        run_dir.mkdir()

        # Scores: one problem with winner=2 but empty mec_details → failure-tie
        scores = {
            "1": {
                "comparisons": [{
                    "idea_0": "0", "idea_1": "1",
                    "winner": 2,
                    "gt_winner": 0,
                    "mec_details": {
                        "novelty": {"fwd_votes": [], "rev_votes": [], "idea_0_scores": [], "idea_1_scores": []},
                    },
                }]
            }
        }
        artifact_path = run_dir / "pairwise_failure"
        create_mock_artifact_dir(artifact_path, "pairwise", scores=scores, metrics={"Accuracy": 0.5})

        instances_path = artifact_path / "test_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({
                "1": {
                    "metadata": {"0": {"title": "A"}, "1": {"title": "B"}},
                    "expected_winners": [0],
                    "ideas": {"0": "idea A", "1": "idea B"},
                }
            }, f)

        report_content = (artifact_path / "accuracy_report.txt").read_text()
        report_content = report_content.replace("test_instances.yaml", str(instances_path))
        (artifact_path / "accuracy_report.txt").write_text(report_content)

        recomputed = merge_utils._recompute_pairwise_metrics(artifact_path, set())
        self.assertIsNotNone(recomputed)
        # Failure-tie reclassified as wrong → accuracy=0.0, n_ties=0, support=1
        self.assertEqual(recomputed["support"], 1)
        self.assertEqual(recomputed["n_ties"], 0)
        self.assertAlmostEqual(recomputed["pairwise_accuracy"], 0.0, places=4)

    def test_pairwise_missing_pid_injected_as_wrong(self):
        """A pid present in the instances YAML but absent from scores.json must count as wrong."""
        run_dir = self.test_dir / "missing_pid_pairwise"
        run_dir.mkdir()

        # Only pid "1" is in scores (correct); pid "2" is missing → should be injected as wrong
        scores = {
            "1": {"comparisons": [{"idea_0": "0", "idea_1": "1", "winner": 0, "gt_winner": 0}]},
        }
        artifact_path = run_dir / "pairwise_missing"
        create_mock_artifact_dir(artifact_path, "pairwise", scores=scores, metrics={"Accuracy": 1.0})

        instances_path = artifact_path / "test_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({
                "1": {
                    "metadata": {"0": {"title": "A"}, "1": {"title": "B"}},
                    "expected_winners": [0],
                    "ideas": {"0": "idea A", "1": "idea B"},
                },
                "2": {
                    "metadata": {"0": {"title": "C"}, "1": {"title": "D"}},
                    "expected_winners": [0],
                    "ideas": {"0": "idea C", "1": "idea D"},
                },
            }, f)

        report_content = (artifact_path / "accuracy_report.txt").read_text()
        report_content = report_content.replace("test_instances.yaml", str(instances_path))
        (artifact_path / "accuracy_report.txt").write_text(report_content)

        recomputed = merge_utils._recompute_pairwise_metrics(artifact_path, set())
        self.assertIsNotNone(recomputed)
        # pid "1" correct (1.0), pid "2" injected as wrong (0.0) → accuracy = 0.5, support = 2
        self.assertEqual(recomputed["support"], 2)
        self.assertAlmostEqual(recomputed["pairwise_accuracy"], 0.5, places=4)

    def test_pointwise_missing_pid_injected_as_wrong(self):
        """A pid present in the instances YAML but absent from scores.json must count as wrong."""
        run_dir = self.test_dir / "missing_pid_pointwise"
        run_dir.mkdir()

        # Only pid "1" is in scores (correct POSITIVE); pid "2" is missing (POSITIVE) → injected as wrong
        scores = {
            "1": {"prediction": 1, "label": "POSITIVE"},
        }
        artifact_path = run_dir / "pointwise_missing"
        create_mock_artifact_dir(artifact_path, "pointwise", scores=scores, metrics={"Accuracy": 1.0})

        instances_path = artifact_path / "test_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({
                "1": {"label": "POSITIVE", "metadata": {"title": "A"}},
                "2": {"label": "POSITIVE", "metadata": {"title": "B"}},
            }, f)

        report_content = (artifact_path / "accuracy_report.txt").read_text()
        report_content = report_content.replace("test_instances.yaml", str(instances_path))
        (artifact_path / "accuracy_report.txt").write_text(report_content)

        recomputed = merge_utils._recompute_pointwise_metrics(artifact_path, set())
        self.assertIsNotNone(recomputed)
        # pid "1" correct (TP), pid "2" injected as wrong (FN) → support=2, accuracy=0.5
        self.assertEqual(recomputed["support"], 2)
        self.assertAlmostEqual(recomputed["accuracy"], 0.5, places=4)

    def test_bootstrap_failure_tie_reclassified(self):
        """_extract_raw_outcomes_for_dir should treat failure-ties as wrong (is_tie=0, is_correct=0)."""
        from novelty_eval.analysis.artifacts import (
            _extract_raw_outcomes_for_dir,
            _RAW_OUTCOMES_CACHE,
        )
        import logging

        run_dir = self.test_dir / "bootstrap_failure_tie"
        run_dir.mkdir()

        scores = {
            "1": {
                "comparisons": [{
                    "idea_0": "0", "idea_1": "1",
                    "winner": 2,
                    "gt_winner": 0,
                    "mec_details": {
                        "novelty": {"fwd_votes": [], "rev_votes": [], "idea_0_scores": [], "idea_1_scores": []},
                    },
                }]
            }
        }
        artifact_path = run_dir / "pairwise_bootstrap_ft"
        create_mock_artifact_dir(artifact_path, "pairwise", scores=scores, metrics={"Accuracy": 0.5})

        instances_path = artifact_path / "test_instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump({
                "1": {
                    "expected_winners": [0],
                    "ideas": {"0": "idea A", "1": "idea B"},
                    "metadata": {"0": {"title": "A"}, "1": {"title": "B"}},
                }
            }, f)

        report_content = (artifact_path / "accuracy_report.txt").read_text()
        report_content = report_content.replace("test_instances.yaml", str(instances_path))
        (artifact_path / "accuracy_report.txt").write_text(report_content)

        # Clear cache to avoid stale hits from other tests
        _RAW_OUTCOMES_CACHE.pop((artifact_path, "pairwise"), None)

        outcomes = _extract_raw_outcomes_for_dir(
            artifact_path, "pairwise", "pairwise_bootstrap_ft",
            logger_inst=logging.getLogger("test"),
        )
        self.assertIsNotNone(outcomes)
        # Failure-tie → is_tie must be 0.0 and is_correct must be 0.0
        self.assertAlmostEqual(outcomes["_pairwise_is_tie"].get("1", -1), 0.0, places=4)
        self.assertAlmostEqual(outcomes["_pairwise_is_correct"].get("1", -1), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
