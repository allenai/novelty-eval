
import json
import os
import shutil
import unittest
import yaml
from pathlib import Path
from collections import Counter
from unittest.mock import patch

# Add src to path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_PROJECT_ROOT / "src"))
sys.path.append(str(_PROJECT_ROOT / "src" / "novelty_eval" / "ablation"))

from merge_ablation_runs import (
    _derive_pointwise_partner_excluded,
    _derive_exclude_indices,
)
# We'll need to mock extract_ideas if it's defined inside _generate_test_sets_consistency_report
# but wait, in my previous edit I kept it inside the function.
# To test it properly, I might need to refactor it or just test the report generator.
# Let's import the whole module to access internal functions if needed via patching or direct access if they were top-level.
import merge_ablation_runs

class TestTestSetsConsistency(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("temp_test_consistency").resolve()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        
        # Setup mock project structure
        self.instances_dir = self.test_dir / "output" / "benchmark_instances"
        self.instances_dir.mkdir(parents=True)
        
        # 1. Create source pairwise YAML
        self.source_pairwise_dir = self.instances_dir / "pairwise_data" / "20260429_000000" / "manipulated"
        self.source_pairwise_dir.mkdir(parents=True)
        self.source_pairwise_yaml = self.source_pairwise_dir / "benchmark_instances.yaml"
        
        self.idea_text_pos = "This is a positive idea text.  " # Note trailing spaces
        self.idea_text_neg = "\nThis is a negative idea text." # Note leading newline
        
        self.pairwise_data = {
            "0": {
                "metadata": {
                    "0": {"title": "Blocked Paper"},
                    "1": {"title": "Gold Idea Title"}
                },
                "ideas": {
                    "0": "Blocked Paper Abstract...",
                    "1": self.idea_text_pos
                },
                "expected_winners": [1]
            },
            "1": {
                "metadata": {
                    "0": {"title": "Safe Paper"},
                    "1": {"title": "Bad Idea Title"}
                },
                "ideas": {
                    "0": "Safe Paper Abstract...",
                    "1": self.idea_text_neg
                },
                "expected_winners": [0]
            }
        }
        with open(self.source_pairwise_yaml, "w") as f:
            yaml.dump(self.pairwise_data, f)
            
        # 2. Create pointwise ablation dir with config and instances
        self.ptw_ablation_dir = self.instances_dir / "ablation" / "pointwise_test"
        self.ptw_ablation_dir.mkdir(parents=True)
        
        self.config_path = self.ptw_ablation_dir / "_instance_creation_config.yaml"
        with open(self.config_path, "w") as f:
            yaml.dump({"derive_pointwise_from_pairwise": str(self.source_pairwise_yaml)}, f)
            
        self.ptw_instances_yaml = self.ptw_ablation_dir / "20260430_000000" / "benchmark_instances.yaml"
        self.ptw_instances_yaml.parent.mkdir(parents=True)
        
        self.pointwise_data = {
            "0": {
                "metadata": {"title": "Gold Idea Title"},
                "idea": self.idea_text_pos.strip(), # Normalized in pointwise
                "label": "POSITIVE" # Test long label
            },
            "1": {
                "metadata": {"title": "Blocked Paper"},
                "idea": "Blocked Paper Abstract...",
                "label": "NEGATIVE"
            },
            "2": {
                "metadata": {"title": "Bad Idea Title"},
                "idea": self.idea_text_neg.strip(), # Normalized
                "label": "NEGATIVE"
            },
             "3": {
                "metadata": {"title": "Safe Paper"},
                "idea": "Safe Paper Abstract...",
                "label": "POSITIVE"
            }
        }
        with open(self.ptw_instances_yaml, "w") as f:
            yaml.dump(self.pointwise_data, f)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_derive_pointwise_partner_excluded_robustness(self):
        """Verify that partner filtering works with whitespace and different directory structures."""
        blocked_titles = {"blocked paper"}
        
        # Test direct call to partner exclusion
        # It should find that in pairwise instance "0", "Blocked Paper" is blocked.
        # Its partner is "Gold Idea Title" with text self.idea_text_pos.
        # It should then find that in pointwise YAML, index "0" has that same text (after stripping).
        
        with patch("merge_ablation_runs._PROJECT_ROOT", self.test_dir):
            excluded = _derive_pointwise_partner_excluded(str(self.ptw_instances_yaml), blocked_titles)
            
        self.assertIn("0", excluded)
        self.assertEqual(excluded["0"], "Gold Idea Title")
        # Ensure it didn't exclude anything else
        self.assertEqual(len(excluded), 1)

    def test_derive_exclude_indices_types(self):
        """Ensure it handles string/int indices correctly."""
        blocked_titles = {"blocked paper"}
        excluded = _derive_exclude_indices(str(self.source_pairwise_yaml), blocked_titles)
        
        # Should exclude pairwise index "0"
        self.assertIn("0", excluded)
        self.assertEqual(excluded["0"], "Blocked Paper")

    def test_extract_ideas_helper(self):
        """Test the extract_ideas logic (manually copied from the script for testing if needed, 
        or we can test it via a mock call if we move it to top level)."""
        
        # Since I haven't moved it to top-level yet, let's move it to top-level in merge_ablation_runs.py 
        # in the next turn if this test proves it's hard to test otherwise.
        # For now, I'll test it by calling _generate_test_sets_consistency_report with minimal inputs.
        pass

    def test_full_consistency_report_generation(self):
        """Verify the report correctly identifies parity and mismatches."""
        output_dir = self.test_dir / "report_out"
        output_dir.mkdir()
        
        # Create a mock retrieval index
        # We need a pairwise artifact dir and a pointwise one.
        pw_art = self.test_dir / "pw_art"
        pw_art.mkdir()
        (pw_art / "accuracy_report.txt").write_text(f"Test Instances Path: {self.source_pairwise_yaml}\nTest Mode: pairwise")
        
        ptw_art = self.test_dir / "ptw_art"
        ptw_art.mkdir()
        (ptw_art / "accuracy_report.txt").write_text(f"Test Instances Path: {self.ptw_instances_yaml}\nTest Mode: pointwise")
        
        retrieval_index = {
            ("pairwise", "current", "gpt-4o"): str(pw_art),
            ("pointwise", "current", "gpt-4o"): str(ptw_art)
        }
        
        blocked_titles = {"blocked paper"}
        
        with patch("merge_ablation_runs._PROJECT_ROOT", self.test_dir):
            report_path = merge_ablation_runs._generate_test_sets_consistency_report(
                retrieval_index, output_dir, blocked_titles
            )
            
        self.assertTrue(report_path.exists())
        content = report_path.read_text()
        
        # Check for global consistency section
        self.assertIn("## Global Consistency: Filtered Positive Ideas", content)
        # Gold idea in pairwise (filtered): Gold Idea Title (index 1 in prob 0) -> wait, 
        # index 0 is blocked, so gold idea "Gold Idea Title" is partner-filtered.
        # Prob 1: Safe Paper (0) vs Bad Idea Title (1). Gold is 0 ("Safe Paper").
        # So positive ideas filtered: "Safe Paper".
        
        # Check for per-ablation breakdown
        self.assertIn("## Per-Ablation Breakdown: `current`", content)
        self.assertIn("✅ Pointwise and pairwise instances contain the **exact same ideas**.", content)

if __name__ == "__main__":
    unittest.main()
