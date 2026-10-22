import unittest
import json
import yaml
import os
import shutil
import tempfile
from pathlib import Path
import sys

# Add src and the project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(1, str(PROJECT_ROOT))

from novelty_eval.retrieval.flatten_retrieval_cache import flatten_cache

class TestFlattenRetrievalCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.pw_yaml_path = os.path.join(self.test_dir, "pw.yaml")
        self.pw_cache_path = os.path.join(self.test_dir, "pw_cache.json")
        self.ptw_yaml_path = os.path.join(self.test_dir, "ptw.yaml")
        self.out_cache_path = os.path.join(self.test_dir, "ptw_cache.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_flatten_cache_mapping(self):
        # 1. Create Mock Pairwise YAML
        pw_data = {
            "prob1": {
                "ideas": {
                    "0": "Idea 1 text",
                    "1": "Idea 2 text " # trailing space
                }
            },
            "prob2": {
                "ideas": {
                    "0": "Idea 3 text"
                }
            }
        }
        with open(self.pw_yaml_path, 'w') as f:
            yaml.dump(pw_data, f)

        # 2. Create Mock Pairwise Cache (nested)
        pw_cache = {
            "prob1": {
                "0": {"candidates": [{"title": "Paper 1"}], "text": "Idea 1 text"},
                "1": {"candidates": [{"title": "Paper 2"}], "text": "Idea 2 text "}
            },
            "prob2": {
                "0": {"candidates": [{"title": "Paper 3"}], "text": "Idea 3 text"}
            }
        }
        with open(self.pw_cache_path, 'w') as f:
            json.dump(pw_cache, f)

        # 3. Create Mock Pointwise YAML
        # Note: pointwise IDs are often 0, 1, 2... and text is stripped
        ptw_data = {
            "0": {"idea": "Idea 1 text"},
            "1": {"idea": "Idea 2 text"}, # no trailing space
            "2": {"idea": "Idea 3 text"},
            "3": {"idea": "Idea 4 unknown"}
        }
        with open(self.ptw_yaml_path, 'w') as f:
            yaml.dump(ptw_data, f)

        # 4. Run Flattening
        # We need to mock MockArgs for the script's internal call to generate_cache_status_report
        # or just test the flatten_cache function if it's separable.
        # Actually, I'll just run the function.
        
        flatten_cache(self.pw_yaml_path, self.pw_cache_path, self.ptw_yaml_path, self.out_cache_path)

        # 5. Verify Output
        with open(self.out_cache_path, 'r') as f:
            out_cache = json.load(f)

        self.assertEqual(len(out_cache), 3)
        self.assertEqual(out_cache["0"]["candidates"][0]["title"], "Paper 1")
        self.assertEqual(out_cache["1"]["candidates"][0]["title"], "Paper 2")
        self.assertEqual(out_cache["2"]["candidates"][0]["title"], "Paper 3")
        self.assertNotIn("3", out_cache)

    def test_flatten_cache_with_duplicate_texts(self):
        """If multiple instances have the same idea text, they should all get the same cache entry."""
        pw_data = {"p1": {"ideas": {"0": "Shared Text"}}}
        with open(self.pw_yaml_path, 'w') as f: yaml.dump(pw_data, f)

        pw_cache = {"p1": {"0": {"candidates": [{"title": "Shared Paper"}]}}}
        with open(self.pw_cache_path, 'w') as f: json.dump(pw_cache, f)

        ptw_data = {
            "inst1": {"idea": "Shared Text"},
            "inst2": {"idea": "Shared Text"}
        }
        with open(self.ptw_yaml_path, 'w') as f: yaml.dump(ptw_data, f)

        flatten_cache(self.pw_yaml_path, self.pw_cache_path, self.ptw_yaml_path, self.out_cache_path)

        with open(self.out_cache_path, 'r') as f:
            out_cache = json.load(f)

        self.assertEqual(out_cache["inst1"]["candidates"][0]["title"], "Shared Paper")
        self.assertEqual(out_cache["inst2"]["candidates"][0]["title"], "Shared Paper")

if __name__ == "__main__":
    unittest.main()
