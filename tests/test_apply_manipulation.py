"""Tests for src/novelty_eval/benchmark_data/utility/apply_manipulation.py"""
import os
import sys
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import shutil

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from novelty_eval.benchmark_data.utility.apply_manipulation import (
    _EMPTY_FIELD_RE,
    _apply_cache,
    _collect_unique_papers,
    _make_cache_key,
    main,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAIRWISE_DATASET = {
    0: {
        "context": "machine learning",
        "expected_winners": [0],
        "ideas": {
            0: "Abstract for paper A about transformers.",
            1: "Abstract for paper B about CNNs.",
        },
        "metadata": {
            0: {"title": "Paper A", "area": "ml", "rating": "7.00",
                "contribution": "3.00", "type": "POSITIVE",
                "positive_signals": ["novel"], "negative_signals": [],
                "similar_papers_mentioned": []},
            1: {"title": "Paper B", "area": "ml", "rating": "4.00",
                "contribution": "2.00", "type": "NEGATIVE",
                "positive_signals": [], "negative_signals": ["incremental"],
                "similar_papers_mentioned": []},
        },
    },
    1: {
        "context": "NLP",
        "expected_winners": [1],
        "ideas": {
            0: "Abstract for paper C about BERT.",
            1: "Abstract for paper D about GPT.",
        },
        "metadata": {
            0: {"title": "Paper C", "area": "nlp", "rating": "5.00",
                "contribution": "2.50", "type": "NEGATIVE",
                "positive_signals": [], "negative_signals": [],
                "similar_papers_mentioned": []},
            1: {"title": "Paper D", "area": "nlp", "rating": "8.00",
                "contribution": "4.00", "type": "POSITIVE",
                "positive_signals": ["original"], "negative_signals": [],
                "similar_papers_mentioned": []},
        },
    },
}

POINTWISE_DATASET = {
    0: {
        "context": "computer vision",
        "idea": "Abstract for paper X about ResNets.",
        "label": "POSITIVE",
        "metadata": {"title": "Paper X", "area": "cv", "rating": "7.00",
                     "contribution": "3.00", "positive_signals": [],
                     "negative_signals": [], "similar_papers_mentioned": []},
    },
    1: {
        "context": "computer vision",
        "idea": "Abstract for paper Y about ViTs.",
        "label": "NEGATIVE",
        "metadata": {"title": "Paper Y", "area": "cv", "rating": "4.00",
                     "contribution": "2.00", "positive_signals": [],
                     "negative_signals": [], "similar_papers_mentioned": []},
    },
}

MANIPULATED = {
    "Paper A": "**context**: ML context.\n**purpose**: Purpose A.\n**mechanism**: Mechanism A.\n**evaluation**: Eval A.\n",
    "Paper B": "**context**: ML context.\n**purpose**: Purpose B.\n**mechanism**: Mechanism B.\n**evaluation**: Eval B.\n",
    "Paper C": "**context**: NLP context.\n**purpose**: Purpose C.\n**mechanism**: Mechanism C.\n**evaluation**: Eval C.\n",
    "Paper D": "**context**: NLP context.\n**purpose**: Purpose D.\n**mechanism**: Mechanism D.\n**evaluation**: Eval D.\n",
}


# ---------------------------------------------------------------------------
# _collect_unique_papers
# ---------------------------------------------------------------------------

class TestCollectUniquePapers(unittest.TestCase):

    def test_pairwise_collects_all_ideas(self):
        unique = _collect_unique_papers(PAIRWISE_DATASET)
        self.assertEqual(len(unique), 4)
        titles = {v["title"] for v in unique.values()}
        self.assertEqual(titles, {"Paper A", "Paper B", "Paper C", "Paper D"})

    def test_pairwise_stores_abstract_and_title(self):
        unique = _collect_unique_papers(PAIRWISE_DATASET)
        entry = next(v for v in unique.values() if v["title"] == "Paper A")
        self.assertEqual(entry["abstract"], "Abstract for paper A about transformers.")
        self.assertEqual(entry["title"], "Paper A")

    def test_pairwise_deduplication_same_title_and_abstract(self):
        shared_abstract = "Shared abstract text."
        dataset = {
            0: {"ideas": {0: shared_abstract}, "metadata": {0: {"title": "Shared Paper"}}},
            1: {"ideas": {0: shared_abstract}, "metadata": {0: {"title": "Shared Paper"}}},
        }
        unique = _collect_unique_papers(dataset)
        self.assertEqual(len(unique), 1)

    def test_pairwise_same_title_different_abstract_not_deduplicated(self):
        dataset = {
            0: {"ideas": {0: "Abstract version one."}, "metadata": {0: {"title": "Same Title"}}},
            1: {"ideas": {0: "Abstract version two."}, "metadata": {0: {"title": "Same Title"}}},
        }
        unique = _collect_unique_papers(dataset)
        self.assertEqual(len(unique), 2)

    def test_pairwise_no_title_uses_compound_key(self):
        abstract = "No title abstract."
        dataset = {0: {"ideas": {0: abstract}, "metadata": {}}}
        unique = _collect_unique_papers(dataset)
        self.assertEqual(len(unique), 1)
        expected_key = _make_cache_key("N/A", abstract)
        self.assertIn(expected_key, unique)

    def test_pointwise_collects_all_ideas(self):
        unique = _collect_unique_papers(POINTWISE_DATASET)
        self.assertEqual(len(unique), 2)
        titles = {v["title"] for v in unique.values()}
        self.assertEqual(titles, {"Paper X", "Paper Y"})

    def test_pointwise_stores_abstract_and_title(self):
        unique = _collect_unique_papers(POINTWISE_DATASET)
        entry = next(v for v in unique.values() if v["title"] == "Paper X")
        self.assertEqual(entry["abstract"], "Abstract for paper X about ResNets.")
        self.assertEqual(entry["title"], "Paper X")

    def test_empty_dataset_returns_empty(self):
        self.assertEqual(_collect_unique_papers({}), {})


# ---------------------------------------------------------------------------
# _apply_cache
# ---------------------------------------------------------------------------

class TestApplyCache(unittest.TestCase):

    def _make_cache(self):
        abstracts = {
            "Paper A": PAIRWISE_DATASET[0]["ideas"][0],
            "Paper B": PAIRWISE_DATASET[0]["ideas"][1],
            "Paper C": PAIRWISE_DATASET[1]["ideas"][0],
            "Paper D": PAIRWISE_DATASET[1]["ideas"][1],
        }
        return {
            _make_cache_key(title, abstract): f"MANIPULATED: {title}"
            for title, abstract in abstracts.items()
        }

    def test_pairwise_ideas_are_replaced(self):
        cache = self._make_cache()
        result = _apply_cache(PAIRWISE_DATASET, cache)
        self.assertEqual(result[0]["ideas"][0], "MANIPULATED: Paper A")
        self.assertEqual(result[0]["ideas"][1], "MANIPULATED: Paper B")
        self.assertEqual(result[1]["ideas"][0], "MANIPULATED: Paper C")
        self.assertEqual(result[1]["ideas"][1], "MANIPULATED: Paper D")

    def test_pairwise_non_ideas_fields_unchanged(self):
        cache = self._make_cache()
        result = _apply_cache(PAIRWISE_DATASET, cache)
        for inst_id, instance in PAIRWISE_DATASET.items():
            self.assertEqual(result[inst_id]["context"], instance["context"])
            self.assertEqual(result[inst_id]["expected_winners"], instance["expected_winners"])
            self.assertEqual(result[inst_id]["metadata"], instance["metadata"])

    def test_pairwise_idea_indices_preserved(self):
        cache = self._make_cache()
        result = _apply_cache(PAIRWISE_DATASET, cache)
        for inst_id in PAIRWISE_DATASET:
            self.assertEqual(
                set(result[inst_id]["ideas"].keys()),
                set(PAIRWISE_DATASET[inst_id]["ideas"].keys()),
            )

    def test_pairwise_instance_ids_preserved(self):
        cache = self._make_cache()
        result = _apply_cache(PAIRWISE_DATASET, cache)
        self.assertEqual(set(result.keys()), set(PAIRWISE_DATASET.keys()))

    def test_pairwise_cache_miss_falls_back_to_original(self):
        result = _apply_cache(PAIRWISE_DATASET, {})
        self.assertEqual(result[0]["ideas"][0], PAIRWISE_DATASET[0]["ideas"][0])

    def test_pointwise_idea_is_replaced(self):
        cache = {
            _make_cache_key("Paper X", POINTWISE_DATASET[0]["idea"]): "MANIPULATED: X",
            _make_cache_key("Paper Y", POINTWISE_DATASET[1]["idea"]): "MANIPULATED: Y",
        }
        result = _apply_cache(POINTWISE_DATASET, cache)
        self.assertEqual(result[0]["idea"], "MANIPULATED: X")
        self.assertEqual(result[1]["idea"], "MANIPULATED: Y")

    def test_pointwise_non_idea_fields_unchanged(self):
        cache = {
            _make_cache_key("Paper X", POINTWISE_DATASET[0]["idea"]): "MANIPULATED: X",
            _make_cache_key("Paper Y", POINTWISE_DATASET[1]["idea"]): "MANIPULATED: Y",
        }
        result = _apply_cache(POINTWISE_DATASET, cache)
        for inst_id, instance in POINTWISE_DATASET.items():
            self.assertEqual(result[inst_id]["context"], instance["context"])
            self.assertEqual(result[inst_id]["label"], instance["label"])
            self.assertEqual(result[inst_id]["metadata"], instance["metadata"])

    def test_original_dataset_not_mutated(self):
        import copy
        original = copy.deepcopy(PAIRWISE_DATASET)
        _apply_cache(PAIRWISE_DATASET, self._make_cache())
        self.assertEqual(PAIRWISE_DATASET, original)


# ---------------------------------------------------------------------------
# _EMPTY_FIELD_RE
# ---------------------------------------------------------------------------

class TestEmptyFieldRegex(unittest.TestCase):

    def _sub(self, text):
        return _EMPTY_FIELD_RE.sub("", text)

    def test_removes_field_with_single_space_before_newline(self):
        self.assertEqual(
            self._sub("**context**: text.\n**evaluation**: \n"),
            "**context**: text.\n",
        )

    def test_removes_field_with_no_space_before_newline(self):
        self.assertEqual(
            self._sub("**context**: text.\n**evaluation**:\n"),
            "**context**: text.\n",
        )

    def test_removes_field_with_multiple_spaces(self):
        self.assertEqual(
            self._sub("**context**: text.\n**evaluation**:   \n"),
            "**context**: text.\n",
        )

    def test_removes_field_with_tab(self):
        self.assertEqual(
            self._sub("**context**: text.\n**evaluation**:\t\n"),
            "**context**: text.\n",
        )

    def test_removes_field_at_end_of_string_no_newline(self):
        self.assertEqual(
            self._sub("**context**: text.\n**evaluation**: "),
            "**context**: text.\n",
        )

    def test_keeps_field_with_non_empty_value(self):
        text = "**context**: Some context.\n**purpose**: Some purpose.\n"
        self.assertEqual(self._sub(text), text)

    def test_removes_multiple_empty_fields(self):
        text = "**context**: text.\n**purpose**: \n**mechanism**: mech.\n**evaluation**:\n"
        self.assertEqual(self._sub(text), "**context**: text.\n**mechanism**: mech.\n")

    def test_all_fields_empty_returns_empty_string(self):
        text = "**context**: \n**purpose**: \n**mechanism**: \n**evaluation**: \n"
        self.assertEqual(self._sub(text), "")


# ---------------------------------------------------------------------------
# Integration tests (main() with mocked LLM)
# ---------------------------------------------------------------------------

def _make_mock_manipulate(mapping: dict):
    """Return a _manipulate_paper replacement that resolves title → content."""
    def _fake(paper, template, model_name, return_plan=False, debug_log=None):
        title = paper.get("title", "N/A")
        key = title if title != "N/A" else str(hash(paper["abstract"]))
        content = mapping.get(key, paper["abstract"])
        if debug_log is not None:
            debug_log.record_success()
        if return_plan:
            return key, content, {}
        return key, content
    return _fake


class TestApplyManipulationIntegration(unittest.TestCase):

    def _run_main(self, dataset: dict, extra_args: list[str] = None,
                  manipulate_map: dict = None) -> tuple[dict, Path]:
        """Write dataset to a temp file, run main(), return (output_dataset, output_path).

        The temp directory is kept alive until the test ends via addCleanup.
        """
        manipulate_map = manipulate_map or MANIPULATED
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, True)

        input_path = Path(tmpdir) / "iclr_test_instances.yaml"
        output_path = Path(tmpdir) / "out" / "iclr_test_instances.yaml"
        with open(input_path, "w") as f:
            yaml.dump(dataset, f, default_flow_style=False, allow_unicode=True)

        argv = [
            "apply_manipulation.py",
            "--dataset", str(input_path),
            "--output", str(output_path),
            "--model", "fake-model",
            "--max-workers", "1",
        ]
        if extra_args:
            argv.extend(extra_args)

        mock_fn = _make_mock_manipulate(manipulate_map)
        with patch("novelty_eval.benchmark_data.utility.apply_manipulation._manipulate_paper", side_effect=mock_fn), \
             patch("sys.argv", argv):
            main()

        with open(output_path) as f:
            result = yaml.safe_load(f)

        return result, output_path

    # --- structure preservation ---

    def test_pairwise_instance_ids_and_order_preserved(self):
        result, _ = self._run_main(PAIRWISE_DATASET)
        self.assertEqual(list(result.keys()), list(PAIRWISE_DATASET.keys()))

    def test_pairwise_idea_indices_preserved(self):
        result, _ = self._run_main(PAIRWISE_DATASET)
        for inst_id in PAIRWISE_DATASET:
            self.assertEqual(
                sorted(result[inst_id]["ideas"].keys()),
                sorted(PAIRWISE_DATASET[inst_id]["ideas"].keys()),
            )

    def test_pairwise_context_unchanged(self):
        result, _ = self._run_main(PAIRWISE_DATASET)
        for inst_id, instance in PAIRWISE_DATASET.items():
            self.assertEqual(result[inst_id]["context"], instance["context"])

    def test_pairwise_expected_winners_unchanged(self):
        result, _ = self._run_main(PAIRWISE_DATASET)
        for inst_id, instance in PAIRWISE_DATASET.items():
            self.assertEqual(result[inst_id]["expected_winners"], instance["expected_winners"])

    def test_pairwise_metadata_unchanged(self):
        result, _ = self._run_main(PAIRWISE_DATASET)
        for inst_id, instance in PAIRWISE_DATASET.items():
            self.assertEqual(result[inst_id]["metadata"], instance["metadata"])

    # --- ideas text changes ---

    def test_pairwise_ideas_are_manipulated(self):
        result, _ = self._run_main(PAIRWISE_DATASET)
        self.assertEqual(result[0]["ideas"][0], MANIPULATED["Paper A"])
        self.assertEqual(result[0]["ideas"][1], MANIPULATED["Paper B"])
        self.assertEqual(result[1]["ideas"][0], MANIPULATED["Paper C"])
        self.assertEqual(result[1]["ideas"][1], MANIPULATED["Paper D"])

    def test_pairwise_ideas_differ_from_originals(self):
        result, _ = self._run_main(PAIRWISE_DATASET)
        for inst_id, instance in PAIRWISE_DATASET.items():
            for idx, orig_abstract in instance["ideas"].items():
                self.assertNotEqual(result[inst_id]["ideas"][idx], orig_abstract)

    # --- deduplication ---

    def test_deduplication_paper_in_two_instances_manipulated_once(self):
        shared_abstract = "Shared abstract for dedup test."
        dataset = {
            0: {"context": "area", "expected_winners": [0],
                "ideas": {0: shared_abstract},
                "metadata": {0: {"title": "Shared Paper", "area": "x",
                                 "rating": "7.00", "contribution": "3.00",
                                 "type": "POSITIVE", "positive_signals": [],
                                 "negative_signals": [], "similar_papers_mentioned": []}}},
            1: {"context": "area", "expected_winners": [0],
                "ideas": {0: shared_abstract},
                "metadata": {0: {"title": "Shared Paper", "area": "x",
                                 "rating": "7.00", "contribution": "3.00",
                                 "type": "POSITIVE", "positive_signals": [],
                                 "negative_signals": [], "similar_papers_mentioned": []}}},
        }
        call_count = {"n": 0}
        original_fake = _make_mock_manipulate({"Shared Paper": "MANIPULATED shared"})

        def counting_fake(paper, template, model_name, return_plan=False, debug_log=None):
            call_count["n"] += 1
            return original_fake(paper, template, model_name, return_plan, debug_log)

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, True)
        input_path = Path(tmpdir) / "iclr_test_instances.yaml"
        output_path = Path(tmpdir) / "out.yaml"
        with open(input_path, "w") as f:
            yaml.dump(dataset, f)
        with patch("novelty_eval.benchmark_data.utility.apply_manipulation._manipulate_paper",
                   side_effect=counting_fake), \
             patch("sys.argv", ["s", "--dataset", str(input_path),
                                "--output", str(output_path),
                                "--model", "x", "--max-workers", "1"]):
            main()

        self.assertEqual(call_count["n"], 1, "LLM should be called once for a deduplicated paper")

        with open(output_path) as f:
            result = yaml.safe_load(f)
        self.assertEqual(result[0]["ideas"][0], "MANIPULATED shared")
        self.assertEqual(result[1]["ideas"][0], "MANIPULATED shared")

    # --- empty field removal stats ---

    def test_empty_field_stats_logged(self):
        manipulate_with_empty = {
            "Paper A": "**context**: ML context.\n**purpose**: \n**mechanism**: Mechanism A.\n**evaluation**:\n",
            "Paper B": "**context**: ML context.\n**purpose**: Purpose B.\n**mechanism**: \n**evaluation**: Eval B.\n",
            "Paper C": "**context**: NLP context.\n**purpose**: Purpose C.\n**mechanism**: Mechanism C.\n**evaluation**: Eval C.\n",
            "Paper D": "**context**: NLP context.\n**purpose**: Purpose D.\n**mechanism**: Mechanism D.\n**evaluation**: Eval D.\n",
        }
        with self.assertLogs("logging_utils", level="INFO") as cm:
            self._run_main(PAIRWISE_DATASET, manipulate_map=manipulate_with_empty)
        log_output = "\n".join(cm.output)
        # purpose removed from Paper A (1 abstract), mechanism from Paper B (1 abstract),
        # evaluation removed from Paper A (1 abstract)
        self.assertIn("purpose", log_output)
        self.assertIn("mechanism", log_output)
        self.assertIn("evaluation", log_output)

    def test_no_empty_fields_logs_zero_message(self):
        all_full = {
            "Paper A": "**context**: ctx.\n**purpose**: p.\n**mechanism**: m.\n**evaluation**: e.\n",
            "Paper B": "**context**: ctx.\n**purpose**: p.\n**mechanism**: m.\n**evaluation**: e.\n",
            "Paper C": "**context**: ctx.\n**purpose**: p.\n**mechanism**: m.\n**evaluation**: e.\n",
            "Paper D": "**context**: ctx.\n**purpose**: p.\n**mechanism**: m.\n**evaluation**: e.\n",
        }
        with self.assertLogs("logging_utils", level="INFO") as cm:
            self._run_main(PAIRWISE_DATASET, manipulate_map=all_full)
        log_output = "\n".join(cm.output)
        self.assertIn("no empty fields", log_output)

    # --- empty field removal ---

    def test_empty_fields_removed_from_output(self):
        manipulate_with_empty = {
            "Paper A": "**context**: ML context.\n**purpose**: \n**mechanism**: Mechanism A.\n**evaluation**:\n",
            "Paper B": "**context**: ML context.\n**purpose**: Purpose B.\n**mechanism**: \n**evaluation**: Eval B.\n",
            "Paper C": "**context**: NLP context.\n**purpose**: Purpose C.\n**mechanism**: Mechanism C.\n**evaluation**: Eval C.\n",
            "Paper D": "**context**: NLP context.\n**purpose**: Purpose D.\n**mechanism**: Mechanism D.\n**evaluation**: Eval D.\n",
        }
        result, _ = self._run_main(PAIRWISE_DATASET, manipulate_map=manipulate_with_empty)
        self.assertNotIn("**purpose**: \n", result[0]["ideas"][0])
        self.assertNotIn("**evaluation**:\n", result[0]["ideas"][0])
        self.assertNotIn("**mechanism**: \n", result[0]["ideas"][1])
        self.assertIn("**context**: ML context.", result[0]["ideas"][0])
        self.assertIn("**mechanism**: Mechanism A.", result[0]["ideas"][0])

    # --- fallback on LLM failure ---

    def test_fallback_to_original_abstract_on_failure(self):
        def failing_fake(paper, template, model_name, return_plan=False, debug_log=None):
            title = paper.get("title", "N/A")
            key = title if title != "N/A" else str(hash(paper["abstract"]))
            # Return original abstract (simulates fallback in _manipulate_paper)
            return key, paper["abstract"]

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, True)
        input_path = Path(tmpdir) / "iclr_test_instances.yaml"
        output_path = Path(tmpdir) / "out.yaml"
        with open(input_path, "w") as f:
            yaml.dump(PAIRWISE_DATASET, f)
        with patch("novelty_eval.benchmark_data.utility.apply_manipulation._manipulate_paper",
                   side_effect=failing_fake), \
             patch("sys.argv", ["s", "--dataset", str(input_path),
                                "--output", str(output_path),
                                "--model", "x", "--max-workers", "1"]):
            main()

        with open(output_path) as f:
            result = yaml.safe_load(f)
        for inst_id, instance in PAIRWISE_DATASET.items():
            for idx, orig_abstract in instance["ideas"].items():
                self.assertEqual(result[inst_id]["ideas"][idx], orig_abstract)

    # --- max instances ---

    def test_max_instances_truncates_dataset(self):
        result, _ = self._run_main(PAIRWISE_DATASET, extra_args=["--max-instances", "1"])
        self.assertEqual(len(result), 1)

    def test_max_instances_preserves_correct_instances(self):
        result, _ = self._run_main(PAIRWISE_DATASET, extra_args=["--max-instances", "1"])
        self.assertIn(0, result)
        self.assertNotIn(1, result)

    # --- pointwise format ---

    def test_pointwise_idea_replaced(self):
        pointwise_map = {
            "Paper X": "**context**: CV context.\n**purpose**: Purpose X.\n**mechanism**: Mech X.\n**evaluation**: Eval X.\n",
            "Paper Y": "**context**: CV context.\n**purpose**: Purpose Y.\n**mechanism**: Mech Y.\n**evaluation**: Eval Y.\n",
        }
        result, _ = self._run_main(POINTWISE_DATASET, manipulate_map=pointwise_map)
        self.assertEqual(result[0]["idea"], pointwise_map["Paper X"])
        self.assertEqual(result[1]["idea"], pointwise_map["Paper Y"])

    def test_pointwise_non_idea_fields_unchanged(self):
        pointwise_map = {"Paper X": "MANIP X", "Paper Y": "MANIP Y"}
        result, _ = self._run_main(POINTWISE_DATASET, manipulate_map=pointwise_map)
        for inst_id, instance in POINTWISE_DATASET.items():
            self.assertEqual(result[inst_id]["context"], instance["context"])
            self.assertEqual(result[inst_id]["label"], instance["label"])
            self.assertEqual(result[inst_id]["metadata"], instance["metadata"])

    # --- output file ---

    def test_output_is_valid_yaml(self):
        result, output_path = self._run_main(PAIRWISE_DATASET)
        self.assertIsInstance(result, dict)
        self.assertTrue(output_path.exists())

    def test_debug_log_is_written(self):
        _, output_path = self._run_main(PAIRWISE_DATASET)
        debug_log_path = output_path.parent / "manipulation_debug.md"
        self.assertTrue(debug_log_path.exists())


if __name__ == "__main__":
    unittest.main()
