"""Tests for the _rebuild_negatives_mode code path in create_benchmark_instances.py.

Core invariant: NOTHING but the negative idea texts should change between the
source dataset and the rebuilt output.

Covered:
  - Positive idea text is preserved verbatim
  - Positive metadata is preserved verbatim
  - Positive slot indices are unchanged
  - expected_winners are unchanged
  - Instance IDs are preserved
  - Negative idea texts are replaced with new generated content
  - Negative metadata is rebuilt with the correct structure
  - Negative slot indices are unchanged
  - Multi-negative instances: both slots filled independently
  - Multi-area routing: area-A negatives never land in area-B instances
  - Shortfall / graceful degradation: original text retained when generator runs short
  - Error paths in async_main: generate_both_versions conflict, missing file
"""
import asyncio
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from novelty_eval.benchmark_data.create_benchmark_instances import (
    ManipulationDebugLog,
    _load_config,
    _rebuild_negatives_mode,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Three instances:
#   inst 0 – "machine learning", positive at slot 1, negative at slot 0
#   inst 1 – "machine learning", positive at slot 0, negative at slot 1
#   inst 2 – "computer vision",  positive at slot 2, negatives at slots 0 and 1
SOURCE_DATASET = {
    0: {
        "context": "machine learning",
        "expected_winners": [1],
        "ideas": {
            0: "ORIGINAL negative text ML-0",
            1: "ORIGINAL positive text ML-0",
        },
        "metadata": {
            0: {
                "title": "Neg-ML-0", "area": "machine learning",
                "rating": "3.00", "contribution": "2.00", "type": "NEGATIVE",
                "positive_signals": [], "negative_signals": ["weak baseline"],
                "similar_papers_mentioned": [],
            },
            1: {
                "title": "Pos-ML-0", "area": "machine learning",
                "rating": "7.50", "contribution": "3.50", "type": "POSITIVE",
                "positive_signals": ["novel approach"], "negative_signals": [],
                "similar_papers_mentioned": ["Paper X"],
            },
        },
    },
    1: {
        "context": "machine learning",
        "expected_winners": [0],
        "ideas": {
            0: "ORIGINAL positive text ML-1",
            1: "ORIGINAL negative text ML-1",
        },
        "metadata": {
            0: {
                "title": "Pos-ML-1", "area": "machine learning",
                "rating": "8.00", "contribution": "4.00", "type": "POSITIVE",
                "positive_signals": ["significant contribution"], "negative_signals": [],
                "similar_papers_mentioned": [],
            },
            1: {
                "title": "Neg-ML-1", "area": "machine learning",
                "rating": "4.00", "contribution": "2.00", "type": "NEGATIVE",
                "positive_signals": [], "negative_signals": ["incremental"],
                "similar_papers_mentioned": [],
            },
        },
    },
    2: {
        "context": "computer vision",
        "expected_winners": [2],
        "ideas": {
            0: "ORIGINAL negative text CV-0",
            1: "ORIGINAL negative text CV-1",
            2: "ORIGINAL positive text CV",
        },
        "metadata": {
            0: {
                "title": "Neg-CV-0", "area": "computer vision",
                "rating": "3.50", "contribution": "1.50", "type": "NEGATIVE",
                "positive_signals": [], "negative_signals": ["limited novelty"],
                "similar_papers_mentioned": [],
            },
            1: {
                "title": "Neg-CV-1", "area": "computer vision",
                "rating": "3.00", "contribution": "1.00", "type": "NEGATIVE",
                "positive_signals": [], "negative_signals": ["weak evaluation"],
                "similar_papers_mentioned": [],
            },
            2: {
                "title": "Pos-CV", "area": "computer vision",
                "rating": "8.00", "contribution": "4.00", "type": "POSITIVE",
                "positive_signals": ["strong results"], "negative_signals": [],
                "similar_papers_mentioned": [],
            },
        },
    },
}

# New negatives the mock generator will return
NEW_NEGATIVES = {
    "machine learning": [
        {"abstract": "NEW neg abstract ML-0", "title": "LLM-machine learning-0",
         "reviews": [], "_generated": True, "_llm_generated": True},
        {"abstract": "NEW neg abstract ML-1", "title": "LLM-machine learning-1",
         "reviews": [], "_generated": True, "_llm_generated": True},
    ],
    "computer vision": [
        {"abstract": "NEW neg abstract CV-0", "title": "LLM-computer vision-0",
         "reviews": [], "_generated": True, "_llm_generated": True},
        {"abstract": "NEW neg abstract CV-1", "title": "LLM-computer vision-1",
         "reviews": [], "_generated": True, "_llm_generated": True},
    ],
}


def _make_args(source_yaml_path: str, output_dir: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        rebuild_negatives_from_yaml=source_yaml_path,
        output_dir=output_dir,
        llm_negatives=True,
        llm_negatives_model="fake-model",
        llm_negatives_prompt="fake-template.jinja2",
        llm_negatives_reasoning_effort=None,
        generate_negatives=False,
        max_workers=1,
        md_report_batch_size=None,
    )


def _run(coro):
    return asyncio.run(coro)


class RebuildNegativesTestBase(unittest.TestCase):
    """Base class: writes the source YAML, runs the rebuild, loads the output."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

        self.source_path = os.path.join(self.tmpdir, "source.yaml")
        with open(self.source_path, "w") as f:
            yaml.dump(SOURCE_DATASET, f, default_flow_style=False, allow_unicode=True)

        self.output_base = os.path.join(self.tmpdir, "output")
        os.makedirs(self.output_base)

        args = _make_args(self.source_path, self.output_base)
        debug_log = ManipulationDebugLog()

        mock_gen = AsyncMock(return_value=NEW_NEGATIVES)
        with patch(
            "novelty_eval.benchmark_data.create_benchmark_instances._generate_llm_negatives",
            new=mock_gen,
        ):
            _run(_rebuild_negatives_mode(args, debug_log))

        # Locate the single timestamped output subdirectory
        output_subdirs = [
            d for d in Path(self.output_base).iterdir()
            if d.is_dir()
        ]
        self.assertEqual(len(output_subdirs), 1, "Expected exactly one timestamped output dir")
        self.output_dir = output_subdirs[0]

        output_yaml = self.output_dir / "benchmark_instances.yaml"
        self.assertTrue(output_yaml.exists(), "benchmark_instances.yaml not found in output dir")
        with open(output_yaml) as f:
            self.result = yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Positive-slot preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestPositiveIdeaTextPreserved(RebuildNegativesTestBase):

    def test_positive_idea_text_inst0(self):
        self.assertEqual(
            self.result[0]["ideas"][1],
            SOURCE_DATASET[0]["ideas"][1],
        )

    def test_positive_idea_text_inst1(self):
        self.assertEqual(
            self.result[1]["ideas"][0],
            SOURCE_DATASET[1]["ideas"][0],
        )

    def test_positive_idea_text_inst2(self):
        self.assertEqual(
            self.result[2]["ideas"][2],
            SOURCE_DATASET[2]["ideas"][2],
        )

    def test_positive_idea_not_replaced_with_new_negative_text(self):
        all_new_texts = {p["abstract"] for papers in NEW_NEGATIVES.values() for p in papers}
        for inst_id, inst in self.result.items():
            for winner_idx in inst["expected_winners"]:
                self.assertNotIn(
                    inst["ideas"][winner_idx], all_new_texts,
                    f"Positive slot {winner_idx} in instance {inst_id} was overwritten with a new negative",
                )


class TestPositiveMetadataPreserved(RebuildNegativesTestBase):

    def _positive_meta(self, inst_id):
        src_inst = SOURCE_DATASET[inst_id]
        winner_idx = src_inst["expected_winners"][0]
        return src_inst["metadata"][winner_idx]

    def test_positive_metadata_inst0_verbatim(self):
        winner_idx = self.result[0]["expected_winners"][0]
        self.assertEqual(self.result[0]["metadata"][winner_idx], self._positive_meta(0))

    def test_positive_metadata_inst1_verbatim(self):
        winner_idx = self.result[1]["expected_winners"][0]
        self.assertEqual(self.result[1]["metadata"][winner_idx], self._positive_meta(1))

    def test_positive_metadata_inst2_verbatim(self):
        winner_idx = self.result[2]["expected_winners"][0]
        self.assertEqual(self.result[2]["metadata"][winner_idx], self._positive_meta(2))

    def test_positive_rating_unchanged(self):
        self.assertEqual(self.result[0]["metadata"][1]["rating"], "7.50")

    def test_positive_signals_unchanged(self):
        self.assertEqual(
            self.result[0]["metadata"][1]["positive_signals"],
            ["novel approach"],
        )

    def test_positive_similar_papers_unchanged(self):
        self.assertEqual(
            self.result[0]["metadata"][1]["similar_papers_mentioned"],
            ["Paper X"],
        )

    def test_positive_type_field_unchanged(self):
        for inst_id, inst in self.result.items():
            for winner_idx in inst["expected_winners"]:
                self.assertEqual(inst["metadata"][winner_idx]["type"], "POSITIVE")


class TestPositiveSlotIndexPreserved(RebuildNegativesTestBase):

    def test_inst0_positive_still_at_slot_1(self):
        self.assertIn(1, self.result[0]["ideas"])
        self.assertEqual(
            self.result[0]["ideas"][1], SOURCE_DATASET[0]["ideas"][1]
        )

    def test_inst1_positive_still_at_slot_0(self):
        self.assertIn(0, self.result[1]["ideas"])
        self.assertEqual(
            self.result[1]["ideas"][0], SOURCE_DATASET[1]["ideas"][0]
        )

    def test_inst2_positive_still_at_slot_2(self):
        self.assertIn(2, self.result[2]["ideas"])
        self.assertEqual(
            self.result[2]["ideas"][2], SOURCE_DATASET[2]["ideas"][2]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Instance structure
# ─────────────────────────────────────────────────────────────────────────────

class TestInstanceStructurePreserved(RebuildNegativesTestBase):

    def test_same_instance_ids(self):
        self.assertEqual(set(self.result.keys()), set(SOURCE_DATASET.keys()))

    def test_same_instance_count(self):
        self.assertEqual(len(self.result), len(SOURCE_DATASET))

    def test_expected_winners_unchanged_inst0(self):
        self.assertEqual(self.result[0]["expected_winners"], SOURCE_DATASET[0]["expected_winners"])

    def test_expected_winners_unchanged_inst1(self):
        self.assertEqual(self.result[1]["expected_winners"], SOURCE_DATASET[1]["expected_winners"])

    def test_expected_winners_unchanged_inst2(self):
        self.assertEqual(self.result[2]["expected_winners"], SOURCE_DATASET[2]["expected_winners"])

    def test_context_unchanged(self):
        for inst_id in SOURCE_DATASET:
            self.assertEqual(self.result[inst_id]["context"], SOURCE_DATASET[inst_id]["context"])

    def test_idea_slot_indices_unchanged(self):
        for inst_id in SOURCE_DATASET:
            self.assertEqual(
                sorted(self.result[inst_id]["ideas"].keys()),
                sorted(SOURCE_DATASET[inst_id]["ideas"].keys()),
            )

    def test_metadata_slot_indices_unchanged(self):
        for inst_id in SOURCE_DATASET:
            self.assertEqual(
                sorted(self.result[inst_id]["metadata"].keys()),
                sorted(SOURCE_DATASET[inst_id]["metadata"].keys()),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Negative-slot replacement
# ─────────────────────────────────────────────────────────────────────────────

class TestNegativeIdeaTextReplaced(RebuildNegativesTestBase):

    def _neg_indices(self, inst_id):
        src = SOURCE_DATASET[inst_id]
        winners = set(src["expected_winners"])
        return [i for i in src["ideas"] if i not in winners]

    def test_inst0_negative_text_changed(self):
        neg_idx = self._neg_indices(0)[0]
        self.assertNotEqual(
            self.result[0]["ideas"][neg_idx],
            SOURCE_DATASET[0]["ideas"][neg_idx],
        )

    def test_inst1_negative_text_changed(self):
        neg_idx = self._neg_indices(1)[0]
        self.assertNotEqual(
            self.result[1]["ideas"][neg_idx],
            SOURCE_DATASET[1]["ideas"][neg_idx],
        )

    def test_inst2_both_negative_texts_changed(self):
        for neg_idx in self._neg_indices(2):
            self.assertNotEqual(
                self.result[2]["ideas"][neg_idx],
                SOURCE_DATASET[2]["ideas"][neg_idx],
            )

    def test_inst0_negative_text_is_from_generator(self):
        self.assertEqual(self.result[0]["ideas"][0], "NEW neg abstract ML-0")

    def test_inst1_negative_text_is_from_generator(self):
        self.assertEqual(self.result[1]["ideas"][1], "NEW neg abstract ML-1")

    def test_inst2_negative_slot0_is_from_generator(self):
        self.assertEqual(self.result[2]["ideas"][0], "NEW neg abstract CV-0")

    def test_inst2_negative_slot1_is_from_generator(self):
        self.assertEqual(self.result[2]["ideas"][1], "NEW neg abstract CV-1")


class TestNegativeMetadataRebuilt(RebuildNegativesTestBase):

    def _neg_meta(self, inst_id, neg_slot):
        return self.result[inst_id]["metadata"][neg_slot]

    def test_type_is_negative(self):
        for inst_id, inst in self.result.items():
            winners = set(inst["expected_winners"])
            for idx, meta in inst["metadata"].items():
                if idx not in winners:
                    self.assertEqual(meta["type"], "NEGATIVE", f"inst {inst_id} slot {idx}")

    def test_rating_is_na(self):
        meta = self._neg_meta(0, 0)
        self.assertEqual(meta["rating"], "N/A")

    def test_contribution_is_na(self):
        meta = self._neg_meta(0, 0)
        self.assertEqual(meta["contribution"], "N/A")

    def test_positive_signals_empty(self):
        meta = self._neg_meta(0, 0)
        self.assertEqual(meta["positive_signals"], [])

    def test_negative_signals_empty(self):
        meta = self._neg_meta(0, 0)
        self.assertEqual(meta["negative_signals"], [])

    def test_similar_papers_empty(self):
        meta = self._neg_meta(0, 0)
        self.assertEqual(meta["similar_papers_mentioned"], [])

    def test_area_matches_context(self):
        for inst_id, inst in self.result.items():
            context = inst["context"]
            winners = set(inst["expected_winners"])
            for idx, meta in inst["metadata"].items():
                if idx not in winners:
                    self.assertEqual(meta["area"], context, f"inst {inst_id} slot {idx}")

    def test_required_keys_present(self):
        required = {"title", "area", "rating", "contribution", "type",
                    "positive_signals", "negative_signals", "similar_papers_mentioned"}
        for inst_id, inst in self.result.items():
            winners = set(inst["expected_winners"])
            for idx, meta in inst["metadata"].items():
                if idx not in winners:
                    self.assertTrue(
                        required.issubset(meta.keys()),
                        f"Missing keys in inst {inst_id} slot {idx}: {required - meta.keys()}",
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-area routing
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiAreaRouting(RebuildNegativesTestBase):

    def test_ml_negatives_not_in_cv_instance(self):
        ml_new_texts = {p["abstract"] for p in NEW_NEGATIVES["machine learning"]}
        cv_winners = set(SOURCE_DATASET[2]["expected_winners"])
        for idx in SOURCE_DATASET[2]["ideas"]:
            if idx not in cv_winners:
                self.assertNotIn(
                    self.result[2]["ideas"][idx], ml_new_texts,
                    "A machine-learning negative was placed in a computer-vision instance",
                )

    def test_cv_negatives_not_in_ml_instances(self):
        cv_new_texts = {p["abstract"] for p in NEW_NEGATIVES["computer vision"]}
        for inst_id in (0, 1):
            ml_winners = set(SOURCE_DATASET[inst_id]["expected_winners"])
            for idx in SOURCE_DATASET[inst_id]["ideas"]:
                if idx not in ml_winners:
                    self.assertNotIn(
                        self.result[inst_id]["ideas"][idx], cv_new_texts,
                        f"A CV negative was placed in ML instance {inst_id}",
                    )

    def test_cv_negative_area_in_metadata(self):
        cv_winners = set(self.result[2]["expected_winners"])
        for idx, meta in self.result[2]["metadata"].items():
            if idx not in cv_winners:
                self.assertEqual(meta["area"], "computer vision")


# ─────────────────────────────────────────────────────────────────────────────
# Shortfall / graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

class TestShortfallGracefulDegradation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

        self.source_path = os.path.join(self.tmpdir, "source.yaml")
        with open(self.source_path, "w") as f:
            yaml.dump(SOURCE_DATASET, f)

        self.output_base = os.path.join(self.tmpdir, "output")
        os.makedirs(self.output_base)

    def _run_with_negatives(self, negatives_map):
        args = _make_args(self.source_path, self.output_base)
        debug_log = ManipulationDebugLog()
        mock_gen = AsyncMock(return_value=negatives_map)
        with patch(
            "novelty_eval.benchmark_data.create_benchmark_instances._generate_llm_negatives",
            new=mock_gen,
        ):
            _run(_rebuild_negatives_mode(args, debug_log))

        subdirs = [d for d in Path(self.output_base).iterdir() if d.is_dir()]
        with open(subdirs[-1] / "benchmark_instances.yaml") as f:
            return yaml.safe_load(f)

    def test_shortfall_retains_original_negative_text(self):
        # Only 1 ML negative generated instead of 2 — inst 1 should fall back
        short_negatives = {
            "machine learning": [
                {"abstract": "NEW neg ML-only-one", "title": "LLM-ml-0",
                 "reviews": [], "_generated": True, "_llm_generated": True},
            ],
            "computer vision": NEW_NEGATIVES["computer vision"],
        }
        result = self._run_with_negatives(short_negatives)

        # inst 0 gets the one generated negative
        self.assertEqual(result[0]["ideas"][0], "NEW neg ML-only-one")
        # inst 1's negative slot falls back to the original text
        self.assertEqual(result[1]["ideas"][1], SOURCE_DATASET[1]["ideas"][1])

    def test_shortfall_does_not_affect_positive_slot(self):
        short_negatives = {
            "machine learning": [],
            "computer vision": NEW_NEGATIVES["computer vision"],
        }
        result = self._run_with_negatives(short_negatives)

        # Positive slots must be intact regardless of shortfall
        self.assertEqual(result[0]["ideas"][1], SOURCE_DATASET[0]["ideas"][1])
        self.assertEqual(result[1]["ideas"][0], SOURCE_DATASET[1]["ideas"][0])


# ─────────────────────────────────────────────────────────────────────────────
# _load_config: rebuild_negatives_from_yaml default
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadConfigRebuildKey(unittest.TestCase):

    def _write_config(self, extra: dict = None) -> str:
        base = {
            "iclr_data": ["/tmp/fake.json"],
            "num_top_papers": 1,
            "num_bottom_papers": 1,
            "output_dir": "/tmp/out",
        }
        if extra:
            base.update(extra)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(base, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_default_is_none_when_absent(self):
        args = _load_config(self._write_config())
        self.assertIsNone(args.rebuild_negatives_from_yaml)

    def test_null_value_is_none(self):
        args = _load_config(self._write_config({"rebuild_negatives_from_yaml": None}))
        self.assertIsNone(args.rebuild_negatives_from_yaml)

    def test_path_value_loaded(self):
        args = _load_config(self._write_config({"rebuild_negatives_from_yaml": "/some/path.yaml"}))
        self.assertEqual(args.rebuild_negatives_from_yaml, "/some/path.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# Error paths (async_main mode-check logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorPaths(unittest.TestCase):
    """Tests for the early-return guards in async_main."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def _write_config(self, extra: dict) -> str:
        base = {
            "iclr_data": ["/tmp/fake.json"],
            "num_top_papers": 1,
            "num_bottom_papers": 1,
            "output_dir": os.path.join(self.tmpdir, "output"),
            "llm_negatives": True,
            "llm_negatives_model": "fake-model",
            "llm_negatives_prompt": os.path.join(self.tmpdir, "fake.jinja2"),
        }
        # Create dummy template so validation passes
        with open(base["llm_negatives_prompt"], "w") as f:
            f.write("Generate an idea for {{ area }}")
        base.update(extra)
        path = os.path.join(self.tmpdir, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(base, f)
        return path

    def test_generate_both_versions_conflict_does_not_call_rebuild(self):
        from novelty_eval.benchmark_data.create_benchmark_instances import async_main
        cfg = self._write_config({
            "generate_both_versions": True,
            "model_name": "fake-model",
            "rebuild_negatives_from_yaml": "/nonexistent/source.yaml",
        })
        mock_rebuild = AsyncMock()
        with patch("sys.argv", ["prog", "--config", cfg]), \
             patch("novelty_eval.benchmark_data.create_benchmark_instances._rebuild_negatives_mode",
                   new=mock_rebuild), \
             patch("novelty_eval.benchmark_data.create_benchmark_instances.load_secrets"):
            _run(async_main())
        mock_rebuild.assert_not_called()

    def test_missing_source_yaml_does_not_call_rebuild(self):
        from novelty_eval.benchmark_data.create_benchmark_instances import async_main
        cfg = self._write_config({
            "rebuild_negatives_from_yaml": "/this/path/does/not/exist.yaml",
        })
        mock_rebuild = AsyncMock()
        with patch("sys.argv", ["prog", "--config", cfg]), \
             patch("novelty_eval.benchmark_data.create_benchmark_instances._rebuild_negatives_mode",
                   new=mock_rebuild), \
             patch("novelty_eval.benchmark_data.create_benchmark_instances.load_secrets"):
            _run(async_main())
        mock_rebuild.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Output files
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputFiles(RebuildNegativesTestBase):

    def test_benchmark_instances_yaml_exists(self):
        self.assertTrue((self.output_dir / "benchmark_instances.yaml").exists())

    def test_readable_summary_exists(self):
        self.assertTrue((self.output_dir / "benchmark_instances_readable.txt").exists())

    def test_md_report_exists(self):
        self.assertTrue((self.output_dir / "benchmark_instances_report.md").exists())

    def test_manipulation_debug_log_exists(self):
        self.assertTrue((self.output_dir / "manipulation_debug.md").exists())

    def test_output_yaml_is_valid_dict(self):
        self.assertIsInstance(self.result, dict)


if __name__ == "__main__":
    unittest.main()
