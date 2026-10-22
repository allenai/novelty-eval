"""Tests for llm_negatives_reasoning_effort config parameter.

Covers:
  - _load_config defaults and YAML override
  - _generate_llm_negatives passes reasoning dict to prompt_openai_client correctly
"""
import asyncio
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from novelty_eval.benchmark_data.create_benchmark_instances import (
    _generate_llm_negatives,
    _load_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# _load_config
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadConfigReasoningEffort(unittest.TestCase):

    def _write_config(self, extra: dict) -> str:
        base = {
            "iclr_data": ["/tmp/fake.json"],
            "num_top_papers": 1,
            "num_bottom_papers": 1,
            "output_dir": "/tmp/out",
            "llm_negatives": True,
            "llm_negatives_model": "claude-sonnet-4-5",
            "llm_negatives_prompt": "templates/idea_generation.jinja2",
        }
        base.update(extra)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(base, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_default_is_none_when_key_absent(self):
        path = self._write_config({})
        args = _load_config(path)
        self.assertIsNone(args.llm_negatives_reasoning_effort)

    def test_null_value_is_none(self):
        path = self._write_config({"llm_negatives_reasoning_effort": None})
        args = _load_config(path)
        self.assertIsNone(args.llm_negatives_reasoning_effort)

    def test_high_effort_loaded(self):
        path = self._write_config({"llm_negatives_reasoning_effort": "high"})
        args = _load_config(path)
        self.assertEqual(args.llm_negatives_reasoning_effort, "high")

    def test_medium_effort_loaded(self):
        path = self._write_config({"llm_negatives_reasoning_effort": "medium"})
        args = _load_config(path)
        self.assertEqual(args.llm_negatives_reasoning_effort, "medium")

    def test_low_effort_loaded(self):
        path = self._write_config({"llm_negatives_reasoning_effort": "low"})
        args = _load_config(path)
        self.assertEqual(args.llm_negatives_reasoning_effort, "low")

    def test_other_llm_negatives_fields_unaffected(self):
        path = self._write_config({"llm_negatives_reasoning_effort": "high"})
        args = _load_config(path)
        self.assertEqual(args.llm_negatives_model, "claude-sonnet-4-5")
        self.assertTrue(args.llm_negatives)


# ─────────────────────────────────────────────────────────────────────────────
# _generate_llm_negatives — reasoning kwarg forwarding
# ─────────────────────────────────────────────────────────────────────────────

_FAKE_TEMPLATE_SRC = "Generate an idea for area: {{ area }}"

_DUMMY_ABSTRACT = "**context**: ctx.\n**purpose**: purpose."


def _run(coro):
    return asyncio.run(coro)


class TestGenerateLlmNegativesReasoningForwarding(unittest.TestCase):
    """Verify prompt_openai_client is called with the correct reasoning kwarg."""

    def _make_template_file(self) -> str:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jinja2", delete=False)
        tmp.write(_FAKE_TEMPLATE_SRC)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def _run_generate(self, reasoning_effort, mock_client):
        template_path = self._make_template_file()
        with patch(
            "novelty_eval.benchmark_data.create_benchmark_instances.prompt_openai_client",
            side_effect=mock_client,
        ):
            return _run(
                _generate_llm_negatives(
                    areas_with_counts={"machine learning": 1},
                    model_name="claude-sonnet-4-5",
                    prompt_template_path=template_path,
                    max_parallel=1,
                    reasoning_effort=reasoning_effort,
                )
            )

    def test_no_reasoning_effort_passes_none(self):
        calls = []

        def fake_client(prompt, engine, reasoning=None):
            calls.append(reasoning)
            return _DUMMY_ABSTRACT

        self._run_generate(None, fake_client)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0])

    def test_high_effort_passes_dict(self):
        calls = []

        def fake_client(prompt, engine, reasoning=None):
            calls.append(reasoning)
            return _DUMMY_ABSTRACT

        self._run_generate("high", fake_client)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], {"effort": "high"})

    def test_medium_effort_passes_dict(self):
        calls = []

        def fake_client(prompt, engine, reasoning=None):
            calls.append(reasoning)
            return _DUMMY_ABSTRACT

        self._run_generate("medium", fake_client)
        self.assertEqual(calls[0], {"effort": "medium"})

    def test_low_effort_passes_dict(self):
        calls = []

        def fake_client(prompt, engine, reasoning=None):
            calls.append(reasoning)
            return _DUMMY_ABSTRACT

        self._run_generate("low", fake_client)
        self.assertEqual(calls[0], {"effort": "low"})

    def test_result_contains_paper_on_success(self):
        def fake_client(prompt, engine, reasoning=None):
            return _DUMMY_ABSTRACT

        result = self._run_generate("high", fake_client)
        self.assertIn("machine learning", result)
        self.assertEqual(len(result["machine learning"]), 1)
        paper = result["machine learning"][0]
        self.assertIn("abstract", paper)
        self.assertTrue(paper.get("_llm_generated"))

    def test_empty_response_skips_paper(self):
        def fake_client(prompt, engine, reasoning=None):
            return ""

        result = self._run_generate("high", fake_client)
        self.assertEqual(result["machine learning"], [])

    def test_multiple_slots_make_multiple_calls(self):
        calls = []

        def fake_client(prompt, engine, reasoning=None):
            calls.append(reasoning)
            return _DUMMY_ABSTRACT

        template_path = self._make_template_file()
        with patch(
            "novelty_eval.benchmark_data.create_benchmark_instances.prompt_openai_client",
            side_effect=fake_client,
        ):
            _run(
                _generate_llm_negatives(
                    areas_with_counts={"ml": 2, "nlp": 1},
                    model_name="claude-sonnet-4-5",
                    prompt_template_path=template_path,
                    max_parallel=1,
                    reasoning_effort="high",
                )
            )

        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c == {"effort": "high"} for c in calls))


if __name__ == "__main__":
    unittest.main()
