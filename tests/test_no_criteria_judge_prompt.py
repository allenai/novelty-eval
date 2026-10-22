"""Tests for the no_criteria judge prompt templates and their run_benchmark integration."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import novelty_eval.judge as judge_module
from novelty_eval.judge import (
    get_pairwise_novelty_prompt,
    get_pointwise_novelty_prompt,
)
from novelty_eval.run_benchmark import _load_config

_TEMPLATES_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "novelty_eval" / "templates"
)
_ABLATIONS_YAML = (
    Path(__file__).resolve().parents[1]
    / "src" / "novelty_eval" / "ablation" / "ablations.yaml"
)

_SENTINEL = "UNIQUE_CRITERION_XYZ_99887766"


class TestNoCriteriaTemplateStructure(unittest.TestCase):
    """no_criteria templates should be identical to the originals except for the removed criteria block."""

    def _read(self, filename: str) -> str:
        return (_TEMPLATES_DIR / filename).read_text(encoding="utf-8")

    def test_pairwise_content_before_criteria_is_identical(self):
        full = self._read("pairwise_novelty.jinja2")
        nc = self._read("pairwise_novelty_no_criteria.jinja2")
        full_before = full.split("Compare these ideas based on the following criteria:")[0]
        nc_before = nc.split("**Comparative Logic:**")[0]
        self.assertEqual(full_before.strip(), nc_before.strip())

    def test_pairwise_content_after_criteria_is_identical(self):
        """The logic + response-format lines between the criteria block and the JSON block
        should be identical in both pairwise templates."""
        full = self._read("pairwise_novelty.jinja2")
        nc = self._read("pairwise_novelty_no_criteria.jinja2")
        full_middle = full.split("**Comparative Logic:**")[1].split("```json")[0]
        nc_middle = nc.split("**Comparative Logic:**")[1].split("```json")[0]
        self.assertEqual(full_middle.strip(), nc_middle.strip())

    def test_pointwise_content_before_criteria_is_identical(self):
        full = self._read("pointwise_novelty.jinja2")
        nc = self._read("pointwise_novelty_no_criteria.jinja2")
        full_before = full.split("Judge this idea based on the following criteria:")[0]
        nc_before = nc.split("** Review Logic:")[0]
        self.assertEqual(full_before.strip(), nc_before.strip())

    def test_pointwise_content_after_criteria_is_identical(self):
        full = self._read("pointwise_novelty.jinja2")
        nc = self._read("pointwise_novelty_no_criteria.jinja2")
        full_middle = full.split("** Review Logic:")[1].split("```json")[0]
        nc_middle = nc.split("** Review Logic:")[1].split("```json")[0]
        self.assertEqual(full_middle.strip(), nc_middle.strip())

    def test_no_criteria_templates_omit_criteria_section(self):
        for filename in (
            "pairwise_novelty_no_criteria.jinja2",
            "pointwise_novelty_no_criteria.jinja2",
        ):
            content = self._read(filename)
            with self.subTest(template=filename):
                self.assertNotIn("following criteria", content)
                self.assertNotIn("evaluation_criteria.items()", content)

    def test_no_criteria_templates_hardcode_novelty_in_json_block(self):
        for filename in (
            "pairwise_novelty_no_criteria.jinja2",
            "pointwise_novelty_no_criteria.jinja2",
        ):
            content = self._read(filename)
            with self.subTest(template=filename):
                self.assertIn('{"novelty": <0 or 1>}', content)


class TestNoCriteriaTemplateRendering(unittest.TestCase):
    """Rendered no_criteria prompts must not include criterion descriptions, but must
    retain idea text and the hardcoded JSON output block."""

    def setUp(self):
        self._orig_pairwise = judge_module._PAIRWISE_JUDGE_TEMPLATE
        self._orig_pointwise = judge_module._POINTWISE_JUDGE_TEMPLATE

    def tearDown(self):
        judge_module._PAIRWISE_JUDGE_TEMPLATE = self._orig_pairwise
        judge_module._POINTWISE_JUDGE_TEMPLATE = self._orig_pointwise

    def test_pairwise_no_criteria_omits_criterion_description(self):
        judge_module._PAIRWISE_JUDGE_TEMPLATE = "pairwise_novelty_no_criteria.jinja2"
        prompt = get_pairwise_novelty_prompt(
            {"text": "Idea zero text", "related_work": ""},
            {"text": "Idea one text", "related_work": ""},
            {"novelty": _SENTINEL},
        )
        self.assertNotIn(_SENTINEL, prompt)
        self.assertIn("Idea zero text", prompt)
        self.assertIn("Idea one text", prompt)
        self.assertIn('{"novelty": <0 or 1>}', prompt)

    def test_pointwise_no_criteria_omits_criterion_description(self):
        judge_module._POINTWISE_JUDGE_TEMPLATE = "pointwise_novelty_no_criteria.jinja2"
        prompt = get_pointwise_novelty_prompt("Some idea text", {"novelty": _SENTINEL})
        self.assertNotIn(_SENTINEL, prompt)
        self.assertIn("Some idea text", prompt)
        self.assertIn('{"novelty": <0 or 1>}', prompt)

    def test_pairwise_full_template_does_include_criterion_description(self):
        """Sanity check: the default pairwise template renders the criterion description."""
        prompt = get_pairwise_novelty_prompt(
            {"text": "Idea zero", "related_work": ""},
            {"text": "Idea one", "related_work": ""},
            {"novelty": _SENTINEL},
        )
        self.assertIn(_SENTINEL, prompt)

    def test_pointwise_full_template_does_include_criterion_description(self):
        prompt = get_pointwise_novelty_prompt("Some idea", {"novelty": _SENTINEL})
        self.assertIn(_SENTINEL, prompt)


class TestNoCriteriaRunBenchmarkWiring(unittest.TestCase):
    """_load_config must expose the judge template params, and setting the module-level
    variables must affect what get_*_novelty_prompt renders."""

    def setUp(self):
        self._orig_pairwise = judge_module._PAIRWISE_JUDGE_TEMPLATE
        self._orig_pointwise = judge_module._POINTWISE_JUDGE_TEMPLATE

    def tearDown(self):
        judge_module._PAIRWISE_JUDGE_TEMPLATE = self._orig_pairwise
        judge_module._POINTWISE_JUDGE_TEMPLATE = self._orig_pointwise

    def _write_config(self, **kwargs) -> str:
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump({'test_inputs': 'in.yaml', 'output_file': 'out.json', 'n': 1, **kwargs}, f)
        f.close()
        return f.name

    def test_config_loads_pairwise_judge_template(self):
        path = self._write_config(pairwise_judge_template='pairwise_novelty_no_criteria.jinja2')
        try:
            args = _load_config(path)
            self.assertEqual(args.pairwise_judge_template, 'pairwise_novelty_no_criteria.jinja2')
        finally:
            os.remove(path)

    def test_config_loads_pointwise_judge_template(self):
        path = self._write_config(pointwise_judge_template='pointwise_novelty_no_criteria.jinja2')
        try:
            args = _load_config(path)
            self.assertEqual(args.pointwise_judge_template, 'pointwise_novelty_no_criteria.jinja2')
        finally:
            os.remove(path)

    def test_template_override_propagates_to_rendered_prompt(self):
        """After the run_benchmark wiring sets the module variable, the rendered prompt
        must reflect the new template (no criterion description)."""
        path = self._write_config(pairwise_judge_template='pairwise_novelty_no_criteria.jinja2')
        try:
            args = _load_config(path)
            if getattr(args, 'pairwise_judge_template', None):
                judge_module._PAIRWISE_JUDGE_TEMPLATE = args.pairwise_judge_template
            prompt = get_pairwise_novelty_prompt(
                {"text": "idea A", "related_work": ""},
                {"text": "idea B", "related_work": ""},
                {"novelty": _SENTINEL},
            )
            self.assertNotIn(_SENTINEL, prompt)
            self.assertIn('{"novelty": <0 or 1>}', prompt)
        finally:
            os.remove(path)


class TestNoCriteriaAblationYaml(unittest.TestCase):
    """ablations.yaml must declare the no_criteria_prompt ablation with valid template file refs."""

    def _load_ablations(self) -> dict:
        with open(_ABLATIONS_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_no_criteria_prompt_entry_exists(self):
        data = self._load_ablations()
        self.assertIn("no_criteria_prompt", data.get("ablations", {}))

    def test_no_criteria_prompt_sweep_keys_present(self):
        data = self._load_ablations()
        sweep = data["ablations"]["no_criteria_prompt"].get("sweep", {})
        self.assertIn("pairwise_judge_template", sweep)
        self.assertIn("pointwise_judge_template", sweep)
        self.assertEqual(sweep["pairwise_judge_template"], "pairwise_novelty_no_criteria.jinja2")
        self.assertEqual(sweep["pointwise_judge_template"], "pointwise_novelty_no_criteria.jinja2")

    def test_no_criteria_prompt_template_files_exist_on_disk(self):
        data = self._load_ablations()
        sweep = data["ablations"]["no_criteria_prompt"].get("sweep", {})
        for key in ("pairwise_judge_template", "pointwise_judge_template"):
            filename = sweep.get(key)
            path = _TEMPLATES_DIR / filename
            self.assertTrue(path.exists(), f"Template file missing: {path}")


if __name__ == "__main__":
    unittest.main()
