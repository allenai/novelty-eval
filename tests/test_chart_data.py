import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.novelty_eval.figures import common, unified_chart_data
from src.novelty_eval.analysis.stats_one_pass import (
    compute_bootstrap_results_pairwise_all,
)

MODELS = ["judge-a", "judge-b"]


def _panel(title, delta):
    """One panel where every model moves by `delta` from a 0.60 baseline."""
    bs = {}
    for m in MODELS:
        bs[(m, "pairwise_accuracy")] = {
            "significant": True, "reliable": True,
            "ci_low": delta - 0.02, "ci_high": delta + 0.02,
        }
    return {
        "title": title,
        "baseline_metrics": {m: {"pairwise_accuracy": 0.60} for m in MODELS},
        "ablation_metrics": {m: {"pairwise_accuracy": 0.60 + delta} for m in MODELS},
        "bootstrap_results": bs,
    }


# --- bootstrap now surfaces the interval, not just the booleans -------------

def test_pairwise_bootstrap_returns_ci_bounds():
    """CI bounds must be returned so charts need no re-bootstrap."""
    # (is_correct, is_tie, gt_winner) — gt must match positionally across the pair.
    base = [(1.0, 0.0, 0.0)] * 50 + [(0.0, 0.0, 0.0)] * 50
    abl = [(1.0, 0.0, 0.0)] * 80 + [(0.0, 0.0, 0.0)] * 20
    res = compute_bootstrap_results_pairwise_all(base, abl, n_resamples=500)
    entry = res["pairwise_accuracy"]
    assert {"significant", "reliable", "ci_low", "ci_high"} <= set(entry)
    assert entry["ci_low"] is not None and entry["ci_high"] is not None
    assert entry["ci_low"] <= entry["ci_high"]


def test_degenerate_bootstrap_ci_is_none_not_nan():
    """Non-finite bounds become None so the payload stays JSON-serialisable."""
    same = [(1.0, 0.0, 0.0)] * 40
    res = compute_bootstrap_results_pairwise_all(same, same, n_resamples=200)
    entry = res["pairwise_accuracy"]
    assert entry["ci_low"] is None and entry["ci_high"] is None
    json.dumps(entry)  # must not raise


# --- on-disk format ---------------------------------------------------------

def test_v1_data_still_loads(tmp_path):
    """Existing v1 files must keep working; they simply carry no interval."""
    p = tmp_path / "unified_pairwise.json"
    p.write_text(json.dumps({
        "schema_version": 1, "generated_at": "", "setup": "pairwise",
        "name_suffix": "", "variant": "unfiltered", "report_file": "r.txt",
        "all_models": MODELS, "metric_labels": {}, "available_metrics": {},
        "rendered": [], "panels": {},
    }))
    assert unified_chart_data.load(p)["schema_version"] == 1


def test_unknown_schema_version_is_rejected(tmp_path):
    p = tmp_path / "unified_pairwise.json"
    p.write_text(json.dumps({"schema_version": 99, "panels": {}}))
    with pytest.raises(ValueError, match="unsupported schema_version"):
        unified_chart_data.load(p)


def test_ci_bounds_survive_a_dump_load_round_trip(tmp_path):
    out = unified_chart_data.dump(
        tmp_path / "unified_pairwise.json",
        setup="pairwise", name_suffix="", report_file="r.txt",
        all_models=MODELS, metric_labels={"pairwise_accuracy": "Acc"},
        available_metrics={"pairwise_accuracy": "Acc"},
        rendered=["vague_criterion"],
        panels={"vague_criterion": dict(_panel("Vague", -0.05),
                                        canonical="vague_criterion")},
    )
    entry = unified_chart_data.load(out)["panels"]["vague_criterion"][
        "bootstrap_results"][("judge-a", "pairwise_accuracy")]
    assert entry["ci_low"] == pytest.approx(-0.07)
    assert entry["ci_high"] == pytest.approx(-0.03)


# --- what every figure has to agree on (common.py) --------------------------

def test_row_key_strips_track_and_setup_tokens():
    """One row entry must cover both setups, so both tokens come off."""
    assert common.row_key("convert_to_plan_form_pairwise_human") == "convert_to_plan_form"
    assert common.row_key("convert_to_plan_form_pointwise_vanilla") == "convert_to_plan_form"
    assert common.row_key("vague_criterion") == "vague_criterion"
    # A prefix that happens to be a token must not be touched.
    assert common.row_key("pointwise_retrieval") == "pointwise_retrieval"
    # A name that is nothing but a token must survive intact.
    assert common.row_key("pairwise") == "pairwise"


def test_chart_data_path_accepts_a_merge_dir_or_its_charts_subdir(tmp_path):
    """Older merges wrote the files at the top level; both must resolve."""
    flat = common.chart_data_path(tmp_path, "pairwise", "filtered")
    assert flat == tmp_path / "unified_pairwise_filtered.json"

    (tmp_path / "unified_charts").mkdir()
    nested = common.chart_data_path(tmp_path, "pairwise", "filtered")
    assert nested == tmp_path / "unified_charts" / "unified_pairwise_filtered.json"
    # Only the unfiltered variant drops the suffix.
    assert common.chart_data_path(tmp_path, "pointwise", "unfiltered").name == \
        "unified_pointwise.json"


def test_unsupported_cells_are_blanked_across_track_variants():
    """A judge that cannot honour the ablation is dropped, however it is named."""
    ablation, model = next(iter(common.UNSUPPORTED_CELLS))
    data = {"panels": {
        f"{ablation}_pairwise_human": {
            "ablation_metrics": {model: {"m": 1.0}, "other-judge": {"m": 2.0}}},
        "vague_criterion": {"ablation_metrics": {model: {"m": 3.0}}},
    }}
    panels = common.drop_unsupported_cells(data)["panels"]
    # Blanked under the ablation it is unsupported for...
    assert model not in panels[f"{ablation}_pairwise_human"]["ablation_metrics"]
    assert "other-judge" in panels[f"{ablation}_pairwise_human"]["ablation_metrics"]
    # ...and left alone everywhere else.
    assert model in panels["vague_criterion"]["ablation_metrics"]
