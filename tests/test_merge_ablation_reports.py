import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.novelty_eval.ablation.merge_ablation_runs import _render_setup_tables

def test_render_setup_tables_with_significance():
    """Test that a single significance star is rendered for significant results."""
    setup = "pointwise"
    all_models = ["gpt-4"]
    baseline_metrics = {
        "gpt-4": {"accuracy": 0.5, "support": 100}
    }
    ablation_metrics = {
        "gpt-4": {"accuracy": 0.8, "support": 100}
    }
    metric_labels = {"accuracy": "Accuracy", "support": "Samples"}

    bootstrap_results = {
        ("gpt-4", "accuracy"): {"significant": True}
    }

    md_lines = _render_setup_tables(
        setup, all_models, baseline_metrics, ablation_metrics, metric_labels, bootstrap_results
    )

    md_content = "\n".join(md_lines)

    # Significant result gets one star
    assert "0.8000 (+30.0%)*" in md_content
    # No double stars — very_significant tier was removed
    assert "0.8000 (+30.0%)**" not in md_content

def test_render_setup_tables_no_significance():
    """Test that no stars are rendered when not significant."""
    setup = "pointwise"
    all_models = ["gpt-4"]
    baseline_metrics = {"gpt-4": {"accuracy": 0.5}}
    ablation_metrics = {"gpt-4": {"accuracy": 0.51}}
    metric_labels = {"accuracy": "Accuracy"}
    bootstrap_results = {
        ("gpt-4", "accuracy"): {"significant": False}
    }

    md_lines = _render_setup_tables(
        setup, all_models, baseline_metrics, ablation_metrics, metric_labels, bootstrap_results
    )
    md_content = "\n".join(md_lines)

    ablation_section = md_content.split("### Ablation Results")[1].split("###")[0]

    assert "0.5100 (+1.0%)" in ablation_section
    assert not re.search(r"\( \+1\.0% \)\*", ablation_section.replace(" ", ""))

def test_heatmap_significance_labels():
    """Test that a single star is used for significant results in the heatmap."""
    try:
        import matplotlib
    except ImportError:
        pytest.skip("matplotlib not installed; skipping heatmap test.")

    from src.novelty_eval.figures.viz import _generate_aggregated_delta_heatmap

    setup = "pointwise"
    metric_labels = {"accuracy": "Accuracy"}
    models = ["gpt-4"]
    baseline_metrics = {"gpt-4": {"accuracy": 0.5}}
    ablation_metrics = {"gpt-4": {"accuracy": 0.8}}
    out_path = Path("test_heatmap.png")

    bootstrap_results = {
        ("gpt-4", "accuracy"): {"significant": True}
    }

    with patch("matplotlib.pyplot.subplots") as mock_subplots:
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        _generate_aggregated_delta_heatmap(
            setup, metric_labels, models, baseline_metrics, ablation_metrics,
            out_path, bootstrap_results=bootstrap_results
        )

        # Significant result should produce exactly one star
        found_star = any("*" in str(call) for call in mock_ax.text.call_args_list)
        assert found_star is True
        found_double_star = any("**" in str(call) for call in mock_ax.text.call_args_list)
        assert found_double_star is False
