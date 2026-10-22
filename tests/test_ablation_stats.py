import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.novelty_eval.analysis.stats_one_pass import (
    compute_bootstrap_results_pointwise_all,
    compute_bootstrap_results_pairwise_all,
)

def test_pointwise_bootstrap_significance():
    """Test that a clear improvement is marked as significant."""
    n = 100
    labels = [1] * n
    base_preds = [1] * 50 + [0] * 50
    abl_preds = [1] * 80 + [0] * 20

    base_pl = list(zip(base_preds, labels))
    abl_pl = list(zip(abl_preds, labels))

    results = compute_bootstrap_results_pointwise_all(base_pl, abl_pl, n_resamples=1000)

    assert "accuracy" in results
    assert results["accuracy"]["significant"]

def test_pointwise_bootstrap_no_difference():
    """Test that identical results are not significant."""
    n = 100
    labels = [1] * n
    base_preds = [1] * 50 + [0] * 50
    abl_preds = [1] * 50 + [0] * 50

    base_pl = list(zip(base_preds, labels))
    abl_pl = list(zip(abl_preds, labels))

    results = compute_bootstrap_results_pointwise_all(base_pl, abl_pl, n_resamples=1000)

    assert not results["accuracy"]["significant"]

def test_pairwise_bootstrap_tie_handling():
    """Test that moving from wrong to tie is handled without errors."""
    n = 20
    # Base: all wrong, Abl: all ties — constant delta, degenerate BCa
    base_pw = [(0.0, 0.0, 1.0)] * n
    abl_pw = [(0.0, 1.0, 1.0)] * n

    results = compute_bootstrap_results_pairwise_all(base_pw, abl_pw, n_resamples=1000)

    # Degenerate case (constant delta across all resamples): BCa CI is NaN → non-significant
    assert "pairwise_accuracy" in results
    assert not results["pairwise_accuracy"]["significant"]

def test_paired_advantage_real_case():
    """
    Paired bootstrap is sensitive to small but consistent improvements.
    95 identical pairs + 5 where ablation wins; the CI should exclude zero.
    """
    n = 100
    base_pw = [(1.0, 0.0, 1.0)] * 95 + [(0.0, 0.0, 1.0)] * 5
    abl_pw  = [(1.0, 0.0, 1.0)] * 100

    results = compute_bootstrap_results_pairwise_all(base_pw, abl_pw, n_resamples=2000)

    assert results["pairwise_accuracy"]["significant"]

def test_mismatched_lengths():
    """Test that mismatched input lengths return empty results gracefully."""
    base_pl = [(1, 1), (0, 1)]
    abl_pl = [(1, 1)]

    results = compute_bootstrap_results_pointwise_all(base_pl, abl_pl)
    assert results == {}

def test_empty_inputs():
    """Test that empty inputs return empty results."""
    assert compute_bootstrap_results_pointwise_all([], []) == {}
    assert compute_bootstrap_results_pairwise_all([], []) == {}

def test_all_metrics_computed():
    """Verify that multiple metrics are returned for both setups."""
    n = 10
    base_pl = [(1, 1), (0, 0)] * 5
    abl_pl = [(1, 1), (1, 0)] * 5

    results = compute_bootstrap_results_pointwise_all(base_pl, abl_pl, n_resamples=100)
    assert "accuracy" in results
    assert "f1_macro" in results
    assert "precision_pos" in results

    base_pw = [(1.0, 0.0, 1.0)] * n
    abl_pw = [(0.0, 1.0, 1.0)] * n
    results_pw = compute_bootstrap_results_pairwise_all(base_pw, abl_pw, n_resamples=100)
    assert "pairwise_accuracy" in results_pw
    assert "pairwise_accuracy_strict" in results_pw
    assert "pairwise_accuracy_no_ties" in results_pw

def test_bca_failure_warning():
    """Test that degenerate data is handled gracefully without crashing."""
    base_pl = [(0, 1), (1, 1)]
    abl_pl = [(1, 1), (0, 1)]

    # Should not raise; degenerate BCa returns non-significant result
    results = compute_bootstrap_results_pointwise_all(base_pl, abl_pl, n_resamples=10)
    assert "accuracy" in results
