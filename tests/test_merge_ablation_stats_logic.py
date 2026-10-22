import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.novelty_eval.ablation.merge_ablation_runs import _run_bootstrap_for_setup
import src.novelty_eval.ablation.merge_ablation_runs as _mruns


# ---------------------------------------------------------------------------
# Shared fixture: wipe the bootstrap cache before and after every test so
# that cached results from one test never bleed into another.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_bootstrap_cache():
    _mruns._BOOTSTRAP_CACHE.clear()
    yield
    _mruns._BOOTSTRAP_CACHE.clear()

def test_run_bootstrap_pointwise_alignment():
    """Test that _run_bootstrap_for_setup correctly aligns PIDs between base and ablation for pointwise."""
    setup = "pointwise"
    all_models = ["gpt-4"]
    baseline_name = "base"
    lookup_abl = "abl"
    retrieval_index = {
        (setup, "base", "gpt-4"): "/path/to/base",
        (setup, "abl", "gpt-4"): "/path/to/abl",
    }
    
    # Mock extract_raw_outcomes_for_dir
    # Base has pids 1, 2
    # Abl has pids 2, 3
    # Intersection is pid 2
    base_outcomes = {
        "_pointwise_pred": {"1": 1.0, "2": 0.0},
        "_pointwise_target": {"1": 1.0, "2": 1.0},
    }
    abl_outcomes = {
        "_pointwise_pred": {"2": 1.0, "3": 0.0},
        "_pointwise_target": {"2": 1.0, "3": 1.0},
    }
    
    with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_raw_outcomes_for_dir") as mock_extract:
        mock_extract.side_effect = lambda path, s, n, l: base_outcomes if "base" in str(path) else abl_outcomes
        with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_instances_path", return_value="instances.yaml"):
            with patch("src.novelty_eval.ablation.merge_ablation_runs.compute_bootstrap_results_pointwise_all") as mock_boot:
                mock_boot.return_value = {"accuracy": {"significant": True, "p_value": 0.01}}
                
                res = _run_bootstrap_for_setup(
                    setup, all_models, baseline_name, lookup_abl, retrieval_index, 
                    "accuracy_report.txt", set(), MagicMock()
                )
                
                # Check that mock_boot was called with aligned data for pid 2 only
                # base_pred[2]=0, target[2]=1 -> (0, 1)
                # abl_pred[2]=1, target[2]=1 -> (1, 1)
                args, _ = mock_boot.call_args
                assert args[0] == [(0, 1)] # base_pl
                assert args[1] == [(1, 1)] # abl_pl
                
                assert res[("gpt-4", "accuracy")]["significant"] is True

def test_run_bootstrap_pairwise_alignment():
    """Test alignment for pairwise setup."""
    setup = "pairwise"
    all_models = ["gpt-4"]
    baseline_name = "base"
    lookup_abl = "abl"
    retrieval_index = {
        (setup, "base", "gpt-4"): "/path/to/base",
        (setup, "abl", "gpt-4"): "/path/to/abl",
    }
    
    base_outcomes = {
        "_pairwise_is_correct": {"1": 1.0, "2": 1.0},
        "_pairwise_is_tie": {"1": 0.0, "2": 0.0},
        "_pairwise_gt_winner": {"1": 0.0, "2": 1.0},
    }
    abl_outcomes = {
        "_pairwise_is_correct": {"1": 0.0, "2": 1.0},
        "_pairwise_is_tie": {"1": 1.0, "2": 0.0},
        "_pairwise_gt_winner": {"1": 0.0, "2": 1.0},
    }
    
    with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_raw_outcomes_for_dir") as mock_extract:
        mock_extract.side_effect = lambda path, s, n, l: base_outcomes if "base" in str(path) else abl_outcomes
        with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_instances_path", return_value="instances.yaml"):
            with patch("src.novelty_eval.ablation.merge_ablation_runs.compute_bootstrap_results_pairwise_all") as mock_boot:
                mock_boot.return_value = {"pairwise_accuracy": {"significant": False}}
                
                _run_bootstrap_for_setup(
                    setup, all_models, baseline_name, lookup_abl, retrieval_index, 
                    "accuracy_report.txt", set(), MagicMock()
                )
                
                args, _ = mock_boot.call_args
                # base_pw: [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
                # abl_pw:  [(0.0, 1.0, 0.0), (1.0, 0.0, 1.0)]
                assert args[0] == [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
                assert args[1] == [(0.0, 1.0, 0.0), (1.0, 0.0, 1.0)]

def test_run_bootstrap_with_exclusions():
    """Test that it respects exclusions when report_file is NOT accuracy_report.txt."""
    setup = "pointwise"
    all_models = ["gpt-4"]
    baseline_name = "base"
    lookup_abl = "abl"
    retrieval_index = {
        (setup, "base", "gpt-4"): "/path/to/base",
        (setup, "abl", "gpt-4"): "/path/to/abl",
    }
    
    base_outcomes = {
        "_pointwise_pred": {"1": 1.0, "2": 1.0},
        "_pointwise_target": {"1": 1.0, "2": 1.0},
    }
    abl_outcomes = {
        "_pointwise_pred": {"1": 0.0, "2": 0.0},
        "_pointwise_target": {"1": 1.0, "2": 1.0},
    }
    
    # Exclude PID "1"
    exclusions = {"1"}
    
    with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_raw_outcomes_for_dir") as mock_extract:
        mock_extract.side_effect = lambda path, s, n, l: base_outcomes if "base" in str(path) else abl_outcomes
        with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_instances_path", return_value="instances.yaml"):
            with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_exclude_set_from_filtered_report", return_value=exclusions):
                with patch("src.novelty_eval.ablation.merge_ablation_runs.compute_bootstrap_results_pointwise_all") as mock_boot:
                    mock_boot.return_value = {"accuracy": {"significant": False}}
                    
                    _run_bootstrap_for_setup(
                        setup, all_models, baseline_name, lookup_abl, retrieval_index, 
                        "filtered_accuracy_report.txt", set(), MagicMock()
                    )
                    
                    args, _ = mock_boot.call_args
                    # Should only have pid "2"
                    assert args[0] == [(1.0, 1.0)]
                    assert args[1] == [(0.0, 1.0)]

def test_run_bootstrap_instance_mismatch_warning():
    """Test that a warning is logged when instance contents differ."""
    setup = "pointwise"
    all_models = ["gpt-4"]
    baseline_name = "base"
    lookup_abl = "abl"
    retrieval_index = {
        (setup, "base", "gpt-4"): "/path/to/base",
        (setup, "abl", "gpt-4"): "/path/to/abl",
    }
    
    logger = MagicMock()
    
    with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_instances_path", side_effect=["i1.yaml", "i2.yaml"]):
        with patch("src.novelty_eval.ablation.merge_ablation_runs._instances_content_matches", return_value=(False, ["1"])):
             with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_raw_outcomes_for_dir", return_value=None):
                _run_bootstrap_for_setup(
                    setup, all_models, baseline_name, lookup_abl, retrieval_index, 
                    "accuracy_report.txt", set(), logger
                )
                
                # Check that logger.warning was called
                logger.warning.assert_called()
                assert "mismatched content" in logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# Bootstrap cache correctness tests
# ---------------------------------------------------------------------------

def _pointwise_retrieval_index(baseline="base", ablation="abl", model="gpt-4"):
    return {
        ("pointwise", baseline, model): f"/path/{baseline}",
        ("pointwise", ablation, model): f"/path/{ablation}",
    }


def _pointwise_outcomes():
    return {
        "_pointwise_pred":   {"1": 1.0, "2": 0.0},
        "_pointwise_target": {"1": 1.0, "2": 1.0},
    }


def test_bootstrap_cache_empty_at_test_start():
    """The autouse fixture guarantees a clean cache at the start of every test."""
    assert _mruns._BOOTSTRAP_CACHE == {}


def test_second_call_hits_cache_not_recomputed():
    """Identical (setup, baseline, ablation, model, report_file) → compute once, serve twice."""
    ri = _pointwise_retrieval_index()
    outcomes = _pointwise_outcomes()

    with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_raw_outcomes_for_dir", return_value=outcomes), \
         patch("src.novelty_eval.ablation.merge_ablation_runs._extract_instances_path", return_value="inst.yaml"), \
         patch("src.novelty_eval.ablation.merge_ablation_runs.compute_bootstrap_results_pointwise_all") as mock_boot:

        mock_boot.return_value = {"accuracy": {"significant": True}}

        r1 = _run_bootstrap_for_setup("pointwise", ["gpt-4"], "base", "abl", ri, "accuracy_report.txt", set(), MagicMock())
        r2 = _run_bootstrap_for_setup("pointwise", ["gpt-4"], "base", "abl", ri, "accuracy_report.txt", set(), MagicMock())

        assert mock_boot.call_count == 1, "bootstrap fn must be called exactly once; second call should be a cache hit"
        assert r1 == r2


def test_different_ablation_gets_separate_cache_entry():
    """Two different ablation names → two independent computations, no cross-contamination."""
    outcomes = _pointwise_outcomes()
    ri = {
        ("pointwise", "base",  "gpt-4"): "/path/base",
        ("pointwise", "abl_a", "gpt-4"): "/path/abl_a",
        ("pointwise", "abl_b", "gpt-4"): "/path/abl_b",
    }

    with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_raw_outcomes_for_dir", return_value=outcomes), \
         patch("src.novelty_eval.ablation.merge_ablation_runs._extract_instances_path", return_value="inst.yaml"), \
         patch("src.novelty_eval.ablation.merge_ablation_runs.compute_bootstrap_results_pointwise_all") as mock_boot:

        mock_boot.side_effect = [
            {"accuracy": {"significant": True}},
            {"accuracy": {"significant": False}},
        ]

        ra = _run_bootstrap_for_setup("pointwise", ["gpt-4"], "base", "abl_a", ri, "accuracy_report.txt", set(), MagicMock())
        rb = _run_bootstrap_for_setup("pointwise", ["gpt-4"], "base", "abl_b", ri, "accuracy_report.txt", set(), MagicMock())

        assert mock_boot.call_count == 2
        assert ra[("gpt-4", "accuracy")]["significant"] is True
        assert rb[("gpt-4", "accuracy")]["significant"] is False


def test_filtered_and_unfiltered_cache_independently():
    """report_file is part of the cache key: filtered and unfiltered don't share an entry."""
    ri = _pointwise_retrieval_index()
    outcomes = _pointwise_outcomes()

    with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_raw_outcomes_for_dir", return_value=outcomes), \
         patch("src.novelty_eval.ablation.merge_ablation_runs._extract_instances_path", return_value="inst.yaml"), \
         patch("src.novelty_eval.ablation.merge_ablation_runs.compute_bootstrap_results_pointwise_all") as mock_boot:

        mock_boot.return_value = {"accuracy": {"significant": True}}

        _run_bootstrap_for_setup("pointwise", ["gpt-4"], "base", "abl", ri, "accuracy_report.txt",          set(), MagicMock())
        _run_bootstrap_for_setup("pointwise", ["gpt-4"], "base", "abl", ri, "filtered_accuracy_report.txt", set(), MagicMock())

        assert mock_boot.call_count == 2, "unfiltered and filtered must be cached under separate keys"


def test_cache_result_content_is_identical_to_fresh():
    """The value returned on a cache hit is byte-for-byte equal to the first computation."""
    ri = _pointwise_retrieval_index()
    outcomes = _pointwise_outcomes()
    expected = {"accuracy": {"significant": True, "ci_low": -0.05, "ci_high": 0.12}}

    with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_raw_outcomes_for_dir", return_value=outcomes), \
         patch("src.novelty_eval.ablation.merge_ablation_runs._extract_instances_path", return_value="inst.yaml"), \
         patch("src.novelty_eval.ablation.merge_ablation_runs.compute_bootstrap_results_pointwise_all", return_value=expected):

        fresh  = _run_bootstrap_for_setup("pointwise", ["gpt-4"], "base", "abl", ri, "accuracy_report.txt", set(), MagicMock())
        cached = _run_bootstrap_for_setup("pointwise", ["gpt-4"], "base", "abl", ri, "accuracy_report.txt", set(), MagicMock())

        assert fresh == cached
        assert cached[("gpt-4", "accuracy")] == expected["accuracy"]


def test_multi_model_each_model_cached_independently():
    """With two models, each model's result is cached under its own key."""
    ri = {
        ("pointwise", "base",  "gpt-4"):   "/path/base",
        ("pointwise", "abl",   "gpt-4"):   "/path/abl",
        ("pointwise", "base",  "claude-3"): "/path/base",
        ("pointwise", "abl",   "claude-3"): "/path/abl",
    }
    outcomes = _pointwise_outcomes()

    with patch("src.novelty_eval.ablation.merge_ablation_runs._extract_raw_outcomes_for_dir", return_value=outcomes), \
         patch("src.novelty_eval.ablation.merge_ablation_runs._extract_instances_path", return_value="inst.yaml"), \
         patch("src.novelty_eval.ablation.merge_ablation_runs.compute_bootstrap_results_pointwise_all") as mock_boot:

        mock_boot.return_value = {"accuracy": {"significant": True}}

        # First run: 2 models → 2 cache entries, bootstrap called twice
        _run_bootstrap_for_setup("pointwise", ["gpt-4", "claude-3"], "base", "abl", ri, "accuracy_report.txt", set(), MagicMock())
        assert mock_boot.call_count == 2

        # Second run: both models already cached → bootstrap not called again
        _run_bootstrap_for_setup("pointwise", ["gpt-4", "claude-3"], "base", "abl", ri, "accuracy_report.txt", set(), MagicMock())
        assert mock_boot.call_count == 2, "all model results already in cache; no recomputation"
