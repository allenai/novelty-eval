"""
Tests for n_runs > 1 aggregation behavior.

Covers two independent aggregation paths that must stay consistent:
  - artifacts.py path (_extract_raw_outcomes_for_dir): averages per-pid scores
    across runs before feeding into bootstrap.
  - filtering.py path (_recompute_pairwise/pointwise_metrics): computes metrics
    per run, then averages those per-run metric values.

Test categories:
  1. Per-pid outcome averaging (artifacts.py path)
  2. Per-run metric averaging (filtering.py path)
  3. Filtering with n_runs > 1
  4. Agreement / documented divergence between the two paths
  5. n_runs=1 backward compatibility
"""
import json
import statistics
import sys
import pytest
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import src.novelty_eval.analysis.artifacts as _artifacts_mod
import src.novelty_eval.analysis.filtering as _filtering_mod
from src.novelty_eval.analysis.artifacts import _extract_raw_outcomes_for_dir
from src.novelty_eval.analysis.filtering import (
    _recompute_pairwise_metrics,
    _recompute_pointwise_metrics,
)


# ---------------------------------------------------------------------------
# Cache-clearing fixture
# Run before and after every test so cached results never cross test boundaries.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_caches():
    _artifacts_mod._RAW_OUTCOMES_CACHE.clear()
    _filtering_mod._YAML_CACHE.clear()
    yield
    _artifacts_mod._RAW_OUTCOMES_CACHE.clear()
    _filtering_mod._YAML_CACHE.clear()


# ---------------------------------------------------------------------------
# Fixture-writing helpers
# ---------------------------------------------------------------------------

def _write_pairwise_scores(run_dir: Path, entries: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "scores.json", "w") as f:
        json.dump(entries, f)


def _write_pointwise_scores(run_dir: Path, entries: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "scores.json", "w") as f:
        json.dump(entries, f)


def _write_pairwise_instances(path: Path, pid_to_gt: dict[str, int]) -> None:
    """Minimal pairwise test_inputs.yaml (ground-truth winners per pid)."""
    data = {
        pid: {"expected_winners": [gt], "ideas": {"0": "idea A", "1": "idea B"}}
        for pid, gt in pid_to_gt.items()
    }
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_pointwise_instances(path: Path, pid_to_label: dict[str, str]) -> None:
    """Minimal pointwise instances YAML."""
    data = {
        pid: {"label": label, "idea": f"idea text for {pid}"}
        for pid, label in pid_to_label.items()
    }
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_accuracy_report(artifact_dir: Path, mode: str, instances_path: str) -> None:
    """Minimal accuracy_report.txt with the required instances path field."""
    content = (
        f"Test Mode: {mode.capitalize()}\n"
        f"Test Instances Path: {instances_path}\n"
        "Number of Test Instances Processed: 1\n"
    )
    (artifact_dir / "accuracy_report.txt").write_text(content)


def _pairwise_entry(winner: int, gt_winner: int, mec_details: dict | None = None) -> dict:
    """Build a single pairwise scores.json entry."""
    comp = {"winner": winner, "gt_winner": gt_winner}
    if mec_details is not None:
        comp["mec_details"] = mec_details
    return {"comparisons": [comp]}


# ---------------------------------------------------------------------------
# 1. Per-pid outcome averaging — artifacts.py path
# ---------------------------------------------------------------------------

class TestArtifactsPathAveraging:

    def test_pairwise_correct_then_wrong_averages_to_half(self, tmp_path):
        """Pid correct in run_0 and wrong in run_1: is_correct should be 0.5."""
        art = tmp_path / "artifact"
        art.mkdir()
        instances = art / "test_inputs.yaml"
        _write_pairwise_instances(instances, {"1": 0})
        _write_accuracy_report(art, "pairwise", str(instances))

        _write_pairwise_scores(art / "run_pairwise_0", {"1": _pairwise_entry(0, 0)})  # correct
        _write_pairwise_scores(art / "run_pairwise_1", {"1": _pairwise_entry(1, 0)})  # wrong

        result = _extract_raw_outcomes_for_dir(art, "pairwise", "fake_abl", None)

        assert result is not None
        assert result["_pairwise_is_correct"]["1"] == pytest.approx(0.5)
        assert result["_pairwise_is_tie"]["1"] == pytest.approx(0.0)

    def test_pairwise_all_correct_averages_to_one(self, tmp_path):
        """Pid correct in all 3 runs: is_correct should be 1.0."""
        art = tmp_path / "artifact"
        art.mkdir()
        instances = art / "test_inputs.yaml"
        _write_pairwise_instances(instances, {"1": 0})
        _write_accuracy_report(art, "pairwise", str(instances))

        for i in range(3):
            _write_pairwise_scores(art / f"run_pairwise_{i}", {"1": _pairwise_entry(0, 0)})

        result = _extract_raw_outcomes_for_dir(art, "pairwise", "fake_abl", None)

        assert result is not None
        assert result["_pairwise_is_correct"]["1"] == pytest.approx(1.0)

    def test_pointwise_correct_then_wrong_averages_prediction(self, tmp_path):
        """Pid predicted correctly (1) in run_0 and wrongly (0) in run_1:
        _pointwise_pred should average to 0.5 while target stays fixed."""
        inst = tmp_path / "instances.yaml"
        _write_pointwise_instances(inst, {"1": "POSITIVE"})  # target=1

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pointwise", str(inst))

        _write_pointwise_scores(art / "run_pointwise_0", {"1": {"prediction": 1, "label": "POSITIVE"}})
        _write_pointwise_scores(art / "run_pointwise_1", {"1": {"prediction": 0, "label": "POSITIVE"}})

        result = _extract_raw_outcomes_for_dir(art, "pointwise", "fake_abl", None)

        assert result is not None
        assert result["_pointwise_pred"]["1"] == pytest.approx(0.5)
        assert result["_pointwise_target"]["1"] == pytest.approx(1.0)

    def test_genuine_tie_then_wrong_gives_half_tie_score(self, tmp_path):
        """A genuine tie in run_0 and wrong in run_1: is_tie=0.5, is_correct=0.0.

        Genuine tie: winner=2 with no mec_details (not a failure tie).
        """
        art = tmp_path / "artifact"
        art.mkdir()
        instances = art / "test_inputs.yaml"
        _write_pairwise_instances(instances, {"1": 0})
        _write_accuracy_report(art, "pairwise", str(instances))

        _write_pairwise_scores(art / "run_pairwise_0", {"1": _pairwise_entry(2, 0)})   # genuine tie
        _write_pairwise_scores(art / "run_pairwise_1", {"1": _pairwise_entry(1, 0)})   # wrong

        result = _extract_raw_outcomes_for_dir(art, "pairwise", "fake_abl", None)

        assert result is not None
        assert result["_pairwise_is_tie"]["1"] == pytest.approx(0.5)
        assert result["_pairwise_is_correct"]["1"] == pytest.approx(0.0)

    def test_pid_absent_from_one_run_uses_union_not_intersection(self, tmp_path):
        """A pid absent from run_0 but present in run_1 is still included
        in the final outcomes (union). Its score reflects only run_1."""
        art = tmp_path / "artifact"
        art.mkdir()
        instances = art / "test_inputs.yaml"
        _write_pairwise_instances(instances, {"1": 0, "2": 0})
        _write_accuracy_report(art, "pairwise", str(instances))

        # run_0: only pid "1" (correct)
        _write_pairwise_scores(art / "run_pairwise_0", {"1": _pairwise_entry(0, 0)})
        # run_1: pid "1" (correct) and pid "2" (correct)
        _write_pairwise_scores(art / "run_pairwise_1", {
            "1": _pairwise_entry(0, 0),
            "2": _pairwise_entry(0, 0),
        })

        result = _extract_raw_outcomes_for_dir(art, "pairwise", "fake_abl", None)

        assert result is not None
        assert "1" in result["_pairwise_is_correct"]
        assert "2" in result["_pairwise_is_correct"]
        # pid "2" only in run_1 where it was correct — no penalty for absence from run_0
        assert result["_pairwise_is_correct"]["2"] == pytest.approx(1.0)

    def test_averaged_prediction_at_threshold_rounds_to_positive(self, tmp_path):
        """Documents a bootstrap-path subtlety: when a pid's averaged prediction
        is exactly 0.5 (one correct run, one wrong run), int(0.5 >= 0.5) == 1
        (POSITIVE), so the bootstrap treats it as a positive prediction.

        This means equal-split pids count as correct when the ground truth is
        POSITIVE — a slight optimistic bias vs. the filtering.py mean accuracy."""
        inst = tmp_path / "instances.yaml"
        _write_pointwise_instances(inst, {"1": "POSITIVE"})  # target=1

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pointwise", str(inst))

        _write_pointwise_scores(art / "run_pointwise_0", {"1": {"prediction": 1, "label": "POSITIVE"}})
        _write_pointwise_scores(art / "run_pointwise_1", {"1": {"prediction": 0, "label": "POSITIVE"}})

        result = _extract_raw_outcomes_for_dir(art, "pointwise", "fake_abl", None)
        assert result is not None

        avg_pred = result["_pointwise_pred"]["1"]
        target = result["_pointwise_target"]["1"]
        bootstrap_pred_class = int(avg_pred >= 0.5)
        target_class = int(target >= 0.5)

        # Bootstrap sees this as correct (tp), even though filtering.py gives 0.5 accuracy
        assert bootstrap_pred_class == target_class == 1


# ---------------------------------------------------------------------------
# 2. Per-run metric averaging — filtering.py path
# ---------------------------------------------------------------------------

class TestFilteringPathAveraging:

    def test_pairwise_recompute_averages_per_run_accuracies(self, tmp_path):
        """run_0 acc=1.0, run_1 acc=0.0 → mean pairwise_accuracy=0.5.
        per_run_accuracies_with_ties preserves the individual values."""
        inst = tmp_path / "instances.yaml"
        _write_pairwise_instances(inst, {"1": 0})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pairwise", str(inst))

        _write_pairwise_scores(art / "run_pairwise_0", {"1": _pairwise_entry(0, 0)})  # correct
        _write_pairwise_scores(art / "run_pairwise_1", {"1": _pairwise_entry(1, 0)})  # wrong

        result = _recompute_pairwise_metrics(art, set())

        assert result is not None
        assert result["pairwise_accuracy"] == pytest.approx(0.5)
        assert sorted(result["per_run_accuracies_with_ties"]) == [0.0, 1.0]

    def test_pointwise_recompute_averages_per_run_accuracies(self, tmp_path):
        """Pointwise: per-run accuracy is averaged across runs."""
        inst = tmp_path / "instances.yaml"
        _write_pointwise_instances(inst, {"1": "POSITIVE"})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pointwise", str(inst))

        _write_pointwise_scores(art / "run_pointwise_0", {"1": {"prediction": 1, "label": "POSITIVE"}})
        _write_pointwise_scores(art / "run_pointwise_1", {"1": {"prediction": 0, "label": "POSITIVE"}})

        result = _recompute_pointwise_metrics(art, set())

        assert result is not None
        assert result["accuracy"] == pytest.approx(0.5)
        assert sorted(result["per_run_accuracies"]) == [0.0, 1.0]

    def test_three_pairwise_runs_averaged_correctly(self, tmp_path):
        """n_runs=3: run_0 correct, run_1 wrong, run_2 correct → mean=2/3."""
        inst = tmp_path / "instances.yaml"
        _write_pairwise_instances(inst, {"1": 0})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pairwise", str(inst))

        _write_pairwise_scores(art / "run_pairwise_0", {"1": _pairwise_entry(0, 0)})  # correct
        _write_pairwise_scores(art / "run_pairwise_1", {"1": _pairwise_entry(1, 0)})  # wrong
        _write_pairwise_scores(art / "run_pairwise_2", {"1": _pairwise_entry(0, 0)})  # correct

        result = _recompute_pairwise_metrics(art, set())

        assert result is not None
        assert result["pairwise_accuracy"] == pytest.approx(2 / 3, rel=1e-5)
        assert len(result["per_run_accuracies_with_ties"]) == 3

    def test_pairwise_absent_pid_injected_as_wrong_per_run(self, tmp_path):
        """A pid missing from run_0 but present (correct) in run_1 is injected
        as wrong in run_0. This differs from the artifacts.py path, which omits
        the pid from the per-run average entirely."""
        inst = tmp_path / "instances.yaml"
        _write_pairwise_instances(inst, {"1": 0, "2": 0})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pairwise", str(inst))

        # run_0 has only pid "1" (correct); pid "2" will be injected as wrong
        _write_pairwise_scores(art / "run_pairwise_0", {"1": _pairwise_entry(0, 0)})
        # run_1 has both pids, both correct
        _write_pairwise_scores(art / "run_pairwise_1", {
            "1": _pairwise_entry(0, 0),
            "2": _pairwise_entry(0, 0),
        })

        result = _recompute_pairwise_metrics(art, set())

        assert result is not None
        # run_0: pid "1" correct (1.0), pid "2" injected wrong (0.0) → acc=0.5
        # run_1: both correct → acc=1.0
        # mean=0.75
        assert result["pairwise_accuracy"] == pytest.approx(0.75)
        assert sorted(result["per_run_accuracies_with_ties"]) == [0.5, 1.0]


# ---------------------------------------------------------------------------
# 3. Filtering (exclusion) with n_runs > 1
# ---------------------------------------------------------------------------

class TestFilteringWithMultipleRuns:

    def test_pairwise_exclude_set_applied_to_every_run(self, tmp_path):
        """An excluded pid is removed before computing metrics in each run."""
        inst = tmp_path / "instances.yaml"
        _write_pairwise_instances(inst, {"1": 0, "2": 0})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pairwise", str(inst))

        # Both runs: pid "1" correct, pid "2" wrong
        for i in range(2):
            _write_pairwise_scores(art / f"run_pairwise_{i}", {
                "1": _pairwise_entry(0, 0),  # correct
                "2": _pairwise_entry(1, 0),  # wrong
            })

        result_full = _recompute_pairwise_metrics(art, set())
        result_excl = _recompute_pairwise_metrics(art, {"1"})

        assert result_full is not None and result_excl is not None
        # Excluding pid "1" (correct) leaves only pid "2" (wrong) → accuracy=0.0
        assert result_excl["pairwise_accuracy"] == pytest.approx(0.0)
        # Without exclusion: one correct, one wrong → 0.5
        assert result_full["pairwise_accuracy"] == pytest.approx(0.5)
        assert "1" in result_excl["excluded"]

    def test_pointwise_exclude_set_applied_to_every_run(self, tmp_path):
        """Pointwise: excluded pids are removed from every run before computing metrics."""
        inst = tmp_path / "instances.yaml"
        _write_pointwise_instances(inst, {"1": "POSITIVE", "2": "POSITIVE"})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pointwise", str(inst))

        for i in range(2):
            _write_pointwise_scores(art / f"run_pointwise_{i}", {
                "1": {"prediction": 1, "label": "POSITIVE"},  # correct
                "2": {"prediction": 0, "label": "POSITIVE"},  # wrong
            })

        result_excl = _recompute_pointwise_metrics(art, {"1"})
        result_full = _recompute_pointwise_metrics(art, set())

        assert result_excl is not None
        assert result_excl["accuracy"] == pytest.approx(0.0)
        assert result_full["accuracy"] == pytest.approx(0.5)

    def test_exclude_set_does_not_affect_other_pids(self, tmp_path):
        """Excluding one pid should not change the reported accuracy of other pids."""
        inst = tmp_path / "instances.yaml"
        _write_pairwise_instances(inst, {"1": 0, "2": 0, "3": 0})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pairwise", str(inst))

        for i in range(2):
            _write_pairwise_scores(art / f"run_pairwise_{i}", {
                "1": _pairwise_entry(0, 0),  # correct
                "2": _pairwise_entry(0, 0),  # correct
                "3": _pairwise_entry(1, 0),  # wrong
            })

        result_excl = _recompute_pairwise_metrics(art, {"3"})

        assert result_excl is not None
        # After excluding pid "3", only pids "1" and "2" (both correct) remain
        assert result_excl["pairwise_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Agreement and documented divergence between the two paths
# ---------------------------------------------------------------------------

class TestPathsConsistency:

    def test_paths_agree_when_all_pids_present_in_all_runs(self, tmp_path):
        """When all pids appear in all runs, both paths should yield the same accuracy.

        filtering.py: mean of per-run accuracies.
        artifacts.py: mean of per-pid averaged is_correct scores.
        These are algebraically equivalent when no pids are missing.
        """
        art = tmp_path / "artifact"
        art.mkdir()
        instances = art / "test_inputs.yaml"
        _write_pairwise_instances(instances, {"1": 0, "2": 0, "3": 0})
        _write_accuracy_report(art, "pairwise", str(instances))

        # run_0: "1" correct, "2" wrong, "3" correct → acc=2/3
        _write_pairwise_scores(art / "run_pairwise_0", {
            "1": _pairwise_entry(0, 0),
            "2": _pairwise_entry(1, 0),
            "3": _pairwise_entry(0, 0),
        })
        # run_1: "1" wrong, "2" correct, "3" correct → acc=2/3
        _write_pairwise_scores(art / "run_pairwise_1", {
            "1": _pairwise_entry(1, 0),
            "2": _pairwise_entry(0, 0),
            "3": _pairwise_entry(0, 0),
        })

        filter_result = _recompute_pairwise_metrics(art, set())
        art_result = _extract_raw_outcomes_for_dir(art, "pairwise", "fake_abl", None)

        assert filter_result is not None and art_result is not None
        assert filter_result["pairwise_accuracy"] == pytest.approx(2 / 3, rel=1e-5)

        is_correct = art_result["_pairwise_is_correct"]
        manual_acc = statistics.mean(is_correct[pid] for pid in ["1", "2", "3"])
        assert manual_acc == pytest.approx(2 / 3, rel=1e-5)

    def test_paths_diverge_when_pid_absent_from_some_runs(self, tmp_path):
        """Documents the known divergence between the two paths when a pid
        is present in only some runs.

        filtering.py penalises the absence (injects as wrong in missing runs).
        artifacts.py only averages over runs where the pid was present.

        This means filtering.py gives a more conservative accuracy estimate.
        """
        art = tmp_path / "artifact"
        art.mkdir()
        instances = art / "test_inputs.yaml"
        _write_pairwise_instances(instances, {"1": 0, "2": 0})
        _write_accuracy_report(art, "pairwise", str(instances))

        # run_0: only pid "1" correct
        _write_pairwise_scores(art / "run_pairwise_0", {"1": _pairwise_entry(0, 0)})
        # run_1: pid "1" correct AND pid "2" correct
        _write_pairwise_scores(art / "run_pairwise_1", {
            "1": _pairwise_entry(0, 0),
            "2": _pairwise_entry(0, 0),
        })

        filter_result = _recompute_pairwise_metrics(art, set())
        # run_0: injects "2" as wrong → acc=0.5; run_1: both correct → acc=1.0 → mean=0.75
        assert filter_result is not None
        assert filter_result["pairwise_accuracy"] == pytest.approx(0.75)

        art_result = _extract_raw_outcomes_for_dir(art, "pairwise", "fake_abl", None)
        # pid "2" only from run_1 (correct) → is_correct["2"]=1.0; no penalty for absence from run_0
        assert art_result is not None
        is_correct = art_result["_pairwise_is_correct"]
        assert is_correct["1"] == pytest.approx(1.0)
        assert is_correct["2"] == pytest.approx(1.0)  # not penalised for being absent from run_0

        # The two paths disagree on the accuracy for these two pids:
        # filtering.py = 0.75 (run_0 penalises absence); artifacts.py = 1.0 on the same pids
        manual_acc_on_test_pids = statistics.mean(is_correct[pid] for pid in ["1", "2"])
        assert manual_acc_on_test_pids == pytest.approx(1.0)
        assert filter_result["pairwise_accuracy"] != pytest.approx(manual_acc_on_test_pids)


# ---------------------------------------------------------------------------
# 5. n_runs=1 backward compatibility
# ---------------------------------------------------------------------------

class TestSingleRunCompatibility:

    def test_single_pairwise_run_outcomes(self, tmp_path):
        """n_runs=1: a single run_pairwise_0 directory yields per-pid scores directly."""
        art = tmp_path / "artifact"
        art.mkdir()
        instances = art / "test_inputs.yaml"
        _write_pairwise_instances(instances, {"1": 0, "2": 1})
        _write_accuracy_report(art, "pairwise", str(instances))

        _write_pairwise_scores(art / "run_pairwise_0", {
            "1": _pairwise_entry(0, 0),  # correct
            "2": _pairwise_entry(0, 1),  # wrong (gt=1, but winner=0)
        })

        result = _extract_raw_outcomes_for_dir(art, "pairwise", "fake_abl", None)

        assert result is not None
        assert result["_pairwise_is_correct"]["1"] == pytest.approx(1.0)
        assert result["_pairwise_is_correct"]["2"] == pytest.approx(0.0)

    def test_single_pointwise_run_outcomes(self, tmp_path):
        """n_runs=1: a single run_pointwise_0 directory yields per-pid scores directly."""
        inst = tmp_path / "instances.yaml"
        _write_pointwise_instances(inst, {"1": "POSITIVE", "2": "NEGATIVE"})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pointwise", str(inst))

        _write_pointwise_scores(art / "run_pointwise_0", {
            "1": {"prediction": 1, "label": "POSITIVE"},   # correct
            "2": {"prediction": 1, "label": "NEGATIVE"},   # wrong (predicted POS, is NEG)
        })

        result = _extract_raw_outcomes_for_dir(art, "pointwise", "fake_abl", None)

        assert result is not None
        assert result["_pointwise_pred"]["1"] == pytest.approx(1.0)
        assert result["_pointwise_pred"]["2"] == pytest.approx(1.0)
        assert result["_pointwise_target"]["1"] == pytest.approx(1.0)  # POSITIVE → 1
        assert result["_pointwise_target"]["2"] == pytest.approx(0.0)  # NEGATIVE → 0

    def test_single_run_recompute_pairwise_gives_run_accuracy(self, tmp_path):
        """n_runs=1: recomputed accuracy reflects that single run, not an average."""
        inst = tmp_path / "instances.yaml"
        _write_pairwise_instances(inst, {"1": 0, "2": 0})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pairwise", str(inst))

        _write_pairwise_scores(art / "run_pairwise_0", {
            "1": _pairwise_entry(0, 0),  # correct
            "2": _pairwise_entry(1, 0),  # wrong
        })

        result = _recompute_pairwise_metrics(art, set())

        assert result is not None
        assert result["pairwise_accuracy"] == pytest.approx(0.5)
        assert len(result["per_run_accuracies_with_ties"]) == 1

    def test_single_run_recompute_pointwise_gives_run_accuracy(self, tmp_path):
        """n_runs=1: pointwise recomputed accuracy reflects that single run."""
        inst = tmp_path / "instances.yaml"
        _write_pointwise_instances(inst, {"1": "POSITIVE", "2": "POSITIVE"})

        art = tmp_path / "artifact"
        art.mkdir()
        _write_accuracy_report(art, "pointwise", str(inst))

        _write_pointwise_scores(art / "run_pointwise_0", {
            "1": {"prediction": 1, "label": "POSITIVE"},  # correct
            "2": {"prediction": 0, "label": "POSITIVE"},  # wrong
        })

        result = _recompute_pointwise_metrics(art, set())

        assert result is not None
        assert result["accuracy"] == pytest.approx(0.5)
        assert len(result["per_run_accuracies"]) == 1
