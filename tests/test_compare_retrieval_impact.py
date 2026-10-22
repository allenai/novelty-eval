"""
Tests for compare_retrieval_impact.py

Covers:
  - Mode detection (ranking / pairwise / pointwise)
  - Per-mode parsing (scores, gold answers, metrics)
  - End-to-end report generation for all three modes
  - Report content / formatting correctness
  - Retrospective sweep-dir pair discovery
  - Merge integration (merge_ablation_runs.py generates reports automatically)
  - Idempotency (existing reports are not overwritten on re-merge)

File structure notes:
  - accuracy_report.txt     : aggregate stats (Mean Accuracy, Mean F1, etc.)
                              Written by report_writer.save_report() as output_file.
                              This is what detect_mode() and the metric parsers read.
  - debug_accuracy_report.txt : per-problem scores / wrong pairs
                              Written as debug_file. For pointwise this is ~empty.
  - debug_accuracy_report.txt.md : full markdown detail (titles, ratings, retrieval)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.novelty_eval.ablation.compare_retrieval_impact import (
    detect_mode,
    find_report_files,
    find_retrieval_pairs,
    parse_gold_from_md,
    parse_iclr_ratings_from_md,
    parse_idea_texts_from_md,
    parse_idea_titles_from_md,
    parse_pairwise_accuracy,
    parse_pointwise_metrics,
    parse_pointwise_problems_from_md,
    parse_ranking_scores,
    parse_retrieved_related_work_from_md,
    parse_retrieval_data_from_md,
    rank_of,
    run_comparison,
    scores_to_ranking,
)


# ── Fixtures: plain-text report bodies ──────────────────────────────────────────

RANKING_TXT_BASELINE = textwrap.dedent("""\
    Swiss Tournament Results
    ------------------------
    Mean NDCG: 0.7500

    Scores per Run:
      Run 1:
        Problem 1:
          Idea 0: {'novelty': 8.0, 'overall': 8.0}
          Idea 1: {'novelty': 5.0, 'overall': 5.0}
          Idea 2: {'novelty': 6.0, 'overall': 6.0}
        Problem 2:
          Idea 0: {'novelty': 4.0, 'overall': 4.0}
          Idea 1: {'novelty': 6.0, 'overall': 6.0}
          Idea 2: {'novelty': 7.0, 'overall': 7.0}
        Problem 3:
          Idea 0: {'novelty': 9.0, 'overall': 9.0}
          Idea 1: {'novelty': 7.0, 'overall': 7.0}
""")

# Retrieval nudges Problem 2 gold (idea 1) from rank 2 → rank 1 (improved),
# but pushes Problem 3 gold (idea 0) from rank 1 → rank 2 (degraded).
# Problem 1 is unchanged correct (gold=0 stays at rank 1).
RANKING_TXT_RETRIEVAL = textwrap.dedent("""\
    Swiss Tournament Results
    ------------------------
    Mean NDCG: 0.8000

    Scores per Run:
      Run 1:
        Problem 1:
          Idea 0: {'novelty': 8.0, 'overall': 8.0}
          Idea 1: {'novelty': 5.0, 'overall': 5.0}
          Idea 2: {'novelty': 6.0, 'overall': 6.0}
        Problem 2:
          Idea 0: {'novelty': 4.0, 'overall': 4.0}
          Idea 1: {'novelty': 8.0, 'overall': 8.0}
          Idea 2: {'novelty': 7.0, 'overall': 7.0}
        Problem 3:
          Idea 0: {'novelty': 6.0, 'overall': 6.0}
          Idea 1: {'novelty': 8.0, 'overall': 8.0}
""")

RANKING_MD_TEMPLATE = textwrap.dedent("""\
    # Debug Accuracy Report

    ### Problem 1
    **Gold Answer:** 0
    **Predicted Rank:** 0, 2, 1

    **Gold Winner Details:**
    - **Idea 0**
      - **Title:** Novel Framework Alpha
      - **ICLR Rating:** 8.0

    **Bad Rankings:**
    - **Idea 1** ranked >= **Gold Idea 0**
      - **Title:** Incremental Method Beta
      - **ICLR Rating:** 5.0
    - **Idea 2** ranked >= **Gold Idea 0**
      - **Title:** Standard Approach Gamma
      - **ICLR Rating:** 6.0

    <details>
    <summary>Comparison Reasoning (click to expand)</summary>
    Idea 0 is more novel because it introduces a genuinely new framework.
    </details>

    ### Problem 2
    **Gold Answer:** 1
    **Predicted Rank:** 2, 0, 1

    **Gold Winner Details:**
    - **Idea 1**
      - **Title:** Transformer-Based Method
      - **ICLR Rating:** 7.0

    **Bad Rankings:**
    - **Idea 2** ranked >= **Gold Idea 1**
      - **Title:** Classical ML Approach
      - **ICLR Rating:** 6.5

    ### Problem 3
    **Gold Answer:** 0
    **Predicted Rank:** 0, 1

    **Gold Winner Details:**
    - **Idea 0**
      - **Title:** Contrastive Learning Method
      - **ICLR Rating:** 9.0

    **Good Rankings:**
    - **Idea 1** correctly ranked below **Gold Idea 0**
      - **Title:** Supervised Baseline
      - **ICLR Rating:** 7.0
""")

# Retrieval MD: same structure as template, but Problem 2 has retrieval data
# embedded inline within the problem section (no duplicate ### Problem headers).
RANKING_MD_RETRIEVAL = textwrap.dedent("""\
    # Debug Accuracy Report

    ### Problem 1
    **Gold Answer:** 0
    **Predicted Rank:** 0, 2, 1

    **Gold Winner Details:**
    - **Idea 0**
      - **Title:** Novel Framework Alpha
      - **ICLR Rating:** 8.0

    **Bad Rankings:**
    - **Idea 1** ranked >= **Gold Idea 0**
      - **Title:** Incremental Method Beta
      - **ICLR Rating:** 5.0
    - **Idea 2** ranked >= **Gold Idea 0**
      - **Title:** Standard Approach Gamma
      - **ICLR Rating:** 6.0

    <details>
    <summary>Comparison Reasoning (click to expand)</summary>
    Idea 0 is more novel because it introduces a genuinely new framework.
    </details>

    ### Problem 2
    **Gold Answer:** 1
    **Predicted Rank:** 1, 2, 0

    **Gold Winner Details:**
    - **Idea 1**
      - **Title:** Transformer-Based Method
      - **ICLR Rating:** 7.0

    **Good Rankings:**
    - **Idea 0** correctly ranked below **Gold Idea 1**
      - **Title:** Classical ML Approach
      - **ICLR Rating:** 4.0
    - **Idea 2** correctly ranked below **Gold Idea 1**
      - **Title:** Standard Baseline
      - **ICLR Rating:** 6.5

    **Idea 0**
    <details>
    <summary>Retrieval Queries (click to expand)</summary>

    - classical machine learning approaches
    - SVM feature extraction
    </details>
    <details>
    <summary>Candidates Confidence Scores (click to expand)</summary>

    Average confidence: `0.7200`
    1. **Support Vector Machines Survey** — confidence: `0.7500`
    2. **Feature Engineering Review** — confidence: `0.6900`
    </details>

    **Idea 1**
    <details>
    <summary>Retrieval Queries (click to expand)</summary>

    - transformer architecture novelty
    - attention mechanism survey
    </details>
    <details>
    <summary>Candidates Confidence Scores (click to expand)</summary>

    Average confidence: `0.8900`
    1. **Attention Is All You Need** — confidence: `0.9200`
    2. **BERT: Pre-training Deep Bidirectional Transformers** — confidence: `0.8600`
    </details>

    ### Problem 3
    **Gold Answer:** 0
    **Predicted Rank:** 0, 1

    **Gold Winner Details:**
    - **Idea 0**
      - **Title:** Contrastive Learning Method
      - **ICLR Rating:** 9.0

    **Good Rankings:**
    - **Idea 1** correctly ranked below **Gold Idea 0**
      - **Title:** Supervised Baseline
      - **ICLR Rating:** 7.0
""")


# accuracy_report.txt content for pairwise (aggregate stats)
PAIRWISE_ACC_BASELINE = textwrap.dedent("""\
    Comparative Evaluator Accuracy Report
    =====================================

    Test Mode: pairwise

    Pairwise
    --------
    Mean LLM Pairwise Accuracy (with ties):    0.6000  (support=10)
    Mean LLM Pairwise Accuracy (without ties): 0.6667  (support=9)
    Mean Number of Ties: 1.0
    Individual Run Accuracies (with ties): [0.6]
    Individual Run Ties: [1]
""")

PAIRWISE_ACC_RETRIEVAL = textwrap.dedent("""\
    Comparative Evaluator Accuracy Report
    =====================================

    Test Mode: pairwise

    Pairwise
    --------
    Mean LLM Pairwise Accuracy (with ties):    0.7000  (support=10)
    Mean LLM Pairwise Accuracy (without ties): 0.7778  (support=9)
    Mean Number of Ties: 1.0
    Individual Run Accuracies (with ties): [0.7]
    Individual Run Ties: [1]
""")

# debug_accuracy_report.txt content for pairwise (per-problem wrong/good pairs)
PAIRWISE_TXT_BASELINE = textwrap.dedent("""\
    Comparative Evaluator Debug Artifact
    ====================================

    Pairwise Details
    ----------------
    Problematic Rankings:
      Run 1:
        - Problem 1: Expected Winners: [0]; Wrong/tied comparisons: 1/1
        - Problem 2: Expected Winners: [1]; Wrong/tied comparisons: 1/1

    Scores per Run:
      Run 1:
        Problem 1:
          Idea (0, 1): {'novelty': 5.0, 'overall': 5.0}
        Problem 2:
          Idea (1, 0): {'novelty': 4.0, 'overall': 4.0}
""")

PAIRWISE_TXT_RETRIEVAL = textwrap.dedent("""\
    Comparative Evaluator Debug Artifact
    ====================================

    Pairwise Details
    ----------------
    Problematic Rankings:
      Run 1:
        - Problem 2: Expected Winners: [1]; Wrong/tied comparisons: 1/1

    Scores per Run:
      Run 1:
        Problem 2:
          Idea (1, 0): {'novelty': 4.0, 'overall': 4.0}
""")

PAIRWISE_MD_BASELINE = textwrap.dedent("""\
    # Pairwise Debug Report

    ## Run 1 — Bad Comparisons

    ### Problem 1
    **Expected Winners:** 0

    - **Pair:** Idea 0 vs Idea 1 — **Idea-1 chosen (wrong)** (expected gt_winner=0)
      - **Idea 0**
        - **Title:** Novel Framework Alpha
        - **Text:**
          > This is the text for idea 0 in problem 1.

    ### Problem 2
    **Expected Winners:** 1

    - **Pair:** Idea 0 vs Idea 1 — **Idea-0 chosen (wrong)** (expected gt_winner=1)
      - **Idea 0**
        - **Title:** Classical ML Approach
        - **Text:**
          > This is the text for idea 0 in problem 2.
      - **Idea 1**
        - **Title:** Transformer-Based Method
        - **Text:**
          > This is the text for idea 1 in problem 2.

    ## Run 1 — Good Comparisons

    ### Problem 3 ✓
    **Expected Winners:** 0

    - **Pair:** Idea 0 vs Idea 1 — correctly chose Idea-0
      - **Idea 0**
        - **Title:** Contrastive Learning Method
        - **Text:**
          > This is the text for idea 0 in problem 3.
""")

PAIRWISE_MD_RETRIEVAL = textwrap.dedent("""\
    # Pairwise Debug Report

    ## Run 1 — Bad Comparisons

    ### Problem 2
    **Expected Winners:** 1

    - **Pair:** Idea 0 vs Idea 1 — **Idea-0 chosen (wrong)** (expected gt_winner=1)
      - **Idea 0**
        - **Title:** Classical ML Approach
        - **Text:**
          > This is the text for idea 0 in problem 2.
      - **Idea 1**
        - **Title:** Transformer-Based Method
        - **Text:**
          > This is the text for idea 1 in problem 2.

    ## Run 1 — Good Comparisons

    ### Problem 1 ✓
    **Expected Winners:** 0

    - **Pair:** Idea 0 vs Idea 1 — correctly chose Idea-0
      - **Idea 0**
        - **Title:** Novel Framework Alpha
        - **Text:**
          > This is the text for idea 0 in problem 1.

    ### Problem 3 ✓
    **Expected Winners:** 0

    - **Pair:** Idea 0 vs Idea 1 — correctly chose Idea-0
      - **Idea 0**
        - **Title:** Contrastive Learning Method
        - **Text:**
          > This is the text for idea 0 in problem 3.
""")


# accuracy_report.txt content for pointwise (aggregate stats)
POINTWISE_ACC_BASELINE = textwrap.dedent("""\
    Comparative Evaluator Accuracy Report
    =====================================

    Test Mode: pointwise

    Pointwise Classification Results
    ---------------------------------
    Mean Accuracy:              0.6500
    Mean F1 (macro):            0.6300
    Mean Precision (POSITIVE):  0.7000
    Mean Precision (NEGATIVE):  0.5800
    Mean Recall    (POSITIVE):  0.6800
    Mean Recall    (NEGATIVE):  0.5700
    Mean F1        (POSITIVE):  0.6900
    Mean F1        (NEGATIVE):  0.5700
    Individual Run Accuracies:  [0.65]
""")

POINTWISE_ACC_RETRIEVAL = textwrap.dedent("""\
    Comparative Evaluator Accuracy Report
    =====================================

    Test Mode: pointwise

    Pointwise Classification Results
    ---------------------------------
    Mean Accuracy:              0.7500
    Mean F1 (macro):            0.7200
    Mean Precision (POSITIVE):  0.7800
    Mean Precision (NEGATIVE):  0.7100
    Mean Recall    (POSITIVE):  0.7600
    Mean Recall    (NEGATIVE):  0.6900
    Mean F1        (POSITIVE):  0.7700
    Mean F1        (NEGATIVE):  0.7000
    Individual Run Accuracies:  [0.75]
""")

# debug_accuracy_report.txt content for pointwise — _write_debug_mode is never called
# for pointwise, so the debug file is nearly empty.
POINTWISE_TXT_BASELINE = textwrap.dedent("""\
    Comparative Evaluator Debug Artifact
    ====================================

""")

POINTWISE_TXT_RETRIEVAL = textwrap.dedent("""\
    Comparative Evaluator Debug Artifact
    ====================================

""")

POINTWISE_MD_BASELINE = textwrap.dedent("""\
    # Comparative Evaluator Debug Report

    # Pointwise Classification

    ## Run 1 — Wrong Classifications (1)

    - **Instance 1**
      - **Title:** Novel Hypothesis Generator
      - **Ground Truth:** POSITIVE | **Predicted:** NEGATIVE (not novel)
      - **ICLR Rating:** 7.50 | **Contribution Score:** 3.00
      - **Idea Text:**
        > This paper presents a generative framework for scientific hypothesis generation.
        > It uses large language models combined with a structured knowledge graph to produce
        > testable and novel research hypotheses automatically.

    ## Run 1 — Correct Classifications (2)

    - **Instance 2**
      - **Title:** Incremental Baseline Method
      - **Ground Truth:** NEGATIVE | **Predicted:** NEGATIVE (not novel)
      - **ICLR Rating:** N/A | **Contribution Score:** N/A
      - **Idea Text:**
        > A simple extension of standard fine-tuning procedures. The method adds a minor
        > regularization term to the training objective.

    - **Instance 3**
      - **Title:** [Expert]-I-3
      - **Ground Truth:** POSITIVE | **Predicted:** POSITIVE (novel)
      - **ICLR Rating:** 8.00 | **Contribution Score:** 4.00
      - **Idea Text:**
        > We introduce a new framework for continual learning that prevents catastrophic
        > forgetting via dynamic architectural expansion.
""")

POINTWISE_MD_RETRIEVAL = textwrap.dedent("""\
    # Comparative Evaluator Debug Report

    # Pointwise Classification

    ## Run 1 — Wrong Classifications (1)

    - **Instance 3**
      - **Title:** [Expert]-I-3
      - **Ground Truth:** POSITIVE | **Predicted:** NEGATIVE (not novel)
      - **ICLR Rating:** 8.00 | **Contribution Score:** 4.00
      - **Idea Text:**
        > We introduce a new framework for continual learning that prevents catastrophic
        > forgetting via dynamic architectural expansion.

    ## Run 1 — Correct Classifications (2)

    - **Instance 1**
      - **Title:** Novel Hypothesis Generator
      - **Ground Truth:** POSITIVE | **Predicted:** POSITIVE (novel)
      - **ICLR Rating:** 7.50 | **Contribution Score:** 3.00
      - **Idea Text:**
        > This paper presents a generative framework for scientific hypothesis generation.

    - **Instance 2**
      - **Title:** Incremental Baseline Method
      - **Ground Truth:** NEGATIVE | **Predicted:** NEGATIVE (not novel)
      - **ICLR Rating:** N/A | **Contribution Score:** N/A
      - **Idea Text:**
        > A simple extension of standard fine-tuning procedures.
""")

# Legacy stub for places that don't need per-instance data
POINTWISE_MD_STUB = POINTWISE_MD_BASELINE


# ── Helper to build an artifact dir ─────────────────────────────────────────────

def _write_artifact_dir(
    tmp_path: Path,
    name: str,
    txt: str,
    md: str,
    accuracy_txt: str | None = None,
) -> Path:
    """
    Create a minimal accuracy_test_artifacts/<timestamp> directory structure.

    Args:
        txt:          Content for debug_accuracy_report.txt (per-problem debug info).
        md:           Content for debug_accuracy_report.txt.md (full markdown detail).
        accuracy_txt: Content for accuracy_report.txt (aggregate stats). When provided
                      this is what detect_mode() and the metric parsers actually read.

    Returns the model-level parent dir (what run_ablations.py calls artifact_dir).
    """
    art = tmp_path / name / "accuracy_test_artifacts" / "20260413_120000"
    art.mkdir(parents=True)
    (art / "debug_accuracy_report.txt").write_text(txt, encoding="utf-8")
    (art / "debug_accuracy_report.txt.md").write_text(md, encoding="utf-8")
    if accuracy_txt is not None:
        (art / "accuracy_report.txt").write_text(accuracy_txt, encoding="utf-8")
    return tmp_path / name  # model-level dir


# ── Mode detection ───────────────────────────────────────────────────────────────

class TestDetectMode:
    def test_ranking(self, tmp_path):
        # Ranking has no separate accuracy_report.txt in old runs;
        # detect_mode falls back to scanning the debug file (no pairwise/pointwise markers → ranking).
        f = tmp_path / "debug_accuracy_report.txt"
        f.write_text(RANKING_TXT_BASELINE)
        assert detect_mode(f) == "ranking"

    def test_pairwise(self, tmp_path):
        # Pairwise: accuracy_report.txt contains "Test Mode: pairwise".
        f = tmp_path / "debug_accuracy_report.txt"
        f.write_text(PAIRWISE_TXT_BASELINE)
        (tmp_path / "accuracy_report.txt").write_text(PAIRWISE_ACC_BASELINE)
        assert detect_mode(f) == "pairwise"

    def test_pointwise(self, tmp_path):
        # Pointwise: debug file is nearly empty; mode is in accuracy_report.txt.
        f = tmp_path / "debug_accuracy_report.txt"
        f.write_text(POINTWISE_TXT_BASELINE)
        (tmp_path / "accuracy_report.txt").write_text(POINTWISE_ACC_BASELINE)
        assert detect_mode(f) == "pointwise"

    def test_pairwise_fallback_when_no_accuracy_report(self, tmp_path):
        """Older runs without accuracy_report.txt should still be detected via debug file."""
        f = tmp_path / "debug_accuracy_report.txt"
        # Write content with pairwise-specific string directly in the debug file
        f.write_text("Mean LLM Pairwise Accuracy (with ties): 0.7\n")
        assert detect_mode(f) == "pairwise"


# ── Ranking: unit tests ──────────────────────────────────────────────────────────

class TestRankingParsers:
    def test_parse_scores_extracts_all_problems(self):
        scores = parse_ranking_scores(RANKING_TXT_BASELINE)
        assert set(scores.keys()) == {1, 2, 3}

    def test_parse_scores_values(self):
        scores = parse_ranking_scores(RANKING_TXT_BASELINE)
        assert scores[1][0] == pytest.approx(8.0)
        assert scores[1][1] == pytest.approx(5.0)
        # Problem 2: idea 1 scores 6.0, idea 2 scores 7.0 (idea 2 beats gold in baseline)
        assert scores[2][1] == pytest.approx(6.0)
        assert scores[2][2] == pytest.approx(7.0)

    def test_parse_gold_from_md(self, tmp_path):
        md = tmp_path / "report.md"
        md.write_text(RANKING_MD_TEMPLATE)
        gold = parse_gold_from_md(md)
        assert gold == {1: 0, 2: 1, 3: 0}

    def test_parse_idea_titles(self, tmp_path):
        md = tmp_path / "report.md"
        md.write_text(RANKING_MD_TEMPLATE)
        titles = parse_idea_titles_from_md(md)
        assert titles[1][0] == "Novel Framework Alpha"
        assert titles[2][1] == "Transformer-Based Method"

    def test_parse_iclr_ratings(self, tmp_path):
        md = tmp_path / "report.md"
        md.write_text(RANKING_MD_TEMPLATE)
        ratings = parse_iclr_ratings_from_md(md)
        assert ratings[1][0] == pytest.approx(8.0)
        assert ratings[3][0] == pytest.approx(9.0)

    def test_scores_to_ranking_order(self):
        scores = {0: 8.0, 1: 5.0, 2: 6.0}
        ranking = scores_to_ranking(scores)
        assert ranking == [0, 2, 1]

    def test_scores_to_ranking_tie_broken_by_id(self):
        scores = {0: 7.0, 1: 7.0, 2: 5.0}
        ranking = scores_to_ranking(scores)
        assert ranking[0] == 0  # lower ID wins on tie
        assert ranking[1] == 1

    def test_rank_of(self):
        assert rank_of([0, 2, 1], 0) == 1
        assert rank_of([0, 2, 1], 2) == 2
        assert rank_of([0, 2, 1], 1) == 3

    def test_parse_retrieval_data(self, tmp_path):
        md = tmp_path / "retrieval.md"
        md.write_text(RANKING_MD_RETRIEVAL)
        data = parse_retrieval_data_from_md(md)
        # Problem 2 should have retrieval data for ideas 0 and 1
        assert 2 in data
        assert 0 in data[2]
        assert 1 in data[2]
        assert data[2][1]["avg_confidence"] == pytest.approx(0.89)
        assert len(data[2][1]["candidates"]) == 2
        assert data[2][1]["candidates"][0]["title"] == "Attention Is All You Need"
        assert "transformer architecture novelty" in data[2][1]["queries"]


# ── Pairwise: unit tests ─────────────────────────────────────────────────────────

class TestPairwiseParsers:
    def test_parse_accuracy(self, tmp_path):
        # Mirrors real file structure: debug file + accuracy_report.txt sibling.
        f = tmp_path / "debug_accuracy_report.txt"
        f.write_text(PAIRWISE_TXT_BASELINE)
        (tmp_path / "accuracy_report.txt").write_text(PAIRWISE_ACC_BASELINE)
        acc = parse_pairwise_accuracy(f)
        assert acc["acc_with_ties"] == pytest.approx(0.6)
        assert acc["acc_no_ties"] == pytest.approx(0.6667, abs=1e-4)
        assert acc["support"] == 10
        assert acc["n_ties"] == pytest.approx(1.0)

    def test_parse_accuracy_retrieval(self, tmp_path):
        f = tmp_path / "debug_accuracy_report.txt"
        f.write_text(PAIRWISE_TXT_RETRIEVAL)
        (tmp_path / "accuracy_report.txt").write_text(PAIRWISE_ACC_RETRIEVAL)
        acc = parse_pairwise_accuracy(f)
        assert acc["acc_with_ties"] == pytest.approx(0.7)


# ── Pointwise: unit tests ────────────────────────────────────────────────────────

class TestPointwiseParsers:
    def test_parse_metrics_baseline(self, tmp_path):
        # Mirrors real file structure: debug file is nearly empty; metrics live in accuracy_report.txt.
        f = tmp_path / "debug_accuracy_report.txt"
        f.write_text(POINTWISE_TXT_BASELINE)
        (tmp_path / "accuracy_report.txt").write_text(POINTWISE_ACC_BASELINE)
        m = parse_pointwise_metrics(f)
        assert m["accuracy"] == pytest.approx(0.65)
        assert m["f1_macro"] == pytest.approx(0.63)
        assert m["precision_pos"] == pytest.approx(0.70)
        assert m["recall_neg"] == pytest.approx(0.57)

    def test_parse_metrics_all_keys_present(self, tmp_path):
        f = tmp_path / "debug_accuracy_report.txt"
        f.write_text(POINTWISE_TXT_RETRIEVAL)
        (tmp_path / "accuracy_report.txt").write_text(POINTWISE_ACC_RETRIEVAL)
        m = parse_pointwise_metrics(f)
        expected_keys = {"accuracy", "f1_macro", "precision_pos", "precision_neg",
                         "recall_pos", "recall_neg", "f1_pos", "f1_neg"}
        assert set(m.keys()) == expected_keys


# ── End-to-end report generation ────────────────────────────────────────────────

class TestRankingEndToEnd:
    """Full ranking comparison: 3 problems, 1 improved, 1 degraded, 1 unchanged-correct."""

    @pytest.fixture
    def dirs(self, tmp_path):
        base_dir = _write_artifact_dir(tmp_path, "ranking_current-gpt-5.4",
                                       RANKING_TXT_BASELINE, RANKING_MD_TEMPLATE)
        ret_dir = _write_artifact_dir(tmp_path, "ranking_retrieval-gpt-5.4",
                                      RANKING_TXT_RETRIEVAL, RANKING_MD_RETRIEVAL)
        return base_dir, ret_dir

    def test_report_is_written(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_report_mode_header(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "**Mode:** `ranking`" in content

    def test_summary_section_present(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "## Summary" in content
        assert "Total problems" in content
        assert "Hits@1" in content

    def test_improved_degraded_unchanged_counts(self, dirs, tmp_path):
        """
        Problem 1: gold=0 at rank 1 both ways → unchanged correct
        Problem 2: gold=1 at rank 3 baseline, rank 1 retrieval → improved
        Problem 3: gold=0 at rank 1 baseline, rank 2 retrieval → degraded
        """
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()

        assert "| Retrieval **improved** gold rank | 1 |" in content
        assert "| Retrieval **degraded** gold rank | 1 |" in content
        assert "| Unchanged — both correct (rank 1) | 1 |" in content

    def test_hits_at_1_stats(self, dirs, tmp_path):
        """
        Baseline: Problem 1 (gold 0, rank 1) + Problem 3 (gold 0, rank 1) = 2/3
        Retrieval: Problem 1 (rank 1) + Problem 2 (improved to rank 1) = 2/3
        Both are 2/3 because one improved case cancels one degraded case.
        """
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "Baseline Hits@1:** 2/3" in content
        assert "Retrieval Hits@1:** 2/3" in content

    def test_improved_section_contains_problem(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        # Problem 2 should appear in the "Improved" section
        improved_section = content.split("## 🟢 Cases Where Retrieval IMPROVED")[1].split("## 🔴")[0]
        assert "Problem 2" in improved_section

    def test_degraded_section_contains_problem(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        degraded_section = content.split("## 🔴 Cases Where Retrieval DEGRADED")[1].split("## ⚪")[0]
        assert "Problem 3" in degraded_section

    def test_retrieval_data_rendered(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        # Retrieval queries and confidence scores for Problem 2 should appear
        assert "Attention Is All You Need" in content
        assert "0.8900" in content

    def test_idea_titles_rendered(self, dirs, tmp_path):
        """
        Only improved/degraded/unchanged-wrong problems get full detail sections;
        unchanged-correct problems (Problem 1) only appear in a compact table row.
        Titles from Problem 2 (improved) and Problem 3 (degraded) should render.
        """
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        # Problem 2 (improved) — full detail section rendered
        assert "Transformer-Based Method" in content
        # Problem 3 (degraded) — full detail section rendered
        assert "Contrastive Learning Method" in content

    def test_score_delta_table_present(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "Score Distribution Changes" in content
        assert "Score Δ" in content

    def test_improved_degraded_table_present(self, dirs, tmp_path):
        """
        Ranking summary table:
          Problem 2: b_rank=3 → r_rank=1 → improved
          Problem 3: b_rank=1 → r_rank=2 → degraded
        """
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "Retrieval **improved** gold rank | 1" in content
        assert "Retrieval **degraded** gold rank | 1" in content


class TestPairwiseEndToEnd:
    @pytest.fixture
    def dirs(self, tmp_path):
        base_dir = _write_artifact_dir(tmp_path, "pairwise_current-gpt-5.4",
                                       PAIRWISE_TXT_BASELINE, PAIRWISE_MD_BASELINE,
                                       accuracy_txt=PAIRWISE_ACC_BASELINE)
        ret_dir = _write_artifact_dir(tmp_path, "pairwise_retrieval-gpt-5.4",
                                      PAIRWISE_TXT_RETRIEVAL, PAIRWISE_MD_RETRIEVAL,
                                      accuracy_txt=PAIRWISE_ACC_RETRIEVAL)
        return base_dir, ret_dir

    def test_mode_is_pairwise(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        assert "**Mode:** `pairwise`" in out.read_text()

    def test_summary_shows_accuracy_improvement(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "0.6000" in content   # baseline accuracy
        assert "0.7000" in content   # retrieval accuracy
        assert "+0.1000" in content  # positive delta

    def test_improvement_message(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        assert "improved" in out.read_text().lower()

    def test_problem_level_flips(self, dirs, tmp_path):
        """Problem 1 flips from wrong (baseline) to correct (retrieval)."""
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "Problem-Level Outcome Changes" in content
        # 1 problem fixed, 0 broken
        assert "fixed) | 1" in content
        assert "broken) | 0" in content

    def test_fixed_transition_matrix_present(self, dirs, tmp_path):
        """
        Fixed section: Problem 1 had baseline predict 1 (wrong) and retrieval predict 0 (correct).
        The transition matrix row `1` → column `0` must show count = 1.
        """
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "Fixed — prediction transitions" in content
        # Matrix header contains both labels
        fixed_section = content.split("Fixed — prediction transitions")[1]
        assert "`0`" in fixed_section
        assert "`1`" in fixed_section
        # Row for baseline=1 with count 1 in the 0-column
        # Table row looks like: | `1` | 1 | 0 | (sorted labels: 0, 1)
        assert "| `1` | 1 |" in fixed_section

    def test_broken_transition_matrix_absent_when_no_broken(self, dirs, tmp_path):
        """No broken problems in this fixture, so the broken matrix should not appear."""
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        assert "Broken — prediction transitions" not in out.read_text()


class TestPointwiseEndToEnd:
    @pytest.fixture
    def dirs(self, tmp_path):
        base_dir = _write_artifact_dir(tmp_path, "pointwise_current-gpt-5.4",
                                       POINTWISE_TXT_BASELINE, POINTWISE_MD_BASELINE,
                                       accuracy_txt=POINTWISE_ACC_BASELINE)
        ret_dir = _write_artifact_dir(tmp_path, "pointwise_retrieval-gpt-5.4",
                                      POINTWISE_TXT_RETRIEVAL, POINTWISE_MD_RETRIEVAL,
                                      accuracy_txt=POINTWISE_ACC_RETRIEVAL)
        return base_dir, ret_dir

    def test_mode_is_pointwise(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        assert "**Mode:** `pointwise`" in out.read_text()

    def test_all_metrics_in_table(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        for metric in ["Accuracy", "F1 (macro)", "Precision (POSITIVE)", "Precision (NEGATIVE)",
                       "Recall (POSITIVE)", "Recall (NEGATIVE)", "F1 (POSITIVE)", "F1 (NEGATIVE)"]:
            assert metric in content, f"Expected metric '{metric}' in report"

    def test_delta_column_shows_improvement(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        # Accuracy delta: 0.75 - 0.65 = +0.1000
        assert "+0.1000" in content

    def test_improvement_message(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        assert "improved" in out.read_text().lower()

    def test_improvement_checkmarks_on_positive_deltas(self, dirs, tmp_path):
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        # All metrics improved; each row should have the ✅ emoji
        assert "✅" in content

    def test_instance_level_outcome_table_present(self, dirs, tmp_path):
        """
        Pointwise fixed/broken from POINTWISE_MD_BASELINE vs POINTWISE_MD_RETRIEVAL:
          Instance 1: baseline wrong (GT=POS, pred=NEG), retrieval correct (GT=POS, pred=POS) → fixed
          Instance 2: both correct (GT=NEG, pred=NEG) → no change
          Instance 3: baseline correct (GT=POS, pred=POS), retrieval wrong (GT=POS, pred=NEG) → broken
        """
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "Instance-Level Outcome Changes" in content
        assert "Baseline wrong → Retrieval correct (🟢 fixed) | 1" in content
        assert "Baseline correct → Retrieval wrong (🔴 broken) | 1" in content
        assert "No change | 1" in content

    def test_fixed_instance_title_shown(self, dirs, tmp_path):
        """Fixed instance (Instance 1) should appear in the 'Fixed by Retrieval' section."""
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        fixed_section = content.split("🟢 Instances Fixed by Retrieval")[1].split("###")[0]
        assert "Novel Hypothesis Generator" in fixed_section

    def test_broken_section_shows_retrieval_prediction(self, dirs, tmp_path):
        """
        Broken instances should show the RETRIEVAL prediction (which is wrong),
        not the baseline prediction (which is correct). Instance 3 is broken:
        baseline correct (GT=POS, pred=POS), retrieval wrong (GT=POS, pred=NEG).
        The report should show 'Retrieval predicted: NEGATIVE' for instance 3.
        """
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        broken_section = content.split("🔴 Instances Broken by Retrieval")[1]
        # Instance 3: baseline correct (POS), retrieval wrong (NEG)
        assert "Baseline predicted: `POSITIVE`" in broken_section
        assert "Retrieval predicted: `NEGATIVE`" in broken_section

    def test_idea_text_in_fixed_instance(self, dirs, tmp_path):
        """Idea text should be shown (in a collapsible block) for fixed/broken instances."""
        base_dir, ret_dir = dirs
        out = tmp_path / "report.md"
        run_comparison(base_dir, ret_dir, out)
        content = out.read_text()
        assert "generative framework for scientific hypothesis generation" in content


class TestPointwiseParsers_PerInstance:
    """Unit tests for parse_pointwise_problems_from_md."""

    def test_parse_baseline_problems(self, tmp_path):
        f = tmp_path / "report.md"
        f.write_text(POINTWISE_MD_BASELINE)
        problems = parse_pointwise_problems_from_md(f)
        assert 1 in problems
        assert problems[1]["ground_truth"] == "POSITIVE"
        assert problems[1]["predicted"] == "NEGATIVE"
        assert problems[1]["correct"] is False
        assert "Novel Hypothesis Generator" in problems[1]["title"]

    def test_parse_correct_instances(self, tmp_path):
        f = tmp_path / "report.md"
        f.write_text(POINTWISE_MD_BASELINE)
        problems = parse_pointwise_problems_from_md(f)
        assert problems[2]["correct"] is True
        assert problems[3]["correct"] is True

    def test_parse_retrieval_problems(self, tmp_path):
        f = tmp_path / "report.md"
        f.write_text(POINTWISE_MD_RETRIEVAL)
        problems = parse_pointwise_problems_from_md(f)
        assert problems[1]["correct"] is True   # fixed
        assert problems[3]["correct"] is False  # broken

    def test_idea_text_captured(self, tmp_path):
        f = tmp_path / "report.md"
        f.write_text(POINTWISE_MD_BASELINE)
        problems = parse_pointwise_problems_from_md(f)
        assert "generative framework" in problems[1]["idea_text"]


# ── Retrospective sweep-dir discovery ───────────────────────────────────────────

class TestFindRetrievalPairs:
    def _make_sweep(self, tmp_path: Path, tracks: list[str]) -> Path:
        """Create a minimal sweep dir structure with baseline+retrieval for each track."""
        sweep = tmp_path / "sweep_20260413"
        for track in tracks:
            for ablation, txt_content, md_content in [
                (f"{track}_current",   RANKING_TXT_BASELINE,   RANKING_MD_TEMPLATE),
                (f"{track}_retrieval", RANKING_TXT_RETRIEVAL,  RANKING_MD_RETRIEVAL),
            ]:
                config_name = f"{ablation}-gpt-5.4"
                art = sweep / ablation / config_name / "accuracy_test_artifacts" / "20260413_120000"
                art.mkdir(parents=True)
                (art / "debug_accuracy_report.txt").write_text(txt_content)
                (art / "debug_accuracy_report.txt.md").write_text(md_content)
        return sweep

    def test_finds_single_track_pair(self, tmp_path):
        sweep = self._make_sweep(tmp_path, ["ranking"])
        pairs = find_retrieval_pairs(sweep)
        assert len(pairs) == 1
        assert pairs[0]["track"] == "ranking"

    def test_finds_multiple_track_pairs(self, tmp_path):
        sweep = self._make_sweep(tmp_path, ["ranking", "pairwise", "pointwise"])
        pairs = find_retrieval_pairs(sweep)
        assert len(pairs) == 3
        tracks_found = {p["track"] for p in pairs}
        assert tracks_found == {"ranking", "pairwise", "pointwise"}

    def test_track_filter_limits_results(self, tmp_path):
        sweep = self._make_sweep(tmp_path, ["ranking", "pairwise"])
        pairs = find_retrieval_pairs(sweep, track_filter="ranking")
        assert len(pairs) == 1
        assert pairs[0]["track"] == "ranking"

    def test_no_pairs_when_baseline_missing(self, tmp_path):
        sweep = tmp_path / "sweep"
        # Only create retrieval, no baseline
        art = sweep / "ranking_retrieval" / "ranking_retrieval-gpt-5.4" / "accuracy_test_artifacts" / "ts"
        art.mkdir(parents=True)
        (art / "debug_accuracy_report.txt").write_text(RANKING_TXT_RETRIEVAL)
        (art / "debug_accuracy_report.txt.md").write_text(RANKING_MD_RETRIEVAL)
        pairs = find_retrieval_pairs(sweep)
        assert pairs == []

    def test_output_paths_are_within_retrieval_dir(self, tmp_path):
        sweep = self._make_sweep(tmp_path, ["ranking"])
        pairs = find_retrieval_pairs(sweep)
        assert "ranking_retrieval" in str(pairs[0]["output_path"])
        assert pairs[0]["output_path"].name == "retrieval_impact_comparison.md"

    def test_find_report_files_works_on_model_dir(self, tmp_path):
        sweep = self._make_sweep(tmp_path, ["ranking"])
        pairs = find_retrieval_pairs(sweep)
        # Verify that find_report_files succeeds on the dirs returned by find_retrieval_pairs
        for pair in pairs:
            txt, md = find_report_files(pair["baseline_dir"])
            assert txt.name == "debug_accuracy_report.txt"
            assert md.name == "debug_accuracy_report.txt.md"


class TestRetrospectiveSweepEndToEnd:
    """Runs run_comparison on all pairs found in a sweep dir and checks report output."""

    def _make_sweep(self, tmp_path: Path) -> Path:
        sweep = tmp_path / "sweep"
        for track, txt_b, acc_b, md_b, txt_r, acc_r, md_r in [
            ("ranking",
             RANKING_TXT_BASELINE,   None,                   RANKING_MD_TEMPLATE,
             RANKING_TXT_RETRIEVAL,  None,                   RANKING_MD_RETRIEVAL),
            ("pairwise",
             PAIRWISE_TXT_BASELINE,  PAIRWISE_ACC_BASELINE,  PAIRWISE_MD_BASELINE,
             PAIRWISE_TXT_RETRIEVAL, PAIRWISE_ACC_RETRIEVAL, PAIRWISE_MD_RETRIEVAL),
            ("pointwise",
             POINTWISE_TXT_BASELINE, POINTWISE_ACC_BASELINE, POINTWISE_MD_BASELINE,
             POINTWISE_TXT_RETRIEVAL,POINTWISE_ACC_RETRIEVAL,POINTWISE_MD_RETRIEVAL),
        ]:
            for ablation, txt, acc, md in [
                (f"{track}_current",   txt_b, acc_b, md_b),
                (f"{track}_retrieval", txt_r, acc_r, md_r),
            ]:
                config = f"{ablation}-gpt-5.4"
                art = sweep / ablation / config / "accuracy_test_artifacts" / "ts"
                art.mkdir(parents=True)
                (art / "debug_accuracy_report.txt").write_text(txt)
                (art / "debug_accuracy_report.txt.md").write_text(md)
                if acc is not None:
                    (art / "accuracy_report.txt").write_text(acc)
        return sweep

    def test_all_three_tracks_generate_reports(self, tmp_path):
        sweep = self._make_sweep(tmp_path)
        pairs = find_retrieval_pairs(sweep)
        assert len(pairs) == 3

        for pair in pairs:
            run_comparison(pair["baseline_dir"], pair["retrieval_dir"], pair["output_path"])

        for pair in pairs:
            assert pair["output_path"].exists(), f"Report missing for {pair['track']}"
            content = pair["output_path"].read_text()
            assert f"**Mode:** `{pair['track']}`" in content

    def test_each_report_written_inside_retrieval_dir(self, tmp_path):
        sweep = self._make_sweep(tmp_path)
        pairs = find_retrieval_pairs(sweep)
        for pair in pairs:
            run_comparison(pair["baseline_dir"], pair["retrieval_dir"], pair["output_path"])
            # Report should live inside the retrieval ablation's model dir
            assert "retrieval" in str(pair["output_path"])


# ── merge_ablation_runs.py integration ───────────────────────────────────────────

class TestMergeRetrieval:
    """
    Tests that merge_ablation_runs.py generates retrieval impact reports
    when the merged set contains both *_current and *_retrieval ablations.

    We invoke the merge script as a subprocess (the same way run_ablations.py
    calls other scripts) so the test exercises the real end-to-end path without
    mocking internals.
    """

    def _make_ablation_sweep(
        self,
        base: Path,
        ablation: str,
        mode: str,
        txt: str,
        md: str,
        model: str = "gpt-5.4",
    ) -> Path:
        """
        Create a minimal ablation sweep directory containing:
          <base>/
            ablation_summary.json
            <ablation>/
              <ablation>-<model>/
                accuracy_test_artifacts/
                  <ts>/
                    debug_accuracy_report.txt
                    debug_accuracy_report.txt.md
                    accuracy_report.txt        ← needed by _extract_mode_from_artifact_dir
                    debug_*.md                 ← needed by _extract_model_from_artifact_dir
        """
        config = f"{ablation}-{model}"
        art = base / ablation / config / "accuracy_test_artifacts" / "20260413_120000"
        art.mkdir(parents=True)
        (art / "debug_accuracy_report.txt").write_text(txt)
        (art / "debug_accuracy_report.txt.md").write_text(md)

        # accuracy_report.txt — mode detection and metric parsing.
        # Use the same fixture content as the end-to-end tests so that the merge
        # script can parse accurate metric values (not just the mode header).
        _ACC_BY_MODE_AND_TXT = {
            ("pairwise",  PAIRWISE_TXT_BASELINE):   PAIRWISE_ACC_BASELINE,
            ("pairwise",  PAIRWISE_TXT_RETRIEVAL):  PAIRWISE_ACC_RETRIEVAL,
            ("pointwise", POINTWISE_TXT_BASELINE):  POINTWISE_ACC_BASELINE,
            ("pointwise", POINTWISE_TXT_RETRIEVAL): POINTWISE_ACC_RETRIEVAL,
        }
        acc_content = _ACC_BY_MODE_AND_TXT.get(
            (mode, txt),
            f"Test Mode: {mode}\n",  # fallback for ranking (no separate accuracy_report.txt needed)
        )
        (art / "accuracy_report.txt").write_text(acc_content)

        # debug_*.md — model name extraction
        (art / f"debug_{config}.md").write_text(
            f"# Debug\n## Command Line Arguments\n```json\n"
            f'{{"llm_engine": "{model}"}}\n```\n'
        )

        # ablation_summary.json at the sweep root
        summary_path = base / "ablation_summary.json"
        summary: dict = {"timestamp": "20260413_120000", "runs": []}
        if summary_path.exists():
            import json as _json
            summary = _json.loads(summary_path.read_text())

        summary["runs"].append({
            "ablation": ablation,
            "description": ablation,
            "status": "OK",
            "artifact_dirs": [str(art)],
        })
        import json as _json
        summary_path.write_text(_json.dumps(summary))
        return base

    def _run_merge(self, tmp_path: Path, sweep_dirs: list[Path]) -> Path:
        """Run merge_ablation_runs.py and return the output dir."""
        import subprocess
        import sys

        output_dir = tmp_path / "merged_output"
        output_dir.mkdir(exist_ok=True)

        merge_script = (
            Path(__file__).resolve().parents[1]
            / "src" / "novelty_eval" / "ablation" / "merge_ablation_runs.py"
        )

        # Write a minimal config listing the sweep dirs
        config_path = tmp_path / "merge_config.yaml"
        import yaml as _yaml
        config_path.write_text(
            _yaml.dump({
                "dirs": [str(d) for d in sweep_dirs],
                "output_dir": str(output_dir),
            })
        )

        env = {**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        result = subprocess.run(
            [sys.executable, str(merge_script), "--config", str(config_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        return output_dir, result

    def test_merge_generates_retrieval_impact_report(self, tmp_path):
        """
        When the merged set has ranking_current + ranking_retrieval for the same
        model, merge should produce a retrieval_impact/*.md report.
        """
        sweep1 = tmp_path / "sweep_current"
        sweep2 = tmp_path / "sweep_retrieval"
        self._make_ablation_sweep(sweep1, "ranking_current",   "ranking",
                                  RANKING_TXT_BASELINE,  RANKING_MD_TEMPLATE)
        self._make_ablation_sweep(sweep2, "ranking_retrieval", "ranking",
                                  RANKING_TXT_RETRIEVAL, RANKING_MD_RETRIEVAL)

        output_dir, result = self._run_merge(tmp_path, [sweep1, sweep2])

        retrieval_impact_dir = output_dir / "retrieval_impact"
        assert retrieval_impact_dir.is_dir(), (
            f"No retrieval_impact dir created.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        reports = list(retrieval_impact_dir.glob("**/*.md"))
        assert len(reports) >= 1, (
            f"No comparison reports found.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_merge_report_content_is_correct(self, tmp_path):
        """The generated comparison report should contain expected ranking analysis."""
        sweep1 = tmp_path / "sweep_current"
        sweep2 = tmp_path / "sweep_retrieval"
        self._make_ablation_sweep(sweep1, "ranking_current",   "ranking",
                                  RANKING_TXT_BASELINE,  RANKING_MD_TEMPLATE)
        self._make_ablation_sweep(sweep2, "ranking_retrieval", "ranking",
                                  RANKING_TXT_RETRIEVAL, RANKING_MD_RETRIEVAL)

        output_dir, result = self._run_merge(tmp_path, [sweep1, sweep2])
        reports = list((output_dir / "retrieval_impact").glob("**/*.md"))
        assert reports, f"No reports found.\nstdout: {result.stdout}\nstderr: {result.stderr}"

        content = reports[0].read_text()
        assert "**Mode:** `ranking`" in content
        assert "## Summary" in content
        # Problem 2 improved, problem 3 degraded
        assert "| Retrieval **improved** gold rank | 1 |" in content
        assert "| Retrieval **degraded** gold rank | 1 |" in content

    def test_merge_no_report_when_no_retrieval(self, tmp_path):
        """No retrieval impact report should be generated when there is no retrieval ablation."""
        sweep = tmp_path / "sweep_current_only"
        self._make_ablation_sweep(sweep, "ranking_current", "ranking",
                                  RANKING_TXT_BASELINE, RANKING_MD_TEMPLATE)

        output_dir, _ = self._run_merge(tmp_path, [sweep])

        retrieval_impact_dir = output_dir / "retrieval_impact"
        reports = list(retrieval_impact_dir.glob("**/*.md")) if retrieval_impact_dir.exists() else []
        assert reports == [], f"Unexpected reports: {reports}"

    def test_merge_skips_existing_report(self, tmp_path):
        """
        If the output file already exists, merge should not regenerate it
        (idempotent — useful when re-running a merge that partially completed).
        """
        sweep1 = tmp_path / "sweep_current"
        sweep2 = tmp_path / "sweep_retrieval"
        self._make_ablation_sweep(sweep1, "ranking_current",   "ranking",
                                  RANKING_TXT_BASELINE,  RANKING_MD_TEMPLATE)
        self._make_ablation_sweep(sweep2, "ranking_retrieval", "ranking",
                                  RANKING_TXT_RETRIEVAL, RANKING_MD_RETRIEVAL)

        output_dir, _ = self._run_merge(tmp_path, [sweep1, sweep2])

        # Grab the generated report and record its modification time
        reports = list((output_dir / "retrieval_impact").glob("**/*.md"))
        assert len(reports) == 1
        mtime_before = reports[0].stat().st_mtime

        # Run merge a second time into the same output dir
        import time
        time.sleep(0.05)  # ensure clock advances
        self._run_merge(tmp_path, [sweep1, sweep2])

        # mtime should be unchanged — the file was not re-written
        mtime_after = reports[0].stat().st_mtime
        assert mtime_after == mtime_before, "Report was regenerated despite already existing"

    def test_merge_ranking_baseline_named_all(self, tmp_path):
        """
        Historically the ranking baseline was stored as 'ranking_all' (not 'ranking_current').
        merge_ablation_runs.py must fall back to 'ranking_all' and still produce a report.
        """
        sweep_base = tmp_path / "sweep_ranking_all"
        sweep_ret  = tmp_path / "sweep_ranking_retrieval"
        # Baseline stored under legacy name 'ranking_all'
        self._make_ablation_sweep(sweep_base, "ranking_all",       "ranking",
                                  RANKING_TXT_BASELINE,  RANKING_MD_TEMPLATE)
        self._make_ablation_sweep(sweep_ret,  "ranking_retrieval", "ranking",
                                  RANKING_TXT_RETRIEVAL, RANKING_MD_RETRIEVAL)

        output_dir, result = self._run_merge(tmp_path, [sweep_base, sweep_ret])

        retrieval_impact_dir = output_dir / "retrieval_impact"
        assert retrieval_impact_dir.is_dir(), (
            f"No retrieval_impact dir created.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        reports = list(retrieval_impact_dir.glob("**/*.md"))
        assert len(reports) >= 1, (
            f"Expected a ranking retrieval report when baseline is 'ranking_all', got none.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_merge_ranking_baseline_named_c4(self, tmp_path):
        """
        'c4' is a legacy name for 'current' registered in ablations.yaml.
        merge_ablation_runs.py must resolve it as a valid baseline for
        'ranking_retrieval' via the YAML alias map.
        """
        sweep_base = tmp_path / "sweep_ranking_c4"
        sweep_ret  = tmp_path / "sweep_ranking_retrieval"
        # Baseline stored under legacy name 'c4'
        self._make_ablation_sweep(sweep_base, "c4",                "ranking",
                                  RANKING_TXT_BASELINE,  RANKING_MD_TEMPLATE)
        self._make_ablation_sweep(sweep_ret,  "ranking_retrieval", "ranking",
                                  RANKING_TXT_RETRIEVAL, RANKING_MD_RETRIEVAL)

        output_dir, result = self._run_merge(tmp_path, [sweep_base, sweep_ret])

        retrieval_impact_dir = output_dir / "retrieval_impact"
        assert retrieval_impact_dir.is_dir(), (
            f"No retrieval_impact dir created.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        reports = list(retrieval_impact_dir.glob("**/*.md"))
        assert len(reports) >= 1, (
            f"Expected a ranking retrieval report when baseline is 'c4', got none.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_merge_generates_reports_for_multiple_tracks(self, tmp_path):
        """One report per (mode, model) pair with both current and retrieval."""
        for track, txt_b, md_b, txt_r, md_r in [
            ("ranking",   RANKING_TXT_BASELINE,   RANKING_MD_TEMPLATE,
                          RANKING_TXT_RETRIEVAL,   RANKING_MD_RETRIEVAL),
            ("pairwise",  PAIRWISE_TXT_BASELINE,  PAIRWISE_MD_BASELINE,
                          PAIRWISE_TXT_RETRIEVAL,  PAIRWISE_MD_RETRIEVAL),
            ("pointwise", POINTWISE_TXT_BASELINE, POINTWISE_MD_BASELINE,
                          POINTWISE_TXT_RETRIEVAL, POINTWISE_MD_RETRIEVAL),
        ]:
            sweep_c = tmp_path / f"sweep_{track}_current"
            sweep_r = tmp_path / f"sweep_{track}_retrieval"
            self._make_ablation_sweep(sweep_c, f"{track}_current",   track, txt_b, md_b)
            self._make_ablation_sweep(sweep_r, f"{track}_retrieval", track, txt_r, md_r)

        all_sweeps = [
            tmp_path / f"sweep_{t}_{kind}"
            for t in ("ranking", "pairwise", "pointwise")
            for kind in ("current", "retrieval")
        ]
        output_dir, result = self._run_merge(tmp_path, all_sweeps)

        retrieval_impact_dir = output_dir / "retrieval_impact"
        assert retrieval_impact_dir.is_dir(), (
            f"No retrieval_impact dir.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        reports = list(retrieval_impact_dir.glob("**/*.md"))
        assert len(reports) == 3, (
            f"Expected 3 reports (one per track), got {[r.name for r in reports]}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
