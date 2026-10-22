"""
tests/test_scoring_and_tip.py
─────────────────────────────
Tests for:
  1. change_winner_to_score  — basic ELO accumulation
  2. Criteria / _init_state  — no spurious meta-key dimensions
  3. judge_idea_mec          — correct BPC+MEC aggregation  (non-batch)
  4. run_rr_tournament       — end-to-end scoring + clean output  (non-batch)
  5. get_pairwise_novelty_prompt — tip appended with correct indentation
  6. End-to-end tip threading   — tip reaches the LLM call  (non-batch)
  7. run_benchmark tip selection — correct tip chosen per retrieval mode
  8. judge_idea_mec (batch)  — returns list of requests; correct count
  9. run_rr_tournament (batch) — returns flat list; correct count; tip in prompts
 10. End-to-end tip threading (batch) — tip embedded in every batch request

Tests 3–4 and 6 cover the non-batch path; sections 8–10 mirror them for
batch mode so every regression is caught in both execution modes.
"""

import asyncio
import tempfile
import threading
import shutil
import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from novelty_eval.scoring import (
    change_winner_to_score,
    aggregate_bidirectional_winner,
    aggregate_unidirectional_comparisons,
)
from novelty_eval.tournament import _init_state, run_rr_tournament
from novelty_eval.judge import (
    EVALUATION_CRITERIA,
    EVALUATION_CRITERIA_RETRIEVAL,
    NOVELTY_PAIRWISE_TIP,
    NOVELTY_RETRIEVAL_PAIRWISE_TIP,
    get_pairwise_novelty_prompt,
    judge_idea_mec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ideas(*names):
    """Return a minimal ideas dict suitable for all tournament calls."""
    return {n: {"text": f"Idea text for {n}", "related_work": ""} for n in names}


# ---------------------------------------------------------------------------
# 1. change_winner_to_score — unit tests
# ---------------------------------------------------------------------------

class TestChangeWinnerToScore(unittest.TestCase):

    def test_idea0_wins_increments_score1(self):
        s1, s2 = change_winner_to_score(0, 0.0, 0.0)
        self.assertEqual(s1, 1.0)
        self.assertEqual(s2, 0.0)

    def test_idea1_wins_increments_score2(self):
        s1, s2 = change_winner_to_score(1, 0.0, 0.0)
        self.assertEqual(s1, 0.0)
        self.assertEqual(s2, 1.0)

    def test_tie_leaves_scores_unchanged(self):
        s1, s2 = change_winner_to_score(2, 3.0, 5.0)
        self.assertEqual(s1, 3.0)
        self.assertEqual(s2, 5.0)

    def test_arbitrary_winner_value_treated_as_tie(self):
        """Any value other than 0 or 1 must not change scores."""
        s1, s2 = change_winner_to_score(99, 1.0, 2.0)
        self.assertEqual(s1, 1.0)
        self.assertEqual(s2, 2.0)

    def test_increments_are_cumulative(self):
        """Scores should accumulate across multiple calls."""
        s1, s2 = change_winner_to_score(0, 5.0, 3.0)  # idea0 wins
        s1, s2 = change_winner_to_score(1, s1, s2)     # idea1 wins
        self.assertEqual(s1, 6.0)
        self.assertEqual(s2, 4.0)


# ---------------------------------------------------------------------------
# 2. Criteria dicts and _init_state — no meta-key pollution
# ---------------------------------------------------------------------------

class TestCriteriaMetaKeys(unittest.TestCase):
    """The evaluation-criteria dicts must contain only real scoring dimensions."""

    def _underscore_keys(self, d):
        return [k for k in d if k.startswith("_")]

    def test_evaluation_criteria_has_no_meta_keys(self):
        self.assertEqual(
            [], self._underscore_keys(EVALUATION_CRITERIA),
            "EVALUATION_CRITERIA must not contain _-prefixed keys",
        )

    def test_evaluation_criteria_retrieval_has_no_meta_keys(self):
        self.assertEqual(
            [], self._underscore_keys(EVALUATION_CRITERIA_RETRIEVAL),
            "EVALUATION_CRITERIA_RETRIEVAL must not contain _-prefixed keys",
        )

    def test_init_state_dimensions_are_clean(self):
        _, _, _, dims, _, _, _ = _init_state(_make_ideas("A", "B", "C"), EVALUATION_CRITERIA)
        bad = [d for d in dims if d.startswith("_")]
        self.assertEqual([], bad, f"_init_state produced spurious meta-dimensions: {bad}")

    def test_init_state_dimensional_elo_keys_are_clean(self):
        _, _, _, _, _, dim_elo, _ = _init_state(_make_ideas("A", "B", "C"), EVALUATION_CRITERIA)
        bad = [k for k in dim_elo if k.startswith("_")]
        self.assertEqual([], bad, f"dimensional_elo_scores contains meta keys: {bad}")

    def test_pairwise_tip_not_in_dimensions(self):
        """Explicit regression guard for _pairwise_tip."""
        _, _, _, dims, _, dim_elo, _ = _init_state(_make_ideas("A", "B"), EVALUATION_CRITERIA)
        self.assertNotIn("_pairwise_tip", dims)
        self.assertNotIn("_pairwise_tip", dim_elo)


# ---------------------------------------------------------------------------
# 3. judge_idea_mec — BPC+MEC aggregation
# ---------------------------------------------------------------------------

class TestJudgeIdeaMEC(unittest.IsolatedAsyncioTestCase):
    """Tests for the bidirectional aggregation logic in judge_idea_mec.

    judge_idea is mocked at module level so we control fwd/rev outcomes
    independently without any LLM or file I/O.
    """

    async def _run_mec(self, fwd_winner: int, rev_winner: int) -> dict:
        """Run with mec_k=1, bidirectional=True; ideas are keyed "I" and "J"."""
        async def mock_judge(pos_i, pos_j, *args, **kwargs):
            # Identify forward vs reverse call by the first positional key
            winner = fwd_winner if str(pos_i) == "I" else rev_winner
            return pos_i, pos_j, {"novelty": winner}

        semaphore = asyncio.Semaphore(10)
        with patch('novelty_eval.judge.judge_idea', side_effect=mock_judge):
            _, _, scores = await judge_idea_mec(
                "I", "J",
                {"text": "idea I"}, {"text": "idea J"},
                llm_engine='dummy', semaphore=semaphore,
                output_path='', problem_name='test',
                evaluation_criteria=EVALUATION_CRITERIA,
                mec_k=1, bidirectional=True,
            )
        return scores

    async def test_consistent_agreement_idea_i_wins(self):
        """fwd=0 (I wins fwd) + rev=1 (I also wins rev) → aggregated = 0."""
        scores = await self._run_mec(fwd_winner=0, rev_winner=1)
        self.assertEqual(scores["novelty"], 0)

    async def test_consistent_agreement_idea_j_wins(self):
        """fwd=1 (J wins fwd) + rev=0 (J also wins rev) → aggregated = 1."""
        scores = await self._run_mec(fwd_winner=1, rev_winner=0)
        self.assertEqual(scores["novelty"], 1)

    async def test_positional_disagreement_fwd_fwd_is_tie(self):
        """fwd=0, rev=0 — both orientations favour the first-listed idea → tie (2)."""
        scores = await self._run_mec(fwd_winner=0, rev_winner=0)
        self.assertEqual(scores["novelty"], 2)

    async def test_positional_disagreement_rev_rev_is_tie(self):
        """fwd=1, rev=1 — both orientations favour the second-listed idea → tie (2)."""
        scores = await self._run_mec(fwd_winner=1, rev_winner=1)
        self.assertEqual(scores["novelty"], 2)

    async def test_mec_details_key_present(self):
        scores = await self._run_mec(fwd_winner=0, rev_winner=1)
        self.assertIn("_mec_details", scores)

    async def test_no_pairwise_tip_in_returned_scores(self):
        """_pairwise_tip must never appear as a dimension in MEC output."""
        scores = await self._run_mec(fwd_winner=0, rev_winner=1)
        self.assertNotIn("_pairwise_tip", scores)


# ---------------------------------------------------------------------------
# 4. run_rr_tournament — end-to-end scoring correctness + clean output
# ---------------------------------------------------------------------------

class TestRRTournamentScoring(unittest.IsolatedAsyncioTestCase):

    async def _run_rr(self, winner_map, ideas=None, evaluation_criteria=None,
                      pairwise_tip=None, output_path=None):
        """Run run_rr_tournament with judge_idea_mec mocked to use winner_map.

        winner_map: {(i_key, j_key): winner_int} — canonical alphabetical pair order.
        """
        if ideas is None:
            ideas = _make_ideas("A", "B", "C")
        if evaluation_criteria is None:
            evaluation_criteria = EVALUATION_CRITERIA

        async def mock_mec(i, j, *args, **kwargs):
            key = (str(i), str(j))
            rev = (str(j), str(i))
            if key in winner_map:
                winner = winner_map[key]
            elif rev in winner_map:
                w = winner_map[rev]
                winner = (1 - w) if w in (0, 1) else w   # flip
            else:
                winner = 2  # default tie
            return i, j, {"novelty": winner, "_mec_details": {}}

        semaphore = asyncio.Semaphore(10)
        with patch('novelty_eval.judge.judge_idea_mec', side_effect=mock_mec):
            return await run_rr_tournament(
                ideas=ideas,
                llm_engine='dummy',
                semaphore=semaphore,
                output_path=output_path or '',
                problem_name='test',
                evaluation_criteria=evaluation_criteria,
                pairwise_tip=pairwise_tip,
            )

    async def test_correct_winner_selected(self):
        """A beats B and C; B beats C → A should be elo_selected."""
        result = await self._run_rr({("A", "B"): 0, ("A", "C"): 0, ("B", "C"): 0})
        self.assertEqual(result["ideas"][result["elo_selected"]], "A")

    async def test_elo_scores_ordered_by_wins(self):
        """A(2 wins) > B(1 win) > C(0 wins) → elo_scores must follow this order."""
        result = await self._run_rr({("A", "B"): 0, ("A", "C"): 0, ("B", "C"): 0})
        idx = {k: i for i, k in enumerate(result["ideas"])}
        elo = result["elo_scores"]
        self.assertGreater(elo[idx["A"]], elo[idx["B"]])
        self.assertGreater(elo[idx["B"]], elo[idx["C"]])

    async def test_exact_elo_scores(self):
        """A wins 2 matches, B wins 1, C wins 0 → exact scores 2.0 / 1.0 / 0.0."""
        result = await self._run_rr({("A", "B"): 0, ("A", "C"): 0, ("B", "C"): 0})
        idx = {k: i for i, k in enumerate(result["ideas"])}
        elo = result["elo_scores"]
        self.assertEqual(elo[idx["A"]], 2.0)
        self.assertEqual(elo[idx["B"]], 1.0)
        self.assertEqual(elo[idx["C"]], 0.0)

    async def test_elo_selected_is_argmax_of_elo_scores(self):
        """elo_selected must be the index of the maximum elo_score, not just the
        name of the best idea independently computed."""
        result = await self._run_rr({("A", "B"): 0, ("A", "C"): 0, ("B", "C"): 0})
        elo = result["elo_scores"]
        self.assertEqual(result["elo_selected"], elo.index(max(elo)))

    async def test_all_ties_produce_equal_elo_scores(self):
        result = await self._run_rr({("A", "B"): 2, ("A", "C"): 2, ("B", "C"): 2})
        self.assertEqual(len(set(result["elo_scores"])), 1,
                         "All-tie tournament must yield equal elo scores")

    async def test_output_has_no_meta_keys_in_dimensional_scores(self):
        """dimensional_elo_scores in the output must not contain _-prefixed keys."""
        result = await self._run_rr({("A", "B"): 0, ("A", "C"): 0, ("B", "C"): 0})
        bad = [k for k in result["dimensional_elo_scores"] if k.startswith("_")]
        self.assertEqual([], bad,
                         f"Output dimensional_elo_scores contains meta keys: {bad}")

    async def test_comparisons_count(self):
        """RR must produce exactly C(n, 2) comparison records."""
        from math import comb
        result = await self._run_rr({("A", "B"): 0, ("A", "C"): 0, ("B", "C"): 0})
        self.assertEqual(len(result["comparisons"]), comb(len(result["ideas"]), 2))

    async def test_each_comparison_has_valid_winner_field(self):
        result = await self._run_rr({("A", "B"): 0, ("A", "C"): 0, ("B", "C"): 0})
        for comp in result["comparisons"]:
            self.assertIn("winner", comp)
            self.assertIn(comp["winner"], {0, 1, 2},
                          f"winner must be 0, 1, or 2; got {comp['winner']!r}")

    async def test_comparisons_winner_consistent_with_elo_scores(self):
        """For every comparison record, winner=0 means idea_0 scored higher overall;
        winner=1 means idea_1 scored higher.  Ties (2) are excluded."""
        result = await self._run_rr({("A", "B"): 0, ("A", "C"): 0, ("B", "C"): 0})
        idea_to_elo = {k: result["elo_scores"][i] for i, k in enumerate(result["ideas"])}
        for comp in result["comparisons"]:
            w = comp["winner"]
            if w == 2:
                continue
            winner_idea = comp["idea_0"] if w == 0 else comp["idea_1"]
            loser_idea  = comp["idea_1"] if w == 0 else comp["idea_0"]
            self.assertGreater(
                idea_to_elo[winner_idea], idea_to_elo[loser_idea],
                f"Comparison winner={w} ({comp['idea_0']} vs {comp['idea_1']}) "
                f"inconsistent with elo_scores",
            )

    async def test_dimensional_elo_scores_match_elo_scores_for_single_dimension(self):
        """With one scoring dimension (novelty), dimensional_elo_scores['novelty']
        must be identical to elo_scores."""
        result = await self._run_rr({("A", "B"): 0, ("A", "C"): 0, ("B", "C"): 0})
        self.assertEqual(
            result["elo_scores"],
            result["dimensional_elo_scores"]["novelty"],
        )

    async def test_regression_meta_key_in_criteria_pollutes_dimensional_scores(self):
        """Regression guard: if _pairwise_tip is re-added to the criteria dict, it
        must show up as a spurious key in dimensional_elo_scores — confirming this
        test would catch the regression.
        """
        # Simulate the old bug: criteria dict contains the meta key
        corrupted = {**EVALUATION_CRITERIA, "_pairwise_tip": "some tip"}

        # Run the full call-chain (mock at prompt_openai_client so judge_idea_mec
        # runs its real aggregation logic and creates the spurious dimension).
        captured_lock = threading.Lock()

        def mock_client(prompt, *args, **kwargs):
            return '{"novelty": 0}'

        semaphore = asyncio.Semaphore(10)
        outdir = tempfile.mkdtemp()
        try:
            with patch('novelty_eval.judge.prompt_openai_client', mock_client):
                result = await run_rr_tournament(
                    ideas=_make_ideas("A", "B"),
                    llm_engine='dummy',
                    semaphore=semaphore,
                    output_path=outdir,
                    problem_name='regression',
                    evaluation_criteria=corrupted,
                    mec_k=1,
                    bidirectional=False,  # 1 call per pair, simplest
                    pairwise_tip=None,
                )
        finally:
            shutil.rmtree(outdir)

        # With the corrupted criteria, _pairwise_tip must appear in the output
        # (this is the symptom we fixed — now a canary)
        self.assertIn(
            "_pairwise_tip", result["dimensional_elo_scores"],
            "Corrupted criteria should produce a _pairwise_tip key in dimensional_elo_scores",
        )

        # And with the clean criteria, it must NOT appear
        outdir2 = tempfile.mkdtemp()
        try:
            with patch('novelty_eval.judge.prompt_openai_client', mock_client):
                clean_result = await run_rr_tournament(
                    ideas=_make_ideas("A", "B"),
                    llm_engine='dummy',
                    semaphore=semaphore,
                    output_path=outdir2,
                    problem_name='clean',
                    evaluation_criteria=EVALUATION_CRITERIA,
                    mec_k=1,
                    bidirectional=False,
                    pairwise_tip=None,
                )
        finally:
            shutil.rmtree(outdir2)

        self.assertNotIn("_pairwise_tip", clean_result["dimensional_elo_scores"])


# ---------------------------------------------------------------------------
# 5. get_pairwise_novelty_prompt — tip appended with correct indentation
# ---------------------------------------------------------------------------

class TestPairwiseTipAddition(unittest.TestCase):

    _IDEA = {"text": "Some research idea.", "related_work": ""}

    def _tip_line(self, prompt: str, tip: str) -> str:
        """Return the rendered line that contains the tip text."""
        for line in prompt.splitlines():
            if tip.strip() in line:
                return line
        return ""

    def _leading_spaces(self, line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    # ---- presence / absence ------------------------------------------------

    def test_tip_present_in_prompt(self):
        p = get_pairwise_novelty_prompt(
            self._IDEA, self._IDEA, EVALUATION_CRITERIA, NOVELTY_PAIRWISE_TIP
        )
        self.assertIn(NOVELTY_PAIRWISE_TIP, p)

    def test_no_tip_when_pairwise_tip_is_none(self):
        p = get_pairwise_novelty_prompt(
            self._IDEA, self._IDEA, EVALUATION_CRITERIA, pairwise_tip=None
        )
        self.assertNotIn(NOVELTY_PAIRWISE_TIP, p)
        self.assertNotIn(NOVELTY_RETRIEVAL_PAIRWISE_TIP, p)

    def test_retrieval_tip_in_retrieval_prompt(self):
        p = get_pairwise_novelty_prompt(
            self._IDEA, self._IDEA,
            EVALUATION_CRITERIA_RETRIEVAL, NOVELTY_RETRIEVAL_PAIRWISE_TIP,
        )
        self.assertIn(NOVELTY_RETRIEVAL_PAIRWISE_TIP, p)

    def test_base_tip_absent_when_retrieval_tip_used(self):
        """Base tip and retrieval tip have different wording; they must not cross-contaminate."""
        if NOVELTY_PAIRWISE_TIP == NOVELTY_RETRIEVAL_PAIRWISE_TIP:
            self.skipTest("Tips are identical — cross-contamination check n/a")
        p = get_pairwise_novelty_prompt(
            self._IDEA, self._IDEA,
            EVALUATION_CRITERIA_RETRIEVAL, NOVELTY_RETRIEVAL_PAIRWISE_TIP,
        )
        self.assertNotIn(NOVELTY_PAIRWISE_TIP, p)

    # ---- indentation -------------------------------------------------------

    def test_tip_indentation_matches_normal_criterion(self):
        """Normal Python criterion has 9-space indent; tip line must match."""
        expected = len(
            next(l for l in EVALUATION_CRITERIA["novelty"].splitlines() if l.strip())
        ) - len(
            next(l for l in EVALUATION_CRITERIA["novelty"].splitlines() if l.strip()).lstrip()
        )
        p = get_pairwise_novelty_prompt(
            self._IDEA, self._IDEA, EVALUATION_CRITERIA, NOVELTY_PAIRWISE_TIP
        )
        tip_line = self._tip_line(p, NOVELTY_PAIRWISE_TIP)
        self.assertNotEqual("", tip_line, "Tip must appear in rendered prompt")
        self.assertEqual(
            expected, self._leading_spaces(tip_line),
            f"Expected {expected} leading spaces on the tip line, got {self._leading_spaces(tip_line)}",
        )

    def test_tip_indentation_matches_vague_criterion_zero_indent(self):
        """YAML-parsed vague criterion has 0 leading spaces; tip must also be unindented."""
        vague = (
            "The substantive originality of the idea, characterized by:\n"
            "- The introduction of new frameworks.\n"
            "- The non-trivial synthesis of existing frameworks.\n"
        )
        p = get_pairwise_novelty_prompt(
            self._IDEA, self._IDEA, {"novelty": vague}, NOVELTY_PAIRWISE_TIP
        )
        tip_line = self._tip_line(p, NOVELTY_PAIRWISE_TIP)
        self.assertNotEqual("", tip_line)
        self.assertEqual(
            0, self._leading_spaces(tip_line),
            "Vague criterion has 0 indent; tip must also have 0 leading spaces",
        )

    def test_tip_indentation_adapts_to_arbitrary_indent(self):
        """Tip indentation must follow the first content line of whatever criterion is passed."""
        criterion = "    Four-space indented criterion text.\n    More text.\n"
        p = get_pairwise_novelty_prompt(
            self._IDEA, self._IDEA, {"novelty": criterion}, NOVELTY_PAIRWISE_TIP
        )
        tip_line = self._tip_line(p, NOVELTY_PAIRWISE_TIP)
        self.assertEqual(4, self._leading_spaces(tip_line))


# ---------------------------------------------------------------------------
# 6. End-to-end tip threading — tip must reach the LLM call
# ---------------------------------------------------------------------------

class TestTipThreadingEndToEnd(unittest.IsolatedAsyncioTestCase):
    """Run the full stack (run_rr_tournament → judge_idea_mec → judge_idea →
    get_pairwise_novelty_prompt → LLM) and verify the tip appears in each prompt.
    """

    async def _capture_prompts(self, pairwise_tip, evaluation_criteria=None):
        if evaluation_criteria is None:
            evaluation_criteria = EVALUATION_CRITERIA
        captured = []
        lock = threading.Lock()

        def mock_client(prompt, *args, **kwargs):
            with lock:
                captured.append(prompt)
            return '{"novelty": 0}'

        outdir = tempfile.mkdtemp()
        try:
            with patch('novelty_eval.judge.prompt_openai_client', mock_client):
                await run_rr_tournament(
                    ideas=_make_ideas("X", "Y"),
                    llm_engine='dummy',
                    semaphore=asyncio.Semaphore(10),
                    output_path=outdir,
                    problem_name='tip_e2e',
                    evaluation_criteria=evaluation_criteria,
                    mec_k=1,
                    bidirectional=False,  # 1 LLM call total — enough to verify
                    pairwise_tip=pairwise_tip,
                )
        finally:
            shutil.rmtree(outdir)

        return captured

    async def test_tip_appears_in_every_llm_prompt(self):
        prompts = await self._capture_prompts(NOVELTY_PAIRWISE_TIP)
        self.assertTrue(len(prompts) > 0, "Expected at least one LLM call")
        for p in prompts:
            self.assertIn(NOVELTY_PAIRWISE_TIP, p,
                          "pairwise_tip must appear in the prompt sent to the LLM")

    async def test_no_tip_when_none_passed(self):
        prompts = await self._capture_prompts(pairwise_tip=None)
        self.assertTrue(len(prompts) > 0)
        for p in prompts:
            self.assertNotIn(NOVELTY_PAIRWISE_TIP, p)
            self.assertNotIn(NOVELTY_RETRIEVAL_PAIRWISE_TIP, p)

    async def test_retrieval_tip_reaches_llm_end_to_end(self):
        prompts = await self._capture_prompts(
            pairwise_tip=NOVELTY_RETRIEVAL_PAIRWISE_TIP,
            evaluation_criteria=EVALUATION_CRITERIA_RETRIEVAL,
        )
        self.assertTrue(len(prompts) > 0)
        for p in prompts:
            self.assertIn(NOVELTY_RETRIEVAL_PAIRWISE_TIP, p)


# ---------------------------------------------------------------------------
# 7. run_benchmark tip selection — correct tip chosen per retrieval mode
# ---------------------------------------------------------------------------

class TestRunBenchmarkTipSelection(unittest.IsolatedAsyncioTestCase):
    """Verify that run_pairwise_experiment forwards the right pairwise_tip to
    evaluate_ideas depending on whether retrieval is active.
    """

    _INPUTS = {
        'p1': {
            'context': 'topic',
            'ideas': {'1': 'idea 1', '2': 'idea 2'},
            'expected_winners': ['1'],
        }
    }
    _EVAL_RESULT = {
        'elo_scores': [1.0, 0.0],
        'elo_selected': 0,
        'ideas': ['1', '2'],
        'dimensional_elo_scores': {'novelty': [1.0, 0.0]},
        'comparisons': [{'idea_0': '1', 'idea_1': '2', 'winner': 0}],
    }

    async def _run_and_capture(self, retrieval_cache_path):
        from novelty_eval.run_benchmark import run_pairwise_experiment

        captured = {}

        async def mock_evaluate(enriched, *args, **kwargs):
            captured.update(kwargs)
            return self._EVAL_RESULT

        retrieval_cache = {'p1': {
            '1': {'text': 'idea 1', 'related_work': 'papers'},
            '2': {'text': 'idea 2', 'related_work': 'papers'},
        }} if retrieval_cache_path else {}

        outdir = tempfile.mkdtemp()
        try:
            with patch('novelty_eval.run_benchmark.evaluate_ideas',
                       side_effect=mock_evaluate):
                await run_pairwise_experiment(
                    n_runs=1,
                    llm_engine='dummy',
                    output_dir=outdir,
                    inputs=self._INPUTS,
                    max_workers=1,
                    timestamp='test',
                    retrieval_cache=retrieval_cache,
                    retrieval_cache_path=retrieval_cache_path,
                    use_batch_api=False,
                    effort='none',
                )
        finally:
            shutil.rmtree(outdir)

        return captured

    async def test_base_tip_passed_without_retrieval(self):
        kwargs = await self._run_and_capture(retrieval_cache_path=None)
        self.assertEqual(
            kwargs.get('pairwise_tip'), NOVELTY_PAIRWISE_TIP,
            "Without retrieval, NOVELTY_PAIRWISE_TIP must be forwarded to evaluate_ideas",
        )

    async def test_retrieval_tip_passed_with_retrieval(self):
        kwargs = await self._run_and_capture(retrieval_cache_path='cache.json')
        self.assertEqual(
            kwargs.get('pairwise_tip'), NOVELTY_RETRIEVAL_PAIRWISE_TIP,
            "With retrieval, NOVELTY_RETRIEVAL_PAIRWISE_TIP must be forwarded to evaluate_ideas",
        )

    async def test_tips_are_different(self):
        """Sanity check: the two tip strings must differ so the tests above are meaningful."""
        self.assertNotEqual(
            NOVELTY_PAIRWISE_TIP, NOVELTY_RETRIEVAL_PAIRWISE_TIP,
            "Base tip and retrieval tip must differ — if they are the same the "
            "tip-selection tests cannot distinguish correct from incorrect behaviour",
        )


# ---------------------------------------------------------------------------
# aggregate_bidirectional_winner — unit tests
# ---------------------------------------------------------------------------

class TestAggregateBidirectionalWinner(unittest.TestCase):
    """Direct unit tests for the fwd+rev → aggregated-winner logic."""

    def _call(self, fwd, rev):
        return aggregate_bidirectional_winner({"novelty": fwd}, {"novelty": rev}, "novelty")

    def test_fwd0_rev1_idea0_wins(self):
        """idea_0 wins fwd AND wins rev (appears as idea_1 in rev) → 0."""
        self.assertEqual(self._call(0, 1), 0)

    def test_fwd1_rev0_idea1_wins(self):
        """idea_1 wins fwd AND wins rev (appears as idea_0 in rev) → 1."""
        self.assertEqual(self._call(1, 0), 1)

    def test_fwd0_rev0_is_tie(self):
        """Both orientations favour the first-listed idea → disagreement → 2."""
        self.assertEqual(self._call(0, 0), 2)

    def test_fwd1_rev1_is_tie(self):
        """Both orientations favour the second-listed idea → disagreement → 2."""
        self.assertEqual(self._call(1, 1), 2)


# ---------------------------------------------------------------------------
# aggregate_unidirectional_comparisons — unit tests (batch post-processing path)
# ---------------------------------------------------------------------------

class TestAggregateUnidirectionalComparisons(unittest.TestCase):
    """Tests for the scoring.py function that converts raw batch comparisons into
    per-problem ELO scores, elo_selected, dimensional scores, and re-derived
    pair_comparisons.

    This is the code path exercised after batch API results are retrieved.
    """

    def _make_input(self, idea_wins):
        """Build the comparisons_by_problem input from a list of (i, j, winner) tuples.

        winner=0 → idea_i wins, winner=1 → idea_j wins.
        """
        comps = [
            {"idea_i": str(i), "idea_j": str(j), "scores": {"novelty": w}}
            for i, j, w in idea_wins
        ]
        return {"p1": comps}

    # ---- basic correctness ------------------------------------------------

    def test_winner_selected_correctly(self):
        """A beats B and C; B beats C → elo_selected points to A."""
        out = aggregate_unidirectional_comparisons(
            self._make_input([("A", "B", 0), ("A", "C", 0), ("B", "C", 0)])
        )
        p = out["p1"]
        self.assertEqual(p["ideas"][p["elo_selected"]], "A")

    def test_exact_overall_scores(self):
        """A wins 2, B wins 1, C wins 0 → overall = [2.0, 1.0, 0.0] in idea order."""
        out = aggregate_unidirectional_comparisons(
            self._make_input([("A", "B", 0), ("A", "C", 0), ("B", "C", 0)])
        )
        p = out["p1"]
        idx = {k: i for i, k in enumerate(p["ideas"])}
        self.assertEqual(p["overall"][idx["A"]], 2.0)
        self.assertEqual(p["overall"][idx["B"]], 1.0)
        self.assertEqual(p["overall"][idx["C"]], 0.0)

    def test_elo_selected_is_argmax_of_overall(self):
        out = aggregate_unidirectional_comparisons(
            self._make_input([("A", "B", 0), ("A", "C", 0), ("B", "C", 0)])
        )
        p = out["p1"]
        self.assertEqual(p["elo_selected"], p["overall"].index(max(p["overall"])))

    def test_dimensional_scores_match_overall_for_single_dimension(self):
        """With one dimension (novelty), dimensional score must equal overall."""
        out = aggregate_unidirectional_comparisons(
            self._make_input([("A", "B", 0), ("A", "C", 0), ("B", "C", 0)])
        )
        p = out["p1"]
        self.assertEqual(p["novelty"], p["overall"])

    # ---- pair_comparisons winner field ------------------------------------

    def test_pair_comparisons_winner_consistent_with_overall(self):
        """In the re-derived pair_comparisons, winner must match the elo ordering."""
        out = aggregate_unidirectional_comparisons(
            self._make_input([("A", "B", 0), ("A", "C", 0), ("B", "C", 0)])
        )
        p = out["p1"]
        idea_to_elo = {k: p["overall"][i] for i, k in enumerate(p["ideas"])}
        for comp in p["comparisons"]:
            w = comp["winner"]
            if w == 2:
                continue
            winner_idea = comp["idea_0"] if w == 0 else comp["idea_1"]
            loser_idea  = comp["idea_1"] if w == 0 else comp["idea_0"]
            self.assertGreater(
                idea_to_elo[winner_idea], idea_to_elo[loser_idea],
                f"pair_comparisons winner={w} inconsistent with overall scores",
            )

    def test_pair_comparisons_tie_when_scores_equal(self):
        """All ties → all pair_comparisons winners should be 2."""
        out = aggregate_unidirectional_comparisons(
            self._make_input([("A", "B", 2), ("A", "C", 2), ("B", "C", 2)])
        )
        p = out["p1"]
        for comp in p["comparisons"]:
            self.assertEqual(comp["winner"], 2)

    def test_pair_comparisons_count(self):
        """Re-derived pair_comparisons must cover every unordered pair."""
        from math import comb
        out = aggregate_unidirectional_comparisons(
            self._make_input([("A", "B", 0), ("A", "C", 0), ("B", "C", 0)])
        )
        p = out["p1"]
        self.assertEqual(len(p["comparisons"]), comb(len(p["ideas"]), 2))

    # ---- edge cases -------------------------------------------------------

    def test_unparsable_winner_is_skipped(self):
        """A comparison whose score cannot be cast to int must be silently skipped."""
        comps = [
            {"idea_i": "A", "idea_j": "B", "scores": {"novelty": None}},  # unparsable
            {"idea_i": "A", "idea_j": "C", "scores": {"novelty": 0}},     # A wins
        ]
        out = aggregate_unidirectional_comparisons({"p1": comps})
        p = out["p1"]
        idx = {k: i for i, k in enumerate(p["ideas"])}
        # Only the valid comparison scores; A gets 1 win, others 0
        self.assertEqual(p["overall"][idx["A"]], 1.0)
        self.assertEqual(p["overall"][idx["B"]], 0.0)

    def test_excluded_dimensions_not_in_output(self):
        """overall_winner and thinking_process must be excluded from dimensional scores."""
        comps = [{
            "idea_i": "A", "idea_j": "B",
            "scores": {"novelty": 0, "overall_winner": 0, "thinking_process": "..."},
        }]
        out = aggregate_unidirectional_comparisons({"p1": comps})
        p = out["p1"]
        self.assertNotIn("overall_winner", p)
        self.assertNotIn("thinking_process", p)
        self.assertIn("novelty", p)


# ---------------------------------------------------------------------------
# Batch-mode helpers
# ---------------------------------------------------------------------------

def _prompt_from_batch_request(req: dict) -> str:
    """Extract the user-facing prompt from a batch request dict.

    Handles both the OpenAI (/v1/chat/completions body.messages) format and
    the Anthropic (params.messages) format.
    """
    if "body" in req:      # OpenAI format
        return req["body"]["messages"][0]["content"]
    if "params" in req:    # Anthropic format
        return req["params"]["messages"][0]["content"]
    return ""


# ---------------------------------------------------------------------------
# 8. judge_idea_mec (batch) — returns list; count driven by mec_k / bidirectional
# ---------------------------------------------------------------------------

class TestJudgeIdeaMECBatch(unittest.IsolatedAsyncioTestCase):
    """judge_idea_mec with use_batch_api=True returns (i, j, list_of_requests).

    In batch mode no LLM is called and no file I/O happens, so these tests
    run without any mocking and verify structure and count only.
    """

    async def _run(self, mec_k=1, bidirectional=True, pairwise_tip=None):
        ri, rj, requests = await judge_idea_mec(
            "I", "J",
            {"text": "idea I"}, {"text": "idea J"},
            llm_engine='dummy',
            semaphore=asyncio.Semaphore(10),
            output_path='',
            problem_name='batch_mec',
            evaluation_criteria=EVALUATION_CRITERIA,
            mec_k=mec_k,
            bidirectional=bidirectional,
            use_batch_api=True,
            pairwise_tip=pairwise_tip,
        )
        return ri, rj, requests

    async def test_returns_list_not_scores_dict(self):
        _, _, requests = await self._run()
        self.assertIsInstance(requests, list)

    async def test_unidirectional_mec_k1_one_request(self):
        _, _, requests = await self._run(mec_k=1, bidirectional=False)
        self.assertEqual(len(requests), 1)

    async def test_bidirectional_mec_k1_two_requests(self):
        _, _, requests = await self._run(mec_k=1, bidirectional=True)
        self.assertEqual(len(requests), 2)

    async def test_bidirectional_mec_k3_six_requests(self):
        _, _, requests = await self._run(mec_k=3, bidirectional=True)
        self.assertEqual(len(requests), 6)

    async def test_tip_embedded_in_all_request_prompts(self):
        _, _, requests = await self._run(
            mec_k=1, bidirectional=True, pairwise_tip=NOVELTY_PAIRWISE_TIP
        )
        for req in requests:
            self.assertIn(NOVELTY_PAIRWISE_TIP, _prompt_from_batch_request(req),
                          "pairwise_tip must appear in every batch request prompt")

    async def test_no_tip_when_none_in_batch_mode(self):
        _, _, requests = await self._run(mec_k=1, bidirectional=True, pairwise_tip=None)
        for req in requests:
            prompt = _prompt_from_batch_request(req)
            self.assertNotIn(NOVELTY_PAIRWISE_TIP, prompt)
            self.assertNotIn(NOVELTY_RETRIEVAL_PAIRWISE_TIP, prompt)

    async def test_all_custom_ids_unique(self):
        """Batch API matches responses by custom_id — duplicates would corrupt results."""
        _, _, requests = await self._run(mec_k=3, bidirectional=True)
        ids = [req.get("custom_id") for req in requests]
        self.assertEqual(len(ids), len(set(ids)), "All custom_ids must be unique")


# ---------------------------------------------------------------------------
# 9. run_rr_tournament (batch) — flat list; count; tip in each request
# ---------------------------------------------------------------------------

class TestRRTournamentBatch(unittest.IsolatedAsyncioTestCase):
    """run_rr_tournament with use_batch_api=True must return a flat list of
    batch requests instead of a scores dict.  No LLM / file I/O, no mocking.
    """

    async def _run(self, ideas=None, mec_k=1, bidirectional=False,
                   pairwise_tip=None, evaluation_criteria=None):
        if ideas is None:
            ideas = _make_ideas("A", "B", "C")
        return await run_rr_tournament(
            ideas=ideas,
            llm_engine='dummy',
            semaphore=asyncio.Semaphore(10),
            output_path='',
            problem_name='batch_rr',
            evaluation_criteria=evaluation_criteria or EVALUATION_CRITERIA,
            mec_k=mec_k,
            bidirectional=bidirectional,
            use_batch_api=True,
            pairwise_tip=pairwise_tip,
        )

    async def test_returns_list_not_dict(self):
        result = await self._run()
        self.assertIsInstance(result, list,
                              "Batch mode must return a list of requests, not a scores dict")

    async def test_two_ideas_unidirectional_mec_k1_one_request(self):
        result = await self._run(ideas=_make_ideas("A", "B"))
        self.assertEqual(len(result), 1)   # C(2,2)=1 pair × 1 call

    async def test_three_ideas_unidirectional_mec_k1_three_requests(self):
        result = await self._run()
        self.assertEqual(len(result), 3)   # C(3,2)=3 pairs × 1 call

    async def test_three_ideas_bidirectional_mec_k1_six_requests(self):
        result = await self._run(bidirectional=True, mec_k=1)
        self.assertEqual(len(result), 6)   # 3 pairs × 2 directions

    async def test_three_ideas_bidirectional_mec_k3_eighteen_requests(self):
        result = await self._run(bidirectional=True, mec_k=3)
        self.assertEqual(len(result), 18)  # 3 pairs × 3 fwd + 3 rev

    async def test_all_custom_ids_unique(self):
        result = await self._run(bidirectional=True, mec_k=2)
        ids = [req.get("custom_id") for req in result]
        self.assertEqual(len(ids), len(set(ids)), "All batch custom_ids must be unique")

    async def test_tip_embedded_in_all_batch_prompts(self):
        result = await self._run(pairwise_tip=NOVELTY_PAIRWISE_TIP)
        self.assertTrue(len(result) > 0)
        for req in result:
            self.assertIn(NOVELTY_PAIRWISE_TIP, _prompt_from_batch_request(req),
                          "pairwise_tip must appear in every batch request prompt")

    async def test_no_tip_when_none_in_batch_mode(self):
        result = await self._run(pairwise_tip=None)
        for req in result:
            prompt = _prompt_from_batch_request(req)
            self.assertNotIn(NOVELTY_PAIRWISE_TIP, prompt)
            self.assertNotIn(NOVELTY_RETRIEVAL_PAIRWISE_TIP, prompt)

    async def test_retrieval_tip_embedded_in_batch_prompts(self):
        result = await self._run(pairwise_tip=NOVELTY_RETRIEVAL_PAIRWISE_TIP)
        for req in result:
            self.assertIn(NOVELTY_RETRIEVAL_PAIRWISE_TIP, _prompt_from_batch_request(req))

    async def test_regression_meta_key_in_criteria_corrupts_batch_prompts(self):
        """Regression guard: if _pairwise_tip is re-inserted into the criteria dict
        it will be rendered as a spurious numbered criterion in the prompt
        (even when pairwise_tip=None).  This test makes that visible.
        """
        corrupted = {**EVALUATION_CRITERIA, "_pairwise_tip": NOVELTY_PAIRWISE_TIP}

        # With the bug: tip appears as a rendered criterion (pairwise_tip param NOT used)
        corrupted_result = await self._run(
            ideas=_make_ideas("A", "B"),
            pairwise_tip=None,
            evaluation_criteria=corrupted,
        )
        for req in corrupted_result:
            self.assertIn(
                NOVELTY_PAIRWISE_TIP, _prompt_from_batch_request(req),
                "Corrupted criteria (with _pairwise_tip key) must render tip as a criterion — canary",
            )

        # With clean criteria + pairwise_tip=None: tip must be absent
        clean_result = await self._run(
            ideas=_make_ideas("A", "B"),
            pairwise_tip=None,
            evaluation_criteria=EVALUATION_CRITERIA,
        )
        for req in clean_result:
            self.assertNotIn(NOVELTY_PAIRWISE_TIP, _prompt_from_batch_request(req))


if __name__ == '__main__':
    unittest.main()
