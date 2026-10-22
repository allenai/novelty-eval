import asyncio
import json
import os
import sys
import tempfile
import unittest
import yaml
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import novelty_eval.retrieval.retrieve_candidates as rc
from novelty_eval.retrieval.retrieve_candidates import (
    extract_search_queries,
    retrieve_candidates_for_idea,
    RetrievalConfig,
)
from novelty_eval.retrieval.retrieval_common import (
    load_retrieval_cache,
    load_query_outcomes_log,
    save_query_outcomes_log,
    render_prompt,
    save_retrieval_cache,
    _filter_and_select_candidates,
    _select_candidates_per_query,
)
from novelty_eval.retrieval.retrieval_reports import (
    format_retrieval_debug_info,
    generate_cache_status_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(title="Paper A", abstract="some text", year=2020,
                    publication_date=None, relevance_score=0.5,
                    paper_id="pid1", corpus_id="cid1", source_query="q"):
    return {
        "title": title,
        "abstract": abstract,
        "text": abstract,
        "year": year,
        "publication_date": publication_date,
        "paperId": paper_id,
        "corpus_id": corpus_id,
        "source_query": source_query,
        "relevance_judgement": {"relevance_score": relevance_score},
    }


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Template rendering
# ---------------------------------------------------------------------------

class TestTemplateRendering(unittest.TestCase):

    def test_contribution_user_prompt_contains_idea_and_n(self):
        rendered = render_prompt("contribution_user.jinja2", idea_text="My idea", n_contributions=4)
        self.assertIn("My idea", rendered)
        self.assertIn("4", rendered)

    def test_query_gen_user_prompt_contains_idea_and_contributions(self):
        rendered = render_prompt(
            "query_gen_user.jinja2",
            idea_text="My idea",
            contributions_summary="Methodology:\n  - novel approach",
        )
        self.assertIn("My idea", rendered)
        self.assertIn("novel approach", rendered)


# ---------------------------------------------------------------------------
# 2. extract_search_queries — LLM failure modes
# ---------------------------------------------------------------------------

class TestExtractSearchQueries(unittest.TestCase):

    def _patch_llm(self, side_effect):
        """Patch call_llm_json in the retrieve_candidates module."""
        return patch.object(rc, "call_llm_json", side_effect=side_effect)

    def test_happy_path_returns_queries_and_contributions(self):
        contributions_response = {"Methodology": ["A novel training scheme"]}
        queries_response = {"queries": ["Research on novel training schemes for NLP models"]}

        with self._patch_llm(side_effect=[contributions_response, queries_response]):
            queries, contribs = extract_search_queries("Some idea", model_name="test-model")

        self.assertEqual(queries, {"all_contributions": ["Research on novel training schemes for NLP models"]})
        self.assertEqual(contribs, contributions_response)

    def test_empty_contributions_returns_empty_dicts(self):
        with self._patch_llm(side_effect=[{}]):
            queries, contribs = extract_search_queries("Some idea", model_name="test-model")

        self.assertEqual(queries, {})
        self.assertEqual(contribs, {})

    def test_contributions_with_no_valid_list_values_returns_empty_queries(self):
        # All dimension values are non-list — no valid contribution statements
        bad_contributions = {"Methodology": "not a list", "Theory": None}

        with self._patch_llm(side_effect=[bad_contributions]):
            queries, contribs = extract_search_queries("Some idea", model_name="test-model")

        self.assertEqual(queries, {})
        self.assertEqual(contribs, bad_contributions)

    def test_query_generation_step_fails_returns_empty_queries(self):
        contributions_response = {"Methodology": ["A novel training scheme"]}

        # Step 2 returns no 'queries' key
        with self._patch_llm(side_effect=[contributions_response, {"other_key": []}]):
            queries, contribs = extract_search_queries("Some idea", model_name="test-model")

        self.assertEqual(queries, {})
        self.assertEqual(contribs, contributions_response)

    def test_query_generation_step_returns_empty_dict(self):
        contributions_response = {"Methodology": ["A novel training scheme"]}

        with self._patch_llm(side_effect=[contributions_response, {}]):
            queries, contribs = extract_search_queries("Some idea", model_name="test-model")

        self.assertEqual(queries, {})
        self.assertEqual(contribs, contributions_response)

    def test_nonsensical_idea_returns_empty_queries(self):
        # Even if the idea is "Hello", if LLM fails to extract valid contributions, it should handle gracefully
        with self._patch_llm(side_effect=[{}]):
            queries, contribs = extract_search_queries("Hello", model_name="test-model")
            
        self.assertEqual(queries, {})
        self.assertEqual(contribs, {})

    def test_idea_with_latex_and_special_chars_renders_safely(self):
        # Ensure that strings with latex/special chars don't cause template rendering to crash
        idea_text = "Idea with $math$, \\textbf{bold}, and brackets [like this]."
        rendered = render_prompt("contribution_user.jinja2", idea_text=idea_text, n_contributions=4)
        self.assertIn("$math$", rendered)
        self.assertIn("\\textbf{bold}", rendered)
        self.assertIn("[like this]", rendered)


# ---------------------------------------------------------------------------
# 3. Cutoff date filtering  (tested via retrieve_candidates_for_idea)
# ---------------------------------------------------------------------------

class TestCutoffDateFiltering(unittest.TestCase):
    """
    These tests mock out LLM calls and paper search so we can exercise
    the date-filtering logic in isolation.
    """

    def _run_with_candidates(self, candidates, cutoff_date):
        with patch.object(rc, "extract_search_queries",
                          return_value=({"all": ["q1"]}, {})), \
             patch("novelty_eval.retrieval.retrieve_candidates.search_papers",
                   return_value={"doc_collection": {"documents": candidates}}):
            result_candidates, _ = _run(
                retrieve_candidates_for_idea(
                    "idea text", llm_engine="test", cutoff_date=cutoff_date,
                    top_k=100,
                )
            )
        return result_candidates

    def test_paper_before_cutoff_is_kept(self):
        c = _make_candidate(publication_date="2020-06-01")
        kept = self._run_with_candidates([c], cutoff_date="2021-01-01")
        self.assertEqual(len(kept), 1)

    def test_paper_on_cutoff_date_is_filtered(self):
        c = _make_candidate(publication_date="2021-01-01")
        kept = self._run_with_candidates([c], cutoff_date="2021-01-01")
        self.assertEqual(len(kept), 0)

    def test_paper_after_cutoff_is_filtered(self):
        c = _make_candidate(publication_date="2022-03-15")
        kept = self._run_with_candidates([c], cutoff_date="2021-01-01")
        self.assertEqual(len(kept), 0)

    def test_year_fallback_keeps_paper_strictly_before_cutoff_year(self):
        c = _make_candidate(year=2019, publication_date=None)
        kept = self._run_with_candidates([c], cutoff_date="2021-06-01")
        self.assertEqual(len(kept), 1)

    def test_year_fallback_filters_paper_at_cutoff_year(self):
        # Year == cutoff year is excluded (can't verify exact date)
        c = _make_candidate(year=2021, publication_date=None)
        kept = self._run_with_candidates([c], cutoff_date="2021-06-01")
        self.assertEqual(len(kept), 0)

    def test_paper_with_no_date_is_filtered(self):
        c = _make_candidate(year=None, publication_date=None)
        kept = self._run_with_candidates([c], cutoff_date="2021-01-01")
        self.assertEqual(len(kept), 0)

    def test_invalid_cutoff_format_returns_no_candidates(self):
        c = _make_candidate(publication_date="2020-01-01")
        kept = self._run_with_candidates([c], cutoff_date="not-a-date")
        self.assertEqual(len(kept), 0)


# ---------------------------------------------------------------------------
# 4. Candidate post-processing: filtering, dedup, sorting, top_k
# ---------------------------------------------------------------------------

class TestCandidatePostProcessing(unittest.TestCase):

    def _run_with_candidates(self, candidates, top_k=100, idea_text="idea text",
                             min_relevance_score=0.0, idea_title=None):
        with patch.object(rc, "extract_search_queries",
                          return_value=({"all": ["q1"]}, {})), \
             patch("novelty_eval.retrieval.retrieve_candidates.search_papers",
                   return_value={"doc_collection": {"documents": candidates}}):
            result_candidates, _ = _run(
                retrieve_candidates_for_idea(
                    idea_text, llm_engine="test", top_k=top_k,
                    min_relevance_score=min_relevance_score,
                    idea_title=idea_title,
                )
            )
        return result_candidates

    def test_same_title_candidate_is_filtered(self):
        """A candidate whose title exactly matches the idea's known title should be removed."""
        c = _make_candidate(title="My Research Idea", relevance_score=0.99)
        kept = self._run_with_candidates([c], idea_title="My Research Idea")
        self.assertEqual(len(kept), 0)

    def test_same_title_candidate_case_insensitive(self):
        """Title matching is case-insensitive."""
        c = _make_candidate(title="My Research Idea", relevance_score=0.99)
        kept = self._run_with_candidates([c], idea_title="my research idea")
        self.assertEqual(len(kept), 0)

    def test_different_title_candidate_is_kept(self):
        """A high-relevance candidate with a different title should NOT be filtered."""
        c = _make_candidate(title="A Different Paper", relevance_score=0.99)
        kept = self._run_with_candidates([c], idea_title="My Research Idea")
        self.assertEqual(len(kept), 1)

    def test_candidate_not_filtered_without_idea_title(self):
        """When no idea_title is provided, no title-based filtering occurs."""
        c = _make_candidate(relevance_score=0.99)
        kept = self._run_with_candidates([c])
        self.assertEqual(len(kept), 1)

    def test_candidate_matching_idea_text_is_filtered(self):
        idea = "exact idea text"
        c = _make_candidate(abstract=idea)
        kept = self._run_with_candidates([c], idea_text=idea)
        self.assertEqual(len(kept), 0)

    def test_duplicate_abstract_is_deduped(self):
        c1 = _make_candidate(title="Paper 1", abstract="shared abstract",
                              paper_id="pid1", corpus_id="cid1")
        c2 = _make_candidate(title="Paper 2", abstract="shared abstract",
                              paper_id="pid2", corpus_id="cid2")
        kept = self._run_with_candidates([c1, c2])
        self.assertEqual(len(kept), 1)

    def test_duplicate_corpus_id_is_deduped(self):
        c1 = _make_candidate(title="Paper 1", abstract="abstract one",
                              paper_id="pid1", corpus_id="cid-same")
        c2 = _make_candidate(title="Paper 2", abstract="abstract two",
                              paper_id="pid2", corpus_id="cid-same")
        kept = self._run_with_candidates([c1, c2])
        self.assertEqual(len(kept), 1)

    def test_candidates_sorted_by_relevance_score_descending(self):
        c1 = _make_candidate(title="Low", abstract="abstract low",
                              paper_id="pid1", corpus_id="cid1", relevance_score=0.3)
        c2 = _make_candidate(title="High", abstract="abstract high",
                              paper_id="pid2", corpus_id="cid2", relevance_score=0.8)
        c3 = _make_candidate(title="Mid", abstract="abstract mid",
                              paper_id="pid3", corpus_id="cid3", relevance_score=0.5)
        kept = self._run_with_candidates([c1, c2, c3])
        scores = [(c.get("relevance_judgement") or {}).get("relevance_score", 0) for c in kept]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_limits_results(self):
        candidates = [
            _make_candidate(title=f"Paper {i}", abstract=f"abstract {i}",
                            paper_id=f"pid{i}", corpus_id=f"cid{i}")
            for i in range(10)
        ]
        kept = self._run_with_candidates(candidates, top_k=3)
        self.assertEqual(len(kept), 3)

    def test_min_relevance_score_filters_low_score_candidates(self):
        c_low = _make_candidate(title="Low", abstract="abstract low",
                                paper_id="pid1", corpus_id="cid1", relevance_score=0.2)
        c_high = _make_candidate(title="High", abstract="abstract high",
                                 paper_id="pid2", corpus_id="cid2", relevance_score=0.8)
        kept = self._run_with_candidates([c_low, c_high], min_relevance_score=0.5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["title"], "High")

    def test_same_title_candidate_appears_in_skipped_debug_data(self):
        """Candidates filtered because their title matches the idea title must appear
        in debug_data['skipped_same_title'] so the debug log captures them."""
        c_match = _make_candidate(title="My Idea Title", abstract="abstract match",
                                  paper_id="pid1", corpus_id="cid1", relevance_score=0.9)
        c_other = _make_candidate(title="Unrelated Paper", abstract="abstract other",
                                  paper_id="pid2", corpus_id="cid2", relevance_score=0.7)

        with patch.object(rc, "extract_search_queries",
                          return_value=({"all": ["q1"]}, {})), \
             patch("novelty_eval.retrieval.retrieve_candidates.search_papers",
                   return_value={"doc_collection": {"documents": [c_match, c_other]}}):
            candidates, debug_data = _run(
                retrieve_candidates_for_idea(
                    "idea text", llm_engine="test", top_k=100,
                    idea_title="My Idea Title",
                )
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["title"], "Unrelated Paper")
        skipped = debug_data.get("skipped_same_title", [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["title"], "My Idea Title")


# ---------------------------------------------------------------------------
# 5. load_retrieval_cache
# ---------------------------------------------------------------------------

class TestLoadRetrievalCache(unittest.TestCase):

    def test_returns_empty_dict_when_path_is_none(self):
        self.assertEqual(load_retrieval_cache(None), {})

    def test_returns_empty_dict_when_path_is_empty_string(self):
        self.assertEqual(load_retrieval_cache(""), {})

    def test_returns_empty_dict_when_file_does_not_exist(self):
        self.assertEqual(load_retrieval_cache("/nonexistent/path/cache.json"), {})

    def test_appends_json_suffix_and_loads_correctly(self):
        data = {"key": "value"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            path_without_suffix = f.name[:-5]  # strip .json
        try:
            result = load_retrieval_cache(path_without_suffix)
            self.assertEqual(result, data)
        finally:
            os.remove(f.name)

    def test_loads_existing_cache_correctly(self):
        data = {"p1": {"idea_1": {"candidates": [{"title": "Paper A"}]}}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            result = load_retrieval_cache(path)
            self.assertEqual(result, data)
        finally:
            os.remove(path)

    def test_returns_empty_dict_on_malformed_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ not valid json }")
            path = f.name
        try:
            result = load_retrieval_cache(path)
            self.assertEqual(result, {})
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# 6. save_retrieval_cache
# ---------------------------------------------------------------------------

class TestSaveRetrievalCache(unittest.TestCase):

    def test_does_nothing_when_path_is_none(self):
        # Should not raise
        save_retrieval_cache(None, {"key": "value"})

    def test_does_nothing_when_path_is_empty_string(self):
        save_retrieval_cache("", {"key": "value"})

    def test_appends_json_suffix_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_without_suffix = os.path.join(tmpdir, "cache")
            save_retrieval_cache(path_without_suffix, {"k": "v"})
            self.assertTrue(os.path.exists(path_without_suffix + ".json"))

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "a", "b", "c", "cache.json")
            save_retrieval_cache(nested_path, {"k": "v"})
            self.assertTrue(os.path.exists(nested_path))

    def test_round_trip_save_and_load(self):
        data = {"p1": {"idea_1": {"candidates": [{"title": "Paper A", "year": 2020}]}}}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_retrieval_cache(path, data)
            result = load_retrieval_cache(path)
            self.assertEqual(result, data)
        finally:
            os.remove(path)

    def test_overwrites_existing_cache(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"old": "data"}, f)
            path = f.name
        try:
            save_retrieval_cache(path, {"new": "data"})
            result = load_retrieval_cache(path)
            self.assertEqual(result, {"new": "data"})
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# 7. Search failure / continue
# ---------------------------------------------------------------------------

class TestSearchFailureContinue(unittest.TestCase):
    """
    Verifies that a RuntimeError from one search query does not abort the whole
    retrieval — results from the remaining queries are still collected.
    """

    def _run_with_search(self, search_side_effects):
        """
        search_side_effects: list aligned to queries — each element is either
        a return value dict or an Exception instance to raise.
        """
        call_index = [0]

        def fake_search(query):
            idx = call_index[0]
            call_index[0] += 1
            effect = search_side_effects[idx % len(search_side_effects)]
            if isinstance(effect, Exception):
                raise effect
            return effect

        with patch.object(rc, "extract_search_queries",
                          return_value=({"all": ["q1", "q2"]}, {})), \
             patch("novelty_eval.retrieval.retrieve_candidates.search_papers",
                   side_effect=fake_search):
            candidates, _ = asyncio.run(
                retrieve_candidates_for_idea("idea text", llm_engine="test", top_k=100)
            )
        return candidates

    def _good_result(self, paper_id="pid1", corpus_id="cid1", title="Paper A"):
        return {
            "doc_collection": {
                "documents": [_make_candidate(
                    title=title, paper_id=paper_id, corpus_id=corpus_id
                )]
            }
        }

    def test_one_failing_query_still_returns_results_from_other_query(self):
        effects = [RuntimeError("network error"), self._good_result("pid2", "cid2", "Paper B")]
        kept = self._run_with_search(effects)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["title"], "Paper B")

    def test_all_queries_fail_returns_empty_candidates(self):
        effects = [RuntimeError("fail"), RuntimeError("fail")]
        kept = self._run_with_search(effects)
        self.assertEqual(kept, [])

    def test_no_queries_generated_returns_empty_candidates(self):
        with patch.object(rc, "extract_search_queries", return_value=({}, {})):
            candidates, _ = asyncio.run(
                retrieve_candidates_for_idea("idea text", llm_engine="test")
            )
        self.assertEqual(candidates, [])


# ---------------------------------------------------------------------------
# 8. generate_cache_status_report
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Minimal fake args object for generate_cache_status_report."""
    class Args:
        llm_engine = "gpt-test"
        top_k_candidates = 10
        cutoff_date = None
        max_contributions = 3
        n_queries = 3
        use_semantic_scholar = False
        max_search_workers = 5
        nr_examples = None
        test_inputs = "/fake/inputs.yaml"
    a = Args()
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _make_candidates(n):
    return [{"title": f"Paper {i}", "abstract": f"abstract {i}"} for i in range(n)]


class TestCacheStatusReport(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.tmpdir, "cache.json")
        self.args = _make_args(test_inputs=os.path.join(self.tmpdir, "inputs.yaml"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _run_report(self, cache, inputs):
        return generate_cache_status_report(cache, inputs, self.output_file, self.args)

    def _read_report(self, path):
        with open(path) as f:
            return f.read()

    # ---- coverage numbers ------------------------------------------------

    def test_all_ideas_cached_shows_full_coverage(self):
        inputs = {"p1": {"ideas": {"1": "idea one", "2": "idea two"}}}
        cache = {
            "p1": {
                "1": {"candidates": _make_candidates(3)},
                "2": {"candidates": _make_candidates(2)},
            }
        }
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("100.0%", report)
        self.assertIn("Problems fully covered | 1 / 1", report)
        self.assertIn("Ideas missing from cache | 0", report)

    def test_no_ideas_cached_shows_zero_coverage(self):
        inputs = {"p1": {"ideas": {"1": "idea one", "2": "idea two"}}}
        cache = {}
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("0.0%", report)
        self.assertIn("Ideas missing from cache | 2", report)
        self.assertIn("Problems fully covered | 0 / 1", report)

    def test_partial_cache_shows_correct_counts(self):
        inputs = {"p1": {"ideas": {"1": "idea one", "2": "idea two"}}}
        cache = {"p1": {"1": {"candidates": _make_candidates(4)}}}
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("Ideas cached (with candidates) | 1", report)
        self.assertIn("Ideas missing from cache | 1", report)
        self.assertIn("50.0%", report)

    def test_cached_entry_with_empty_candidates_counted_separately(self):
        inputs = {"p1": {"ideas": {"1": "idea one"}}}
        cache = {"p1": {"1": {"candidates": []}}}  # cached but 0 candidates
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("Ideas cached but 0 candidates | 1", report)
        self.assertIn("Ideas cached (with candidates) | 0", report)

    def test_cached_entry_missing_candidates_key_handled_gracefully(self):
        # Simulates an older cache schema where 'candidates' might be missing
        inputs = {"p1": {"ideas": {"1": "idea one"}}}
        cache = {"p1": {"1": {"related_work": "some text"}}}  # missing 'candidates' entirely
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        # The system should treat this as 0 candidates or un-cached without crashing
        self.assertIn("Total ideas in input | 1", report)

    # ---- per-problem status symbols --------------------------------------

    def test_fully_covered_problem_shows_checkmark(self):
        inputs = {"p1": {"ideas": {"1": "idea one"}}}
        cache = {"p1": {"1": {"candidates": _make_candidates(2)}}}
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("✅", report)

    def test_uncovered_problem_shows_cross(self):
        inputs = {"p1": {"ideas": {"1": "idea one"}}}
        cache = {}
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("❌", report)

    def test_partially_covered_problem_shows_warning(self):
        inputs = {"p1": {"ideas": {"1": "idea one", "2": "idea two"}}}
        cache = {"p1": {"1": {"candidates": _make_candidates(1)}}}
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("⚠️", report)

    # ---- candidate statistics --------------------------------------------

    def test_candidate_stats_are_correct(self):
        inputs = {
            "p1": {"ideas": {"1": "a", "2": "b"}},
        }
        cache = {
            "p1": {
                "1": {"candidates": _make_candidates(3)},
                "2": {"candidates": _make_candidates(7)},
            }
        }
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("Total candidates across all ideas | 10", report)
        self.assertIn("Min candidates per idea | 3", report)
        self.assertIn("Max candidates per idea | 7", report)
        self.assertIn("Avg candidates per idea | 5.0", report)

    def test_empty_cache_candidate_stats_are_zero(self):
        inputs = {"p1": {"ideas": {"1": "idea one"}}}
        cache = {}
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("Total candidates across all ideas | 0", report)
        self.assertIn("Avg candidates per idea | 0.0", report)

    # ---- pointwise format ------------------------------------------------

    def test_pointwise_fully_cached_shows_full_coverage(self):
        inputs = {"0": {"idea": "some idea", "label": "POSITIVE"}}
        cache = {"0": {"candidates": _make_candidates(5)}}
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("100.0%", report)
        self.assertIn("✅", report)

    def test_pointwise_missing_shows_zero_coverage(self):
        inputs = {"0": {"idea": "some idea", "label": "POSITIVE"}}
        cache = {}
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("0.0%", report)
        self.assertIn("❌", report)

    # ---- multiple problems -----------------------------------------------

    def test_multiple_problems_counts_are_aggregated(self):
        inputs = {
            "p1": {"ideas": {"1": "a", "2": "b"}},
            "p2": {"ideas": {"3": "c"}},
        }
        cache = {
            "p1": {
                "1": {"candidates": _make_candidates(2)},
                "2": {"candidates": _make_candidates(2)},
            },
            # p2 not cached at all
        }
        path = self._run_report(cache, inputs)
        report = self._read_report(path)
        self.assertIn("Total ideas in input | 3", report)
        self.assertIn("Ideas cached (with candidates) | 2", report)
        self.assertIn("Ideas missing from cache | 1", report)
        self.assertIn("Problems fully covered | 1 / 2", report)

    # ---- report file location --------------------------------------------

    def test_report_is_written_next_to_cache_file(self):
        inputs = {"p1": {"ideas": {"1": "a"}}}
        cache = {"p1": {"1": {"candidates": _make_candidates(1)}}}
        report_path = self._run_report(cache, inputs)
        expected = os.path.join(self.tmpdir, "cache_status.md")
        self.assertEqual(report_path, expected)
        self.assertTrue(os.path.exists(expected))


# ---------------------------------------------------------------------------
# 9. Save everything we retrieve (debug_data completeness)
# ---------------------------------------------------------------------------

class TestDebugDataCompleteness(unittest.TestCase):
    """
    Verifies that retrieve_candidates_for_idea returns a debug_data dict
    containing all fields needed for caching: contributions, search_queries_dict,
    and candidates.
    """

    def _run_retrieval(self, contributions_response, queries_response, papers):
        with patch.object(rc, "call_llm_json",
                          side_effect=[contributions_response, queries_response]), \
             patch("novelty_eval.retrieval.retrieve_candidates.search_papers",
                   return_value={"doc_collection": {"documents": papers}}):
            candidates, debug_data = asyncio.run(
                retrieve_candidates_for_idea("idea text", llm_engine="test", top_k=100)
            )
        return candidates, debug_data

    def test_debug_data_contains_contributions(self):
        contributions = {"Methodology": ["A novel training scheme"]}
        queries = {"queries": ["Papers about novel training schemes"]}
        papers = [_make_candidate()]
        _, debug_data = self._run_retrieval(contributions, queries, papers)
        self.assertIn("contributions", debug_data)
        self.assertEqual(debug_data["contributions"], contributions)

    def test_debug_data_contains_search_queries_dict(self):
        contributions = {"Methodology": ["A novel training scheme"]}
        queries = {"queries": ["Papers about novel training schemes"]}
        papers = [_make_candidate()]
        _, debug_data = self._run_retrieval(contributions, queries, papers)
        self.assertIn("search_queries_dict", debug_data)
        self.assertIn("all_contributions", debug_data["search_queries_dict"])

    def test_debug_data_contains_candidates(self):
        contributions = {"Methodology": ["A novel training scheme"]}
        queries = {"queries": ["Papers about novel training schemes"]}
        papers = [_make_candidate()]
        _, debug_data = self._run_retrieval(contributions, queries, papers)
        self.assertIn("candidates", debug_data)
        self.assertEqual(len(debug_data["candidates"]), 1)

    def test_all_debug_fields_present_even_when_no_candidates(self):
        contributions = {"Methodology": ["A novel training scheme"]}
        queries = {"queries": ["Papers about novel training schemes"]}
        _, debug_data = self._run_retrieval(contributions, queries, papers=[])
        self.assertIn("contributions", debug_data)
        self.assertIn("search_queries_dict", debug_data)
        self.assertIn("candidates", debug_data)
        self.assertEqual(debug_data["candidates"], [])


# ---------------------------------------------------------------------------
# 10. format_retrieval_debug_info
# ---------------------------------------------------------------------------

def _make_debug_candidate(title="Paper A", url="https://example.com", year=2021,
                           publication_date="2021-03-01", relevance_score=0.8,
                           abstract="An abstract.", snippet="A snippet.", snippet_kind="body",
                           snippet_section="Introduction", source_query="some query",
                           relevance_summary=None, criteria_judgements=None):
    rj = {"relevance_score": relevance_score}
    if relevance_summary is not None:
        rj["relevance_summary"] = relevance_summary
    if criteria_judgements is not None:
        rj["relevance_criteria_judgements"] = criteria_judgements
    return {
        "title": title,
        "url": url,
        "year": year,
        "publication_date": publication_date,
        "abstract": abstract,
        "snippet": snippet,
        "snippet_kind": snippet_kind,
        "snippet_section": snippet_section,
        "source_query": source_query,
        "relevance_judgement": rj,
    }


def _render_debug(candidates, idea_key="0", idea_text="My idea", topic="ML"):
    debug_data = {
        "contributions": {"Methodology": ["A novel approach"]},
        "search_queries_dict": {"all_contributions": ["query one"]},
        "candidates": candidates,
    }
    return format_retrieval_debug_info(idea_key, idea_text, topic, debug_data)


class TestFormatRetrievalDebugInfo(unittest.TestCase):

    def test_basic_fields_present(self):
        c = _make_debug_candidate()
        out = _render_debug([c])
        self.assertIn("Paper A", out)
        self.assertIn("2021", out)
        self.assertIn("2021-03-01", out)
        self.assertIn("0.8", out)
        self.assertIn("some query", out)
        self.assertIn("An abstract.", out)

    def test_relevance_grade_shown_as_x_over_3(self):
        rj = {"relevance_score": 0.8, "relevance": 2}
        c = _make_debug_candidate()
        c["relevance_judgement"] = rj
        out = _render_debug([c])
        self.assertIn("2/3", out)

    def test_relevance_grade_na_when_missing(self):
        c = _make_debug_candidate()
        c["relevance_judgement"] = {"relevance_score": 0.8}
        out = _render_debug([c])
        self.assertIn("N/A", out)

    def test_criteria_table_rendered_with_scores(self):
        criteria = [
            {"name": "novelty", "relevance": 3},
            {"name": "feasibility", "relevance": 2},
        ]
        c = _make_debug_candidate(criteria_judgements=criteria)
        out = _render_debug([c])
        self.assertIn("Relevance by Criterion", out)
        self.assertIn("novelty", out)
        self.assertIn("3/3", out)
        self.assertIn("feasibility", out)
        self.assertIn("2/3", out)

    def test_criteria_top_snippet_in_evidence_column(self):
        criteria = [
            {"name": "novelty", "relevance": 3,
             "relevant_snippets": [{"text": "This is a novel approach."}, {"text": "Second snippet."}]},
        ]
        c = _make_debug_candidate(criteria_judgements=criteria)
        out = _render_debug([c])
        self.assertIn("Evidence", out)
        self.assertIn("This is a novel approach.", out)
        self.assertNotIn("Second snippet.", out)

    def test_criterion_name_bolded_in_evidence_column(self):
        criteria = [{"name": "novelty", "relevance": 3,
                     "relevant_snippets": [{"text": "The novelty of this method is clear."}]}]
        c = _make_debug_candidate(criteria_judgements=criteria)
        out = _render_debug([c])
        self.assertIn("**novelty**", out)

    def test_criterion_names_bolded_in_abstract(self):
        criteria = [
            {"name": "attention", "relevance": 3, "relevant_snippets": []},
            {"name": "transformers", "relevance": 2, "relevant_snippets": []},
        ]
        c = _make_debug_candidate(
            abstract="We study attention in transformers.",
            criteria_judgements=criteria,
        )
        out = _render_debug([c])
        self.assertIn("**attention**", out)
        self.assertIn("**transformers**", out)

    def test_highlighting_is_case_insensitive(self):
        criteria = [{"name": "Attention", "relevance": 3,
                     "relevant_snippets": [{"text": "attention mechanisms are key."}]}]
        c = _make_debug_candidate(criteria_judgements=criteria)
        out = _render_debug([c])
        self.assertIn("**attention**", out)

    def test_criteria_evidence_empty_when_no_relevant_snippets(self):
        criteria = [{"name": "novelty", "relevance": 2, "relevant_snippets": []}]
        c = _make_debug_candidate(criteria_judgements=criteria)
        out = _render_debug([c])
        self.assertIn("| novelty | 2/3 |  |", out)

    def test_criteria_evidence_empty_when_snippet_text_empty(self):
        criteria = [{"name": "novelty", "relevance": 2,
                     "relevant_snippets": [{"text": ""}]}]
        c = _make_debug_candidate(criteria_judgements=criteria)
        out = _render_debug([c])
        self.assertIn("| novelty | 2/3 |  |", out)


    def test_criteria_table_absent_when_not_provided(self):
        c = _make_debug_candidate(criteria_judgements=None)
        out = _render_debug([c])
        self.assertNotIn("Relevance by Criterion", out)

    def test_relevance_summary_rendered(self):
        c = _make_debug_candidate(relevance_summary="This paper is highly relevant because it covers X.")
        out = _render_debug([c])
        self.assertIn("Why Retrieved", out)
        self.assertIn("This paper is highly relevant because it covers X.", out)

    def test_relevance_summary_absent_when_not_provided(self):
        c = _make_debug_candidate(relevance_summary=None)
        out = _render_debug([c])
        self.assertNotIn("Why Retrieved", out)

    def test_both_summary_and_criteria_rendered_together(self):
        criteria = [{"name": "methodology", "relevance": 3}]
        c = _make_debug_candidate(
            relevance_summary="Strong methodological fit.",
            criteria_judgements=criteria,
        )
        out = _render_debug([c])
        self.assertIn("Why Retrieved", out)
        self.assertIn("Strong methodological fit.", out)
        self.assertIn("Relevance by Criterion", out)
        self.assertIn("methodology", out)

    def test_no_candidates_renders_placeholder(self):
        out = _render_debug([])
        self.assertIn("No candidates found", out)

    def test_missing_relevance_judgement_renders_na(self):
        c = _make_debug_candidate()
        del c["relevance_judgement"]
        out = _render_debug([c])
        self.assertIn("N/A", out)
        self.assertNotIn("Relevance by Criterion", out)
        self.assertNotIn("Why Retrieved", out)

    def test_multiple_candidates_all_rendered(self):
        c1 = _make_debug_candidate(title="Paper One", abstract="abstract one")
        c2 = _make_debug_candidate(title="Paper Two", abstract="abstract two")
        out = _render_debug([c1, c2])
        self.assertIn("Paper One", out)
        self.assertIn("Paper Two", out)


class TestFormatPapersForPrompt(unittest.TestCase):
    """Tests for format_papers_for_prompt()."""

    def _candidate(self, title="A Paper", abstract="Some abstract.", relevance_summary=None):
        c = {"title": title, "url": "https://example.com", "abstract": abstract}
        if relevance_summary is not None:
            c["relevance_judgement"] = {"relevance_summary": relevance_summary}
        return c

    def _render(self, candidates, **kwargs):
        from novelty_eval.retrieval.retrieve_candidates import format_papers_for_prompt
        return format_papers_for_prompt(candidates, **kwargs)

    def test_why_retrieved_always_present_in_metadata(self):
        c = self._candidate(relevance_summary="Because it is very relevant.")
        out = self._render([c])
        self.assertNotIn("[why-retrieved-1]", out)
        self.assertIn("why_relevant: Because it is very relevant.", out)

    def test_why_retrieved_present_regardless_of_flag(self):
        c = self._candidate(relevance_summary="Because it is very relevant.")
        out = self._render([c])
        self.assertNotIn("[why-retrieved-1]", out)
        self.assertIn("why_relevant: Because it is very relevant.", out)

    def test_why_relevant_empty_when_summary_missing(self):
        c = self._candidate()  # no relevance_judgement key
        out = self._render([c])
        self.assertIn("why_relevant: \n", out)

    def test_why_relevant_empty_when_summary_empty(self):
        c = self._candidate(relevance_summary="")
        out = self._render([c])
        self.assertIn("why_relevant: \n", out)

    def test_why_relevant_empty_when_no_relevance_judgement(self):
        c = {"title": "T", "url": "u", "abstract": "A"}  # no relevance_judgement at all
        out = self._render([c])
        self.assertIn("why_relevant: \n", out)

    def test_abstract_always_present(self):
        c = self._candidate(abstract="My abstract text.", relevance_summary="Why.")
        out = self._render([c])
        self.assertIn("[abstract-1]", out)
        self.assertIn("My abstract text.", out)

    def test_metadata_fields_indexed_correctly(self):
        c1 = self._candidate(title="P1", relevance_summary="Why 1.")
        c2 = self._candidate(title="P2", relevance_summary="Why 2.")
        out = self._render([c1, c2])
        self.assertIn("[metadata-1]", out)
        self.assertIn("[metadata-2]", out)
        self.assertIn("Why 1.", out)
        self.assertIn("Why 2.", out)

    def test_title_with_brackets_does_not_break_markdown(self):
        # Titles with brackets could conflict with prompt structure if not handled
        c = self._candidate(title="[Draft] A Novel Approach [2023]")
        out = self._render([c])
        self.assertIn("[Draft] A Novel Approach [2023]", out)

    def test_missing_abstract_handled_gracefully(self):
        # What if the API returned a candidate without an abstract?
        c = self._candidate(abstract=None)
        out = self._render([c])
        self.assertIsInstance(out, str)

    def test_extremely_long_abstract_truncation_or_render(self):
        # Ensure very long abstracts don't crash the renderer
        long_abstract = "word " * 10000
        c = self._candidate(abstract=long_abstract)
        out = self._render([c])
        self.assertIsInstance(out, str)
        self.assertIn("word word", out)


# ---------------------------------------------------------------------------
# 11. _select_candidates_per_query
# ---------------------------------------------------------------------------

class TestSelectCandidatesPerQuery(unittest.TestCase):
    """Unit tests for the per-query quota + round-robin fill selection logic."""

    def _make_pool(self, query_counts):
        """
        Build a candidate list where each query contributes the given number of
        candidates, scored 1.0 downwards so they arrive pre-sorted descending.
        query_counts: dict {query_name: count}
        """
        candidates = []
        for q, n in query_counts.items():
            for i in range(n):
                score = 1.0 - i * 0.01
                candidates.append(_make_candidate(
                    title=f"{q}-paper-{i}",
                    abstract=f"{q} abstract {i}",
                    paper_id=f"{q}-pid{i}",
                    corpus_id=f"{q}-cid{i}",
                    relevance_score=score,
                    source_query=q,
                ))
        return candidates

    def _query_counts(self, selected):
        counts = {}
        for c in selected:
            q = c["source_query"]
            counts[q] = counts.get(q, 0) + 1
        return counts

    def test_even_distribution_when_all_queries_have_surplus(self):
        # 2 queries × 10 candidates each, top_k=4 → quota=2 per query
        pool = self._make_pool({"q1": 10, "q2": 10})
        result = _select_candidates_per_query(pool, top_k=4)
        self.assertEqual(len(result), 4)
        counts = self._query_counts(result)
        self.assertEqual(counts.get("q1", 0), 2)
        self.assertEqual(counts.get("q2", 0), 2)

    def test_round_robin_fill_when_one_query_runs_short(self):
        # quota = ceil(6/2) = 3; q1 only has 1 candidate → q2 fills the remaining 2
        pool = self._make_pool({"q1": 1, "q2": 10})
        result = _select_candidates_per_query(pool, top_k=6)
        self.assertEqual(len(result), 6)
        counts = self._query_counts(result)
        self.assertEqual(counts.get("q1", 0), 1)
        self.assertEqual(counts.get("q2", 0), 5)

    def test_total_never_exceeds_top_k(self):
        pool = self._make_pool({"q1": 20, "q2": 20, "q3": 20})
        result = _select_candidates_per_query(pool, top_k=7)
        self.assertLessEqual(len(result), 7)

    def test_returns_all_candidates_when_pool_smaller_than_top_k(self):
        pool = self._make_pool({"q1": 2, "q2": 2})
        result = _select_candidates_per_query(pool, top_k=100)
        self.assertEqual(len(result), 4)

    def test_empty_pool_returns_empty_list(self):
        result = _select_candidates_per_query([], top_k=10)
        self.assertEqual(result, [])

    def test_single_query_contributes_all_top_k(self):
        pool = self._make_pool({"q1": 10})
        result = _select_candidates_per_query(pool, top_k=5)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(c["source_query"] == "q1" for c in result))

    def test_fill_alternates_between_queries_with_surplus(self):
        # quota = ceil(2/2) = 1 per query; top_k=4 means 2 fill slots split between q1 and q2
        pool = self._make_pool({"q1": 5, "q2": 5})
        result = _select_candidates_per_query(pool, top_k=4)
        counts = self._query_counts(result)
        # Each query should contribute exactly 2 (1 from quota + 1 from round-robin)
        self.assertEqual(counts.get("q1", 0), 2)
        self.assertEqual(counts.get("q2", 0), 2)


# ---------------------------------------------------------------------------
# 12. select_and_format_candidates
# ---------------------------------------------------------------------------

class TestSelectAndFormatCandidates(unittest.TestCase):
    """Tests that select_and_format_candidates applies _filter_and_select_candidates
    correctly — it is the single authoritative path for building related_work, shared
    by the cache-write path (retrieve_candidates.py) and all eval paths (run_benchmark.py)."""

    def _make_pool(self, specs):
        """specs: list of (title, score, query)"""
        return [
            _make_candidate(
                title=t,
                abstract=f"abstract for {t}",
                paper_id=f"pid-{i}",
                corpus_id=f"cid-{i}",
                relevance_score=s,
                source_query=q,
            )
            for i, (t, s, q) in enumerate(specs)
        ]

    def test_no_top_k_formats_candidates_in_place(self):
        """Without top_k, candidates are formatted as-is without re-ordering."""
        from novelty_eval.retrieval.retrieve_candidates import (
            select_and_format_candidates, format_papers_for_prompt,
        )
        candidates = self._make_pool([("P1", 0.9, "q1"), ("P2", 0.5, "q2")])
        self.assertEqual(
            select_and_format_candidates(candidates),
            format_papers_for_prompt(candidates),
        )

    def test_top_k_applies_per_query_quota_not_pure_relevance(self):
        """With top_k=2 and 2 queries, per-query quota (1 each) must be respected.
        Pure relevance sort would take both from q1 since they have higher scores —
        this was the bug in the old sort+slice approach."""
        from novelty_eval.retrieval.retrieve_candidates import select_and_format_candidates
        candidates = self._make_pool([
            ("Q1-Best",   0.9, "q1"),
            ("Q1-Second", 0.7, "q1"),
            ("Q2-Best",   0.4, "q2"),
            ("Q2-Second", 0.2, "q2"),
        ])
        result = select_and_format_candidates(candidates, top_k=2)
        self.assertIn("Q1-Best", result)
        self.assertIn("Q2-Best", result)
        self.assertNotIn("Q1-Second", result)
        self.assertNotIn("Q2-Second", result)

    def test_why_relevant_in_metadata_with_top_k(self):
        """why_relevant is always present in metadata when top_k is set."""
        from novelty_eval.retrieval.retrieve_candidates import select_and_format_candidates
        c = _make_candidate(title="Paper W", abstract="some text")
        c["relevance_judgement"]["relevance_summary"] = "Why this paper matters."
        result = select_and_format_candidates([c], top_k=1)
        self.assertIn("why_relevant: Why this paper matters.", result)

    def test_why_relevant_in_metadata_without_top_k(self):
        """why_relevant is always present in metadata when top_k is not set."""
        from novelty_eval.retrieval.retrieve_candidates import select_and_format_candidates
        c = _make_candidate(title="Paper Y", abstract="some text")
        c["relevance_judgement"]["relevance_summary"] = "Relevant because X."
        result = select_and_format_candidates([c])
        self.assertIn("why_relevant: Relevant because X.", result)


# ---------------------------------------------------------------------------
# 13. Cost stage attribution in call_llm
# ---------------------------------------------------------------------------

class TestCallLlmCostStage(unittest.TestCase):
    """call_llm() must attribute its LLM call to the 'retrieval' pipeline stage."""

    def test_retrieval_stage_tag(self):
        try:
            from src.cost_tracker import _current_stage
        except ImportError:
            try:
                from cost_tracker import _current_stage
            except ImportError:
                self.skipTest("cost_tracker not available")

        recorded_stage = []

        def fake_prompt(prompt, engine):
            recorded_stage.append(_current_stage.get())
            return "response"

        with patch.object(rc, 'prompt_openai_client', fake_prompt):
            rc.call_llm("user prompt", "system", "gpt-test")

        self.assertEqual(recorded_stage, ["retrieval"])


# ---------------------------------------------------------------------------
# 14. Cost report integration in main()
# ---------------------------------------------------------------------------

class TestCostReportInMain(unittest.TestCase):
    """Verify that main() resets the cost tracker before processing, writes a
    JSON cost report afterwards, calls _write_cost_report_md when available,
    and skips all reporting when GLOBAL_COST_TRACKER is None."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.tmpdir, "retrieval_cache.json")
        self.inputs_file = os.path.join(self.tmpdir, "inputs.yaml")
        self.config_file = os.path.join(self.tmpdir, "config.yaml")

        with open(self.inputs_file, 'w') as f:
            yaml.dump({"p1": {"context": "ML context", "ideas": {"a": "A research idea."}}}, f)
        with open(self.config_file, 'w') as f:
            yaml.dump({"test_inputs": self.inputs_file, "output_file": self.output_file,
                       "llm_engine": "gpt-test"}, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _make_mock_tracker(self):
        mock = MagicMock()
        mock.get_report.return_value = {
            "models": {},
            "by_stage": {"retrieval": {
                "total_cost_usd": 0.001, "total_calls": 2,
                "total_input_tokens": 100, "total_output_tokens": 50,
                "total_cached_input_tokens": 0, "total_cache_creation_tokens": 0,
            }},
            "total_input_tokens": 100,
            "total_cached_input_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_output_tokens": 50,
            "total_calls": 2,
            "total_cost_usd": 0.001,
        }
        return mock

    def _run_main(self):
        with patch.object(sys, 'argv', ['retrieve_candidates.py', '--config', self.config_file]):
            asyncio.run(rc.main())

    def _base_patches(self, mock_tracker, mock_write_md=None):
        return [
            patch.object(rc, 'GLOBAL_COST_TRACKER', mock_tracker),
            patch.object(rc, '_write_cost_report_md', mock_write_md),
            patch.object(rc, 'extract_search_queries', return_value=({}, {})),
            patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                  return_value={"doc_collection": {"documents": []}}),
        ]

    def test_cost_tracker_reset_at_start(self):
        mock_tracker = self._make_mock_tracker()
        with patch.object(rc, 'GLOBAL_COST_TRACKER', mock_tracker), \
             patch.object(rc, '_write_cost_report_md', None), \
             patch.object(rc, 'extract_search_queries', return_value=({}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": []}}):
            self._run_main()

        mock_tracker.reset.assert_called_once()

    def test_cost_report_json_written(self):
        mock_tracker = self._make_mock_tracker()
        with patch.object(rc, 'GLOBAL_COST_TRACKER', mock_tracker), \
             patch.object(rc, '_write_cost_report_md', None), \
             patch.object(rc, 'extract_search_queries', return_value=({}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": []}}):
            self._run_main()

        cost_json = os.path.join(self.tmpdir, "retrieval_cost_report.json")
        self.assertTrue(os.path.exists(cost_json))
        with open(cost_json) as f:
            report = json.load(f)
        self.assertIn("total_cost_usd", report)
        self.assertIn("total_calls", report)

    def test_cost_report_json_contains_instances_processed(self):
        mock_tracker = self._make_mock_tracker()
        with patch.object(rc, 'GLOBAL_COST_TRACKER', mock_tracker), \
             patch.object(rc, '_write_cost_report_md', None), \
             patch.object(rc, 'extract_search_queries', return_value=({}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": []}}):
            self._run_main()

        cost_json = os.path.join(self.tmpdir, "retrieval_cost_report.json")
        with open(cost_json) as f:
            report = json.load(f)
        self.assertIn("instances_processed", report)
        self.assertEqual(report["instances_processed"], 1)

    def test_write_cost_report_md_called_when_available(self):
        mock_tracker = self._make_mock_tracker()
        mock_write_md = MagicMock()
        with patch.object(rc, 'GLOBAL_COST_TRACKER', mock_tracker), \
             patch.object(rc, '_write_cost_report_md', mock_write_md), \
             patch.object(rc, 'extract_search_queries', return_value=({}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": []}}):
            self._run_main()

        mock_write_md.assert_called_once()
        md_path = mock_write_md.call_args[0][1]
        self.assertTrue(md_path.endswith("retrieval_cost_report.md"))

    def test_md_title_includes_engine_name(self):
        mock_tracker = self._make_mock_tracker()
        mock_write_md = MagicMock()
        with patch.object(rc, 'GLOBAL_COST_TRACKER', mock_tracker), \
             patch.object(rc, '_write_cost_report_md', mock_write_md), \
             patch.object(rc, 'extract_search_queries', return_value=({}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": []}}):
            self._run_main()

        title = mock_write_md.call_args[1]["title"]
        self.assertIn("gpt-test", title)

    def test_no_cost_report_when_tracker_is_none(self):
        """When cost_tracker import failed (GLOBAL_COST_TRACKER is None) no files are written."""
        with patch.object(rc, 'GLOBAL_COST_TRACKER', None), \
             patch.object(rc, 'extract_search_queries', return_value=({}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": []}}):
            self._run_main()

        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "retrieval_cost_report.json")))
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "retrieval_cost_report.md")))

    def test_json_report_written_even_when_md_writer_unavailable(self):
        """JSON is always written; the markdown writer is optional."""
        mock_tracker = self._make_mock_tracker()
        with patch.object(rc, 'GLOBAL_COST_TRACKER', mock_tracker), \
             patch.object(rc, '_write_cost_report_md', None), \
             patch.object(rc, 'extract_search_queries', return_value=({}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": []}}):
            self._run_main()

        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "retrieval_cost_report.json")))
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "retrieval_cost_report.md")))


# ---------------------------------------------------------------------------
# 15. Per-query retry on PaperFinderAPIError
# ---------------------------------------------------------------------------

class TestQueryRetry(unittest.TestCase):

    def _run(self, search_fn, max_retries=3):
        with patch.object(rc, 'extract_search_queries', return_value=({"all": ["q1"]}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   side_effect=search_fn):
            _, debug_data = _run(
                retrieve_candidates_for_idea(
                    "idea text", llm_engine="test",
                    search_max_retries=max_retries, search_retry_delay=0,
                )
            )
        return debug_data["query_outcomes"]

    def test_succeeds_after_transient_failures(self):
        from paper_finder_api import PaperFinderAPIError
        good = {"doc_collection": {"documents": [_make_candidate()]}}
        call_count = [0]

        def flaky(q):
            call_count[0] += 1
            if call_count[0] < 3:
                raise PaperFinderAPIError("busy")
            return good

        outcomes = self._run(flaky, max_retries=3)
        self.assertEqual(outcomes[0]["status"], "success")
        self.assertEqual(outcomes[0]["attempts"], 3)
        self.assertEqual(call_count[0], 3)

    def test_exhausted_retries_produce_failed_status(self):
        from paper_finder_api import PaperFinderAPIError
        call_count = [0]

        def always_fail(q):
            call_count[0] += 1
            raise PaperFinderAPIError("down")

        outcomes = self._run(always_fail, max_retries=3)
        self.assertEqual(outcomes[0]["status"], "failed")
        self.assertEqual(outcomes[0]["attempts"], 3)
        self.assertEqual(call_count[0], 3)

    def test_empty_result_is_not_retried(self):
        call_count = [0]

        def empty(q):
            call_count[0] += 1
            return {"doc_collection": {"documents": []}}

        outcomes = self._run(empty, max_retries=3)
        self.assertEqual(outcomes[0]["status"], "empty")
        self.assertEqual(outcomes[0]["attempts"], 1)
        self.assertEqual(call_count[0], 1)

    def test_success_on_first_attempt_records_attempts_as_one(self):
        good = {"doc_collection": {"documents": [_make_candidate()]}}
        outcomes = self._run(lambda q: good, max_retries=3)
        self.assertEqual(outcomes[0]["status"], "success")
        self.assertEqual(outcomes[0]["attempts"], 1)

    def test_failed_query_contributes_zero_candidates(self):
        from paper_finder_api import PaperFinderAPIError

        with patch.object(rc, 'extract_search_queries', return_value=({"all": ["q1"]}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   side_effect=lambda q: (_ for _ in ()).throw(PaperFinderAPIError("down"))):
            candidates, _ = _run(
                retrieve_candidates_for_idea("idea text", llm_engine="test",
                                             search_max_retries=1, search_retry_delay=0)
            )
        self.assertEqual(candidates, [])


# ---------------------------------------------------------------------------
# 16. Query outcome status discrimination
# ---------------------------------------------------------------------------

class TestQueryOutcomeStatus(unittest.TestCase):

    def _run_single(self, search_fn, max_retries=1):
        with patch.object(rc, 'extract_search_queries', return_value=({"all": ["q1"]}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   side_effect=search_fn):
            _, debug_data = _run(
                retrieve_candidates_for_idea("idea text", llm_engine="test",
                                             search_max_retries=max_retries, search_retry_delay=0)
            )
        return debug_data["query_outcomes"][0]

    def test_success_when_papers_returned(self):
        outcome = self._run_single(lambda q: {"doc_collection": {"documents": [_make_candidate()]}})
        self.assertEqual(outcome["status"], "success")
        self.assertGreater(outcome["papers_found"], 0)

    def test_empty_when_api_ok_but_no_papers(self):
        outcome = self._run_single(lambda q: {"doc_collection": {"documents": []}})
        self.assertEqual(outcome["status"], "empty")
        self.assertEqual(outcome["papers_found"], 0)

    def test_failed_when_api_raises(self):
        from paper_finder_api import PaperFinderAPIError

        def raise_err(q):
            raise PaperFinderAPIError("unavailable")

        outcome = self._run_single(raise_err, max_retries=1)
        self.assertEqual(outcome["status"], "failed")
        self.assertEqual(outcome["papers_found"], 0)

    def test_multiple_queries_tracked_independently(self):
        call_index = [0]
        responses = [
            {"doc_collection": {"documents": [_make_candidate()]}},
            {"doc_collection": {"documents": []}},
        ]

        def multi(q):
            r = responses[call_index[0] % len(responses)]
            call_index[0] += 1
            return r

        with patch.object(rc, 'extract_search_queries',
                          return_value=({"all": ["q1", "q2"]}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   side_effect=multi):
            _, debug_data = _run(
                retrieve_candidates_for_idea("idea text", llm_engine="test",
                                             search_max_retries=1, search_retry_delay=0)
            )

        outcomes = {o["query"]: o["status"] for o in debug_data["query_outcomes"]}
        self.assertEqual(outcomes["q1"], "success")
        self.assertEqual(outcomes["q2"], "empty")


# ---------------------------------------------------------------------------
# 17. Per-query filter pipeline stats
# ---------------------------------------------------------------------------

class TestFilterPipelineStats(unittest.TestCase):

    def test_raw_count_equals_api_result(self):
        candidates = [
            _make_candidate(title=f"P{i}", abstract=f"a{i}", paper_id=f"p{i}",
                            corpus_id=f"c{i}", source_query="q1")
            for i in range(4)
        ]
        _, _, stats = _filter_and_select_candidates(candidates, top_k=10)
        self.assertEqual(stats["q1"]["raw"], 4)

    def test_after_date_reflects_cutoff(self):
        candidates = [
            _make_candidate(title="Old", abstract="old", paper_id="p1", corpus_id="c1",
                            publication_date="2020-01-01", source_query="q1"),
            _make_candidate(title="New", abstract="new", paper_id="p2", corpus_id="c2",
                            publication_date="2026-01-01", source_query="q1"),
        ]
        _, _, stats = _filter_and_select_candidates(candidates, top_k=10, cutoff_date="2021-01-01")
        self.assertEqual(stats["q1"]["raw"], 2)
        self.assertEqual(stats["q1"]["after_date"], 1)

    def test_after_date_is_none_without_cutoff(self):
        c = _make_candidate(source_query="q1")
        _, _, stats = _filter_and_select_candidates([c], top_k=5)
        self.assertIsNone(stats["q1"]["after_date"])

    def test_selected_respects_per_query_quota(self):
        candidates = [
            _make_candidate(title=f"Q1-{i}", abstract=f"q1 abstract {i}",
                            paper_id=f"p1-{i}", corpus_id=f"c1-{i}", source_query="q1")
            for i in range(5)
        ] + [
            _make_candidate(title=f"Q2-{i}", abstract=f"q2 abstract {i}",
                            paper_id=f"p2-{i}", corpus_id=f"c2-{i}", source_query="q2")
            for i in range(5)
        ]
        _, _, stats = _filter_and_select_candidates(candidates, top_k=2)
        self.assertEqual(stats["q1"]["selected"] + stats["q2"]["selected"], 2)

    def test_stats_tracked_per_query_independently(self):
        c1 = _make_candidate(title="Q1", abstract="q1 abstract", paper_id="p1",
                              corpus_id="c1", source_query="q1")
        c2 = _make_candidate(title="Q2", abstract="q2 abstract", paper_id="p2",
                              corpus_id="c2", source_query="q2")
        _, _, stats = _filter_and_select_candidates([c1, c2], top_k=10)
        self.assertEqual(stats["q1"]["raw"], 1)
        self.assertEqual(stats["q2"]["raw"], 1)

    def test_filter_stats_merged_into_query_outcomes(self):
        paper = _make_candidate(publication_date="2020-01-01")
        with patch.object(rc, 'extract_search_queries', return_value=({"all": ["q1"]}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": [paper]}}):
            _, debug_data = _run(
                retrieve_candidates_for_idea("idea text", llm_engine="test",
                                             top_k=5, cutoff_date="2021-01-01")
            )
        outcome = debug_data["query_outcomes"][0]
        for key in ("raw", "after_date", "after_dedup", "after_id_dedup", "selected"):
            self.assertIn(key, outcome, f"missing key: {key}")
        self.assertEqual(outcome["raw"], 1)
        self.assertEqual(outcome["after_date"], 1)


# ---------------------------------------------------------------------------
# 18. Outcome merge on re-run (no duplication)
# ---------------------------------------------------------------------------

class TestOutcomeMergeOnRerun(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _cfg(self):
        return RetrievalConfig(test_inputs="x",
                               output_file=os.path.join(self.tmpdir, "cache.json"))

    def _process(self, problem_id, data, cfg, retrieval_cache, outcomes_log, papers):
        semaphore = asyncio.Semaphore(10)
        with patch.object(rc, 'extract_search_queries', return_value=({"all": ["q1"]}, {})), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": papers}}):
            asyncio.run(rc.process_problem(
                problem_id, data, cfg, retrieval_cache, [], outcomes_log, semaphore, self.tmpdir
            ))

    def test_old_entries_replaced_not_appended(self):
        outcomes_log = [{
            "problem_id": "p1", "idea_key": "a", "query": "old-query",
            "status": "failed", "papers_found": 0, "attempts": 3,
            "raw": 0, "after_date": None, "after_dedup": 0, "after_id_dedup": 0, "selected": 0,
        }]
        self._process("p1", {"ideas": {"a": "some idea"}}, self._cfg(),
                      {}, outcomes_log, [_make_candidate()])

        p1a = [o for o in outcomes_log
               if str(o["problem_id"]) == "p1" and str(o["idea_key"]) == "a"]
        self.assertFalse(any(o["query"] == "old-query" for o in p1a),
                         "stale entry was not replaced")
        self.assertGreater(len(p1a), 0, "new entries were not added")

    def test_other_ideas_outcomes_preserved(self):
        other = {
            "problem_id": "p1", "idea_key": "b", "query": "b-query",
            "status": "success", "papers_found": 5, "attempts": 1,
        }
        outcomes_log = [other.copy()]
        self._process("p1", {"ideas": {"a": "some idea"}}, self._cfg(),
                      {}, outcomes_log, [_make_candidate()])

        b_entries = [o for o in outcomes_log if str(o["idea_key"]) == "b"]
        self.assertEqual(len(b_entries), 1)
        self.assertEqual(b_entries[0]["query"], "b-query")

    def test_no_duplication_across_two_reruns(self):
        outcomes_log = []
        cfg = self._cfg()

        self._process("p1", {"ideas": {"a": "some idea"}}, cfg,
                      {}, outcomes_log, [_make_candidate()])
        count_after_first = len([o for o in outcomes_log
                                  if str(o["problem_id"]) == "p1" and str(o["idea_key"]) == "a"])

        # Second run with cleared cache forces re-retrieval of same idea
        self._process("p1", {"ideas": {"a": "some idea"}}, cfg,
                      {}, outcomes_log, [_make_candidate()])
        count_after_second = len([o for o in outcomes_log
                                   if str(o["problem_id"]) == "p1" and str(o["idea_key"]) == "a"])

        self.assertEqual(count_after_second, count_after_first,
                         "outcome count doubled — entries were appended instead of replaced")


# ---------------------------------------------------------------------------
# 19. retry_failed_queries gate
# ---------------------------------------------------------------------------

class TestRetryFailedQueriesGate(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _cfg(self, retry_failed_queries=False):
        return RetrievalConfig(
            test_inputs="x",
            output_file=os.path.join(self.tmpdir, "cache.json"),
            retry_failed_queries=retry_failed_queries,
        )

    def _cached_idea(self):
        return {"candidates": [_make_candidate()], "related_work": "x",
                "title": "a", "text": "idea", "search_queries_dict": {}, "contributions": {}}

    def _count_retrievals(self, data, cfg, retrieval_cache, outcomes_log):
        """Returns number of times extract_search_queries was called (one per retrieved idea)."""
        call_count = [0]

        def counting_extract(*args, **kwargs):
            call_count[0] += 1
            return ({"all": ["q1"]}, {})

        semaphore = asyncio.Semaphore(10)
        with patch.object(rc, 'extract_search_queries', side_effect=counting_extract), \
             patch('novelty_eval.retrieval.retrieve_candidates.search_papers',
                   return_value={"doc_collection": {"documents": [_make_candidate()]}}):
            asyncio.run(rc.process_problem(
                "p1", data, cfg, retrieval_cache, [], outcomes_log, semaphore, self.tmpdir
            ))
        return call_count[0]

    def test_cached_idea_with_no_past_failures_is_skipped(self):
        cfg = self._cfg(retry_failed_queries=True)
        cache = {"p1": {"a": self._cached_idea()}}
        outcomes = [{"problem_id": "p1", "idea_key": "a", "query": "q",
                     "status": "success", "papers_found": 3, "attempts": 1}]
        calls = self._count_retrievals({"ideas": {"a": "idea a"}}, cfg, cache, outcomes)
        self.assertEqual(calls, 0)

    def test_cached_idea_with_past_failure_retried_when_flag_on(self):
        cfg = self._cfg(retry_failed_queries=True)
        cache = {"p1": {"a": self._cached_idea()}}
        outcomes = [{"problem_id": "p1", "idea_key": "a", "query": "q",
                     "status": "failed", "papers_found": 0, "attempts": 3}]
        calls = self._count_retrievals({"ideas": {"a": "idea a"}}, cfg, cache, outcomes)
        self.assertGreater(calls, 0)

    def test_cached_idea_with_past_failure_skipped_when_flag_off(self):
        cfg = self._cfg(retry_failed_queries=False)
        cache = {"p1": {"a": self._cached_idea()}}
        outcomes = [{"problem_id": "p1", "idea_key": "a", "query": "q",
                     "status": "failed", "papers_found": 0, "attempts": 3}]
        calls = self._count_retrievals({"ideas": {"a": "idea a"}}, cfg, cache, outcomes)
        self.assertEqual(calls, 0)

    def test_only_idea_with_failure_is_retried(self):
        cfg = self._cfg(retry_failed_queries=True)
        cache = {"p1": {"a": self._cached_idea(), "b": self._cached_idea()}}
        outcomes = [
            {"problem_id": "p1", "idea_key": "a", "query": "q",
             "status": "failed", "papers_found": 0, "attempts": 3},
            {"problem_id": "p1", "idea_key": "b", "query": "q",
             "status": "success", "papers_found": 3, "attempts": 1},
        ]
        calls = self._count_retrievals({"ideas": {"a": "idea a", "b": "idea b"}},
                                       cfg, cache, outcomes)
        self.assertEqual(calls, 1)


# ---------------------------------------------------------------------------
# 20. Query outcomes log persistence
# ---------------------------------------------------------------------------

class TestOutcomesLogPersistence(unittest.TestCase):

    def test_round_trip(self):
        data = [{
            "problem_id": "p1", "idea_key": "a", "query": "q1",
            "status": "success", "papers_found": 3, "attempts": 1,
            "raw": 5, "after_date": 4, "after_dedup": 3, "after_id_dedup": 3, "selected": 3,
        }]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "query_outcomes_log.json")
            save_query_outcomes_log(path, data)
            self.assertEqual(load_query_outcomes_log(path), data)

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(load_query_outcomes_log("/tmp/nonexistent_outcomes_xyz_456.json"), [])

    def test_malformed_file_returns_empty_list(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ not valid json }")
            path = f.name
        try:
            self.assertEqual(load_query_outcomes_log(path), [])
        finally:
            os.unlink(path)

    def test_accumulated_across_simulated_runs(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "query_outcomes_log.json")

            run1_entry = {"problem_id": "p1", "idea_key": "a", "query": "q1",
                          "status": "success", "papers_found": 3, "attempts": 1}
            save_query_outcomes_log(path, [run1_entry])

            log = load_query_outcomes_log(path)
            run2_entry = {"problem_id": "p2", "idea_key": "b", "query": "q2",
                          "status": "empty", "papers_found": 0, "attempts": 1}
            log.append(run2_entry)
            save_query_outcomes_log(path, log)

            final = load_query_outcomes_log(path)

        self.assertEqual(len(final), 2)
        self.assertIn(run1_entry, final)
        self.assertIn(run2_entry, final)


# ---------------------------------------------------------------------------
# 21. Query status report rendering
# ---------------------------------------------------------------------------

class TestQueryStatusReportRendering(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write_and_read(self, outcomes):
        from novelty_eval.retrieval.retrieval_reports import write_query_status_report
        write_query_status_report(outcomes, self.tmpdir)
        path = os.path.join(self.tmpdir, "retrieval_debug", "query_status.md")
        with open(path) as f:
            return f.read()

    def test_pipeline_shows_raw_and_selected(self):
        outcomes = [{"problem_id": "p1", "idea_key": "a", "query": "q",
                     "status": "success", "papers_found": 10, "attempts": 1,
                     "raw": 10, "after_date": None, "after_dedup": 8,
                     "after_id_dedup": 8, "selected": 3}]
        report = self._write_and_read(outcomes)
        self.assertIn("10", report)
        self.assertIn("**3 selected**", report)

    def test_pipeline_shows_date_step_when_cutoff_applied(self):
        outcomes = [{"problem_id": "p1", "idea_key": "a", "query": "q",
                     "status": "success", "papers_found": 10, "attempts": 1,
                     "raw": 10, "after_date": 7, "after_dedup": 5,
                     "after_id_dedup": 5, "selected": 2}]
        report = self._write_and_read(outcomes)
        self.assertIn("7 (date)", report)
        self.assertIn("5 (dedup)", report)
        self.assertIn("**2 selected**", report)

    def test_pipeline_omits_date_step_without_cutoff(self):
        outcomes = [{"problem_id": "p1", "idea_key": "a", "query": "q",
                     "status": "success", "papers_found": 5, "attempts": 1,
                     "raw": 5, "after_date": None, "after_dedup": 5,
                     "after_id_dedup": 5, "selected": 5}]
        report = self._write_and_read(outcomes)
        self.assertNotIn("(date)", report)

    def test_failed_query_shows_dash_in_pipeline(self):
        outcomes = [{"problem_id": "p1", "idea_key": "a", "query": "q",
                     "status": "failed", "papers_found": 0, "attempts": 3}]
        report = self._write_and_read(outcomes)
        self.assertIn("❌ failed", report)
        self.assertIn("—", report)

    def test_summary_totals_correct(self):
        outcomes = [
            {"problem_id": "p1", "idea_key": "a", "query": "q1", "status": "success",
             "papers_found": 3, "attempts": 1, "raw": 3, "after_date": None,
             "after_dedup": 3, "after_id_dedup": 3, "selected": 3},
            {"problem_id": "p1", "idea_key": "a", "query": "q2", "status": "empty",
             "papers_found": 0, "attempts": 1, "raw": 0, "after_date": None,
             "after_dedup": 0, "after_id_dedup": 0, "selected": 0},
            {"problem_id": "p1", "idea_key": "b", "query": "q3", "status": "failed",
             "papers_found": 0, "attempts": 3},
        ]
        report = self._write_and_read(outcomes)
        self.assertIn("Total queries | 3", report)
        # 1 idea with failed query out of 2 distinct ideas
        self.assertIn("1 / 2", report)

    def test_empty_outcomes_renders_placeholder(self):
        report = self._write_and_read([])
        self.assertIn("Query Status Report", report)
        self.assertIn("No queries were executed", report)


if __name__ == "__main__":
    unittest.main()
