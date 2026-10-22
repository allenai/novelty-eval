import asyncio
import threading
import unittest
import os
import sys
import yaml
import json
import tempfile
import shutil
from unittest.mock import MagicMock, patch, AsyncMock

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from novelty_eval.run_benchmark import (
    _annotate_gt_winner,
    _finalize_stats,
    read_inputs,
    run_ranking_experiment,
    run_pairwise_experiment,
    run_pointwise_experiment,
    run_random_experiment,
    reprocess_experiment_results,
    _load_config,
    ExperimentStats
)
from novelty_eval.retrieval.retrieve_candidates import format_papers_for_prompt, select_and_format_candidates
from novelty_eval.metrics import compute_pointwise_metrics

class TestAccuracyTest(unittest.TestCase):
    def test_annotate_gt_winner(self):
        comparisons = [
            {'idea_0': '1', 'idea_1': '2'},
            {'idea_0': '2', 'idea_1': '3'},
            {'idea_0': '1', 'idea_1': '3'}
        ]
        expected_winners = {'1'}
        _annotate_gt_winner(comparisons, expected_winners)
        
        self.assertEqual(comparisons[0]['gt_winner'], 0) # 1 wins
        self.assertEqual(comparisons[1]['gt_winner'], None) # neither 2 nor 3 in expected_winners
        self.assertEqual(comparisons[2]['gt_winner'], 0) # 1 wins

    def test_finalize_stats(self):
        stats = ExperimentStats()
        stats.ndcg_scores = [0.8, 0.9, 0.7]
        stats.mrr_scores = [0.5, 1.0, 0.5]
        stats.mr_scores = [2.0, 1.0, 2.0]
        stats.hits_scores[1] = [1.0, 0.0, 1.0]
        
        _finalize_stats(stats)
        
        self.assertAlmostEqual(stats.mean_ndcg, 0.8)
        self.assertAlmostEqual(stats.mean_mrr, 0.6666666666666666)
        self.assertAlmostEqual(stats.mean_mr, 1.6666666666666666)
        self.assertAlmostEqual(stats.mean_hits[1], 0.6666666666666666)

    def test_read_inputs(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'p1': {'expected_winners': ['1'], 'ideas': {'1': 'txt', '2': 'txt'}}}, f)
            temp_path = f.name
        
        try:
            data = read_inputs(temp_path)
            self.assertIn('p1', data)
            self.assertEqual(data['p1']['expected_winners'], ['1'])
        finally:
            os.remove(temp_path)

    def test_load_config_defaults(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test_inputs': 'in.yaml', 'output_file': 'out.json', 'n': 5}, f)
            temp_path = f.name

        try:
            cfg = _load_config(temp_path)
            self.assertEqual(cfg.llm_engine, "gpt-5.2") # from _CONFIG_DEFAULTS
            self.assertEqual(cfg.n, 5)
        finally:
            os.remove(temp_path)

    def test_load_config_pointwise_mode(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test_inputs': 'in.yaml', 'output_file': 'out.json', 'n': 1,
                       'test_mode': 'pointwise'}, f)
            temp_path = f.name
        try:
            cfg = _load_config(temp_path)
            self.assertEqual(cfg.test_mode, 'pointwise')
        finally:
            os.remove(temp_path)

    def test_load_config_rejects_invalid_mode(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test_inputs': 'in.yaml', 'output_file': 'out.json', 'n': 1,
                       'test_mode': 'invalid'}, f)
            temp_path = f.name
        try:
            with self.assertRaises(ValueError):
                _load_config(temp_path)
        finally:
            os.remove(temp_path)

    def test_read_inputs_pointwise(self):
        """read_inputs should not warn when instances have 'label' instead of 'expected_winners'."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                '0': {'idea': 'some idea text', 'label': 'POSITIVE', 'context': 'ml'},
                '1': {'idea': 'another idea',   'label': 'NEGATIVE', 'context': 'ml'},
            }, f)
            temp_path = f.name
        try:
            data = read_inputs(temp_path)
            self.assertIn('0', data)
            self.assertEqual(data['0']['label'], 'POSITIVE')
        finally:
            os.remove(temp_path)

    def test_compute_pointwise_metrics_perfect(self):
        results = {
            '0': {'prediction': 1, 'label': 'POSITIVE'},
            '1': {'prediction': 0, 'label': 'NEGATIVE'},
            '2': {'prediction': 1, 'label': 'POSITIVE'},
            '3': {'prediction': 0, 'label': 'NEGATIVE'},
        }
        m = compute_pointwise_metrics(results, {})
        self.assertAlmostEqual(m['accuracy'], 1.0)
        self.assertAlmostEqual(m['f1_pos'],   1.0)
        self.assertAlmostEqual(m['f1_neg'],   1.0)
        self.assertAlmostEqual(m['f1_macro'], 1.0)
        self.assertEqual(m['support_pos'], 2)
        self.assertEqual(m['support_neg'], 2)

    def test_compute_pointwise_metrics_all_wrong(self):
        results = {
            '0': {'prediction': 0, 'label': 'POSITIVE'},  # FN
            '1': {'prediction': 1, 'label': 'NEGATIVE'},  # FP
        }
        m = compute_pointwise_metrics(results, {})
        self.assertAlmostEqual(m['accuracy'], 0.0)
        self.assertAlmostEqual(m['f1_pos'],   0.0)
        self.assertAlmostEqual(m['f1_neg'],   0.0)
        self.assertAlmostEqual(m['f1_macro'], 0.0)

    def test_compute_pointwise_metrics_mixed(self):
        # 3 correct out of 4: TP=2, TN=1, FP=0, FN=1
        results = {
            '0': {'prediction': 1, 'label': 'POSITIVE'},   # TP
            '1': {'prediction': 1, 'label': 'POSITIVE'},   # TP
            '2': {'prediction': 0, 'label': 'POSITIVE'},   # FN
            '3': {'prediction': 0, 'label': 'NEGATIVE'},   # TN
        }
        m = compute_pointwise_metrics(results, {})
        self.assertAlmostEqual(m['accuracy'], 0.75)
        self.assertAlmostEqual(m['precision_pos'], 1.0)    # TP/(TP+FP) = 2/2
        self.assertAlmostEqual(m['recall_pos'],    2/3)    # TP/(TP+FN) = 2/3
        self.assertAlmostEqual(m['precision_neg'], 1/2)    # TN/(TN+FN) = 1/2
        self.assertAlmostEqual(m['recall_neg'],    1.0)    # TN/(TN+FP) = 1/1
        self.assertAlmostEqual(m['f1_macro'], (m['f1_pos'] + m['f1_neg']) / 2)

    def test_finalize_stats_pointwise(self):
        stats = ExperimentStats()
        stats.pointwise_accuracies   = [0.8, 0.9]
        stats.pointwise_f1_pos       = [0.7, 0.8]
        stats.pointwise_f1_neg       = [0.6, 0.7]
        stats.pointwise_f1_macro     = [0.65, 0.75]
        stats.pointwise_precision_pos = [1.0, 0.9]
        stats.pointwise_precision_neg = [0.5, 0.6]
        stats.pointwise_recall_pos    = [0.6, 0.7]
        stats.pointwise_recall_neg    = [0.8, 0.9]
        _finalize_stats(stats)
        self.assertAlmostEqual(stats.mean_pointwise_accuracy,  0.85)
        self.assertAlmostEqual(stats.mean_pointwise_f1_macro,  0.70)
        self.assertAlmostEqual(stats.mean_pointwise_f1_pos,    0.75)
        self.assertAlmostEqual(stats.mean_pointwise_f1_neg,    0.65)

class TestAccuracyTestAsync(unittest.IsolatedAsyncioTestCase):
    async def test_run_ranking_experiment(self):
        # Setup temp output dir
        output_dir = tempfile.mkdtemp()
        try:
            # Inputs
            inputs = {
                'p1': {
                    'context': 'topic',
                    'ideas': {'1': 'txt1', '2': 'txt2'},
                    'expected_winners': ['1']
                }
            }
            
            # Mock evaluate_ideas
            mock_evaluate_ideas = AsyncMock()
            mock_evaluate_ideas.return_value = {
                'elo_scores': [1200, 1000],
                'elo_selected': [True, False],
                'ideas': ['1', '2'],
                'dimensional_elo_scores': {'overall': [1200, 1000]},
                'comparisons': [
                    {'idea_0': '1', 'idea_1': '2', 'winner': 0}
                ]
            }

            with patch('novelty_eval.run_benchmark.evaluate_ideas', mock_evaluate_ideas):
                stats = await run_ranking_experiment(
                    n_runs=1,
                    llm_engine='dummy',
                    output_dir=output_dir,
                    inputs=inputs,
                    swiss_tournament=False,
                    max_workers=1,
                    timestamp='test',
                    retrieval_cache={},
                    retrieval_cache_path=None,
                    use_batch_api=False,
                    effort='none'
                )
            
            self.assertEqual(len(stats.run_paths), 1)
            self.assertAlmostEqual(stats.mean_ndcg, 1.0)
            self.assertEqual(mock_evaluate_ideas.call_count, 1)

        finally:
            shutil.rmtree(output_dir)

    async def test_run_pairwise_experiment(self):
        output_dir = tempfile.mkdtemp()
        try:
            inputs = {
                'p1': {
                    'context': 'topic',
                    'ideas': {'1': 'txt1', '2': 'txt2'},
                    'expected_winners': ['1']
                }
            }
            
            mock_evaluate_ideas = AsyncMock()
            mock_evaluate_ideas.return_value = {
                'elo_scores': [1200, 1000],
                'elo_selected': [True, False],
                'ideas': ['1', '2'],
                'dimensional_elo_scores': {'overall': [1200, 1000]},
                'comparisons': [
                    {'idea_0': '1', 'idea_1': '2', 'winner': 0}
                ]
            }

            with patch('novelty_eval.run_benchmark.evaluate_ideas', mock_evaluate_ideas):
                stats = await run_pairwise_experiment(
                    n_runs=1,
                    llm_engine='dummy',
                    output_dir=output_dir,
                    inputs=inputs,
                    max_workers=1,
                    timestamp='test',
                    retrieval_cache={},
                    retrieval_cache_path=None,
                    use_batch_api=False,
                    effort='none'
                )
            
            self.assertEqual(len(stats.run_paths), 1)
            self.assertAlmostEqual(stats.mean_pairwise_accuracy, 1.0)
            self.assertEqual(mock_evaluate_ideas.call_count, 1)

        finally:
            shutil.rmtree(output_dir)

    async def test_run_pointwise_experiment(self):
        output_dir = tempfile.mkdtemp()
        try:
            inputs = {
                '0': {'context': 'ml', 'idea': 'novel idea',     'label': 'POSITIVE'},
                '1': {'context': 'ml', 'idea': 'obvious idea',   'label': 'NEGATIVE'},
                '2': {'context': 'ml', 'idea': 'creative idea',  'label': 'POSITIVE'},
                '3': {'context': 'ml', 'idea': 'derivative idea','label': 'NEGATIVE'},
            }

            # LLM always predicts correctly
            side_effects = {'0': 1, '1': 0, '2': 1, '3': 0}
            async def mock_pointwise(problem_name, *args, **kwargs):
                return side_effects[str(problem_name)]

            with patch('novelty_eval.run_benchmark.evaluate_idea_pointwise',
                       side_effect=mock_pointwise):
                stats = await run_pointwise_experiment(
                    n_runs=1,
                    llm_engine='dummy',
                    output_dir=output_dir,
                    inputs=inputs,
                    max_workers=1,
                    timestamp='test',
                    retrieval_cache={},
                    retrieval_cache_path=None,
                    effort='none',
                )

            self.assertEqual(len(stats.run_paths), 1)
            self.assertAlmostEqual(stats.mean_pointwise_accuracy, 1.0)
            self.assertAlmostEqual(stats.mean_pointwise_f1_pos,   1.0)
            self.assertAlmostEqual(stats.mean_pointwise_f1_neg,   1.0)
            self.assertAlmostEqual(stats.mean_pointwise_f1_macro, 1.0)

        finally:
            shutil.rmtree(output_dir)

    async def test_run_pointwise_experiment_with_retrieval(self):
        output_dir = tempfile.mkdtemp()
        try:
            inputs = {
                '0': {'context': 'ml', 'idea': 'novel idea', 'label': 'POSITIVE'},
                '1': {'context': 'ml', 'idea': 'boring idea', 'label': 'NEGATIVE'},
            }
            retrieval_cache = {
                '0': {'related_work': 'some related papers'},
                '1': {'related_work': 'other related papers'},
            }

            captured_related_work = {}
            async def mock_pointwise(problem_name, idea_text, llm_engine, semaphore,
                                     eval_criteria, effort, related_work='', **kwargs):
                captured_related_work[str(problem_name)] = related_work
                return 1 if str(problem_name) == '0' else 0

            with patch('novelty_eval.run_benchmark.evaluate_idea_pointwise',
                       side_effect=mock_pointwise):
                stats = await run_pointwise_experiment(
                    n_runs=1,
                    llm_engine='dummy',
                    output_dir=output_dir,
                    inputs=inputs,
                    max_workers=1,
                    timestamp='test',
                    retrieval_cache=retrieval_cache,
                    retrieval_cache_path='cache.json',  # non-None triggers retrieval
                    effort='none',
                )

            self.assertEqual(captured_related_work['0'], 'some related papers')
            self.assertEqual(captured_related_work['1'], 'other related papers')
            self.assertAlmostEqual(stats.mean_pointwise_accuracy, 1.0)

        finally:
            shutil.rmtree(output_dir)

    async def test_top_k_limits_candidates_from_cache(self):
        """Cache has 5 candidates stored in low-to-high score order; top_k=2 must select the 2 highest-scored ones."""
        output_dir = tempfile.mkdtemp()
        try:
            # Candidates stored in ascending score order (low → high) to verify that
            # consumption-side sorting picks the highest scores, not just the first entries.
            candidates = [
                {'title': f'Paper {i}', 'url': f'http://url{i}', 'abstract': f'Abstract {i}', 'year': 2020,
                 'corpus_id': f'cid-{i}', 'source_query': 'q1',
                 'relevance_judgement': {'relevance_score': i}}
                for i in range(5)  # scores: 0, 1, 2, 3, 4
            ]
            # Cache stores all 5, as if retrieved with a higher limit than top_k_candidates
            retrieval_cache = {
                '0': {
                    'related_work': format_papers_for_prompt(candidates),
                    'candidates': candidates,
                },
            }
            inputs = {'0': {'context': 'ml', 'idea': 'novel idea', 'label': 'POSITIVE'}}

            captured_related_work = {}
            async def mock_pointwise(problem_name, idea_text, llm_engine, semaphore,
                                     eval_criteria, effort, related_work='', **kwargs):
                captured_related_work[str(problem_name)] = related_work
                return 1

            with patch('novelty_eval.run_benchmark.evaluate_idea_pointwise',
                       side_effect=mock_pointwise):
                await run_pointwise_experiment(
                    n_runs=1,
                    llm_engine='dummy',
                    output_dir=output_dir,
                    inputs=inputs,
                    max_workers=1,
                    timestamp='test',
                    retrieval_cache=retrieval_cache,
                    retrieval_cache_path='cache.json',
                    effort='none',
                    top_k_candidates=2,
                )

            rw = captured_related_work['0']
            # Highest-scored candidates (score 4 and 3) must be included
            self.assertIn('Paper 4', rw)
            self.assertIn('Paper 3', rw)
            # Lower-scored candidates must be excluded
            self.assertNotIn('Paper 0', rw)
            self.assertNotIn('Paper 1', rw)
            self.assertNotIn('Paper 2', rw)

        finally:
            shutil.rmtree(output_dir)

    async def test_top_k_limits_candidates_pairwise(self):
        """Pairwise path: top_k_candidates selects highest-scored candidates from cache."""
        output_dir = tempfile.mkdtemp()
        try:
            candidates = [
                {'title': f'Paper {i}', 'url': f'http://url{i}', 'abstract': f'Abstract {i}', 'year': 2020,
                 'corpus_id': f'cid-{i}', 'source_query': 'q1',
                 'relevance_judgement': {'relevance_score': i}}
                for i in range(5)  # scores: 0, 1, 2, 3, 4
            ]
            retrieval_cache = {
                'p1': {
                    '1': {'title': '1', 'text': 'idea 1', 'related_work': format_papers_for_prompt(candidates), 'candidates': candidates},
                    '2': {'title': '2', 'text': 'idea 2', 'related_work': format_papers_for_prompt(candidates), 'candidates': candidates},
                }
            }
            inputs = {
                'p1': {
                    'context': 'topic',
                    'ideas': {'1': 'idea 1', '2': 'idea 2'},
                    'expected_winners': ['1'],
                }
            }

            captured_enriched = {}
            async def mock_evaluate_ideas(enriched_ideas, *args, **kwargs):
                captured_enriched.update(enriched_ideas)
                return {
                    'elo_scores': [1200, 1000], 'elo_selected': [True, False],
                    'ideas': ['1', '2'], 'dimensional_elo_scores': {'overall': [1200, 1000]},
                    'comparisons': [{'idea_0': '1', 'idea_1': '2', 'winner': 0}],
                }

            with patch('novelty_eval.run_benchmark.evaluate_ideas',
                       side_effect=mock_evaluate_ideas):
                await run_pairwise_experiment(
                    n_runs=1,
                    llm_engine='dummy',
                    output_dir=output_dir,
                    inputs=inputs,
                    max_workers=1,
                    timestamp='test',
                    retrieval_cache=retrieval_cache,
                    retrieval_cache_path='cache.json',
                    use_batch_api=False,
                    effort='none',
                    top_k_candidates=2,
                )

            for idea_key in ('1', '2'):
                rw = captured_enriched[idea_key]['related_work']
                self.assertIn('Paper 4', rw)
                self.assertIn('Paper 3', rw)
                self.assertNotIn('Paper 0', rw)
                self.assertNotIn('Paper 1', rw)
                self.assertNotIn('Paper 2', rw)

        finally:
            shutil.rmtree(output_dir)

    async def test_top_k_limits_candidates_ranking(self):
        """Ranking path: top_k_candidates selects highest-scored candidates from cache."""
        output_dir = tempfile.mkdtemp()
        try:
            candidates = [
                {'title': f'Paper {i}', 'url': f'http://url{i}', 'abstract': f'Abstract {i}', 'year': 2020,
                 'corpus_id': f'cid-{i}', 'source_query': 'q1',
                 'relevance_judgement': {'relevance_score': i}}
                for i in range(5)  # scores: 0, 1, 2, 3, 4
            ]
            retrieval_cache = {
                'p1': {
                    '1': {'title': '1', 'text': 'idea 1', 'related_work': format_papers_for_prompt(candidates), 'candidates': candidates},
                    '2': {'title': '2', 'text': 'idea 2', 'related_work': format_papers_for_prompt(candidates), 'candidates': candidates},
                }
            }
            inputs = {
                'p1': {
                    'context': 'topic',
                    'ideas': {'1': 'idea 1', '2': 'idea 2'},
                    'expected_winners': ['1'],
                }
            }

            captured_enriched = {}
            async def mock_evaluate_ideas(enriched_ideas, *args, **kwargs):
                captured_enriched.update(enriched_ideas)
                return {
                    'elo_scores': [1200, 1000], 'elo_selected': [True, False],
                    'ideas': ['1', '2'], 'dimensional_elo_scores': {'overall': [1200, 1000]},
                    'comparisons': [{'idea_0': '1', 'idea_1': '2', 'winner': 0}],
                }

            with patch('novelty_eval.run_benchmark.evaluate_ideas',
                       side_effect=mock_evaluate_ideas):
                await run_ranking_experiment(
                    n_runs=1,
                    llm_engine='dummy',
                    output_dir=output_dir,
                    inputs=inputs,
                    swiss_tournament=False,
                    max_workers=1,
                    timestamp='test',
                    retrieval_cache=retrieval_cache,
                    retrieval_cache_path='cache.json',
                    use_batch_api=False,
                    effort='none',
                    top_k_candidates=2,
                )

            for idea_key in ('1', '2'):
                rw = captured_enriched[idea_key]['related_work']
                self.assertIn('Paper 4', rw)
                self.assertIn('Paper 3', rw)
                self.assertNotIn('Paper 0', rw)
                self.assertNotIn('Paper 1', rw)
                self.assertNotIn('Paper 2', rw)

        finally:
            shutil.rmtree(output_dir)

    async def test_run_pointwise_experiment_uses_flattened_cache_correctly(self):
        """Verify that pointwise experiment uses flat cache keys (instance_id) to look up retrieved candidates."""
        output_dir = tempfile.mkdtemp()
        try:
            # Pointwise instances are usually keyed by "0", "1", "2"...
            inputs = {
                "0": {"idea": "Idea A text", "label": "POSITIVE"},
                "1": {"idea": "Idea B text", "label": "NEGATIVE"},
            }
            # Flattened cache produced by scripts/flatten_retrieval_cache.py
            retrieval_cache = {
                "0": {"related_work": "Context for Idea A", "text": "Idea A text"},
                "1": {"related_work": "Context for Idea B", "text": "Idea B text"}
            }
            
            captured_related_work = {}
            async def mock_pointwise(problem_name, *args, related_work="", **kwargs):
                captured_related_work[str(problem_name)] = related_work
                return 1

            with patch("novelty_eval.run_benchmark.evaluate_idea_pointwise", side_effect=mock_pointwise):
                await run_pointwise_experiment(
                    n_runs=1,
                    llm_engine="dummy",
                    output_dir=output_dir,
                    inputs=inputs,
                    max_workers=1,
                    timestamp="test",
                    retrieval_cache=retrieval_cache,
                    retrieval_cache_path="ptw_cache.json",
                    effort="none",
                )
            
            self.assertEqual(captured_related_work["0"], "Context for Idea A")
            self.assertEqual(captured_related_work["1"], "Context for Idea B")

        finally:
            shutil.rmtree(output_dir)

    async def test_run_random_experiment(self):
        inputs = {
            'p1': {
                'context': 'topic',
                'ideas': {'1': 'txt1', '2': 'txt2'},
                'expected_winners': ['1']
            }
        }
        stats = await run_random_experiment(n_runs=10, inputs=inputs)
        self.assertGreater(len(stats.ndcg_scores), 0)
        # Random should have some mean NDCG, usually around 0.5-0.7 for small sets
        self.assertGreater(stats.mean_ndcg, 0)

    async def test_reprocess_experiment_results(self):
        output_dir = tempfile.mkdtemp()
        try:
            # Create dummy artifacts
            timestamp = 'test_reprocess'
            artifacts_dir = os.path.join(output_dir, "accuracy_test_artifacts", timestamp)
            run_dir = os.path.join(artifacts_dir, "run_rr_0")
            os.makedirs(run_dir)
            
            dummy_scores = {
                'p1': {
                    'overall': [1200, 1000],
                    'ideas': ['1', '2']
                }
            }
            with open(os.path.join(run_dir, "scores.json"), 'w') as f:
                json.dump(dummy_scores, f)
            
            inputs = {
                'p1': {
                    'ideas': {'1': 'txt1', '2': 'txt2'},
                    'expected_winners': ['1']
                }
            }
            
            stats = await reprocess_experiment_results(
                run_prefix='rr',
                artifacts_dir=artifacts_dir,
                inputs=inputs,
                test_mode='ranking'
            )
            
            self.assertEqual(len(stats.run_paths), 1)
            self.assertAlmostEqual(stats.mean_ndcg, 1.0)
            
        finally:
            shutil.rmtree(output_dir)

    async def test_run_pointwise_experiment_passes_mec_k(self):
        """mec_k passed to run_pointwise_experiment must reach evaluate_idea_pointwise."""
        output_dir = tempfile.mkdtemp()
        try:
            inputs = {
                '0': {'context': 'ml', 'idea': 'novel idea', 'label': 'POSITIVE'},
            }
            captured = {}

            async def mock_pw(problem_name, *args, mec_k=1, **kwargs):
                captured['mec_k'] = mec_k
                return 1

            with patch('novelty_eval.run_benchmark.evaluate_idea_pointwise',
                       side_effect=mock_pw):
                await run_pointwise_experiment(
                    n_runs=1,
                    llm_engine='dummy',
                    output_dir=output_dir,
                    inputs=inputs,
                    max_workers=1,
                    timestamp='test',
                    retrieval_cache={},
                    retrieval_cache_path=None,
                    effort='none',
                    mec_k=3,
                )

            self.assertEqual(captured.get('mec_k'), 3)
        finally:
            shutil.rmtree(output_dir)


class TestEvaluateIdeaPointwiseMEC(unittest.IsolatedAsyncioTestCase):
    """Unit tests for evaluate_idea_pointwise majority-vote (mec_k > 1)."""

    async def _call(self, responses: list, mec_k: int) -> "int | None":
        """Run evaluate_idea_pointwise with a mocked LLM that returns *responses* in round-robin.

        responses: list of JSON strings or None (simulating LLM failure).
        Thread-safe counter ensures deterministic consumption across concurrent threads.
        """
        from novelty_eval.judge import evaluate_idea_pointwise
        semaphore = asyncio.Semaphore(10)
        idx = [0]
        lock = threading.Lock()

        def mock_prompt(*args, **kwargs):
            with lock:
                r = responses[idx[0] % len(responses)]
                idx[0] += 1
            return r

        with patch('novelty_eval.judge.prompt_openai_client', mock_prompt):
            return await evaluate_idea_pointwise(
                'test', 'idea text', 'dummy', semaphore,
                {'novelty': 'criterion'}, effort='none', mec_k=mec_k,
            )

    async def test_all_novel_returns_1(self):
        result = await self._call(['{"novelty": 1}'] * 3, mec_k=3)
        self.assertEqual(result, 1)

    async def test_all_not_novel_returns_0(self):
        result = await self._call(['{"novelty": 0}'] * 3, mec_k=3)
        self.assertEqual(result, 0)

    async def test_majority_novel(self):
        # 2 novel + 1 not novel → majority vote → 1
        result = await self._call(['{"novelty": 1}', '{"novelty": 1}', '{"novelty": 0}'], mec_k=3)
        self.assertEqual(result, 1)

    async def test_majority_not_novel(self):
        # 1 novel + 2 not novel → majority vote → 0
        result = await self._call(['{"novelty": 0}', '{"novelty": 0}', '{"novelty": 1}'], mec_k=3)
        self.assertEqual(result, 0)

    async def test_tie_defaults_to_not_novel(self):
        # sum=1, len=2 → 1 > 1.0 is False → 0
        result = await self._call(['{"novelty": 1}', '{"novelty": 0}'], mec_k=2)
        self.assertEqual(result, 0)

    async def test_partial_failure_aggregates_valid_votes(self):
        # One failure (None) + two "novel" → valid_votes=[1,1] → majority=1
        result = await self._call([None, '{"novelty": 1}', '{"novelty": 1}'], mec_k=3)
        self.assertEqual(result, 1)

    async def test_all_fail_returns_none(self):
        result = await self._call([None] * 3, mec_k=3)
        self.assertIsNone(result)

    async def test_mec_k_1_matches_single_call_behavior(self):
        # mec_k=1 should behave identically to the original single-call logic
        result = await self._call(['{"novelty": 1}'], mec_k=1)
        self.assertEqual(result, 1)

        result = await self._call(['{"novelty": 0}'], mec_k=1)
        self.assertEqual(result, 0)


class TestRelatedWorkConsistencyAcrossPaths(unittest.IsolatedAsyncioTestCase):
    """Verifies that all three eval paths (ranking, pairwise, pointwise) produce related_work
    via select_and_format_candidates — the same function used by the cache-write path —
    including the per-query quota selection that was missing before the refactor."""

    def _make_multi_query_candidates(self):
        """4 candidates across 2 queries: q1 has high scores, q2 has low scores.
        With top_k=2, pure relevance sort would take both from q1.
        Per-query quota must yield 1 from each query."""
        return [
            {'title': 'Q1-Best',   'url': 'http://q1b', 'abstract': 'q1 best abstract',
             'year': 2020, 'source_query': 'q1',
             'relevance_judgement': {'relevance_score': 0.9},
             'corpus_id': 'q1-cid-0', 'paperId': 'q1-pid-0'},
            {'title': 'Q1-Second', 'url': 'http://q1s', 'abstract': 'q1 second abstract',
             'year': 2020, 'source_query': 'q1',
             'relevance_judgement': {'relevance_score': 0.7},
             'corpus_id': 'q1-cid-1', 'paperId': 'q1-pid-1'},
            {'title': 'Q2-Best',   'url': 'http://q2b', 'abstract': 'q2 best abstract',
             'year': 2020, 'source_query': 'q2',
             'relevance_judgement': {'relevance_score': 0.4},
             'corpus_id': 'q2-cid-0', 'paperId': 'q2-pid-0'},
            {'title': 'Q2-Second', 'url': 'http://q2s', 'abstract': 'q2 second abstract',
             'year': 2020, 'source_query': 'q2',
             'relevance_judgement': {'relevance_score': 0.2},
             'corpus_id': 'q2-cid-1', 'paperId': 'q2-pid-1'},
        ]

    async def _capture_pointwise(self, candidates, top_k):
        output_dir = tempfile.mkdtemp()
        try:
            cache = {'0': {'candidates': candidates, 'related_work': '', 'text': 'idea'}}
            inputs = {'0': {'idea': 'test idea', 'label': 'POSITIVE'}}
            captured = {}
            async def mock_pw(problem_name, *args, related_work='', **kwargs):
                captured['rw'] = related_work
                return 1
            with patch('novelty_eval.run_benchmark.evaluate_idea_pointwise', side_effect=mock_pw):
                await run_pointwise_experiment(
                    n_runs=1, llm_engine='dummy', output_dir=output_dir, inputs=inputs,
                    max_workers=1, timestamp='test', retrieval_cache=cache,
                    retrieval_cache_path='cache.json', effort='none', top_k_candidates=top_k,
                )
            return captured.get('rw', '')
        finally:
            shutil.rmtree(output_dir)

    async def _capture_pairwise(self, candidates, top_k):
        output_dir = tempfile.mkdtemp()
        try:
            cache = {'p1': {
                '1': {'candidates': candidates, 'related_work': '', 'text': 'idea 1', 'title': '1'},
                '2': {'candidates': candidates, 'related_work': '', 'text': 'idea 2', 'title': '2'},
            }}
            inputs = {'p1': {'ideas': {'1': 'idea 1', '2': 'idea 2'}, 'expected_winners': ['1']}}
            captured = {}
            async def mock_eval(enriched_ideas, *args, **kwargs):
                captured.update(enriched_ideas)
                return {'elo_scores': [1200, 1000], 'elo_selected': [True, False],
                        'ideas': ['1', '2'], 'dimensional_elo_scores': {'overall': [1200, 1000]},
                        'comparisons': [{'idea_0': '1', 'idea_1': '2', 'winner': 0}]}
            with patch('novelty_eval.run_benchmark.evaluate_ideas', side_effect=mock_eval):
                await run_pairwise_experiment(
                    n_runs=1, llm_engine='dummy', output_dir=output_dir, inputs=inputs,
                    max_workers=1, timestamp='test', retrieval_cache=cache,
                    retrieval_cache_path='cache.json', use_batch_api=False, effort='none',
                    top_k_candidates=top_k,
                )
            return captured.get('1', {}).get('related_work', '')
        finally:
            shutil.rmtree(output_dir)

    async def _capture_ranking(self, candidates, top_k):
        output_dir = tempfile.mkdtemp()
        try:
            cache = {'p1': {
                '1': {'candidates': candidates, 'related_work': '', 'text': 'idea 1', 'title': '1'},
                '2': {'candidates': candidates, 'related_work': '', 'text': 'idea 2', 'title': '2'},
            }}
            inputs = {'p1': {'ideas': {'1': 'idea 1', '2': 'idea 2'}, 'expected_winners': ['1']}}
            captured = {}
            async def mock_eval(enriched_ideas, *args, **kwargs):
                captured.update(enriched_ideas)
                return {'elo_scores': [1200, 1000], 'elo_selected': [True, False],
                        'ideas': ['1', '2'], 'dimensional_elo_scores': {'overall': [1200, 1000]},
                        'comparisons': [{'idea_0': '1', 'idea_1': '2', 'winner': 0}]}
            with patch('novelty_eval.run_benchmark.evaluate_ideas', side_effect=mock_eval):
                await run_ranking_experiment(
                    n_runs=1, llm_engine='dummy', output_dir=output_dir, inputs=inputs,
                    swiss_tournament=False, max_workers=1, timestamp='test', retrieval_cache=cache,
                    retrieval_cache_path='cache.json', use_batch_api=False, effort='none',
                    top_k_candidates=top_k,
                )
            return captured.get('1', {}).get('related_work', '')
        finally:
            shutil.rmtree(output_dir)

    async def test_all_eval_paths_match_select_and_format_candidates(self):
        """The related_work string passed to the LLM by all three eval paths must equal
        select_and_format_candidates(candidates, top_k) — the same function used at cache-write time."""
        candidates = self._make_multi_query_candidates()
        top_k = 2
        expected = select_and_format_candidates(candidates, top_k=top_k)

        pointwise_rw = await self._capture_pointwise(candidates, top_k)
        pairwise_rw  = await self._capture_pairwise(candidates, top_k)
        ranking_rw   = await self._capture_ranking(candidates, top_k)

        self.assertEqual(pointwise_rw, expected, "Pointwise path diverges from select_and_format_candidates")
        self.assertEqual(pairwise_rw,  expected, "Pairwise path diverges from select_and_format_candidates")
        self.assertEqual(ranking_rw,   expected, "Ranking path diverges from select_and_format_candidates")

    async def test_per_query_quota_preserved_not_pure_relevance_sort(self):
        """Regression test: before the fix, all 3 paths used sort+slice, which takes both
        candidates from q1 (higher scores) and omits q2 entirely. Per-query quota must
        include one candidate from each query."""
        candidates = self._make_multi_query_candidates()
        # top_k=2, 2 queries → quota=1 per query → must get Q1-Best AND Q2-Best

        pointwise_rw = await self._capture_pointwise(candidates, top_k=2)
        pairwise_rw  = await self._capture_pairwise(candidates, top_k=2)
        ranking_rw   = await self._capture_ranking(candidates, top_k=2)

        for path, rw in [("pointwise", pointwise_rw), ("pairwise", pairwise_rw), ("ranking", ranking_rw)]:
            with self.subTest(path=path):
                self.assertIn("Q1-Best",   rw, f"{path}: top q1 candidate missing")
                self.assertIn("Q2-Best",   rw, f"{path}: top q2 candidate missing — pure sort+slice regression")
                self.assertNotIn("Q1-Second", rw, f"{path}: second q1 candidate should be excluded")
                self.assertNotIn("Q2-Second", rw, f"{path}: second q2 candidate should be excluded")


if __name__ == '__main__':
    unittest.main()
