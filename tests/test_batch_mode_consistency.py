import asyncio
import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add src to path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from novelty_eval.run_benchmark import (
    run_pairwise_experiment,
    run_pointwise_experiment,
    ExperimentStats
)
from tests.utils import generate_mock_batch_jsonl

class TestBatchModeConsistency(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_dir = Path("temp_test_batch")
        self.test_dir.mkdir(exist_ok=True)
        self.inputs = {
            "p1": {
                "expected_winners": ["1"],
                "ideas": {"1": "idea1 content", "2": "idea2 content"},
                "metadata": {"1": {"title": "Title 1"}, "2": {"title": "Title 2"}},
                "topic": "Topic",
                "label": 1 # Required for pointwise
            }
        }
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch("novelty_eval.judge.save_json_artifact")
    @patch("novelty_eval.judge.submit_batch_job")
    async def test_pointwise_consistency(self, mock_submit, mock_save):
        """Test that pointwise batch mode generation works without crashing."""
        # Mock evaluate_idea_pointwise to return a batch request
        mock_request = {"custom_id": "ptw|p1|0", "method": "POST"}
        with patch("novelty_eval.run_benchmark.evaluate_idea_pointwise", AsyncMock(return_value=[mock_request])):
            stats = await run_pointwise_experiment(
                n_runs=1,
                llm_engine="gpt-4o",
                output_dir=str(self.test_dir),
                inputs=self.inputs,
                max_workers=10,
                timestamp="test",
                retrieval_cache={},
                retrieval_cache_path=None,
                include_topic=False,
                effort="none",
                use_batch_api=True
            )
            
        # In batch mode, run_pointwise_experiment returns empty stats and calls submit_batch_job
        self.assertIsInstance(stats, ExperimentStats)
        mock_submit.assert_called_once()
        args, _ = mock_submit.call_args
        self.assertEqual(args[0], [mock_request])

    @patch("novelty_eval.judge.save_json_artifact")
    @patch("novelty_eval.judge.submit_batch_job")
    async def test_pairwise_consistency(self, mock_submit, mock_save):
        """Test that pairwise batch mode generation works without crashing."""
        mock_request = {"custom_id": "cmp|p1|1|2|0", "method": "POST"}
        with patch("novelty_eval.run_benchmark.evaluate_ideas", AsyncMock(return_value=[mock_request])):
            stats = await run_pairwise_experiment(
                n_runs=1,
                llm_engine="gpt-4o",
                output_dir=str(self.test_dir),
                inputs=self.inputs,
                max_workers=10,
                timestamp="test",
                retrieval_cache={},
                retrieval_cache_path=None,
                include_topic=False,
                use_batch_api=True,
                effort="none"
            )
            
        self.assertIsInstance(stats, ExperimentStats)
        mock_submit.assert_called_once()
        args, _ = mock_submit.call_args
        self.assertEqual(args[0], [mock_request])

    @patch("novelty_eval.judge.save_json_artifact")
    @patch("novelty_eval.judge.submit_batch_job")
    async def test_ranking_consistency(self, mock_submit, mock_save):
        """Test that ranking batch mode generation works without crashing."""
        from novelty_eval.run_benchmark import run_ranking_experiment
        
        mock_request = {"custom_id": "cmp|p1|1|2|0", "method": "POST"}
        with patch("novelty_eval.run_benchmark.evaluate_ideas", AsyncMock(return_value=[mock_request])):
            stats = await run_ranking_experiment(
                n_runs=1,
                llm_engine="gpt-4o",
                output_dir=str(self.test_dir),
                inputs=self.inputs,
                swiss_tournament=False, # Use RR
                max_workers=10,
                timestamp="test",
                retrieval_cache={},
                retrieval_cache_path=None,
                include_topic=False,
                ranking_modes=["rr"],
                use_batch_api=True,
                effort="none"
            )
            
        self.assertIsInstance(stats, ExperimentStats)
        mock_submit.assert_called_once()
        args, _ = mock_submit.call_args
        self.assertEqual(args[0], [mock_request])

    def test_batch_retrieval_aggregation(self):
        """Test that retrieve_batch_results correctly aggregates multiple votes via majority."""
        from novelty_eval.retrieve_batch_results import process_batch_results
        
        batch_output_file = self.test_dir / "batch_results.jsonl"
        # Simulate 3 votes for instance 'p1': two 1s, one 0
        responses = [
            {"custom_id": "ptw|p1|0", "text": '{"novelty": 1}'},
            {"custom_id": "ptw|p1|1", "text": '{"novelty": 1}'},
            {"custom_id": "ptw|p1|2", "text": '{"novelty": 0}'},
        ]
        generate_mock_batch_jsonl(batch_output_file, "openai", responses)
        
        results = process_batch_results(batch_output_file.read_text())
        
        # p1 should be 1 (majority of [1, 1, 0])
        self.assertEqual(results["p1"]["prediction"], 1)

    def test_batch_cost_tracking(self):
        """Test that process_batch_results records cost in GLOBAL_COST_TRACKER with 50% discount."""
        from novelty_eval.retrieve_batch_results import process_batch_results
        from cost_tracker import GLOBAL_COST_TRACKER, MODEL_PRICING
        
        GLOBAL_COST_TRACKER.reset()
        self.assertEqual(GLOBAL_COST_TRACKER.total_cost_usd(), 0.0)
        
        batch_output_file = self.test_dir / "batch_results_cost.jsonl"
        # Mock OpenAI response with usage
        model = "gpt-4o"
        prompt_tokens = 1000
        completion_tokens = 50
        entry = {
            "custom_id": "ptw|p1|0",
            "response": {
                "status_code": 200,
                "body": {
                    "model": model,
                    "choices": [{"message": {"content": '{"novelty": 1}'}}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
            }
        }
        batch_output_file.write_text(json.dumps(entry) + "\n")
        
        process_batch_results(batch_output_file.read_text())
        
        # Check if cost was recorded with 50% discount
        pricing = MODEL_PRICING[model]
        expected_full_cost = (prompt_tokens / 1_000_000 * pricing["input"]) + (completion_tokens / 1_000_000 * pricing["output"])
        expected_batch_cost = expected_full_cost * 0.5
        
        actual_cost = GLOBAL_COST_TRACKER.total_cost_usd()
        self.assertAlmostEqual(actual_cost, expected_batch_cost, places=6)

    def test_cost_tracker_merge_and_batch_discount(self):
        """Test that CostTracker correctly applies batch discount and merges reports."""
        from cost_tracker import CostTracker, MODEL_PRICING
        tracker = CostTracker()
        
        model = "gpt-4o"
        pricing = MODEL_PRICING[model]
        prompt_tokens = 1000000 # 1M tokens
        output_tokens = 0
        
        # 1. Standard record
        tracker.record(model, prompt_tokens, output_tokens, is_batch=False)
        self.assertEqual(tracker.total_cost_usd(), pricing["input"])
        
        # 2. Batch record (50% discount)
        tracker.reset()
        tracker.record(model, prompt_tokens, output_tokens, is_batch=True)
        self.assertEqual(tracker.total_cost_usd(), pricing["input"] * 0.5)
        
        # 3. Merge report
        report = tracker.get_report()
        new_tracker = CostTracker()
        new_tracker.merge_report(report)
        self.assertEqual(new_tracker.total_cost_usd(), pricing["input"] * 0.5)
        
        # Verify it can be merged multiple times
        new_tracker.merge_report(report)
        self.assertEqual(new_tracker.total_cost_usd(), pricing["input"]) # 0.5 + 0.5

if __name__ == "__main__":
    unittest.main()
