import unittest
import os
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from utils import extract_thinking_process
from novelty_eval.comparison_logger import (
    write_pointwise_to_file,
    write_comparison_to_file,
    _append_batch_pointwise_entry,
    _append_batch_comparison_entry,
    write_pointwise_logs_from_batch_results,
    write_comparison_logs_from_batch_results
)
from novelty_eval.retrieve_batch_results import process_batch_results
from novelty_eval.md_report_writer import (
    extract_pointwise_reasonings,
    extract_all_comparison_reasonings
)

class TestReasoningExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.run_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_utils_extract_thinking_process(self):
        """Verify basic extraction from <thinking> blocks."""
        text = "<thinking>I should pick A.</thinking>\n```json\n{\"choice\": \"A\"}\n```"
        self.assertEqual(extract_thinking_process(text), "I should pick A.")
        
        # Multiline and whitespace
        text = "<thinking>\n  Line 1\n  Line 2\n</thinking>"
        self.assertEqual(extract_thinking_process(text), "Line 1\n  Line 2")
        
        # Missing block
        self.assertIsNone(extract_thinking_process("Just some text"))

    def test_pointwise_logging_sequential(self):
        """Verify sequential pointwise logging writes reasoning to .txt."""
        instance_id = "inst_1"
        thinking = "Pointwise thinking"
        choice = {"novelty": 1}
        
        write_pointwise_to_file(
            self.test_dir, instance_id, "idea text", "related work",
            thinking, choice, call_index=0
        )
        
        # Verify .txt file exists and contains thinking
        txt_path = self.run_path / "pointwise" / f"{instance_id}.txt"
        self.assertTrue(txt_path.exists())
        content = txt_path.read_text()
        self.assertIn("Thinking Process:", content)
        self.assertIn(thinking, content)

    def test_pointwise_logging_batch(self):
        """Verify batch pointwise logging writes reasoning to .txt."""
        instance_id = "inst_batch"
        thinking = "Batch pointwise thinking"
        choice = {"novelty": 0}
        
        _append_batch_pointwise_entry(self.test_dir, instance_id, thinking, choice, call_index="1")
        
        txt_path = self.run_path / "pointwise" / f"{instance_id}.txt"
        content = txt_path.read_text()
        self.assertIn("Pointwise evaluation of instance inst_batch [call 1]", content)
        self.assertIn(thinking, content)

    def test_comparison_logging_sequential(self):
        """Verify sequential pairwise logging writes reasoning to .txt."""
        problem_name = "prob_1"
        thinking = "Pairwise thinking"
        choice = {"novelty": 0}
        
        write_comparison_to_file(
            self.test_dir, problem_name, 0, 1, "idea0", "idea1",
            thinking, choice
        )
        
        txt_path = self.run_path / "comparisons" / f"{problem_name}.txt"
        self.assertTrue(txt_path.exists())
        content = txt_path.read_text()
        self.assertIn(thinking, content)

    def test_comparison_logging_batch(self):
        """Verify batch pairwise logging writes reasoning to .txt."""
        problem_name = "prob_batch"
        thinking = "Batch pairwise thinking"
        choice = {"novelty": 1}
        
        _append_batch_comparison_entry(self.test_dir, problem_name, "0", "1", thinking, choice)
        
        txt_path = self.run_path / "comparisons" / f"{problem_name}.txt"
        content = txt_path.read_text()
        self.assertIn("Comparison between Idea 0 [idea0] and Idea 1 [idea1]", content)
        self.assertIn(thinking, content)

    def test_batch_process_results_extracts_reasoning(self):
        """Verify retrieve_batch_results.process_batch_results extracts and logs reasoning."""
        # Mock OpenAI batch output line
        thinking_text = "Reasoning in batch"
        content_with_thinking = f"<thinking>{thinking_text}</thinking>\n```json\n{{\"novelty\": 1}}\n```"
        
        batch_line = {
            "custom_id": "ptw|inst_abc|0",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": content_with_thinking}}],
                    "model": "gpt-4o",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                }
            }
        }
        jsonl_content = json.dumps(batch_line)
        
        # Call process_batch_results (it should call loggers internally now)
        process_batch_results(jsonl_content, output_dir=self.test_dir)
        
        # Verify the pointwise log was created with thinking
        txt_path = self.run_path / "pointwise" / "inst_abc.txt"
        self.assertTrue(txt_path.exists())
        self.assertIn(thinking_text, txt_path.read_text())

    def test_md_report_writer_extracts_reasoning(self):
        """Verify md_report_writer can read back the reasoning from .txt files."""
        instance_id = "inst_readback"
        thinking = "Thinking to be read back"
        choice = {"novelty": 1}
        
        # 1. Write the log
        _append_batch_pointwise_entry(self.test_dir, instance_id, thinking, choice, call_index="0")
        
        # 2. Extract using md_report_writer's logic
        entries = extract_pointwise_reasonings(self.test_dir, instance_id)
        
        self.assertEqual(len(entries), 1)
        # The actual implementation includes headers like "Thinking Process:" in the text field
        self.assertIn(thinking, entries[0]["text"])
        self.assertIn("Thinking Process:", entries[0]["text"])
        self.assertEqual(entries[0]["choice"], choice)

    def test_reprocess_logs_from_jsonl(self):
        """Verify write_pointwise_logs_from_batch_results re-populates logs."""
        thinking_text = "Reprocessed reasoning"
        jsonl_path = self.run_path / "batch_results.jsonl"
        batch_line = {
            "custom_id": "ptw|inst_repro|2",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": f"<thinking>{thinking_text}</thinking>{{}}"}}]
                }
            }
        }
        jsonl_path.write_text(json.dumps(batch_line))
        
        n = write_pointwise_logs_from_batch_results(str(jsonl_path), self.test_dir)
        self.assertEqual(n, 1)
        
        txt_path = self.run_path / "pointwise" / "inst_repro.txt"
        self.assertIn("[call 2]", txt_path.read_text())
        self.assertIn(thinking_text, txt_path.read_text())

    def test_reprocess_anthropic_logs_from_jsonl(self):
        """Verify write_pointwise_logs_from_batch_results supports Anthropic format."""
        thinking_text = "Anthropic reprocessed reasoning"
        jsonl_path = self.run_path / "anthropic_batch.jsonl"
        batch_line = {
            "custom_id": "ptw|inst_anthro|1",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": f"<thinking>{thinking_text}</thinking>{{}}"}]
                }
            }
        }
        jsonl_path.write_text(json.dumps(batch_line))
        
        n = write_pointwise_logs_from_batch_results(str(jsonl_path), self.test_dir)
        self.assertEqual(n, 1)
        
        txt_path = self.run_path / "pointwise" / "inst_anthro.txt"
        self.assertIn(thinking_text, txt_path.read_text())

if __name__ == "__main__":
    unittest.main()
