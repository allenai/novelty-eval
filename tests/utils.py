import json
import os
import shutil
import yaml
from pathlib import Path

def create_mock_artifact_dir(path: Path, mode: str, scores: dict = None, metrics: dict = None, model: str = "gpt-4o"):
    """Scaffold a temporary ablation artifact directory."""
    path.mkdir(parents=True, exist_ok=True)
    
    if scores:
        # Create run_pairwise_0/scores.json or similar
        run_dir = path / f"run_{mode}_0"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "scores.json", "w") as f:
            json.dump(scores, f)
            
    # accuracy_report.txt
    report_content = f"Test Mode: {mode.capitalize()}\n"
    report_content += f"Test Instances Path: test_instances.yaml\n"
    report_content += "Number of Test Instances Processed: 1\n\n"
    
    if mode == "pairwise":
        report_content += "Pairwise\n--------\n"
        report_content += f"Mean LLM Pairwise Accuracy (with ties): {metrics.get('Accuracy', 0.0)} (support=1.0)\n"
        report_content += f"Mean LLM Pairwise Accuracy (without ties): {metrics.get('Accuracy', 0.0)} (support=1.0)\n"
        report_content += "Mean Number of Ties: 0.0\n"
    elif mode == "pointwise":
        report_content += "Pointwise\n---------\n"
        report_content += f"Mean Accuracy: {metrics.get('Accuracy', 0.0)}\n"
        report_content += f"Mean F1 (macro): {metrics.get('Accuracy', 0.0)}\n"
        report_content += f"Mean F1 (POSITIVE): {metrics.get('Accuracy', 0.0)}\n"
        report_content += f"Mean F1 (NEGATIVE): {metrics.get('Accuracy', 0.0)}\n"
    elif mode == "ranking":
        report_content += "3. Swiss Tournament\n-------------------\n"
        report_content += f"NDCG: {metrics.get('Accuracy', 0.0)}\n"
        report_content += f"MRR: {metrics.get('Accuracy', 0.0)}\n"

    report_content += "\nSummary\n-------\n"
    if metrics:
        for k, v in metrics.items():
            report_content += f"{k}: {v}\n"
            
    with open(path / "accuracy_report.txt", "w") as f:
        f.write(report_content)
        
    # cost_report.json
    with open(path / "cost_report.json", "w") as f:
        json.dump({"total_cost_usd": 0.5, "model_costs": {}}, f)

    # debug_info.md
    debug_content = f"""
# Debug Info
## Command Line Arguments
```json
{{
    "llm_engine": "{model}",
    "test_mode": "{mode}"
}}
```
"""
    with open(path / f"debug_{mode}.md", "w") as f:
        f.write(debug_content)

def generate_mock_batch_jsonl(path: Path, provider: str, responses: list[dict]):
    """Generate mock JSONL batch response file."""
    with open(path, "w") as f:
        for resp in responses:
            if provider == "anthropic":
                # Mock Anthropic Message Batch format
                entry = {
                    "custom_id": resp["custom_id"],
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [{"type": "text", "text": resp["text"]}]
                        }
                    }
                }
                if "error" in resp:
                    entry["result"] = {"type": "errored", "error": {"type": resp["error"]}}
            else:
                # Mock OpenAI Batch format
                entry = {
                    "custom_id": resp["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"content": resp["text"]}}]
                        }
                    }
                }
                if "error" in resp:
                    entry["response"] = {"status_code": 400, "error": {"code": resp["error"]}}
            f.write(json.dumps(entry) + "\n")

def create_mock_llm_failures(path: Path, failures: list[dict]):
    """Create a mock llm_failures_debug.md file."""
    content = "# LLM Failures Debug\n\n"
    for f in failures:
        content += f"### {f['index']} · Problem `{f['problem']}`\n"
        content += "| Metric | Value |\n"
        content += "| --- | --- |\n"
        content += f"| Error code | `{f['error_code']}` |\n"
        content += f"| Error message | {f['error_message']} |\n\n"
        
    with open(path / "llm_failures_debug.md", "w") as f:
        f.write(content)
