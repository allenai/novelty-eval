import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import save_json_artifact, extract_json_choice, extract_thinking_process
from batch_api import check_batch_status, retrieve_raw_batch_content  # noqa: F401 — check_batch_status re-exported
try:
    from cost_tracker import GLOBAL_COST_TRACKER, cost_stage
except ImportError:
    from src.cost_tracker import GLOBAL_COST_TRACKER, cost_stage
from novelty_eval.scoring import aggregate_unidirectional_comparisons
from novelty_eval.comparison_logger import (
    _append_batch_comparison_entry,
    _append_batch_pointwise_entry,
    _parse_comparison_custom_id,
    _parse_pointwise_custom_id,
)
from logging_utils import setup_logger

LOGGER = setup_logger(output_dir=os.getenv("OUTPUT_DIR", "."))

# ---------------------------------------------------------------------------
# batch_info helpers
# ---------------------------------------------------------------------------

def _read_batch_info(batch_id: str, output_dir: str) -> dict:
    """Return the batch_info_*.json dict for batch_id found under output_dir, or {}."""
    if output_dir and os.path.isdir(output_dir):
        for fn in os.listdir(output_dir):
            if fn.startswith("batch_info_") and fn.endswith(".json"):
                try:
                    with open(os.path.join(output_dir, fn)) as f:
                        info = json.load(f)
                    if info.get("batch_id") == batch_id:
                        return info
                except Exception:
                    pass
    return {}


def _detect_provider(batch_id: str, output_dir: str) -> str:
    return _read_batch_info(batch_id, output_dir).get("provider", "openai")


# ---------------------------------------------------------------------------
# Response parsing — format-aware
# ---------------------------------------------------------------------------

def _parse_single_response(custom_id: str, message_content: str,
                            comparisons: "defaultdict[str, list]",
                            pointwise: "defaultdict[str, list]",
                            usage: dict | None = None,
                            model: str | None = None,
                            output_dir: str | None = None) -> bool | None:
    """Route one batch response into comparisons or pointwise based on custom_id format.

    Returns:
        True   – successfully parsed and recorded.
        False  – recognised custom_id but JSON extraction failed (dropped, not logged).
        None   – unrecognised custom_id format.
    """
    if "__" in custom_id:
        parts = custom_id.split("__")
    else:
        parts = custom_id.split("|")

    # Record cost if usage and model are provided
    if usage and model and GLOBAL_COST_TRACKER:
        # Infer stage from custom_id
        stage = "unknown"
        if custom_id.startswith("cmp|") or custom_id.startswith("cmp__") or (len(parts) == 3 and not (custom_id.startswith("ptw|") or custom_id.startswith("ptw__"))):
            stage = "judge_pairwise"
        elif custom_id.startswith("ptw|") or custom_id.startswith("ptw__") or len(parts) == 1:
            stage = "judge_pointwise"

        with cost_stage(stage):
            # Record usage. Handles both OpenAI and Anthropic usage formats.
            # OpenAI: prompt_tokens, completion_tokens, cached_tokens (optional)
            # Anthropic: input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens
            
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            
            # OpenAI prompt caching details
            cached_input = 0
            prompt_details = usage.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                cached_input = prompt_details.get("cached_tokens", 0)
            
            # Anthropic prompt caching details
            cached_input = cached_input or usage.get("cache_read_input_tokens") or 0
            cache_creation = usage.get("cache_creation_input_tokens") or 0
            
            GLOBAL_COST_TRACKER.record(
                model=model,
                input_tokens=input_tokens - cached_input, # Subtract cached from base input
                output_tokens=output_tokens,
                cached_input_tokens=cached_input,
                cache_creation_tokens=cache_creation,
                is_batch=True
            )

    choice_json = extract_json_choice(message_content)
    thinking = extract_thinking_process(message_content)

    # 1) Try parsing as comparison
    cmp_parsed = _parse_comparison_custom_id(custom_id)
    if cmp_parsed:
        problem_name, idea_i, idea_j = cmp_parsed
        if choice_json is None:
            LOGGER.warning(
                f"Batch parse failure: JSON extraction returned None for comparison "
                f"{custom_id!r} — dropping from results (no comparison log entry written)."
            )
            return False
        comparisons[problem_name].append({
            "idea_i": idea_i,
            "idea_j": idea_j,
            "scores": choice_json,
        })
        if output_dir:
            _append_batch_comparison_entry(output_dir, problem_name, idea_i, idea_j, thinking, choice_json)
        return True

    # 2) Try parsing as pointwise
    ptw_parsed = _parse_pointwise_custom_id(custom_id)
    if ptw_parsed:
        instance_id, call_index = ptw_parsed
        if choice_json is None:
            LOGGER.warning(
                f"Batch parse failure: JSON extraction returned None for pointwise "
                f"{custom_id!r} — dropping from results (no pointwise log entry written)."
            )
            return False
        votes = [v for v in choice_json.values() if isinstance(v, int) and v in (0, 1)]
        if votes:
            pointwise[instance_id].append(1 if sum(votes) > len(votes) / 2 else 0)
        if output_dir:
            _append_batch_pointwise_entry(output_dir, instance_id, thinking, choice_json, call_index)
        return True

    LOGGER.warning(f"Unrecognised custom_id format ({len(parts)} parts): {custom_id}")
    return None


def _assemble_results(comparisons: "defaultdict[str, list]",
                      pointwise: "defaultdict[str, list]") -> dict:
    """Merge parsed comparisons and pointwise predictions into a single results dict."""
    if comparisons and pointwise:
        LOGGER.warning("Batch output contains both comparison and pointwise responses — "
                       "they will be merged, but this is unexpected.")
    results: dict = {}
    if comparisons:
        results.update(aggregate_unidirectional_comparisons(comparisons))
    if pointwise:
        # Pointwise: {instance_id: {"prediction": 0|1}}
        # Aggregate multiple votes per instance via majority vote.
        for iid, votes in pointwise.items():
            final_pred = 1 if sum(votes) > len(votes) / 2 else 0
            results[iid] = {"prediction": final_pred}
    return results


def retrieve_batch_results(batch_id: str, output_dir: str, provider: str | None = None):
    """Retrieve batch results, dispatching to the correct provider.

    provider is auto-detected from batch_info_*.json when not given.
    """
    if provider is None:
        provider = _detect_provider(batch_id, output_dir)

    content = retrieve_raw_batch_content(batch_id, output_dir, provider)
    if content is None:
        return None
    return process_batch_results(content, output_dir=output_dir)


def process_batch_results(content: str, output_dir: str | None = None) -> dict:
    """Parse JSONL batch output (OpenAI or Anthropic). Handles both comparison and pointwise formats."""
    comparisons: defaultdict = defaultdict(list)
    pointwise: defaultdict = defaultdict(list)
    batch_parse_failures = 0

    for line in content.splitlines():
        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            LOGGER.warning(f"Failed to parse JSON line: {e}")
            continue

        custom_id = item.get("custom_id")
        if not custom_id:
            continue

        message_content = ""
        usage = None
        model = None

        # 1) Detect Anthropic Format
        if "result" in item and isinstance(item["result"], dict):
            res_obj = item["result"]
            if res_obj.get("type") != "succeeded":
                LOGGER.warning(f"Request {custom_id} did not succeed: {res_obj.get('type')}")
                continue
            
            message = res_obj.get("message", {})
            model = message.get("model")
            usage = message.get("usage")
            # Anthropic usage might need conversion if it's still a dict
            
            for block in message.get("content", []):
                if block.get("type") == "text":
                    message_content = block.get("text", "")
                    break
        
        # 2) Detect OpenAI Format
        elif "response" in item and isinstance(item["response"], dict):
            response = item.get("response", {})
            body = response.get("body", {})
            model = body.get("model")
            usage = body.get("usage")
            choices = body.get("choices", [])
            if not choices:
                continue
            message_content = choices[0].get("message", {}).get("content", "")
        
        else:
            LOGGER.warning(f"Unrecognized batch result format for custom_id: {custom_id}")
            continue

        result = _parse_single_response(
            custom_id, message_content, comparisons, pointwise,
            usage=usage, model=model, output_dir=output_dir,
        )
        if result is False:
            batch_parse_failures += 1

    if batch_parse_failures:
        LOGGER.warning(
            f"Batch results: {batch_parse_failures} response(s) dropped due to JSON parse failures "
            f"(choice_json=None). Failure rate is underestimated relative to live runs unless "
            f"these are accounted for. See per-entry warnings above for details."
        )

    return _assemble_results(comparisons, pointwise)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and process batch API results (OpenAI or Anthropic)")
    parser.add_argument("--batch-id", type=str, help="The Batch ID to retrieve")
    parser.add_argument("--batch-info-file", type=str, help="Path to the batch info JSON file")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save results")
    parser.add_argument("--provider", type=str, default=None, choices=["openai", "anthropic"],
                        help="Provider (auto-detected from batch_info file if omitted)")

    args = parser.parse_args()

    batch_id = args.batch_id
    provider = args.provider
    if not batch_id and args.batch_info_file:
        if os.path.exists(args.batch_info_file):
            with open(args.batch_info_file, "r") as f:
                info = json.load(f)
                batch_id = info.get("batch_id")
                if provider is None:
                    provider = info.get("provider", "openai")
        else:
            LOGGER.error(f"Batch info file not found: {args.batch_info_file}")
            sys.exit(1)

    if not batch_id:
        LOGGER.error("Please provide either --batch-id or --batch-info-file")
        sys.exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    results = retrieve_batch_results(batch_id, args.output_dir, provider=provider)

    if results:
        save_json_artifact(args.output_dir, results, "scores_from_batch")
        LOGGER.info(f"Processed results saved to {args.output_dir}/scores_from_batch.json")
