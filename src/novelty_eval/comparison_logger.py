"""
comparison_logger.py — Writes per-comparison debug output to disk.
"""
import json
import os
from typing import Dict, List, Any


def format_candidates_readable(candidates: List[Dict]) -> str:
    if not candidates:
        return ""
    lines = []
    for idx, c in enumerate(candidates, 1):
        title = c.get('title', 'Unknown Title')
        year = c.get('year', 'N/A')
        url = c.get('url', 'N/A')
        abstract = c.get('abstract') or c.get('text', 'No abstract available')
        lines.append(f"{idx}. {title} ({year})\n   URL: {url}\n   Abstract: {abstract}\n")
    return "\n".join(lines)


def _idea_text(idea: Any) -> str:
    return idea['text'] if isinstance(idea, dict) else idea


def _idea_related_work(idea: Any) -> str:
    return idea.get('related_work', '') if isinstance(idea, dict) else ''


def _idea_candidates(idea: Any) -> List[Dict]:
    return idea.get('candidates', []) if isinstance(idea, dict) else []


def write_comparison_to_file(
    output_path: str,
    problem_name: str,
    i, j,
    idea0: Any,
    idea1: Any,
    thinking_process: str,
    json_response: Any,
) -> None:
    """Append a single pairwise comparison record to <output_path>/comparisons/<problem_name>.txt."""
    comparison_dir = os.path.join(output_path, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    file_path = os.path.join(comparison_dir, f"{problem_name}.txt")

    idea0_text = _idea_text(idea0)
    idea0_rw = _idea_related_work(idea0)
    idea0_candidates = _idea_candidates(idea0)

    idea1_text = _idea_text(idea1)
    idea1_rw = _idea_related_work(idea1)
    idea1_candidates = _idea_candidates(idea1)

    sep = "=" * 80
    lines = [
        sep,
        f"Comparison between Idea {i} [idea0] and Idea {j} [idea1]",
        sep,
        "",
        f"Idea {i} [idea0]:",
        idea0_text,
    ]

    if idea0_candidates:
        lines += [f"Related Work for Idea {i} [idea0]:", format_candidates_readable(idea0_candidates)]
    elif idea0_rw:
        lines += [f"Related Work for Idea {i} [idea0]:", idea0_rw]
    lines.append("")

    lines += [f"Idea {j} [idea1]:", idea1_text]
    if idea1_candidates:
        lines += [f"Related Work for Idea {j} [idea1]:", format_candidates_readable(idea1_candidates)]
    elif idea1_rw:
        lines += [f"Related Work for Idea {j} [idea1]:", idea1_rw]
    lines.append("")

    lines += [
        "Thinking Process:",
        thinking_process,
        "",
        "Choice:",
        json.dumps(json_response, indent=2) if json_response else "None",
        "",
    ]

    with open(file_path, "a") as f:
        f.write("\n".join(line if line is not None else "" for line in lines) + "\n")


def _append_batch_comparison_entry(
    run_path: str,
    problem_name: str,
    idea_i: str,
    idea_j: str,
    thinking_process: str,
    choice_json: Any,
) -> None:
    """Append one batch comparison entry to comparisons/<problem_name>.txt (no idea texts)."""
    comparison_dir = os.path.join(run_path, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    file_path = os.path.join(comparison_dir, f"{problem_name}.txt")

    sep = "=" * 80
    lines = [
        sep,
        f"Comparison between Idea {idea_i} [idea0] and Idea {idea_j} [idea1]",
        sep,
        "",
        "Thinking Process:",
        thinking_process or "(no thinking recorded)",
        "",
        "Choice:",
        json.dumps(choice_json, indent=2) if choice_json else "None",
        "",
    ]
    with open(file_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _append_batch_pointwise_entry(
    run_path: str,
    instance_id: str,
    thinking_process: str,
    choice_json: Any,
    call_index: str = "0",
) -> None:
    """Append one batch pointwise entry to pointwise/<instance_id>.txt (no idea text)."""
    pointwise_dir = os.path.join(run_path, "pointwise")
    os.makedirs(pointwise_dir, exist_ok=True)
    file_path = os.path.join(pointwise_dir, f"{instance_id}.txt")

    sep = "=" * 80
    lines = [
        sep,
        f"Pointwise evaluation of instance {instance_id} [call {call_index}]",
        sep,
        "",
        "Thinking Process:",
        thinking_process or "(no thinking recorded)",
        "",
        "Choice:",
        json.dumps(choice_json, indent=2) if choice_json else "None",
        "",
    ]
    with open(file_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _parse_comparison_custom_id(custom_id: str):
    """Return (problem_name, idea_i, idea_j) from a cmp custom_id, or None if not a comparison."""
    if "__" in custom_id:
        parts = custom_id.split("__")
    else:
        parts = custom_id.split("|")

    if len(parts) >= 4 and parts[0] == "cmp":
        return parts[1], parts[2], parts[3]
    if len(parts) == 3 and not (custom_id.startswith("ptw|") or custom_id.startswith("ptw__")):
        return parts[0], parts[1], parts[2]
    if len(parts) == 1 and parts[0].startswith("cmp-"):
        sub = parts[0].split("-")
        if len(sub) >= 5:
            return "-".join(sub[1:-3]), sub[-3], sub[-2]
    return None


def _parse_pointwise_custom_id(custom_id: str):
    """Return (instance_id, call_index) from a ptw custom_id, or None if not pointwise."""
    if "__" in custom_id:
        parts = custom_id.split("__")
    else:
        parts = custom_id.split("|")

    if len(parts) >= 3 and parts[0] == "ptw":
        return parts[1], parts[2]
    if len(parts) == 2 and parts[0] == "ptw":
         return parts[1], "0"
    if len(parts) == 1 and parts[0].startswith("ptw-"):
        sub = parts[0].split("-")
        if len(sub) >= 3:
            return "-".join(sub[1:-1]), sub[-1]
    if len(parts) == 1 and not (parts[0].startswith("cmp|") or parts[0].startswith("cmp__") or parts[0].startswith("cmp-")):
        # Legacy/Simple pointwise
        return parts[0], "0"
    return None


def _extract_content_from_batch_item(item: dict) -> str:
    """Extract message content from either OpenAI or Anthropic batch result formats."""
    # OpenAI format: response.body.choices
    choices = item.get("response", {}).get("body", {}).get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    
    # Anthropic format: result.message.content
    result = item.get("result", {})
    if result.get("type") == "succeeded" and "message" in result:
        content = result["message"].get("content", [])
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return ""


def write_comparison_logs_from_batch_results(jsonl_path: str, run_path: str) -> int:
    """Parse a batch results JSONL (OpenAI or Anthropic) and write comparison log files.

    Skips writing if the comparisons directory already exists and is non-empty.
    Returns the number of entries written.
    """
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils import extract_json_choice, extract_thinking_process

    comparisons_dir = os.path.join(run_path, "comparisons")
    if os.path.isdir(comparisons_dir) and any(os.scandir(comparisons_dir)):
        return 0  # already populated

    try:
        with open(jsonl_path) as f:
            lines = f.readlines()
    except Exception:
        return 0

    written = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        custom_id = item.get("custom_id", "")
        parsed = _parse_comparison_custom_id(custom_id)
        if parsed is None:
            continue
        problem_name, idea_i, idea_j = parsed

        content = _extract_content_from_batch_item(item)
        if not content:
            continue

        thinking = extract_thinking_process(content) or ""
        choice_json = extract_json_choice(content)

        _append_batch_comparison_entry(run_path, problem_name, idea_i, idea_j, thinking, choice_json)
        written += 1

    return written


def write_pointwise_logs_from_batch_results(jsonl_path: str, run_path: str) -> int:
    """Parse a batch results JSONL (OpenAI or Anthropic) and write pointwise log files.

    Skips writing if the pointwise directory already exists and is non-empty.
    Returns the number of entries written.
    """
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils import extract_json_choice, extract_thinking_process

    pointwise_dir = os.path.join(run_path, "pointwise")
    if os.path.isdir(pointwise_dir) and any(os.scandir(pointwise_dir)):
        return 0  # already populated

    try:
        with open(jsonl_path) as f:
            lines = f.readlines()
    except Exception:
        return 0

    written = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        custom_id = item.get("custom_id", "")
        parsed = _parse_pointwise_custom_id(custom_id)
        if parsed is None:
            continue
        instance_id, call_index = parsed

        content = _extract_content_from_batch_item(item)
        if not content:
            continue

        thinking = extract_thinking_process(content) or ""
        choice_json = extract_json_choice(content)

        _append_batch_pointwise_entry(run_path, instance_id, thinking, choice_json, call_index)
        written += 1

    return written


def write_pointwise_to_file(
    output_path: str,
    instance_id: str,
    idea_text: str,
    related_work: str,
    thinking_process: str,
    json_response: Any,
    call_index: int,
) -> None:
    """Append a single pointwise reasoning record to <output_path>/pointwise/<instance_id>.txt."""
    pointwise_dir = os.path.join(output_path, "pointwise")
    os.makedirs(pointwise_dir, exist_ok=True)
    file_path = os.path.join(pointwise_dir, f"{instance_id}.txt")

    sep = "=" * 80
    lines = [
        sep,
        f"Pointwise evaluation of instance {instance_id} [call {call_index}]",
        sep,
        "",
        f"Idea:",
        idea_text,
    ]

    if related_work:
        lines += ["", "Related Work:", related_work]

    lines += [
        "",
        "Thinking Process:",
        thinking_process,
        "",
        "Choice:",
        json.dumps(json_response, indent=2) if json_response else "None",
        "",
    ]

    with open(file_path, "a") as f:
        f.write("\n".join(line if line is not None else "" for line in lines) + "\n")
