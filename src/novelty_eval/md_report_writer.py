"""
Markdown debug report generation for accuracy_test experiments.

The main entry point is `save_md_debug_report`.
Internal helpers handle idea-detail rendering, paper/signal formatting, and comparison
reasoning — all shared across RR, Swiss, and Bi-Swiss tournament modes.
"""
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .experiment_stats import ExperimentStats
from .judge import EVALUATION_CRITERIA, EVALUATION_CRITERIA_RETRIEVAL


def _get_input(inputs: Dict[str, Any], problem_id) -> Optional[Dict[str, Any]]:
    """Look up a problem in inputs, tolerating str/int key-type mismatches from JSON round-trips."""
    val = inputs.get(problem_id)
    if val is None:
        try:
            val = inputs.get(int(problem_id))
        except (ValueError, TypeError):
            val = inputs.get(str(problem_id))
    return val


# ---------------------------------------------------------------------------
# Low-level MD formatting helpers
# ---------------------------------------------------------------------------

def format_retrieved_papers_for_md(papers: List[Dict[str, Any]]) -> str:
    if not papers:
        return ""
    lines = [
        "  - <details>",
        "    <summary><strong>Retrieved Related Work</strong></summary>\n",
    ]
    for paper in papers:
        title = paper.get("title", "Unknown Title")
        venue = paper.get("venue", "Unknown Venue")
        year = paper.get("year", "Unknown Year")
        url = paper.get("url", "")
        abstract = paper.get("abstract", "")
        link = f" [Link]({url})" if url else ""
        lines.append(f"    - **{title}** ({venue}, {year}){link}")
        if abstract:
            lines.append(f"      > {abstract}")
        lines.append("")
    lines.append("    </details>")
    return "\n".join(lines)


def format_signals_for_md(title: str, signals: List[str]) -> str:
    if not signals:
        return ""
    lines = [
        "  - <details>",
        f"    <summary><strong>{title}</strong></summary>\n",
    ]
    for sig in signals:
        lines.append(f"    - {sig}")
    lines += ["", "    </details>"]
    return "\n".join(lines)


def format_retrieval_queries_for_md(search_queries_dict: Dict[str, List[str]]) -> str:
    if not search_queries_dict:
        return ""
    lines = [
        "  - <details>",
        "    <summary><strong>Retrieval Queries</strong></summary>\n",
    ]
    for dimension, queries in search_queries_dict.items():
        lines.append(f"    **{dimension}:**")
        for q in queries:
            lines.append(f"    - {q}")
        lines.append("")
    lines.append("    </details>")
    return "\n".join(lines)


def format_candidates_confidence_for_md(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return ""
    lines = [
        "  - <details>",
        "    <summary><strong>Candidates Confidence Scores</strong></summary>\n",
    ]
    scores = []
    for i, c in enumerate(candidates, 1):
        title = c.get("title", "Unknown Title")
        score = (c.get("relevance_judgement") or {}).get("relevance_score", None)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        lines.append(f"    {i}. **{title}** — confidence: `{score_str}`")
        if isinstance(score, (int, float)):
            scores.append(score)
    lines.append("")
    if scores:
        avg = sum(scores) / len(scores)
        lines.append(f"    **Average confidence: `{avg:.4f}`** ({len(scores)} candidates)")
    else:
        lines.append("    **Average confidence: N/A** (no scores available)")
    lines += ["", "    </details>"]
    return "\n".join(lines)


def format_reasoning_for_md(reasoning: str) -> str:
    """Clean up raw comparison log text for Markdown display."""
    reasoning = re.sub(r"(Thinking Process:)", r"**\1**", reasoning)
    reasoning = re.sub(r"(Choice:)", r"**\1**", reasoning)

    lines = reasoning.split("\n")
    filtered_lines: List[str] = []
    idea_headers: List[str] = []
    skip = False

    for line in lines:
        s_line = line.strip()
        idea_match = re.match(r"^(Idea \d+ \[idea\d+\]):", s_line)
        if idea_match:
            idea_headers.append(idea_match.group(1))
            skip = True
            continue
        if skip:
            if (re.match(r"^Idea \d+ \[idea\d+\]:", s_line)
                    or "**Thinking Process:**" in s_line
                    or "**Choice:**" in s_line):
                skip = False
            else:
                continue
        filtered_lines.append(line)

    summary = f"**Compared ideas:** {', '.join(idea_headers)}"
    final_lines: List[str] = []
    inserted = False
    for line in filtered_lines:
        final_lines.append(line)
        if line.strip().startswith("Topic:") and not inserted:
            final_lines += ["", summary]
            inserted = True

    if not inserted and idea_headers:
        final_lines.insert(0, summary)

    return "\n".join(final_lines).strip()


# ---------------------------------------------------------------------------
# Log-file extraction helpers
# ---------------------------------------------------------------------------

def extract_comparison_reasoning(run_path: str, problem_id: str, idea1: str, idea2: str) -> str:
    comparison_file = os.path.join(run_path, "comparisons", f"{problem_id}.txt")
    if not os.path.exists(comparison_file):
        return "Comparison log not found."
    try:
        with open(comparison_file, "r") as fh:
            content = fh.read()
    except Exception as e:
        return f"Error reading comparison log: {e}"

    header1 = f"Comparison between Idea {idea1} [idea0] and Idea {idea2} [idea1]"
    header2 = f"Comparison between Idea {idea2} [idea0] and Idea {idea1} [idea1]"
    idx = content.find(header1)
    if idx == -1:
        idx = content.find(header2)
    if idx == -1:
        return "Direct comparison not found in logs."

    separator = "=" * 80
    sep_start = content.find(separator, idx)
    if sep_start == -1:
        return content[idx:].strip()
    content_start = sep_start + len(separator)
    next_sep = content.find(separator, content_start)
    return content[content_start:next_sep].strip() if next_sep != -1 else content[content_start:].strip()


def extract_all_comparison_reasonings(
    run_path: str, problem_id: str, idea_a: str, idea_b: str
) -> List[Dict[str, Any]]:
    """Return all per-call reasoning entries for a pair (both directions), in file order.

    Each entry is a dict with: idea0, idea1, winner (raw int or None), text (reasoning).
    """
    comparison_file = os.path.join(run_path, "comparisons", f"{problem_id}.txt")
    if not os.path.exists(comparison_file):
        return []
    try:
        with open(comparison_file, "r") as fh:
            content = fh.read()
    except Exception:
        return []

    separator = "=" * 80

    # Collect positions of all headers for both directions
    candidates = []
    for i0, i1 in [(idea_a, idea_b), (idea_b, idea_a)]:
        header = f"Comparison between Idea {i0} [idea0] and Idea {i1} [idea1]"
        start = 0
        while True:
            idx = content.find(header, start)
            if idx == -1:
                break
            candidates.append((idx, i0, i1))
            start = idx + 1

    candidates.sort(key=lambda x: x[0])

    entries = []
    for pos, i0, i1 in candidates:
        sep_after = content.find(separator, pos)
        if sep_after == -1:
            continue
        content_start = sep_after + len(separator)
        next_sep = content.find(separator, content_start)
        text = content[content_start:next_sep].strip() if next_sep != -1 else content[content_start:].strip()

        winner = None
        choice_idx = text.find("Choice:")
        if choice_idx != -1:
            choice_text = text[choice_idx + len("Choice:"):].strip()
            try:
                end = choice_text.find("\n\n")
                choice_json = json.loads(choice_text[:end] if end != -1 else choice_text)
                if isinstance(choice_json, dict):
                    winner = next(iter(choice_json.values()), None)
            except (json.JSONDecodeError, StopIteration):
                pass

        entries.append({"idea0": i0, "idea1": i1, "winner": winner, "text": text})

    return entries


def parse_candidates_text(text: str) -> List[Dict[str, Any]]:
    papers = []
    for chunk in re.split(r"(?:^|\n)\d+\.\s+", text.strip()):
        if not chunk.strip():
            continue
        lines = chunk.strip().split("\n")
        first_line = lines[0].strip()
        year, title = "Unknown Year", first_line
        m = re.search(r"\((.*?)\)$", first_line)
        if m:
            year = m.group(1)
            title = first_line[:m.start()].strip()
        url, abstract = "", ""
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("URL:"):
                url = line[4:].strip()
            elif line.startswith("Abstract:"):
                abstract = line[9:].strip()
            elif abstract:
                abstract += " " + line
        papers.append({"title": title, "year": year, "url": url, "abstract": abstract, "venue": "Unknown Venue"})
    return papers


def extract_retrieved_papers_from_log(run_path: str, problem_id: str, idea_key: str) -> List[Dict[str, Any]]:
    log_file = os.path.join(run_path, "comparisons", f"{problem_id}.txt")
    if not os.path.exists(log_file):
        return []
    try:
        with open(log_file, "r") as fh:
            content = fh.read()
    except Exception:
        return []
    escaped = re.escape(str(idea_key))
    pattern = re.compile(
        rf"Related Work for Idea {escaped} \[idea\d\]:\n(.*?)\n(?=\n(?:Idea |Thinking Process:|={80}))",
        re.DOTALL,
    )
    match = pattern.search(content)
    return parse_candidates_text(match.group(1)) if match else []


def parse_retrieval_debug_log(debug_file_path: str) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    """Parse a retrieval debug log file for search queries and candidate scores."""
    if not os.path.exists(debug_file_path):
        return {}, []
    try:
        with open(debug_file_path, "r") as fh:
            content = fh.read()
    except Exception:
        return {}, []

    search_queries_dict: Dict[str, List[str]] = {}
    qm = re.search(r"--- GENERATED SEARCH QUERIES ---\n(.*?)(?=--- SELECTED CANDIDATES ---)", content, re.DOTALL)
    if qm:
        current_dim = None
        for line in qm.group(1).split("\n"):
            line = line.strip()
            if line.startswith("Dimension: "):
                current_dim = line[len("Dimension: "):]
                search_queries_dict[current_dim] = []
            elif line.startswith("- ") and current_dim is not None:
                search_queries_dict[current_dim].append(line[2:])

    candidates: List[Dict[str, Any]] = []
    cm = re.search(r"--- SELECTED CANDIDATES ---\n(.*)", content, re.DOTALL)
    if cm:
        for block in re.split(r"(?=Candidate #\d+)", cm.group(1)):
            block = block.strip()
            if not block.startswith("Candidate #"):
                continue
            title_m = re.search(r"Title: (.+)", block)
            score_m = re.search(r"Score: (.+)", block)
            year_m = re.search(r"Year: (.+)", block)
            title = title_m.group(1).strip() if title_m else "Unknown Title"
            year = year_m.group(1).strip() if year_m else "N/A"
            score_str = score_m.group(1).strip() if score_m else "N/A"
            try:
                relevance_judgement = {"relevance_score": float(score_str)}
            except (ValueError, TypeError):
                relevance_judgement = {}
            candidates.append({"title": title, "year": year, "relevance_judgement": relevance_judgement})

    return search_queries_dict, candidates


def get_idea_retrieval_data(
    retrieval_cache: Dict[str, Any],
    problem_id: str,
    idea_key: str,
    retrieval_cache_path: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    """Return (search_queries_dict, candidates) from cache, falling back to the debug log file."""
    cached = retrieval_cache.get(str(problem_id), {}).get(str(idea_key), {})
    search_queries_dict = cached.get("search_queries_dict", {})
    candidates = cached.get("candidates", [])

    if not search_queries_dict and retrieval_cache_path:
        cache_dir = os.path.dirname(os.path.abspath(retrieval_cache_path))
        debug_file = os.path.join(cache_dir, "retrieval_debug", str(problem_id), f"{idea_key}_retrieval.txt")
        parsed_queries, parsed_candidates = parse_retrieval_debug_log(debug_file)
        if parsed_queries:
            search_queries_dict = parsed_queries
        if not candidates and parsed_candidates:
            candidates = parsed_candidates

    return search_queries_dict, candidates


# ---------------------------------------------------------------------------
# Per-idea block writer (shared by bad/good comparison sections)
# ---------------------------------------------------------------------------

def _write_idea_block(
    f,
    idea_key: str,
    inputs: Dict[str, Any],
    problem_id: str,
    run_path: str,
    retrieved_papers: Dict[str, Any],
    retrieval_cache: Dict[str, Any],
    retrieval_cache_path: Optional[str],
    prefix: str = "",
) -> None:
    """Write a full idea detail block: text, metadata, retrieved papers, and retrieval debug info."""
    input_data = _get_input(inputs, problem_id) or {}
    instance_metadata = input_data.get("metadata", {})
    paper_metadata = instance_metadata.get(int(idea_key), {})
    idea_text = str((input_data.get("ideas") or {}).get(int(idea_key), f"Idea {idea_key}"))
    title = paper_metadata.get("title", f"Idea {idea_key}")
    rating = paper_metadata.get("rating", "N/A")
    contribution = paper_metadata.get("contribution", "N/A")
    positive_signals = paper_metadata.get("positive_signals", [])
    negative_signals = paper_metadata.get("negative_signals", [])

    f.write(f"{prefix}- **Idea {idea_key}**\n")
    f.write(f"{prefix}  - **Title:** {title}\n")
    f.write(f"{prefix}  - **Text:**\n")
    for line in idea_text.split("\n"):
        f.write(f"{prefix}    > {line}\n")
    f.write(f"{prefix}  - **ICLR Rating:** {rating}\n")
    f.write(f"{prefix}  - **Contribution Score:** {contribution}\n")
    if positive_signals:
        f.write(format_signals_for_md("Positive Novelty Signals", positive_signals) + "\n")
    if negative_signals:
        f.write(format_signals_for_md("Negative Novelty Signals", negative_signals) + "\n")

    papers = retrieved_papers.get(idea_key, [])
    if not papers:
        papers = extract_retrieved_papers_from_log(run_path, str(problem_id), str(idea_key))
    if papers:
        f.write(format_retrieved_papers_for_md(papers) + "\n")

    queries, cands = get_idea_retrieval_data(retrieval_cache, problem_id, idea_key, retrieval_cache_path)
    if queries:
        f.write(format_retrieval_queries_for_md(queries) + "\n")
    if cands:
        f.write(format_candidates_confidence_for_md(cands) + "\n")
    f.write("\n")


def _write_comparison_reasoning_block(f, run_path: str, problem_id: str, gw: str, other_idea: str) -> None:
    entries = extract_all_comparison_reasonings(run_path, str(problem_id), str(gw), str(other_idea))
    f.write("  - <details>\n")
    f.write("    <summary><strong>Comparison Reasoning</strong></summary>\n\n")
    if not entries:
        f.write("    > Comparison log not found.\n")
    for entry in entries:
        winner_str = str(entry["winner"]) if entry["winner"] is not None else "?"
        label = f"Idea {entry['idea0']} vs Idea {entry['idea1']} (winner = {winner_str})"
        f.write("    <details>\n")
        f.write(f"    <summary>{label}</summary>\n\n")
        for line in format_reasoning_for_md(entry["text"]).split("\n"):
            f.write(f"    > {line}\n")
        f.write("    </details>\n\n")
    f.write("    </details>\n\n")


def extract_pointwise_reasonings(run_path: str, instance_id: str) -> List[Dict[str, Any]]:
    """Return all per-call reasoning entries for a pointwise instance, in file order.

    Each entry is a dict with: call_index (int), choice (dict|None), text (reasoning).
    """
    pointwise_file = os.path.join(run_path, "pointwise", f"{instance_id}.txt")
    if not os.path.exists(pointwise_file):
        return []
    try:
        with open(pointwise_file, "r") as fh:
            content = fh.read()
    except Exception:
        return []

    separator = "=" * 80
    entries = []
    start = 0
    while True:
        header_start = content.find(f"Pointwise evaluation of instance {instance_id} [call ", start)
        if header_start == -1:
            break
        sep_after = content.find(separator, header_start)
        if sep_after == -1:
            break

        # Parse call index from header
        header_line = content[header_start:content.find("\n", header_start)]
        call_index = None
        m = re.search(r"\[call (\d+)\]", header_line)
        if m:
            call_index = int(m.group(1))

        content_start = sep_after + len(separator)
        next_header = content.find(f"Pointwise evaluation of instance {instance_id} [call ", content_start)
        block = content[content_start:next_header].strip() if next_header != -1 else content[content_start:].strip()

        choice = None
        choice_idx = block.find("Choice:")
        if choice_idx != -1:
            choice_text = block[choice_idx + len("Choice:"):].strip()
            try:
                end = choice_text.find("\n\n")
                choice = json.loads(choice_text[:end] if end != -1 else choice_text)
            except (json.JSONDecodeError, ValueError):
                pass

        entries.append({"call_index": call_index, "choice": choice, "text": block})
        start = header_start + 1

    return entries


def _write_pointwise_reasoning_block(f, run_path: str, instance_id: str) -> None:
    entries = extract_pointwise_reasonings(run_path, str(instance_id))
    f.write("  - <details>\n")
    f.write("    <summary><strong>LLM Reasoning</strong></summary>\n\n")
    if not entries:
        f.write("    > Reasoning log not found.\n")
    for entry in entries:
        call_str = f"Call {entry['call_index']}" if entry["call_index"] is not None else "Call ?"
        choice = entry.get("choice")
        choice_str = f" — choice: {json.dumps(choice)}" if choice is not None else ""
        f.write("    <details>\n")
        f.write(f"    <summary>{call_str}{choice_str}</summary>\n\n")
        for line in entry["text"].split("\n"):
            f.write(f"    > {line}\n")
        f.write("    </details>\n\n")
    f.write("    </details>\n\n")


# ---------------------------------------------------------------------------
# Bad / Good comparison section writers
# ---------------------------------------------------------------------------

def _write_pairwise_bad_section(
    f,
    wrong_list: List[Dict[str, Any]],
    run_path: str,
    inputs: Dict[str, Any],
    retrieval_cache: Dict[str, Any],
    retrieval_cache_path: Optional[str],
) -> None:
    """Write bad comparisons for pairwise mode (entries have 'comparisons' key)."""
    if not wrong_list:
        f.write("No bad comparisons found.\n")
        return

    for wp in wrong_list:
        if not isinstance(wp, dict):
            continue
        problem_id = wp["problem_id"]
        expected_winners = wp.get("expected_winners", [])
        comps = wp.get("comparisons", [])

        f.write(f"### Problem {problem_id}\n")
        f.write(f"**Expected Winners:** {', '.join(str(w) for w in expected_winners)}\n\n")

        for comp in comps:
            winner = comp.get("winner")
            gt_winner = comp.get("gt_winner")
            idea_0 = str(comp.get("idea_0", "?"))
            idea_1 = str(comp.get("idea_1", "?"))
            try:
                w = int(winner)
            except (TypeError, ValueError):
                w = -1

            if gt_winner is None:
                continue  # no GT — skip

            is_wrong = (w == 2) or (w != gt_winner)
            if not is_wrong:
                continue

            outcome = "TIE" if w == 2 else f"Idea-{w} chosen (wrong)"
            f.write(f"- **Pair:** Idea {idea_0} vs Idea {idea_1} — **{outcome}** (expected gt_winner={gt_winner})\n")
            retrieved_papers = comp.get("retrieved_papers", {})
            _write_idea_block(f, idea_0, inputs, problem_id, run_path, retrieved_papers, retrieval_cache, retrieval_cache_path, prefix="  ")
            _write_idea_block(f, idea_1, inputs, problem_id, run_path, retrieved_papers, retrieval_cache, retrieval_cache_path, prefix="  ")
            _write_comparison_reasoning_block(f, run_path, problem_id, idea_0, idea_1)


def _write_pairwise_good_section(
    f,
    good_list: List[Dict[str, Any]],
    run_path: str,
    inputs: Dict[str, Any],
    retrieval_cache: Dict[str, Any],
    retrieval_cache_path: Optional[str],
) -> None:
    """Write good comparisons for pairwise mode (entries have 'comparisons' key)."""
    if not good_list:
        f.write("No good comparisons found.\n")
        return

    for gp in good_list:
        if not isinstance(gp, dict):
            continue
        problem_id = gp["problem_id"]
        expected_winners = gp.get("expected_winners", [])
        comps = gp.get("comparisons", [])

        f.write(f"### Problem {problem_id} \u2713\n")
        f.write(f"**Expected Winners:** {', '.join(str(w) for w in expected_winners)}\n\n")

        for comp in comps:
            winner = comp.get("winner")
            gt_winner = comp.get("gt_winner")
            idea_0 = str(comp.get("idea_0", "?"))
            idea_1 = str(comp.get("idea_1", "?"))
            if gt_winner is None:
                continue
            f.write(f"- **Pair:** Idea {idea_0} vs Idea {idea_1} — correctly chose Idea-{winner}\n")
            retrieved_papers = comp.get("retrieved_papers", {})
            _write_idea_block(f, idea_0, inputs, problem_id, run_path, retrieved_papers, retrieval_cache, retrieval_cache_path, prefix="  ")
            _write_idea_block(f, idea_1, inputs, problem_id, run_path, retrieved_papers, retrieval_cache, retrieval_cache_path, prefix="  ")
            _write_comparison_reasoning_block(f, run_path, problem_id, idea_0, idea_1)


def _is_pairwise_entries(pair_list: List[Dict[str, Any]]) -> bool:
    """Return True if the list contains pairwise-mode entries (have 'comparisons' key)."""
    for entry in pair_list:
        if isinstance(entry, dict):
            return "comparisons" in entry
    return False


def _write_bad_comparisons_section(
    f,
    wrong_list: List[Dict[str, Any]],
    run_path: str,
    inputs: Dict[str, Any],
    retrieval_cache: Dict[str, Any],
    retrieval_cache_path: Optional[str],
) -> None:
    if not wrong_list:
        f.write("No bad comparisons found.\n")
        return

    if _is_pairwise_entries(wrong_list):
        _write_pairwise_bad_section(f, wrong_list, run_path, inputs, retrieval_cache, retrieval_cache_path)
        return

    for wp in wrong_list:
        if not isinstance(wp, dict):
            continue
        problem_id = wp["problem_id"]
        predicted_rank = wp["predicted_rank"]
        gold_winners = wp["gold_winners"]
        gold_ranks = wp["gold_ranks"]
        scores = wp["scores"]
        retrieved_papers = wp.get("retrieved_papers", {})

        f.write(f"### Problem {problem_id}\n")
        f.write(f"**Gold Answer:** {', '.join(gold_ranks)}\n")
        f.write(f"**Predicted Rank:** {', '.join(predicted_rank)}\n\n")
        f.write("**Gold Winner Details:**\n")
        for gw in gold_winners:
            _write_idea_block(f, gw, inputs, problem_id, run_path, retrieved_papers, retrieval_cache, retrieval_cache_path)

        f.write("**Bad Comparisons**\n")
        bad_found = False
        for gw in gold_winners:
            gw_score = scores.get(gw, -1)
            for other_idea in predicted_rank:
                if other_idea in gold_winners:
                    continue
                if scores.get(other_idea, -1) >= gw_score:
                    bad_found = True
                    f.write(f"- **Idea {other_idea}** ranked >= **Gold Idea {gw}**\n")
                    _write_idea_block(f, other_idea, inputs, problem_id, run_path, retrieved_papers, retrieval_cache, retrieval_cache_path, prefix="  ")
                    _write_comparison_reasoning_block(f, run_path, problem_id, gw, other_idea)

        if not bad_found:
            f.write(
                "No specific bad comparisons (gold winners tied or ranked correctly against others, "
                "but maybe internal ordering issue).\n"
            )


def _write_good_comparisons_section(
    f,
    good_list: List[Dict[str, Any]],
    run_path: str,
    inputs: Dict[str, Any],
    retrieval_cache: Dict[str, Any],
    retrieval_cache_path: Optional[str],
) -> None:
    if not good_list:
        f.write("No good comparisons found.\n")
        return

    if _is_pairwise_entries(good_list):
        _write_pairwise_good_section(f, good_list, run_path, inputs, retrieval_cache, retrieval_cache_path)
        return

    for gp in good_list:
        if not isinstance(gp, dict):
            continue
        problem_id = gp["problem_id"]
        predicted_rank = gp["predicted_rank"]
        gold_winners = gp["gold_winners"]
        gold_ranks = gp["gold_ranks"]
        scores = gp["scores"]
        retrieved_papers = gp.get("retrieved_papers", {})

        f.write(f"### Problem {problem_id} \u2713\n")
        f.write(f"**Gold Answer:** {', '.join(gold_ranks)}\n")
        f.write(f"**Predicted Rank:** {', '.join(predicted_rank)}\n\n")
        f.write("**Gold Winner Details:**\n")
        for gw in gold_winners:
            _write_idea_block(f, gw, inputs, problem_id, run_path, retrieved_papers, retrieval_cache, retrieval_cache_path)

        f.write("**Correctly Ranked Non-Winners:**\n")
        for gw in gold_winners:
            gw_score = scores.get(gw, -1)
            for other_idea in predicted_rank:
                if other_idea in gold_winners:
                    continue
                if scores.get(other_idea, -1) < gw_score:
                    f.write(f"- **Idea {other_idea}** correctly ranked below **Gold Idea {gw}**\n")
                    _write_idea_block(f, other_idea, inputs, problem_id, run_path, retrieved_papers, retrieval_cache, retrieval_cache_path, prefix="  ")
                    _write_comparison_reasoning_block(f, run_path, problem_id, gw, other_idea)


# ---------------------------------------------------------------------------
# Pointwise classification section writer
# ---------------------------------------------------------------------------

def _write_pointwise_instance_block(
    f,
    instance_id: str,
    data: Dict[str, Any],
    prediction: int,
    retrieval_cache: Optional[Dict[str, Any]] = None,
    run_path: str = "",
    prefix: str = "",
) -> None:
    """Write a single pointwise instance: label, prediction, metadata, idea text, and retrieval info."""
    label = data.get("label", "?")
    metadata = data.get("metadata", {})
    title = metadata.get("title", f"Instance {instance_id}")
    rating = metadata.get("rating", "N/A")
    contribution = metadata.get("contribution", "N/A")
    positive_signals = metadata.get("positive_signals", [])
    negative_signals = metadata.get("negative_signals", [])
    idea_text = str(data.get("idea", ""))

    pred_str = "POSITIVE (novel)" if prediction == 1 else "NEGATIVE (not novel)"
    f.write(f"{prefix}- **Instance {instance_id}**\n")
    f.write(f"{prefix}  - **Title:** {title}\n")
    f.write(f"{prefix}  - **Ground Truth:** {label} | **Predicted:** {pred_str}\n")
    f.write(f"{prefix}  - **ICLR Rating:** {rating} | **Contribution Score:** {contribution}\n")
    f.write(f"{prefix}  - **Idea Text:**\n")
    for line in idea_text.split("\n"):
        f.write(f"{prefix}    > {line}\n")
    if positive_signals:
        f.write(format_signals_for_md("Positive Novelty Signals", positive_signals) + "\n")
    if negative_signals:
        f.write(format_signals_for_md("Negative Novelty Signals", negative_signals) + "\n")

    if retrieval_cache is not None:
        cached = retrieval_cache.get(str(instance_id), {})
        search_queries_dict = cached.get("search_queries_dict", {})
        candidates = cached.get("candidates", [])
        if search_queries_dict:
            f.write(format_retrieval_queries_for_md(search_queries_dict) + "\n")
        if candidates:
            f.write(format_retrieved_papers_for_md(candidates) + "\n")
            f.write(format_candidates_confidence_for_md(candidates) + "\n")

    if run_path:
        _write_pointwise_reasoning_block(f, run_path, instance_id)

    f.write("\n")


def _write_pointwise_section(
    f,
    anchor_prefix: str,
    title: str,
    stats: ExperimentStats,
    inputs: Dict[str, Any],
    retrieval_cache: Optional[Dict[str, Any]] = None,
) -> None:
    f.write(f"<a id='{anchor_prefix}'></a>\n")
    f.write(f"# {title}\n\n")

    for i, run_path in enumerate(stats.run_paths):
        scores_path = os.path.join(run_path, "scores.json")
        if not os.path.exists(scores_path):
            f.write(f"## Run {i + 1} — scores.json not found\n\n")
            continue

        with open(scores_path) as fh:
            run_results: Dict[str, Any] = json.load(fh)

        wrong: List[Tuple[str, int, str]] = []   # (instance_id, prediction, label)
        correct: List[Tuple[str, int, str]] = []

        for iid, entry in run_results.items():
            pred = entry.get("prediction", -1)
            instance_data = _get_input(inputs, iid) or {}
            label = instance_data.get("label", entry.get("label", "?"))
            gt = 1 if label == "POSITIVE" else 0
            if pred == gt:
                correct.append((iid, pred, label))
            else:
                wrong.append((iid, pred, label))

        f.write(f"<a id='{anchor_prefix}-run-{i + 1}-wrong'></a>\n")
        f.write(f"## Run {i + 1} — Wrong Classifications ({len(wrong)})\n\n")
        if not wrong:
            f.write("All instances classified correctly.\n\n")
        else:
            for iid, pred, _label in wrong:
                instance_data = inputs.get(iid) or inputs.get(int(iid), {})
                _write_pointwise_instance_block(f, iid, instance_data, pred, retrieval_cache, run_path)

        f.write(f"<a id='{anchor_prefix}-run-{i + 1}-correct'></a>\n")
        f.write(f"## Run {i + 1} — Correct Classifications ({len(correct)})\n\n")
        if not correct:
            f.write("No instances classified correctly.\n\n")
        else:
            for iid, pred, _label in correct:
                instance_data = inputs.get(iid) or inputs.get(int(iid), {})
                _write_pointwise_instance_block(f, iid, instance_data, pred, retrieval_cache, run_path)


# ---------------------------------------------------------------------------
# Tournament section writer — reused for RR, Swiss, Bi-Swiss
# ---------------------------------------------------------------------------

def _write_tournament_section(
    f,
    anchor_prefix: str,
    title: str,
    stats: ExperimentStats,
    inputs: Dict[str, Any],
    retrieval_cache: Dict[str, Any],
    retrieval_cache_path: Optional[str],
) -> None:
    f.write(f"<a id='{anchor_prefix}'></a>\n")
    f.write(f"# {title}\n")

    for i, wrong_list in enumerate(stats.wrong_pairs):
        run_path = stats.run_paths[i] if i < len(stats.run_paths) else ""
        f.write(f"<a id='{anchor_prefix}-run-{i + 1}-bad'></a>\n")
        f.write(f"## Run {i + 1} \u2014 Bad Comparisons\n")
        _write_bad_comparisons_section(f, wrong_list, run_path, inputs, retrieval_cache, retrieval_cache_path)

    for i, good_list in enumerate(stats.good_pairs):
        run_path = stats.run_paths[i] if i < len(stats.run_paths) else ""
        f.write(f"<a id='{anchor_prefix}-run-{i + 1}-good'></a>\n")
        f.write(f"## Run {i + 1} \u2014 Good Comparisons\n")
        _write_good_comparisons_section(f, good_list, run_path, inputs, retrieval_cache, retrieval_cache_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def save_md_debug_report(
    output_file: str,
    rr_stats: ExperimentStats,
    swiss_stats: ExperimentStats,
    inputs: Dict[str, Any],
    modes: List[str],
    retrieval_cache: Dict[str, Any],
    args: argparse.Namespace,
    pairwise_stats: ExperimentStats = None,
    pointwise_stats: ExperimentStats = None,
) -> None:
    if pairwise_stats is None:
        pairwise_stats = ExperimentStats.empty()
    if pointwise_stats is None:
        pointwise_stats = ExperimentStats.empty()

    retrieval_cache_path: Optional[str] = getattr(args, "retrieval_cache_file", None)

    with open(output_file, "w") as f:
        # Table of Contents
        f.write("# Comparative Evaluator Debug Report\n\n")
        f.write("## Table of Contents\n")
        f.write("- [Settings](#settings)\n")
        if "pointwise" in modes:
            f.write("- [Pointwise Classification](#pointwise-classification)\n")
            for i in range(len(pointwise_stats.run_paths)):
                f.write(f"  - [Run {i+1} - Wrong Classifications](#pointwise-run-{i+1}-wrong)\n")
                f.write(f"  - [Run {i+1} - Correct Classifications](#pointwise-run-{i+1}-correct)\n")
        if "pairwise" in modes:
            f.write("- [Pairwise Comparisons](#pairwise-comparisons)\n")
            for i in range(len(pairwise_stats.wrong_pairs)):
                f.write(f"  - [Run {i+1} - Bad Comparisons](#pairwise-run-{i+1}-bad)\n")
            for i in range(len(pairwise_stats.good_pairs)):
                f.write(f"  - [Run {i+1} - Good Comparisons](#pairwise-run-{i+1}-good)\n")
        if "rr" in modes:
            f.write("- [Round Robin Tournament](#round-robin-tournament)\n")
            for i in range(len(rr_stats.wrong_pairs)):
                f.write(f"  - [Run {i+1} - Bad Comparisons](#rr-run-{i+1}-bad)\n")
            for i in range(len(rr_stats.good_pairs)):
                f.write(f"  - [Run {i+1} - Good Comparisons](#rr-run-{i+1}-good)\n")
        if "swiss" in modes:
            f.write("- [Swiss Tournament](#swiss-tournament)\n")
            for i in range(len(swiss_stats.wrong_pairs)):
                f.write(f"  - [Run {i+1} - Bad Comparisons](#swiss-run-{i+1}-bad)\n")
            for i in range(len(swiss_stats.good_pairs)):
                f.write(f"  - [Run {i+1} - Good Comparisons](#swiss-run-{i+1}-good)\n")
        f.write("\n")

        # Settings section
        f.write("<a id='settings'></a>\n")
        f.write("<summary><h1>Settings</h1></summary>\n\n")
        f.write("### Command Line Arguments\n```json\n")
        f.write(json.dumps(vars(args), indent=2, default=str))
        f.write("\n```\n\n")

        f.write("### Evaluation Criteria\n")
        criteria = EVALUATION_CRITERIA_RETRIEVAL if retrieval_cache_path else EVALUATION_CRITERIA
        f.write("```json\n")
        f.write(json.dumps(criteria, indent=2, default=str))
        f.write("\n```\n\n")

        f.write("### Prompt Template\n")
        template_name = "pointwise_novelty.jinja2" if "pointwise" in modes else "pairwise_novelty.jinja2"
        template_path = os.path.join(os.path.dirname(__file__), "templates", template_name)
        if os.path.exists(template_path):
            with open(template_path, "r") as tf:
                content = tf.read()
            max_ticks = max((len(m.group(0)) for m in re.finditer(r"`+", content)), default=2)
            fence = "`" * max(3, max_ticks + 1)
            f.write(f"{fence}\n{content}\n{fence}\n\n")
        else:
            f.write(f"Template file not found at {template_path}\n")

        # Pointwise classification section
        if "pointwise" in modes:
            _write_pointwise_section(
                f, "pointwise-classification", "Pointwise Classification",
                pointwise_stats, inputs, retrieval_cache if retrieval_cache_path else None,
            )

        # Tournament sections
        if "pairwise" in modes:
            _write_tournament_section(
                f, "pairwise-comparisons", "Pairwise Comparisons",
                pairwise_stats, inputs, retrieval_cache, retrieval_cache_path,
            )
        if "rr" in modes:
            _write_tournament_section(
                f, "round-robin-tournament", "Round Robin Tournament",
                rr_stats, inputs, retrieval_cache, retrieval_cache_path,
            )
        if "swiss" in modes:
            _write_tournament_section(
                f, "swiss-tournament", "Swiss Tournament",
                swiss_stats, inputs, retrieval_cache, retrieval_cache_path,
            )


