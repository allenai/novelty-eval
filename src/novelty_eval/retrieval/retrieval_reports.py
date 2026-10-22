#!/usr/bin/env python3
import os
import re
import sys
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import LOGGER


def format_retrieval_debug_info(idea_key: str, idea_text: str, topic: str, debug_data: Dict,
                                judge_debug: Dict = None) -> str:
    """Render the per-idea retrieval debug markdown.

    `judge_debug` is an optional block (used by web_search_novelty_judge.py) carrying the
    judge's own reasoning trail — verdict, raw cited-paper list, grounding outcome, and the
    full prompt/response — so the judge's decision can be audited without opening the raw
    JSON debug artifact.
    """
    contributions = debug_data.get('contributions', {})
    search_queries_dict = debug_data.get('search_queries_dict', {})
    candidates = debug_data.get('candidates', [])
    skipped_same_title = debug_data.get('skipped_same_title', [])

    def blockquote(text: str) -> str:
        return "\n".join(f"> {line}" for line in text.splitlines()) if text else "> *(empty)*"

    def highlight_terms(text: str, terms: List[str]) -> str:
        for term in terms:
            if term:
                text = re.sub(f'({re.escape(term)})', r'**\1**', text, flags=re.IGNORECASE)
        return text

    def norm_arxiv_id(s: str) -> str:
        s = str(s or '').strip()
        s = re.sub(r'(?i)^arxiv:', '', s).strip()
        s = re.sub(r'v\d+$', '', s)
        return s.lower()

    lines = []
    lines.append(f"# Retrieval Debug — Idea {idea_key}")
    lines.append("")
    lines.append(f"**Topic:** {topic}")
    lines.append("")
    lines.append("## Idea Text")
    lines.append("")
    lines.append(blockquote(idea_text))
    lines.append("")

    if judge_debug is not None:
        self_judgement = judge_debug.get('self_judgement')
        gold_label = judge_debug.get('gold_label')  # 'POSITIVE' (novel) | 'NEGATIVE' (not novel)
        lines.append("## Judge Verdict")
        lines.append("")
        if gold_label:
            lines.append(f"**Gold label:** {gold_label}")
        if self_judgement:
            novelty = self_judgement.get('novelty')
            verdict_label = {1: "**NOVEL** (1)", 0: "**NOT NOVEL** (0)"}.get(novelty, "N/A")
            lines.append(f"**Novelty verdict:** {verdict_label}")
            if gold_label in ("POSITIVE", "NEGATIVE") and novelty in (0, 1):
                gold_novelty = 1 if gold_label == "POSITIVE" else 0
                match = "✅ **CORRECT**" if novelty == gold_novelty else "❌ **WRONG**"
                lines.append(f"**Match:** {match}")
            lines.append("")
            criteria = self_judgement.get('criteria') or {}
            if criteria:
                lines.append("| Criterion | Label |")
                lines.append("|---|---|")
                for k, v in criteria.items():
                    lines.append(f"| {k} | {v} |")
                lines.append("")
            lines.append("**Reasoning (model's `<thinking>` block)**")
            lines.append("")
            lines.append(blockquote(self_judgement.get('reasoning') or ''))
            lines.append("")
        else:
            lines.append("*No verdict parsed for this idea (parse failure or missing response).*")
            lines.append("")

        raw_items = judge_debug.get('raw_items') or []
        unresolved = {norm_arxiv_id(a) for a in (judge_debug.get('unresolved_arxiv_ids') or [])}
        lines.append("## Cited Papers — Raw Model Output (pre-grounding)")
        lines.append("")
        if not raw_items:
            lines.append("*Model did not cite any papers in its `[PAPERS_START]...[PAPERS_END]` block.*")
            lines.append("")
        else:
            lines.append("| # | arXiv ID | Relevance Score | Grounded? | Why Retrieved |")
            lines.append("|---|---|---|---|---|")
            for i, it in enumerate(raw_items, 1):
                if not isinstance(it, dict):
                    continue
                aid = it.get('arxiv_id', 'N/A')
                score = it.get('relevance_score', 'N/A')
                why = (it.get('why_retrieve') or '').replace('\n', ' ')
                grounded = "❌ unresolved" if norm_arxiv_id(aid) in unresolved else "✅"
                lines.append(f"| {i} | `{aid}` | {score} | {grounded} | {why} |")
            lines.append("")

    lines.append("## Step 1 — Extracted Contributions")
    lines.append("")
    if not contributions:
        lines.append("*No contributions extracted.*")
    else:
        for dimension, statements in contributions.items():
            if not isinstance(statements, list) or not statements:
                continue
            lines.append(f"### {dimension}")
            lines.append("")
            for s in statements:
                lines.append(blockquote(s))
            lines.append("")

    lines.append("## Step 2 — Generated Search Queries")
    lines.append("")
    if not search_queries_dict:
        lines.append("*No queries generated.*")
    else:
        for dimension, queries in search_queries_dict.items():
            for q in queries:
                lines.append(f"- {q}")
        lines.append("")

    lines.append("## Selected Candidates")
    lines.append("")
    if not candidates:
        lines.append("*No candidates found.*")
    else:
        for i, c in enumerate(candidates, 1):
            title = c.get('title', 'Unknown Title')
            year = c.get('year', 'N/A')
            publication_date = c.get('publication_date', 'N/A')
            url = c.get('url', 'N/A')
            score = (c.get('relevance_judgement') or {}).get('relevance_score', 'N/A')
            abstract = c.get('abstract') or c.get('text') or ''
            source_query = c.get('source_query', '')

            relevance_judgement = c.get('relevance_judgement') or {}
            relevance_summary = relevance_judgement.get('relevance_summary', '')
            criteria_judgements = relevance_judgement.get('relevance_criteria_judgements') or []
            relevance_int = relevance_judgement.get('relevance')
            relevance_grade = f"{relevance_int}/3" if relevance_int is not None else 'N/A'

            lines.append(f"### {i}. [{title}]({url})")
            lines.append("")
            lines.append(f"| Field | Value |")
            lines.append(f"|---|---|")
            lines.append(f"| Year | {year} |")
            lines.append(f"| Publication Date | {publication_date} |")
            lines.append(f"| Relevance | {relevance_grade} |")
            lines.append(f"| Relevance Score | {score} |")
            if source_query:
                lines.append(f"| Source Query | *{source_query}* |")
            lines.append("")
            if criteria_judgements:
                lines.append("**Relevance by Criterion**")
                lines.append("")
                lines.append("| Criterion | Score | Evidence |")
                lines.append("|---|---|---|")
                for cj in criteria_judgements:
                    cj_name = cj.get('name', '')
                    cj_score = cj.get('relevance', 'N/A')
                    top_snippet = ((cj.get('relevant_snippets') or [{}])[0]).get('text', '')
                    evidence = highlight_terms(top_snippet, [cj_name]).replace('\n', ' ') if top_snippet else ''
                    lines.append(f"| {cj_name} | {cj_score}/3 | {evidence} |")
                lines.append("")
            if relevance_summary:
                lines.append("**Why Retrieved**")
                lines.append("")
                lines.append(blockquote(relevance_summary))
                lines.append("")
            if abstract:
                criterion_names = [cj.get('name', '') for cj in criteria_judgements]
                lines.append("**Abstract**")
                lines.append("")
                lines.append(blockquote(highlight_terms(abstract, criterion_names)))
                lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## ⚠️ Skipped — Same Title as Idea")
    lines.append("")
    if not skipped_same_title:
        lines.append("*No candidates were skipped for this reason.*")
        lines.append("")
    else:
        lines.append(
            f"The following {len(skipped_same_title)} candidate(s) were **removed** because their "
            f"title exactly matches the title of the idea being evaluated, "
            f"indicating they are the same paper."
        )
        lines.append("")

        lines.append("### 📄 Idea Being Evaluated")
        lines.append("")
        lines.append(f"**Key:** `{idea_key}`")
        lines.append("")
        lines.append("**Text**")
        lines.append("")
        lines.append(blockquote(idea_text))
        lines.append("")

        lines.append("### 🔁 Skipped Duplicates")
        lines.append("")
        for i, c in enumerate(skipped_same_title, 1):
            title = c.get('title', 'Unknown Title')
            url = c.get('url', '#')
            score = (c.get('relevance_judgement') or {}).get('relevance_score', 'N/A')
            year = c.get('year', 'N/A')
            source_query = c.get('source_query', '')
            abstract = c.get('abstract') or c.get('text') or ''

            lines.append(f"#### {i}. [{title}]({url})")
            lines.append("")
            lines.append(f"| Field | Value |")
            lines.append(f"|---|---|")
            lines.append(f"| Relevance Score | **{score}** |")
            lines.append(f"| Year | {year} |")
            if source_query:
                lines.append(f"| Source Query | *{source_query}* |")
            lines.append("")
            if abstract:
                lines.append("**Abstract / Text**")
                lines.append("")
                lines.append(blockquote(abstract))
                lines.append("")

    if judge_debug is not None and (judge_debug.get('prompt') or judge_debug.get('raw_text')):
        lines.append("---")
        lines.append("")
        lines.append("## Full Trace")
        lines.append("")
        prompt = judge_debug.get('prompt')
        if prompt:
            lines.append("<details>")
            lines.append("<summary><strong>Compiled Prompt</strong></summary>")
            lines.append("")
            lines.append("```")
            lines.append(prompt)
            lines.append("```")
            lines.append("</details>")
            lines.append("")
        raw_text = judge_debug.get('raw_text')
        if raw_text:
            lines.append("<details>")
            lines.append("<summary><strong>Raw Model Answer</strong></summary>")
            lines.append("")
            lines.append("```")
            lines.append(raw_text)
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)


def generate_cache_status_report(retrieval_cache: Dict, inputs: Dict, output_file: str, args) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = os.path.dirname(os.path.abspath(output_file))
    report_path = os.path.join(output_path, "cache_status.md")

    total_problems_in_input = len(inputs)
    total_ideas_in_input = 0
    total_ideas_cached = 0
    total_ideas_missing = 0
    total_ideas_empty_candidates = 0
    total_candidates = 0
    candidate_counts = []
    per_problem_lines = []

    for problem_id, data in inputs.items():
        ideas = data.get('ideas', {})
        is_pointwise = not ideas and 'idea' in data

        if is_pointwise:
            ideas = {'idea': data['idea']}

        num_ideas = len(ideas)
        total_ideas_in_input += num_ideas

        str_problem = str(problem_id)
        cached_entry = retrieval_cache.get(str_problem, {})

        cached_keys = []
        missing_keys = []
        empty_keys = []

        if is_pointwise:
            if isinstance(cached_entry, dict) and cached_entry.get('candidates'):
                cached_keys.append('idea')
                n_cands = len(cached_entry['candidates'])
                total_candidates += n_cands
                candidate_counts.append(n_cands)
            elif isinstance(cached_entry, dict) and cached_entry:
                empty_keys.append('idea')
            else:
                missing_keys.append('idea')
        else:
            cached_problem = cached_entry
            for key in ideas:
                str_key = str(key)
                if str_key in cached_problem:
                    entry = cached_problem[str_key]
                    cands = entry.get('candidates', [])
                    if cands:
                        cached_keys.append(str_key)
                        n_cands = len(cands)
                        total_candidates += n_cands
                        candidate_counts.append(n_cands)
                    else:
                        empty_keys.append(str_key)
                else:
                    missing_keys.append(str_key)

        total_ideas_cached += len(cached_keys)
        total_ideas_missing += len(missing_keys)
        total_ideas_empty_candidates += len(empty_keys)

        if len(missing_keys) == 0 and len(empty_keys) == 0:
            status = "✅"
        elif len(cached_keys) > 0:
            status = "⚠️"
        else:
            status = "❌"

        per_problem_lines.append(f"| {status} `{problem_id}` | {num_ideas} | {len(cached_keys)} | {len(missing_keys)} | {len(empty_keys)} |")

    coverage_pct = (total_ideas_cached / total_ideas_in_input * 100) if total_ideas_in_input > 0 else 0

    def _is_fully_covered(pid, data):
        entry = retrieval_cache.get(str(pid), {})
        if not data.get('ideas') and 'idea' in data:
            return isinstance(entry, dict) and bool(entry.get('candidates'))
        return all(
            str(k) in entry and entry.get(str(k), {}).get('candidates')
            for k in data.get('ideas', {})
        )

    problems_fully_covered = sum(
        1 for pid, data in inputs.items()
        if _is_fully_covered(pid, data)
    )

    avg_candidates = (total_candidates / total_ideas_cached) if total_ideas_cached > 0 else 0
    min_candidates = min(candidate_counts) if candidate_counts else 0
    max_candidates = max(candidate_counts) if candidate_counts else 0

    lines = [
        f"# 📦 Cache Status Report",
        f"",
        f"**Generated:** {timestamp}  ",
        f"**Cache file:** `{os.path.abspath(output_file)}`  ",
        f"**Input file:** `{os.path.abspath(args.test_inputs)}`  ",
        f"",
        f"---",
        f"",
        f"## Overview",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total problems in input | {total_problems_in_input} |",
        f"| Problems fully covered | {problems_fully_covered} / {total_problems_in_input} |",
        f"| Total ideas in input | {total_ideas_in_input} |",
        f"| Ideas cached (with candidates) | {total_ideas_cached} |",
        f"| Ideas missing from cache | {total_ideas_missing} |",
        f"| Ideas cached but 0 candidates | {total_ideas_empty_candidates} |",
        f"| **Coverage** | **{coverage_pct:.1f}%** |",
        f"",
        f"## Candidate Statistics",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total candidates across all ideas | {total_candidates} |",
        f"| Avg candidates per idea | {avg_candidates:.1f} |",
        f"| Min candidates per idea | {min_candidates} |",
        f"| Max candidates per idea | {max_candidates} |",
        f"",
        f"## Run Configuration",
        f"",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| LLM engine | `{args.llm_engine}` |",
        f"| Top-K candidates | {args.top_k_candidates} |",
        f"| Cutoff date | {args.cutoff_date if args.cutoff_date else 'None'} |",
        f"| Max contributions | {args.max_contributions} |",
        f"| N queries | {args.n_queries} |",
        f"| Use Semantic Scholar | {args.use_semantic_scholar} |",
        f"| Max search workers | {args.max_search_workers} |",
        f"| Nr examples limit | {args.nr_examples if args.nr_examples else 'All'} |",
        f"",
        f"## Per-Problem Breakdown",
        f"",
        f"| Problem | Total Ideas | Cached | Missing | Empty (0 candidates) |",
        f"|---|---|---|---|---|",
    ]
    lines.extend(per_problem_lines)
    lines.extend([
        f"",
        f"**Legend:** ✅ = fully covered | ⚠️ = partially covered | ❌ = no ideas cached",
    ])

    report_content = "\n".join(lines) + "\n"

    try:
        os.makedirs(os.path.dirname(report_path) if os.path.dirname(report_path) else '.', exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_content)
        LOGGER.info(f"Cache status report saved to {report_path}")
    except Exception as e:
        LOGGER.error(f"Failed to save cache status report: {e}")

    return report_path


def write_skipped_duplicates_log(log_entries: List[Dict], output_path: str):
    log_path = os.path.join(output_path, "retrieval_debug", "skipped_duplicates_log.md")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    by_problem: Dict[str, List[Dict]] = {}
    for entry in log_entries:
        pid = str(entry['problem_id'])
        by_problem.setdefault(pid, []).append(entry)

    n_problems_affected = len(by_problem)
    n_ideas_affected = len({(e['problem_id'], e['idea_key']) for e in log_entries})
    n_total_skipped = len(log_entries)

    lines = [
        "# 🔁 Skipped Duplicates Log",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Filter:** title match — candidate title equals the idea's known title  ",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Problems affected | {n_problems_affected} |",
        f"| Ideas with ≥1 skipped candidate | {n_ideas_affected} |",
        f"| Total skipped candidates | {n_total_skipped} |",
        "",
        "---",
        "",
    ]

    if not log_entries:
        lines.append("*No candidates were skipped in this run.*")
    else:
        for pid, entries in by_problem.items():
            lines.append(f"## Problem `{pid}`")
            lines.append("")
            lines.append("| Idea Key | Candidate Title | Relevance Score | Year | Source Query |")
            lines.append("|---|---|---|---|---|")
            for e in entries:
                title_cell = f"[{e['candidate_title']}]({e['candidate_url']})" if e.get('candidate_url', '#') != '#' else e['candidate_title']
                sq = e.get('source_query', '')
                lines.append(
                    f"| `{e['idea_key']}` | {title_cell} | **{e['relevance_score']}** | {e['year']} | {sq} |"
                )
            lines.append("")

    content = "\n".join(lines) + "\n"
    try:
        with open(log_path, 'w') as f:
            f.write(content)
        LOGGER.info(f"Skipped-duplicates log saved to {log_path}")
    except Exception as exc:
        LOGGER.error(f"Failed to write skipped-duplicates log: {exc}")


def write_query_status_report(all_outcomes: List[Dict], output_path: str):
    log_path = os.path.join(output_path, "retrieval_debug", "query_status.md")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_total = len(all_outcomes)
    n_success = sum(1 for o in all_outcomes if o["status"] == "success")
    n_empty = sum(1 for o in all_outcomes if o["status"] == "empty")
    n_failed = sum(1 for o in all_outcomes if o["status"] == "failed")

    ideas = {(str(o["problem_id"]), str(o["idea_key"])) for o in all_outcomes}
    ideas_with_failed = {(str(o["problem_id"]), str(o["idea_key"])) for o in all_outcomes if o["status"] == "failed"}
    ideas_all_failed = {
        (pid, ik) for pid, ik in ideas
        if all(
            o["status"] == "failed"
            for o in all_outcomes
            if str(o["problem_id"]) == pid and str(o["idea_key"]) == ik
        )
    }

    def pct(n):
        return f"{n / n_total * 100:.1f}%" if n_total > 0 else "N/A"

    STATUS_ICON = {"success": "✅", "empty": "⚠️", "failed": "❌"}

    by_problem: Dict[str, List[Dict]] = {}
    for o in all_outcomes:
        pid = str(o["problem_id"])
        by_problem.setdefault(pid, []).append(o)

    lines = [
        "# 🔍 Query Status Report",
        "",
        f"**Generated:** {timestamp}  ",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total queries | {n_total} |",
        f"| ✅ Successful (≥1 paper) | {n_success} ({pct(n_success)}) |",
        f"| ⚠️ Empty (0 papers, API OK) | {n_empty} ({pct(n_empty)}) |",
        f"| ❌ Failed (API error) | {n_failed} ({pct(n_failed)}) |",
        f"| Ideas with ≥1 failed query | {len(ideas_with_failed)} / {len(ideas)} |",
        f"| Ideas where all queries failed | {len(ideas_all_failed)} |",
        "",
        "---",
        "",
        "## Per-Problem Breakdown",
        "",
    ]

    def _pipeline(o: Dict) -> str:
        """Render the filter pipeline for one query outcome as 'raw→date→dedup→selected'."""
        raw = o.get("raw")
        if raw is None:
            return "—"
        parts = [str(raw)]
        after_date = o.get("after_date")
        if after_date is not None:
            parts.append(f"{after_date} (date)")
        after_dedup = o.get("after_dedup")
        if after_dedup is not None and after_dedup != (after_date if after_date is not None else raw):
            parts.append(f"{after_dedup} (dedup)")
        after_id_dedup = o.get("after_id_dedup")
        if after_id_dedup is not None and after_id_dedup != after_dedup:
            parts.append(f"{after_id_dedup} (id-dedup)")
        selected = o.get("selected")
        if selected is not None:
            parts.append(f"**{selected} selected**")
        return " → ".join(parts)

    if not all_outcomes:
        lines.append("*No queries were executed in this run.*")
    else:
        for pid, outcomes in by_problem.items():
            lines.append(f"### Problem `{pid}`")
            lines.append("")
            lines.append("| Idea Key | Query | Status | Attempts | Filter pipeline |")
            lines.append("|---|---|---|---|---|")
            for o in outcomes:
                q = o["query"]
                q_display = (q[:80] + "…") if len(q) > 80 else q
                icon = STATUS_ICON.get(o["status"], "?")
                lines.append(
                    f"| `{o['idea_key']}` | {q_display} | {icon} {o['status']} | {o['attempts']} | {_pipeline(o)} |"
                )
            lines.append("")

    content = "\n".join(lines) + "\n"
    try:
        with open(log_path, "w") as f:
            f.write(content)
        LOGGER.info(f"Query status report saved to {log_path}")
    except Exception as exc:
        LOGGER.error(f"Failed to write query status report: {exc}")
