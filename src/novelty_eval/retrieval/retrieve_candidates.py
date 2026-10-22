#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
import yaml
import json
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import LOGGER as _BASE_LOGGER, prompt_openai_client, extract_json_choice
try:
    from logging_utils import setup_logger as _setup_logger
except ImportError:
    try:
        from src.logging_utils import setup_logger as _setup_logger
    except ImportError:
        _setup_logger = None

LOGGER = _BASE_LOGGER
from paper_finder_api import STATS as PAPER_FINDER_STATS, search_papers, PaperFinderAPIError
from semantic_scholar import SEMANTIC_API
try:
    from .retrieval_reports import format_retrieval_debug_info, generate_cache_status_report, write_skipped_duplicates_log, write_query_status_report
except ImportError:
    from retrieval_reports import format_retrieval_debug_info, generate_cache_status_report, write_skipped_duplicates_log, write_query_status_report

# Shared, method-agnostic retrieval utilities (cache I/O, filtering, formatting).
try:
    from .retrieval_common import (
        RetrievalConfigBase,
        render_prompt,
        load_retrieval_cache,
        save_retrieval_cache,
        load_query_outcomes_log,
        save_query_outcomes_log,
        _filter_and_select_candidates,
        format_papers_for_prompt,
        select_and_format_candidates,
        read_inputs,
        run_in_chunks,
        set_logger as _set_common_logger,
    )
except ImportError:
    from retrieval_common import (
        RetrievalConfigBase,
        render_prompt,
        load_retrieval_cache,
        save_retrieval_cache,
        load_query_outcomes_log,
        save_query_outcomes_log,
        _filter_and_select_candidates,
        format_papers_for_prompt,
        select_and_format_candidates,
        read_inputs,
        run_in_chunks,
        set_logger as _set_common_logger,
    )

try:
    from src.cost_tracker import cost_stage, GLOBAL_COST_TRACKER
except ImportError:
    try:
        from cost_tracker import cost_stage, GLOBAL_COST_TRACKER
    except ImportError:
        from contextlib import nullcontext as cost_stage
        GLOBAL_COST_TRACKER = None

try:
    from novelty_eval.cost_report_writer import write_cost_report_md as _write_cost_report_md
except ImportError:
    try:
        from cost_report_writer import write_cost_report_md as _write_cost_report_md
    except ImportError:
        _write_cost_report_md = None


@dataclass
class RetrievalConfig(RetrievalConfigBase):
    llm_engine: str = 'gpt-5.2'
    max_search_workers: int = 10
    use_semantic_scholar: bool = False
    chunk_size: int = 50
    max_contributions: int = 3
    n_queries: int = 3
    search_max_retries: int = 3
    search_retry_delay: float = 2.0
    retry_failed_queries: bool = False


def call_llm(full_prompt: str, system_prompt: str, model_name: str) -> str:
    combined_prompt = f"{system_prompt}\n\n{full_prompt}"
    with cost_stage("retrieval"):
        return prompt_openai_client(combined_prompt, engine=model_name)

def call_llm_json(full_prompt: str, system_prompt: str, model_name: str, max_retries: int = 3) -> Dict:
    for attempt in range(max_retries):
        response_text = call_llm(full_prompt, system_prompt, model_name)
        result = extract_json_choice(response_text)
        if result:
            return result
        LOGGER.warning(f"Failed to parse JSON (attempt {attempt + 1}/{max_retries}). Retrying...")
    return {}

def extract_search_queries(idea_text: str,
                           model_name: str,
                           max_contributions: int = 3,
                           n_queries: int = 3) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Two-step query generation: first extract contributions, then generate queries.
    Returns (queries_dict, contributions_dict).
    """
    # Step 1: Extract contributions
    LOGGER.debug("Step 1: Extracting contributions from idea...")

    user_prompt_1 = render_prompt("contribution_user.jinja2", idea_text=idea_text, n_contributions=max_contributions)
    system_prompt_1 = render_prompt("contribution_system.jinja2")

    contributions = call_llm_json(user_prompt_1, system_prompt_1, model_name)

    if not contributions:
        LOGGER.warning("No contributions extracted (JSON parse failed or empty response).")
        return {}, {}

    LOGGER.debug(f"Extracted contributions: {json.dumps(contributions, indent=2)}")

    # Step 2: Generate queries for all contributions at once
    LOGGER.debug("Step 2: Generating queries for all contributions...")

    contributions_lines = []
    for dimension, statements in contributions.items():
        if not isinstance(statements, list) or not statements:
            continue
        contributions_lines.append(f"{dimension}:")
        for s in statements:
            contributions_lines.append(f"  - {s}")

    contributions_summary = "\n".join(contributions_lines)

    if not contributions_summary:
        LOGGER.warning("No valid contribution statements found.")
        return {}, contributions

    LOGGER.debug(f"Contributions summary:\n{contributions_summary}")

    system_prompt = render_prompt("query_gen_system.jinja2", n_queries=n_queries)
    user_prompt_2 = render_prompt("query_gen_user.jinja2", idea_text=idea_text, contributions_summary=contributions_summary)

    response = call_llm_json(user_prompt_2, system_prompt, model_name)

    generated_queries = {}
    if response and 'queries' in response:
        queries = response['queries']
        LOGGER.debug(f"Generated {len(queries)} queries: {queries}")
        generated_queries["all_contributions"] = queries
    else:
        LOGGER.warning("Failed to generate queries from contributions.")

    return generated_queries, contributions


async def retrieve_candidates_for_idea(idea_text: str, llm_engine: str, top_k: int = 10,
                                       cutoff_date: Optional[str] = None, search_semaphore: Optional[asyncio.Semaphore] = None,
                                       use_semantic_scholar: bool = False,
                                       min_relevance_score: float = 0.0,
                                       max_contributions: int = 3, n_queries: int = 3,
                                       idea_title: Optional[str] = None,
                                       search_max_retries: int = 3,
                                       search_retry_delay: float = 2.0) -> Tuple[List[Dict], Dict]:

    candidates = []

    # 1. Extract search queries using two-step method
    search_queries, contributions = await asyncio.to_thread(
        extract_search_queries,
        idea_text=idea_text,
        model_name=llm_engine,
        max_contributions=max_contributions,
        n_queries=n_queries,
    )

    all_queries = [q for queries in search_queries.values() for q in queries]
    LOGGER.debug(f"Extracted {len(all_queries)} search queries for idea.")

    # 2. Search papers
    query_outcomes = []
    if all_queries:
        async def search_task(q):
            for attempt in range(1, search_max_retries + 1):
                try:
                    async def _do():
                        if use_semantic_scholar:
                            return await asyncio.to_thread(SEMANTIC_API.search_papers, q, max_results=top_k)
                        else:
                            return await asyncio.to_thread(search_papers, q)

                    if search_semaphore:
                        async with search_semaphore:
                            raw = await _do()
                    else:
                        raw = await _do()

                    if use_semantic_scholar:
                        papers = list(raw.values())
                        for p in papers:
                            if 'text' not in p:
                                p['text'] = p.get('abstract', '')
                            if 'corpus_id' not in p:
                                p['corpus_id'] = p.get('paperId')
                            if 'url' not in p:
                                p['url'] = f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}"
                    else:
                        papers = raw.get("doc_collection", {}).get("documents", [])

                    status = "success" if papers else "empty"
                    return papers, status, attempt
                except PaperFinderAPIError as e:
                    if attempt < search_max_retries:
                        LOGGER.warning(f"Query '{q}' API error (attempt {attempt}/{search_max_retries}): {e}. Retrying in {search_retry_delay}s...")
                        await asyncio.sleep(search_retry_delay)
                    else:
                        LOGGER.error(f"Query '{q}' failed after {search_max_retries} attempt(s): {e}")
            return [], "failed", search_max_retries

        raw_results = await asyncio.gather(*[search_task(q) for q in all_queries], return_exceptions=True)

        results_papers = []
        for q, result in zip(all_queries, raw_results):
            if isinstance(result, Exception):
                LOGGER.error(f"Unexpected error for query '{q}': {result}")
                results_papers.append([])
                query_outcomes.append({"query": q, "status": "failed", "papers_found": 0, "attempts": 0})
            else:
                papers, status, attempts = result
                results_papers.append(papers)
                query_outcomes.append({"query": q, "status": status, "papers_found": len(papers), "attempts": attempts})

        for q, papers in zip(all_queries, results_papers):
            LOGGER.debug(f"Query: '{q}' retrieved {len(papers)} candidates.")
            for p in papers:
                p['source_query'] = q
            candidates.extend(papers)

    # 3–5. Filter, deduplicate, sort, and select
    final_candidates, skipped_same_title, per_query_stats = _filter_and_select_candidates(
        candidates, top_k, idea_text, idea_title, min_relevance_score, cutoff_date
    )

    for o in query_outcomes:
        stats = per_query_stats.get(o["query"], {})
        o.update(stats)

    debug_data = {
        "contributions": contributions,
        "search_queries_dict": search_queries,
        "candidates": final_candidates,
        "skipped_same_title": skipped_same_title,
        "query_outcomes": query_outcomes,
    }

    return final_candidates, debug_data


async def process_problem(
    problem_id,
    data: Dict,
    cfg: RetrievalConfig,
    retrieval_cache: Dict,
    skipped_duplicates_log: List[Dict],
    query_outcomes_log: List[Dict],
    search_semaphore: asyncio.Semaphore,
    output_path: str,
):
    topic = data.get('context')
    ideas = data.get('ideas', {})
    metadata = data.get('metadata') or {}

    # Pointwise format: each instance has a single 'idea' string instead of an 'ideas' dict.
    # The accuracy_test.py lookup for pointwise is flat: retrieval_cache[instance_id] = result.
    is_pointwise = not ideas and 'idea' in data
    if is_pointwise:
        ideas = {'idea': data['idea']}

    str_problem = str(problem_id)

    # Build a set of idea keys that had ≥1 failed query in the persisted outcomes log.
    # Scanned once here so per-idea checks below are O(1).
    ideas_with_past_failures: set = set()
    if cfg.retry_failed_queries:
        ideas_with_past_failures = {
            str(o["idea_key"])
            for o in query_outcomes_log
            if str(o["problem_id"]) == str_problem and o["status"] in ("failed", "empty")
        }

    if is_pointwise:
        # For pointwise, check if the flat cache entry already exists and has candidates.
        existing = retrieval_cache.get(str_problem)
        if isinstance(existing, dict) and existing.get('candidates'):
            if "idea" not in ideas_with_past_failures:
                return
    else:
        if str_problem not in retrieval_cache:
            retrieval_cache[str_problem] = {}

    cached_problem = retrieval_cache[str_problem] if not is_pointwise else {}

    ideas_to_retrieve = {}
    for key, text in ideas.items():
        str_key = str(key)
        should_retrieve = False
        if str_key not in cached_problem:
            should_retrieve = True
        elif not cached_problem[str_key].get("candidates"):
            LOGGER.info(f"Idea {str_key} in problem {problem_id} has 0 candidates in cache. Retrying retrieval.")
            should_retrieve = True
        elif str_key in ideas_with_past_failures:
            n_failed = sum(
                1 for o in query_outcomes_log
                if str(o["problem_id"]) == str_problem and str(o["idea_key"]) == str_key and o["status"] in ("failed", "empty")
            )
            LOGGER.info(f"Idea {str_key} in problem {problem_id} had {n_failed} failed/empty query/queries in previous run. Retrying retrieval.")
            should_retrieve = True

        if should_retrieve:
            ideas_to_retrieve[key] = text

    if not ideas_to_retrieve:
        return

    n_total = len(ideas_to_retrieve)
    completed = 0

    def _get_idea_title(key):
        str_key = str(key)
        if is_pointwise:
            return metadata.get('title') if isinstance(metadata, dict) else None
        idea_meta = metadata.get(key)
        if idea_meta is None:
            idea_meta = metadata.get(str_key)
        return idea_meta.get('title') if isinstance(idea_meta, dict) else None

    async def retrieve_and_cache(idea_key, idea_text, idea_title=None):
        nonlocal completed
        completed += 1
        LOGGER.debug(f"Problem {problem_id} [{completed}/{n_total}] — idea {idea_key}")
        candidates, debug_data = await retrieve_candidates_for_idea(
            idea_text,
            cfg.llm_engine,
            top_k=cfg.top_k_candidates,
            cutoff_date=cfg.cutoff_date,
            search_semaphore=search_semaphore,
            use_semantic_scholar=cfg.use_semantic_scholar,
            min_relevance_score=cfg.min_relevance_score,
            max_contributions=cfg.max_contributions,
            n_queries=cfg.n_queries,
            idea_title=idea_title,
            search_max_retries=cfg.search_max_retries,
            search_retry_delay=cfg.search_retry_delay,
        )

        # Save debug info
        debug_dir = os.path.join(output_path, "retrieval_debug", str(problem_id))
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f"{idea_key}_retrieval.md")
        with open(debug_file, "w") as f:
            f.write(format_retrieval_debug_info(str(idea_key), idea_text, topic if topic else "N/A", debug_data))

        # Replace any prior entries for this (problem_id, idea_key) with the current run's outcomes
        str_pid, str_ik = str(problem_id), str(idea_key)
        query_outcomes_log[:] = [
            o for o in query_outcomes_log
            if not (str(o["problem_id"]) == str_pid and str(o["idea_key"]) == str_ik)
        ]
        for o in debug_data.get('query_outcomes', []):
            query_outcomes_log.append({"problem_id": problem_id, "idea_key": idea_key, **o})

        # Record any skipped-duplicate hits in the global log
        for c in debug_data.get('skipped_same_title', []):
            skipped_duplicates_log.append({
                'problem_id': problem_id,
                'idea_key': idea_key,
                'candidate_title': c.get('title', 'Unknown'),
                'candidate_url': c.get('url', '#'),
                'relevance_score': (c.get('relevance_judgement') or {}).get('relevance_score', 'N/A'),
                'year': c.get('year', 'N/A'),
                'source_query': c.get('source_query', ''),
            })

        if not candidates:
            LOGGER.warning(f"No candidates retrieved for idea {idea_key} in problem {problem_id}. Skipping cache save.")
            return

        formatted_candidates = select_and_format_candidates(candidates)
        result_val = {
            "title": str(idea_key),
            "text": idea_text,
            "related_work": formatted_candidates,
            "candidates": candidates,
            "search_queries_dict": debug_data.get("search_queries_dict", {}),
            "contributions": debug_data.get("contributions", {}),
        }

        if is_pointwise:
            # Flat cache structure: retrieval_cache[instance_id] = result (matches accuracy_test.py lookup)
            retrieval_cache[str_problem] = result_val
        else:
            retrieval_cache[str_problem][str(idea_key)] = result_val

    tasks = [retrieve_and_cache(k, v, idea_title=_get_idea_title(k)) for k, v in ideas_to_retrieve.items()]
    await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser(description="Retrieve candidates for ideas and save to cache.")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file.")
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        raw = yaml.safe_load(f) or {}

    if 'test_inputs' not in raw:
        raise ValueError("Config must specify 'test_inputs'.")

    cfg_field_names = {f.name for f in fields(RetrievalConfig)}
    cfg = RetrievalConfig(**{k: v for k, v in raw.items() if k in cfg_field_names})

    global LOGGER
    if _setup_logger is not None:
        output_path_for_log = os.path.dirname(os.path.abspath(cfg.output_file))
        LOGGER = _setup_logger(output_dir=output_path_for_log, console_level="INFO")
        _set_common_logger(LOGGER)
    LOGGER.info(f"Loaded config from {cli_args.config}")

    if GLOBAL_COST_TRACKER is not None:
        GLOBAL_COST_TRACKER.reset()

    retrieval_cache = load_retrieval_cache(cfg.output_file) if os.path.exists(cfg.output_file) else {}
    LOGGER.info(f"Starting retrieval process. Output will be saved to: {cfg.output_file}")

    skipped_duplicates_log: List[Dict] = []
    output_path = os.path.dirname(os.path.abspath(cfg.output_file))
    outcomes_log_path = os.path.join(output_path, "query_outcomes_log.json")
    query_outcomes_log: List[Dict] = load_query_outcomes_log(outcomes_log_path)
    search_semaphore = asyncio.Semaphore(cfg.max_search_workers)

    inputs = {}
    all_tasks = []
    try:
        inputs = read_inputs(cfg.test_inputs)
        for i, (problem_id, data) in enumerate(inputs.items()):
            if cfg.nr_examples is not None and i >= cfg.nr_examples:
                break
            all_tasks.append(
                process_problem(problem_id, data, cfg, retrieval_cache, skipped_duplicates_log, query_outcomes_log, search_semaphore, output_path)
            )
    except Exception as e:
        LOGGER.error(f"Failed to read or process {cfg.test_inputs}: {e}")

    def save_progress():
        save_retrieval_cache(cfg.output_file, retrieval_cache)
        save_query_outcomes_log(outcomes_log_path, query_outcomes_log)
        LOGGER.info(f"Saved progress to {cfg.output_file}")

    await run_in_chunks(all_tasks, cfg.chunk_size, on_chunk_done=save_progress)

    save_retrieval_cache(cfg.output_file, retrieval_cache)
    save_query_outcomes_log(outcomes_log_path, query_outcomes_log)
    LOGGER.info(f"Retrieval complete. Cache saved to {cfg.output_file}")

    stats = PAPER_FINDER_STATS.get_stats()
    LOGGER.info(f"Paper Finder Stats: {stats}")

    if inputs:
        generate_cache_status_report(retrieval_cache, inputs, cfg.output_file, cfg)

    write_skipped_duplicates_log(skipped_duplicates_log, output_path)
    write_query_status_report(query_outcomes_log, output_path)

    if GLOBAL_COST_TRACKER is not None:
        cost_report = GLOBAL_COST_TRACKER.get_report()
        cost_report["instances_processed"] = len(inputs)

        cost_json_path = os.path.join(output_path, "retrieval_cost_report.json")
        with open(cost_json_path, "w") as f:
            json.dump(cost_report, f, indent=2)

        if _write_cost_report_md is not None:
            cost_md_path = os.path.join(output_path, "retrieval_cost_report.md")
            _write_cost_report_md(cost_report, cost_md_path, title=f"Cost Report — Retrieval (`{cfg.llm_engine}`)")
            LOGGER.info(
                f"Estimated retrieval cost: ${cost_report['total_cost_usd']:.4f} "
                f"({cost_report['total_calls']} LLM calls, "
                f"{cost_report['total_input_tokens']:,} input + "
                f"{cost_report['total_output_tokens']:,} output tokens) — "
                f"saved to {cost_md_path}"
            )
        else:
            LOGGER.info(
                f"Estimated retrieval cost: ${cost_report['total_cost_usd']:.4f} "
                f"({cost_report['total_calls']} LLM calls, "
                f"{cost_report['total_input_tokens']:,} input + "
                f"{cost_report['total_output_tokens']:,} output tokens) — "
                f"saved to {cost_json_path}"
            )

if __name__ == '__main__':
    asyncio.run(main())
