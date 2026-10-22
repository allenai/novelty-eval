#!/usr/bin/env python3
"""retrieval_common.py — Method-agnostic retrieval utilities.

Shared by every retrieval-cache producer (paper-finder / Semantic Scholar in
retrieve_candidates.py, and the GPT web-search novelty judge in
web_search_novelty_judge.py) so that cache handling, candidate
filtering/selection, and prompt formatting stay identical across backends.
"""
import asyncio
import hashlib
import math
import os
import sys
import json
import yaml
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional, Tuple

from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

# Add src to path to allow `from utils import ...` when imported standalone.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import LOGGER as _BASE_LOGGER

# Module-level logger. main() entry points may swap this for a file-backed
# logger via set_logger() so the moved helpers log to the same run.log.
LOGGER = _BASE_LOGGER


def set_logger(logger) -> None:
    """Point the shared helpers at a configured logger (e.g. file-backed)."""
    global LOGGER
    LOGGER = logger


_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_jinja_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), keep_trailing_newline=True)


def render_prompt(template_name: str, **kwargs) -> str:
    return _jinja_env.get_template(template_name).render(**kwargs)


# ---------------------------------------------------------------------------
# Content-addressed idea identity (for recycling caches across datasets)
# ---------------------------------------------------------------------------

def normalize_idea_text(idea_text: str) -> str:
    """Canonical normalization for deciding when two idea texts are "the same".

    The single source of truth for idea identity: `idea_identity` (the content-addressed
    cache key), candidate dedup, and any dataset-join keyed on idea text MUST route
    through this so "same idea" means the same thing everywhere. Diverging normalizations
    (e.g. a join that strips but doesn't lower-case) silently make the join and the cache
    disagree, so projection misses on ideas that differ only in case/whitespace.
    """
    return (idea_text or "").strip().lower()


def idea_identity(idea_text: str, cutoff_date: Optional[str] = None) -> str:
    """Stable, content-addressed id for an idea's retrieval result.

    Retrieval depends only on the idea text and the cutoff_date — nothing about a
    dataset's positional `problem_id` enters the prompt. Two datasets that share an
    idea therefore share this id, so a master store keyed by `idea_identity` can be
    recycled across datasets (pairwise vs pointwise, same-positives/different-negatives)
    regardless of how each dataset numbers its instances.

    The text is normalized via `normalize_idea_text` (`strip().lower()`) — the same
    normalization used by candidate dedup and dataset joins — so "same idea" means the
    same thing everywhere. `cutoff_date` is folded in because it changes which
    candidates survive date filtering.
    """
    norm = normalize_idea_text(idea_text)
    payload = f"{cutoff_date or ''}\n{norm}".encode("utf-8")
    return "idea_" + hashlib.sha1(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Shared config base
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfigBase:
    """Fields common to every retrieval backend."""
    test_inputs: str
    output_file: Optional[str] = None
    llm_engine: str = 'gpt-5.2'
    cutoff_date: Optional[str] = None
    top_k_candidates: int = 10
    min_relevance_score: float = 0.0
    nr_examples: Optional[int] = None
    chunk_size: int = 50

    def __post_init__(self):
        if not self.output_file:
            self.output_file = os.path.join(
                os.path.dirname(os.path.abspath(self.test_inputs)), "retrieval_cache.json"
            )


# ---------------------------------------------------------------------------
# Cache + outcome-log I/O
# ---------------------------------------------------------------------------

def load_retrieval_cache(cache_path: str) -> Dict:
    if not cache_path:
        return {}

    if not cache_path.endswith('.json'):
        cache_path += '.json'

    if os.path.exists(cache_path):
        LOGGER.info(f"Loading retrieval cache from {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            LOGGER.error(f"Failed to load cache from {cache_path}: {e}")
            return {}
    return {}


def save_retrieval_cache(cache_path: str, cache: Dict):
    if not cache_path:
        return

    if not cache_path.endswith('.json'):
        cache_path += '.json'

    LOGGER.info(f"Saving retrieval cache to {cache_path}")
    try:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        LOGGER.error(f"Failed to save cache to {cache_path}: {e}")


def load_query_outcomes_log(path: str) -> List[Dict]:
    if os.path.exists(path):
        LOGGER.info(f"Loading query outcomes log from {path}")
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            LOGGER.error(f"Failed to load query outcomes log from {path}: {e}")
    return []


def save_query_outcomes_log(path: str, log: List[Dict]):
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        LOGGER.error(f"Failed to save query outcomes log to {path}: {e}")


# ---------------------------------------------------------------------------
# Candidate filtering / selection
# ---------------------------------------------------------------------------

def _filter_by_date(candidates: List[Dict], cutoff_date: str) -> List[Dict]:
    try:
        cutoff_date_obj = datetime.strptime(cutoff_date, '%Y-%m-%d').date()
        cutoff_year = cutoff_date_obj.year
        filtered_by_date = []

        total_before_filter = len(candidates)
        filtered_by_pub_date = 0
        filtered_by_year = 0
        skipped_after_cutoff_pub_date = 0
        skipped_after_cutoff_year = 0
        skipped_no_date = 0

        for c in candidates:
            pub_date_str = c.get('publication_date')
            included = False
            has_valid_date = False

            # Try to use publication_date field first
            if pub_date_str:
                try:
                    pub_date_obj = datetime.strptime(pub_date_str, '%Y-%m-%d').date()
                    has_valid_date = True  # Mark that we successfully parsed a date
                    if pub_date_obj < cutoff_date_obj:
                        filtered_by_date.append(c)
                        included = True
                        filtered_by_pub_date += 1
                    else:
                        skipped_after_cutoff_pub_date += 1
                        LOGGER.debug(f"Skipped candidate '{c.get('title', 'Unknown')}' with publication_date {pub_date_str} (after cutoff {cutoff_date})")
                except ValueError:
                    LOGGER.warning(f"Could not parse publication_date '{pub_date_str}' for candidate '{c.get('title', 'Unknown')}'. Trying fallback...")

            # If no valid publication_date was parsed, try year field as fallback
            if not has_valid_date and not included:
                year = c.get('year')
                if year:
                    try:
                        year_int = int(year)
                        if year_int < cutoff_year:
                            filtered_by_date.append(c)
                            included = True
                            filtered_by_year += 1
                        else:
                            skipped_after_cutoff_year += 1
                            LOGGER.debug(f"Skipped candidate '{c.get('title', 'Unknown')}' with year {year_int} (>= cutoff year {cutoff_year})")
                        # Note: Don't include year == cutoff_year since we can't verify exact date
                    except (ValueError, TypeError):
                        LOGGER.debug(f"Could not parse year '{year}' for candidate '{c.get('title', 'Unknown')}'")

            # If still not included and no date info, warn and skip
            if not included and not has_valid_date and not c.get('year'):
                LOGGER.debug(f"Candidate '{c.get('title', 'Unknown')}' has no publication_date or year field. Skipping.")
                skipped_no_date += 1

        LOGGER.debug(f"Date filter ({cutoff_date}): {total_before_filter} in → {len(filtered_by_date)} kept "
                     f"({filtered_by_pub_date} by pub_date, {filtered_by_year} by year fallback; "
                     f"{skipped_after_cutoff_pub_date + skipped_after_cutoff_year} after cutoff, {skipped_no_date} no date)")
        return filtered_by_date
    except ValueError:
        LOGGER.error(f"Invalid cutoff_date format '{cutoff_date}'. Expected YYYY-MM-DD format.")
        return []


def _select_candidates_per_query(unique_candidates: List[Dict], top_k: int) -> List[Dict]:
    """Select up to top_k candidates with per-query quota and round-robin fill."""
    query_groups: Dict[str, List[Dict]] = defaultdict(list)
    for c in unique_candidates:
        query_groups[c.get('source_query', '')].append(c)

    n_per_query = math.floor(top_k / len(query_groups)) if query_groups else top_k

    selected: List[Dict] = []
    surplus: Dict[str, List[Dict]] = {}
    for q, papers in query_groups.items():
        selected.extend(papers[:n_per_query])
        surplus[q] = list(papers[n_per_query:])

    # Round-robin fill to reach top_k
    query_keys = list(query_groups.keys())
    i = 0
    while len(selected) < top_k:
        advanced = False
        for _ in range(len(query_keys)):
            q = query_keys[i % len(query_keys)]
            i += 1
            if surplus[q]:
                selected.append(surplus[q].pop(0))
                advanced = True
                break
        if not advanced:
            break

    return selected[:top_k]


def _deduplicate_candidates(
    candidates: List[Dict],
    idea_text: str,
    idea_title: Optional[str],
    min_relevance_score: float,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    filtered_candidates = []
    skipped_same_title = []
    seen_texts = set()
    drop_counts: Dict[str, int] = {"below_threshold": 0, "same_title": 0, "same_text": 0, "duplicate_text": 0}
    idea_text_normalized = idea_text.strip().lower()
    idea_title_normalized = idea_title.strip().lower() if idea_title else None

    for c in candidates:
        relevance_score = (c.get('relevance_judgement') or {}).get('relevance_score', 0)

        if min_relevance_score > 0 and relevance_score < min_relevance_score:
            LOGGER.debug(f"Skipping candidate '{c.get('title', 'Unknown')}' with relevance score {relevance_score} (below min {min_relevance_score})")
            drop_counts["below_threshold"] += 1
            continue

        if idea_title_normalized:
            candidate_title_normalized = c.get('title', '').strip().lower()
            if candidate_title_normalized and candidate_title_normalized == idea_title_normalized:
                LOGGER.debug(f"Skipping candidate '{c.get('title', 'Unknown')}' — title matches idea title")
                skipped_same_title.append(c)
                drop_counts["same_title"] += 1
                continue

        candidate_text = (c.get('abstract') or c.get('text') or '').strip().lower()
        if candidate_text and candidate_text == idea_text_normalized:
            LOGGER.debug(f"Skipping candidate '{c.get('title', 'Unknown')}' with text matching the idea text")
            drop_counts["same_text"] += 1
            continue

        if candidate_text and candidate_text in seen_texts:
            LOGGER.debug(f"Skipping candidate '{c.get('title', 'Unknown')}' with duplicate text")
            drop_counts["duplicate_text"] += 1
            continue

        if candidate_text:
            seen_texts.add(candidate_text)
        filtered_candidates.append(c)

    return filtered_candidates, skipped_same_title, drop_counts


def _filter_and_select_candidates(
    candidates: List[Dict],
    top_k: int,
    idea_text: str = "",
    idea_title: Optional[str] = None,
    min_relevance_score: float = 0.0,
    cutoff_date: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], Dict[str, Dict]]:
    """Filter, deduplicate, sort, and select candidates.

    Returns (final_candidates, skipped_same_title, per_query_stats).
    """
    def _counts_by_query(lst: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for c in lst:
            counts[c.get('source_query', '(unknown)')] += 1
        return counts

    all_queries = list(_counts_by_query(candidates).keys())
    raw_by_query = _counts_by_query(candidates)
    n_raw = len(candidates)

    if cutoff_date is not None:
        candidates = _filter_by_date(candidates, cutoff_date)
    after_date_by_query = _counts_by_query(candidates)
    n_after_date = len(candidates)

    candidates, skipped_same_title, drop_counts = _deduplicate_candidates(
        candidates, idea_text, idea_title, min_relevance_score
    )
    after_dedup_by_query = _counts_by_query(candidates)
    n_after_dedup = len(candidates)

    candidates.sort(key=lambda c: (c.get('relevance_judgement') or {}).get('relevance_score', 0), reverse=True)

    unique_candidates = []
    seen_ids = set()
    for c in candidates:
        c_id = c.get('corpus_id') or c.get('paperId')
        if c_id and c_id not in seen_ids:
            unique_candidates.append(c)
            seen_ids.add(c_id)
    after_id_dedup_by_query = _counts_by_query(unique_candidates)
    n_after_id_dedup = len(unique_candidates)

    final_candidates = _select_candidates_per_query(unique_candidates, top_k)
    selected_by_query = _counts_by_query(final_candidates)

    # Per-query breakdown
    q_lines = []
    for q in all_queries:
        label = f"'{q[:60]}'"
        chain = [str(raw_by_query[q])]
        if cutoff_date is not None:
            chain.append(str(after_date_by_query.get(q, 0)))
        if n_after_date != n_after_dedup:
            chain.append(str(after_dedup_by_query.get(q, 0)))
        if n_after_dedup != n_after_id_dedup:
            chain.append(str(after_id_dedup_by_query.get(q, 0)))
        chain.append(f"{selected_by_query.get(q, 0)} selected")
        q_lines.append(f"  {label}: {' → '.join(chain)}")

    # Overall summary
    overall_parts = [f"{n_raw} raw"]
    if cutoff_date is not None:
        overall_parts.append(f"{n_after_date} after date (-{n_raw - n_after_date})")
    if n_after_date != n_after_dedup:
        reasons = [f"-{v} {k.replace('_', ' ')}" for k, v in drop_counts.items() if v > 0]
        overall_parts.append(f"{n_after_dedup} after dedup ({', '.join(reasons)})")
    if n_after_dedup != n_after_id_dedup:
        overall_parts.append(f"{n_after_id_dedup} after ID dedup (-{n_after_dedup - n_after_id_dedup})")
    suffix = f"{len(final_candidates)}/{top_k} selected"
    if len(final_candidates) < top_k:
        suffix += " (insufficient)"
    overall_parts.append(suffix)

    LOGGER.debug("Candidates: " + " → ".join(overall_parts) + "\n" + "\n".join(q_lines))

    if len(final_candidates) == 0:
        LOGGER.error("No candidates found for idea.")

    per_query_stats: Dict[str, Dict] = {}
    for q in all_queries:
        per_query_stats[q] = {
            "raw": raw_by_query.get(q, 0),
            "after_date": after_date_by_query.get(q, 0) if cutoff_date is not None else None,
            "after_dedup": after_dedup_by_query.get(q, 0),
            "after_id_dedup": after_id_dedup_by_query.get(q, 0),
            "selected": selected_by_query.get(q, 0),
        }

    return final_candidates, skipped_same_title, per_query_stats


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_papers_for_prompt(doc_list):
    related_work_blocks = []
    for rw_idx, entry in enumerate(doc_list, 1):
        title = entry.get('title', 'Unknown Title')
        retrieval_query = entry.get('source_query', '')
        why_relevant = (entry.get('relevance_judgement') or {}).get('relevance_summary', '')
        abstract = entry.get('abstract') or entry.get('text', 'No abstract available')

        rw_tmp = (
            f"[metadata-{rw_idx}]\n"
            f"title: {title}\n"
            f"retrieval_query: {retrieval_query}\n"
            f"why_relevant: {why_relevant}\n"
            f"[abstract-{rw_idx}]\n{abstract}\n"
        )
        related_work_blocks.append(rw_tmp)

    return '\n'.join(related_work_blocks)


def select_and_format_candidates(
    candidates: List[Dict],
    top_k: Optional[int] = None,
) -> str:
    if top_k is not None:
        candidates, _, __ = _filter_and_select_candidates(candidates, top_k)
    return format_papers_for_prompt(candidates)


# ---------------------------------------------------------------------------
# I/O + async helpers
# ---------------------------------------------------------------------------

def read_inputs(input_path: str) -> Dict[str, Any]:
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


async def run_in_chunks(tasks: List, chunk_size: int, on_chunk_done: Optional[Callable] = None):
    with tqdm(total=len(tasks), desc="Processing papers") as pbar:
        async def tracked(coro):
            result = await coro
            pbar.update(1)
            return result

        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            await asyncio.gather(*[tracked(t) for t in chunk])
            if on_chunk_done:
                on_chunk_done()
