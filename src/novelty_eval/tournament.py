"""
tournament.py — Async tournament runners for idea evaluation.

Two modes are supported:
  - run_rr_tournament    : Round-robin (every unordered pair, MEC+BPC)
  - run_swiss_tournament : Swiss (one MEC+BPC match per pair per round)

All modes delegate scoring to scoring.py.
"""
import asyncio
import math
import random
import os
import sys
from itertools import combinations
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import LOGGER
from novelty_eval.scoring import change_winner_to_score

# judge_idea is imported lazily inside each function to avoid a circular import
# with judge.py (which also imports from this module).


def _init_state(ideas: Dict, evaluation_criteria: Dict):
    """Return common tournament state structures."""
    idea_keys = list(ideas.keys())
    key_to_idx = {key: idx for idx, key in enumerate(idea_keys)}
    n_ideas = len(idea_keys)
    dimensions = list(evaluation_criteria.keys())
    elo_scores = [0.0] * n_ideas
    dimensional_elo_scores = {dim: [0.0] * n_ideas for dim in dimensions}
    comparisons: List[Dict] = []
    return idea_keys, key_to_idx, n_ideas, dimensions, elo_scores, dimensional_elo_scores, comparisons


def _award_bye(
    p1_idx: int,
    idea_keys: List,
    dimensions: List[str],
    elo_scores: List[float],
    dimensional_elo_scores: Dict[str, List[float]],
) -> None:
    LOGGER.debug(f"Idea {idea_keys[p1_idx]} gets a bye.")
    elo_scores[p1_idx] += float(len(dimensions))
    for dim in dimensions:
        dimensional_elo_scores[dim][p1_idx] += 1.0


def _pair_round_tasks_swiss(
    sorted_indices: List[int],
    n_ideas: int,
    idea_keys: List,
    ideas: Dict,
    played_pairs: set,
    elo_scores: List[float],
    dimensional_elo_scores: Dict[str, List[float]],
    dimensions: List[str],
    llm_engine: str,
    semaphore: asyncio.Semaphore,
    output_path: str,
    problem_name: str,
    evaluation_criteria: Dict,
    effort: str,
    mec_k: int = 1,
    bidirectional: bool = True,
    failure_log=None,
    run_label: str = "",
    use_prompt_caching: bool = False,
    use_batch_api: bool = False,
    pairwise_tip: str | None = None,
    tools: list | None = None,
    cutoff_date: str | None = None,
):
    """Build the task list for one Swiss round.

    Each match calls judge_idea_mec once, which internally runs k fwd + k rev
    calls (BPC+MEC). Returns a list of coroutines.
    """
    from novelty_eval.judge import judge_idea_mec

    round_tasks = []
    used = [False] * n_ideas
    idx_ptr = 0

    while idx_ptr < n_ideas:
        p1_idx = sorted_indices[idx_ptr]
        if used[p1_idx]:
            idx_ptr += 1
            continue

        p2_idx = -1
        # Prefer an opponent not yet played
        for offset in range(1, n_ideas - idx_ptr):
            candidate_idx = sorted_indices[idx_ptr + offset]
            if not used[candidate_idx]:
                c_id1 = idea_keys[p1_idx]
                c_id2 = idea_keys[candidate_idx]
                if tuple(sorted((c_id1, c_id2))) not in played_pairs:
                    p2_idx = candidate_idx
                    break

        # Fallback: any available opponent
        if p2_idx == -1:
            for offset in range(1, n_ideas - idx_ptr):
                candidate_idx = sorted_indices[idx_ptr + offset]
                if not used[candidate_idx]:
                    p2_idx = candidate_idx
                    break

        if p2_idx != -1:
            used[p1_idx] = True
            used[p2_idx] = True
            id1 = idea_keys[p1_idx]
            id2 = idea_keys[p2_idx]
            played_pairs.add(tuple(sorted((id1, id2))))
            round_tasks.append(
                judge_idea_mec(id1, id2, ideas[id1], ideas[id2], llm_engine,
                               semaphore, output_path, problem_name, evaluation_criteria,
                               effort=effort, mec_k=mec_k, bidirectional=bidirectional,
                               failure_log=failure_log, run_label=run_label,
                               use_prompt_caching=use_prompt_caching,
                               use_batch_api=use_batch_api,
                               pairwise_tip=pairwise_tip,
                               tools=tools,
                               cutoff_date=cutoff_date))
        else:
            used[p1_idx] = True
            _award_bye(p1_idx, idea_keys, dimensions, elo_scores, dimensional_elo_scores)

        idx_ptr += 1

    return round_tasks


# ---------------------------------------------------------------------------
# Public tournament runners
# ---------------------------------------------------------------------------

async def run_rr_tournament(
    ideas: Dict,
    llm_engine: str,
    semaphore: asyncio.Semaphore,
    output_path: str,
    problem_name: str,
    evaluation_criteria: Dict,
    effort: str = "none",
    mec_k: int = 1,
    bidirectional: bool = True,
    failure_log=None,
    run_label: str = "",
    use_prompt_caching: bool = False,
    use_batch_api: bool = False,
    pairwise_tip: str | None = None,
    tools: list | None = None,
    cutoff_date: str | None = None,
) -> Dict[str, Any] | List[Dict]:
    """Round-robin: every unordered pair is evaluated once via judge_idea_mec,
    which internally runs k fwd + k rev calls (BPC+MEC).
    """
    from novelty_eval.judge import judge_idea_mec

    idea_keys, key_to_idx, n_ideas, dimensions, elo_scores, dimensional_elo_scores, comparisons = (
        _init_state(ideas, evaluation_criteria)
    )

    tasks = [
        judge_idea_mec(i, j, ideas[i], ideas[j], llm_engine, semaphore, output_path,
                       problem_name, evaluation_criteria, effort=effort, mec_k=mec_k,
                       bidirectional=bidirectional, failure_log=failure_log, run_label=run_label,
                       use_prompt_caching=use_prompt_caching, use_batch_api=use_batch_api,
                       pairwise_tip=pairwise_tip, tools=tools, cutoff_date=cutoff_date)
        for i, j in combinations(idea_keys, 2)
    ]
    results = await asyncio.gather(*tasks)

    if use_batch_api:
        # Collect all requests from (ri, rj, requests) tuples
        all_requests = []
        for _, _, requests in results:
            all_requests.extend(requests)
        return all_requests

    first_dim = dimensions[0] if dimensions else None
    for ri, rj, scores in results:
        idx_i, idx_j = key_to_idx[ri], key_to_idx[rj]
        for dim_name in dimensions:
            score = scores.get(dim_name, '')
            elo_scores[idx_i], elo_scores[idx_j] = change_winner_to_score(
                score, elo_scores[idx_i], elo_scores[idx_j])
            dimensional_elo_scores[dim_name][idx_i], dimensional_elo_scores[dim_name][idx_j] = (
                change_winner_to_score(score,
                                       dimensional_elo_scores[dim_name][idx_i],
                                       dimensional_elo_scores[dim_name][idx_j]))
        comparisons.append({'idea_0': str(ri), 'idea_1': str(rj),
                            'winner': scores.get(first_dim) if first_dim else None,
                            'mec_details': scores.get('_mec_details')})

    return _build_output(elo_scores, dimensional_elo_scores, idea_keys, comparisons)


async def run_swiss_tournament(
    ideas: Dict,
    llm_engine: str,
    semaphore: asyncio.Semaphore,
    output_path: str,
    problem_name: str,
    evaluation_criteria: Dict,
    effort: str = "none",
    mec_k: int = 1,
    bidirectional: bool = True,
    failure_log=None,
    run_label: str = "",
    use_prompt_caching: bool = False,
    use_batch_api: bool = False,
    pairwise_tip: str | None = None,
    tools: list | None = None,
    cutoff_date: str | None = None,
) -> Dict[str, Any]:
    """Swiss: one MEC match per pair per round (k fwd + k rev calls via judge_idea_mec)."""

    idea_keys, key_to_idx, n_ideas, dimensions, elo_scores, dimensional_elo_scores, comparisons = (
        _init_state(ideas, evaluation_criteria)
    )

    num_rounds = max(1, math.ceil(math.log2(n_ideas)))
    LOGGER.info(f"Starting Swiss Tournament with {num_rounds} rounds for {n_ideas} ideas.")
    played_pairs: set = set()

    for round_num in range(num_rounds):
        LOGGER.info(f"Round {round_num + 1}/{num_rounds}")
        indices = list(range(n_ideas))
        random.shuffle(indices)
        sorted_indices = sorted(indices, key=lambda idx: elo_scores[idx], reverse=True)

        round_tasks = _pair_round_tasks_swiss(
            sorted_indices, n_ideas, idea_keys, ideas, played_pairs,
            elo_scores, dimensional_elo_scores, dimensions,
            llm_engine, semaphore, output_path, problem_name,
            evaluation_criteria, effort, mec_k=mec_k, bidirectional=bidirectional,
            failure_log=failure_log, run_label=run_label,
            use_prompt_caching=use_prompt_caching,
            pairwise_tip=pairwise_tip,
            tools=tools,
            cutoff_date=cutoff_date,
        )

        if round_tasks:
            results = await asyncio.gather(*round_tasks)
            first_dim = dimensions[0] if dimensions else None
            for i, j, scores in results:
                idx_i, idx_j = key_to_idx[i], key_to_idx[j]
                for dim_name in dimensions:
                    score = scores.get(dim_name, '')
                    elo_scores[idx_i], elo_scores[idx_j] = change_winner_to_score(
                        score, elo_scores[idx_i], elo_scores[idx_j])
                    dimensional_elo_scores[dim_name][idx_i], dimensional_elo_scores[dim_name][idx_j] = (
                        change_winner_to_score(score,
                                               dimensional_elo_scores[dim_name][idx_i],
                                               dimensional_elo_scores[dim_name][idx_j])
                    )
                raw_winner = scores.get(first_dim, None) if first_dim else None
                comparisons.append({'idea_0': str(i), 'idea_1': str(j), 'winner': raw_winner,
                                    'mec_details': scores.get('_mec_details')})

    return _build_output(elo_scores, dimensional_elo_scores, idea_keys, comparisons)



def _build_output(
    elo_scores: List[float],
    dimensional_elo_scores: Dict[str, List[float]],
    idea_keys: List,
    comparisons: List[Dict],
) -> Dict[str, Any]:
    LOGGER.debug(f"Final elo_scores: {elo_scores}")
    LOGGER.debug(f"Dimensional scores: {dimensional_elo_scores}")
    try:
        elo_selected = elo_scores.index(max(elo_scores))
    except (ValueError, IndexError):
        elo_selected = 0
    LOGGER.debug(f"elo_selected: {elo_selected}")
    return {
        "elo_scores": elo_scores,
        "dimensional_elo_scores": dimensional_elo_scores,
        "elo_selected": elo_selected,
        "ideas": idea_keys,
        "comparisons": comparisons,
    }






