"""
scoring.py — Score-update helpers for pairwise comparisons.

All tournament modes (RR, Swiss, Bi-Swiss) share the same logic for
translating an LLM winner verdict into Elo-style score increments and
for aggregating two bidirectional calls into a single winner.
"""
import logging
from typing import Dict, List, Tuple, Any

LOGGER = logging.getLogger(__name__)


def change_winner_to_score(winner, score_1: float, score_2: float) -> Tuple[float, float]:
    """Return updated (score_1, score_2) given a raw LLM winner value.

    winner == 0  →  idea_0 wins  (+1 to score_1)
    winner == 1  →  idea_1 wins  (+1 to score_2)
    anything else (2, None, …) → tie, no change
    """
    winner = int(winner)
    if winner == 0:
        return score_1 + 1, score_2
    if winner == 1:
        return score_1, score_2 + 1
    return score_1, score_2  # tie


def aggregate_bidirectional_winner(
        scores_fwd: Dict[str, Any],
        scores_rev: Dict[str, Any],
        dim_name: str,
) -> int:
    """Determine the aggregated winner for one dimension from two opposite-direction calls.

    Forward call  (A vs B): winner=0 → A wins, winner=1 → B wins
    Reverse call  (B vs A): winner=0 → B wins, winner=1 → A wins

    Returns:
        0  if A (the forward idea_0) wins both directions
        1  if B (the forward idea_1) wins both directions
        2  if the calls disagree (tie)
    """
    w_fwd = int(scores_fwd[dim_name])
    w_rev = int(scores_rev[dim_name])

    # A wins fwd (w_fwd==0) AND A wins rev (w_rev==1, since in reverse A is idea_1)
    if w_fwd == 0 and w_rev == 1:
        return 0
    # B wins fwd (w_fwd==1) AND B wins rev (w_rev==0)
    if w_fwd == 1 and w_rev == 0:
        return 1
    return 2  # disagreement → tie


def group_results_by_pair(results) -> Dict[tuple, list]:
    """Group a flat list of (i, j, scores) triples by unordered {i, j} pair key."""
    pair_results: Dict[tuple, list] = {}
    for ri, rj, rscores in results:
        pair_key = tuple(sorted((str(ri), str(rj))))
        pair_results.setdefault(pair_key, []).append((ri, rj, rscores))
    return pair_results


def aggregate_unidirectional_comparisons(
    comparisons_by_problem: Dict[str, List[Dict]],
    evaluation_criteria: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Aggregate raw unidirectional pairwise comparisons into ELO score arrays.

    Input format:
        {problem_name: [{"idea_i": str, "idea_j": str, "scores": {dim: winner_int, ...}}, ...]}

    Output format:
        {problem_name: {"overall": [...], dim: [...], "elo_selected": int, "ideas": [str, ...]}}

    Dimensions named "overall_winner" and "thinking_process" are excluded.
    Unparsable winner values (non-integer, None) are skipped for that dimension.

    Args:
        comparisons_by_problem: Raw pairwise comparison results keyed by problem name.
        evaluation_criteria: Optional criteria dict used during judging.  When provided,
            a warning is emitted if the dimensions inferred from the first comparison do
            not exactly match the criteria keys (extra or missing dimensions are listed).
    """
    final_results: Dict[str, Any] = {}

    for problem_name, comparisons in comparisons_by_problem.items():
        ideas: set = set()
        for comp in comparisons:
            ideas.add(comp["idea_i"])
            ideas.add(comp["idea_j"])

        idea_keys = sorted(ideas)
        key_to_idx = {key: idx for idx, key in enumerate(idea_keys)}
        n_ideas = len(idea_keys)

        elo_scores = [0.0] * n_ideas
        dimensions: List[str] = (
            [k for k in comparisons[0]["scores"] if k not in ("overall_winner", "thinking_process")]
            if comparisons else []
        )
        if evaluation_criteria is not None:
            expected = set(evaluation_criteria.keys())
            actual = set(dimensions)
            if actual != expected:
                extra = actual - expected
                missing = expected - actual
                parts = []
                if extra:
                    parts.append(f"unexpected dimensions in comparisons: {sorted(extra)}")
                if missing:
                    parts.append(f"dimensions missing from comparisons: {sorted(missing)}")
                LOGGER.warning(
                    "[%s] Comparison dimensions do not match evaluation_criteria — %s",
                    problem_name,
                    "; ".join(parts),
                )
        dimensional_elo_scores: Dict[str, List[float]] = {dim: [0.0] * n_ideas for dim in dimensions}

        for comp in comparisons:
            idx_i = key_to_idx[comp["idea_i"]]
            idx_j = key_to_idx[comp["idea_j"]]
            scores = comp["scores"]
            for dim in dimensions:
                try:
                    val = int(scores.get(dim))
                except (TypeError, ValueError):
                    continue
                dimensional_elo_scores[dim][idx_i], dimensional_elo_scores[dim][idx_j] = change_winner_to_score(
                    val, dimensional_elo_scores[dim][idx_i], dimensional_elo_scores[dim][idx_j]
                )
                elo_scores[idx_i], elo_scores[idx_j] = change_winner_to_score(
                    val, elo_scores[idx_i], elo_scores[idx_j]
                )

        elo_selected = elo_scores.index(max(elo_scores)) if elo_scores else 0

        # Build one comparison entry per unordered pair from the aggregated ELO scores.
        # This mirrors what run_rr_tournament stores for non-batch runs so that
        # calculate_pairwise_accuracy can iterate over them.
        pair_comparisons = []
        for idx_i in range(n_ideas):
            for idx_j in range(idx_i + 1, n_ideas):
                si, sj = elo_scores[idx_i], elo_scores[idx_j]
                if si > sj:
                    winner = 0
                elif sj > si:
                    winner = 1
                else:
                    winner = 2
                pair_comparisons.append({"idea_0": idea_keys[idx_i], "idea_1": idea_keys[idx_j], "winner": winner})

        problem_data: Dict[str, Any] = dict(dimensional_elo_scores)
        problem_data["overall"] = elo_scores
        problem_data["elo_selected"] = elo_selected
        problem_data["ideas"] = idea_keys
        problem_data["comparisons"] = pair_comparisons
        final_results[problem_name] = problem_data

    return final_results
