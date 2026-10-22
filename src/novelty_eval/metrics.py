"""
Ranking evaluation metrics: pairwise accuracy, NDCG, MRR, MR, Hits@k, and LLM pairwise accuracy.
"""
import logging
import math
import statistics
from typing import Dict, Any, List, Tuple

LOGGER = logging.getLogger(__name__)


def _sorted_by_score(idea_keys: List[str], elo_scores: List[float]) -> List[Tuple[str, float]]:
    """Return (key, score) pairs sorted by score descending."""
    return sorted(zip(idea_keys, elo_scores), key=lambda x: x[1], reverse=True)


def _get_ranked_keys_and_scores(result_data: Dict[str, Any]) -> Tuple[List[str], Dict[str, float]]:
    idea_keys = [str(k) for k in result_data['ideas']]
    elo_scores = result_data['overall']
    key_to_score = dict(zip(idea_keys, elo_scores))
    ranked_keys = [k for k, _ in _sorted_by_score(idea_keys, elo_scores)]
    return ranked_keys, key_to_score


def _iter_labeled_results(results: Dict[str, Any], inputs: Dict[str, Any]):
    """Yield (problem_id, result_data, expected_winners_set) for each valid instance."""
    for problem_id, result_data in results.items():
        # Handle key-type mismatch: JSON round-trips integer keys as strings while
        # YAML safe_load returns them as ints, so try both.
        input_data = inputs.get(problem_id)
        if input_data is None:
            try:
                input_data = inputs.get(int(problem_id))
            except (ValueError, TypeError):
                input_data = inputs.get(str(problem_id))
        if input_data is None:
            continue
        if 'expected_winners' not in input_data:
            continue
        expected_winners = set(str(w) for w in input_data['expected_winners'])
        yield problem_id, result_data, expected_winners


def calculate_ranking_accuracy(
    results: Dict[str, Any],
    inputs: Dict[str, Any],
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pairwise ranking accuracy: fraction of (winner, loser) pairs that are correctly ordered.

    Returns:
        accuracy         – fraction of correct pairwise comparisons (ties score 0.5)
        wrong_pairs      – list of problem entries where at least one pair was wrong
        good_pairs       – list of problem entries where all pairs were correct
    """
    correct_pairs = 0.0
    total_pairs = 0
    wrong_pairs: List[Dict[str, Any]] = []
    good_pairs: List[Dict[str, Any]] = []

    for problem_id, result_data, expected_winners in _iter_labeled_results(results, inputs):
        idea_keys = [str(k) for k in result_data['ideas']]
        elo_scores = result_data['overall']
        key_to_score = dict(zip(idea_keys, elo_scores))

        winners = [(k, s) for k, s in key_to_score.items() if k in expected_winners]
        losers  = [(k, s) for k, s in key_to_score.items() if k not in expected_winners]

        problem_has_error = False
        for _, w_score in winners:
            for _, l_score in losers:
                if w_score > l_score:
                    correct_pairs += 1
                elif w_score == l_score:
                    correct_pairs += 0.5
                    problem_has_error = True
                else:
                    problem_has_error = True
                total_pairs += 1

        ranked_keys = [k for k, _ in _sorted_by_score(idea_keys, elo_scores)]
        gold_winners = list(expected_winners)
        gold_ranks = [
            f"{gw} (Rank: {ranked_keys.index(gw) + 1})" if gw in ranked_keys else f"{gw} (Not found)"
            for gw in gold_winners
        ]

        raw_retrieved = result_data.get('retrieved_papers', {})
        entry = {
            'problem_id': problem_id,
            'predicted_rank': ranked_keys,
            'gold_winners': gold_winners,
            'gold_ranks': gold_ranks,
            'scores': key_to_score,
            'retrieved_papers': {str(k): v for k, v in raw_retrieved.items()},
        }

        (wrong_pairs if problem_has_error else good_pairs).append(entry)

    if total_pairs == 0:
        return 0.0, [], []
    return correct_pairs / total_pairs, wrong_pairs, good_pairs


def calculate_ndcg(results: Dict[str, Any], inputs: Dict[str, Any]) -> float:
    """
    Mean NDCG across all instances (binary relevance: expected winners = 1, others = 0).
    """
    ndcg_scores = []
    for _, result_data, expected_winners in _iter_labeled_results(results, inputs):
        idea_keys = [str(k) for k in result_data['ideas']]
        elo_scores = result_data['overall']

        items = [
            {'relevance': int(k in expected_winners), 'score': s}
            for k, s in zip(idea_keys, elo_scores)
        ]

        dcg  = sum(item['relevance'] / math.log2(i + 2) for i, item in enumerate(sorted(items, key=lambda x: x['score'], reverse=True)) if item['relevance'] > 0)
        idcg = sum(item['relevance'] / math.log2(i + 2) for i, item in enumerate(sorted(items, key=lambda x: x['relevance'], reverse=True)) if item['relevance'] > 0)

        ndcg_scores.append(0.0 if idcg == 0 else dcg / idcg)

    return statistics.mean(ndcg_scores) if ndcg_scores else 0.0


def calculate_mrr(results: Dict[str, Any], inputs: Dict[str, Any]) -> float:
    """
    Mean Reciprocal Rank of the first expected winner in the predicted ranking.
    """
    reciprocal_ranks = []
    for _, result_data, expected_winners in _iter_labeled_results(results, inputs):
        idea_keys = [str(k) for k in result_data['ideas']]
        elo_scores = result_data['overall']
        sorted_keys = [k for k, _ in _sorted_by_score(idea_keys, elo_scores)]
        rank = next((i + 1 for i, k in enumerate(sorted_keys) if k in expected_winners), 0)
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
    return statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_mr(results: Dict[str, Any], inputs: Dict[str, Any]) -> float:
    """
    Mean Rank of the first expected winner in the predicted ranking.
    """
    ranks = []
    for _, result_data, expected_winners in _iter_labeled_results(results, inputs):
        idea_keys = [str(k) for k in result_data['ideas']]
        elo_scores = result_data['overall']
        sorted_keys = [k for k, _ in _sorted_by_score(idea_keys, elo_scores)]
        rank = next((i + 1 for i, k in enumerate(sorted_keys) if k in expected_winners), 0)
        if rank > 0:
            ranks.append(rank)
    return statistics.mean(ranks) if ranks else 0.0


def calculate_hits_at_k(
    results: Dict[str, Any],
    inputs: Dict[str, Any],
    k_values: List[int],
) -> Dict[int, float]:
    """
    Hits@k: proportion of instances where at least one expected winner is in the top-k results.
    """
    hits: Dict[int, List[int]] = {k: [] for k in k_values}

    for _, result_data, expected_winners in _iter_labeled_results(results, inputs):
        idea_keys = [str(k) for k in result_data['ideas']]
        elo_scores = result_data['overall']
        sorted_keys = [k for k, _ in _sorted_by_score(idea_keys, elo_scores)]
        for k in k_values:
            hits[k].append(int(any(key in expected_winners for key in sorted_keys[:k])))

    return {k: (statistics.mean(hits[k]) if hits[k] else 0.0) for k in k_values}


def calculate_pairwise_accuracy(results: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    True pairwise accuracy: fraction of individual LLM comparisons where the correct idea won.

    - Only comparisons where exactly one of the two ideas is a gold winner are counted.
    - Ties (winner == 2 or unparsable) count as 0.5 correct in accuracy_with_ties,
      and are excluded from both numerator and denominator of accuracy_without_ties
      (MT-Bench S2 convention: correct / (total - ties)).

    Returns a dict with:
        accuracy_with_ties     – accuracy counting ties as 0.5 (Deutsch et al. PA)
        accuracy_without_ties  – accuracy on non-tie pairs only (MT-Bench S2)
        n_ties                 – number of tied comparisons
        support                – total eligible comparisons (including ties)
        support_without_ties   – eligible comparisons excluding ties
    """
    correct = 0.0
    correct_with_ties = 0.0
    correct_strict = 0.0
    total = 0
    ties = 0

    for problem_id, result_data, expected_winners in _iter_labeled_results(results, inputs):
        for comp in result_data.get('comparisons', []):
            winner = comp.get('winner', None)
            gt_winner = comp.get('gt_winner', None)

            if gt_winner is None:
                continue

            total += 1
            w = int(winner)

            if w == 2:  # tie
                ties += 1
                correct_with_ties += 0.5
            elif gt_winner == winner:
                correct += 1.0
                correct_with_ties += 1.0
                correct_strict += 1.0

    support_no_ties = total - ties
    return {
        "accuracy_with_ties": correct_with_ties / total if total > 0 else 0.0,
        "accuracy_strict": correct_strict / total if total > 0 else 0.0,
        "accuracy_without_ties": correct / support_no_ties if support_no_ties > 0 else 0.0,
        "n_ties": ties,
        "support": total,
        "support_without_ties": support_no_ties,
    }


def collect_run_scores(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten per-dimension ELO score lists into a per-idea dict of {dimension: score}.
    Used to build the human-readable scores section of the debug report.
    """
    run_scores: Dict[str, Any] = {}
    for problem_id, res in results.items():
        idea_keys = res['ideas']
        dims = [k for k in res if k not in ('ideas', 'elo_selected', 'comparisons')]
        scores_map = {}
        for idx, idea_key in enumerate(idea_keys):
            idea_scores = {
                dim: res[dim][idx]
                for dim in dims
                if dim in res and idx < len(res[dim])
            }
            scores_map[str(idea_key)] = idea_scores
        run_scores[problem_id] = scores_map
    return run_scores


def compute_ranking_metrics(
    results: Dict[str, Any],
    inputs: Dict[str, Any],
) -> Tuple[float, float, float, Dict[int, float]]:
    """
    Metrics for ranking test mode: NDCG, MRR, MR, Hits@k.

    Returns:
        (ndcg, mrr, mr, hits)
    """
    ndcg = calculate_ndcg(results, inputs)
    mrr  = calculate_mrr(results, inputs)
    mr   = calculate_mr(results, inputs)
    hits = calculate_hits_at_k(results, inputs, [1, 2, 3])
    return ndcg, mrr, mr, hits


def compute_pairwise_metrics(
    results: Dict[str, Any],
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Metrics for pairwise test mode: LLM pairwise accuracy with and without ties, plus tie count and support.

    Returns:
        dict with accuracy_with_ties, accuracy_without_ties, n_ties, support
    """
    return calculate_pairwise_accuracy(results, inputs)


def compute_pairwise_wrong_good(
    results: Dict[str, Any],
    inputs: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build per-problem bad/good comparison lists for the pairwise test mode debug report.

    Each entry in wrong_pairs / good_pairs is a dict with:
        problem_id, comparisons (list of comp dicts), expected_winners

    A problem is "bad" if any of its comparisons has an incorrect or tied prediction.
    A problem is "good" if all its comparisons are correctly predicted.

    Returns:
        (wrong_pairs, good_pairs)
    """
    wrong_pairs: List[Dict[str, Any]] = []
    good_pairs: List[Dict[str, Any]] = []

    for problem_id, result_data, expected_winners in _iter_labeled_results(results, inputs):
        comps = result_data.get('comparisons', [])
        if not comps:
            continue

        problem_has_error = False
        for comp in comps:
            winner = comp.get('winner', None)
            gt_winner = comp.get('gt_winner', None)
            if gt_winner is None:
                continue  # no GT available for this comparison
            try:
                w = int(winner)
            except (TypeError, ValueError):
                problem_has_error = True
                continue
            if w == 2 or w != gt_winner:  # tie or wrong
                problem_has_error = True

        entry = {
            'problem_id': problem_id,
            'comparisons': comps,
            'expected_winners': list(expected_winners),
        }
        (wrong_pairs if problem_has_error else good_pairs).append(entry)

    return wrong_pairs, good_pairs


def compute_pointwise_metrics(
    results: Dict[str, Any],
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Binary classification metrics for pointwise single-idea evaluation.

    results: {instance_id: {'prediction': 0|1, 'label': 'POSITIVE'|'NEGATIVE'}}
    inputs:  original instances with 'label' field

    Returns accuracy, per-class precision/recall/f1, and support counts.
    """
    tp = tn = fp = fn = 0

    for instance_id, result_data in results.items():
        pred = result_data.get('prediction')
        label = result_data.get('label')
        if pred is None or label is None:
            LOGGER.warning(
                "compute_pointwise_metrics: skipping instance %r — "
                "%s. Instance will not contribute to metric counts.",
                instance_id,
                "prediction is None" if pred is None else "label is None",
            )
            continue
        is_positive = (label == 'POSITIVE')
        pred_positive = (int(pred) == 1)
        if is_positive and pred_positive:
            tp += 1
        elif is_positive and not pred_positive:
            fn += 1
        elif not is_positive and pred_positive:
            fp += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)
              if (precision_pos + recall_pos) > 0 else 0.0)

    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = (2 * precision_neg * recall_neg / (precision_neg + recall_neg)
              if (precision_neg + recall_neg) > 0 else 0.0)

    f1_macro = (f1_pos + f1_neg) / 2

    return {
        'accuracy': accuracy,
        'precision_pos': precision_pos,
        'precision_neg': precision_neg,
        'recall_pos': recall_pos,
        'recall_neg': recall_neg,
        'f1_pos': f1_pos,
        'f1_neg': f1_neg,
        'f1_macro': f1_macro,
        'support_pos': tp + fn,
        'support_neg': tn + fp,
    }


def compute_all_metrics(
    results: Dict[str, Any],
    inputs: Dict[str, Any],
) -> Tuple[float, float, float, Dict[int, float], Dict[str, Any]]:
    """
    Compute all metrics for a single run in one call.

    Returns:
        (ndcg, mrr, mr, hits, pairwise_metrics)
        where pairwise_metrics is a dict with accuracy_with_ties, accuracy_without_ties, n_ties, support.
    """
    ndcg, mrr, mr, hits = compute_ranking_metrics(results, inputs)
    pairwise_metrics = compute_pairwise_metrics(results, inputs)
    return ndcg, mrr, mr, hits, pairwise_metrics


