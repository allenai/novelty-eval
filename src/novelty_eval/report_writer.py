"""
Plain-text report generation for accuracy_test experiments.
"""
from typing import List

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .experiment_stats import ExperimentStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ranking_stats(f, label: str, stats: ExperimentStats) -> None:
    f.write(f"{label}\n")
    f.write(f"{'-' * len(label)}\n")
    f.write(f"Mean NDCG: {stats.mean_ndcg:.4f}\n")
    f.write(f"Mean MRR: {stats.mean_mrr:.4f}\n")
    f.write(f"Mean MR: {stats.mean_mr:.4f}\n")
    f.write(f"Mean Hits@1: {stats.mean_hits[1]:.4f}\n")
    f.write(f"Mean Hits@2: {stats.mean_hits[2]:.4f}\n")
    f.write(f"Mean Hits@3: {stats.mean_hits[3]:.4f}\n")
    f.write(f"Individual Run NDCG: {stats.ndcg_scores}\n")
    f.write(f"Individual Run MRR: {stats.mrr_scores}\n")
    f.write(f"Individual Run MR: {stats.mr_scores}\n")
    f.write(f"Individual Run Hits@1: {stats.hits_scores[1]}\n")
    f.write(f"Individual Run Hits@2: {stats.hits_scores[2]}\n")
    f.write(f"Individual Run Hits@3: {stats.hits_scores[3]}\n")
    f.write("\n")


def _write_pairwise_stats(f, label: str, stats: ExperimentStats) -> None:
    f.write(f"{label}\n")
    f.write(f"{'-' * len(label)}\n")
    f.write(f"Mean LLM Pairwise Accuracy (with ties):    {stats.mean_pairwise_accuracy:.4f}  (support={stats.mean_pairwise_support:.1f})\n")
    f.write(f"Mean LLM Pairwise Accuracy (strict):       {stats.mean_pairwise_accuracy_strict:.4f}  (support={stats.mean_pairwise_support:.1f})\n")
    f.write(f"Mean LLM Pairwise Accuracy (without ties): {stats.mean_pairwise_accuracy_no_ties:.4f}  (support={stats.mean_pairwise_support_no_ties:.1f})\n")
    f.write(f"Mean Number of Ties: {stats.mean_pairwise_n_ties:.1f}\n")
    f.write(f"Individual Run Accuracies (with ties): {stats.pairwise_accuracies}\n")
    f.write(f"Individual Run Accuracies (strict):    {stats.pairwise_accuracies_strict}\n")
    f.write(f"Individual Run Accuracies (without ties): {stats.pairwise_accuracies_no_ties}\n")
    f.write(f"Individual Run Ties: {stats.pairwise_n_ties}\n")
    f.write(f"Individual Run Support (with ties):    {stats.pairwise_support}\n")
    f.write(f"Individual Run Support (without ties): {stats.pairwise_support_no_ties}\n")
    f.write("\n")


def _write_pointwise_stats(f, label: str, stats: ExperimentStats) -> None:
    f.write(f"{label}\n")
    f.write(f"{'-' * len(label)}\n")
    f.write(f"Mean Accuracy:              {stats.mean_pointwise_accuracy:.4f}\n")
    f.write(f"Mean F1 (macro):            {stats.mean_pointwise_f1_macro:.4f}\n")
    f.write(f"Mean Precision (POSITIVE):  {stats.mean_pointwise_precision_pos:.4f}\n")
    f.write(f"Mean Precision (NEGATIVE):  {stats.mean_pointwise_precision_neg:.4f}\n")
    f.write(f"Mean Recall    (POSITIVE):  {stats.mean_pointwise_recall_pos:.4f}\n")
    f.write(f"Mean Recall    (NEGATIVE):  {stats.mean_pointwise_recall_neg:.4f}\n")
    f.write(f"Mean F1        (POSITIVE):  {stats.mean_pointwise_f1_pos:.4f}\n")
    f.write(f"Mean F1        (NEGATIVE):  {stats.mean_pointwise_f1_neg:.4f}\n")
    f.write(f"Individual Run Accuracies:  {stats.pointwise_accuracies}\n")
    f.write(f"Individual Run F1 (macro):  {stats.pointwise_f1_macro}\n")
    f.write(f"Individual Run F1 (POS):    {stats.pointwise_f1_pos}\n")
    f.write(f"Individual Run F1 (NEG):    {stats.pointwise_f1_neg}\n")
    f.write("\n")


def _write_pairwise_summary(f, stats: ExperimentStats) -> None:
    f.write("\nLLM Pairwise Accuracy:\n")
    f.write(f"  Accuracy (with ties):    {stats.mean_pairwise_accuracy:.4f}  (support={stats.mean_pairwise_support:.1f})\n")
    f.write(f"  Accuracy (strict):       {stats.mean_pairwise_accuracy_strict:.4f}  (support={stats.mean_pairwise_support:.1f})\n")
    f.write(f"  Accuracy (without ties): {stats.mean_pairwise_accuracy_no_ties:.4f}  (support={stats.mean_pairwise_support_no_ties:.1f})\n")
    f.write(f"  Mean Ties:               {stats.mean_pairwise_n_ties:.1f}\n")


def _write_pointwise_summary(f, stats: ExperimentStats) -> None:
    f.write("\nPointwise Classification:\n")
    f.write(f"  Accuracy:            {stats.mean_pointwise_accuracy:.4f}\n")
    f.write(f"  F1 (macro):          {stats.mean_pointwise_f1_macro:.4f}\n")
    f.write(f"  F1 (POSITIVE):       {stats.mean_pointwise_f1_pos:.4f}\n")
    f.write(f"  F1 (NEGATIVE):       {stats.mean_pointwise_f1_neg:.4f}\n")
    f.write(f"  Precision (POSITIVE):{stats.mean_pointwise_precision_pos:.4f}\n")
    f.write(f"  Recall    (POSITIVE):{stats.mean_pointwise_recall_pos:.4f}\n")
    f.write(f"  Precision (NEGATIVE):{stats.mean_pointwise_precision_neg:.4f}\n")
    f.write(f"  Recall    (NEGATIVE):{stats.mean_pointwise_recall_neg:.4f}\n")


def _write_mode_stats(f, label: str, stats: ExperimentStats, test_mode: str) -> None:
    if test_mode == 'pairwise':
        _write_pairwise_stats(f, label, stats)
    elif test_mode == 'pointwise':
        _write_pointwise_stats(f, label, stats)
    else:
        _write_ranking_stats(f, label, stats)


def _write_debug_mode(f, label: str, stats: ExperimentStats) -> None:
    f.write(f"{label} Details\n")
    f.write(f"{'-' * (len(label) + 8)}\n")
    f.write("Problematic Rankings:\n")
    for i, wrong_list in enumerate(stats.wrong_pairs):
        if wrong_list:
            f.write(f"  Run {i + 1}:\n")
            for wp in wrong_list:
                if isinstance(wp, dict):
                    if 'comparisons' in wp:
                        # pairwise mode entry
                        comps = wp['comparisons']
                        wrong_comps = [
                            c for c in comps
                            if c.get('gt_winner') is not None
                            and (int(c.get('winner', -1)) == 2 or int(c.get('winner', -1)) != c.get('gt_winner'))
                        ]
                        msg = (
                            f"Problem {wp['problem_id']}: "
                            f"Expected Winners: {wp.get('expected_winners', [])}; "
                            f"Wrong/tied comparisons: {len(wrong_comps)}/{len(comps)}"
                        )
                    else:
                        # ranking mode entry
                        msg = (
                            f"Problem {wp['problem_id']}: "
                            f"Predicted Rank: {wp['predicted_rank']}; "
                            f"Gold Winners: {wp['gold_ranks']}"
                        )
                    f.write(f"    - {msg}\n")
                else:
                    f.write(f"    - {wp}\n")
    f.write("\n")

    f.write("Scores per Run:\n")
    for i, scores in enumerate(stats.all_scores):
        f.write(f"  Run {i + 1}:\n")
        for prob_id, prob_scores in scores.items():
            f.write(f"    Problem {prob_id}:\n")
            for idea_key, dim_scores in prob_scores.items():
                f.write(f"      Idea {idea_key}: {dim_scores}\n")
    f.write("\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_report(
    output_file: str,
    debug_file: str,
    rr_stats: ExperimentStats,
    swiss_stats: ExperimentStats,
    random_stats: ExperimentStats,
    n_runs: int,
    test_inputs_path: str,
    num_instances: int,
    modes: List[str],
    test_mode: str,
    pairwise_stats: ExperimentStats = None,
    pointwise_stats: ExperimentStats = None,
) -> None:
    if pairwise_stats is None:
        pairwise_stats = ExperimentStats.empty()
    if pointwise_stats is None:
        pointwise_stats = ExperimentStats.empty()

    is_pairwise = test_mode == 'pairwise'
    is_pointwise = test_mode == 'pointwise'

    with open(output_file, 'w') as f:
        f.write("Comparative Evaluator Accuracy Report\n")
        f.write("=====================================\n\n")
        f.write(f"Test Instances Path: {test_inputs_path}\n")
        f.write(f"Number of Test Instances Processed: {num_instances}\n")
        f.write(f"Test Mode: {test_mode}\n\n")

        if is_pairwise:
            f.write("Metrics:\n")
            f.write("- LLM Pairwise Accuracy (with ties): ties counted as 0.5 correct\n")
            f.write("- LLM Pairwise Accuracy (without ties): tied comparisons excluded entirely\n")
            f.write("- Number of Ties: comparisons where the LLM declared a tie\n")
            f.write("- Support: total eligible comparisons (one gold winner vs. one non-winner)\n\n")
        elif is_pointwise:
            f.write("Metrics:\n")
            f.write("- Accuracy: fraction of correctly classified ideas\n")
            f.write("- F1 (macro): unweighted mean of per-class F1 scores\n")
            f.write("- Precision/Recall/F1 (POSITIVE): for the novel class\n")
            f.write("- Precision/Recall/F1 (NEGATIVE): for the non-novel class\n\n")
        else:
            f.write("Metrics:\n")
            f.write("1. NDCG (Normalized Discounted Cumulative Gain)\n")
            f.write("2. MRR (Mean Reciprocal Rank)\n")
            f.write("3. MR (Mean Rank)\n")
            f.write("4. Hits@k (Proportion of instances where at least one relevant item appears in the top k results)\n\n")

        f.write(f"Number of runs per experiment: {n_runs}\n\n")

        if 'random' in modes:
            _write_mode_stats(f, "1. Random Ranking", random_stats, test_mode)
        if 'rr' in modes:
            _write_mode_stats(f, "2. Round Robin Tournament", rr_stats, test_mode)
        if 'swiss' in modes:
            _write_mode_stats(f, "3. Swiss Tournament", swiss_stats, test_mode)
        if 'pairwise' in modes:
            _write_mode_stats(f, "Pairwise", pairwise_stats, test_mode)
        if 'pointwise' in modes:
            _write_mode_stats(f, "Pointwise", pointwise_stats, test_mode)

        f.write("Summary\n")
        f.write("-------\n")

        if is_pairwise:
            if 'pairwise' in modes:
                _write_pairwise_summary(f, pairwise_stats)
        elif is_pointwise:
            if 'pointwise' in modes:
                _write_pointwise_summary(f, pointwise_stats)
        else:
            for metric, attr in [
                ("NDCG Comparison", "mean_ndcg"),
                ("MRR Comparison", "mean_mrr"),
            ]:
                f.write(f"\n{metric}:\n")
                if 'random' in modes:
                    f.write(f"Random Ranking: {getattr(random_stats, attr):.4f}\n")
                if 'rr' in modes:
                    f.write(f"Round Robin: {getattr(rr_stats, attr):.4f}\n")
                if 'swiss' in modes:
                    f.write(f"Swiss Tournament: {getattr(swiss_stats, attr):.4f}\n")

    with open(debug_file, 'w') as f:
        f.write("Comparative Evaluator Debug Artifact\n")
        f.write("====================================\n\n")
        if 'pairwise' in modes:
            _write_debug_mode(f, "Pairwise", pairwise_stats)
        if 'rr' in modes:
            _write_debug_mode(f, "1. Round Robin Tournament", rr_stats)
        if 'swiss' in modes:
            _write_debug_mode(f, "2. Swiss Tournament", swiss_stats)
