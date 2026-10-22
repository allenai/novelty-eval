"""
Dataclass representing the results of a single experiment mode (rr, swiss, bi-swiss, random).
Replaces the unwieldy 16-element tuples returned by run_experiment and friends.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any

EMPTY_HITS: Dict[int, List[float]] = {1: [], 2: [], 3: []}
EMPTY_MEAN_HITS: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}


@dataclass
class ExperimentStats:
    accuracies: List[float] = field(default_factory=list)
    mean_accuracy: float = 0.0

    ndcg_scores: List[float] = field(default_factory=list)
    mean_ndcg: float = 0.0

    mrr_scores: List[float] = field(default_factory=list)
    mean_mrr: float = 0.0

    mr_scores: List[float] = field(default_factory=list)
    mean_mr: float = 0.0

    hits_scores: Dict[int, List[float]] = field(default_factory=lambda: {1: [], 2: [], 3: []})
    mean_hits: Dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0, 3: 0.0})

    wrong_pairs: List[List[Dict[str, Any]]] = field(default_factory=list)
    good_pairs: List[List[Dict[str, Any]]] = field(default_factory=list)

    all_scores: List[Dict[str, Any]] = field(default_factory=list)
    run_paths: List[str] = field(default_factory=list)

    pairwise_accuracies: List[float] = field(default_factory=list)         # accuracy_with_ties per run
    mean_pairwise_accuracy: float = 0.0                                    # mean accuracy_with_ties

    pairwise_accuracies_strict: List[float] = field(default_factory=list)  # accuracy_strict per run (ties=0)
    mean_pairwise_accuracy_strict: float = 0.0

    pairwise_accuracies_no_ties: List[float] = field(default_factory=list) # accuracy_without_ties per run
    mean_pairwise_accuracy_no_ties: float = 0.0

    pairwise_n_ties: List[int] = field(default_factory=list)               # n_ties per run
    mean_pairwise_n_ties: float = 0.0

    pairwise_support: List[int] = field(default_factory=list)              # support per run (incl. ties)
    mean_pairwise_support: float = 0.0

    pairwise_support_no_ties: List[int] = field(default_factory=list)      # support per run (excl. ties)
    mean_pairwise_support_no_ties: float = 0.0

    # Pointwise binary classification metrics (per run)
    pointwise_accuracies: List[float] = field(default_factory=list)
    mean_pointwise_accuracy: float = 0.0

    pointwise_precision_pos: List[float] = field(default_factory=list)
    mean_pointwise_precision_pos: float = 0.0

    pointwise_precision_neg: List[float] = field(default_factory=list)
    mean_pointwise_precision_neg: float = 0.0

    pointwise_recall_pos: List[float] = field(default_factory=list)
    mean_pointwise_recall_pos: float = 0.0

    pointwise_recall_neg: List[float] = field(default_factory=list)
    mean_pointwise_recall_neg: float = 0.0

    pointwise_f1_pos: List[float] = field(default_factory=list)
    mean_pointwise_f1_pos: float = 0.0

    pointwise_f1_neg: List[float] = field(default_factory=list)
    mean_pointwise_f1_neg: float = 0.0

    pointwise_f1_macro: List[float] = field(default_factory=list)
    mean_pointwise_f1_macro: float = 0.0

    pointwise_support_pos: List[int] = field(default_factory=list)
    pointwise_support_neg: List[int] = field(default_factory=list)

    def to_tuple(self):
        """Backward-compat: return the old 16-element tuple."""
        return (
            self.accuracies, self.mean_accuracy,
            self.ndcg_scores, self.mean_ndcg,
            self.mrr_scores, self.mean_mrr,
            self.mr_scores, self.mean_mr,
            self.hits_scores, self.mean_hits,
            self.wrong_pairs, self.good_pairs,
            self.all_scores, self.run_paths,
            self.pairwise_accuracies, self.mean_pairwise_accuracy,
        )

    @staticmethod
    def empty() -> "ExperimentStats":
        return ExperimentStats()

