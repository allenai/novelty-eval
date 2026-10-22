"""
One-pass bootstrap: a single scipy bootstrap call computes all metrics at once.

Public entry points:
  - compute_bootstrap_results_pairwise_all  — pairwise accuracy metrics (accuracy, strict, no-ties)
  - compute_bootstrap_results_pointwise_all — pointwise classification metrics (accuracy, F1, precision, recall)

Both return {metric_key: {"significant": bool, "reliable": bool}}.
"""
import warnings
import numpy as np
from scipy.stats import bootstrap


def _run_bootstrap_multi(
        data_tuple: tuple,
        statistic_fn,
        metric_keys: list[str],
        n_resamples: int,
        random_state: int,
        label: str,
        _logger,
        alternative: str = "two-sided",
) -> dict[str, dict]:
    """Runs scipy bootstrap over multiple metrics at once.

    Performs paired bootstrapping at the corpus level:
      1. Samples an array of indices [1...n] with repetitions
      2. Uses the same indices for all arrays in data_tuple
      3. Calls statistic_fn on the resampled tuple — must return a length-M array of metric deltas

    Args:
        data_tuple: Arrays passed directly to scipy bootstrap (e.g. base scores, ablation scores).
        statistic_fn: Callable(*data_tuple) → np.ndarray of shape (n_metrics,); computes per-metric deltas.
        metric_keys: Names for each metric, in the same order as statistic_fn's output.
        n_resamples: Number of bootstrap resamples.
        random_state: Seed for reproducibility.
        label: Context string used only in warning messages (e.g. "[condition_name]").
        _logger: Logger instance for scipy warnings; pass None to suppress.
        alternative: Hypothesis direction — "two-sided", "greater", or "less".

    Returns:
        {metric_key: {"significant": bool, "reliable": bool}}
    """
    rng = np.random.default_rng(random_state)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = bootstrap(
            data_tuple,
            statistic_fn,
            paired=True, # resamples an array of indices and uses the same indices for all arrays in data
            vectorized=False,
            n_resamples=n_resamples,
            random_state=rng,
            alternative=alternative,
        )

    lows = res.confidence_interval.low  # shape (n_metrics,)
    highs = res.confidence_interval.high  # shape (n_metrics,)
    dist = res.bootstrap_distribution  # shape (n_metrics, n_resamples)
    out: dict[str, dict] = {}

    for i, k in enumerate(metric_keys):
        # Check test results reliability.
        low, high = float(lows[i]), float(highs[i])
        zero_variance = float(np.std(dist[i])) == 0.0
        if alternative == "greater":
            finite_ok = bool(np.isfinite(low))
        elif alternative == "less":
            finite_ok = bool(np.isfinite(high))
        else:
            finite_ok = bool(np.isfinite(low) and np.isfinite(high))
        reliable = finite_ok and not zero_variance

        # Check significance
        if alternative == "greater":
            significant = bool(low > 0) if reliable else False
        elif alternative == "less":
            significant = bool(high < 0) if reliable else False
        else:
            significant = bool(low > 0 or high < 0) if reliable else False
        # ci_low/ci_high are kept (not just the booleans derived from them) so
        # dot-and-interval figures can be drawn without re-running the bootstrap.
        # None where the bound is non-finite, to stay JSON-safe.
        out[k] = {
            "significant": significant,
            "reliable": reliable,
            "ci_low": low if np.isfinite(low) else None,
            "ci_high": high if np.isfinite(high) else None,
        }

    if caught and _logger:
        unreliable = [k for k, v in out.items() if not v["reliable"]]
        reliability_note = f" — non-reliable: {unreliable}" if unreliable else " — all CIs reliable"
        for w in caught:
            _logger.warning("scipy bootstrap warning%s: %s%s", label, w.message, reliability_note)

    return out


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------

_PAIRWISE_METRIC_KEYS = [
    "pairwise_accuracy",
    "pairwise_accuracy_strict",
    "pairwise_accuracy_no_ties",
]


def _pairwise_all_statistic(base_correct, base_tie, abl_correct, abl_tie):
    """Compute deltas for all 3 pairwise metrics; returns a length-3 array.

    Assumes correct and tie are mutually exclusive (an instance cannot be both).
    """
    # accuracy with ties: correct=1 → 1.0, tie=1 → 0.5
    d_acc = np.mean(abl_correct + 0.5 * abl_tie) - np.mean(base_correct + 0.5 * base_tie)

    # strict accuracy: correct=1 -> 1.0, ties get no credit
    d_strict = np.mean(abl_correct) - np.mean(base_correct)

    # accuracy excluding ties: mean(correct) over non-tie instances only
    base_nt = base_correct[base_tie < 0.5]
    abl_nt = abl_correct[abl_tie < 0.5]
    b = float(np.mean(base_nt)) if len(base_nt) else 0.0
    a = float(np.mean(abl_nt)) if len(abl_nt) else 0.0
    d_no_ties = a - b

    return np.array([d_acc, d_strict, d_no_ties])


def compute_bootstrap_results_pairwise_all(
        base_pairwise_data: list[tuple[float, float, float]],
        abl_pairwise_data: list[tuple[float, float, float]],
        n_resamples: int = 100000,
        random_state: int = 42,
        context_label: str = "",
        _logger=None,
        alternative: str = "two-sided",
) -> dict[str, dict]:
    """
    Corpus-level paired bootstrap for all pairwise accuracy metrics in one call.

    Each tuple is (is_correct, is_tie, gt_winner).
    Returns {metric_key: {"significant": bool, "reliable": bool}} for:
      pairwise_accuracy, pairwise_accuracy_strict, pairwise_accuracy_no_ties
    """
    if (
            not base_pairwise_data
            or not abl_pairwise_data
            or len(base_pairwise_data) != len(abl_pairwise_data)
    ):
        return {}

    base_arr = np.array(base_pairwise_data, dtype=float)
    abl_arr = np.array(abl_pairwise_data, dtype=float)
    label = f" [{context_label}]" if context_label else ""

    _gt_mismatch = np.where(base_arr[:, 2] != abl_arr[:, 2])[0]
    if len(_gt_mismatch) > 0:
        _idx = _gt_mismatch[:10].tolist()
        _pairs = [(float(base_arr[i, 2]), float(abl_arr[i, 2])) for i in _idx]
        raise ValueError(
            f"base_target and abl_target must be identical for paired bootstrap"
            f"{label} — {len(_gt_mismatch)} mismatched positions; "
            f"first positions {_idx} have (base_gt, abl_gt)={_pairs}"
        )

    data_tuple = (base_arr[:, 0], base_arr[:, 1], abl_arr[:, 0], abl_arr[:, 1])

    return _run_bootstrap_multi(
        data_tuple,
        _pairwise_all_statistic,
        _PAIRWISE_METRIC_KEYS,
        n_resamples=n_resamples,
        random_state=random_state,
        label=label,
        _logger=_logger,
        alternative=alternative,
    )


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------

_POINTWISE_METRIC_KEYS = [
    "accuracy",
    "f1_macro",
    "f1_pos",
    "f1_neg",
    "precision_pos",
    "precision_neg",
    "recall_pos",
    "recall_neg",
]


def _counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for binary arrays using 0.5 threshold."""
    tp = int(np.sum((pred >= 0.5) & (target >= 0.5)))
    fp = int(np.sum((pred >= 0.5) & (target < 0.5)))
    fn = int(np.sum((pred < 0.5) & (target >= 0.5)))
    tn = int(np.sum((pred < 0.5) & (target < 0.5)))
    return tp, fp, fn, tn


def _all_pointwise_metrics(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute all 8 pointwise metrics in the order defined by _POINTWISE_METRIC_KEYS; returns a length-8 array."""
    tp, fp, fn, tn = _counts(pred, target)
    total = tp + fp + fn + tn

    acc = (tp + tn) / total if total else 0.0

    p_pos = tp / (tp + fp) if (tp + fp) else 0.0
    r_pos = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) else 0.0

    p_neg = tn / (tn + fn) if (tn + fn) else 0.0
    r_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) else 0.0

    f1_macro = (f1_pos + f1_neg) / 2

    return np.array([acc, f1_macro, f1_pos, f1_neg, p_pos, p_neg, r_pos, r_neg])


def _pointwise_all_statistic(base_pred, abl_pred, target):
    """Compute deltas for all 8 pointwise metrics; returns a length-8 array."""
    return _all_pointwise_metrics(abl_pred, target) - _all_pointwise_metrics(base_pred, target)


def compute_bootstrap_results_pointwise_all(
        base_pred_labels: list[tuple[int, int]],
        abl_pred_labels: list[tuple[int, int]],
        n_resamples: int = 100000,
        random_state: int = 42,
        context_label: str = "",
        _logger=None,
        alternative: str = "two-sided",
) -> dict[str, dict]:
    """
    Corpus-level paired bootstrap for all pointwise metrics in one call.

    Each tuple is (pred, target) as binary ints.
    Returns {metric_key: {"significant": bool, "reliable": bool}} for:
      accuracy, f1_macro, f1_pos, f1_neg,
      precision_pos, precision_neg, recall_pos, recall_neg
    """
    if (
            not base_pred_labels
            or not abl_pred_labels
            or len(base_pred_labels) != len(abl_pred_labels)
    ):
        return {}

    base_arr = np.array(base_pred_labels, dtype=float)
    abl_arr = np.array(abl_pred_labels, dtype=float)
    base_pred = base_arr[:, 0]
    abl_pred = abl_arr[:, 0]

    if not np.array_equal(base_arr[:, 1], abl_arr[:, 1]):
        raise ValueError("base_target and abl_target must be identical for paired bootstrap")
    target = base_arr[:, 1]

    data_tuple = (base_pred, abl_pred, target)
    label = f" [{context_label}]" if context_label else ""

    return _run_bootstrap_multi(
        data_tuple,
        _pointwise_all_statistic,
        _POINTWISE_METRIC_KEYS,
        n_resamples=n_resamples,
        random_state=random_state,
        label=label,
        _logger=_logger,
        alternative=alternative,
    )
