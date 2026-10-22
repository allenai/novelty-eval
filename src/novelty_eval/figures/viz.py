"""
Heatmap visualization for ablation delta comparisons.

Generates matplotlib figures showing metric deltas (ablation − baseline)
across models, with optional bootstrap significance annotations.
"""
import numpy as np
from pathlib import Path

from novelty_eval.figures.common import PDF_RCPARAMS


def _generate_overview_heatmap(
    ablation_labels: list[str],
    metric_specs: list[tuple[str, str]],
    delta_matrix,
    sig_matrix,
    out_path: Path,
    model: str = "",
    n_pointwise: int = 0,
    base_matrix=None,
    abs_matrix=None,
    reliable_matrix=None,
) -> "Path | None":
    """
    Overview heatmap: rows = ablations, columns = metrics.

    Rendered as two side-by-side GridSpec panels — left for pointwise metrics,
    right for pairwise — sharing the y-axis (ablation labels).  Cells show
    delta vs baseline with (base → ablation) below; * marks a significant delta,
    ? marks a cell where the bootstrap CI was unreliable.

    Args:
        ablation_labels: display name for each ablation row.
        metric_specs: [(display_label, setup), ...] ordered same as delta_matrix columns.
        delta_matrix: (n_ablations × n_metrics) float array, NaN = missing.
        sig_matrix: (n_ablations × n_metrics) bool array, True = significant.
        out_path: where to write the PNG.
        model: judge-model name, used in the figure title.
        n_pointwise: number of leading columns that belong to pointwise setup.
        base_matrix: (n_ablations × n_metrics) baseline absolute values (optional).
        abs_matrix: (n_ablations × n_metrics) ablation absolute values (optional).
        reliable_matrix: (n_ablations × n_metrics) bool array, False = CI unreliable (optional).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.gridspec import GridSpec
    except ImportError:
        return None

    n_abl = len(ablation_labels)
    n_metrics = len(metric_specs)
    n_pairwise = n_metrics - n_pointwise
    if n_abl == 0 or n_metrics == 0:
        return None

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg_delta", ["#d64045", "#f7f4ef", "#3d9970"], N=256
    )
    cmap.set_bad(color="#e2e8f0")

    abs_max = float(np.nanmax(np.abs(delta_matrix))) if not np.all(np.isnan(delta_matrix)) else 0.1
    abs_max = max(abs_max, 0.01)
    vmin, vmax = -abs_max, abs_max

    fig_w = max(6.0, n_metrics * 1.9 + 2.5)
    fig_h = max(3.0, n_abl * 0.7 + 2.2)

    has_both = n_pointwise > 0 and n_pairwise > 0
    if has_both:
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = GridSpec(1, 2, width_ratios=[n_pointwise, n_pairwise], figure=fig, wspace=0.06)
        ax_ptw = fig.add_subplot(gs[0])
        ax_pair = fig.add_subplot(gs[1], sharey=ax_ptw)
        sections = [
            (ax_ptw, 0, n_pointwise, "Pointwise"),
            (ax_pair, n_pointwise, n_metrics, "Pairwise"),
        ]
    else:
        fig, ax_only = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
        sections = [(ax_only, 0, n_metrics, "")]

    show_transition = base_matrix is not None and abs_matrix is not None

    def _render_section(ax, col_start, col_end, section_label):
        ax.set_facecolor("white")
        n_cols = col_end - col_start
        sec_delta    = delta_matrix[:, col_start:col_end]
        sec_sig      = sig_matrix[:, col_start:col_end]
        sec_reliable = reliable_matrix[:, col_start:col_end] if reliable_matrix is not None else None
        sec_base = base_matrix[:, col_start:col_end] if show_transition else None
        sec_abs  = abs_matrix[:, col_start:col_end]  if show_transition else None
        col_labels = [metric_specs[j][0] for j in range(col_start, col_end)]

        ax.imshow(sec_delta, cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect="auto", interpolation="nearest")

        for i in range(n_abl):
            for j in range(n_cols):
                v = sec_delta[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#bbbbbb")
                else:
                    norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    text_color = "white" if norm_v < 0.25 or norm_v > 0.82 else "#1a1a1a"
                    if sec_reliable is not None and not sec_reliable[i, j]:
                        sig_marker = "?"
                    elif sec_sig[i, j]:
                        sig_marker = "*"
                    else:
                        sig_marker = ""
                    cell_text = f"{'+' if v >= 0 else ''}{v * 100:.1f}%{sig_marker}"
                    y_delta = i - 0.13 if show_transition else i
                    ax.text(j, y_delta, cell_text, ha="center", va="center",
                            fontsize=9, color=text_color, fontweight="bold")
                    if show_transition:
                        bv = sec_base[i, j]
                        av = sec_abs[i, j]
                        if not (np.isnan(bv) or np.isnan(av)):
                            trans = f"({bv * 100:.1f} → {av * 100:.1f})"
                            ax.text(j, i + 0.18, trans, ha="center", va="center",
                                    fontsize=7, color=text_color, alpha=0.88,
                                    fontstyle="italic")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=9, rotation=0, ha="center")
        ax.tick_params(length=0)
        if section_label:
            ax.set_title(section_label, fontsize=11, fontweight="bold", pad=18)

        for x in range(n_cols + 1):
            ax.axvline(x - 0.5, color="white", linewidth=1.5)
        for y in range(n_abl + 1):
            ax.axhline(y - 0.5, color="white", linewidth=1.5)

        if col_start == 0:
            ax.set_yticks(range(n_abl))
            ax.set_yticklabels(ablation_labels, fontsize=9)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    for ax, col_start, col_end, section_label in sections:
        _render_section(ax, col_start, col_end, section_label)

    fig.suptitle(model or "Ablation Overview", fontsize=12, fontweight="bold", y=1.06)
    fig.text(
        0.5, -0.01,
        "* significant (95% BCa CI); ? = CI unreliable — corpus-level paired bootstrap, n=100,000",
        ha="center", va="top", fontsize=7.5, color="#666666", style="italic",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _generate_unified_heatmap(
    setup: str,
    panels: list[dict],
    metric_labels: dict[str, str],
    all_models: list[str],
    out_path: "Path",
    formats=None,
) -> "Path | None":
    """
    Unified side-by-side heatmap: one panel per ablation, all on one figure.

    Each panel shows rows = judge models, columns = metrics (delta vs baseline),
    using a **shared** colour scale so cells are visually comparable across panels.

    panels: list of dicts, each with:
        desc (str)               — ablation description used as panel title
        baseline_metrics (dict)  — model → {metric_key → value}
        ablation_metrics (dict)  — model → {metric_key → value}
        bootstrap_results (dict) — (model, metric_key) → bootstrap result dict
    all_models: ordered list of judge models (rows, same for every panel).
    metric_labels: ordered dict of metric_key → display label (columns per panel).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return None

    if not panels or not metric_labels or not all_models:
        return None

    n_panels  = len(panels)
    n_models  = len(all_models)
    metric_keys = list(metric_labels.keys())
    n_metrics = len(metric_keys)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        **PDF_RCPARAMS,
    })

    # ── Build one matrix per panel ────────────────────────────────────────────
    all_delta_vals: list[float] = []
    panel_data: list[tuple] = []  # (delta, base, abs_, sig, relbl, bs)
    for panel in panels:
        bm_all = panel.get("baseline_metrics", {})
        am_all = panel.get("ablation_metrics", {})
        bs     = panel.get("bootstrap_results", {})

        delta = np.full((n_models, n_metrics), np.nan)
        base  = np.full((n_models, n_metrics), np.nan)
        abs_  = np.full((n_models, n_metrics), np.nan)
        sig   = np.zeros((n_models, n_metrics), dtype=bool)
        relbl = np.ones((n_models, n_metrics),  dtype=bool)

        for i, model in enumerate(all_models):
            bm = bm_all.get(model, {})
            am = am_all.get(model, {})
            for j, mk in enumerate(metric_keys):
                bv = bm.get(mk)
                av = am.get(mk)
                if bv is not None and av is not None:
                    d = av - bv
                    delta[i, j] = d
                    base[i, j]  = bv
                    abs_[i, j]  = av
                    all_delta_vals.append(d)
                    bs_e = bs.get((model, mk), {})
                    sig[i, j]   = bool(bs_e.get("significant", False))
                    relbl[i, j] = bool(bs_e.get("reliable", True))

        panel_data.append((delta, base, abs_, sig, relbl, bs))

    # ── Shared colour scale across all panels ─────────────────────────────────
    abs_max = max(abs(v) for v in all_delta_vals) if all_delta_vals else 0.1
    abs_max = max(abs_max, 0.01)
    vmin, vmax = -abs_max, abs_max

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg_delta", ["#d64045", "#f7f4ef", "#3d9970"], N=256
    )
    cmap.set_bad(color="#e2e8f0")

    # ── Figure layout ─────────────────────────────────────────────────────────
    panel_w = max(1.4, n_metrics * 1.6 + 0.2)

    # Wrap titles to fit panel width before creating the figure so we can
    # reserve the correct amount of vertical space for multi-line titles.
    import textwrap as _tw
    _chars_per_line = max(12, int(panel_w * 7.0))  # ~7 chars per inch at fontsize 9
    wrapped_titles = [_tw.fill(p["desc"], width=_chars_per_line) for p in panels]
    _max_title_lines = max((t.count("\n") + 1 for t in wrapped_titles), default=1)

    fig_w   = panel_w * n_panels + 2.2 + (n_panels - 1) * 0.15
    # Reserve 0.30 in per extra title line beyond the first
    fig_h   = max(2.5, n_models * 0.75 + 2.2 + max(0, _max_title_lines - 1) * 0.30)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_w, fig_h),
        sharey=True,
        facecolor="white",
        squeeze=False,
    )
    axes = axes[0]  # shape → (n_panels,)

    has_bootstrap = any(bool(p.get("bootstrap_results")) for p in panels)

    for p_idx, (ax, panel) in enumerate(zip(axes, panels)):
        ax.set_facecolor("white")
        delta, base, abs_, sig, relbl, bs = panel_data[p_idx]

        ax.imshow(delta, cmap=cmap, vmin=vmin, vmax=vmax,
                  aspect="auto", interpolation="nearest")

        for i in range(n_models):
            for j in range(n_metrics):
                v   = delta[i, j]
                mk  = metric_keys[j]
                mdl = all_models[i]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color="#bbbbbb")
                else:
                    norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    tc = "white" if norm_v < 0.25 or norm_v > 0.82 else "#1a1a1a"
                    bs_e = bs.get((mdl, mk), {})
                    star = ("?" if not relbl[i, j]
                            else ("*" if sig[i, j] else ""))
                    bv, av = base[i, j], abs_[i, j]
                    if "n_ties" in mk or "support" in mk:
                        top = f"{'+' if v > 0 else ''}{v:.1f}{star}"
                        bot = f"({bv:.1f}→{av:.1f})"
                    else:
                        top = f"{'+' if v >= 0 else ''}{v * 100:.1f}%{star}"
                        bot = f"({bv * 100:.1f}→{av * 100:.1f})"
                    ax.text(j, i - 0.13, top, ha="center", va="center",
                            fontsize=8, color=tc, fontweight="bold")
                    if not (np.isnan(bv) or np.isnan(av)):
                        ax.text(j, i + 0.18, bot, ha="center", va="center",
                                fontsize=6.5, color=tc, alpha=0.88,
                                fontstyle="italic")

        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(
            [metric_labels[k] for k in metric_keys],
            fontsize=8, rotation=0, ha="center",
        )
        ax.tick_params(length=0)
        ax.set_title(wrapped_titles[p_idx], fontsize=9, fontweight="bold", pad=6)

        for x in range(n_metrics + 1):
            ax.axvline(x - 0.5, color="white", linewidth=1.5)
        for y in range(n_models + 1):
            ax.axhline(y - 0.5, color="white", linewidth=1.5)

        if p_idx == 0:
            ax.set_yticks(range(n_models))
            ax.set_yticklabels(all_models, fontsize=9)
            ax.set_ylabel("Judge Model", fontsize=11, fontweight="bold", labelpad=8)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Push suptitle above the tallest (possibly multi-line) panel title
    _suptitle_y = 1.03 + max(0, _max_title_lines - 1) * 0.04
    fig.suptitle(
        f"{setup.capitalize()} — Unified Ablation Comparison",
        fontsize=12, fontweight="bold", y=_suptitle_y,
    )
    if has_bootstrap:
        fig.text(
            0.5, -0.01,
            "* significant (95% BCa CI); ? = CI unreliable — "
            "corpus-level paired bootstrap, n=100,000",
            ha="center", va="top", fontsize=7.5, color="#666666", style="italic",
        )

    plt.subplots_adjust(wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets = [out_path]
    for ext in (formats or []):
        cand = out_path.with_suffix("." + str(ext).lstrip("."))
        if cand not in targets:
            targets.append(cand)
    for target in targets:
        fig.savefig(str(target), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _generate_aggregated_delta_heatmap(
    setup: str,
    metric_labels: dict[str, str],
    models: list[str],
    baseline_metrics: dict[str, dict],
    ablation_metrics: dict[str, dict],
    out_path: Path,
    after_desc: str = "",
    before_desc: str = "",
    ablation_name: str = "",
    bootstrap_results: dict[tuple[str, str], dict] = None,
) -> Path | None:
    """
    Generate a single heatmap for a setup, showing deltas for all metrics across all models.
    Rows = Models, Columns = Metrics.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return None

    if not models or not metric_labels:
        return None

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })

    n_models = len(models)
    metric_keys = list(metric_labels.keys())
    n_metrics = len(metric_keys)

    # Build delta matrix (models x metrics), absolute value matrix, and baseline matrix
    delta_mat = np.zeros((n_models, n_metrics))
    abs_mat = np.zeros((n_models, n_metrics))
    base_mat = np.zeros((n_models, n_metrics))
    for i, model in enumerate(models):
        bm = baseline_metrics.get(model, {})
        am = ablation_metrics.get(model, {})
        for j, mk in enumerate(metric_keys):
            bv = bm.get(mk)
            av = am.get(mk)
            if bv is not None and av is not None:
                delta_mat[i, j] = av - bv
                abs_mat[i, j] = av
                base_mat[i, j] = bv
            else:
                delta_mat[i, j] = np.nan
                abs_mat[i, j] = np.nan
                base_mat[i, j] = np.nan

    display_mat = delta_mat
    display_abs = abs_mat
    display_base = base_mat
    display_keys = metric_keys
    display_labels = metric_labels
    n_display = len(display_keys)

    cmap_delta = mcolors.LinearSegmentedColormap.from_list(
        "rg_delta", ["#d64045", "#f7f4ef", "#3d9970"], N=256
    )
    cmap_delta.set_bad(color="#e2e8f0")

    # Symmetric scale around 0, based on max absolute delta across all metrics
    delta_abs_max = float(np.nanmax(np.abs(delta_mat))) if not np.all(np.isnan(delta_mat)) else 0.1
    delta_abs_max = max(delta_abs_max, 0.01)
    vmin, vmax = -delta_abs_max, delta_abs_max

    # Dynamically size figure based on number of models and metrics
    fig_w = max(4.5, n_display * 1.6 + 1.5)
    fig_h = max(2.5, n_models * 0.75 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    plt.subplots_adjust(top=0.92, bottom=0.10)

    im = ax.imshow(display_mat, cmap=cmap_delta, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")

    # Add text labels to each cell
    for i in range(n_models):
        for j in range(n_display):
            v = display_mat[i, j]
            abs_v = display_abs[i, j]
            base_v = display_base[i, j]
            model = models[i]
            mk = display_keys[j]

            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="#bbbbbb")
            else:
                norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                # Use white text on dark backgrounds
                text_color = "white" if norm_v < 0.25 or norm_v > 0.82 else "#1a1a1a"

                sig_stars = ""
                if bootstrap_results and (model, mk) in bootstrap_results:
                    res = bootstrap_results[(model, mk)]
                    if not res.get("reliable", True):
                        sig_stars = "?"
                    elif res.get("significant"):
                        sig_stars = "*"

                if "n_ties" in mk or "support" in mk:
                    delta_label = f"{'+' if v > 0 else ''}{v:.1f}{sig_stars}"
                    abs_label = f"({base_v:.1f} → {abs_v:.1f})"
                else:
                    delta_label = f"{'+' if v > 0 else ''}{v*100:.1f}%{sig_stars}"
                    abs_label = f"({base_v*100:.1f} → {abs_v*100:.1f})"

                # Delta (large)
                ax.text(j, i - 0.12, delta_label, ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold")
                # Result (smaller, italic transition)
                ax.text(j, i + 0.18, abs_label, ha="center", va="center",
                        fontsize=7.0, color=text_color, alpha=0.9, fontstyle="italic")

    # Setup axes
    ax.set_xticks(range(n_display))
    ax.set_xticklabels([display_labels[k] for k in display_keys],
                       fontsize=10, rotation=0, ha="center")
    ax.tick_params(length=0)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Judge Model", fontsize=13, fontweight="bold", labelpad=8)

    # Grid lines to separate cells
    for x in range(n_display + 1):
        ax.axvline(x - 0.5, color="white", linewidth=2.0)
    for y in range(n_models + 1):
        ax.axhline(y - 0.5, color="white", linewidth=2.0)

    if bootstrap_results:
        fig.text(
            0.5, 0.0,
            "* significant (95% BCa CI); ? = CI unreliable — corpus-level paired bootstrap, n=100,000",
            ha="center", va="top",
            fontsize=8, color="#666666", style="italic",
            transform=fig.transFigure,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
