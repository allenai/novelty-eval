"""
Shared figure logic
===================
What every figure entry point — the notebooks and the remaining scripts — has to
agree on: where a merge's saved chart data lives, which cells are artefacts and
must be blanked, how the same ablation is recognised across differently-named
tracks, and the delta palette.

This module holds the things that would silently drift apart if each entry
point kept its own copy, so a figure regenerated from a notebook and one
regenerated from a script cannot disagree about the numbers they may show.

`TwoTrackFigure` at the bottom is the one full figure that lives here rather
than with its caller: two notebooks draw it over different experiments, so the
drawing is shared and only the wording is theirs.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

# Type 3 fonts are rejected by many venues; 42 embeds TrueType instead.
PDF_RCPARAMS = {"pdf.fonttype": 42, "ps.fonttype": 42}

# The paper's typeface rather than the plotting library's: matplotlib's DejaVu
# Sans reads as a default, and a figure set in it looks unlike the page it sits
# on. Names are tried in turn, so a machine without the first still gets a
# grotesque instead of falling back to DejaVu. The same stack as the other
# figure notebooks, so no two figures in the paper are set differently.
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

# Diverging red → off-white → green, for a delta centred on zero.
DELTA_COLORS = ["#d64045", "#f7f4ef", "#3d9970"]

# claude-sonnet-4-5 has no reasoning-effort control, so any number it produces
# under that ablation is an artefact of the harness rather than a result: every
# figure blanks the cell instead of drawing it.
UNSUPPORTED_CELLS = {("low_judge_reasoning", "claude-sonnet-4-5")}

# The same conceptual ablation is named per track *and* per setup when its
# instance set differs: convert_to_plan_form_pairwise_human in one place,
# convert_to_plan_form_pointwise_vanilla in another. Rows are matched on the
# name with these trailing tokens stripped, so one entry covers both setups.
TRACK_SUFFIXES = ("human", "vanilla", "hvh", "vanilla_ai", "ai",
                  "pairwise", "pointwise", "ranking")


def chart_data_path(merge_dir, setup: str, variant: str = "filtered") -> Path:
    """
    Locate unified_<setup>[_<variant>].json under a merge output directory.

    Accepts either the merge directory or its unified_charts/ subdirectory, since
    older merges wrote the files at the top level.
    """
    merge_dir = Path(merge_dir)
    base = merge_dir / "unified_charts" if (merge_dir / "unified_charts").is_dir() \
        else merge_dir
    suffix = "" if variant == "unfiltered" else f"_{variant}"
    path = base / f"unified_{setup}{suffix}.json"
    # The merge writes a _filtered file only when the blocklist removed something.
    # On the released data/ (already filtered) it never does, so unfiltered is filtered.
    if variant == "filtered" and not path.exists():
        path = base / f"unified_{setup}.json"
    return path


def row_key(name: str, suffixes: tuple[str, ...] = TRACK_SUFFIXES) -> str:
    """
    Strip trailing track/setup tokens so per-track and per-setup variants of one
    ablation share a row key. Applied repeatedly: `..._pairwise_human` loses both.

    Never strips the whole name — a token that is the entire name is left alone.
    """
    ordered = sorted(suffixes, key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        for suffix in ordered:
            token = "_" + suffix
            if name.endswith(token) and len(name) > len(token):
                name = name[: -len(token)]
                changed = True
                break
    return name


def drop_unsupported_cells(data: dict) -> dict:
    """Blank cells for judges that cannot honour the ablation being drawn."""
    for name, panel in data.get("panels", {}).items():
        key = row_key(name)
        for ablation, model in UNSUPPORTED_CELLS:
            if key == ablation:
                panel.get("ablation_metrics", {}).pop(model, None)
    return data


def delta_cmap(bad: str = "#e2e8f0"):
    """The diverging delta colormap, with `bad` for cells that have no number."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("delta", DELTA_COLORS, N=256)
    cmap.set_bad(bad)
    return cmap


def save_figure(fig, out_path, formats=None, dpi: int = 300,
                **savefig_kw) -> "list[Path]":
    """
    Write a figure to out_path plus any extra formats (by extension swap).

    Returns every path written. PDF output is vector; dpi only affects rasters.
    Extra keywords go to `savefig` unchanged, and the same ones to every format,
    so a figure trimmed to its ink (`bbox_inches="tight", pad_inches=EDGE_IN`)
    is trimmed identically in the PDF and the PNG.
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    targets = [out_path]
    for ext in (formats or []):
        cand = out_path.with_suffix("." + str(ext).lstrip("."))
        if cand not in targets:
            targets.append(cand)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for target in targets:
        fig.savefig(str(target), dpi=dpi, facecolor="white", **savefig_kw)
        written.append(target)
    plt.close(fig)
    return written


def load_config(path=None) -> dict:
    """
    Read a YAML config into a dict; None means "no config given", i.e. {}.

    Exits with a readable message rather than a traceback — a mistyped path or a
    stray tab in the YAML is a user error, not a bug worth a stack trace.
    """
    if path is None:
        return {}
    path = Path(path)
    try:
        import yaml
    except ImportError:
        raise SystemExit("Reading a config file needs PyYAML installed.")
    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"config is not valid YAML ({path}): {exc}")


# ---------------------------------------------------------------------------
# The two-track ablation figure
# ---------------------------------------------------------------------------
# Rows = ablations, columns = judge models, one column block per data track.
# Drawn twice from one table: once as the change each ablation made, once as
# the score it landed on.

# Shorter judge names for the column headers.
MODEL_LABELS = {"claude-sonnet-4-5": "sonnet-4-5",
                "claude-opus-4-5": "opus-4-5",
                "claude-opus-4-6": "opus-4-6"}

# The baseline strip of a delta figure, a shade of its own: it is not a delta,
# so it must never read as one of zero.
BASELINE_BG = "#f4f6f8"

# Chance on a two-way decision, and the point the red→green scale is white at
# when the figure colours scores rather than changes.
ABSOLUTE_CENTER = 50.0

# White text only on the dark ends of the scale, which is red→green either way.
WHITE_TEXT_BAND = (0.25, 0.82)

# Layout in inches, so a cell is the size it says it is. Every gap on the page
# is one number here: no layout engine to reverse-engineer when it looks wrong.
CELL_IN = 0.29                    # height of one ablation row: two lines of cell
                                  # text, the change over the score it landed on
STRIP_ROWS = 0.56                 # baseline strip height, in rows — shorter than
                                  # a data row, so it reads as a header
GUTTER_IN = 0.17                  # between the two blocks
MARGIN_IN = {"left": 1.51,        # room for the row names and the axis title,
                                  # resized at draw time to what they take
                                  # top: the block title; bottom: the judge
                                  # names and the axis title under them
                                  # Both are what their type measures plus the
                                  # gap it needs, and nothing over: the figure
                                  # ends EDGE_IN past its outermost ink on
                                  # every side, so a trimmed edge is the same
                                  # width all round. What bottom holds over the
                                  # type is the air between the judge names and
                                  # the row under them, near enough the gap the
                                  # title's `pad` leaves above the blocks: the
                                  # grid sits in the same white top and bottom
             "right": 0.04, "top": 0.18, "bottom": 0.34}
EDGE_IN = 0.04                    # gap kept between the longest name and the
                                  # edge, and what every other edge leaves too:
                                  # the figure is trimmed, not padded, so the
                                  # page around it is the document's to set
XLABEL_IN = 0.04                  # the column axis title, up from the bottom

# A row group's bracket, drawn in the left margin outside the row names. Every
# offset is in inches from the left block's edge, measured outwards: names,
# gap, bracket. The group's own name sits *on* the bracket, in a chip that
# hides the line behind it — so it costs the margin only the width of the chip
# rather than a column of its own, and cannot be read as belonging to a row.
# The chip is what makes the interruption read as deliberate, so this is the
# one shaded label left on the page; the block titles are plain text.
GROUP_GAP_IN = 0.06               # between the longest row name and the bracket
GROUP_TICK_IN = 0.045             # the bracket's end ticks, pointing at the rows
GROUP_LABEL_PAD = 0.16            # the chip's padding, in units of its font size
GROUP_LABEL_ROUND = 0.16          # its corner radius, same units
GROUP_INSET = 0.10                # rows the bracket stops short of its span, so
                                  # it reads as a span rather than a cell border
GROUP_LINE_MIN_IN = 0.08          # bracket left visible either side of the chip;
                                  # a group too short for it is named beside it
GROUP_COLOUR = "#8a94a6"
GROUP_LABEL_BG = "#eef1f5"        # the chip behind a group name, as the titles
LABELPAD_IN = 5 / 72              # the axis title's labelpad, in inches


@dataclass
class TwoTrackFigure:
    """
    One two-track ablation figure: what it says, and how it is drawn.

    Everything that differs between experiments is a field, so nothing in the
    drawing knows which one it is drawing. `table` is the whole data path — one
    tidy row per (track, ablation, judge), everything already in points — and
    the two draw methods are two readings of that table: `draw` colours the
    change an ablation made, `draw_absolute` the score it landed on. Same
    layout either way, so the pair can be read side by side.

    tracks: (block title, merge dir under output/ablation_sweeps, name token),
        left to right. The token fills {track} in a row name.
    rows: (panel name, row label), top to bottom, with an optional third
        element naming the setups the row belongs to — an ablation only one
        setup was run for is left out of the other rather than drawn as a row
        of dashes. {setup} and {track} expand in the panel name, for ablations
        whose instance set differs per track or per setup.
    metrics: the metric to read per setup — the two setups score different
        things. It lands in the filename, never on the figure, so the caption
        has to say which it is.
    row_groups: (group label, panel names in it) — a bracket in the left margin
        naming what its rows have in common. Named by panel name rather than by
        position, so a setup that draws only some of a group's rows brackets
        what it actually drew; a group whose rows are not drawn together is a
        mistake in the figure, not something to draw around, so it raises. A
        group label of "" draws the bracket unnamed, for a group too short to
        set the name beside.
    """

    root: Path
    figure: str                       # output subdirectory, and the filename
                                      # prefix unless `file_prefix` gives one
    tracks: Sequence[tuple[str, str, str]]
    rows: Sequence[tuple]             # (panel name, row label[, setups])
    models: Sequence[str]
    metrics: Mapping[str, str]
    row_axis_label: str
    row_groups: Sequence[tuple] = ()  # (group label, panel names in it)
    baseline_label: str = "baseline"  # the un-ablated strip, named like a row
    col_axis_label: str = "Judge model"
    sig_note: str = "* p < 0.05 (two-sided paired bootstrap)"
                                      # the key to the marked cells, set in the
                                      # bottom margin beside the column title.
                                      # Delta reading only — the absolute one
                                      # marks nothing — and "" drops it
    file_prefix: str = ""             # what the filenames start with, when that
                                      # is not the subdirectory's name: two
                                      # readings of one experiment belong in one
                                      # directory and have to be told apart in
                                      # it. "" prefixes them with `figure`
    variant: str = "filtered"         # which accuracy report: filtered | unfiltered
    model_labels: Mapping[str, str] = field(
        default_factory=lambda: dict(MODEL_LABELS))
    width: float = 7.1                # figure width, inches

    def rows_for(self, setup: str) -> list:
        """The (panel name, row label) rows one setup draws, top to bottom."""
        return [(name, label) for name, label, *setups in self.rows
                if not setups or setup in setups[0]]

    def labels_for(self, setup: str) -> list:
        """Row labels of one setup, top to bottom."""
        return [label for _, label in self.rows_for(setup)]

    def group_spans(self, setup: str) -> list:
        """
        (group label, first row, last row) for the groups one setup draws.

        Row indices are into `rows_for(setup)`, so a group the setup only
        partly drew brackets the part it drew. A group with no drawn rows is
        left out; one whose drawn rows are not adjacent raises, because a
        bracket over them would claim a grouping the figure does not have.
        """
        drawn = [name for name, _ in self.rows_for(setup)]
        spans = []
        for label, names in self.row_groups:
            at = [i for i, name in enumerate(drawn) if name in set(names)]
            if not at:
                continue
            if at != list(range(at[0], at[-1] + 1)):
                raise ValueError(
                    f"row group {label!r} is not drawn as adjacent rows in "
                    f"{setup}: rows {at}. Reorder `rows` so the group is "
                    f"contiguous, or drop the group.")
            spans.append((label, at[0], at[-1]))
        return spans

    @property
    def judges(self) -> list:
        """Column headers, left to right."""
        return [self.model_labels.get(m, m) for m in self.models]

    def panels(self, merge_dir, setup: str) -> dict:
        """The panel dict of one merge's saved chart data."""
        path = chart_data_path(Path(self.root) / "output/ablation_sweeps" / merge_dir,
                               setup, self.variant)
        return json.loads(path.read_text())["panels"]

    def table(self, setup: str):
        """One row per (track, ablation, judge). Scores and deltas are in points."""
        import pandas as pd

        metric = self.metrics[setup]
        recs = []
        for title, merge_dir, token in self.tracks:
            ps = self.panels(merge_dir, setup)
            for name, label in self.rows_for(setup):
                panel = ps.get(name.format(setup=setup, track=token))
                for model in self.models:
                    if panel is None or (name, model) in UNSUPPORTED_CELLS:
                        base = abl = sig = None
                    else:
                        base = panel["baseline_metrics"].get(model, {}).get(metric)
                        abl = panel["ablation_metrics"].get(model, {}).get(metric)
                        # Bootstrap results are keyed "<model>||<metric>" on disk.
                        sig = panel["bootstrap_results"].get(
                            f"{model}||{metric}", {}).get("significant")
                    recs.append({
                        "track": title,
                        "ablation": label,
                        "judge": self.model_labels.get(model, model),
                        "baseline": None if base is None else base * 100,
                        "ablated": None if abl is None else abl * 100,
                        "delta": None if base is None or abl is None
                                 else (abl - base) * 100,
                        "significant": bool(sig),
                    })
        return pd.DataFrame(recs)

    def draw(self, setup: str) -> Path:
        """Draw the delta reading of one setup: save PDF + PNG, return the PNG."""
        return self._draw(setup, absolute=False)

    def draw_absolute(self, setup: str, pad: float = 2.0,
                      center: float = ABSOLUTE_CENTER) -> Path:
        """
        Draw the absolute reading of one setup: the same grid on the same
        red→green scale, but a cell is the score itself rather than the change,
        and the baseline is an ordinary row of it.

        `center` is the score the scale is white at, red below and green above;
        `pad` is the headroom in points left past the extreme cells, so neither
        end of the palette is reached flat.
        """
        return self._draw(setup, absolute=True, pad=pad, center=center)

    def _draw(self, setup: str, absolute: bool, pad: float = 2.0,
              center: float = ABSOLUTE_CENTER) -> Path:
        """
        One `pcolormesh` per track, then the cell text on top of it.

        Every cell prints its number, so colour is never the only encoding, and
        both readings share one scale across the blocks: a colour means the same
        thing left and right.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.transforms import blended_transform_factory

        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": FONT_STACK, **PDF_RCPARAMS})
        df = self.table(setup)
        labels, judges = self.labels_for(setup), self.judges
        cmap = delta_cmap()

        if absolute:
            # One scale for every number on the page, baseline row included, so
            # a shade means the same score in either block. Kept symmetric about
            # `center`, which is what makes red mean "below chance" rather than
            # "low in whatever range this figure happens to cover".
            vals = pd.concat([df["ablated"], df["baseline"]]).astype(float)
            span = np.nanmax(np.abs(vals - center)) + pad
            vmin, vmax = center - span, center + span
        else:
            span = max(1.0, np.nanmax(np.abs(df["delta"].astype(float))))
            vmin, vmax = -span, span

        def text_colour(v):
            """Cell text: white only on the dark ends of the scale."""
            shade = (v - vmin) / (vmax - vmin)
            dark = shade < WHITE_TEXT_BAND[0] or shade > WHITE_TEXT_BAND[1]
            return "white" if dark else "#1a1a1a"

        # The delta reading measures against the baseline, so the baseline sits
        # outside the grid as a strip; the absolute reading has one kind of
        # number throughout, so there the baseline is simply the top row.
        strip_rows = 1.0 if absolute else STRIP_ROWS
        strip_top = 1 - strip_rows

        grid_in = CELL_IN * (len(labels) + strip_rows)
        fig_h = grid_in + MARGIN_IN["top"] + MARGIN_IN["bottom"]
        blocks_in = self.width - MARGIN_IN["left"] - MARGIN_IN["right"]
        block_in = (blocks_in - GUTTER_IN * (len(self.tracks) - 1)) / len(self.tracks)

        fig = plt.figure(figsize=(self.width, fig_h))
        gs = fig.add_gridspec(
            1, len(self.tracks), wspace=GUTTER_IN / block_in,
            left=MARGIN_IN["left"] / self.width,
            right=1 - MARGIN_IN["right"] / self.width,
            top=1 - MARGIN_IN["top"] / fig_h, bottom=MARGIN_IN["bottom"] / fig_h)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(self.tracks))]

        for ax, (title, _, _) in zip(axes, self.tracks):
            block = df[df["track"] == title].set_index(["ablation", "judge"])
            grid = lambda col: block[col].unstack().reindex(index=labels,
                                                            columns=judges)
            delta, ablated, sig = grid("delta"), grid("ablated"), grid("significant")
            # Every ablation in a track shares one baseline run, so it is one row
            # of its own rather than a number repeated down every row.
            base = grid("baseline").bfill().iloc[0]

            mesh = dict(cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolors="white", linewidth=1.2)
            if absolute:
                ax.pcolormesh(np.arange(len(judges) + 1),
                              np.arange(len(labels) + 2),
                              np.ma.masked_invalid(np.vstack(
                                  [base.to_numpy(float), ablated.to_numpy(float)])),
                              **mesh)
            else:
                ax.pcolormesh(np.arange(len(judges) + 1),
                              np.arange(1, len(labels) + 2),
                              np.ma.masked_invalid(delta.to_numpy(float)), **mesh)
                # Outside the grid, in a shade of its own, so the baseline never
                # reads as a delta of zero.
                ax.axhspan(strip_top, 0.995, facecolor=BASELINE_BG, zorder=0)
                ax.vlines(range(1, len(judges)), strip_top, 0.995,
                          color="white", linewidth=1.2)
            ax.set_xlim(0, len(judges))
            ax.set_ylim(len(labels) + 1, strip_top)      # baseline on top

            # Weight is significance and nothing else: a bold number is a
            # change the bootstrap called significant, so the eye finds them
            # without reading a single asterisk. Everything that is not such a
            # claim — the baseline, the absolute reading — is set regular.
            for j, judge in enumerate(judges):
                ax.text(j + 0.5, (strip_top + 1) / 2, f"{base[judge]:.1f}",
                        ha="center", va="center",
                        fontsize=7.0 if absolute else 6.4,
                        color=text_colour(base[judge]) if absolute else "#333333")
                for i, label in enumerate(labels):
                    d, a = delta.loc[label, judge], ablated.loc[label, judge]
                    lead = a if absolute else d      # the number that colours the cell
                    if not np.isfinite(float(lead if lead is not None else np.nan)):
                        ax.text(j + 0.5, i + 1.5, "—", ha="center", va="center",
                                fontsize=6, color="#999999")
                        continue
                    colour = text_colour(lead)
                    if absolute:
                        # The score alone. Significance is a claim about the
                        # change, so it is marked where the change is drawn.
                        ax.text(j + 0.5, i + 1.5, f"{a:.1f}", ha="center",
                                va="center", fontsize=7.0, color=colour)
                        continue
                    marked = bool(sig.loc[label, judge])
                    star = "*" if marked else ""
                    # The two lines sit close enough to read as one cell: the
                    # row is only just taller than they are, so the gap between
                    # rows has to stay the widest white on the page.
                    ax.text(j + 0.5, i + 1.31, f"{d:+.1f}{star}", ha="center",
                            va="center", fontsize=6.8,
                            fontweight="bold" if marked else "normal",
                            color=colour)
                    # Italic, so the absolute never reads as a second delta.
                    ax.text(j + 0.5, i + 1.69, f"({a:.1f})", ha="center",
                            va="center", fontsize=5.0,
                            color=colour, alpha=0.72)

            # Set horizontally: the longest judge name is narrower than a cell
            # at this size, so there is nothing for a rotation to rescue, and a
            # column header the eye has to tilt for is one it reads slowly.
            ax.set_xticks(np.arange(len(judges)) + 0.5, judges, fontsize=5.8)
            ax.set_yticks(
                [(strip_top + 1) / 2] + [i + 1.5 for i in range(len(labels))],
                [self.baseline_label] + labels, fontsize=7)
            ax.set_title(title, fontsize=7.0, fontweight="bold", pad=3.5)
            ax.tick_params(length=0, pad=2)
            for spine in ax.spines.values():
                spine.set_visible(False)

        # The row names, and both axis titles, belong to the left-hand block only.
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
        if not absolute:
            # The strip is labelled like a row, but lighter: it is not an
            # ablation. In the absolute figure it *is* an ordinary row.
            axes[0].get_yticklabels()[0].set(style="italic", color="#555555")
        axes[0].set_ylabel(self.row_axis_label, fontsize=7.2, fontweight="bold",
                           color="#333333", labelpad=5)

        # Group names are drawn now and positioned after the figure has been
        # widened, because their width is part of what it must be widened by.
        # The chip interrupts its own bracket, so a group too short to keep
        # some line either side of it has its name set beside the bracket
        # instead, unchipped, and the bracket drawn whole.
        spans = self.group_spans(setup)
        names = []                   # (text, beside) per span; text None if unnamed
        for label, i0, i1 in spans:
            if not label:
                names.append((None, False))
                continue
            # Set small: the chip hides the bracket behind it, and a short
            # group has to keep enough line either side of the name to read as
            # a span rather than as a label parked beside its rows.
            t = axes[0].text(0, (i0 + i1) / 2 + 1.5, label, rotation=90,
                             ha="center", va="center", fontsize=5.8,
                             fontweight="bold", color=GROUP_COLOUR,
                             clip_on=False, zorder=6,
                             bbox=dict(boxstyle=f"round,pad={GROUP_LABEL_PAD},"
                                                f"rounding_size={GROUP_LABEL_ROUND}",
                                       fc=GROUP_LABEL_BG, ec="none"))
            box = t.get_window_extent(fig.canvas.get_renderer())
            chip_in = 2 * GROUP_LABEL_PAD * t.get_fontsize() / 72
            bracket_in = (i1 - i0 + 1 - 2 * GROUP_INSET) * CELL_IN
            beside = box.height / fig.dpi + chip_in + 2 * GROUP_LINE_MIN_IN > bracket_in
            if beside:
                t.set_bbox(None)
            names.append((t, beside))

        def name_in(t, beside):
            """How far a name reaches out past the bracket, in inches."""
            w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
            if beside:
                return GROUP_TICK_IN + w
            # The chip straddles the bracket, so only its half sticks out past it.
            return (w + 2 * GROUP_LABEL_PAD * t.get_fontsize() / 72) / 2

        reach = [name_in(t, beside) for t, beside in names if t is not None]
        group_in = 0.0 if not spans else (
            GROUP_GAP_IN + GROUP_TICK_IN + max(reach, default=0.0))

        # The left margin is room for the row names and their brackets, and
        # `MARGIN_IN["left"]` is only the guess the cells were sized against.
        # Measure what the finished labels take and resize the figure by the
        # difference — out where a name would otherwise be cut, in where the
        # names are short and the slack would print as white. Either way the
        # figure ends EDGE_IN past the longest name and every cell keeps the
        # size it was given.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overrun = -min(ax.get_tightbbox(renderer).x0 for ax in axes) / fig.dpi
        left_in = MARGIN_IN["left"] + overrun + group_in + EDGE_IN
        width = self.width + (left_in - MARGIN_IN["left"])
        if abs(width - self.width) > 1e-3:
            fig.set_size_inches(width, fig_h)
            gs.update(left=left_in / width, right=1 - MARGIN_IN["right"] / width)

        # Now the block is its final width, so an offset in inches is a fixed
        # fraction of it: the bracket sits just outside the longest row name,
        # the name just outside the bracket, and the axis title outside both.
        if spans:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax0 = axes[0]
            block_px = ax0.get_window_extent(renderer).width
            rows_in = (ax0.get_window_extent(renderer).x0
                       - min(t.get_window_extent(renderer).x0
                             for t in ax0.get_yticklabels())) / fig.dpi
            out = lambda dx_in: -(rows_in + dx_in) * fig.dpi / block_px
            trans = blended_transform_factory(ax0.transAxes, ax0.transData)
            spine, tick = out(GROUP_GAP_IN + GROUP_TICK_IN), out(GROUP_GAP_IN)
            line = dict(transform=trans, color=GROUP_COLOUR, linewidth=0.8,
                        clip_on=False, solid_capstyle="round", zorder=5)
            for (_, i0, i1), (t, beside) in zip(spans, names):
                y0, y1 = i0 + 1 + GROUP_INSET, i1 + 2 - GROUP_INSET
                ax0.plot([spine, spine], [y0, y1], **line)
                ax0.plot([spine, tick], [y0, y0], **line)
                ax0.plot([spine, tick], [y1, y1], **line)
                if t is not None:
                    # Centred on the line it interrupts, or just outside it.
                    w = t.get_window_extent(renderer).width / fig.dpi
                    t.set_x(out(GROUP_GAP_IN + 2 * GROUP_TICK_IN + w / 2)
                            if beside else spine)
                    t.set_transform(trans)
            axes[0].yaxis.set_label_coords(out(group_in + LABELPAD_IN), 0.5)

        fig.text((left_in + blocks_in / 2) / width, XLABEL_IN / fig_h,
                 self.col_axis_label, ha="center", va="baseline", fontsize=7.2,
                 fontweight="bold", color="#333333")

        # The key to the asterisk, on the column title's own baseline: that row
        # of the bottom margin is empty either side of a centred title, so the
        # key costs the figure no height at all. It goes in the bottom-left
        # corner, under the row names, where a legend is looked for and where
        # nothing competes with it, and is set as dark as the axis titles and
        # as large as the cell numbers it explains, so it is read rather than
        # skipped. Only the delta reading marks anything to key.
        if not absolute and self.sig_note:
            fig.text(EDGE_IN / width, XLABEL_IN / fig_h,
                     self.sig_note, ha="left", va="baseline", fontsize=6.4,
                     color="#333333")

        prefix = self.file_prefix or self.figure
        stem = f"{prefix}_{setup}_{self.variant}_{self.metrics[setup]}"
        out_dir = Path(self.root) / "output/figures" / self.figure
        out_path = out_dir / f"{stem}{'_absolute' if absolute else ''}.pdf"
        save_figure(fig, out_path, formats=["png"])
        return out_path.with_suffix(".png")
