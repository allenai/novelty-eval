#!/usr/bin/env python3
"""
Judge Tie-Rate Strip Plot
=========================
One row per judge model, one dot per (ablation × track) configuration, placed at
the share of pairs that judge scored as a tie. It answers "which judge ties, and
how much does the ablation move that" in a single glance, which the per-ablation
tables cannot.

Reads each run's `accuracy_report.txt` straight from the sweep directories, so
the numbers are the same ones the merged reports print.

    python chart_tie_rates.py \
        --track "Human + generated"=src/novelty_eval/ablation/configs/merge/merge_config_vanilla.yaml \
        --track "Human-only"=src/novelty_eval/ablation/configs/merge/merge_config_hvh.yaml \
        --out output/figures/tie-rates/tie_rates.pdf --formats pdf png

A --track value is either a merge config (its `dirs:` list is used) or a sweep
directory, and several may be comma-separated. Prefer the config: the merged
`artifacts/` copies are lossy, a config also encodes which of two competing runs
of an ablation is the live one, and its `tracks:` scopes the scan to one data
type. That scoping matters when two tracks share a sweep directory — the runs
differ only by their track prefix, so a bare directory would give both tracks
the same numbers.

Options worth knowing:

    --ablations a b c     canonical names, in order. Default: the ablations
                          every track has, which keeps the judges comparable.
    --models m1 m2        row order top to bottom. Default: descending median
                          tie rate.
    --variant filtered    read filtered_accuracy_report.txt instead.
    --style acl           camera-ready styling: type at caption size, no
                          in-figure title block (the LaTeX caption carries it),
                          and a per-track median tick on each row.
    --config FILE         YAML with row_labels / model_labels / track_titles.
                          The same file the figure notebook reads.
    --csv PATH            write the plotted numbers. Defaults to the figure
                          path with a .csv suffix; --no-csv turns it off.
    --keep-unsupported    plot the configurations a judge cannot honour, which
                          common.py's UNSUPPORTED_CELLS drops.

Tie rates are only comparable between runs that sampled the judge the same way:
a tie needs an even number of votes to be reachable at all, so `unidirectional`
(one vote) can never tie and `mec_k_1` (two) ties whenever the two directions
disagree. Those are excluded by default; naming them in --ablations pulls them
back in and prints a warning, because the resulting dots do not mean the same
thing as the rest.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from novelty_eval.figures.common import (
    TRACK_SUFFIXES,
    UNSUPPORTED_CELLS,
    load_config,
    row_key,
    save_figure,
)

# Sampling-depth ablations. Their tie counts are on a different scale by
# construction (see the module docstring), so they stay out unless asked for.
DEPTH_ABLATIONS = ("unidirectional", "mec_k_1")

# Track colours. Blue/orange survives the common colour-vision deficiencies and
# stays separable in greyscale; keep them if you add a third track only after
# checking the new hue against both.
TRACK_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

DEFAULT_TITLE = "Tie rate by judge, one dot per configuration"

# Two presentations of the same plot. `screen` is the standalone figure: a bold
# title block, generous type, one dot per run. `acl` is what goes in the paper —
# same typeface, sized down to sit next to body text, no title (the LaTeX
# caption is the title), a median tick per track so a row can be read at a
# glance without counting dots.
STYLES = {
    "screen": {
        "rc": {"font.family": "sans-serif"},
        "titles": True,
        "medians": False,
        "row_h": 0.42,
        "label_w": 0.075,      # inches of left margin per character of label
        "left_base": 0.14,
        "bottom_pad": 0.62,
        "legend_pad": 0.34,
        "marker_size": 40,
        "marker_edge": 0.9,
        "max_offset": 0.17,
        "fs_ylabel": 10,
        "fs_xtick": 9,
        "fs_xlabel": 10,
        "fs_legend": 9.5,
        "legend_axespad": 0.5,
        "legend_colspace": 1.5,
        "xlabel": "share of pairs the judge tied",
        "labelpad": 7,
        "xtick_params": {},
        "row_line": {"color": "#dfe2e8", "lw": 0.8, "ls": "-"},
    },
    "acl": {
        "rc": {"font.family": "sans-serif"},
        "titles": False,
        "medians": True,
        "row_h": 0.33,
        "label_w": 0.068,
        "left_base": 0.12,
        "bottom_pad": 0.46,
        "legend_pad": 0.24,
        "marker_size": 21,
        "marker_edge": 0.6,
        "max_offset": 0.15,
        "fs_ylabel": 9,
        "fs_xtick": 8,
        "fs_xlabel": 9,
        "fs_legend": 8.5,
        "legend_axespad": 0.4,
        "legend_colspace": 1.4,
        "xlabel": "Tie rate (share of pairs judged a tie)",
        "labelpad": 5,
        "xtick_params": {"length": 3, "width": 0.7},
        "row_line": {"color": "#d9dde4", "lw": 0.6, "ls": (0, (1, 2))},
    },
}

_TIES_RE = re.compile(r"^Individual Run Ties:\s*\[([^\]]*)\]", re.M)
_SUPPORT_RE = re.compile(r"^Individual Run Support \(with ties\):\s*\[([^\]]*)\]", re.M)
_MODE_RE = re.compile(r"^Test Mode:\s*(\S+)", re.M)


# ---------------------------------------------------------------------------
# Reading runs
# ---------------------------------------------------------------------------

def _resolve_sweep_dirs(spec: str, repo_root: Path) -> list[tuple[Path, tuple[str, ...]]]:
    """Expand one --track path into (sweep directory, track prefixes) pairs.

    A merge config contributes its `dirs:` list; a directory contributes itself.
    A config's `tracks:` list rides along as a filter on the runs inside those
    directories — see `_scan`. A bare directory has no config to scope it, so it
    contributes every run it holds.
    """
    dirs: list[tuple[Path, tuple[str, ...]]] = []
    for raw in spec.split(","):
        p = Path(raw.strip())
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            raise SystemExit(f"--track path does not exist: {p}")
        if p.is_file():
            cfg = load_config(p)
            listed = cfg.get("dirs") or []
            if not listed:
                raise SystemExit(f"--track config has no `dirs:` entries: {p}")
            prefixes = tuple(str(t) for t in (cfg.get("tracks") or []))
            for d in listed:
                cand = Path(str(d).rstrip("/"))
                cand = cand if cand.is_absolute() else repo_root / cand
                if cand.exists():
                    dirs.append((cand, prefixes))
                else:
                    print(f"warning: {p.name} lists a missing directory, skipping: {d}",
                          file=sys.stderr)
        else:
            dirs.append((p, ()))
    return dirs


def _ablation_descriptions() -> dict:
    """Canonical name -> description, from ablations.yaml."""
    # The ablation registry stays with the ablation package.
    path = Path(__file__).resolve().parent.parent / "ablation" / "ablations.yaml"
    if not path.exists():
        return {}
    data = load_config(path).get("ablations") or {}
    out = {}
    for name, spec in data.items():
        if isinstance(spec, dict) and spec.get("description"):
            out[name] = spec["description"]
    return out


def _label_for(key: str, run_names: set, overrides: dict, descriptions: dict) -> str:
    """Prefer an explicit override, then ablations.yaml, then the bare key.

    Both are keyed inconsistently in practice — a config may label
    `convert_to_plan_form_pairwise_human` while the row groups under
    `convert_to_plan_form` — so each source is tried under every name the
    ablation is known by.
    """
    for source in (overrides, descriptions):
        if key in source:
            return source[key]
        for name in sorted(run_names):
            if name in source:
                return source[name]
    return key


def _parse_track(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f'--track expects "Display name"=path, got: {spec!r}')
    title, raw = spec.split("=", 1)
    return title.strip(), raw.strip()


def _first_int_list(pattern: re.Pattern, text: str) -> list[int] | None:
    m = pattern.search(text)
    if not m or not m.group(1).strip():
        return None
    try:
        return [int(float(x)) for x in m.group(1).split(",")]
    except ValueError:
        return None


# Judge names appear as a trailing `-<engine>` token on the run directory. Used
# only when a run ships no settings file at all.
_KNOWN_ENGINE = re.compile(
    r"-((?:claude|gpt|gemini|llama|mistral|o\d)[\w.\-]*)$", re.I)


def _engine_from_dir(leaf_name: str) -> str | None:
    m = _KNOWN_ENGINE.search(leaf_name)
    return m.group(1) if m else None


def _run_settings(leaf: Path, artifact: Path) -> dict:
    """Read a run's settings, whichever form this sweep wrote them in.

    Normal runs get `_run_config.yaml` beside the artifacts. Runs produced by
    derive_subrun_artifacts.py (mec_k_1, unidirectional) instead carry a small
    `debug_args.md` JSON stub, and record no sampling parameters at all — the
    caller has to fall back on the ablation name for those.
    """
    cfg_path = leaf / "_run_config.yaml"
    if cfg_path.exists():
        return load_config(cfg_path)
    stub = artifact / "debug_args.md"
    if stub.exists():
        import json
        text = stub.read_text(errors="replace").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return {}


def _scan(sweep_dir: Path, variant: str,
          track_prefixes: tuple[str, ...] = ()) -> list[dict]:
    """Collect every pairwise run under one sweep directory.

    `track_prefixes` keeps only runs whose directory carries one of those track
    prefixes, mirroring a merge config's `tracks:`. One sweep can hold several
    data types side by side under names that differ only by that prefix, so
    without the filter two --track arguments reading the same sweep collect the
    same runs and the tracks come out identical.
    """
    name = "accuracy_report.txt" if variant == "unfiltered" \
        else f"{variant}_accuracy_report.txt"
    found = []
    for report in sorted(sweep_dir.rglob(f"accuracy_test_artifacts/*/{name}")):
        artifact = report.parent
        leaf = artifact.parent.parent            # <ablation>-<engine>
        if track_prefixes and leaf.name.split("_", 1)[0] not in track_prefixes:
            continue
        cfg = _run_settings(leaf, artifact)
        engine = cfg.get("llm_engine") or _engine_from_dir(leaf.name)
        if not engine:
            print(f"warning: cannot tell which judge ran {leaf.name}, skipping.",
                  file=sys.stderr)
            continue
        text = report.read_text(errors="replace")
        mode = _MODE_RE.search(text)
        if (mode.group(1) if mode else cfg.get("test_mode")) != "pairwise":
            continue
        ties = _first_int_list(_TIES_RE, text)
        support = _first_int_list(_SUPPORT_RE, text)
        if not ties or not support:
            print(f"warning: no tie/support numbers in {report}, skipping.",
                  file=sys.stderr)
            continue

        # <track prefix>_<ablation>-<engine> -> <ablation>
        ablation = leaf.name
        if "_" in ablation:
            ablation = ablation.split("_", 1)[1]
        if ablation.endswith("-" + engine):
            ablation = ablation[: -len(engine) - 1]

        found.append({
            "ablation": ablation,
            "model": engine,
            # Several runs of one config are pooled, matching how the reports
            # average them, rather than averaging the per-run rates.
            "ties": sum(ties),
            "support": sum(support),
            "runs": len(ties),
            # Absent in derived runs; None means "not recorded", which is not
            # the same as the default and must not be reported as one.
            "mec_k": cfg.get("mec_k"),
            "bidirectional": cfg.get("bidirectional"),
            "artifact": artifact,
        })
    return found


def _collect(track_specs: list[str], variant: str, repo_root: Path,
             on_duplicate: str) -> list[tuple[str, dict]]:
    """Return [(track_title, {(ablation, model): record}), ...]."""
    tracks = []
    for spec in track_specs:
        title, raw = _parse_track(spec)
        cells: dict[tuple[str, str], dict] = {}
        for sweep, prefixes in _resolve_sweep_dirs(raw, repo_root):
            for rec in _scan(sweep, variant, prefixes):
                key = (rec["ablation"], rec["model"])
                prev = cells.get(key)
                if prev is None:
                    cells[key] = rec
                    continue
                # A sweep can hold the same run under two groupings (a
                # `baseline_reasoning/` copy beside the `..._current/` one).
                # Those agree to the number, so only a real disagreement — two
                # separate executions — is worth interrupting for.
                if (prev["ties"], prev["support"]) == (rec["ties"], rec["support"]):
                    continue
                msg = (f"{title}: {rec['ablation']} / {rec['model']} has two runs "
                       f"that disagree\n"
                       f"    {prev['ties']}/{prev['support']} ties  {prev['artifact']}\n"
                       f"    {rec['ties']}/{rec['support']} ties  {rec['artifact']}")
                if on_duplicate == "error":
                    raise SystemExit("error: " + msg +
                                     "\n  Pass a merge config, or --on-duplicate newest.")
                keep = max(prev, rec, key=lambda r: r["artifact"].name)
                print(f"warning: {msg}\n    using the newer: {keep['artifact'].name}",
                      file=sys.stderr)
                cells[key] = keep
        if not cells:
            raise SystemExit(f"no pairwise runs found for track {title!r} ({raw})")
        tracks.append((title, cells))
    return tracks


# ---------------------------------------------------------------------------
# Selecting what to draw
# ---------------------------------------------------------------------------

def _shared_ablations(tracks, requested, include_depth):
    """Resolve the ablation list, and map each track's own naming onto it.

    Returns (row_keys, per_track_lookup) where per_track_lookup[i][row_key] is
    the ablation name that track actually uses — `..._pairwise_human` in one
    track lines up with `..._pairwise_vanilla` in another.
    """
    indexes = []
    for _, cells in tracks:
        idx = defaultdict(set)
        for ablation, _model in cells:
            idx[row_key(ablation, TRACK_SUFFIXES)].add(ablation)
            idx[ablation].add(ablation)
        indexes.append(idx)

    if requested:
        keys = [row_key(a, TRACK_SUFFIXES) for a in requested]
    else:
        common = set(indexes[0])
        for idx in indexes[1:]:
            common &= set(idx)
        keys = sorted(k for k in common
                      if k == row_key(k, TRACK_SUFFIXES))
        if not include_depth:
            keys = [k for k in keys if k not in DEPTH_ABLATIONS]

    resolved, lookups = [], [dict() for _ in tracks]
    for key in keys:
        names = [idx.get(key) for idx in indexes]
        if not any(names):
            print(f"warning: ablation {key!r} is in no track — skipping.",
                  file=sys.stderr)
            continue
        for lookup, found in zip(lookups, names):
            if found:
                lookup[key] = sorted(found)[0]
        resolved.append(key)
    return resolved, lookups


def _run_names(tracks, ablations, lookups) -> dict:
    """Row key -> every on-disk ablation name it covers, across tracks."""
    names = {key: set() for key in ablations}
    for lookup in lookups:
        for key in ablations:
            if lookup.get(key):
                names[key].add(lookup[key])
    return names


def _rows(tracks, ablations, lookups, requested_models, keep_unsupported=False):
    """Gather plot rows, keyed by model, sorted by median tie rate descending.

    Configurations a judge cannot actually run — see UNSUPPORTED_CELLS, shared
    with common.py so every figure blanks the same cells — are dropped
    rather than plotted, because the number they produced measures nothing.
    """
    per_model = defaultdict(list)
    dropped = set()
    for t, ((title, cells), lookup) in enumerate(zip(tracks, lookups)):
        for key in ablations:
            name = lookup.get(key)
            if name is None:
                continue
            for (ablation, model), rec in cells.items():
                if ablation != name:
                    continue
                if not rec["support"]:
                    continue
                if not keep_unsupported and (key, model) in UNSUPPORTED_CELLS:
                    dropped.add((key, model))
                    continue
                per_model[model].append({
                    **rec,
                    # `key` is the shared name the row is grouped under;
                    # rec["ablation"] is the track-specific one on disk. Set
                    # these after **rec so the spread cannot clobber them.
                    "track_index": t, "track": title,
                    "ablation": key, "ablation_run": rec["ablation"],
                    "rate": rec["ties"] / rec["support"],
                })
    for key, model in sorted(dropped):
        print(f"note: dropping {key} / {model} — the judge has no such control, "
              f"so the run measures nothing. Pass --keep-unsupported to plot it.")
    if requested_models:
        missing = [m for m in requested_models if m not in per_model]
        for m in missing:
            print(f"warning: no data for model {m!r} — skipping row.", file=sys.stderr)
        order = [m for m in requested_models if m in per_model]
    else:
        import statistics
        order = sorted(per_model,
                       key=lambda m: -statistics.median(p["rate"] for p in per_model[m]))
    return order, per_model


def _warn_mixed_sampling(per_model, ablations):
    """Tie rates only compare across runs that sampled the judge identically.

    Derived runs record no sampling parameters, so the ablation name is the
    fallback signal — and the one that catches the case that matters.
    """
    depth_named = [a for a in ablations if a in DEPTH_ABLATIONS]
    other = [a for a in ablations if a not in DEPTH_ABLATIONS]
    recorded = {(p["mec_k"], p["bidirectional"])
                for points in per_model.values() for p in points
                if p["mec_k"] is not None}
    if len(recorded) < 2 and not (depth_named and other):
        return
    detail = ", ".join(f"mec_k={k}{'' if b else ' unidirectional'}"
                       for k, b in sorted(recorded, key=lambda d: (d[0], d[1])))
    named = ", ".join(depth_named)
    reason = f"sampling depths ({detail})" if len(recorded) > 1 else f"{named}"
    print(f"warning: the selection mixes {reason} with the standard runs. "
          f"A tie needs an even vote count to be reachable at all, so those dots "
          f"are not on a comparable scale.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _beeswarm(values, min_gap, max_offset=0.17):
    """Vertical offsets that stop overlapping dots hiding each other.

    Points are walked left to right and pushed off the row centre only while a
    neighbour sits within min_gap, so an isolated dot always stays on its line
    and a row of well-separated dots reads as a clean strip.
    """
    offsets = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    cluster = []
    for i in order:
        if cluster and values[i] - values[cluster[-1]] > min_gap:
            _spread(cluster, offsets, max_offset)
            cluster = []
        cluster.append(i)
    _spread(cluster, offsets, max_offset)
    return offsets


def _spread(cluster, offsets, max_offset):
    """Fan one run of near-equal values symmetrically about the row line."""
    n = len(cluster)
    if n < 2:
        return
    step = min(2 * max_offset / (n - 1), 0.11)
    start = -step * (n - 1) / 2
    for rank, i in enumerate(cluster):
        offsets[i] = start + rank * step


def _draw(order, per_model, tracks, model_labels, out_path, formats, width_in,
          dpi, title, subtitle, style, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    st = STYLES[style]
    plt.rcParams.update({**st["rc"], "pdf.fonttype": 42, "ps.fonttype": 42})
    title = title if st["titles"] else None
    subtitle = subtitle if st["titles"] else None

    labels = [model_labels.get(m, m) for m in order]
    hi = max(p["rate"] for pts in per_model.values() for p in pts)
    xmax = min(1.0, (int(hi * 10) + 1) / 10)

    # Lay the figure out in inches and convert to fractions, so the title block
    # never eats into the axes the way tight_layout + suptitle does.
    row_h = st["row_h"]
    top_pad = ((0.30 if title else 0) + (0.21 if subtitle else 0)
               + st["legend_pad"])
    bottom_pad = st["bottom_pad"]
    plot_h = row_h * len(order)
    height = top_pad + plot_h + bottom_pad
    left_pad = st["left_base"] + st["label_w"] * max(len(t) for t in labels)
    # The axis title sits outside the tick labels, so it needs margin of its own
    # — matplotlib would otherwise run it off the left edge of the canvas.
    if ylabel:
        left_pad += st["fs_xlabel"] / 72 + 0.10
    right_pad = 0.22

    fig = plt.figure(figsize=(width_in, height))
    ax = fig.add_axes([left_pad / width_in, bottom_pad / height,
                       1 - (left_pad + right_pad) / width_in, plot_h / height])

    for row, model in enumerate(order):
        y = len(order) - 1 - row
        points = per_model[model]
        rates = [p["rate"] for p in points]
        offsets = _beeswarm(rates, min_gap=xmax * 0.018,
                            max_offset=st["max_offset"])
        ax.axhline(y, zorder=1, **st["row_line"])
        if st["medians"]:
            _median_ticks(ax, y, points, st["max_offset"])
        for p, dy in zip(points, offsets):
            ax.scatter(p["rate"], y + dy, s=st["marker_size"],
                       color=TRACK_COLORS[p["track_index"] % len(TRACK_COLORS)],
                       edgecolor="white", linewidth=st["marker_edge"],
                       alpha=0.9, zorder=3)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=st["fs_ylabel"])
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_xlim(0, xmax)
    ticks = [i / 100 for i in range(0, int(round(xmax * 100)) + 1, 10)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{round(t * 100)}%" for t in ticks],
                       fontsize=st["fs_xtick"])
    ax.set_xlabel(st["xlabel"], fontsize=st["fs_xlabel"], labelpad=st["labelpad"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=st["fs_xlabel"], labelpad=6)
    ax.xaxis.grid(True, color="#e9ebf0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c4c8d2")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color="#c4c8d2", labelcolor="#52514e",
                   **st["xtick_params"])

    handles = [Line2D([], [], marker="o", linestyle="",
                      markersize=7 if style == "screen" else 5,
                      markerfacecolor=TRACK_COLORS[i % len(TRACK_COLORS)],
                      markeredgecolor="white", label=t)
               for i, (t, _) in enumerate(tracks)]
    if st["medians"]:
        handles.append(Line2D([], [], color="#3f4a5a", lw=1.1,
                              label="track median"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0),
              borderaxespad=st["legend_axespad"], ncol=len(handles),
              frameon=False, fontsize=st["fs_legend"], handletextpad=0.3,
              columnspacing=st["legend_colspace"])

    y_in = height
    if title:
        y_in -= 0.26
        fig.text(left_pad / width_in, y_in / height, title, ha="left", va="baseline",
                 fontsize=12.5, fontweight="bold")
    if subtitle:
        y_in -= 0.20
        fig.text(left_pad / width_in, y_in / height, subtitle, ha="left",
                 va="baseline", fontsize=9, color="#52514e")
    return save_figure(fig, out_path, formats=formats, dpi=dpi)


def _median_ticks(ax, y, points, half_height):
    """A short vertical rule at each track's median rate on one row.

    The dots show the spread the ablations produce; this shows where the judge
    sits once that spread is collapsed, which is the comparison between judges
    the row ordering is already making. Drawn under the dots so a median that
    lands on a dot does not hide it, but taller than the beeswarm band so its
    two ends stay visible when it does.
    """
    import statistics
    from collections import defaultdict

    reach = half_height * 1.7
    by_track = defaultdict(list)
    for p in points:
        by_track[p["track_index"]].append(p["rate"])
    for track_index, rates in by_track.items():
        ax.vlines(statistics.median(rates), y - reach, y + reach,
                  color=TRACK_COLORS[track_index % len(TRACK_COLORS)],
                  lw=1.1, alpha=0.55, zorder=2)


def _write_csv(path, order, per_model, model_labels, ablation_labels):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "model_label", "ablation", "ablation_label",
                    "ablation_run", "track", "ties", "support", "tie_rate", "runs",
                    "mec_k", "bidirectional", "artifact"])
        for model in order:
            for p in sorted(per_model[model], key=lambda p: (p["track_index"], p["ablation"])):
                w.writerow([model, model_labels.get(model, model), p["ablation"],
                            ablation_labels.get(p["ablation"], p["ablation"]),
                            p["ablation_run"], p["track"], p["ties"], p["support"],
                            f"{p['rate']:.6f}", p["runs"], p["mec_k"],
                            p["bidirectional"], p["artifact"]])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip plot of judge tie rates across ablations and tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--track", action="append", required=True, metavar='"Name"=PATH',
                        help="Merge config or sweep dir(s). Repeat once per track.")
    parser.add_argument("--out", required=True, help="Figure path, e.g. fig/tie_rates.pdf")
    parser.add_argument("--ablations", nargs="+", metavar="NAME",
                        help="Canonical names to include (default: shared across tracks).")
    parser.add_argument("--models", nargs="+", metavar="NAME",
                        help="Row order (default: descending median tie rate).")
    parser.add_argument("--variant", default="unfiltered",
                        choices=["unfiltered", "filtered", "subset"],
                        help="Which accuracy report to read (default: unfiltered).")
    parser.add_argument("--config", help="YAML with row_labels / model_labels / track_titles.")
    parser.add_argument("--keep-unsupported", action="store_true",
                        help="Plot configurations a judge cannot run "
                             "(e.g. reasoning effort on a model without one), "
                             "which are dropped by default.")
    parser.add_argument("--include-depth-ablations", action="store_true",
                        help=f"Also plot {', '.join(DEPTH_ABLATIONS)}, whose tie "
                             f"rates are not on a comparable scale.")
    parser.add_argument("--on-duplicate", default="newest", choices=["newest", "error"],
                        help="What to do when one configuration has two runs.")
    parser.add_argument("--formats", nargs="+", default=[], metavar="EXT",
                        help="Extra formats beyond --out's own extension.")
    parser.add_argument("--width", type=float, default=7.1,
                        help="Figure width in inches (default: 7.1, ~full text width).")
    parser.add_argument("--dpi", type=int, default=300, help="Raster resolution.")
    parser.add_argument("--style", default="screen", choices=sorted(STYLES),
                        help="Presentation preset (default: screen). `acl` is "
                             "camera-ready: caption-sized type, no title block.")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--ylabel", default="Judge model",
                        help='Y-axis title (default: "Judge model"). '
                             'Pass "" to leave the axis unnamed.')
    parser.add_argument("--subtitle", default=None,
                        help="Line under the title (default: describes the selection).")
    parser.add_argument("--no-title", action="store_true", help="Draw no title block.")
    parser.add_argument("--csv", help="Where to write the plotted numbers "
                                      "(default: --out with a .csv suffix).")
    parser.add_argument("--no-csv", action="store_true", help="Skip the CSV sidecar.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    cfg = load_config(Path(args.config)) if args.config else {}
    model_labels = cfg.get("model_labels") or {}
    ablation_labels = cfg.get("row_labels") or {}
    track_titles = cfg.get("track_titles") or {}

    tracks = _collect(args.track, args.variant, repo_root, args.on_duplicate)
    tracks = [(track_titles.get(t, t), c) for t, c in tracks]

    ablations, lookups = _shared_ablations(
        tracks, args.ablations, args.include_depth_ablations)
    if not ablations:
        raise SystemExit("no ablations to draw — the tracks have none in common. "
                         "Name them explicitly with --ablations.")
    run_names = _run_names(tracks, ablations, lookups)
    descriptions = _ablation_descriptions()
    ablation_labels = {key: _label_for(key, run_names[key], ablation_labels, descriptions)
                       for key in ablations}

    order, per_model = _rows(tracks, ablations, lookups, args.models,
                             args.keep_unsupported)
    if not order:
        raise SystemExit("no judge models to draw.")
    _warn_mixed_sampling(per_model, ablations)

    n_points = sum(len(v) for v in per_model.values())
    print(f"{len(ablations)} ablations x {len(order)} judges x {len(tracks)} tracks "
          f"-> {n_points} points")
    for key in ablations:
        print(f"  {ablation_labels.get(key, key)}")
    print()
    width = max(len(model_labels.get(m, m)) for m in order)
    for model in order:
        pts = sorted(per_model[model], key=lambda p: p["rate"])
        cells = "  ".join(f"{p['rate']*100:>5.1f}%" for p in pts)
        print(f"  {model_labels.get(model, model):<{width}}  {cells}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    subtitle = args.subtitle
    if subtitle is None and not args.no_title:
        shared = "The ablations run on every track." if not args.ablations else \
            "The selected ablations."
        subtitle = f"{shared} {n_points} runs, {args.variant} reports."
    written = _draw(order, per_model, tracks, model_labels, out_path,
                    args.formats, args.width, args.dpi,
                    None if args.no_title else args.title,
                    None if args.no_title else subtitle, args.style, args.ylabel)
    print()
    for p in written:
        print(f"figure -> {p}")
    if not args.no_csv:
        csv_path = Path(args.csv) if args.csv else out_path.with_suffix(".csv")
        if not csv_path.is_absolute():
            csv_path = repo_root / csv_path
        print(f"data   -> {_write_csv(csv_path, order, per_model, model_labels, ablation_labels)}")


if __name__ == "__main__":
    main()
