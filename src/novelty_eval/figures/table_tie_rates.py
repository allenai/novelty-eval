#!/usr/bin/env python3
"""
Judge Tie-Rate LaTeX Table
==========================
The numbers behind the tie-rate strip plot, as the paper's table: one row per
configuration, one column per judge, column blocks per track.

Reads the CSV that chart_tie_rates.py writes beside its figure, so the table and
the figure can never drift apart — regenerate the figure and rerun this.

    python table_tie_rates.py output/figures/tie-rates/tie_rates_filtered.csv \
        --tracks "Human-only" "Human + generated" \
        --track-title "Human-only=Human-Only" \
        --track-title "Human + generated=Human+Generated" \
        --models claude-sonnet-4-5 claude-opus-4-5 claude-opus-4-6 \
                 gpt-5.1 gpt-5.2 gpt-5.4 \
        --out output/figures/tie-rates/tie_rates_filtered.tex

Ordering, all overridable:

    --tracks       column blocks, left to right. Default: order of appearance.
    --models       columns inside a block. Default: descending mean tie rate,
                   which is the order the figure stacks its rows in.
    --rows         configurations, top to bottom, by canonical ablation name.
                   Default: ascending mean tie rate with --baseline pinned to
                   the top, so the table reads as "least ties first, relative
                   to the run the paper calls current".
    --baseline     ablation to pin first and mark "(baseline)". Default:
                   `current`; pass "" for no pinned row.

Every cell is a tie count over that run's own support, taken from the CSV rather
than recomputed. The caption states n only when every cell shares one support —
otherwise the sentence would be false, so it is dropped and a warning printed.
"""

import argparse
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

# Judge abbreviations for the column headers. Anything not listed falls back to
# _abbreviate, which just strips the vendor prefix.
DEFAULT_ABBREV = {
    "claude-sonnet-4-5": "son-4-5",
    "claude-opus-4-5": "op-4-5",
    "claude-opus-4-6": "op-4-6",
    "gpt-5.1": "5.1",
    "gpt-5.2": "5.2",
    "gpt-5.4": "5.4",
}

_LATEX_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                  "_": r"\_", "{": r"\{", "}": r"\}"}


def _escape(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(c, c) for c in str(text))


def _abbreviate(model: str) -> str:
    """claude-sonnet-4-5 -> son-4-5, gpt-5.1 -> 5.1, o3-mini -> o3-mini."""
    short = re.sub(r"^(claude|anthropic|openai|google)-", "", model)
    short = re.sub(r"^gpt-", "", short)
    for long, brief in (("sonnet", "son"), ("opus", "op"), ("haiku", "hai")):
        short = short.replace(long, brief)
    return short


def _parse_pairs(specs) -> dict:
    """--track-title "a=b" pairs into {a: b}."""
    out = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f'expected "key=value", got: {spec!r}')
        key, value = spec.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _read(csv_path: Path):
    """CSV -> (cells, labels, supports, order-of-appearance lists).

    cells is {(track, ablation, model): rate in %}; labels maps a canonical
    ablation name to the display label the chart config gave it.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    missing = {"track", "ablation", "model", "tie_rate", "support"} - set(rows[0])
    if missing:
        raise SystemExit(f"{csv_path} is missing column(s): {', '.join(sorted(missing))}. "
                         f"Is it a chart_tie_rates.py CSV?")

    cells, labels, supports = {}, {}, set()
    tracks, models, ablations = OrderedDict(), OrderedDict(), OrderedDict()
    for r in rows:
        key = (r["track"], r["ablation"], r["model"])
        if key in cells:
            raise SystemExit(f"{csv_path} has two rows for {key} — "
                             f"regenerate it rather than editing by hand.")
        cells[key] = float(r["tie_rate"]) * 100
        labels.setdefault(r["ablation"], r.get("ablation_label") or r["ablation"])
        supports.add(int(r["support"]))
        tracks[r["track"]] = None
        models[r["model"]] = None
        ablations[r["ablation"]] = None
    return cells, labels, supports, list(tracks), list(models), list(ablations)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _order(requested, seen, what):
    """Honour an explicit order, dropping and reporting names that are absent."""
    if not requested:
        return None
    known = set(seen)
    keep = [x for x in requested if x in known]
    for x in requested:
        if x not in known:
            print(f"warning: no {what} named {x!r} in the CSV — skipping.",
                  file=sys.stderr)
    if not keep:
        raise SystemExit(f"none of the requested {what}s are in the CSV.")
    return keep


def _render(cells, labels, tracks, models, ablations, abbrev, track_titles,
            baseline, caption, label, mean_row, missing):
    n_models = len(models)
    span = len(tracks) * n_models
    col_spec = "l " + " ".join(["c" * n_models] * len(tracks))

    def cell(track, ablation, model):
        v = cells.get((track, ablation, model))
        return missing if v is None else f"{v:.1f}"

    def body_row(name, values, width):
        return " & ".join([f"{name:<{width}}"] + [f"{v:>4}" for v in values]) + r" \\"

    header_groups, rules, first = [], [], 2
    for track in tracks:
        title = track_titles.get(track, track)
        header_groups.append(
            f"\\multicolumn{{{n_models}}}{{c}}{{\\textbf{{{_escape(title)}}}}}")
        rules.append(f"\\cmidrule(lr){{{first}-{first + n_models - 1}}}")
        first += n_models
    judge_block = " & ".join(f"\\texttt{{{_escape(abbrev[m])}}}" for m in models)

    # First column, already in its final form, so the padding lines the `&` up
    # in the .tex source too — someone will end up reading this by hand.
    names = [_escape(labels[a] + (" (baseline)" if a == baseline else ""))
             for a in ablations]
    if mean_row:
        names.append(r"\textit{Mean}")
    width = max(len(n) for n in names)

    # One source line per track block, continued under the first, so the header
    # stays readable at 13 columns instead of running off the screen.
    header_indent = " " * len(r"\textbf{Configuration} ")
    judge_lines = [r"\textbf{Configuration} & " + judge_block]
    judge_lines += [header_indent + "& " + judge_block for _ in tracks[1:]]
    judge_lines[-1] += r" \\"

    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "& " + " & ".join(header_groups) + r" \\",
        " ".join(rules),
        *judge_lines,
        r"\midrule",
    ]
    for ablation, name in zip(ablations, names):
        lines.append(body_row(
            name, [cell(t, ablation, m) for t in tracks for m in models], width))
    if mean_row:
        means = [_mean([cells.get((t, a, m)) for a in ablations])
                 for t in tracks for m in models]
        lines += [r"\midrule",
                  body_row(names[-1],
                           [missing if v is None else f"{v:.1f}" for v in means],
                           width)]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table*}",
    ]
    # `span` only exists to catch a table too wide to be worth a table* — the
    # caller sees the warning and can switch to a rotated or split layout.
    if span > 14:
        print(f"warning: {span} numeric columns is wide even for a table* — "
              f"consider splitting the tracks into two tables.", file=sys.stderr)
    return "\n".join(lines) + "\n"


def _caption(models, abbrev, supports, custom, note):
    if custom:
        return f"{custom} {note}".strip() if note else custom
    named = ", ".join(f"\\texttt{{{_escape(abbrev[m])}}} is \\texttt{{{_escape(m)}}}"
                      for m in models)
    text = (r"\textbf{Tie rates (\%) across controlled configurations}, the "
            r"numbers behind Figure~\ref{fig:tie_rates}. Judges are "
            f"abbreviated: {named}.")
    if len(supports) == 1:
        n = next(iter(supports))
        text += (f" Every cell is computed over the same $n = {n}$ pairwise "
                 f"examples, so the absolute number of ties is the reported "
                 f"rate $\\times\\,{n}$.")
    return f"{text} {note}".strip() if note else text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LaTeX table of judge tie rates, from a chart_tie_rates.py CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv", help="CSV written by chart_tie_rates.py.")
    parser.add_argument("--out", help="Write here instead of stdout.")
    parser.add_argument("--tracks", nargs="+", metavar="NAME",
                        help="Column blocks, left to right.")
    parser.add_argument("--models", nargs="+", metavar="NAME",
                        help="Judge columns inside each block.")
    parser.add_argument("--rows", nargs="+", metavar="NAME",
                        help="Configurations top to bottom, by ablation name.")
    parser.add_argument("--baseline", default="current",
                        help='Ablation pinned first and marked "(baseline)". '
                             'Default: current. Pass "" to pin nothing.')
    parser.add_argument("--track-title", action="append", metavar="CSV=DISPLAY",
                        help="Rename a track in the header. Repeatable.")
    parser.add_argument("--abbrev", action="append", metavar="MODEL=SHORT",
                        help="Override a judge abbreviation. Repeatable.")
    parser.add_argument("--no-mean", action="store_true",
                        help="Leave off the trailing mean row.")
    parser.add_argument("--missing", default="",
                        help="What to print where the CSV has no run "
                             "(default: an empty cell).")
    parser.add_argument("--note", help="Sentence appended to the caption, e.g. "
                                       "why a cell is blank.")
    parser.add_argument("--caption", help="Replace the generated caption.")
    parser.add_argument("--label", default="tab:tie_rates")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"no such CSV: {csv_path}")
    cells, labels, supports, csv_tracks, csv_models, csv_ablations = _read(csv_path)

    tracks = _order(args.tracks, csv_tracks, "track") or csv_tracks
    models = _order(args.models, csv_models, "model") or sorted(
        csv_models,
        key=lambda m: -(_mean([cells.get((t, a, m))
                               for t in tracks for a in csv_ablations]) or 0))
    ablations = _order(args.rows, csv_ablations, "configuration")
    if not ablations:
        ablations = sorted(
            csv_ablations,
            key=lambda a: _mean([cells.get((t, a, m))
                                 for t in tracks for m in models]) or 0)
        if args.baseline in ablations:
            ablations.remove(args.baseline)
            ablations.insert(0, args.baseline)

    abbrev = {m: DEFAULT_ABBREV.get(m) or _abbreviate(m) for m in models}
    abbrev.update({m: s for m, s in _parse_pairs(args.abbrev).items() if m in abbrev})
    if len(set(abbrev.values())) < len(abbrev):
        print("warning: two judges share an abbreviation — pass --abbrev to "
              "disambiguate.", file=sys.stderr)
    if len(supports) > 1:
        print(f"warning: the runs do not share one support ({sorted(supports)}), "
              f"so the caption cannot state a single n.", file=sys.stderr)

    tex = _render(cells, labels, tracks, models, ablations, abbrev,
                  _parse_pairs(args.track_title), args.baseline,
                  _caption(models, abbrev, supports, args.caption, args.note),
                  args.label, not args.no_mean, args.missing)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex, encoding="utf-8")
        print(f"table -> {out}")
    else:
        sys.stdout.write(tex)


if __name__ == "__main__":
    main()
