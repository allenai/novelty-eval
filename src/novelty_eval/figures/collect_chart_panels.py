#!/usr/bin/env python3
"""
Collect panels from several merges into one chart-data set
==========================================================
`two_track_figure.ipynb` draws one row per panel of a single merge's
`unified_*.json`. That is enough when every row shares a baseline, because one
merge can only hold one baseline (`baseline:` is global to a merge).

When each row needs its *own* comparator — "what does retrieval do, holding the
negative generator fixed?" is one delta per generator, each against that
generator's own no-retrieval run — the rows necessarily come from different
merges. Each panel already carries its own `baseline_metrics`, `ablation_metrics`
and `bootstrap_results`, so gathering panels from several merges into one file
produces a valid chart-data set whose rows are each measured against their own
baseline.

The output directory mimics a merge output (`<out>/unified_charts/unified_*.json`),
so the figure notebook consumes it unchanged as one more track. Drop the
baseline strip when drawing it: the strip assumes one baseline for the whole
grid, which is exactly what this breaks.

    python collect_chart_panels.py --setup pairwise \\
        --out output/ablation_sweeps/retrieval_by_generator \\
        --from output/ablation_sweeps/merge_combined:retrieval \\
        --from output/ablation_sweeps/retrieval_given_gpt_5.1_pairwise

`--from DIR` takes every panel in DIR; `--from DIR:NAME[,NAME...]` takes only the
named ones. Panels keep their source order, which becomes the row order.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from novelty_eval.figures.common import chart_data_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="sources", action="append", required=True,
                        metavar="DIR[:NAME,...]",
                        help="Merge directory, optionally with the panels to take. Repeatable; "
                             "order becomes row order.")
    parser.add_argument("--setup", required=True, choices=["pairwise", "pointwise"])
    parser.add_argument("--variant", default="filtered", choices=["filtered", "unfiltered"])
    parser.add_argument("--out", required=True, help="Directory to write the chart data into.")
    args = parser.parse_args()

    panels: dict = {}
    models: list[str] | None = None
    template: dict | None = None

    for src in args.sources:
        raw, _, wanted = src.partition(":")
        path = chart_data_path(Path(raw), args.setup, args.variant)
        if not path.exists():
            raise SystemExit(f"no {args.setup} chart data in {raw} (looked for {path})")
        data = json.loads(path.read_text())
        template = template or data

        names = [n.strip() for n in wanted.split(",") if n.strip()] or list(data.get("panels", {}))
        for name in names:
            panel = data.get("panels", {}).get(name)
            if panel is None:
                raise SystemExit(f"panel '{name}' not in {path}\n"
                                 f"  available: {', '.join(data.get('panels', {})) or '(none)'}")
            if name in panels:
                raise SystemExit(f"panel '{name}' collected twice — rows must be distinct")
            panels[name] = panel

        # Judge models must be common to every source, or a row would be blank
        # for a judge some other row has.
        src_models = data.get("all_models") or []
        models = src_models if models is None else [m for m in models if m in src_models]

    if not panels:
        raise SystemExit("no panels collected")
    if not models:
        raise SystemExit("the sources share no judge models")

    out_dir = Path(args.out) / "unified_charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.variant == "unfiltered" else f"_{args.variant}"
    out_path = out_dir / f"unified_{args.setup}{suffix}.json"

    out_path.write_text(json.dumps({**template, "all_models": models, "panels": panels,
                                    "collected_from": args.sources}, indent=2))
    print(f"{len(panels)} panel(s), {len(models)} judge(s) -> {out_path}")
    for name in panels:
        print(f"    {name}  (vs {panels[name].get('baseline_name', '?')})")


if __name__ == "__main__":
    main()
