#!/usr/bin/env python3
"""
Rechart Unified Ablation Figures
================================
Re-render the unified side-by-side charts from the .json data files that
merge_ablation_runs.py writes next to them, without re-running the merge —
no artifact copying, no report generation, no bootstrap.  Iterating on the
figure format is a sub-second loop.

Both the unfiltered and filtered variants are redrawn by default.

Usage:
    # Redraw everything found (both setups, both variants)
    python rechart_unified.py output/ablation_sweeps/merge_vanilla

    # See what's in the data files without drawing anything
    python rechart_unified.py output/ablation_sweeps/merge_vanilla --list

    # Just the filtered pairwise figure, with a subset of panels
    python rechart_unified.py output/ablation_sweeps/merge_vanilla \
        --setup pairwise --variant filtered \
        --panels vague_criterion retrieval low_judge_reasoning

    # Retitle a panel, drop a judge model, add a metric column back
    python rechart_unified.py output/ablation_sweeps/merge_vanilla \
        --titles retrieval="No Retrieval" \
        --exclude-models gpt-5.2 \
        --metrics pairwise_accuracy pairwise_accuracy_strict pairwise_accuracy_no_ties

    # Write variants side by side instead of overwriting the original PNG
    python rechart_unified.py output/ablation_sweeps/merge_vanilla --suffix wide

To change the look of the figure itself (fonts, cell text, colours, spacing),
edit _generate_unified_heatmap in viz.py and re-run this script.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from novelty_eval.figures import unified_chart_data
from novelty_eval.figures.viz import _generate_unified_heatmap


def _find_data_files(raw_paths: list[str]) -> list[Path]:
    """Resolve CLI paths (files, a merge dir, or a unified_charts dir) to .json files."""
    found: list[Path] = []
    for raw in raw_paths:
        p = Path(raw)
        if p.is_file():
            found.append(p)
            continue
        if not p.is_dir():
            print(f"warning: no such path — {p}", file=sys.stderr)
            continue
        search_dir = p / "unified_charts" if (p / "unified_charts").is_dir() else p
        found.extend(sorted(search_dir.glob("unified_*.json")))

    # Preserve order while dropping duplicates from overlapping arguments.
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in found:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _parse_titles(pairs: list[str]) -> dict[str, str]:
    """Parse --titles name=Title arguments."""
    titles: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--titles expects name=Title, got: {pair!r}")
        name, title = pair.split("=", 1)
        titles[name.strip()] = title.strip()
    return titles


def _output_path(data_path: Path, args) -> Path:
    """Where the redrawn PNG goes, per --out / --overwrite / --suffix."""
    if args.out:
        return Path(args.out)
    png = data_path.with_suffix(".png")
    if args.overwrite:
        return png
    suffix = f"_rechart_{args.suffix}" if args.suffix else "_rechart"
    return png.with_name(png.stem + suffix + png.suffix)


def _describe(data: dict, path: Path) -> None:
    """Print the contents of one data file for --list."""
    panels = data.get("panels", {})
    rendered = data.get("rendered", [])
    print(f"\n{path}")
    print(f"  setup      : {data['setup']}")
    print(f"  variant    : {data['variant']}  (from {data['report_file']})")
    print(f"  generated  : {data['generated_at']}")
    print(f"  models     : {', '.join(data['all_models'])}")
    print(f"  columns    : {', '.join(data['metric_labels'])}")
    available = [k for k in data["available_metrics"] if k not in data["metric_labels"]]
    if available:
        print(f"  also avail.: {', '.join(available)}")
    print(f"  panels drawn ({len(rendered)}):")
    for name in rendered:
        print(f"    - {name}  \"{panels[name]['title']}\"")
    extra = [n for n in sorted(panels) if n not in rendered]
    if extra:
        print(f"  panels available but not drawn ({len(extra)}):")
        for name in extra:
            print(f"    - {name}  \"{panels[name]['title']}\"")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-render unified ablation charts from their saved data files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="unified_*.json files, a unified_charts/ directory, or a merge output directory.",
    )
    parser.add_argument("--setup", choices=["pointwise", "pairwise"],
                        help="Only redraw this setup (default: both).")
    parser.add_argument("--variant", choices=["unfiltered", "filtered"],
                        help="Only redraw this variant (default: both).")
    parser.add_argument("--panels", nargs="+", metavar="NAME",
                        help="Canonical ablation names to draw, in order "
                             "(default: whatever the original figure drew).")
    parser.add_argument("--titles", nargs="+", default=[], metavar="NAME=TITLE",
                        help="Override a panel title.")
    parser.add_argument("--metrics", nargs="+", metavar="KEY",
                        help="Metric columns to show, in order (default: the original columns). "
                             "Keys that don't belong to a file's setup are ignored for that file.")
    parser.add_argument("--models", nargs="+", metavar="NAME",
                        help="Judge models to show as rows, in order (default: all).")
    parser.add_argument("--exclude-models", nargs="+", default=[], metavar="NAME",
                        help="Judge models to drop from the rows.")
    parser.add_argument("--out", help="Exact output PNG path (only valid with a single input file).")
    parser.add_argument("--suffix", help="Tag the output as <name>_rechart_<suffix>.png.")
    parser.add_argument("--formats", nargs="+", default=[], metavar="EXT",
                        help="Extra formats to write alongside the PNG, e.g. pdf. "
                             "PDF is vector and embeds Type-42 fonts.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Write over the original <name>.png instead of a _rechart copy.")
    parser.add_argument("--list", action="store_true",
                        help="Print what each data file contains and exit.")
    args = parser.parse_args()

    data_files = _find_data_files(args.paths)
    if not data_files:
        raise SystemExit(
            "No unified_*.json data files found.\n"
            "They are written next to the PNGs by merge_ablation_runs.py. For an older "
            "merge directory, regenerate them with:\n"
            "  python merge_ablation_runs.py --config <merge_config.yaml> --charts-only"
        )

    title_overrides = _parse_titles(args.titles)
    loaded: list[tuple[Path, dict]] = []
    for path in data_files:
        data = unified_chart_data.load(path)
        if args.setup and data["setup"] != args.setup:
            continue
        if args.variant and data["variant"] != args.variant:
            continue
        loaded.append((path, data))

    if not loaded:
        raise SystemExit("No data files matched the --setup / --variant filters.")

    if args.list:
        for path, data in loaded:
            _describe(data, path)
        return

    if args.out and len(loaded) > 1:
        raise SystemExit(
            f"--out takes a single output path but {len(loaded)} files matched. "
            f"Narrow it with --setup / --variant, or use --suffix instead."
        )

    # A panel or metric that belongs to only one setup is fine — it is dropped
    # for the files that lack it. One that matches nothing anywhere is a typo.
    requested_panels = set(args.panels or [])
    requested_metrics = set(args.metrics or [])
    matched_panels: set[str] = set()
    matched_metrics: set[str] = set()

    written: list[Path] = []
    for path, data in loaded:
        available_metrics = data["available_metrics"]

        if args.metrics:
            metric_keys = [k for k in args.metrics if k in available_metrics]
            matched_metrics.update(metric_keys)
            if not metric_keys:
                print(f"skipping {path.name}: none of --metrics apply to {data['setup']}",
                      file=sys.stderr)
                continue
            metric_labels = {k: available_metrics[k] for k in metric_keys}
        else:
            metric_labels = data["metric_labels"]

        if args.panels:
            canonicals = [c for c in args.panels if c in data["panels"]]
            matched_panels.update(canonicals)
            if not canonicals:
                print(f"skipping {path.name}: none of --panels exist in {data['setup']}",
                      file=sys.stderr)
                continue
        else:
            canonicals = None

        models = args.models or data["all_models"]
        models = [m for m in models if m not in set(args.exclude_models)]
        if not models:
            raise SystemExit(f"{path.name}: no judge models left after --models/--exclude-models.")
        unknown_models = [m for m in models if m not in data["all_models"]]
        if unknown_models:
            raise SystemExit(
                f"{path.name}: unknown judge model(s) {unknown_models}. "
                f"Available: {data['all_models']}"
            )

        panels = unified_chart_data.build_render_panels(
            data, canonicals=canonicals, titles=title_overrides
        )
        if not panels:
            print(f"skipping {path.name}: no panels selected", file=sys.stderr)
            continue

        out_path = _output_path(path, args)
        result = _generate_unified_heatmap(
            setup=data["setup"],
            panels=panels,
            metric_labels=metric_labels,
            all_models=models,
            out_path=out_path,
            formats=args.formats,
        )
        if result:
            written.append(result)
            print(f"{data['setup']:>9} / {data['variant']:<10} → {result}")
        else:
            print(f"failed to render {path}", file=sys.stderr)

    unknown_panels = requested_panels - matched_panels
    if unknown_panels:
        print(f"\nwarning: --panels not found in any data file: {sorted(unknown_panels)}",
              file=sys.stderr)
    unknown_metrics = requested_metrics - matched_metrics
    if unknown_metrics:
        print(f"warning: --metrics not found in any data file: {sorted(unknown_metrics)}",
              file=sys.stderr)

    if not written:
        raise SystemExit("Nothing was rendered.")


if __name__ == "__main__":
    main()
