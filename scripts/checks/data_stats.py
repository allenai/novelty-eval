#!/usr/bin/env python3
"""Print label/area statistics for pointwise or pairwise YAML data files.

Optional --extract_other: filter to 'other topics in ML' entries, save a
filtered YAML and a human-readable Markdown report.
"""

import sys
import argparse
from collections import Counter
from pathlib import Path
from textwrap import dedent

import yaml


BAR_WIDTH = 30
OTHER_AREA = "other topics in machine learning (i.e., none of the above)"


def make_bar(count: int, total: int, width: int = BAR_WIDTH) -> str:
    filled = round(count / total * width) if total else 0
    return "█" * filled + "░" * (width - filled)


def detect_format(data: dict) -> str:
    first = next(iter(data.values()))
    if "ideas" in first and "metadata" in first and isinstance(first["metadata"], dict):
        inner = next(iter(first["metadata"].values()), None)
        if isinstance(inner, dict) and "type" in inner:
            return "pairwise"
    if "label" in first:
        return "pointwise"
    raise ValueError("Cannot detect file format (expected 'label' or pairwise 'ideas'/'metadata' structure)")


def parse_pointwise(data: dict) -> tuple[Counter, dict[str, Counter]]:
    labels: Counter = Counter()
    by_area: dict[str, Counter] = {}
    for entry in data.values():
        label = entry.get("label", "UNKNOWN")
        area = entry.get("metadata", {}).get("area", "UNKNOWN")
        labels[label] += 1
        by_area.setdefault(area, Counter())[label] += 1
    return labels, by_area


def parse_pairwise(data: dict) -> tuple[Counter, dict[str, Counter]]:
    labels: Counter = Counter()
    by_area: dict[str, Counter] = {}
    for pair in data.values():
        metadata = pair.get("metadata", {})
        for idea_meta in metadata.values():
            if not isinstance(idea_meta, dict):
                continue
            label = idea_meta.get("type", "UNKNOWN")
            area = idea_meta.get("area", "UNKNOWN")
            labels[label] += 1
            by_area.setdefault(area, Counter())[label] += 1
    return labels, by_area


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("─" * len(title))


def print_label_histogram(labels: Counter) -> None:
    total = sum(labels.values())
    print_section("LABEL DISTRIBUTION")
    for label in sorted(labels):
        count = labels[label]
        pct = count / total * 100 if total else 0
        bar = make_bar(count, total)
        print(f"  {label:<10} [{bar}] {count:>4}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<10}  {' ' * BAR_WIDTH}  {total:>4}")


def print_area_histogram(by_area: dict[str, Counter], all_labels: list[str]) -> None:
    total_ideas = sum(sum(c.values()) for c in by_area.values())
    sorted_areas = sorted(by_area.items(), key=lambda kv: -sum(kv[1].values()))

    label_header = "  ".join(f"{lbl:>8}" for lbl in all_labels)
    print_section(f"AREA DISTRIBUTION  ({label_header}  )")

    # Compute max area name length for alignment
    max_name = max(len(a) for a in by_area) if by_area else 20
    max_name = min(max_name, 55)

    for area, counts in sorted_areas:
        area_total = sum(counts.values())
        pct = area_total / total_ideas * 100 if total_ideas else 0
        bar = make_bar(area_total, total_ideas)
        label_counts = "  ".join(f"{counts.get(lbl, 0):>8}" for lbl in all_labels)
        truncated = area[:max_name].ljust(max_name)
        print(f"  {truncated}  [{bar}]  {area_total:>4} ({pct:4.1f}%)  {label_counts}")


def _abstract_block(text: str) -> str:
    """Wrap abstract text in a styled pre block that word-wraps and adapts to dark/light themes."""
    escaped = text.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        '<pre style="white-space: pre-wrap; word-wrap: break-word; '
        'color: inherit; background: rgba(128,128,128,0.08); padding:12px; border-radius:6px; '
        'border:1px solid rgba(128,128,128,0.3); font-family:inherit; font-size:1em;">'
        f"{escaped}</pre>"
    )


def _extract_other_pointwise(data: dict) -> dict:
    return {
        k: v
        for k, v in data.items()
        if v.get("metadata", {}).get("area") == OTHER_AREA
    }


def _extract_other_pairwise(data: dict) -> dict:
    return {
        k: v
        for k, v in data.items()
        if v.get("context") == OTHER_AREA
    }


def _write_pointwise_md(subset: dict, out_path: Path) -> None:
    lines = [
        f"# Other Topics in Machine Learning — Abstracts (Pointwise)",
        f"",
        f"> Area: *{OTHER_AREA}*  ",
        f"> Total entries: **{len(subset)}**",
        f"",
        "---",
        "",
    ]
    for idx, (key, entry) in enumerate(sorted(subset.items(), key=lambda kv: str(kv[0])), 1):
        meta = entry.get("metadata", {})
        title = meta.get("title", f"Entry {key}")
        label = entry.get("label", "?")
        rating = meta.get("rating", "?")
        contribution = meta.get("contribution", "?")
        abstract = entry.get("idea", "")

        lines += [
            f"## {idx}. {title}",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Label | `{label}` |",
            f"| Rating | {rating} |",
            f"| Contribution | {contribution} |",
            f"",
            _abstract_block(abstract),
            "",
            "---",
            "",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_pairwise_md(subset: dict, out_path: Path) -> None:
    lines = [
        f"# Other Topics in Machine Learning — Abstracts (Pairwise)",
        f"",
        f"> Area: *{OTHER_AREA}*  ",
        f"> Total pairs: **{len(subset)}**",
        f"",
        "---",
        "",
    ]
    for idx, (key, pair) in enumerate(sorted(subset.items(), key=lambda kv: str(kv[0])), 1):
        ideas = pair.get("ideas", {})
        metadata = pair.get("metadata", {})
        expected_winners = pair.get("expected_winners", [])

        lines += [f"## Pair {idx}  (key: `{key}`)", ""]

        for idea_key in sorted(ideas.keys(), key=str):
            idea_text = ideas[idea_key]
            idea_meta = metadata.get(idea_key, {})
            title = idea_meta.get("title", f"Idea {idea_key}")
            idea_type = idea_meta.get("type", "?")
            rating = idea_meta.get("rating", "?")
            contribution = idea_meta.get("contribution", "?")
            winner_marker = " ✓ *(expected winner)*" if str(idea_key) in [str(w) for w in expected_winners] else ""

            lines += [
                f"### Idea {idea_key} — {title}{winner_marker}",
                f"",
                f"| Field | Value |",
                f"|-------|-------|",
                f"| Type | `{idea_type}` |",
                f"| Rating | {rating} |",
                f"| Contribution | {contribution} |",
                f"",
                _abstract_block(idea_text),
                "",
            ]

        lines += ["---", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def extract_other(yaml_path: Path, output_dir: Path, fmt: str, data: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = yaml_path.stem

    if fmt == "pointwise":
        subset = _extract_other_pointwise(data)
        yaml_out = output_dir / f"{stem}_other_topics.yaml"
        md_out = output_dir / f"{stem}_other_topics.md"
        with open(yaml_out, "w") as f:
            yaml.dump(subset, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        _write_pointwise_md(subset, md_out)
    else:
        subset = _extract_other_pairwise(data)
        yaml_out = output_dir / f"{stem}_other_topics.yaml"
        md_out = output_dir / f"{stem}_other_topics.md"
        with open(yaml_out, "w") as f:
            yaml.dump(subset, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        _write_pairwise_md(subset, md_out)

    print(f"\n  Extracted {len(subset)} entries → {yaml_out}")
    print(f"  Markdown report    → {md_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print histogram statistics for pointwise or pairwise YAML data files."
    )
    parser.add_argument("--yaml_file", type=Path, help="Path to the YAML file")
    parser.add_argument(
        "--extract_other",
        action="store_true",
        help=f'Filter to "{OTHER_AREA}" entries and write a YAML subset + Markdown report.',
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write --extract_other outputs (default: same directory as --yaml_file).",
    )
    args = parser.parse_args()

    if not args.yaml_file.exists():
        print(f"Error: file not found: {args.yaml_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.yaml_file) as f:
        data = yaml.safe_load(f)

    fmt = detect_format(data)

    print(f"\n{'=' * 70}")
    print(f"  File : {args.yaml_file.name}")
    print(f"  Path : {args.yaml_file.parent}")

    if fmt == "pointwise":
        entry_count = len(data)
        print(f"  Type : POINTWISE  |  entries: {entry_count}")
        print(f"{'=' * 70}")
        labels, by_area = parse_pointwise(data)
    else:
        pair_count = len(data)
        idea_count = sum(
            sum(1 for m in pair.get("metadata", {}).values() if isinstance(m, dict))
            for pair in data.values()
        )
        print(f"  Type : PAIRWISE   |  pairs: {pair_count}  |  ideas: {idea_count}")
        print(f"{'=' * 70}")
        labels, by_area = parse_pairwise(data)

    all_labels = sorted(labels.keys())
    print_label_histogram(labels)
    print_area_histogram(by_area, all_labels)
    print()

    if args.extract_other:
        out_dir = args.output_dir if args.output_dir else args.yaml_file.parent
        extract_other(args.yaml_file, out_dir, fmt, data)


if __name__ == "__main__":
    main()
