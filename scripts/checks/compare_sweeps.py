"""Compare accuracy results across multiple ablation sweeps.

Shows per-(setup, model): accuracy from each sweep, and how many
predictions changed on the shared instances between consecutive sweeps.

Usage:
    python scripts/compare_sweeps.py --config scripts/compare_sweeps.yaml
    python scripts/compare_sweeps.py --config scripts/compare_sweeps.yaml --output report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunResult:
    sweep_label: str
    setup: str
    model: str
    predictions: dict[str, int] = field(default_factory=dict)  # instance_id -> prediction


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_DIGIT_HYPHEN_DIGIT_RE = re.compile(r"(\d)-(\d)")


def _normalize_model(name: str) -> str:
    """gpt-5-2 → gpt-5.2"""
    return _DIGIT_HYPHEN_DIGIT_RE.sub(r"\1.\2", name)


def _parse_scores(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    return {str(k): int(v["elo_selected"]) for k, v in data.items()}


def _infer_setup_and_model(report_dir: Path) -> tuple[str, str]:
    parts = report_dir.parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "accuracy_test_artifacts")
    except StopIteration:
        setup_model = report_dir.parent.name
        setup = report_dir.parent.parent.name
        model = setup_model[len(setup) + 1:] if setup_model.startswith(setup + "-") else setup_model
        return setup, _normalize_model(model)
    setup_model = parts[idx - 1]
    setup = parts[idx - 2]
    model = setup_model[len(setup) + 1:] if setup_model.startswith(setup + "-") else setup_model
    return setup, _normalize_model(model)


def collect_results(sweep_path: Path, label: str) -> list[RunResult]:
    results = []
    for scores_path in sweep_path.rglob("scores.json"):
        report_dir = scores_path.parent.parent  # run_*/ -> accuracy_test_artifacts/timestamp/
        setup, model = _infer_setup_and_model(report_dir)
        predictions = _parse_scores(scores_path)
        results.append(RunResult(label, setup, model, predictions))
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def compare_and_print(
    all_results: list[RunResult],
    sweep_labels: list[str],
    markdown_output: str | None = None,
) -> None:
    lookup = {(r.setup, r.model, r.sweep_label): r for r in all_results}
    pairs = sorted({(r.setup, r.model) for r in all_results})
    has_two = len(sweep_labels) >= 2

    md_lines: list[str] = []

    current_setup = None
    for setup, model in pairs:
        if setup != current_setup:
            current_setup = setup
            print(f"\n### {setup}")
            # Header row
            print(f"  {'Model':<30}", end="")
            if has_two:
                print(f"  {'Changed/N':>12}", end="")
            print()
            print("  " + "-" * (30 + (16 if has_two else 0)))

            md_lines.append(f"\n#### {setup}\n")
            md_header = ["Model"] + (["Changed / N"] * (len(sweep_labels) - 1) if has_two else [])
            md_lines.append("| " + " | ".join(md_header) + " |")
            md_lines.append("|" + "|".join(["---"] * len(md_header)) + "|")

        runs = [lookup.get((setup, model, lbl)) for lbl in sweep_labels]

        print(f"  {model:<30}", end="")
        md_row = [model]

        if has_two:
            for i in range(len(sweep_labels) - 1):
                r_a, r_b = runs[i], runs[i + 1]
                if r_a and r_b:
                    common = set(r_a.predictions) & set(r_b.predictions)
                    n_changed = sum(1 for iid in common if r_a.predictions[iid] != r_b.predictions[iid])
                    changed_str = f"{n_changed} / {len(common)}"
                else:
                    changed_str = "—"
                print(f"  {changed_str:>12}", end="")
                md_row.append(changed_str)
        print()
        md_lines.append("| " + " | ".join(md_row) + " |")

    if markdown_output:
        Path(markdown_output).write_text("\n".join(md_lines) + "\n")
        print(f"\nMarkdown written to: {markdown_output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    config = yaml.safe_load(config_path.read_text())
    repo_root = Path(__file__).parent.parent.parent

    all_results: list[RunResult] = []
    sweep_labels: list[str] = []

    for entry in config.get("sweeps", []):
        raw_path = entry.get("path", "")
        label = entry.get("label", raw_path)
        sweep_path = Path(raw_path) if Path(raw_path).is_absolute() else repo_root / raw_path

        if not sweep_path.exists():
            print(f"Warning: not found, skipping: {sweep_path}", file=sys.stderr)
            continue

        print(f"Scanning {sweep_path}  [{label}]")
        results = collect_results(sweep_path, label)
        print(f"  {len(results)} run(s) found.")
        all_results.extend(results)
        if label not in sweep_labels:
            sweep_labels.append(label)

    if not all_results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    compare_and_print(all_results, sweep_labels, markdown_output=args.output)


if __name__ == "__main__":
    main()
