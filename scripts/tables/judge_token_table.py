#!/usr/bin/env python3
"""
Build a judge token-usage table from ablation sweep artifacts.

Reads per-call token usage straight out of the batch output JSONL files, so the
table reports the *distribution* of tokens per comparison (mean, median,
percentiles) rather than only the run totals stored in cost_report.json.

Usage (YAML config):
    python scripts/judge_token_table.py scripts/judge_token_table.yaml
    python scripts/judge_token_table.py scripts/judge_token_table.yaml --style slide
    python scripts/judge_token_table.py scripts/judge_token_table.yaml --model gpt-5.4 --out table.md

YAML config format:
    models: [gpt-5.4]              # one table per model
    style: full                    # full | slide
    variants:                      # ablation name -> label shown in the table
      - {name: ai_researcher_base, label: base}
      - {name: ai_researcher_no_review, label: no review mention}
    groups:                        # one column block (slide) / section (full)
      - label: Human+AI
        artifacts: output/ablation_sweeps/merge_ai_researcher_vanilla_ai-plan/artifacts
        prefix: pairwise-vanilla-ai-plan_
        instances: output/.../iclr_test_instances.yaml   # optional, enables the
                                                         # correct-vs-wrong split

Styles:
  full  — one section per model, one row per (group, variant), full distribution.
  slide — one compact table per model: rows are variants, one mean/median column
          per group plus accuracy.

Notes:
  * Anthropic runs without extended thinking emit a constant ~10 output tokens
    per call; the distribution columns are degenerate there by design.
  * `--split-correct` adds mean output tokens for correct vs wrong decisions.
    Requires `instances` on the group (to read gold labels).
"""

import argparse
import json
import re
import statistics as st
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class Cell:
    """Token stats for one (group, model, variant) run."""
    group: str
    model: str
    variant: str
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    correct_out: list[int] = field(default_factory=list)
    wrong_out: list[int] = field(default_factory=list)
    accuracy: float | None = None
    artifact_dir: Path | None = None
    note: str | None = None

    @property
    def n_calls(self) -> int:
        return len(self.output_tokens)


# ── artifact discovery ────────────────────────────────────────────────────────

def find_artifact_dir(artifacts: Path, prefix: str, variant: str, model: str) -> Path | None:
    """Newest artifact dir matching '<prefix><variant>-<model>_<timestamp>'."""
    matches = sorted(artifacts.glob(f"{prefix}{variant}-{model}_*"))
    return matches[-1] if matches else None


def find_batch_outputs(artifact_dir: Path) -> list[Path]:
    return sorted(artifact_dir.glob("run_*/batch_output*.jsonl"))


# ── usage parsing ─────────────────────────────────────────────────────────────

def parse_usage(record: dict) -> tuple[int, int] | None:
    """Return (input_tokens, output_tokens) for one batch record, or None.

    Handles both the OpenAI batch shape (response.body.usage) and the Anthropic
    one (result.message.usage).
    """
    body = record.get("response", {}).get("body")
    if isinstance(body, dict) and "usage" in body:
        u = body["usage"]
        return u["prompt_tokens"], u["completion_tokens"]

    message = record.get("result", {}).get("message")
    if isinstance(message, dict) and "usage" in message:
        u = message["usage"]
        billed_input = (
            u.get("input_tokens", 0)
            + u.get("cache_read_input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0)
        )
        return billed_input, u["output_tokens"]

    return None


def instance_index(custom_id: str) -> int | None:
    """Instance index out of a custom_id like 'cmp__21__0__1__0'."""
    nums = re.findall(r"\d+", custom_id or "")
    return int(nums[0]) if nums else None


# ── loading ───────────────────────────────────────────────────────────────────

def load_gold(instances_path: Path) -> dict[int, int]:
    data = yaml.safe_load(instances_path.read_text())
    return {int(k): v["expected_winners"][0] for k, v in data.items()}


def load_picks(artifact_dir: Path) -> dict[int, int]:
    """Instance index -> chosen idea index, from the first run's scores.json."""
    picks: dict[int, int] = {}
    for scores_path in sorted(artifact_dir.glob("run_*/scores.json")):
        for key, entry in json.loads(scores_path.read_text()).items():
            comparisons = entry.get("comparisons") or []
            if comparisons:
                picks[int(key)] = comparisons[0]["winner"]
        break
    return picks


def read_accuracy(artifact_dir: Path) -> float | None:
    report = artifact_dir / "accuracy_report.txt"
    if not report.exists():
        return None
    m = re.search(r"Accuracy \(with ties\):\s*([\d.]+)", report.read_text())
    return float(m.group(1)) if m else None


def read_cost_totals(artifact_dir: Path) -> tuple[int, int, int] | None:
    """Fallback when no batch outputs exist: (input, output, calls) totals."""
    path = artifact_dir / "cost_report.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d["total_input_tokens"], d["total_output_tokens"], d["total_calls"]


def load_cell(group: dict, model: str, variant: dict, split_correct: bool) -> Cell:
    label_group, label_variant = group["label"], variant["label"]
    cell = Cell(group=label_group, model=model, variant=label_variant)

    artifacts = Path(group["artifacts"])
    artifact_dir = find_artifact_dir(artifacts, group.get("prefix", ""), variant["name"], model)
    if artifact_dir is None:
        cell.note = "no artifact dir found"
        return cell

    cell.artifact_dir = artifact_dir
    cell.accuracy = read_accuracy(artifact_dir)

    gold: dict[int, int] = {}
    picks: dict[int, int] = {}
    if split_correct and group.get("instances"):
        gold = load_gold(Path(group["instances"]))
        picks = load_picks(artifact_dir)

    batch_files = find_batch_outputs(artifact_dir)
    if not batch_files:
        totals = read_cost_totals(artifact_dir)
        if totals is None:
            cell.note = "no batch outputs and no cost_report.json"
            return cell
        # Synthesise a flat distribution so means stay correct; flag it.
        total_in, total_out, calls = totals
        cell.input_tokens = [round(total_in / calls)] * calls
        cell.output_tokens = [round(total_out / calls)] * calls
        cell.note = "totals only (no per-call data)"
        return cell

    for batch_file in batch_files:
        for line in batch_file.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            usage = parse_usage(record)
            if usage is None:
                continue
            tin, tout = usage
            cell.input_tokens.append(tin)
            cell.output_tokens.append(tout)

            idx = instance_index(record.get("custom_id", ""))
            if gold and idx in gold and idx in picks:
                (cell.correct_out if picks[idx] == gold[idx] else cell.wrong_out).append(tout)
        break  # first run only; multi-run sweeps are averaged elsewhere

    return cell


# ── formatting ────────────────────────────────────────────────────────────────

def pct(values: list[int], p: float) -> int:
    s = sorted(values)
    return s[int(p * (len(s) - 1))]


def fmt_acc(acc: float | None) -> str:
    return f"{acc:.3f}" if acc is not None else "—"


def fmt_mean(values: list[int]) -> str:
    return f"{round(st.mean(values)):,}" if values else "—"


def render_full(cells: list[Cell], model: str, split_correct: bool) -> list[str]:
    header = ["Group", "Prompt", "In/call", "Out mean", "Out med", "p25–p75", "p90", "max", "Acc"]
    if split_correct:
        header += ["Out correct", "Out wrong"]

    lines = [f"### {model}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]

    for c in cells:
        if not c.output_tokens:
            lines.append(f"| {c.group} | {c.variant} | " + " | ".join(["—"] * (len(header) - 2)) + " |")
            continue
        out = c.output_tokens
        row = [
            c.group,
            c.variant,
            f"{round(st.mean(c.input_tokens)):,}",
            f"{round(st.mean(out)):,}",
            f"{round(st.median(out)):,}",
            f"{pct(out, .25):,}–{pct(out, .75):,}",
            f"{pct(out, .90):,}",
            f"{max(out):,}",
            fmt_acc(c.accuracy),
        ]
        if split_correct:
            row += [fmt_mean(c.correct_out), fmt_mean(c.wrong_out)]
        lines.append("| " + " | ".join(row) + " |")

    notes = {c.note for c in cells if c.note}
    if notes:
        lines += ["", "*" + "; ".join(sorted(notes)) + "*"]
    return lines + [""]


def render_slide(cells: list[Cell], model: str, groups: list[str], variants: list[str]) -> list[str]:
    by_key = {(c.group, c.variant): c for c in cells}

    header = ["Prompt"] + [f"{g} mean/med" for g in groups] + [f"Acc ({g})" for g in groups]
    lines = [f"### {model}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]

    for v in variants:
        row = [v]
        for g in groups:
            c = by_key.get((g, v))
            if c and c.output_tokens:
                row.append(f"{round(st.mean(c.output_tokens)):,} / {round(st.median(c.output_tokens)):,}")
            else:
                row.append("—")
        for g in groups:
            c = by_key.get((g, v))
            row.append(fmt_acc(c.accuracy) if c else "—")
        lines.append("| " + " | ".join(row) + " |")

    in_per_call = [
        f"{groups[i]}: {pct(c.input_tokens, .5):,}"
        for i, g in enumerate(groups)
        for c in [by_key.get((g, variants[0]))]
        if c and c.input_tokens
    ]
    if in_per_call:
        lines += ["", f"*Input tokens/call (median, base prompt): {'; '.join(in_per_call)}*"]
    return lines + [""]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--style", choices=["full", "slide"], default=None,
                        help="Override the style set in the config")
    parser.add_argument("--model", action="append", default=None,
                        help="Restrict to these models (repeatable)")
    parser.add_argument("--split-correct", action="store_true",
                        help="Add mean output tokens for correct vs wrong decisions "
                             "(needs `instances` on each group)")
    parser.add_argument("--out", default=None, help="Write Markdown here instead of stdout")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    style = args.style or cfg.get("style", "full")
    models = args.model or cfg["models"]
    groups = cfg["groups"]
    variants = cfg["variants"]
    split_correct = args.split_correct or cfg.get("split_correct", False)

    if style == "slide" and split_correct:
        print("warning: --split-correct is ignored in slide style", file=sys.stderr)

    lines: list[str] = ["# Judge token usage per comparison", ""]
    for model in models:
        cells = [
            load_cell(group, model, variant, split_correct)
            for group in groups
            for variant in variants
        ]
        missing = [c for c in cells if c.note == "no artifact dir found"]
        for c in missing:
            print(f"warning: no run found for {c.group} / {model} / {c.variant}", file=sys.stderr)

        if style == "slide":
            lines += render_slide(cells, model,
                                  [g["label"] for g in groups],
                                  [v["label"] for v in variants])
        else:
            lines += render_full(cells, model, split_correct)

    output = "\n".join(lines)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"Table written to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
