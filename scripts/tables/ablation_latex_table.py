#!/usr/bin/env python3
"""
Print a LaTeX booktabs table from a per-ablation report markdown file.

Usage (CLI):
    python scripts/ablation_latex_table.py <report.md> --mode pointwise --metrics accuracy f1_macro
    python scripts/ablation_latex_table.py <report.md> --mode pairwise  --metrics pairwise_accuracy pairwise_accuracy_no_ties

Usage (YAML config):
    python scripts/ablation_latex_table.py config.yaml
    python scripts/ablation_latex_table.py config.yaml --mode pairwise   # override a single field

YAML config format:
    report: output/ablation_sweeps/my_merge_test/per_ablation_reports/low_judge_reasoning_filtered.md
    mode: pointwise
    metrics:
      - accuracy
      - f1_macro
      - f1_pos

Available metrics:
  pointwise : accuracy, f1_macro, f1_pos, f1_neg, precision_pos, precision_neg,
              recall_pos, recall_neg, support, cost_usd
  pairwise  : pairwise_accuracy, pairwise_accuracy_strict, pairwise_accuracy_no_ties,
              n_ties, support, cost_usd
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

# Internal key → markdown column header (mirrors _SETUP_METRIC_LABELS in merge_ablation_runs.py)
_MD_COL_HEADER: dict[str, dict[str, str]] = {
    "pairwise": {
        "pairwise_accuracy":        "Acc (with ties)",
        "pairwise_accuracy_strict": "Acc (with ties, strict)",
        "pairwise_accuracy_no_ties":"Acc (w/o ties)",
        "n_ties":                   "Ties",
        "support":                  "Samples",
        "cost_usd":                 "Cost ($)",
    },
    "pointwise": {
        "accuracy":      "Accuracy",
        "f1_macro":      "F1 macro",
        "f1_pos":        "F1 POS",
        "f1_neg":        "F1 NEG",
        "precision_pos": "Prec POS",
        "precision_neg": "Prec NEG",
        "recall_pos":    "Rec POS",
        "recall_neg":    "Rec NEG",
        "support":       "Samples",
        "cost_usd":      "Cost ($)",
    },
}

# Internal key → LaTeX column header displayed in the table
_LATEX_LABEL: dict[str, str] = {
    "accuracy":                 "Accuracy",
    "f1_macro":                 "F1 macro",
    "f1_pos":                   "F1 POS",
    "f1_neg":                   "F1 NEG",
    "precision_pos":            "Prec POS",
    "precision_neg":            "Prec NEG",
    "recall_pos":               "Rec POS",
    "recall_neg":               "Rec NEG",
    "support":                  "Samples",
    "cost_usd":                 "Cost (\\$)",
    "pairwise_accuracy":        "Acc (w/ ties)",
    "pairwise_accuracy_strict": "Acc (strict)",
    "pairwise_accuracy_no_ties":"Acc (w/o ties)",
    "n_ties":                   "Ties",
}

_INTEGER_METRICS = {"support", "n_ties"}


def _find_section(content: str, mode: str) -> str:
    header = "## Pointwise Results" if mode == "pointwise" else "## Pairwise Results"
    start = content.find(header)
    if start == -1:
        return ""
    nxt = re.search(r"\n## ", content[start + 1:])
    end = start + 1 + nxt.start() if nxt else len(content)
    return content[start:end]


def _extract_subsection(section: str, sub_header: str) -> list[str]:
    start = section.find(sub_header)
    if start == -1:
        return []
    end = section.find("\n### ", start + 1)
    chunk = section[start: end if end != -1 else len(section)]
    return chunk.splitlines()


def _parse_md_table(lines: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    table_lines = [l.strip() for l in lines if l.strip().startswith("|")]
    if not table_lines:
        return [], []
    headers = [h.strip() for h in table_lines[0].split("|")[1:-1]]
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:  # skip separator row
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return headers, rows


def _clean_model(name: str) -> str:
    return name.strip("*").strip()


def _strip_delta(val: str) -> str:
    """'0.7781 (-1.2%)*' → '0.7781',  '320.0 (0.0)' → '320.0',  '—' → '—'"""
    val = val.strip()
    m = re.match(r"^([0-9.]+)\s*\(", val)
    if m:
        return m.group(1)
    return re.sub(r"[*?]+$", "", val).strip()


def _fmt(val: str, metric: str) -> str:
    cleaned = _strip_delta(val)
    if cleaned in ("—", "", "-"):
        return "{---}"
    try:
        f = float(cleaned)
        if metric in _INTEGER_METRICS:
            return f"{f:.0f}"
        return f"{f:.3f}"
    except ValueError:
        return cleaned


def _load_config(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        sys.exit(f"Error reading config {path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a LaTeX booktabs table from a per-ablation report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "report",
        help="Path to the per-ablation .md report file, or a .yaml config file.",
    )
    parser.add_argument("--mode", choices=["pointwise", "pairwise"],
                        help="Evaluation mode (overrides config file).")
    parser.add_argument("--metrics", nargs="+",
                        help="Metric keys to include (overrides config file).")
    args = parser.parse_args()

    # If the positional arg is a YAML file, load it and merge with CLI overrides.
    first_arg = Path(args.report)
    if first_arg.suffix in (".yaml", ".yml"):
        cfg = _load_config(first_arg)
        report_path = Path(cfg.get("report", ""))
        mode        = args.mode    or cfg.get("mode")
        metrics     = args.metrics or cfg.get("metrics") or []
        model_order = cfg.get("model_order") or []
    else:
        report_path = first_arg
        mode        = args.mode
        metrics     = args.metrics or []
        model_order = []

    if not report_path or not str(report_path):
        sys.exit("Error: no report path provided.")
    if not report_path.exists():
        sys.exit(f"Error: not found: {report_path}")
    if not mode:
        sys.exit("Error: --mode is required (or set 'mode' in the config file).")
    if not metrics:
        sys.exit("Error: --metrics is required (or set 'metrics' in the config file).")
    available = _MD_COL_HEADER[mode]
    bad = [m for m in metrics if m not in available]
    if bad:
        sys.exit(
            f"Unknown metric(s) for {mode}: {', '.join(bad)}\n"
            f"Available: {', '.join(available)}"
        )

    content = report_path.read_text(encoding="utf-8", errors="ignore")
    section = _find_section(content, mode)
    if not section:
        sys.exit(f"No '{mode}' results section found in {report_path.name}.")

    baseline_lines = _extract_subsection(section, "### Baseline Results")
    ablation_lines = _extract_subsection(section, "### Ablation Results")
    _, baseline_rows = _parse_md_table(baseline_lines)
    _, ablation_rows  = _parse_md_table(ablation_lines)
    if not baseline_rows:
        sys.exit("Could not parse baseline results table.")

    baseline_by_model = {_clean_model(r["Model"]): r for r in baseline_rows}
    ablation_by_model  = {_clean_model(r["Model"]): r for r in ablation_rows}
    models_in_report = [_clean_model(r["Model"]) for r in baseline_rows]
    if model_order:
        models = [m for m in model_order if m in set(models_in_report)]
        models += [m for m in models_in_report if m not in set(model_order)]
    else:
        models = models_in_report

    col_hdrs   = [available[m]           for m in metrics]
    latex_hdrs = [_LATEX_LABEL.get(m, m) for m in metrics]

    n = len(metrics)
    col_spec   = "l" + "r" * n
    metric_row = " & ".join(latex_hdrs)

    def _make_table(title: str, label: str, row_data: dict[str, dict]) -> list[str]:
        t: list[str] = []
        t.append(r"\begin{table}[htbp]")
        t.append(r"    \centering")
        t.append(r"    \small")
        t.append(f"    \\begin{{tabular}}{{{col_spec}}}")
        t.append(r"        \toprule")
        t.append(f"        Model & {metric_row} \\\\")
        t.append(r"        \midrule")
        for model in models:
            r = row_data.get(model, {})
            vals = [_fmt(r.get(ch, "—"), mk) for ch, mk in zip(col_hdrs, metrics)]
            t.append("        " + " & ".join(["\\texttt{" + model + "}"] + vals) + " \\\\")
        t.append(r"        \bottomrule")
        t.append(r"    \end{tabular}")
        t.append(f"    \\caption{{{title}}}")
        t.append(f"    \\label{{tab:{label}}}")
        t.append(r"\end{table}")
        return t

    out = (
        _make_table("Baseline results", "baseline", baseline_by_model)
        + [""]
        + _make_table("Ablation results", "ablation", ablation_by_model)
    )
    print("\n".join(out))


if __name__ == "__main__":
    main()
