#!/usr/bin/env python3
"""Aggregate per-ablation reports into a cross-model delta range table.

Reads a YAML config listing per-ablation Markdown report files and emits one
Markdown table per evaluation setting (pointwise / pairwise). Each table has:
  - row 1+ = one ablation per report
  - columns = [min, max] (range) of deltas between baseline and ablation across models

Usage:
    python scripts/ablation_delta_report.py config.yaml
    python scripts/ablation_delta_report.py config.yaml --out report.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

# Columns that are not evaluation metrics — excluded from computation.
_SKIP_COLS = {"Model", "Samples", "Cost ($)", "Ties"}

_FIRST_FLOAT_RE = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")

# ── low-level table helpers ────────────────────────────────────────────────────

def _extract_float(cell: str) -> float | None:
    """Return the first number in a cell, ignoring delta annotations and markers."""
    cell = cell.strip()
    if not cell or cell == "—":
        return None
    m = _FIRST_FLOAT_RE.search(cell)
    return float(m.group()) if m else None

def _parse_md_table(lines: list[str], header_idx: int) -> tuple[list[str], list[dict]]:
    """Parse a pipe-delimited Markdown table whose header is at `header_idx`.

    Returns (column_names, rows) where rows is a list of {col: float | None}.
    The "Model" column is returned as a plain string (bold markers stripped).
    """
    raw_cols = [c.strip() for c in lines[header_idx].split("|")[1:-1]]
    rows: list[dict] = []
    for line in lines[header_idx + 2:]:  # +2 to skip header + separator
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells:
            break
        row: dict = {"Model": re.sub(r"\*+", "", cells[0]).strip()}
        for col, cell in zip(raw_cols[1:], cells[1:]):
            row[col] = _extract_float(cell)
        rows.append(row)
    return raw_cols, rows

# ── per-report parsing ─────────────────────────────────────────────────────────

def parse_report(path: Path) -> dict:
    """Parse one ablation report and return its key tables."""
    lines = path.read_text(encoding="utf-8").splitlines()

    title = ""
    data: dict = {
        "pointwise": {"baseline": None, "ablation": None},
        "pairwise":  {"baseline": None, "ablation": None},
    }

    cur_section: str | None = None
    cur_subsection: str | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("# Ablation Report:") and not title:
            title = stripped.removeprefix("# Ablation Report:").strip()

        elif stripped == "## Pointwise Results":
            cur_section, cur_subsection = "pointwise", None

        elif stripped == "## Pairwise Results":
            cur_section, cur_subsection = "pairwise", None

        elif stripped == "### Baseline Results" and cur_section:
            cur_subsection = "baseline"

        elif stripped == "### Ablation Results" and cur_section:
            cur_subsection = "ablation"

        elif (
            cur_section
            and cur_subsection
            and data[cur_section][cur_subsection] is None
            and line.startswith("| Model |")
        ):
            cols, rows = _parse_md_table(lines, i)
            data[cur_section][cur_subsection] = {"cols": cols, "rows": rows}
            cur_subsection = None  # captured — don't re-parse later tables

    return {"title": title, **data}

# ── delta computation ───────────────────────────────────────────────────────

def delta_stats_row(baseline_table: dict, ablation_table: dict) -> dict[str, dict[str, float] | None]:
    """Compute min/max/range of deltas between baseline and ablation across models."""
    cols = baseline_table["cols"]
    base_rows = {r["Model"]: r for r in baseline_table["rows"]}
    abl_rows = {r["Model"]: r for r in ablation_table["rows"]}
    
    result: dict[str, dict[str, float] | None] = {}
    for col in cols:
        if col in _SKIP_COLS or col == "Model":
            continue
            
        deltas = []
        for model, base_row in base_rows.items():
            if model in abl_rows:
                v_base = base_row.get(col)
                v_abl = abl_rows[model].get(col)
                if v_base is not None and v_abl is not None:
                    deltas.append(v_abl - v_base)
        
        if deltas:
            d_min = min(deltas)
            d_max = max(deltas)
            result[col] = {
                "min": d_min,
                "max": d_max,
                "range": d_max - d_min
            }
        else:
            result[col] = None
    return result

# ── report rendering ───────────────────────────────────────────────────────────

def render_table(names: list[str], stats_rows: list[dict], metrics: list[str]) -> str:
    # Pre-compute per-column max range for bold highlighting.
    col_max_range: dict[str, float] = {}
    for m in metrics:
        ranges = [r[m]["range"] for r in stats_rows if r.get(m) is not None]
        if ranges:
            col_max_range[m] = max(ranges)

    header = "| Ablation | " + " | ".join(metrics) + " |"
    sep    = "| --- | "    + " | ".join(["---"] * len(metrics)) + " |"
    lines  = [header, sep]
    for name, row in zip(names, stats_rows):
        cells = []
        for m in metrics:
            stats = row.get(m)
            if stats is None:
                cells.append("—")
            else:
                # Scale to points (out of 100)
                d_min = stats['min'] * 100
                d_max = stats['max'] * 100
                formatted = f"{d_min:+.1f} / {d_max:+.1f}"
                if stats['range'] == col_max_range.get(m):
                    formatted = f"**{formatted}**"
                cells.append(formatted)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, help="YAML config file (positional)")
    parser.add_argument("--config", dest="config_flag", type=Path, help="YAML config file (named)")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output Markdown file (default: stdout)",
    )
    args = parser.parse_args()

    config_path = args.config_flag or args.config
    if config_path is None:
        parser.error("a config file is required (positional or --config)")

    if not config_path.exists():
        sys.exit(f"ERROR: config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).parent.parent.parent

    # ── collect report paths ──────────────────────────────────────────────────
    report_paths: list[Path] = []
    for entry in cfg.get("reports", []):
        raw = entry["path"] if isinstance(entry, dict) else entry
        p = Path(raw)
        if not p.is_absolute():
            p = repo_root / p
        report_paths.append(p)

    if not report_paths:
        sys.exit("ERROR: no reports listed in config")

    # ── parse ─────────────────────────────────────────────────────────────────
    parsed: list[dict] = []
    for p in report_paths:
        if not p.exists():
            print(f"WARNING: {p} not found — skipping", file=sys.stderr)
            continue
        print(f"Parsing {p.name} …", file=sys.stderr)
        parsed.append(parse_report(p))

    if not parsed:
        sys.exit("ERROR: no reports could be parsed")

    # optional metric overrides from config
    cfg_pointwise_metrics: list[str] | None = cfg.get("pointwise_metrics")
    cfg_pairwise_metrics:  list[str] | None = cfg.get("pairwise_metrics")

    sections: list[str] = []

    for setting in ("pointwise", "pairwise"):
        # ── locate baseline (use first report that carries it) ────────────────
        baseline_table: dict | None = None
        for p in parsed:
            bl = p[setting]["baseline"]
            if bl:
                baseline_table = bl
                break

        if not baseline_table:
            print(f"WARNING: no {setting} baseline found — skipping section", file=sys.stderr)
            continue

        # ── choose metric columns ─────────────────────────────────────────────
        if setting == "pointwise":
            metrics = cfg_pointwise_metrics or [
                c for c in baseline_table["cols"]
                if c not in _SKIP_COLS and c != "Model"
            ]
        else:
            metrics = cfg_pairwise_metrics or [
                c for c in baseline_table["cols"]
                if c not in _SKIP_COLS and c != "Model"
            ]

        # ── build rows ────────────────────────────────────────────────────────
        names: list[str] = []
        stats_rows: list[dict] = []

        missing = 0
        for p in parsed:
            abl = p[setting]["ablation"]
            if not abl:
                missing += 1
                continue
            names.append(p["title"] or "(unnamed)")
            stats_rows.append(delta_stats_row(baseline_table, abl))

        if not names:
            print(f"WARNING: no {setting} ablation results found", file=sys.stderr)
            continue

        table_md = render_table(names, stats_rows, metrics)
        sections.append(f"## {setting.capitalize()} — Delta Range Across Models (min / max points)\n\n{table_md}")

    if not sections:
        sys.exit("ERROR: no sections could be generated")

    output = "\n\n".join(sections) + "\n"

    out_raw = args.out or (Path(cfg["output"]) if cfg.get("output") else None)
    if out_raw:
        out_path = out_raw if out_raw.is_absolute() else repo_root / out_raw
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Report written to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
