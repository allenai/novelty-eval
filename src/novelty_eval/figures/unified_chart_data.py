"""
On-disk format for unified ablation chart data.

`merge_ablation_runs.py` writes one JSON file per (setup, variant) alongside the
unified chart PNGs; `rechart_unified.py` reads it back to re-render the figure
without re-running the merge pipeline — no artifact copying, no report
subprocesses, no filtered-metric recomputation, and no n=100,000 paired
bootstraps.

Every non-baseline ablation that had data is written, not just the panels that
were rendered, so panel selection, ordering, titles, metric columns, and model
rows can all be changed offline.

Deliberately dependency-light (json + pathlib) so the rechart CLI starts fast.
"""
import json
import math
from datetime import datetime
from pathlib import Path

# v2 added ci_low/ci_high to each bootstrap entry. v1 files still load; their
# panels simply carry no interval, so dot-and-interval figures are unavailable
# for them until the merge is re-run.
SCHEMA_VERSION = 2
_READABLE_SCHEMA_VERSIONS = (1, 2)

# Bootstrap results are keyed by a (model, metric) tuple in memory, but JSON
# object keys must be strings — the two are joined with this separator on disk.
_KEY_SEP = "||"


def _finite_or_none(value):
    """Keep a finite number, else None — NaN/inf are not valid JSON."""
    if value is None or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _clean_metrics(metrics_by_model: dict | None) -> dict:
    """Keep only JSON-safe numeric metric values, dropping anything exotic."""
    return {
        model: {
            k: v
            for k, v in (metrics or {}).items()
            if v is None or isinstance(v, (int, float))
        }
        for model, metrics in (metrics_by_model or {}).items()
    }


def serialize_panel(panel: dict) -> dict:
    """Convert an in-memory unified-chart panel into its on-disk form."""
    return {
        "canonical": panel.get("canonical", ""),
        "title": panel.get("desc", ""),
        "index_key": panel.get("index_key", ""),
        "baseline_name": panel.get("baseline_name", ""),
        "baseline_metrics": _clean_metrics(panel.get("baseline_metrics")),
        "ablation_metrics": _clean_metrics(panel.get("ablation_metrics")),
        "bootstrap_results": {
            f"{model}{_KEY_SEP}{metric}": {
                "significant": bool(res.get("significant", False)),
                "reliable": bool(res.get("reliable", True)),
                "ci_low": _finite_or_none(res.get("ci_low")),
                "ci_high": _finite_or_none(res.get("ci_high")),
            }
            for (model, metric), res in (panel.get("bootstrap_results") or {}).items()
        },
    }


def dump(
    path: Path,
    *,
    setup: str,
    name_suffix: str,
    report_file: str,
    all_models: list[str],
    metric_labels: dict[str, str],
    available_metrics: dict[str, str],
    rendered: list[str],
    panels: dict[str, dict],
) -> Path:
    """
    Write the data behind one unified chart figure.

    Args:
        path: destination .json (conventionally the PNG path with a .json suffix).
        setup: "pointwise" or "pairwise".
        name_suffix: "" for the unfiltered variant, "_filtered" for the filtered one.
        report_file: the accuracy report the metrics were parsed from.
        all_models: judge models, in row order.
        metric_labels: metric_key → label for the columns that were rendered.
        available_metrics: metric_key → label for every metric of this setup,
            so columns can be added back when re-charting.
        rendered: canonical ablation names in the order they were drawn.
        panels: canonical name → in-memory panel dict (may include entries that
            were not rendered).
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "setup": setup,
        "name_suffix": name_suffix,
        "variant": name_suffix.lstrip("_") or "unfiltered",
        "report_file": report_file,
        "all_models": list(all_models),
        "metric_labels": dict(metric_labels),
        "available_metrics": dict(available_metrics),
        "rendered": list(rendered),
        "panels": {name: serialize_panel(p) for name, p in panels.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(path: Path | str) -> dict:
    """
    Read a file written by `dump`, restoring bootstrap keys to (model, metric)
    tuples so panels can be handed straight to `_generate_unified_heatmap`.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("schema_version")
    if version not in _READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"{path}: unsupported schema_version {version!r} "
            f"(this build reads versions {_READABLE_SCHEMA_VERSIONS}). "
            f"Re-run merge_ablation_runs.py --charts-only to regenerate it."
        )

    for panel in data.get("panels", {}).values():
        panel["bootstrap_results"] = {
            tuple(k.split(_KEY_SEP, 1)): v
            for k, v in (panel.get("bootstrap_results") or {}).items()
        }
    return data


def build_render_panels(
    data: dict,
    canonicals: list[str] | None = None,
    titles: dict[str, str] | None = None,
) -> list[dict]:
    """
    Select and order panels from loaded data, shaped for `_generate_unified_heatmap`.

    Args:
        data: the dict returned by `load`.
        canonicals: canonical names to draw, in order. Defaults to the panels
            that were originally rendered.
        titles: canonical name → replacement panel title.

    Raises:
        KeyError: if a requested canonical name is not present in the file.
    """
    panels_by_name = data.get("panels", {})
    wanted = canonicals if canonicals is not None else data.get("rendered", [])
    titles = titles or {}

    missing = [c for c in wanted if c not in panels_by_name]
    if missing:
        raise KeyError(
            f"no data for {missing} — available: {sorted(panels_by_name)}"
        )

    return [
        {
            "desc": titles.get(c, panels_by_name[c].get("title", c)),
            "baseline_metrics": panels_by_name[c]["baseline_metrics"],
            "ablation_metrics": panels_by_name[c]["ablation_metrics"],
            "bootstrap_results": panels_by_name[c]["bootstrap_results"],
        }
        for c in wanted
    ]
