"""
cost_report_writer.py — Shared helper for writing human-readable Markdown cost reports.

Used by both accuracy_test.py (eval subprocess) and create_benchmark_instances.py
(data creation subprocess) so the format is consistent.
"""


def write_cost_report_md(cost_report: dict, path: str, title: str = "Cost Report") -> None:
    """Write a human-readable Markdown cost report to *path*."""
    cached_input      = cost_report.get("total_cached_input_tokens", 0)
    cache_creation    = cost_report.get("total_cache_creation_tokens", 0)
    instances_processed = cost_report.get("instances_processed")
    instances_total     = cost_report.get("instances_total")
    estimated_full_cost = cost_report.get("estimated_full_cost_usd")

    # Build the instances row label
    if instances_processed is not None and instances_total is not None:
        instances_value = f"{instances_processed:,} / {instances_total:,}"
    elif instances_processed is not None:
        instances_value = f"{instances_processed:,}"
    else:
        instances_value = None

    lines = [
        f"# {title}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total cost | **${cost_report['total_cost_usd']:.4f}** |",
        f"| LLM calls | {cost_report['total_calls']:,} |",
        f"| Input tokens | {cost_report['total_input_tokens']:,} |",
        f"| Cached input tokens | {cached_input:,} |",
        f"| Cache creation tokens | {cache_creation:,} |",
        f"| Output tokens | {cost_report['total_output_tokens']:,} |",
    ]

    if instances_value is not None:
        lines.append(f"| Examples processed | {instances_value} |")

    if (
        estimated_full_cost is not None
        and instances_total is not None
        and instances_processed is not None
        and instances_total > instances_processed
    ):
        lines.append(f"| Estimated full set cost | **${estimated_full_cost:.4f}** |")

    lines += [
        "",
        "## Breakdown by Model",
        "",
        "| Model | Calls | Input | Cached input | Cache creation | Output | Cost (USD) |",
        "|-------|------:|------:|-------------:|---------------:|-------:|-----------:|",
    ]
    for m in cost_report.get("models", {}).values():
        lines.append(
            f"| `{m['model']}` "
            f"| {m['call_count']:,} "
            f"| {m['input_tokens']:,} "
            f"| {m['cached_input_tokens']:,} "
            f"| {m['cache_creation_tokens']:,} "
            f"| {m['output_tokens']:,} "
            f"| ${m['cost_usd']:.4f} |"
        )

    by_stage = cost_report.get("by_stage", {})
    # Show stage breakdown only when there is meaningful data
    # Hide the "unknown" stage if it has zero cost (nothing untagged)
    visible_stages = {
        s: d for s, d in by_stage.items()
        if s != "unknown" or d.get("total_cost_usd", 0) > 0
    }
    if visible_stages:
        lines += [
            "",
            "## Breakdown by Stage",
            "",
            "| Stage | Calls | Input | Cached input | Cache creation | Output | Cost (USD) |",
            "|-------|------:|------:|-------------:|---------------:|-------:|-----------:|",
        ]
        for stage, d in sorted(visible_stages.items(), key=lambda kv: kv[1].get("total_cost_usd", 0), reverse=True):
            lines.append(
                f"| `{stage}` "
                f"| {d.get('total_calls', 0):,} "
                f"| {d.get('total_input_tokens', 0):,} "
                f"| {d.get('total_cached_input_tokens', 0):,} "
                f"| {d.get('total_cache_creation_tokens', 0):,} "
                f"| {d.get('total_output_tokens', 0):,} "
                f"| ${d.get('total_cost_usd', 0):.4f} |"
            )

    lines += [
        "",
        f"> {cost_report.get('pricing_note', '')}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
