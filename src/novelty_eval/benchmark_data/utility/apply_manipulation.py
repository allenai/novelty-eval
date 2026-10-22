#!/usr/bin/env python3
"""
Apply an abstract manipulation prompt to every idea in an existing dataset YAML.

Reads a pairwise or pointwise benchmark_instances.yaml, applies the given
Jinja2 manipulation template to each unique abstract via an LLM, and writes
a new YAML with manipulated abstracts while preserving all other fields.

Pairwise format: each instance has an ``ideas`` dict (int -> abstract str).
Pointwise format: each instance has a single ``idea`` str.

Usage:
    python apply_manipulation.py \\
        --dataset output/benchmark_instances/pairwise_data/<ts>/benchmark_instances.yaml \\
        --template src/novelty_eval/benchmark_data/templates/extract_research_plan.jinja2 \\
        [--model gpt-4o-mini] \\
        [--output path/to/output.yaml] \\
        [--max-workers 8] \\
        [--reuse-from path/to/existing/manipulated.yaml]

    When --reuse-from is provided, POSITIVE ideas whose title matches an entry in
    the reference file are reused verbatim (no LLM call). Unmatched ideas are
    generated as normal. Output structure (order, winner IDs, metadata) is always
    taken from --dataset unchanged.
"""
import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Matches a bold field label with an empty or whitespace-only value
_EMPTY_FIELD_RE = re.compile(r"\*\*[^*]+\*\*:[ \t]*(?:\n|$)")
# Same pattern but captures the field name for stats
_EMPTY_FIELD_CAPTURE_RE = re.compile(r"\*\*([^*]+)\*\*:[ \t]*(?:\n|$)")
# Matches a bold field label (possibly mid-line) so we can force a newline before it
_SECTION_HEADER_RE = re.compile(r"(?<!\n)(\*\*[^*]+\*\*:)")

import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

if "SECRETS" not in os.environ:
    _project_root = Path(__file__).resolve().parents[3]
    _secrets_file = _project_root / "secrets.toml"
    if _secrets_file.exists():
        os.environ["SECRETS"] = str(_secrets_file)

from logging_utils import setup_logger
from novelty_eval.benchmark_data.create_benchmark_instances import (
    ManipulationDebugLog,
    _load_manipulation_template,
    _manipulate_paper,
)

try:
    from cost_tracker import GLOBAL_COST_TRACKER
except ImportError:
    GLOBAL_COST_TRACKER = None

try:
    from novelty_eval.cost_report_writer import write_cost_report_md as _write_cost_report_md
except ImportError:
    try:
        from cost_report_writer import write_cost_report_md as _write_cost_report_md
    except ImportError:
        _write_cost_report_md = None

LOGGER = setup_logger(output_dir="..", console_level="INFO")


def _load_secrets() -> None:
    try:
        import toml
    except ImportError:
        return
    project_root = Path(__file__).resolve().parents[3]
    secrets_file = project_root / "secrets.toml"
    if secrets_file.exists():
        secrets = toml.load(secrets_file)
        if "openai_key" in secrets:
            os.environ["OPENAI_API_KEY"] = secrets["openai_key"]


def _build_reuse_lookup(reuse_path: Path) -> dict[str, str]:
    """Build a {title: manipulated_text} mapping from an existing manipulated dataset.

    Only POSITIVE entries are indexed; title is used as the stable identifier.
    """
    with open(reuse_path) as f:
        reuse_data = yaml.safe_load(f)
    lookup: dict[str, str] = {}
    for instance in reuse_data.values():
        ideas = instance.get("ideas")
        metadata = instance.get("metadata", {})
        if ideas is not None:
            for idx, text in ideas.items():
                meta = metadata.get(idx, {})
                if meta.get("type") == "POSITIVE":
                    title = meta.get("title", "")
                    if title and title not in lookup:
                        lookup[title] = text
    return lookup


def _make_cache_key(title: str, abstract: str) -> str:
    """Compound key combining title and abstract so identity requires both to match."""
    return f"{title}||{hash(abstract)}"


def _collect_unique_papers(dataset: dict) -> dict:
    """Return a mapping {cache_key: {abstract, title}} for every unique idea."""
    unique: dict = {}
    for instance in dataset.values():
        # Pairwise: ideas is a dict of {int: str}
        ideas = instance.get("ideas")
        metadata = instance.get("metadata", {})
        if ideas is not None:
            for idx, abstract in ideas.items():
                title = metadata.get(idx, {}).get("title", "N/A")
                key = _make_cache_key(title, abstract)
                unique.setdefault(key, {"abstract": abstract, "title": title})
            continue

        # Pointwise: idea is a single str
        abstract = instance.get("idea")
        if abstract is not None:
            title = metadata.get("title", "N/A")
            key = _make_cache_key(title, abstract)
            unique.setdefault(key, {"abstract": abstract, "title": title})

    return unique


def _apply_cache(dataset: dict, cache: dict) -> dict:
    """Return a new dataset with ideas replaced by their cached manipulated text."""
    new_dataset = {}
    for inst_id, instance in dataset.items():
        inst = dict(instance)

        ideas = instance.get("ideas")
        metadata = instance.get("metadata", {})
        if ideas is not None:
            new_ideas = {}
            for idx, abstract in ideas.items():
                title = metadata.get(idx, {}).get("title", "N/A")
                key = _make_cache_key(title, abstract)
                new_ideas[idx] = cache.get(key, abstract)
            inst["ideas"] = new_ideas
        else:
            abstract = instance.get("idea")
            if abstract is not None:
                title = metadata.get("title", "N/A")
                key = _make_cache_key(title, abstract)
                inst["idea"] = cache.get(key, abstract)

        new_dataset[inst_id] = inst
    return new_dataset


def _write_manipulation_report(
    output_dir: str,
    unique_papers: dict,
    manipulation_cache: dict,
) -> None:
    """Write a human-readable Markdown report with before/after manipulation text for every abstract."""
    lines: list[str] = [
        "# Manipulation Report\n",
        f"**Abstracts processed:** {len(unique_papers)}\n",
        "---\n",
    ]

    for i, (key, paper) in enumerate(unique_papers.items(), start=1):
        title = paper.get("title", "N/A")
        before = paper.get("abstract", "")
        after = manipulation_cache.get(key, before)

        lines.append(f"## {i}. {title}\n")

        lines.append("### Before\n")
        for paragraph in before.splitlines():
            lines.append(f"> {paragraph}" if paragraph.strip() else ">")
        lines.append("\n")

        lines.append("### After\n")
        after_formatted = _SECTION_HEADER_RE.sub(r"\n\1", after).lstrip("\n")
        first_section = True
        for paragraph in after_formatted.splitlines():
            if not paragraph.strip():
                lines.append(">")
            elif _SECTION_HEADER_RE.match(paragraph.strip()):
                if not first_section:
                    lines.append(">")  # blank blockquote line = paragraph break
                lines.append(f"> {paragraph}")
                first_section = False
            else:
                lines.append(f"> {paragraph}")
                first_section = False
        lines.append("\n")

        lines.append("---\n")

    report_path = os.path.join(output_dir, "manipulation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    LOGGER.info(f"Manipulation report written to: {report_path}")


def _write_cost_report(output_dir: str, abstracts_processed: int | None = None) -> None:
    """Write cost_report.json and cost_report.md to output_dir if tracker is available."""
    if GLOBAL_COST_TRACKER is None:
        return
    import json
    cost_report = GLOBAL_COST_TRACKER.get_report()
    cost_report["test_mode"] = "apply_manipulation"
    if abstracts_processed is not None:
        cost_report["abstracts_processed"] = abstracts_processed
    cost_json_path = os.path.join(output_dir, "cost_report.json")
    with open(cost_json_path, "w") as f:
        json.dump(cost_report, f, indent=2)
    LOGGER.info(
        f"Manipulation cost: ${cost_report['total_cost_usd']:.4f} "
        f"({cost_report['total_calls']} LLM calls) — saved to {cost_json_path}"
    )
    if _write_cost_report_md is not None:
        cost_md_path = os.path.join(output_dir, "cost_report.md")
        _write_cost_report_md(cost_report, cost_md_path, title="Apply Manipulation Cost Report")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply an abstract manipulation prompt to all ideas in a dataset YAML."
    )
    parser.add_argument("--dataset", default="data/human-plus-generated/pairwise.yaml", help="Path to input YAML dataset")
    parser.add_argument(
        "--template",
        default=None,
        help="Path to Jinja2 manipulation template (default: extract_research_plan.jinja2)",
    )
    parser.add_argument(
        "--model", default="claude-opus-4-6", help="LLM model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML path (default: <dataset_dir>/manually-manipulated/benchmark_instances.yaml)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Parallel LLM workers (default: 8)"
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="Cap the number of instances processed (default: all)",
    )
    parser.add_argument(
        "--reuse-from",
        default=None,
        help="Path to an existing manipulated YAML; POSITIVE ideas matching by title are reused verbatim.",
    )
    args = parser.parse_args()

    _load_secrets()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        LOGGER.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    LOGGER.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path) as f:
        dataset: dict = yaml.safe_load(f)

    if not dataset:
        LOGGER.error("Dataset is empty.")
        sys.exit(1)

    if args.max_instances is not None and len(dataset) > args.max_instances:
        LOGGER.info(f"Truncating dataset from {len(dataset)} to {args.max_instances} instances")
        dataset = dict(list(dataset.items())[: args.max_instances])

    template = _load_manipulation_template(args.template)
    debug_log = ManipulationDebugLog()

    unique_papers = _collect_unique_papers(dataset)
    LOGGER.info(f"Found {len(unique_papers)} unique abstracts across {len(dataset)} instances")

    reuse_lookup: dict[str, str] = {}
    if args.reuse_from:
        reuse_path = Path(args.reuse_from).resolve()
        if not reuse_path.exists():
            LOGGER.error(f"--reuse-from file not found: {reuse_path}")
            sys.exit(1)
        reuse_lookup = _build_reuse_lookup(reuse_path)
        LOGGER.info(f"Loaded {len(reuse_lookup)} reusable manipulations from {reuse_path}")

    manipulation_cache: dict = {}
    field_removal_counts: dict[str, int] = {}

    # Pre-populate cache with reused manipulations; collect remainder for LLM
    papers_to_generate = {}
    reused_count = 0
    for key, paper in unique_papers.items():
        title = paper.get("title", "")
        if title in reuse_lookup:
            manipulation_cache[key] = reuse_lookup[title]
            reused_count += 1
        else:
            papers_to_generate[key] = paper

    if reused_count:
        LOGGER.info(f"Reused {reused_count}/{len(unique_papers)} manipulations from --reuse-from")
    if papers_to_generate:
        LOGGER.info(f"Generating {len(papers_to_generate)} new manipulations via LLM")

    if papers_to_generate:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    _manipulate_paper, paper, template, args.model, False, debug_log
                ): key
                for key, paper in papers_to_generate.items()
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Manipulating abstracts"
            ):
                submitted_key = futures[future]
                _, content = future.result()
                for field in _EMPTY_FIELD_CAPTURE_RE.findall(content):
                    field_removal_counts[field] = field_removal_counts.get(field, 0) + 1
                manipulation_cache[submitted_key] = _EMPTY_FIELD_RE.sub("", content)

    total = len(unique_papers)
    if field_removal_counts:
        LOGGER.info("Empty field removal stats (%d abstracts total):", total)
        for field, count in sorted(field_removal_counts.items()):
            LOGGER.info("  %-15s removed in %d/%d abstracts (%.1f%%)", field, count, total, 100 * count / total)
    else:
        LOGGER.info("Empty field removal stats: no empty fields found across %d abstracts.", total)

    new_dataset = _apply_cache(dataset, manipulation_cache)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = dataset_path.parent / "manually-manipulated" / dataset_path.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(new_dataset, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    LOGGER.info(f"Manipulated dataset written to: {output_path}")

    debug_log.write_md(str(output_path.parent))
    _write_manipulation_report(str(output_path.parent), unique_papers, manipulation_cache)
    _write_cost_report(str(output_path.parent), abstracts_processed=len(unique_papers))


if __name__ == "__main__":
    main()
