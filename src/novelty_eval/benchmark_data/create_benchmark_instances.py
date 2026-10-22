import json
import argparse
import logging
import random
import datetime
import asyncio
import concurrent.futures
import threading
from collections import Counter
from dataclasses import dataclass

import toml
import yaml
import re
from pathlib import Path
import sys
import os
from jinja2 import Template
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the parent directory to sys.path to allow importing logging_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set SECRETS environment variable if not present, as utils.py expects it
if "SECRETS" not in os.environ:
    project_root = Path(__file__).resolve().parents[3]
    secrets_file = project_root / "secrets.toml"
    if secrets_file.exists():
        os.environ["SECRETS"] = str(secrets_file)

from logging_utils import setup_logger
from utils import prompt_openai_client

try:
    from cost_tracker import GLOBAL_COST_TRACKER, cost_stage
except ImportError:
    GLOBAL_COST_TRACKER = None
    from contextlib import nullcontext as cost_stage

try:
    from novelty_eval.cost_report_writer import write_cost_report_md as _write_cost_report_md
except ImportError:
    try:
        from cost_report_writer import write_cost_report_md as _write_cost_report_md
    except ImportError:
        _write_cost_report_md = None

# Initialize logger
LOGGER = setup_logger(output_dir='..', console_level='INFO')

# Primary areas left out of the benchmark: the catch-all area has no shared topic, so
# its papers make no meaningful same-area pairs. Matched case-insensitively.
EXCLUDED_AREAS = {"other topics in machine learning (i.e., none of the above)"}


# ─────────────────────────────────────────────────────────────────────────────
# Abstract manipulation debug logging
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ManipulationFailure:
    title: str
    abstract: str         # original abstract text
    kind: str             # empty_response | json_decode_recovered | json_decode_fatal | exception
    error_msg: str = ""
    error_code: str = ""  # HTTP/API error code (e.g. 429, "invalid_prompt")
    error_type: str = ""  # exception class name
    raw_response: str = ""  # first 500 chars of LLM response if any
    used_fallback: bool = True  # False only for json_decode_recovered


class ManipulationDebugLog:
    """Thread-safe collector of abstract manipulation issues.

    Call ``write_md(output_dir)`` once all worker threads have finished to
    produce ``manipulation_debug.md`` in the run's output directory.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.failures: list[_ManipulationFailure] = []
        self.total: int = 0
        self.successes: int = 0

    def record_success(self) -> None:
        with self._lock:
            self.total += 1
            self.successes += 1

    def record_failure(self, f: _ManipulationFailure) -> None:
        with self._lock:
            self.total += 1
            self.failures.append(f)

    def write_md(self, output_dir: str) -> None:
        """Write ``manipulation_debug.md`` to *output_dir*."""
        path = os.path.join(output_dir, "manipulation_debug.md")
        hard_kinds = {"empty_response", "json_decode_fatal", "exception"}
        hard_failures = [f for f in self.failures if f.kind in hard_kinds]
        soft_failures = [f for f in self.failures if f.kind == "json_decode_recovered"]

        lines = [
            "# Abstract Manipulation Debug Log",
            "",
            f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total abstracts processed | {self.total} |",
        ]

        pct = f"{100 * self.successes / self.total:.1f}%" if self.total else "—"
        lines.append(f"| Clean successes | {self.successes} ({pct}) |")

        soft_pct = f"{100 * len(soft_failures) / self.total:.1f}%" if self.total else "—"
        lines.append(f"| Soft failures — JSON recovered via regex | {len(soft_failures)} ({soft_pct}) |")

        hard_pct = f"{100 * len(hard_failures) / self.total:.1f}%" if self.total else "—"
        lines.append(f"| Hard failures — fallback to original abstract | {len(hard_failures)} ({hard_pct}) |")

        for kind in ("empty_response", "json_decode_fatal", "exception"):
            count = sum(1 for f in hard_failures if f.kind == kind)
            if count:
                label = {
                    "empty_response": "↳ Empty LLM response",
                    "json_decode_fatal": "↳ JSON decode + regex both failed",
                    "exception": "↳ Exception during LLM call",
                }[kind]
                lines.append(f"|   {label} | {count} |")

        if not self.failures:
            lines += ["", "---", "", "*All abstracts processed successfully — no failures recorded.*", ""]
        else:
            lines += ["", "---", "", "## Failed / Degraded Abstracts", ""]
            for idx, f in enumerate(self.failures, 1):
                severity = "soft failure" if f.kind == "json_decode_recovered" else "hard failure"
                lines.append(f"### {idx} · \"{f.title}\" · `{f.kind}` _{severity}_")
                lines.append("")
                raw = f.raw_response.strip() if f.raw_response else ""
                rows = [
                    ("Fallback", "yes" if f.used_fallback else "no (partial plan extracted)"),
                ]
                if f.error_type:
                    rows.append(("Error type", f"`{f.error_type}`"))
                if f.error_code:
                    rows.append(("Error code", f"`{f.error_code}`"))
                if f.error_msg:
                    rows.append(("Error message", f"`{f.error_msg}`"))
                lines += ["| Field | Value |", "|---|---|"]
                lines += [f"| {k} | {v} |" for k, v in rows]
                lines += [""]
                if raw:
                    lines += [
                        "<details>",
                        "<summary>Raw LLM response (first 500 chars)</summary>",
                        "",
                        "```",
                        raw[:500],
                        "```",
                        "",
                        "</details>",
                        "",
                    ]
                lines += [
                    "<details>",
                    "<summary>Abstract text</summary>",
                    "",
                    f.abstract,
                    "",
                    "</details>",
                    "",
                    "---",
                    "",
                ]

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        LOGGER.info(f"Manipulation debug log written to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Secrets / dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_secrets():
    project_root = Path(__file__).resolve().parents[2]
    secrets_file = project_root / "secrets.toml"
    if secrets_file.exists():
        secrets = toml.load(secrets_file)
        if "anthropic" in secrets:
            os.environ["ANTHROPIC_API_KEY"] = secrets["anthropic"]
        if "openai_key" in secrets:
            os.environ["OPENAI_API_KEY"] = secrets["openai_key"]


def load_clean_dataset(file_path):
    LOGGER.info(f"Loading clean dataset from {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Per-paper score helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_avg_rating(paper):
    if 'average_rating' in paper and paper['average_rating'] is not None:
        try:
            return float(paper['average_rating'])
        except (ValueError, TypeError):
            pass
    return None


def get_avg_contribution(paper):
    """Return the mean contribution score across all reviews, or None."""
    if 'reviews' not in paper or not paper['reviews']:
        return None

    contributions = []
    for review in paper['reviews']:
        contribution = review.get('contribution', {})
        value = contribution.get('value') if isinstance(contribution, dict) else contribution
        if value is not None:
            try:
                if isinstance(value, str):  # e.g. "3: good"
                    value = int(value.split(':')[0].strip())
                contributions.append(float(value))
            except (ValueError, TypeError):
                pass

    return sum(contributions) / len(contributions) if contributions else None


# ─────────────────────────────────────────────────────────────────────────────
# Filtering helpers
# ─────────────────────────────────────────────────────────────────────────────

def filter_papers(papers, min_rating=None, max_rating=None):
    if min_rating is None and max_rating is None:
        return papers
    filtered = []
    for p in papers:
        rating = get_avg_rating(p)
        if rating is None:
            continue
        if min_rating is not None and rating < min_rating:
            continue
        if max_rating is not None and rating > max_rating:
            continue
        filtered.append(p)
    return filtered


def filter_papers_by_contribution(papers, min_contribution=None, max_contribution=None):
    """Filter papers based on average contribution score."""
    if min_contribution is None and max_contribution is None:
        return papers
    filtered = []
    for p in papers:
        contribution = get_avg_contribution(p)
        if contribution is None:
            continue
        if min_contribution is not None and contribution < min_contribution:
            continue
        if max_contribution is not None and contribution > max_contribution:
            continue
        filtered.append(p)
    return filtered


def filter_papers_by_decision(papers, required_decision):
    """Keep only papers whose 'decision' field matches required_decision."""
    return [p for p in papers if p.get('decision') == required_decision]


def filter_papers_strict(papers, is_positive, mode='all'):
    """
    Filter papers based on novelty signal agreement across reviewers.

    Args:
        papers:      list of paper dicts.
        is_positive: True for top/accepted papers, False for bottom/rejected papers.
        mode:        'all'      – every reviewer must agree on the signal direction.
                     'majority' – more than half of reviewers must agree.
    """
    if mode not in ('all', 'majority'):
        raise ValueError(f"Unknown strictness mode '{mode}'. Choose from: 'all', 'majority'.")

    filtered = []
    for p in papers:
        if 'reviews' not in p or not p['reviews']:
            continue
        reviews = p['reviews']
        total = len(reviews)
        matching = sum(
            1 for review in reviews
            if (is_positive and review.get('positive_novelty_signals', []) and not review.get(
                'negative_novelty_signals', []))
            or (not is_positive and review.get('negative_novelty_signals', []) and not review.get(
                'positive_novelty_signals', []))
        )
        passes = (matching == total) if mode == 'all' else (matching > total / 2)
        if passes:
            filtered.append(p)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# LLM abstract manipulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_abstract(raw: str) -> str:
    """Extract the abstract text from a raw LLM response.

    Handles the output formats produced by idea_generation templates:
      - abstract: "..."       (quoted, as instructed)
      - **abstract**: text    (markdown bold label, model drift)
      - abstract: text        (unquoted)
    Falls back to the full stripped response if no pattern matches.
    """
    patterns = [
        r'(?:\*\*)?abstract(?:\*\*)?\s*:\s*["“](.+?)["”]\s*$',  # quoted
        r'(?:\*\*)?abstract(?:\*\*)?\s*:\s*(.+)',                           # unquoted / bold
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
            text = re.sub(r'^\*+\s*', '', text)
            text = re.sub(r'\s*\*+$', '', text)
            return text
    return raw.strip()


def _load_manipulation_template(template_path: str = None):
    """Load the Jinja2 template used to prompt the LLM for abstract manipulation.

    Args:
        template_path: Optional path to a custom template file.  Defaults to
                       templates/extract_research_plan.jinja2.
    """
    if template_path is None:
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'extract_research_plan.jinja2')
    try:
        with open(template_path, 'r') as f:
            return Template(f.read())
    except FileNotFoundError:
        LOGGER.error(f"Template file not found at {template_path}")
        sys.exit(1)


def _manipulate_paper(paper, template, model_name, return_plan=False, debug_log: ManipulationDebugLog | None = None):
    """Call the LLM to manipulate the abstract of one paper.

    Returns (cache_key, idea_content) by default.
    When return_plan=True and the response is JSON-structured, returns
    (cache_key, idea_content, plan_dict) where plan_dict is the raw
    {"context": ..., "purpose": ..., ...} dict.  Used by the fine-grained
    seed extraction to avoid a second LLM call.
    """
    abstract = paper['abstract']
    title = paper.get('title', 'N/A')
    cache_key = title if title != 'N/A' else str(hash(abstract))

    prompt = template.render(abstract=abstract)
    plan_dict = {}
    error_out: list = []
    _log_recorded = False  # tracks whether we've already called record_* for this paper
    try:
        with cost_stage("instance_manipulation"):
            raw_response = prompt_openai_client(prompt, engine=model_name, error_out=error_out)
        if not raw_response:
            LOGGER.warning(f"Empty response from LLM for '{title}', using original abstract")
            idea_content = abstract
            if debug_log is not None:
                err = error_out[0] if error_out else {}
                debug_log.record_failure(_ManipulationFailure(
                    title=title,
                    abstract=abstract,
                    kind="empty_response",
                    error_msg=err.get("msg", ""),
                    error_code=str(err["code"]) if err.get("code") is not None else "",
                    error_type=err.get("type", ""),
                    raw_response="",
                    used_fallback=True,
                ))
                _log_recorded = True
        else:
            cleaned = raw_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            if cleaned.startswith("{"):
                # JSON-structured response (e.g. extract_research_plan template)
                try:
                    plan_dict = json.loads(cleaned).get("idea", {})
                except json.JSONDecodeError as e:
                    # Response was truncated mid-JSON (common with verbatim-extraction templates
                    # on long abstracts).  Recover what we can via regex so the manipulation
                    # result is not completely lost.
                    LOGGER.warning(f"Truncated JSON for '{title}' ({e}); attempting regex recovery")
                    fields = ["context", "purpose", "mechanism", "evaluation"]
                    for field in fields:
                        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
                        if m:
                            plan_dict[field] = m.group(1)
                    if not plan_dict:
                        LOGGER.error(f"Regex recovery failed for '{title}'; using original abstract")
                        idea_content = abstract
                        if debug_log is not None:
                            debug_log.record_failure(_ManipulationFailure(
                                title=title,
                                abstract=abstract,
                                kind="json_decode_fatal",
                                error_msg=str(e),
                                error_type=type(e).__name__,
                                raw_response=raw_response[:500],
                                used_fallback=True,
                            ))
                        if return_plan:
                            return cache_key, idea_content, plan_dict
                        return cache_key, idea_content
                    # Partial recovery succeeded — record as soft failure
                    if debug_log is not None:
                        debug_log.record_failure(_ManipulationFailure(
                            title=title,
                            abstract=abstract,
                            kind="json_decode_recovered",
                            error_msg=str(e),
                            error_type=type(e).__name__,
                            raw_response=raw_response[:500],
                            used_fallback=False,
                        ))
                        _log_recorded = True
                idea_content = "".join(f'**{k}**: {v}\n' for k, v in plan_dict.items())
            else:
                # Plain-text response (e.g. remove_eval_data template)
                idea_content = cleaned
    except Exception as e:
        LOGGER.error(f"Error manipulating abstract for '{title}': {e}")
        idea_content = abstract  # fallback
        if debug_log is not None:
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            debug_log.record_failure(_ManipulationFailure(
                title=title,
                abstract=abstract,
                kind="exception",
                error_msg=str(e),
                error_code=str(code) if code is not None else "",
                error_type=type(e).__name__,
                raw_response="",
                used_fallback=True,
            ))
            _log_recorded = True

    if debug_log is not None and not _log_recorded:
        debug_log.record_success()

    if return_plan:
        return cache_key, idea_content, plan_dict
    return cache_key, idea_content


async def _generate_llm_negatives(
        areas_with_counts: dict,
        model_name: str,
        prompt_template_path: str,
        max_parallel: int = 10,
        reasoning_effort: str | None = None,
) -> dict:
    """
    Generate synthetic negative papers for each area by prompting an LLM
    directly with a configurable Jinja2 template — no multi-agent pipeline.

    Each call produces exactly one idea per slot.  Ideas are returned as plain
    abstract text.

    Args:
        areas_with_counts: Dict mapping area name → number of negatives needed.
        model_name: The LLM to use for generation (e.g. "claude-sonnet-4-6").
        prompt_template_path: Path to the Jinja2 template file.
        max_parallel: Max concurrent LLM calls.

    Returns:
        Dict mapping area → list of synthetic paper dicts.
    """
    with open(prompt_template_path, 'r') as f:
        template = Template(f.read())

    semaphore = asyncio.Semaphore(max_parallel)
    area_papers: dict = {area: [] for area in areas_with_counts}

    async def _generate_one(area: str, sample_idx: int):
        prompt = template.render(area=area)
        async with semaphore:
            LOGGER.info(f"[llm_negatives][{area}][sample {sample_idx}] Prompting {model_name}")
            loop = asyncio.get_event_loop()
            with cost_stage("llm_negative_generation"):
                response = await loop.run_in_executor(
                    None,
                    lambda: prompt_openai_client(
                        prompt, engine=model_name,
                        reasoning={"effort": reasoning_effort} if reasoning_effort else None,
                    ),
                )
        abstract_text = _parse_llm_abstract(response) if response else ""
        if not abstract_text:
            LOGGER.warning(f"[llm_negatives][{area}][sample {sample_idx}] Empty response — skipping")
            return area, None
        paper = {
            "abstract": abstract_text,
            "title": f"LLM-{area}-{sample_idx}",
            "reviews": [],
            "_generated": True,
            "_llm_generated": True,
        }
        return area, paper

    tasks = [
        _generate_one(area, idx)
        for area, count in areas_with_counts.items()
        for idx in range(count)
    ]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating LLM negatives"):
        area, paper = await fut
        if paper is not None:
            area_papers[area].append(paper)
            LOGGER.info(f"[llm_negatives][{area}] collected {len(area_papers[area])} / "
                        f"{areas_with_counts[area]} negatives")

    return area_papers


# ─────────────────────────────────────────────────────────────────────────────
# Per-paper content builders
# ─────────────────────────────────────────────────────────────────────────────

def _collect_novelty_signals(paper):
    """Return (pos_signals, neg_signals, similar_papers) aggregated across all reviews."""
    pos_signals, neg_signals, similar_papers = [], [], []
    for review in paper.get('reviews', []):
        pos_signals.extend(review.get('positive_novelty_signals', []))
        neg_signals.extend(review.get('negative_novelty_signals', []))
        similar_papers.extend(review.get('similar_papers_mentioned', []))
    return pos_signals, neg_signals, similar_papers


def _build_paper_summary_lines(idx, paper_type, title, rating_str, contribution_str, area,
                               idea_label, idea_text, pos_signals, neg_signals, similar_papers):
    """Build the human-readable text block for one paper within an instance summary."""

    def signal_lines(signals):
        return [f"* {s}" for s in signals] if signals else ["None"]

    return [
        f"Paper {idx} ({paper_type})",
        f"Title: {title}",
        f"Average Rating: {rating_str}",
        f"Average Contribution: {contribution_str}",
        f"Area: {area}",
        "-" * 40,
        f"{idea_label}:",
        idea_text,
        "-" * 40,
        "Positive Novelty Signals:",
        *signal_lines(pos_signals),
        "-" * 40,
        "Negative Novelty Signals:",
        *signal_lines(neg_signals),
        "-" * 40,
        "Similar Papers Mentioned:",
        *signal_lines(similar_papers),
        "\n" + "-" * 80 + "\n",
    ]


def _build_paper_md_cell(paper_type, title, rating_str, contribution_str, area,
                         idea_text, pos_signals, neg_signals, similar_papers):
    """Build the markdown table cell string for one paper."""

    def md_list(items):
        safe = [s.replace("|", "\\|").replace("\n", " ") for s in items]
        return "<ul>" + "".join(f"<li>{s}</li>" for s in safe) + "</ul>"

    # Preserve real paragraph breaks (\n\n) as <br><br>;
    # collapse single newlines (PDF line-wrap artifacts) into spaces.
    safe_idea = (idea_text
                 .replace("|", "\\|")
                 .replace("\n\n", "<br><br>")
                 .replace("\n", " "))
    parts = [
        f"**Type**: {paper_type}",
        f"**Title**: {title}",
        f"**Rating**: {rating_str}",
        f"**Contribution**: {contribution_str}",
        f"**Area**: {area}",
        f"**Idea**: {safe_idea}",
    ]
    if pos_signals:
        parts.append(f"**Positive Signals**:{md_list(pos_signals)}")
    if neg_signals:
        parts.append(f"**Negative Signals**:{md_list(neg_signals)}")
    if similar_papers:
        parts.append(f"**Similar Papers**:{md_list(similar_papers)}")
    return "<br><br>".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-instance assembly
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_instance(instance_id, area, all_selected_papers, selected_top,
                       manipulated_cache, model_name, use_abstract_only, generate_both_versions,
                       manipulate_generated_negatives=False):
    """Build all data structures for a single test instance.

    Returns a dict with keys: raw_ideas, [manipulated_ideas], expected_winners,
    paper_metadata, raw_summary_lines, [manipulated_summary_lines],
    raw_md_row_parts, [manipulated_md_row_parts].
    """
    raw_ideas = {}
    manipulated_ideas = {} if generate_both_versions and model_name else None
    expected_winners = []
    paper_metadata = {}

    raw_summary_lines = [f"Instance ID: {instance_id}", f"Context: {area}", "=" * 80]
    manipulated_summary_lines = [f"Instance ID: {instance_id}", f"Context: {area}", "=" * 80] \
        if generate_both_versions and model_name else None

    raw_md_row_parts = [str(instance_id), area.replace("|", "\\|")]
    manipulated_md_row_parts = [str(instance_id), area.replace("|", "\\|")] \
        if generate_both_versions and model_name else None

    for idx, paper in enumerate(all_selected_papers):
        abstract = paper['abstract']
        title = paper.get('title', 'N/A')
        is_generated = paper.get('_generated', False)

        # Generated negatives are used as-is by default
        if is_generated and not manipulate_generated_negatives:
            manipulated_content = abstract
        else:
            cache_key = title if title != 'N/A' else str(hash(abstract))
            manipulated_content = manipulated_cache.get(cache_key) or abstract

        # Decide idea content per version
        if generate_both_versions and model_name:
            raw_ideas[idx] = abstract
            manipulated_ideas[idx] = manipulated_content
        elif use_abstract_only or not model_name:
            raw_ideas[idx] = abstract
        else:
            if is_generated and not manipulate_generated_negatives:
                raw_ideas[idx] = abstract
            else:
                raw_ideas[idx] = manipulated_content

        is_winner = paper in selected_top
        if is_winner:
            expected_winners.append(idx)

        pos_signals, neg_signals, similar_papers = _collect_novelty_signals(paper)
        avg_rating = get_avg_rating(paper)
        avg_contribution = get_avg_contribution(paper)
        rating_str = f"{avg_rating:.2f}" if avg_rating is not None else "N/A"
        contribution_str = f"{avg_contribution:.2f}" if avg_contribution is not None else "N/A"
        paper_type = "POSITIVE" if is_winner else "NEGATIVE"

        # Raw summary + MD cell
        # When single-version mode with manipulation active, show the manipulated
        # content in the report so it matches what's stored in the YAML.
        single_version_manipulated = (not generate_both_versions and model_name
                                      and not use_abstract_only and not is_generated)
        display_content = manipulated_content if single_version_manipulated else abstract
        raw_summary_lines.extend(_build_paper_summary_lines(
            idx, paper_type, title, rating_str, contribution_str, area,
            "Abstract", display_content, pos_signals, neg_signals, similar_papers))
        raw_md_row_parts.append(_build_paper_md_cell(
            paper_type, title, rating_str, contribution_str, area,
            display_content, pos_signals, neg_signals, similar_papers))

        # Manipulated version summary + MD cell (if applicable)
        if generate_both_versions and model_name:
            manipulated_summary_lines.extend(_build_paper_summary_lines(
                idx, paper_type, title, rating_str, contribution_str, area,
                "Manipulated Abstract", manipulated_content, pos_signals, neg_signals, similar_papers))
            manipulated_md_row_parts.append(_build_paper_md_cell(
                paper_type, title, rating_str, contribution_str, area,
                manipulated_content, pos_signals, neg_signals, similar_papers))

        paper_metadata[idx] = {
            'title': title,
            'area': area,
            'rating': rating_str,
            'contribution': contribution_str,
            'type': paper_type,
            'positive_signals': pos_signals,
            'negative_signals': neg_signals,
            'similar_papers_mentioned': similar_papers,
        }
        if paper.get("turn_number") is not None:
            paper_metadata[idx]["turn_number"] = paper["turn_number"]

    expected_winners.sort()
    return dict(
        raw_ideas=raw_ideas, manipulated_ideas=manipulated_ideas,
        expected_winners=expected_winners, paper_metadata=paper_metadata,
        raw_summary_lines=raw_summary_lines, manipulated_summary_lines=manipulated_summary_lines,
        raw_md_row_parts=raw_md_row_parts, manipulated_md_row_parts=manipulated_md_row_parts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main instance-creation entry point
# ─────────────────────────────────────────────────────────────────────────────

def create_test_instances(clean_data_list, num_top_papers, num_bottom_papers,
                          model_name=None, use_abstract_only=False, mix_topics=False,
                          max_workers=1, strictness=None, generate_both_versions=False,
                          manipulation_prompt=None, generate_negatives=False,
                          generated_negatives_map=None, max_instances=None,
                          pre_built_manipulation_cache=None,
                          debug_log: ManipulationDebugLog | None = None,
                          manipulate_generated_negatives=False):
    """
    Create test instances from the clean data list.

    If generate_both_versions is True (and model_name is set), produces two
    correlated versions: raw abstracts and manipulated abstracts with identical
    paper ordering per instance.

    If generate_negatives is True, ``generated_negatives_map`` must be a dict
    mapping area → list of synthetic paper dicts produced by the pipeline.
    These replace the ICLR bottom papers for each area.

    If pre_built_manipulation_cache is provided (dict mapping cache_key →
    idea_content), those entries are used directly and the corresponding papers
    are skipped during the manipulation phase — avoiding a second LLM call
    (used by the fine-grained seed ablation).

    Args:
        manipulate_generated_negatives: If True, apply manipulation to synthetic negatives.

    Returns:
        Single mode:  (test_instances, summaries, md_report)
        Dual mode:    (raw_instances, raw_summaries, raw_md_report,
                       manipulated_instances, manipulated_summaries, manipulated_md_report)
    """
    if generated_negatives_map is None:
        generated_negatives_map = {}

    if mix_topics:
        all_top, all_bottom = [], []
        for clean_data in clean_data_list:
            for data in clean_data.values():
                all_top.extend(data.get('top_papers', []))
                all_bottom.extend(data.get('bottom_papers', []))
        if generate_negatives:
            all_generated = []
            for papers in generated_negatives_map.values():
                all_generated.extend(papers)
            clean_data_list = [{'Mixed Topics': {'top_papers': all_top, 'bottom_papers': all_generated}}]
        else:
            clean_data_list = [{'Mixed Topics': {'top_papers': all_top, 'bottom_papers': all_bottom}}]

    # ── Pass 1: select papers for each instance ──────────────────────────
    # Decide which papers go into each instance *before* manipulation so we
    # only pay for LLM calls on papers that are actually used.
    instance_selections = []  # list of (area, all_selected, selected_top)

    for clean_data in clean_data_list:
        for area, data in clean_data.items():
            top_papers = data.get('top_papers', [])

            if generate_negatives:
                bottom_papers = generated_negatives_map.get(area, [])
                if not bottom_papers:
                    LOGGER.warning(f"Skipping area '{area}': no generated negatives available")
                    continue
            else:
                bottom_papers = data.get('bottom_papers', [])

            if len(top_papers) < num_top_papers or len(bottom_papers) < num_bottom_papers:
                LOGGER.warning(f"Skipping area '{area}': not enough papers "
                               f"(top={len(top_papers)}, bottom={len(bottom_papers)})")
                continue

            # Fine-grained pairing: negatives tagged with _positive_title
            if bottom_papers and bottom_papers[0].get("_positive_title"):
                negatives_by_pos: dict = {}
                for n in bottom_papers:
                    negatives_by_pos.setdefault(n["_positive_title"], []).append(n)

                for top_paper in top_papers:
                    pos_negs = negatives_by_pos.get(top_paper.get("title", ""), [])
                    if len(pos_negs) < num_bottom_papers:
                        LOGGER.warning(
                            f"Skipping top paper '{top_paper.get('title')}' in area '{area}': "
                            f"only {len(pos_negs)} matched negatives (need {num_bottom_papers})")
                        continue
                    all_selected = [top_paper] + pos_negs[:num_bottom_papers]
                    random.shuffle(all_selected)
                    instance_selections.append((area, all_selected, [top_paper]))
            else:
                random.shuffle(top_papers)
                random.shuffle(bottom_papers)
                current_top_idx = 0
                current_bottom_idx = 0

                while current_top_idx + num_top_papers <= len(top_papers):
                    if current_bottom_idx + num_bottom_papers > len(bottom_papers):
                        random.shuffle(bottom_papers)
                        current_bottom_idx = 0

                    selected_top = top_papers[current_top_idx: current_top_idx + num_top_papers]
                    selected_bottom = bottom_papers[current_bottom_idx: current_bottom_idx + num_bottom_papers]
                    current_top_idx += num_top_papers
                    current_bottom_idx += num_bottom_papers

                    all_selected = selected_top + selected_bottom
                    random.shuffle(all_selected)
                    instance_selections.append((area, all_selected, selected_top))

    if max_instances is not None and len(instance_selections) > max_instances:
        LOGGER.info(f"Truncating {len(instance_selections)} instances to max_instances={max_instances}")
        instance_selections = instance_selections[:max_instances]

    # ── Build manipulation cache only for papers that will actually be used ─
    # Seed from pre-built cache to avoid re-manipulating already-processed papers.
    manipulated_cache = dict(pre_built_manipulation_cache) if pre_built_manipulation_cache else {}
    if model_name and not use_abstract_only:
        # Collect unique non-generated papers across all selected instances
        papers_to_manipulate = {}
        for _area, all_selected, _top in instance_selections:
            for p in all_selected:
                if p.get('_generated') and not manipulate_generated_negatives:
                    continue
                title = p.get('title', 'N/A')
                key = title if title != 'N/A' else str(hash(p['abstract']))
                papers_to_manipulate.setdefault(key, p)

        # Skip papers already in the pre-built cache
        papers_to_manipulate = {k: p for k, p in papers_to_manipulate.items()
                                 if k not in manipulated_cache}

        if papers_to_manipulate:
            template = _load_manipulation_template(manipulation_prompt)
            LOGGER.info(f"Found {len(papers_to_manipulate)} unique papers to manipulate.")
            LOGGER.info(f"Starting parallel abstract manipulation with {max_workers} workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_manipulate_paper, p, template, model_name,
                                           False, debug_log): k
                           for k, p in papers_to_manipulate.items()}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Manipulating abstracts"):
                    key, content = future.result()
                    manipulated_cache[key] = content

    # ── Pass 2: assemble instances ───────────────────────────────────────
    total_papers = num_top_papers + num_bottom_papers
    md_header = "| Instance ID | Context | " + " | ".join(f"Paper {i}" for i in range(total_papers)) + " |"
    md_separator = "|---|---| " + " | ".join("---" for _ in range(total_papers)) + " |"
    md_lines = [md_header, md_separator]
    manipulated_md_lines = [md_header, md_separator] if generate_both_versions and model_name else None

    raw_instances, summaries = {}, []
    manipulated_instances, manipulated_summaries = {}, []

    for instance_id, (area, all_selected, selected_top) in enumerate(
            tqdm(instance_selections, desc="Creating instances")):

        inst = _assemble_instance(
            instance_id, area, all_selected, selected_top,
            manipulated_cache, model_name, use_abstract_only, generate_both_versions,
            manipulate_generated_negatives=manipulate_generated_negatives)

        raw_instances[instance_id] = {
            'context': area,
            'expected_winners': inst['expected_winners'],
            'ideas': inst['raw_ideas'],
            'metadata': inst['paper_metadata'],
        }
        summaries.append("\n".join(inst['raw_summary_lines']))
        md_lines.append("| " + " | ".join(inst['raw_md_row_parts']) + " |")

        if generate_both_versions and model_name:
            manipulated_instances[instance_id] = {
                'context': area,
                'expected_winners': inst['expected_winners'],
                'ideas': inst['manipulated_ideas'],
                'metadata': inst['paper_metadata'],
            }
            manipulated_summaries.append("\n".join(inst['manipulated_summary_lines']))
            manipulated_md_lines.append("| " + " | ".join(inst['manipulated_md_row_parts']) + " |")


    md_report = "\n".join(md_lines)
    if generate_both_versions and model_name:
        return raw_instances, summaries, md_report, manipulated_instances, manipulated_summaries, "\n".join(manipulated_md_lines)
    return raw_instances, summaries, md_report


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise instance creation
# ─────────────────────────────────────────────────────────────────────────────

def create_pointwise_instances(clean_data_list, model_name=None, use_abstract_only=False,
                               max_workers=1, manipulation_prompt=None,
                               pointwise_balance=False, pointwise_shuffle=True,
                               max_instances=None, generate_negatives=False,
                               generated_negatives_map=None,
                               pre_built_manipulation_cache=None,
                               debug_log: ManipulationDebugLog | None = None,
                               manipulate_generated_negatives=False):
    """Create pointwise test instances where each instance contains exactly one idea.

    Each instance has the format::

        {
          'context': <area>,
          'idea': <abstract or manipulated text>,
          'label': 'POSITIVE' or 'NEGATIVE',
          'metadata': {
            'title': ..., 'area': ..., 'rating': ..., 'contribution': ...,
            'positive_signals': [...], 'negative_signals': [...],
            'similar_papers_mentioned': [...],
          }
        }

    Args:
        clean_data_list: List of area-keyed dicts with 'top_papers' and 'bottom_papers'.
        model_name: LLM used to manipulate abstracts. None → use raw abstracts.
        use_abstract_only: If True, always use raw abstract even when model_name is set.
        max_workers: Thread pool size for parallel LLM manipulation calls.
        manipulation_prompt: Path to Jinja2 template for manipulation. None → default template.
        pointwise_balance: Truncate the larger set so positives and negatives are equal in count.
        pointwise_shuffle: Shuffle the final instance list before returning (default True).
        max_instances: Cap the total number of instances returned.
        generate_negatives: If True, use generated_negatives_map instead of bottom_papers.
        generated_negatives_map: Dict mapping area → list of synthetic paper dicts.
        manipulate_generated_negatives: If True, apply manipulation to synthetic negatives.

    Returns:
        Dict mapping integer index → instance dict.
    """
    if generated_negatives_map is None:
        generated_negatives_map = {}

    # ── Collect all papers tagged with area and label ─────────────────────────
    positives = []  # list of (area, paper)
    negatives = []

    for clean_data in clean_data_list:
        for area, data in clean_data.items():
            for paper in data.get('top_papers', []):
                positives.append((area, paper))
            if generate_negatives:
                area_negatives = generated_negatives_map.get(area, [])
                if not area_negatives:
                    LOGGER.warning(f"Pointwise: no generated negatives for area '{area}', skipping negatives")
            else:
                area_negatives = data.get('bottom_papers', [])
            for paper in area_negatives:
                negatives.append((area, paper))

    LOGGER.info(f"Pointwise mode: {len(positives)} positive papers, {len(negatives)} negative papers")

    if pointwise_balance:
        target = min(len(positives), len(negatives))
        random.shuffle(positives)
        random.shuffle(negatives)
        positives = positives[:target]
        negatives = negatives[:target]
        LOGGER.info(f"Pointwise balance: truncated to {target} of each label")

    all_tagged = [(area, paper, 'POSITIVE') for area, paper in positives] + \
                 [(area, paper, 'NEGATIVE') for area, paper in negatives]

    # ── Build manipulation cache ──────────────────────────────────────────────
    manipulated_cache = dict(pre_built_manipulation_cache) if pre_built_manipulation_cache else {}
    if model_name and not use_abstract_only:
        papers_to_manipulate = {}
        for area, paper, _label in all_tagged:
            if paper.get('_generated') and not manipulate_generated_negatives:
                continue
            title = paper.get('title', 'N/A')
            key = title if title != 'N/A' else str(hash(paper['abstract']))
            papers_to_manipulate.setdefault(key, paper)

        # Skip papers already in the pre-built cache
        papers_to_manipulate = {k: p for k, p in papers_to_manipulate.items()
                                 if k not in manipulated_cache}

        if papers_to_manipulate:
            template = _load_manipulation_template(manipulation_prompt)
            LOGGER.info(f"Pointwise: manipulating {len(papers_to_manipulate)} unique papers "
                        f"with {max_workers} workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_manipulate_paper, p, template, model_name,
                                           False, debug_log): k
                           for k, p in papers_to_manipulate.items()}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Manipulating abstracts (pointwise)"):
                    key, content = future.result()
                    manipulated_cache[key] = content

    # ── Assemble instances ────────────────────────────────────────────────────
    instances_list = []
    for area, paper, label in all_tagged:
        abstract = paper['abstract']
        title = paper.get('title', 'N/A')
        is_generated = paper.get('_generated', False)

        if model_name and not use_abstract_only and (not is_generated or manipulate_generated_negatives):
            key = title if title != 'N/A' else str(hash(abstract))
            idea_text = manipulated_cache.get(key) or abstract
        else:
            idea_text = abstract

        pos_signals, neg_signals, similar_papers = _collect_novelty_signals(paper)
        avg_rating = get_avg_rating(paper)
        avg_contribution = get_avg_contribution(paper)
        rating_str = f"{avg_rating:.2f}" if avg_rating is not None else "N/A"
        contribution_str = f"{avg_contribution:.2f}" if avg_contribution is not None else "N/A"

        meta = {
            'title': title,
            'area': area,
            'rating': rating_str,
            'contribution': contribution_str,
            'positive_signals': pos_signals,
            'negative_signals': neg_signals,
            'similar_papers_mentioned': similar_papers,
        }
        if paper.get("turn_number") is not None:
            meta["turn_number"] = paper["turn_number"]
        instances_list.append({
            'context': area,
            'idea': idea_text,
            'label': label,
            'metadata': meta,
        })

    if pointwise_shuffle:
        random.shuffle(instances_list)

    if max_instances is not None and len(instances_list) > max_instances:
        LOGGER.info(f"Truncating {len(instances_list)} pointwise instances to max_instances={max_instances}")
        instances_list = instances_list[:max_instances]

    result = {idx: inst for idx, inst in enumerate(instances_list)}
    pos_count = sum(1 for inst in result.values() if inst['label'] == 'POSITIVE')
    neg_count = sum(1 for inst in result.values() if inst['label'] == 'NEGATIVE')
    LOGGER.info(f"Pointwise instances created: {len(result)} total "
                f"({pos_count} POSITIVE, {neg_count} NEGATIVE)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_test_instances(test_instances, output_file):
    LOGGER.info(f"Saving {len(test_instances)} test instances to {output_file}")
    with open(output_file, 'w') as f:
        yaml.dump(test_instances, f, sort_keys=True)


def save_readable_summaries(summaries, output_file):
    LOGGER.info(f"Saving readable summaries to {output_file}")
    with open(output_file, 'w') as f:
        f.write("\n\n".join(summaries))


def save_md_report(md_content, output_file, batch_size=None):
    if not batch_size:
        LOGGER.info(f"Saving markdown report to {output_file}")
        with open(output_file, 'w') as f:
            f.write(md_content)
        return

    # Split only at newlines that begin a new table row (i.e. followed by '|'),
    # so cells containing literal '\n' characters don't break row boundaries.
    all_rows = re.split(r'\n(?=\|)', md_content)
    header, data_rows = all_rows[:2], all_rows[2:]

    if not data_rows:
        LOGGER.info(f"Saving markdown report to {output_file}")
        with open(output_file, 'w') as f:
            f.write(md_content)
        return

    total = (len(data_rows) + batch_size - 1) // batch_size
    stem = os.path.splitext(os.path.basename(output_file))[0]
    batch_dir = os.path.join(os.path.dirname(output_file), stem)
    os.makedirs(batch_dir, exist_ok=True)
    for i, start in enumerate(range(0, len(data_rows), batch_size)):
        batch_file = os.path.join(batch_dir, f"{stem}_{i + 1:03d}.md")
        content = '\n'.join(header + data_rows[start:start + batch_size])
        LOGGER.info(f"Saving markdown report batch {i + 1}/{total} to {batch_file}")
        with open(batch_file, 'w') as f:
            f.write(content)


def save_data_summary(summary_content, output_file):
    LOGGER.info(f"Saving data summary to {output_file}")
    with open(output_file, 'w') as f:
        f.write(summary_content)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers for generate_data_summary
# ─────────────────────────────────────────────────────────────────────────────

def _collect_paper_stats(test_instances):
    """Return title lists, rating/contribution lists, and signal counts."""
    all_titles, pos_titles, neg_titles = [], [], []
    all_ratings, pos_ratings, neg_ratings = [], [], []
    all_contributions, pos_contributions, neg_contributions = [], [], []
    papers_with_pos_signals = papers_with_neg_signals = 0
    total_pos_signals = total_neg_signals = 0

    for inst in test_instances.values():
        for meta in inst.get('metadata', {}).values():
            title = meta.get('title', 'N/A')
            ptype = meta.get('type')
            if title != 'N/A':
                all_titles.append(title)
                (pos_titles if ptype == 'POSITIVE' else neg_titles).append(title)

            for val_str, all_list, typed_list in [
                (meta.get('rating', 'N/A'), all_ratings, pos_ratings if ptype == 'POSITIVE' else neg_ratings),
                (meta.get('contribution', 'N/A'), all_contributions,
                 pos_contributions if ptype == 'POSITIVE' else neg_contributions),
            ]:
                if val_str != 'N/A':
                    try:
                        v = float(val_str)
                        all_list.append(v)
                        typed_list.append(v)
                    except ValueError:
                        pass

            pos_sigs = meta.get('positive_signals', [])
            neg_sigs = meta.get('negative_signals', [])
            if pos_sigs:
                papers_with_pos_signals += 1
                total_pos_signals += len(pos_sigs)
            if neg_sigs:
                papers_with_neg_signals += 1
                total_neg_signals += len(neg_sigs)

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else None

    return dict(
        all_titles=all_titles, pos_titles=pos_titles, neg_titles=neg_titles,
        avg_rating=safe_avg(all_ratings),
        avg_pos_rating=safe_avg(pos_ratings),
        avg_neg_rating=safe_avg(neg_ratings),
        avg_contribution=safe_avg(all_contributions),
        avg_pos_contribution=safe_avg(pos_contributions),
        avg_neg_contribution=safe_avg(neg_contributions),
        papers_with_pos_signals=papers_with_pos_signals,
        papers_with_neg_signals=papers_with_neg_signals,
        total_pos_signals=total_pos_signals,
        total_neg_signals=total_neg_signals,
    )


def _collect_reviewer_stats(test_instances, clean_data_list):
    """Count reviewer-level novelty-signal statistics (unique papers only)."""
    title_to_paper = {}
    for clean_data in clean_data_list:
        for data in clean_data.values():
            for paper in data.get('top_papers', []):
                t = paper.get('title', 'N/A')
                if t != 'N/A':
                    title_to_paper[t] = ('POSITIVE', paper)
            for paper in data.get('bottom_papers', []):
                t = paper.get('title', 'N/A')
                if t != 'N/A':
                    title_to_paper[t] = ('NEGATIVE', paper)

    totals = dict(total=0, pos=0, neg=0, both=0, none=0,
                  pos_paper_total=0, pos_paper_pos=0, pos_paper_neg=0,
                  neg_paper_total=0, neg_paper_pos=0, neg_paper_neg=0)
    counted = set()

    for inst in test_instances.values():
        for meta in inst.get('metadata', {}).values():
            title = meta.get('title', 'N/A')
            if title == 'N/A' or title in counted or title not in title_to_paper:
                continue
            counted.add(title)
            paper_type, paper = title_to_paper[title]
            for review in paper.get('reviews', []):
                has_pos = bool(review.get('positive_novelty_signals'))
                has_neg = bool(review.get('negative_novelty_signals'))
                totals['total'] += 1
                if has_pos: totals['pos'] += 1
                if has_neg: totals['neg'] += 1
                if has_pos and has_neg: totals['both'] += 1
                if not has_pos and not has_neg: totals['none'] += 1
                if paper_type == 'POSITIVE':
                    totals['pos_paper_total'] += 1
                    if has_pos: totals['pos_paper_pos'] += 1
                    if has_neg: totals['pos_paper_neg'] += 1
                else:
                    totals['neg_paper_total'] += 1
                    if has_pos: totals['neg_paper_pos'] += 1
                    if has_neg: totals['neg_paper_neg'] += 1
    return totals


def _collect_source_stats(clean_data_list):
    """Count available papers per area in the (already filtered) source data."""
    source_stats = {}
    for clean_data in clean_data_list:
        for area, data in clean_data.items():
            s = source_stats.setdefault(area, {'top': 0, 'bottom': 0})
            s['top'] += len(data.get('top_papers', []))
            s['bottom'] += len(data.get('bottom_papers', []))
    return source_stats


# ─────────────────────────────────────────────────────────────────────────────
# Data summary markdown generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_data_summary(args, test_instances, output_dir, clean_data_list):
    """Generate a comprehensive data summary markdown file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_both = getattr(args, 'generate_both_versions', False)

    # ── Collect stats ──────────────────────────────────────────────────────
    paper_stats = _collect_paper_stats(test_instances)
    reviewer_stats = _collect_reviewer_stats(test_instances, clean_data_list)
    source_stats = _collect_source_stats(clean_data_list)

    total_instances = len(test_instances)
    total_ideas = sum(len(inst['ideas']) for inst in test_instances.values())
    unique_papers = set(paper_stats['all_titles'])
    unique_pos = set(paper_stats['pos_titles'])
    unique_neg = set(paper_stats['neg_titles'])
    duplicated = {t: c for t, c in Counter(paper_stats['all_titles']).items() if c > 1}
    context_counts = Counter(inst['context'] for inst in test_instances.values())

    def fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"

    def pct(n, d):
        return f"{100 * n / d:.1f}%" if d > 0 else "N/A"

    rs = reviewer_stats  # shorthand

    # ── Build markdown ─────────────────────────────────────────────────────
    repro_num = "5" if is_both else "4"
    stats_num = "4" if is_both else "3"
    files_num = "3" if is_both else "2"

    md = []

    # Header
    md += [
        "# Data Creation Summary", "",
        f"**Generated:** {timestamp}",
        f"**Output Directory:** `{os.path.abspath(output_dir)}`", "",
    ]
    if is_both:
        md += [
            "> 📊 **Dual Version Mode**: This dataset contains two correlated versions.",
            "> Each instance ID corresponds to the exact same papers in both versions.", "",
        ]

    # Table of contents
    md += [
        "---", "", "## Table of Contents", "",
        "1. [Parameters Used](#1-parameters-used)",
        "   - [Input Configuration](#input-configuration)",
        "   - [Filtering Parameters](#filtering-parameters)",
    ]
    if is_both:
        md += [
            "2. [Dataset Versions](#2-dataset-versions)",
            f"{files_num}. [Output Files](#{files_num}-output-files)",
            f"{stats_num}. [Dataset Statistics](#{stats_num}-dataset-statistics)",
        ]
    else:
        md += [
            f"{files_num}. [Output Files](#{files_num}-output-files)",
            f"{stats_num}. [Dataset Statistics](#{stats_num}-dataset-statistics)",
        ]
    md += [
        "   - [Overview](#overview)",
        "   - [Rating Statistics](#rating-statistics)",
        "   - [Contribution Statistics](#contribution-statistics)",
        "   - [Novelty Signals](#novelty-signals)",
        "   - [Reviewer Signal Distribution](#reviewer-signal-distribution-unique-papers)",
        "   - [Distribution by Topic/Context](#distribution-by-topiccontext)",
        "   - [Source Data](#source-data-after-filtering)",
        f"{repro_num}. [Reproducibility Information](#{repro_num}-reproducibility-information)",
        "", "---", "",
    ]

    # Section 1 – Parameters
    md += [
        "## 1. Parameters Used", "",
        "### Input Configuration", "",
        "| Parameter | Value |", "|-----------|-------|",
        f"| ICLR Data | {', '.join(f'`{p}`' for p in args.iclr_data)} |",
        f"| Number of Top Papers per Instance | {args.num_top_papers} |",
        f"| Number of Bottom Papers per Instance | {args.num_bottom_papers} |",
        f"| Model Name | {args.model_name or 'None (using original abstracts)'} |",
        f"| Use Abstract Only | {args.use_abstract_only} |",
        f"| Manipulate Generated Negatives | {getattr(args, 'manipulate_generated_negatives', False)} |",
        f"| Generate Both Versions | {is_both} |",
        f"| Mix Topics | {args.mix_topics} |",
        f"| Strictness Mode | {args.strictness or 'None'} |",
        f"| Max Workers | {args.max_workers} |", "",
        "### Filtering Parameters", "",
        "| Filter Type | Min | Max |", "|-------------|-----|-----|",
        f"| Positive Paper Rating | {args.min_pos_rating or 'None'} | {args.max_pos_rating or 'None'} |",
        f"| Negative Paper Rating | {args.min_neg_rating or 'None'} | {args.max_neg_rating or 'None'} |",
        f"| Positive Paper Contribution | {args.min_pos_contribution or 'None'} | {args.max_pos_contribution or 'None'} |",
        f"| Negative Paper Contribution | {args.min_neg_contribution or 'None'} | {args.max_neg_contribution or 'None'} |",
        "", "---", "",
    ]

    # Section 2 – Dual-version details (optional)
    if is_both:
        md += [
            "## 2. Dataset Versions", "",
            "Both versions contain the same papers in the same order per instance.", "",
            "### Raw Abstract Version", "",
            "| Property | Description |", "|----------|-------------|",
            "| **Directory** | `raw_abstract/` |",
            "| **Content** | Original paper abstracts as-is |",
            "| **Use Case** | Testing evaluators on full, unprocessed academic text |", "",
            "### Manipulated Abstract Version", "",
            "| Property | Description |", "|----------|-------------|",
            "| **Directory** | `manipulated/` |",
            f"| **Content** | LLM-manipulated abstracts (using `{args.model_name}`) |",
            "| **Use Case** | Testing evaluators on structured, condensed idea representations |",
            "", "---", "",
        ]

    # Output files section
    if is_both:
        md += [
            f"## {files_num}. Output Files", "",
            "```",
            f"{os.path.basename(output_dir)}/",
            "├── data_summary.md",
            "├── raw_abstract/",
            "│   ├── benchmark_instances.yaml",
            "│   ├── benchmark_instances_readable.txt",
            "│   └── benchmark_instances_report.md",
            "└── manipulated/",
            "    ├── benchmark_instances.yaml",
            "    ├── benchmark_instances_readable.txt",
            "    └── benchmark_instances_report.md",
            "```", "", "---", "",
        ]
    else:
        md += [
            f"## {files_num}. Output Files", "",
            "| File | Description | Path |", "|------|-------------|------|",
            f"| Test Instances (YAML) | Main data file | `{os.path.join(output_dir, 'benchmark_instances.yaml')}` |",
            f"| Readable Summary | Human-readable text | `{os.path.join(output_dir, 'benchmark_instances_readable.txt')}` |",
            f"| Detailed Report (MD) | Markdown table | `{os.path.join(output_dir, 'benchmark_instances_report.md')}` |",
            f"| Data Summary (MD) | This file | `{os.path.join(output_dir, 'data_summary.md')}` |",
            "", "---", "",
        ]

    # Dataset statistics
    md += [
        f"## {stats_num}. Dataset Statistics", "",
        "### Overview", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Total Test Instances | {total_instances} |",
        f"| Total Ideas/Papers | {total_ideas} |",
        f"| Unique Papers | {len(unique_papers)} |",
        f"| Unique Positive Papers | {len(unique_pos)} |",
        f"| Unique Negative Papers | {len(unique_neg)} |",
        f"| Duplicate Paper Occurrences | {len(duplicated)} |",
        f"| Paper Reuse Rate | {pct(total_ideas - len(unique_papers), total_ideas)} |",
        f"| Papers per Instance | {args.num_top_papers + args.num_bottom_papers} |",
        f"| Positive Papers per Instance | {args.num_top_papers} |",
        f"| Negative Papers per Instance | {args.num_bottom_papers} |", "",
        "### Rating Statistics", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Overall Average Rating | {fmt(paper_stats['avg_rating'])} |",
        f"| Positive Papers Avg Rating | {fmt(paper_stats['avg_pos_rating'])} |",
        f"| Negative Papers Avg Rating | {fmt(paper_stats['avg_neg_rating'])} |",
        f"| Rating Gap (Pos - Neg) | {fmt((paper_stats['avg_pos_rating'] - paper_stats['avg_neg_rating']) if paper_stats['avg_pos_rating'] and paper_stats['avg_neg_rating'] else None)} |",
        "",
        "### Contribution Statistics", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Overall Average Contribution | {fmt(paper_stats['avg_contribution'])} |",
        f"| Positive Papers Avg Contribution | {fmt(paper_stats['avg_pos_contribution'])} |",
        f"| Negative Papers Avg Contribution | {fmt(paper_stats['avg_neg_contribution'])} |",
        f"| Contribution Gap (Pos - Neg) | {fmt((paper_stats['avg_pos_contribution'] - paper_stats['avg_neg_contribution']) if paper_stats['avg_pos_contribution'] and paper_stats['avg_neg_contribution'] else None)} |",
        "",
        "### Novelty Signals", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Papers with Positive Signals | {paper_stats['papers_with_pos_signals']} ({pct(paper_stats['papers_with_pos_signals'], total_ideas)}) |",
        f"| Papers with Negative Signals | {paper_stats['papers_with_neg_signals']} ({pct(paper_stats['papers_with_neg_signals'], total_ideas)}) |",
        f"| Total Positive Signals | {paper_stats['total_pos_signals']} |",
        f"| Total Negative Signals | {paper_stats['total_neg_signals']} |",
        f"| Avg Positive Signals per Paper | {fmt(paper_stats['total_pos_signals'] / total_ideas if total_ideas else None)} |",
        f"| Avg Negative Signals per Paper | {fmt(paper_stats['total_neg_signals'] / total_ideas if total_ideas else None)} |",
        "",
        "### Reviewer Signal Distribution (Unique Papers)", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Total Reviewers | {rs['total']} |",
        f"| Reviewers with Positive Signals | {rs['pos']} ({pct(rs['pos'], rs['total'])}) |",
        f"| Reviewers with Negative Signals | {rs['neg']} ({pct(rs['neg'], rs['total'])}) |",
        f"| Reviewers with Both Signals | {rs['both']} ({pct(rs['both'], rs['total'])}) |",
        f"| Reviewers with No Signals | {rs['none']} ({pct(rs['none'], rs['total'])}) |",
        "",
        "#### Breakdown by Paper Type", "",
        "**Positive Papers (Accepted):**", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Total Reviewers | {rs['pos_paper_total']} |",
        f"| Reviewers with Positive Signals | {rs['pos_paper_pos']} ({pct(rs['pos_paper_pos'], rs['pos_paper_total'])}) |",
        f"| Reviewers with Negative Signals | {rs['pos_paper_neg']} ({pct(rs['pos_paper_neg'], rs['pos_paper_total'])}) |",
        "",
        "**Negative Papers (Rejected):**", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Total Reviewers | {rs['neg_paper_total']} |",
        f"| Reviewers with Positive Signals | {rs['neg_paper_pos']} ({pct(rs['neg_paper_pos'], rs['neg_paper_total'])}) |",
        f"| Reviewers with Negative Signals | {rs['neg_paper_neg']} ({pct(rs['neg_paper_neg'], rs['neg_paper_total'])}) |",
        "",
        "### Distribution by Topic/Context", "",
        "| Topic | Instance Count | Percentage |", "|-------|----------------|------------|",
    ]
    for ctx, cnt in sorted(context_counts.items(), key=lambda x: -x[1]):
        md.append(f"| {ctx} | {cnt} | {pct(cnt, total_instances)} |")

    md += ["", "### Source Data (After Filtering)", "",
           "| Area | Top Papers Available | Bottom Papers Available |",
           "|------|---------------------|------------------------|"]
    for area, s in sorted(source_stats.items()):
        md.append(f"| {area} | {s['top']} | {s['bottom']} |")

    if duplicated:
        md += ["", "### Duplicated Papers", "",
               f"**Note:** {len(duplicated)} papers appear multiple times.", "",
               "| Paper Title | Occurrences |", "|-------------|-------------|"]
        for title, cnt in sorted(duplicated.items(), key=lambda x: -x[1])[:10]:
            safe = title.replace("|", "\\|")[:80] + ("..." if len(title) > 80 else "")
            md.append(f"| {safe} | {cnt} |")
        if len(duplicated) > 10:
            md.append(f"| ... and {len(duplicated) - 10} more | |")

    # Reproducibility
    more = args.model_name or args.use_abstract_only or args.mix_topics or args.strictness or is_both
    cmd_lines = [
        "```bash",
        "python create_benchmark_instances.py \\",
        f"    --iclr_data {' '.join(args.iclr_data)} \\",
        f"    --num_top_papers {args.num_top_papers} \\",
        f"    --num_bottom_papers {args.num_bottom_papers} \\",
        f"    --output_dir {args.output_dir}" + (" \\" if more else ""),
    ]
    if args.model_name:
        cmd_lines.append(f"    --model_name {args.model_name}" +
                         (" \\" if args.use_abstract_only or args.mix_topics or args.strictness or is_both else ""))
    if args.use_abstract_only:
        cmd_lines.append(f"    --use_abstract_only" +
                         (" \\" if args.mix_topics or args.strictness or is_both or getattr(args, 'manipulate_generated_negatives', False) else ""))
    if getattr(args, 'manipulate_generated_negatives', False):
        cmd_lines.append(f"    --manipulate_generated_negatives" +
                         (" \\" if args.mix_topics or args.strictness or is_both else ""))
    if args.mix_topics:
        cmd_lines.append(f"    --mix_topics" + (" \\" if args.strictness or is_both else ""))
    if args.strictness:
        cmd_lines.append(f"    --strictness {args.strictness}" + (" \\" if is_both else ""))
    if is_both:
        cmd_lines.append("    --generate_both_versions")
    for flag, val in [
        ('min_pos_rating', args.min_pos_rating), ('max_pos_rating', args.max_pos_rating),
        ('min_neg_rating', args.min_neg_rating), ('max_neg_rating', args.max_neg_rating),
        ('min_pos_contribution', args.min_pos_contribution), ('max_pos_contribution', args.max_pos_contribution),
        ('min_neg_contribution', args.min_neg_contribution), ('max_neg_contribution', args.max_neg_contribution),
    ]:
        if val is not None:
            cmd_lines.append(f"    --{flag} {val} \\")
    if args.max_workers != 1:
        cmd_lines.append(f"    --max_workers {args.max_workers}")
    cmd_lines.append("```")

    md += [
        "", "---", "",
        f"## {repro_num}. Reproducibility Information", "",
        "### Command to Reproduce", "",
        *cmd_lines, "",
        "### Environment", "",
        f"- **Python Version:** {sys.version.split()[0]}",
        f"- **Script Location:** `{os.path.abspath(__file__)}`",
        f"- **Working Directory:** `{os.getcwd()}`",
        "", "---", "",
        "*This summary was automatically generated by `create_benchmark_instances.py`*",
    ]

    return "\n".join(md)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading + filtering
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_filter_datasets(args):
    """Load all clean datasets from disk and apply all configured filters."""
    clean_data_list = []
    for file_path in args.iclr_data:
        if not os.path.exists(file_path):
            LOGGER.error(f"Clean dataset file not found: {file_path}")
            continue
        clean_data_list.append(load_clean_dataset(file_path))

    if not clean_data_list:
        return None

    def apply_to_area(clean_data, key, fn, label):
        for area in clean_data:
            if key in clean_data[area]:
                before = len(clean_data[area][key])
                clean_data[area][key] = fn(clean_data[area][key])
                LOGGER.info(f"Area '{area}' {key}: {before} -> {len(clean_data[area][key])} after {label}")

    for clean_data in clean_data_list:
        for area in [a for a in clean_data if a.lower() in EXCLUDED_AREAS]:
            LOGGER.info(f"Dropping excluded area '{area}'")
            del clean_data[area]

        apply_to_area(clean_data, 'top_papers',
                      lambda ps: filter_papers_by_decision(ps, 'Accepted'), "decision filtering")
        apply_to_area(clean_data, 'bottom_papers',
                      lambda ps: filter_papers_by_decision(ps, 'Unaccepted'), "decision filtering")

        apply_to_area(clean_data, 'top_papers',
                      lambda ps: filter_papers(ps, args.min_pos_rating, args.max_pos_rating), "rating filtering")
        apply_to_area(clean_data, 'bottom_papers',
                      lambda ps: filter_papers(ps, args.min_neg_rating, args.max_neg_rating), "rating filtering")

        if args.min_pos_contribution is not None or args.max_pos_contribution is not None:
            apply_to_area(clean_data, 'top_papers',
                          lambda ps: filter_papers_by_contribution(ps, args.min_pos_contribution,
                                                                   args.max_pos_contribution),
                          "contribution filtering")
        if args.min_neg_contribution is not None or args.max_neg_contribution is not None:
            apply_to_area(clean_data, 'bottom_papers',
                          lambda ps: filter_papers_by_contribution(ps, args.min_neg_contribution,
                                                                   args.max_neg_contribution),
                          "contribution filtering")

        if args.strictness:
            apply_to_area(clean_data, 'top_papers',
                          lambda ps: filter_papers_strict(ps, is_positive=True, mode=args.strictness),
                          f"strictness filtering ({args.strictness})")
            apply_to_area(clean_data, 'bottom_papers',
                          lambda ps: filter_papers_strict(ps, is_positive=False, mode=args.strictness),
                          f"strictness filtering ({args.strictness})")

    return clean_data_list


def _save_outputs(args, output_dir, base_name,
                  raw_instances, summaries, md_report,
                  manipulated_instances=None, manipulated_summaries=None, manipulated_md_report=None):
    """Persist all output artefacts to disk."""
    if args.generate_both_versions:
        raw_dir = os.path.join(output_dir, 'raw_abstract')
        manipulated_dir = os.path.join(output_dir, 'manipulated')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(manipulated_dir, exist_ok=True)

        batch_size = getattr(args, 'md_report_batch_size', None)

        LOGGER.info("Saving raw abstract version...")
        save_test_instances(raw_instances, os.path.join(raw_dir, f"{base_name}.yaml"))
        save_readable_summaries(summaries, os.path.join(raw_dir, f"{base_name}_readable.txt"))
        save_md_report(md_report, os.path.join(raw_dir, f"{base_name}_report.md"), batch_size)

        LOGGER.info("Saving manipulated abstract version...")
        save_test_instances(manipulated_instances, os.path.join(manipulated_dir, f"{base_name}.yaml"))
        save_readable_summaries(manipulated_summaries, os.path.join(manipulated_dir, f"{base_name}_readable.txt"))
        save_md_report(manipulated_md_report, os.path.join(manipulated_dir, f"{base_name}_report.md"), batch_size)

        LOGGER.info(f"Both versions saved to: {output_dir}")
        return raw_dir, manipulated_dir
    else:
        batch_size = getattr(args, 'md_report_batch_size', None)
        save_test_instances(raw_instances, os.path.join(output_dir, f"{base_name}.yaml"))
        save_readable_summaries(summaries, os.path.join(output_dir, f"{base_name}_readable.txt"))
        save_md_report(md_report, os.path.join(output_dir, f"{base_name}_report.md"), batch_size)
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str):
    """Load a YAML config file and return a SimpleNamespace that mimics argparse's Namespace."""
    import types
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    defaults = dict(
        iclr_data=[],
        num_top_papers=1,
        num_bottom_papers=2,
        output_dir='benchmark_instances',
        model_name=None,
        use_abstract_only=False,
        generate_both_versions=False,
        max_workers=1,
        mix_topics=False,
        strict=False,
        strictness=None,
        manipulation_prompt=None,
        llm_negatives=False,
        llm_negatives_model=None,
        llm_negatives_prompt=None,
        llm_negatives_reasoning_effort=None,
        manipulate_generated_negatives=False,
        min_pos_rating=None, max_pos_rating=None,
        min_neg_rating=None, max_neg_rating=None,
        min_pos_contribution=None, max_pos_contribution=None,
        min_neg_contribution=None, max_neg_contribution=None,
        md_report_batch_size=None,
        max_instances=None,
        pointwise=False,
        pointwise_balance=False,
        pointwise_shuffle=True,
        derive_pointwise_from_pairwise=None,
        pre_built_manipulation_cache_yaml=None,
        rebuild_negatives_from_yaml=None,
        dry_run=False,
    )
    for k, v in cfg.items():
        defaults[k] = v
    return types.SimpleNamespace(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise output helpers
# ─────────────────────────────────────────────────────────────────────────────

def generate_pointwise_readable_summary(pointwise_instances):
    """Return a human-readable text summary of all pointwise instances."""
    blocks = []
    for idx, inst in pointwise_instances.items():
        meta = inst.get('metadata', {})
        idea = inst['idea']
        pos_sigs = meta.get('positive_signals', [])
        neg_sigs = meta.get('negative_signals', [])
        similar = meta.get('similar_papers_mentioned', [])

        def signal_lines(signals):
            return "\n".join(f"  * {s}" for s in signals) if signals else "  None"

        block = "\n".join([
            f"Instance ID: {idx}",
            f"Label:       {inst['label']}",
            f"Context:     {inst['context']}",
            f"Title:       {meta.get('title', 'N/A')}",
            f"Rating:      {meta.get('rating', 'N/A')}",
            f"Contribution:{meta.get('contribution', 'N/A')}",
            "=" * 60,
            "Idea:",
            idea,
            "-" * 60,
            "Positive Novelty Signals:",
            signal_lines(pos_sigs),
            "-" * 60,
            "Negative Novelty Signals:",
            signal_lines(neg_sigs),
            "-" * 60,
            "Similar Papers Mentioned:",
            signal_lines(similar),
            "=" * 80,
        ])
        blocks.append(block)
    return "\n\n".join(blocks)


def generate_pointwise_md_report(pointwise_instances):
    """Return a markdown table with one row per pointwise instance."""
    header = "| ID | Label | Context | Title | Rating | Contribution | Idea |"
    separator = "|---|---|---|---|---|---|---|"
    rows = [header, separator]

    for idx, inst in pointwise_instances.items():
        meta = inst.get('metadata', {})
        idea = inst['idea']
        # Collapse newlines for table cell compatibility
        safe_idea = (idea.replace("|", "\\|")
                        .replace("\n\n", "<br><br>")
                        .replace("\n", " "))
        safe_title = meta.get('title', 'N/A').replace("|", "\\|")
        safe_context = inst['context'].replace("|", "\\|")
        rows.append(
            f"| {idx} | {inst['label']} | {safe_context} | {safe_title} "
            f"| {meta.get('rating', 'N/A')} | {meta.get('contribution', 'N/A')} "
            f"| {safe_idea} |"
        )
    return "\n".join(rows)


def generate_pointwise_data_summary(args, pointwise_instances, output_dir):
    """Generate a markdown data summary for a pointwise dataset."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pos_instances = {k: v for k, v in pointwise_instances.items() if v['label'] == 'POSITIVE'}
    neg_instances = {k: v for k, v in pointwise_instances.items() if v['label'] == 'NEGATIVE'}
    total = len(pointwise_instances)

    def safe_avg(vals):
        return sum(vals) / len(vals) if vals else None

    def fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"

    def pct(n, d):
        return f"{100 * n / d:.1f}%" if d > 0 else "N/A"

    def ratings_for(instances):
        vals = []
        for inst in instances.values():
            r = inst.get('metadata', {}).get('rating', 'N/A')
            if r != 'N/A':
                try:
                    vals.append(float(r))
                except ValueError:
                    pass
        return vals

    all_ratings = ratings_for(pointwise_instances)
    pos_ratings = ratings_for(pos_instances)
    neg_ratings = ratings_for(neg_instances)

    context_counts = Counter(inst['context'] for inst in pointwise_instances.values())

    md = [
        "# Pointwise Data Creation Summary", "",
        f"**Generated:** {timestamp}",
        f"**Output Directory:** `{os.path.abspath(output_dir)}`", "",
        "---", "", "## Parameters", "",
        "| Parameter | Value |", "|-----------|-------|",
        f"| ICLR Data | {', '.join(f'`{p}`' for p in args.iclr_data)} |",
        f"| Model Name | {args.model_name or 'None (raw abstracts)'} |",
        f"| Use Abstract Only | {args.use_abstract_only} |",
        f"| Manipulate Generated Negatives | {getattr(args, 'manipulate_generated_negatives', False)} |",
        f"| Pointwise Balance | {getattr(args, 'pointwise_balance', False)} |",
        f"| Pointwise Shuffle | {getattr(args, 'pointwise_shuffle', True)} |",
        f"| Strictness | {args.strictness or 'None'} |",
        f"| Max Workers | {args.max_workers} |",
        f"| Max Instances | {args.max_instances or 'None'} |", "",
        "### Filtering Parameters", "",
        "| Filter | Min | Max |", "|--------|-----|-----|",
        f"| Positive Rating | {args.min_pos_rating or 'None'} | {args.max_pos_rating or 'None'} |",
        f"| Negative Rating | {args.min_neg_rating or 'None'} | {args.max_neg_rating or 'None'} |",
        f"| Positive Contribution | {args.min_pos_contribution or 'None'} | {args.max_pos_contribution or 'None'} |",
        f"| Negative Contribution | {args.min_neg_contribution or 'None'} | {args.max_neg_contribution or 'None'} |",
        "", "---", "", "## Dataset Statistics", "",
        "### Overview", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Total Instances | {total} |",
        f"| POSITIVE Instances | {len(pos_instances)} ({pct(len(pos_instances), total)}) |",
        f"| NEGATIVE Instances | {len(neg_instances)} ({pct(len(neg_instances), total)}) |", "",
        "### Rating Statistics", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Overall Avg Rating | {fmt(safe_avg(all_ratings))} |",
        f"| POSITIVE Avg Rating | {fmt(safe_avg(pos_ratings))} |",
        f"| NEGATIVE Avg Rating | {fmt(safe_avg(neg_ratings))} |",
        f"| Rating Gap (Pos - Neg) | {fmt((safe_avg(pos_ratings) - safe_avg(neg_ratings)) if pos_ratings and neg_ratings else None)} |",
        "", "### Distribution by Topic/Context", "",
        "| Topic | Count | % of Total |", "|-------|-------|------------|",
    ]
    for ctx, cnt in sorted(context_counts.items(), key=lambda x: -x[1]):
        md.append(f"| {ctx} | {cnt} | {pct(cnt, total)} |")

    md += [
        "", "---", "", "## Output Files", "",
        "| File | Description |", "|------|-------------|",
        f"| `iclr_pointwise_instances.yaml` | Main data file (one idea per instance) |",
        f"| `iclr_pointwise_instances_readable.txt` | Human-readable text summary |",
        f"| `iclr_pointwise_instances_report.md` | Markdown table report |",
        f"| `data_summary.md` | This file |",
        "", "---", "",
        "*Generated by `create_benchmark_instances.py` (pointwise mode)*",
    ]
    return "\n".join(md)


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run report
# ─────────────────────────────────────────────────────────────────────────────

def _print_dry_run_report(args, clean_data_list):
    """Simulate instance selection and print expected counts without any LLM calls."""
    is_pointwise = getattr(args, 'pointwise', False)
    use_negatives = args.llm_negatives
    neg_type = "LLM" if args.llm_negatives else None

    lines = ["", "=" * 64, "  DRY RUN — no LLM calls will be made", "=" * 64, ""]

    if is_pointwise:
        pos_total = 0
        neg_total = 0
        area_rows = []

        data_source = clean_data_list
        if getattr(args, 'mix_topics', False):
            all_top, all_bottom = [], []
            for cd in clean_data_list:
                for d in cd.values():
                    all_top.extend(d.get('top_papers', []))
                    all_bottom.extend(d.get('bottom_papers', []))
            data_source = [{'Mixed Topics': {'top_papers': all_top, 'bottom_papers': all_bottom}}]

        for clean_data in data_source:
            for area, data in clean_data.items():
                pos_c = len(data.get('top_papers', []))
                if use_negatives:
                    neg_c = pos_c
                    neg_note = f"{neg_c} (to be generated via {neg_type})"
                else:
                    neg_c = len(data.get('bottom_papers', []))
                    neg_note = str(neg_c)
                area_rows.append((area, pos_c, neg_c, neg_note))
                pos_total += pos_c
                neg_total += neg_c

        if getattr(args, 'pointwise_balance', False):
            balanced = min(pos_total, neg_total)
            pos_total = neg_total = balanced

        total = pos_total + neg_total
        if args.max_instances is not None:
            total = min(total, int(args.max_instances))

        lines += [f"  Mode: POINTWISE", ""]
        col = f"  {'Area':<48} {'Positives':>10} {'Negatives':>30}"
        lines += [col, "  " + "-" * (len(col) - 2)]
        for area, pos_c, neg_c, neg_note in sorted(area_rows):
            lines.append(f"  {area[:48]:<48} {pos_c:>10} {neg_note:>30}")

        lines += [""]
        if getattr(args, 'pointwise_balance', False):
            lines.append(f"  Positives (after balance): {pos_total}")
            lines.append(f"  Negatives (after balance): {neg_total}")
        else:
            lines.append(f"  Positives: {pos_total}")
            lines.append(f"  Negatives: {neg_total}")
        lines.append(f"  Total instances: {total}")
        if args.max_instances is not None and pos_total + neg_total > int(args.max_instances):
            lines.append(f"    (capped by max_instances={args.max_instances})")

    else:
        # Pairwise
        total_instances = 0
        unique_paper_keys: set = set()
        area_rows = []

        data_source = clean_data_list
        if getattr(args, 'mix_topics', False):
            all_top, all_bottom = [], []
            for cd in clean_data_list:
                for d in cd.values():
                    all_top.extend(d.get('top_papers', []))
                    all_bottom.extend(d.get('bottom_papers', []))
            data_source = [{'Mixed Topics': {'top_papers': all_top, 'bottom_papers': all_bottom}}]

        for clean_data in data_source:
            for area, data in clean_data.items():
                top_papers = data.get('top_papers', [])
                top_c = len(top_papers)

                if use_negatives:
                    if top_c < args.num_top_papers:
                        area_rows.append((area, top_c, "—", 0, "SKIP (not enough top papers)"))
                        continue
                    n_inst = top_c // args.num_top_papers
                    neg_needed = n_inst * args.num_bottom_papers
                    neg_note = f"{neg_needed} (to be generated via {neg_type})"
                else:
                    bottom_papers = data.get('bottom_papers', [])
                    bot_c = len(bottom_papers)
                    if top_c < args.num_top_papers or bot_c < args.num_bottom_papers:
                        area_rows.append((area, top_c, str(bot_c), 0, "SKIP (not enough papers)"))
                        continue
                    n_inst = top_c // args.num_top_papers
                    neg_note = str(bot_c)
                    for p in bottom_papers:
                        k = p.get('title', 'N/A')
                        unique_paper_keys.add(k if k != 'N/A' else str(hash(p['abstract'])))

                for p in top_papers:
                    k = p.get('title', 'N/A')
                    unique_paper_keys.add(k if k != 'N/A' else str(hash(p['abstract'])))

                area_rows.append((area, top_c, neg_note, n_inst, ""))
                total_instances += n_inst

        if args.max_instances is not None and total_instances > int(args.max_instances):
            capped_total = int(args.max_instances)
        else:
            capped_total = total_instances

        lines += [f"  Mode: PAIRWISE",
                  f"  Papers per instance: {args.num_top_papers} positive + {args.num_bottom_papers} negative",
                  ""]
        col = f"  {'Area':<48} {'Top':>5} {'Bottom':>28} {'Instances':>10}  Note"
        lines += [col, "  " + "-" * (len(col) - 2)]
        for area, top_c, neg_note, n_inst, note in sorted(area_rows):
            lines.append(f"  {area[:48]:<48} {top_c:>5} {neg_note:>28} {n_inst:>10}  {note}")

        lines += [""]
        lines.append(f"  Total instances: {capped_total}")
        if total_instances != capped_total:
            lines.append(f"    ({total_instances} before cap; max_instances={args.max_instances})")

        if args.model_name and not getattr(args, 'use_abstract_only', False):
            lines.append(f"  Unique papers requiring manipulation: ~{len(unique_paper_keys)}")
            lines.append(f"    (model: {args.model_name})")

    if use_negatives:
        lines.append(f"  Negative generation ({neg_type}): LLM calls SKIPPED in dry run")

    lines += ["", "=" * 64, ""]
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild-negatives mode
# ─────────────────────────────────────────────────────────────────────────────

async def _rebuild_negatives_mode(args, debug_log: ManipulationDebugLog):
    """Replace negative ideas in an existing benchmark dataset, keeping all positive slots intact.

    Loads ``args.rebuild_negatives_from_yaml`` as the structural skeleton, generates
    fresh negatives via the configured LLM or pipeline path, then re-assembles each
    instance with its original positive slots verbatim and the new negative texts.
    """
    source_yaml = args.rebuild_negatives_from_yaml
    LOGGER.info(f"Rebuild-negatives mode: loading source dataset from {source_yaml}")
    with open(source_yaml, 'r') as f:
        source_data = yaml.safe_load(f)

    # ── Step 1: Extract instance structure ───────────────────────────────────
    instance_structure = {}   # inst_id → {context, expected_winners, pos_slots, neg_indices, pos_metadata}
    areas_with_counts: dict = {}  # area → total negatives needed

    for inst_id, inst in source_data.items():
        context = inst['context']
        expected_winners = inst['expected_winners']
        ideas = inst.get('ideas', {})
        metadata = inst.get('metadata', {})

        pos_indices = {int(w) for w in expected_winners}
        all_indices = sorted(int(k) for k in ideas.keys())
        neg_indices = sorted(i for i in all_indices if i not in pos_indices)

        instance_structure[inst_id] = {
            'context': context,
            'expected_winners': expected_winners,
            'pos_slots': {i: ideas.get(i, ideas.get(str(i), '')) for i in pos_indices},
            'neg_indices': neg_indices,
            'pos_metadata': {
                i: metadata.get(i, metadata.get(str(i), {})) for i in pos_indices
            },
        }
        areas_with_counts[context] = areas_with_counts.get(context, 0) + len(neg_indices)

    LOGGER.info(
        f"Source dataset: {len(instance_structure)} instances, "
        f"{sum(areas_with_counts.values())} negatives to regenerate "
        f"across {len(areas_with_counts)} area(s)"
    )

    # ── Create output dir and attach run log before any LLM work ─────────────
    # Must happen here so that utils.py logs during generation are captured.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    _attach_run_log(output_dir)

    # ── Step 2: Generate new negatives ───────────────────────────────────────
    total_needed = sum(areas_with_counts.values())
    LOGGER.info(f"Generating {total_needed} LLM negatives (max {args.max_workers} parallel)")
    generated_negatives_map = await _generate_llm_negatives(
        areas_with_counts=areas_with_counts,
        model_name=args.llm_negatives_model,
        prompt_template_path=args.llm_negatives_prompt,
        max_parallel=args.max_workers,
        reasoning_effort=getattr(args, 'llm_negatives_reasoning_effort', None),
    )

    # ── Step 3: Re-assemble instances ─────────────────────────────────────────
    area_neg_cursors = {area: 0 for area in areas_with_counts}
    new_instances = {}

    for inst_id, struct in instance_structure.items():
        context = struct['context']
        area_negs = generated_negatives_map.get(context, [])
        cursor = area_neg_cursors[context]

        new_ideas = {}
        new_metadata = {}

        for idx, idea_text in struct['pos_slots'].items():
            new_ideas[idx] = idea_text
            new_metadata[idx] = struct['pos_metadata'][idx]

        for neg_idx in struct['neg_indices']:
            if cursor < len(area_negs):
                neg_paper = area_negs[cursor]
                new_ideas[neg_idx] = neg_paper['abstract']
                new_metadata[neg_idx] = {
                    'title': neg_paper.get('title', 'N/A'),
                    'area': context,
                    'rating': 'N/A',
                    'contribution': 'N/A',
                    'type': 'NEGATIVE',
                    'positive_signals': [],
                    'negative_signals': [],
                    'similar_papers_mentioned': [],
                }
                cursor += 1
            else:
                LOGGER.warning(
                    f"[rebuild_negatives] Not enough generated negatives for area '{context}' "
                    f"(instance {inst_id}, slot {neg_idx}). Retaining original negative text."
                )
                source_inst = source_data[inst_id]
                orig_idea = source_inst['ideas'].get(neg_idx, source_inst['ideas'].get(str(neg_idx), ''))
                orig_meta = source_inst['metadata'].get(neg_idx, source_inst['metadata'].get(str(neg_idx), {}))
                new_ideas[neg_idx] = orig_idea
                new_metadata[neg_idx] = orig_meta

        area_neg_cursors[context] = cursor
        new_instances[inst_id] = {
            'context': context,
            'expected_winners': struct['expected_winners'],
            'ideas': new_ideas,
            'metadata': new_metadata,
        }

    LOGGER.info(f"Re-assembled {len(new_instances)} instances")

    # ── Step 4: Save output ───────────────────────────────────────────────────
    base_name = 'benchmark_instances'
    batch_size = getattr(args, 'md_report_batch_size', None)

    total_papers = max((len(inst['ideas']) for inst in new_instances.values()), default=2)
    md_header = "| Instance ID | Context | " + " | ".join(f"Paper {i}" for i in range(total_papers)) + " |"
    md_separator = "|---|---| " + " | ".join("---" for _ in range(total_papers)) + " |"
    md_rows = [md_header, md_separator]
    summaries = []

    for inst_id, inst in new_instances.items():
        context = inst['context']
        ideas = inst['ideas']
        metadata = inst['metadata']
        expected = {int(w) for w in inst['expected_winners']}

        block = [f"Instance ID: {inst_id}", f"Context: {context}", "=" * 80]
        for idx in sorted(ideas.keys()):
            ptype = "POSITIVE" if idx in expected else "NEGATIVE"
            meta = metadata.get(idx, {})
            block += [
                f"Paper {idx} ({ptype})",
                f"Title: {meta.get('title', 'N/A')}",
                "-" * 40,
                "Idea:",
                ideas[idx],
                "\n" + "-" * 80 + "\n",
            ]
        summaries.append("\n".join(block))

        row_parts = [str(inst_id), context.replace("|", "\\|")]
        for idx in sorted(ideas.keys()):
            ptype = "POSITIVE" if idx in expected else "NEGATIVE"
            meta = metadata.get(idx, {})
            safe_idea = (ideas[idx]
                         .replace("|", "\\|")
                         .replace("\n\n", "<br><br>")
                         .replace("\n", " "))
            row_parts.append(
                f"**Type**: {ptype}<br><br>"
                f"**Title**: {meta.get('title', 'N/A')}<br><br>"
                f"**Idea**: {safe_idea}"
            )
        md_rows.append("| " + " | ".join(row_parts) + " |")

    save_test_instances(new_instances, os.path.join(output_dir, f"{base_name}.yaml"))
    save_readable_summaries(summaries, os.path.join(output_dir, f"{base_name}_readable.txt"))
    save_md_report("\n".join(md_rows), os.path.join(output_dir, f"{base_name}_report.md"), batch_size)
    debug_log.write_md(output_dir)
    _write_instance_creation_cost_report(output_dir, instances_processed=len(new_instances))
    LOGGER.info(f"Rebuilt dataset saved to: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def async_main():
    parser = argparse.ArgumentParser(description="Create ICLR test instances from clean dataset.")
    parser.add_argument(
        '--config', type=str,
        default=os.path.join(os.path.dirname(__file__), 'config', 'create_pairwise_benchmark_instances.yaml'),
        help='Path to the YAML configuration file.',
    )
    parser.add_argument(
        '--derive_pointwise_from_pairwise', type=str,
        help='Path to a pairwise YAML file to convert to pointwise.',
    )
    parser.add_argument(
        '--output_dir', type=str,
        help='Base directory for output files.',
    )
    parser.add_argument(
        '--pre_built_manipulation_cache_yaml', type=str,
        help='Path to a YAML file to load pre-built manipulated abstracts from.',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Print expected instance counts without making any LLM calls.',
    )
    parser.add_argument(
        '--manipulate_generated_negatives', action='store_true',
        help='Apply abstract manipulation to generated negatives as well as positives.',
    )
    parser.add_argument(
        '--rebuild_negatives_from_yaml', type=str,
        help='Path to an existing benchmark YAML. Keeps all positive slots intact and regenerates only the negatives.',
    )
    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.config):
        LOGGER.error(f"Config file not found: {cli_args.config}")
        return

    LOGGER.info(f"Loading configuration from: {cli_args.config}")
    args = _load_config(cli_args.config)
    load_secrets()

    # Overwrite config values with CLI arguments if provided
    if cli_args.derive_pointwise_from_pairwise:
        args.derive_pointwise_from_pairwise = cli_args.derive_pointwise_from_pairwise
    if cli_args.output_dir:
        args.output_dir = cli_args.output_dir
    if cli_args.pre_built_manipulation_cache_yaml:
        args.pre_built_manipulation_cache_yaml = cli_args.pre_built_manipulation_cache_yaml
    if cli_args.dry_run:
        args.dry_run = True
    if cli_args.manipulate_generated_negatives:
        args.manipulate_generated_negatives = True
    if cli_args.rebuild_negatives_from_yaml:
        args.rebuild_negatives_from_yaml = cli_args.rebuild_negatives_from_yaml

    # ── Populate pre-built manipulation cache from existing dataset ──────────
    pre_built_manipulation_cache = {}
    if args.pre_built_manipulation_cache_yaml:
        if not os.path.exists(args.pre_built_manipulation_cache_yaml):
            LOGGER.error(f"Pre-built cache YAML not found: {args.pre_built_manipulation_cache_yaml}")
            return
        LOGGER.info(f"Loading pre-built manipulation cache from: {args.pre_built_manipulation_cache_yaml}")
        with open(args.pre_built_manipulation_cache_yaml, 'r') as f:
            cache_data = yaml.safe_load(f)

        for inst in cache_data.values():
            ideas = inst.get('ideas', {})
            metadata = inst.get('metadata', {})
            for idea_id, idea_text in ideas.items():
                meta = metadata.get(str(idea_id), metadata.get(int(idea_id), {}))
                title = meta.get('title', 'N/A')
                if title != 'N/A':
                    pre_built_manipulation_cache[title] = idea_text
                else:
                    # Fallback to hash-based key if title is missing
                    key = str(hash(idea_text))
                    pre_built_manipulation_cache[key] = idea_text
        LOGGER.info(f"Populated cache with {len(pre_built_manipulation_cache)} manipulated abstracts.")

    # ── Derived Pointwise mode (from Pairwise YAML) ──────────────────────────
    if args.derive_pointwise_from_pairwise:
        LOGGER.info(f"Deriving pointwise instances from pairwise YAML: {args.derive_pointwise_from_pairwise}")
        with open(args.derive_pointwise_from_pairwise, 'r') as f:
            pairwise_data = yaml.safe_load(f)

        pointwise_instances = {}
        pw_idx = 0
        seen_ideas: set[str] = set()
        n_dupes = 0
        for pair_id, inst in pairwise_data.items():
            context = inst.get('context', '')
            expected = inst.get('expected_winners', [])
            ideas = inst.get('ideas', {})
            metadata = inst.get('metadata', {})

            for idea_id, idea_text in ideas.items():
                idea_id_int = int(idea_id)
                label = "POSITIVE" if idea_id_int in expected else "NEGATIVE"
                meta = metadata.get(str(idea_id), metadata.get(idea_id_int, {}))
                if 'type' not in meta:
                    meta['type'] = label

                if idea_text.strip() in seen_ideas:
                    n_dupes += 1
                    continue
                seen_ideas.add(idea_text.strip())

                pointwise_instances[str(pw_idx)] = {
                    'context': context,
                    'idea': idea_text,
                    'label': label,
                    'metadata': meta
                }
                pw_idx += 1

        if n_dupes:
            LOGGER.info(f"Deduplication: removed {n_dupes} duplicate ideas ({len(pointwise_instances)} unique ideas retained)")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        _attach_run_log(output_dir)
        base_name = 'iclr_pointwise_instances'

        batch_size = getattr(args, 'md_report_batch_size', None)
        save_test_instances(pointwise_instances, os.path.join(output_dir, f"{base_name}.yaml"))
        save_readable_summaries(
            [generate_pointwise_readable_summary(pointwise_instances)],
            os.path.join(output_dir, f"{base_name}_readable.txt"),
        )
        save_md_report(
            generate_pointwise_md_report(pointwise_instances),
            os.path.join(output_dir, f"{base_name}_report.md"),
            batch_size,
        )
        save_data_summary(
            generate_pointwise_data_summary(args, pointwise_instances, output_dir),
            os.path.join(output_dir, "data_summary.md"),
        )
        _write_instance_creation_cost_report(output_dir, instances_processed=len(pointwise_instances))
        LOGGER.info(f"Derived pointwise instances saved to: {output_dir}")
        return

    # Validate options
    if args.generate_both_versions:
        if args.use_abstract_only:
            LOGGER.error("generate_both_versions cannot be used with use_abstract_only")
            return
        if not args.model_name:
            LOGGER.error("generate_both_versions requires model_name to be set")
            return
    if args.strictness and args.strictness not in ('all', 'majority'):
        LOGGER.error(f"Invalid strictness value '{args.strictness}'. Choose from: 'all', 'majority'")
        return
    if args.llm_negatives:
        if not args.llm_negatives_model:
            LOGGER.error("llm_negatives requires llm_negatives_model to be set")
            return
        if not args.llm_negatives_prompt or not os.path.exists(args.llm_negatives_prompt):
            LOGGER.error("llm_negatives requires llm_negatives_prompt to be a valid template path")
            return

    # Shared debug log for all manipulation calls in this run
    debug_log = ManipulationDebugLog()

    # ── Rebuild-negatives mode ────────────────────────────────────────────────
    if getattr(args, 'rebuild_negatives_from_yaml', None):
        if args.generate_both_versions:
            LOGGER.error("rebuild_negatives_from_yaml cannot be combined with generate_both_versions")
            return
        if not os.path.exists(args.rebuild_negatives_from_yaml):
            LOGGER.error(f"rebuild_negatives_from_yaml YAML not found: {args.rebuild_negatives_from_yaml}")
            return
        await _rebuild_negatives_mode(args, debug_log)
        return

    # Load and filter datasets
    clean_data_list = _load_and_filter_datasets(args)
    if not clean_data_list:
        LOGGER.error("No valid clean dataset files provided.")
        return

    if getattr(args, 'dry_run', False):
        _print_dry_run_report(args, clean_data_list)
        return

    # Create the timestamped output directory before any LLM work so that all
    # prompt_openai_client / prompt_anthropic_client calls are captured in run.log.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    _attach_run_log(output_dir)

    # Generate negatives by prompting an LLM directly, if requested
    generated_negatives_map = {}
    use_negatives = args.llm_negatives
    if use_negatives:
        # Calculate how many negatives each area needs.
        # Pointwise: 1 negative per positive paper.
        # Pairwise:  (num_top // num_top_papers) * num_bottom_papers.
        areas_with_counts = {}
        is_pointwise = getattr(args, 'pointwise', False)
        for clean_data in clean_data_list:
            for area, data in clean_data.items():
                num_top = len(data.get('top_papers', []))
                if is_pointwise:
                    needed = num_top
                else:
                    if num_top < args.num_top_papers:
                        continue
                    num_instances = num_top // args.num_top_papers
                    needed = num_instances * args.num_bottom_papers
                areas_with_counts[area] = max(areas_with_counts.get(area, 0), needed)

        if not areas_with_counts:
            LOGGER.error("No areas with enough top papers to generate negatives for.")
            return

        # Cap generation to max_instances before the expensive pipeline runs
        if args.max_instances is not None:
            args.max_instances = int(args.max_instances)
            capped: dict = {}
            total_instances_so_far = 0
            for area, count in areas_with_counts.items():
                instances_this_area = count // args.num_bottom_papers
                remaining_budget = args.max_instances - total_instances_so_far
                if remaining_budget <= 0:
                    break
                take = min(instances_this_area, remaining_budget)
                capped[area] = take * args.num_bottom_papers
                total_instances_so_far += take
            areas_with_counts = capped
            LOGGER.info(f"max_instances={args.max_instances}: capped negatives generation to "
                        f"{sum(areas_with_counts.values())} negatives across {len(areas_with_counts)} areas.")

        total_needed = sum(areas_with_counts.values())
        LOGGER.info(f"Generating {total_needed} LLM negatives across {len(areas_with_counts)} areas "
                    f"(max {args.max_workers} parallel)")
        generated_negatives_map = await _generate_llm_negatives(
            areas_with_counts=areas_with_counts,
            model_name=args.llm_negatives_model,
            prompt_template_path=args.llm_negatives_prompt,
            max_parallel=args.max_workers,
            reasoning_effort=getattr(args, 'llm_negatives_reasoning_effort', None),
        )

    # When negatives were capped to max_instances, filter clean_data_list to only
    # the areas we actually generated negatives for, so create_test_instances
    # doesn't warn about every skipped area.
    if use_negatives and generated_negatives_map:
        clean_data_list = [
            {area: data for area, data in clean_data.items() if area in generated_negatives_map}
            for clean_data in clean_data_list
        ]
        clean_data_list = [d for d in clean_data_list if d]  # drop empty dicts

    # ── Pointwise mode ────────────────────────────────────────────────────────
    if getattr(args, 'pointwise', False):
        pointwise_instances = create_pointwise_instances(
            clean_data_list,
            model_name=args.model_name,
            use_abstract_only=args.use_abstract_only,
            max_workers=args.max_workers,
            manipulation_prompt=args.manipulation_prompt,
            pointwise_balance=getattr(args, 'pointwise_balance', False),
            pointwise_shuffle=getattr(args, 'pointwise_shuffle', True),
            max_instances=args.max_instances,
            generate_negatives=use_negatives,
            generated_negatives_map=generated_negatives_map,
            pre_built_manipulation_cache=pre_built_manipulation_cache,
            debug_log=debug_log,
            manipulate_generated_negatives=getattr(args, 'manipulate_generated_negatives', False),
        )

        base_name = 'iclr_pointwise_instances'

        batch_size = getattr(args, 'md_report_batch_size', None)
        save_test_instances(pointwise_instances, os.path.join(output_dir, f"{base_name}.yaml"))
        save_readable_summaries(
            [generate_pointwise_readable_summary(pointwise_instances)],
            os.path.join(output_dir, f"{base_name}_readable.txt"),
        )
        save_md_report(
            generate_pointwise_md_report(pointwise_instances),
            os.path.join(output_dir, f"{base_name}_report.md"),
            batch_size,
        )
        save_data_summary(
            generate_pointwise_data_summary(args, pointwise_instances, output_dir),
            os.path.join(output_dir, "data_summary.md"),
        )
        debug_log.write_md(output_dir)
        _write_instance_creation_cost_report(output_dir, instances_processed=len(pointwise_instances))
        LOGGER.info(f"Pointwise instances saved to: {output_dir}")
        return

    # Create instances
    result = create_test_instances(
        clean_data_list, args.num_top_papers, args.num_bottom_papers,
        args.model_name, args.use_abstract_only, args.mix_topics,
        args.max_workers, args.strictness,
        generate_both_versions=args.generate_both_versions,
        manipulation_prompt=args.manipulation_prompt,
        generate_negatives=use_negatives,
        generated_negatives_map=generated_negatives_map,
        max_instances=args.max_instances,
        pre_built_manipulation_cache=pre_built_manipulation_cache,
        debug_log=debug_log,
        manipulate_generated_negatives=getattr(args, 'manipulate_generated_negatives', False),
    )

    base_name = 'benchmark_instances'

    if args.generate_both_versions:
        raw_instances, summaries, md_report, manipulated_instances, manipulated_summaries, manipulated_md_report = result
        _save_outputs(args, output_dir, base_name,
                      raw_instances, summaries, md_report,
                      manipulated_instances, manipulated_summaries, manipulated_md_report)
        stats_instances = raw_instances
    else:
        raw_instances, summaries, md_report = result
        _save_outputs(args, output_dir, base_name, raw_instances, summaries, md_report)
        stats_instances = raw_instances

    data_summary = generate_data_summary(args, stats_instances, output_dir, clean_data_list)
    save_data_summary(data_summary, os.path.join(output_dir, "data_summary.md"))
    debug_log.write_md(output_dir)
    _write_instance_creation_cost_report(output_dir, instances_processed=len(stats_instances))


def _attach_run_log(output_dir: str) -> None:
    """Add a FileHandler for run.log inside output_dir to LOGGER."""
    log_path = os.path.join(output_dir, "run.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s"
    ))
    handler.setLevel(logging.DEBUG)
    LOGGER.addHandler(handler)
    LOGGER.info(f"Run log: {log_path}")


def _write_instance_creation_cost_report(
    output_dir: str,
    instances_processed: int | None = None,
) -> None:
    """Write cost_report.json and cost_report.md to output_dir if tracker is available."""
    if GLOBAL_COST_TRACKER is None:
        return
    import json
    cost_report = GLOBAL_COST_TRACKER.get_report()
    cost_report["test_mode"] = "instance_creation"
    if instances_processed is not None:
        cost_report["instances_processed"] = instances_processed
    cost_json_path = os.path.join(output_dir, "cost_report.json")
    with open(cost_json_path, "w") as _f:
        json.dump(cost_report, _f, indent=2)
    LOGGER.info(
        f"Instance creation cost: ${cost_report['total_cost_usd']:.4f} "
        f"({cost_report['total_calls']} LLM calls) — saved to {cost_json_path}"
    )
    if _write_cost_report_md is not None:
        cost_md_path = os.path.join(output_dir, "cost_report.md")
        _write_cost_report_md(cost_report, cost_md_path, title="Instance Creation Cost Report")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

