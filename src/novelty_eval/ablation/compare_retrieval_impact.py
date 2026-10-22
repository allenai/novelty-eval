#!/usr/bin/env python3
"""
Compare baseline (no retrieval) vs retrieval-augmented accuracy reports.
Generates a markdown report showing where retrieval helped vs hurt.

Supports three test modes: ranking, pairwise, and pointwise.
Each mode has its own parsing and analysis logic.

Usage — explicit dirs:
    python compare_retrieval_impact.py \\
        --baseline-dir  <accuracy_test_artifacts dir of baseline run> \\
        --retrieval-dir <accuracy_test_artifacts dir of retrieval run> \\
        [--output retrieval_impact_comparison.md]

Usage — retrospective sweep scan (auto-discovers all pairs):
    python compare_retrieval_impact.py \\
        --sweep-dir output/ablation_sweeps/<timestamp> \\
        [--track ranking]          # optional filter
        [--baseline-name ranking_current]   # optional override
        [--retrieval-name ranking_retrieval]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode 1: explicit directories
    explicit = p.add_argument_group("Explicit-dir mode")
    explicit.add_argument(
        "--baseline-dir",
        help="Path to the accuracy_test_artifacts directory of the baseline (no-retrieval) run.",
    )
    explicit.add_argument(
        "--retrieval-dir",
        help="Path to the accuracy_test_artifacts directory of the retrieval-augmented run.",
    )
    explicit.add_argument(
        "--output", default=None,
        help="Output markdown path (default: <retrieval-dir>/retrieval_impact_comparison.md).",
    )

    # Mode 2: retrospective sweep scan
    retro = p.add_argument_group("Retrospective sweep mode")
    retro.add_argument(
        "--sweep-dir",
        help="Scan an existing ablation sweep dir for all baseline+retrieval pairs.",
    )
    retro.add_argument(
        "--track",
        help="Only process this track (e.g. ranking, pairwise, pointwise).",
    )
    retro.add_argument(
        "--baseline-name",
        help="Ablation dir name for the baseline (default: <track>_current).",
    )
    retro.add_argument(
        "--retrieval-name",
        help="Ablation dir name for the retrieval run (default: <track>_retrieval).",
    )

    p.add_argument(
        "--baseline-label", default=None,
        help="Human-readable label for the baseline run.",
    )
    p.add_argument(
        "--retrieval-label", default=None,
        help="Human-readable label for the retrieval run.",
    )
    return p.parse_args()


# ── File discovery ───────────────────────────────────────────────────────────────

def find_report_files(artifacts_dir: Path) -> tuple[Path, Path]:
    """
    Given an accuracy_test_artifacts directory (or any ancestor), find the
    debug_accuracy_report.txt and its companion .md file.
    Raises FileNotFoundError if either is missing.
    """
    txt_candidates = sorted(artifacts_dir.rglob("debug_accuracy_report.txt"))
    if not txt_candidates:
        raise FileNotFoundError(f"No debug_accuracy_report.txt found under {artifacts_dir}")
    txt = txt_candidates[0]

    md_candidates = sorted(artifacts_dir.rglob("debug_accuracy_report.txt.md"))
    if not md_candidates:
        raise FileNotFoundError(f"No debug_accuracy_report.txt.md found under {artifacts_dir}")
    md = md_candidates[0]

    return txt, md


def find_retrieval_pairs(
    sweep_dir: Path,
    track_filter: str | None = None,
    baseline_name: str | None = None,
    retrieval_name: str | None = None,
) -> list[dict]:
    """
    Scan a sweep directory for (baseline_dir, retrieval_dir) pairs.

    Expected layout:
        <sweep_dir>/
          <track>_current/        ← baseline
            <config_name>/
              accuracy_test_artifacts/...
          <track>_retrieval/      ← retrieval
            <config_name>/
              accuracy_test_artifacts/...

    Returns a list of dicts with keys:
        track, model, baseline_dir, retrieval_dir, output_path
    """
    pairs = []

    for ablation_dir in sorted(sweep_dir.iterdir()):
        if not ablation_dir.is_dir() or ablation_dir.name.startswith("_"):
            continue

        name = ablation_dir.name

        # Determine if this dir is a retrieval ablation
        if retrieval_name:
            if name != retrieval_name:
                continue
            # Infer track from the name: everything before _retrieval
            track = name.replace("_retrieval", "") if "_retrieval" in name else name
        else:
            if not name.endswith("_retrieval"):
                continue
            track = name[: -len("_retrieval")]

        if track_filter and track != track_filter:
            continue

        # Find the matching baseline dir
        baseline_ablation = baseline_name or f"{track}_current"
        baseline_ablation_dir = sweep_dir / baseline_ablation
        if not baseline_ablation_dir.is_dir():
            continue

        # Match model sub-dirs by stripping the ablation prefix from config names
        # Config name format: {ablation}-{model}
        for ret_model_dir in sorted(ablation_dir.iterdir()):
            if not ret_model_dir.is_dir() or ret_model_dir.name.startswith("_"):
                continue

            # Derive the corresponding baseline config name by replacing the
            # retrieval ablation prefix with the baseline ablation prefix.
            base_config_name = ret_model_dir.name.replace(name, baseline_ablation, 1)
            base_model_dir = baseline_ablation_dir / base_config_name
            if not base_model_dir.is_dir():
                # Try matching by model suffix (after first dash)
                parts = ret_model_dir.name.split("-", 1)
                if len(parts) == 2:
                    model_suffix = parts[1]
                    candidates = [
                        d for d in baseline_ablation_dir.iterdir()
                        if d.is_dir() and d.name.endswith(model_suffix)
                    ]
                    if len(candidates) == 1:
                        base_model_dir = candidates[0]
                    else:
                        continue
                else:
                    continue

            try:
                _find_ret = find_report_files(ret_model_dir)
                _find_base = find_report_files(base_model_dir)
            except FileNotFoundError:
                continue

            output_path = ret_model_dir / "retrieval_impact_comparison.md"
            pairs.append({
                "track": track,
                "model": ret_model_dir.name,
                "baseline_dir": base_model_dir,
                "retrieval_dir": ret_model_dir,
                "output_path": output_path,
            })

    return pairs


# ── Accuracy report helper ───────────────────────────────────────────────────────

def find_accuracy_report(debug_txt: Path) -> Path | None:
    """
    Return the accuracy_report.txt sibling of a debug_accuracy_report.txt, or None.
    The accuracy report lives in the same artifact directory and contains the
    aggregate statistics (Mean Accuracy, Mean F1, etc.) that are NOT present in
    the debug file.
    """
    candidate = debug_txt.parent / "accuracy_report.txt"
    return candidate if candidate.exists() else None


# ── Mode detection ───────────────────────────────────────────────────────────────

def detect_mode(txt_path: Path) -> str:
    """
    Infer the test mode from the artifact directory.
    Returns one of: 'ranking', 'pairwise', 'pointwise'.

    Reads accuracy_report.txt (which has 'Test Mode: X') first, then falls back
    to scanning debug_accuracy_report.txt for mode-specific strings.
    """
    acc_report = find_accuracy_report(txt_path)
    if acc_report:
        content = acc_report.read_text(encoding="utf-8")
        m = re.search(r"Test Mode:\s*(\w+)", content)
        if m:
            mode = m.group(1).lower()
            if mode in ("pairwise", "pointwise", "ranking"):
                return mode

    # Fallback: scan the debug file itself (for hand-crafted test fixtures or
    # older runs that embedded stats in the debug file).
    content = txt_path.read_text(encoding="utf-8")
    if "Mean LLM Pairwise Accuracy" in content:
        return "pairwise"
    if "Mean F1 (macro):" in content or "Mean Accuracy:" in content:
        return "pointwise"
    return "ranking"


# ── Ranking: parsers ─────────────────────────────────────────────────────────────

def parse_ranking_scores(text: str) -> dict[int, dict[int, float]]:
    """Return {problem_id: {idea_id: overall_score}} from the plain-text ranking report."""
    results: dict[int, dict[int, float]] = {}
    section = text.split("Scores per Run:")[-1] if "Scores per Run:" in text else ""
    current_problem = None
    for line in section.strip().splitlines():
        line = line.strip()
        m = re.match(r"Problem (\d+):", line)
        if m:
            current_problem = int(m.group(1))
            results[current_problem] = {}
            continue
        m = re.match(r"Idea (\d+): \{'novelty': [\d.]+, 'overall': ([\d.]+)}", line)
        if m and current_problem is not None:
            results[current_problem][int(m.group(1))] = float(m.group(2))
    return results


def parse_gold_from_md(md_path: Path) -> dict[int, int]:
    """Parse gold answers (single winner ID) from the ranking markdown report."""
    gold: dict[int, int] = {}
    with open(md_path, encoding="utf-8") as fh:
        current_problem = None
        for line in fh:
            m = re.match(r"^### Problem (\d+)", line)
            if m:
                current_problem = int(m.group(1))
                continue
            m = re.match(r"\*\*Gold Answer:\*\* (\d+)", line)
            if m and current_problem is not None:
                gold[current_problem] = int(m.group(1))
    return gold


def parse_idea_titles_from_md(md_path: Path) -> dict[int, dict[int, str]]:
    """Return {problem_id: {idea_id: title}} from the ranking markdown report."""
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()

    titles: dict[int, dict[int, str]] = {}
    problem_blocks = re.split(r"^### Problem (\d+)", content, flags=re.MULTILINE)
    i = 1
    while i < len(problem_blocks) - 1:
        prob_id = int(problem_blocks[i])
        block = problem_blocks[i + 1]
        titles[prob_id] = {}

        for m in re.finditer(r"\*\*Idea (\d+)\*\*\s*\n\s+- \*\*Title:\*\*\s*(.+)", block):
            titles[prob_id][int(m.group(1))] = m.group(2).strip()
        for m in re.finditer(r"\*\*Idea (\d+)\*\* ranked.*?\n\s+- \*\*Title:\*\*\s*(.+)", block):
            titles[prob_id][int(m.group(1))] = m.group(2).strip()
        for m in re.finditer(r"\*\*Idea (\d+)\*\* correctly ranked.*?\n\s+- \*\*Title:\*\*\s*(.+)", block):
            titles[prob_id][int(m.group(1))] = m.group(2).strip()

        i += 2
    return titles


def parse_iclr_ratings_from_md(md_path: Path) -> dict[int, dict[int, float]]:
    """Return {problem_id: {idea_id: iclr_rating}} from the ranking markdown."""
    ratings: dict[int, dict[int, float]] = {}
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()

    problem_blocks = re.split(r"^### Problem (\d+)", content, flags=re.MULTILINE)
    i = 1
    while i < len(problem_blocks) - 1:
        prob_id = int(problem_blocks[i])
        block = problem_blocks[i + 1]
        ratings[prob_id] = {}
        for m in re.finditer(
            r"\*\*Idea (\d+)\*\*.*?ICLR Rating:\*\*\s*([\d.]+)", block, re.DOTALL
        ):
            idea_id = int(m.group(1))
            if idea_id not in ratings[prob_id]:
                ratings[prob_id][idea_id] = float(m.group(2))
        i += 2
    return ratings


def parse_comparison_reasoning_from_md(md_path: Path) -> dict[int, list[str]]:
    """Return {problem_id: [reasoning_texts]} from <details> blocks in the MD."""
    reasonings: dict[int, list[str]] = {}
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()

    problem_blocks = re.split(r"^### Problem (\d+)", content, flags=re.MULTILINE)
    i = 1
    while i < len(problem_blocks) - 1:
        prob_id = int(problem_blocks[i])
        block = problem_blocks[i + 1]
        reasonings[prob_id] = []
        for m in re.finditer(
            r"Comparison Reasoning.*?</summary>\s*\n(.*?)</details>", block, re.DOTALL
        ):
            reasonings[prob_id].append(m.group(1).strip())
        i += 2
    return reasonings


def parse_retrieval_data_from_md(md_path: Path) -> dict[int, dict[int, dict]]:
    """Return retrieval queries and confidence scores from the MD report."""
    data: dict[int, dict[int, dict]] = {}
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()

    problem_blocks = re.split(r"^### Problem (\d+)", content, flags=re.MULTILINE)
    i = 1
    while i < len(problem_blocks) - 1:
        prob_id = int(problem_blocks[i])
        block = problem_blocks[i + 1]
        data[prob_id] = {}

        idea_chunks = re.split(r"\*\*Idea (\d+)\*\*", block)
        j = 1
        while j < len(idea_chunks) - 1:
            idea_id = int(idea_chunks[j])
            chunk = idea_chunks[j + 1]

            idea_data: dict = {"queries": [], "avg_confidence": None, "candidates": []}

            q_match = re.search(
                r"Retrieval Queries.*?</summary>\s*\n(.*?)</details>", chunk, re.DOTALL
            )
            if q_match:
                for line in q_match.group(1).strip().splitlines():
                    line = line.strip()
                    if line.startswith("- ") and not line.startswith("- **"):
                        idea_data["queries"].append(line[2:].strip())

            c_match = re.search(
                r"Candidates Confidence Scores.*?</summary>\s*\n(.*?)</details>",
                chunk, re.DOTALL,
            )
            if c_match:
                c_block = c_match.group(1).strip()
                avg_m = re.search(r"Average confidence:\s*`([\d.]+)`", c_block)
                if avg_m:
                    idea_data["avg_confidence"] = float(avg_m.group(1))
                for cm in re.finditer(
                    r"\d+\.\s+\*\*(.+?)\*\*\s*—\s*confidence:\s*`([\d.]+)`", c_block
                ):
                    idea_data["candidates"].append(
                        {"title": cm.group(1), "confidence": float(cm.group(2))}
                    )

            data[prob_id][idea_id] = idea_data
            j += 2

        i += 2
    return data


def parse_idea_texts_from_md(md_path: Path) -> dict[int, dict[int, str]]:
    """Return {problem_id: {idea_id: idea_text}} from the ranking markdown report."""
    texts: dict[int, dict[int, str]] = {}
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()

    problem_blocks = re.split(r"^### Problem (\d+)", content, flags=re.MULTILINE)
    i = 1
    while i < len(problem_blocks) - 1:
        prob_id = int(problem_blocks[i])
        block = problem_blocks[i + 1]
        texts[prob_id] = {}

        # Walk through each **Idea N** occurrence and extract the **Text:** block that follows.
        for idea_match in re.finditer(r"\*\*Idea (\d+)\*\*", block):
            idea_id = int(idea_match.group(1))
            if idea_id in texts[prob_id]:
                continue  # Use the first occurrence (gold winner details)
            tail = block[idea_match.end():]
            # Stop search at the next **Idea N** to avoid bleeding into another idea's text
            next_idea_m = re.search(r"\*\*Idea \d+\*\*", tail)
            search_region = tail[:next_idea_m.start()] if next_idea_m else tail
            text_m = re.search(r"\*\*Text:\*\*\s*\n((?:\s+>\s.*\n?)+)", search_region)
            if text_m:
                raw_lines = text_m.group(1).strip().splitlines()
                stripped = [re.sub(r"^\s*>\s?", "", line) for line in raw_lines]
                texts[prob_id][idea_id] = " ".join(stripped).strip()
        i += 2
    return texts


def parse_retrieved_related_work_from_md(md_path: Path) -> dict[int, dict[int, list[dict]]]:
    """
    Return retrieved related work entries per idea per problem.

    Structure: {problem_id: {idea_id: [{"title": str, "abstract": str, "url": str}]}}
    """
    work: dict[int, dict[int, list[dict]]] = {}
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()

    problem_blocks = re.split(r"^### Problem (\d+)", content, flags=re.MULTILINE)
    i = 1
    while i < len(problem_blocks) - 1:
        prob_id = int(problem_blocks[i])
        block = problem_blocks[i + 1]
        work[prob_id] = {}

        idea_chunks = re.split(r"\*\*Idea (\d+)\*\*", block)
        j = 1
        while j < len(idea_chunks) - 1:
            idea_id = int(idea_chunks[j])
            chunk = idea_chunks[j + 1]

            rw_match = re.search(
                r"Retrieved Related Work.*?</summary>\s*\n(.*?)</details>", chunk, re.DOTALL
            )
            if rw_match:
                rw_block = rw_match.group(1)
                papers = []
                # Each paper: "- **Title** (venue, year) [Link](url)\n  > abstract..."
                for pm in re.finditer(
                    r"-\s+\*\*(.+?)\*\*[^\n]*(?:\[Link\]\(([^)]+)\))?\s*\n((?:\s+>\s.*\n?)*)",
                    rw_block,
                ):
                    title = pm.group(1).strip()
                    url = pm.group(2) or ""
                    abstract_lines = pm.group(3).strip().splitlines()
                    abstract = " ".join(
                        re.sub(r"^\s*>\s?", "", ln) for ln in abstract_lines
                    ).strip()
                    papers.append({"title": title, "url": url, "abstract": abstract})
                if papers:
                    work[prob_id][idea_id] = papers
            j += 2
        i += 2
    return work


def parse_pointwise_problems_from_md(md_path: Path) -> dict[int, dict]:
    """
    Parse per-instance classification results from the pointwise markdown report.

    Returns {instance_id: {"ground_truth": str, "predicted": str, "correct": bool,
                            "title": str, "idea_text": str}}
    The report has "Wrong Classifications" and "Correct Classifications" sections,
    each listing instances with their ground truth and predicted labels.
    """
    problems: dict[int, dict] = {}
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()

    for m in re.finditer(
        r"-\s+\*\*Instance (\d+)\*\*\s*\n"
        r"(?:.*?\n)*?"
        r"\s+-\s+\*\*Ground Truth:\*\*\s+(\w+).*?\*\*Predicted:\*\*\s+(\w+)",
        content, re.DOTALL,
    ):
        inst_id = int(m.group(1))
        gt = m.group(2).strip().upper()
        pred = m.group(3).strip().upper()

        # Extract title (optional)
        region = content[m.start():m.start() + 500]
        title_m = re.search(r"\*\*Title:\*\*\s+(.+)", region)
        title = title_m.group(1).strip() if title_m else ""

        # Extract idea text (optional — under "**Idea Text:**" as blockquotes)
        text_m = re.search(r"\*\*Idea Text:\*\*\s*\n((?:\s+>\s.*\n?)+)", region)
        if not text_m:
            # Also look a bit further ahead
            region2 = content[m.start():m.start() + 2000]
            text_m = re.search(r"\*\*Idea Text:\*\*\s*\n((?:\s+>\s.*\n?)+)", region2)
        if text_m:
            raw = text_m.group(1).strip().splitlines()
            idea_text = " ".join(re.sub(r"^\s*>\s?", "", ln) for ln in raw).strip()
        else:
            idea_text = ""

        problems[inst_id] = {
            "ground_truth": gt,
            "predicted": pred,
            "correct": gt == pred,
            "title": title,
            "idea_text": idea_text,
        }

    return problems


def scores_to_ranking(scores: dict[int, float]) -> list[int]:
    """Highest score → rank 1."""
    return [idea for idea, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def rank_of(ranking: list[int], target: int) -> int:
    """1-based rank."""
    return ranking.index(target) + 1


# ── Pairwise: parsers ────────────────────────────────────────────────────────────

def parse_pairwise_accuracy(txt_path: Path) -> dict[str, float | int]:
    """
    Parse aggregate pairwise accuracy metrics.
    Reads accuracy_report.txt (where the stats actually live); falls back to the
    given file so hand-crafted test fixtures still work.
    Returns dict with: acc_with_ties, acc_no_ties, n_ties, support
    """
    acc_report = find_accuracy_report(txt_path)
    content = (acc_report or txt_path).read_text(encoding="utf-8")
    result: dict[str, float | int] = {}

    m = re.search(r"Mean LLM Pairwise Accuracy \(with ties\):\s*([\d.]+)\s+\(support=([\d.]+)\)", content)
    if m:
        result["acc_with_ties"] = float(m.group(1))
        result["support"] = int(float(m.group(2)))

    m = re.search(r"Mean LLM Pairwise Accuracy \(without ties\):\s*([\d.]+)\s+\(support=([\d.]+)\)", content)
    if m:
        result["acc_no_ties"] = float(m.group(1))
        result["support_no_ties"] = int(float(m.group(2)))

    m = re.search(r"Mean Number of Ties:\s*([\d.]+)", content)
    if m:
        result["n_ties"] = float(m.group(1))

    return result


def parse_pairwise_problems_from_md(md_path: Path) -> dict[int, dict]:
    """
    Parse pairwise comparison results from the markdown report.
    Returns {problem_id: {gold_answer, predicted, correct, raw_block}}

    The real format uses:
      - ``**Expected Winners:** X`` for the gold answer
      - ``**Pair:** ... — **Idea-N chosen (wrong)**`` for wrong predictions
      - ``### Problem N ✓`` header for correct predictions (``✓`` follows the digits)
    """
    problems: dict[int, dict] = {}
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()

    # After splitting, problem_blocks[i] = digit string, problem_blocks[i+1] = rest of block.
    # Good comparisons have " ✓" immediately after the digits (still at the start of block).
    problem_blocks = re.split(r"^### Problem (\d+)", content, flags=re.MULTILINE)
    i = 1
    while i < len(problem_blocks) - 1:
        prob_id = int(problem_blocks[i])
        block = problem_blocks[i + 1]

        # "### Problem N ✓" — the ✓ ends up at the start of the next block segment
        correct = block.startswith(" ✓")

        gold_m = re.search(r"\*\*Expected Winners:\*\*\s*(.+)", block)
        # Bad:  "**Idea-1 chosen (wrong)**"
        # Good: "correctly chose Idea-0"
        # Tie:  "**TIE**"
        pred_m = re.search(r"Idea-?(\d+) chosen|correctly chose Idea-?(\d+)|\*\*TIE\*\*", block)
        if pred_m:
            if pred_m.group(0) == "**TIE**":
                predicted = "TIE"
            else:
                predicted = (pred_m.group(1) or pred_m.group(2)).strip()
        else:
            predicted = None

        problems[prob_id] = {
            "gold_answer": gold_m.group(1).strip() if gold_m else None,
            "predicted": predicted,
            "correct": correct,
            "raw_block": block,
        }
        i += 2

    return problems


# ── Pointwise: parsers ───────────────────────────────────────────────────────────

def parse_pointwise_metrics(txt_path: Path) -> dict[str, float]:
    """
    Parse aggregate pointwise classification metrics.
    Reads accuracy_report.txt (where the stats actually live); falls back to the
    given file so hand-crafted test fixtures still work.
    Returns dict with: accuracy, f1_macro, precision_pos, precision_neg,
                       recall_pos, recall_neg, f1_pos, f1_neg
    """
    acc_report = find_accuracy_report(txt_path)
    content = (acc_report or txt_path).read_text(encoding="utf-8")
    result: dict[str, float] = {}

    _metric_patterns = [
        ("accuracy",      r"Mean Accuracy:\s*([\d.]+)"),
        ("f1_macro",      r"Mean F1 \(macro\):\s*([\d.]+)"),
        ("precision_pos", r"Mean Precision \(POSITIVE\):\s*([\d.]+)"),
        ("precision_neg", r"Mean Precision \(NEGATIVE\):\s*([\d.]+)"),
        ("recall_pos",    r"Mean Recall\s+\(POSITIVE\):\s*([\d.]+)"),
        ("recall_neg",    r"Mean Recall\s+\(NEGATIVE\):\s*([\d.]+)"),
        ("f1_pos",        r"Mean F1\s+\(POSITIVE\):\s*([\d.]+)"),
        ("f1_neg",        r"Mean F1\s+\(NEGATIVE\):\s*([\d.]+)"),
    ]
    for key, pattern in _metric_patterns:
        m = re.search(pattern, content)
        if m:
            result[key] = float(m.group(1))

    return result


# ── Report generation: ranking ───────────────────────────────────────────────────

def generate_ranking_report(
    lines: list[str],
    baseline_scores: dict[int, dict[int, float]],
    retrieval_scores: dict[int, dict[int, float]],
    all_gold: dict[int, int],
    titles: dict[int, dict[int, str]],
    ratings: dict[int, dict[int, float]],
    reasoning_baseline: dict[int, list[str]],
    reasoning_retrieval: dict[int, list[str]],
    retrieval_data: dict[int, dict[int, dict]],
    idea_texts: dict[int, dict[int, str]] | None = None,
    retrieved_work: dict[int, dict[int, list[dict]]] | None = None,
) -> None:
    w = lines.append

    all_problems = sorted(
        set(baseline_scores) & set(retrieval_scores) & set(all_gold)
    )

    if not all_problems:
        lines.append("*No problems found in common between the baseline and retrieval reports.*")
        lines.append("")
        lines.append("This usually means `detect_mode` picked the wrong mode, or the report "
                     "files are from different test instances.")
        return

    improved, degraded, unchanged_correct, unchanged_wrong = [], [], [], []
    for prob in all_problems:
        gold = all_gold[prob]
        b_ranking = scores_to_ranking(baseline_scores[prob])
        r_ranking = scores_to_ranking(retrieval_scores[prob])
        b_rank = rank_of(b_ranking, gold)
        r_rank = rank_of(r_ranking, gold)
        entry = dict(problem=prob, gold=gold, b_ranking=b_ranking,
                     r_ranking=r_ranking, b_rank=b_rank, r_rank=r_rank)
        if r_rank < b_rank:
            improved.append(entry)
        elif r_rank > b_rank:
            degraded.append(entry)
        elif b_rank == 1:
            unchanged_correct.append(entry)
        else:
            unchanged_wrong.append(entry)

    all_entries = improved + degraded + unchanged_correct + unchanged_wrong
    n = len(all_problems)
    b_hits1 = sum(1 for e in all_entries if e["b_rank"] == 1)
    r_hits1 = sum(1 for e in all_entries if e["r_rank"] == 1)

    w("## Summary")
    w("")
    w("| Category | Count |")
    w("|---|---|")
    w(f"| Total problems | {n} |")
    w(f"| Retrieval **improved** gold rank | {len(improved)} |")
    w(f"| Retrieval **degraded** gold rank | {len(degraded)} |")
    w(f"| Unchanged — both correct (rank 1) | {len(unchanged_correct)} |")
    w(f"| Unchanged — both wrong (same rank) | {len(unchanged_wrong)} |")
    w("")
    w(f"**Baseline Hits@1:** {b_hits1}/{n} ({b_hits1/n*100:.1f}%)")
    w(f"**Retrieval Hits@1:** {r_hits1}/{n} ({r_hits1/n*100:.1f}%)")
    w("")
    # ── Per-problem renderer ──────────────────────────────────────────────────────
    def _render_problem(e: dict, section_type: str) -> None:
        p_id = e["problem"]
        p_gold = e["gold"]
        b_r = e["b_ranking"]
        r_r = e["r_ranking"]
        b_rk = e["b_rank"]
        r_rk = e["r_rank"]

        p_titles = titles.get(p_id, {})
        p_ratings = ratings.get(p_id, {})
        gold_title = p_titles.get(p_gold, "(title not found)")
        gold_rating = p_ratings.get(p_gold, "?")

        emoji = "🟢" if section_type == "improved" else ("🔴" if section_type == "degraded" else "⚪")

        w(f"### {emoji} Problem {p_id}")
        w("")
        w("| | Baseline | Retrieval |")
        w("|---|---|---|")
        w(f"| **Predicted Ranking** | {', '.join(str(x) for x in b_r)} | {', '.join(str(x) for x in r_r)} |")
        w(f"| **Gold Idea Rank** | {b_rk} | {r_rk} |")
        w("")
        w(f"**Gold Winner:** Idea {p_gold} — *{gold_title}* (ICLR: {gold_rating})")
        w("")

        p_texts = (idea_texts or {}).get(p_id, {})
        p_rw    = (retrieved_work or {}).get(p_id, {})

        w("**All Ideas:**")
        w("")
        all_ids = sorted(set(list(p_titles) + list(p_ratings)))
        if not all_ids:
            all_ids = sorted(baseline_scores.get(p_id, {}))
        for idea_id in all_ids:
            t = p_titles.get(idea_id, "(title not found)")
            r = p_ratings.get(idea_id, "?")
            marker = " ⭐" if idea_id == p_gold else ""
            w(f"- **Idea {idea_id}:** *{t}* (ICLR: {r}){marker}")
            idea_text = p_texts.get(idea_id, "")
            if idea_text:
                w("")
                w("  <details>")
                w("  <summary>Idea text (click to expand)</summary>")
                w("")
                w("  ```")
                # Wrap at ~100 chars for readability inside the code block
                import textwrap as _tw
                for wrapped_line in _tw.wrap(idea_text, width=100) or [idea_text]:
                    w(f"  {wrapped_line}")
                w("  ```")
                w("")
                w("  </details>")
                w("")
        w("")

        if section_type in ("improved", "degraded"):
            b_above = set(b_r[:b_rk - 1])
            r_above = set(r_r[:r_rk - 1])
            newly_below = b_above - r_above
            newly_above = r_above - b_above
            if newly_below:
                w("**Ideas retrieval correctly moved below gold:** "
                  + ", ".join(f"Idea {i}" for i in sorted(newly_below)))
            if newly_above:
                w("**Ideas retrieval incorrectly moved above gold:** "
                  + ", ".join(f"Idea {i}" for i in sorted(newly_above)))
            w("")

        prob_retrieval = retrieval_data.get(p_id, {})
        if prob_retrieval:
            p_all_confs = [
                d["avg_confidence"]
                for d in prob_retrieval.values()
                if d.get("avg_confidence") is not None
            ]
            if p_all_confs:
                avg_conf = sum(p_all_confs) / len(p_all_confs)
                w(f"**📎 Average Retrieval Relevance Score (across all ideas): `{avg_conf:.4f}`**")
                w("")
            w("<details>")
            w("<summary><strong>Retrieval Queries, Relevance Scores & Retrieved Papers per Idea"
              " (click to expand)</strong></summary>")
            w("")
            for idea_id in sorted(prob_retrieval):
                idea_d = prob_retrieval[idea_id]
                idea_title = p_titles.get(idea_id, "(title not found)")
                marker = " ⭐" if idea_id == p_gold else ""
                w(f"#### Idea {idea_id}{marker}: *{idea_title}*")
                w("")
                if idea_d["queries"]:
                    w("**Retrieval Queries:**")
                    for q in idea_d["queries"]:
                        w(f"- `{q}`")
                    w("")
                if idea_d["avg_confidence"] is not None:
                    w(f"**Average Relevance Score: `{idea_d['avg_confidence']:.4f}`** "
                      f"({len(idea_d['candidates'])} candidates)")
                    w("")
                if idea_d["candidates"]:
                    w("| # | Retrieved Paper | Relevance |")
                    w("|---|---|---|")
                    for idx_c, cand in enumerate(idea_d["candidates"], 1):
                        w(f"| {idx_c} | {cand['title']} | `{cand['confidence']:.4f}` |")
                    w("")
                # Retrieved related work abstracts
                rw_papers = p_rw.get(idea_id, [])
                if rw_papers:
                    w("<details>")
                    w("<summary>Retrieved related work abstracts (click to expand)</summary>")
                    w("")
                    for paper in rw_papers:
                        w(f"**{paper['title']}**")
                        if paper.get("url"):
                            w(f"[Link]({paper['url']})")
                        if paper.get("abstract"):
                            w("")
                            w("```")
                            import textwrap as _tw
                            for wrapped_line in _tw.wrap(paper["abstract"], width=100):
                                w(wrapped_line)
                            w("```")
                        w("")
                    w("</details>")
                    w("")
            w("</details>")
            w("")

        r_reasons = reasoning_retrieval.get(p_id, [])
        b_reasons = reasoning_baseline.get(p_id, [])
        if section_type in ("improved", "degraded") and (r_reasons or b_reasons):
            w("<details>")
            w("<summary><strong>Key Comparison Reasoning (click to expand)</strong></summary>")
            w("")
            for label, reasons in [("Retrieval", r_reasons), ("Baseline", b_reasons)]:
                if reasons:
                    w(f"**{label} reasoning excerpts:**")
                    for reason in reasons[:3]:
                        snippet = reason[:500].replace("\n", " ").strip()
                        if len(reason) > 500:
                            snippet += "..."
                        w(f"> {snippet}")
                        w("")
            w("</details>")
            w("")
        w("---")
        w("")

    w("## 🟢 Cases Where Retrieval IMPROVED Rankings")
    w("")
    w(f"In these {len(improved)} problems, retrieval moved the gold idea closer to rank 1.")
    w("")
    for e in sorted(improved, key=lambda x: x["b_rank"] - x["r_rank"], reverse=True):
        _render_problem(e, "improved")

    w("## 🔴 Cases Where Retrieval DEGRADED Rankings")
    w("")
    w(f"In these {len(degraded)} problems, retrieval moved the gold idea further from rank 1.")
    w("")
    for e in sorted(degraded, key=lambda x: x["r_rank"] - x["b_rank"], reverse=True):
        _render_problem(e, "degraded")

    w("## ⚪ Cases Where Retrieval Had No Effect on Gold Rank")
    w("")
    w(f"### Both Correct ({len(unchanged_correct)} problems)")
    w("")
    w("| Problem | Gold Idea | Baseline Ranking | Retrieval Ranking | Identical? |")
    w("|---|---|---|---|---|")
    for e in unchanged_correct:
        same = "✅" if e["b_ranking"] == e["r_ranking"] else "❌ (non-gold order differs)"
        w(f"| {e['problem']} | Idea {e['gold']} | "
          f"{', '.join(str(x) for x in e['b_ranking'])} | "
          f"{', '.join(str(x) for x in e['r_ranking'])} | {same} |")
    w("")
    w(f"### Both Wrong at Same Rank ({len(unchanged_wrong)} problems)")
    w("")
    w("| Problem | Gold Idea | Gold Rank | Baseline Ranking | Retrieval Ranking |")
    w("|---|---|---|---|---|")
    for e in unchanged_wrong:
        w(f"| {e['problem']} | Idea {e['gold']} | {e['b_rank']} | "
          f"{', '.join(str(x) for x in e['b_ranking'])} | "
          f"{', '.join(str(x) for x in e['r_ranking'])} |")
    w("")

    w("## 📊 Pattern Analysis")
    w("")
    w("### Gold Winner ICLR Ratings by Category")
    w("")
    for label, entries in [
        ("Improved", improved), ("Degraded", degraded),
        ("Unchanged Correct", unchanged_correct), ("Unchanged Wrong", unchanged_wrong),
    ]:
        if entries:
            gl = [ratings.get(e["problem"], {}).get(e["gold"], 0) for e in entries]
            gl = [r for r in gl if r > 0]
            if gl:
                avg = sum(gl) / len(gl)
                w(f"- **{label}:** avg gold ICLR rating = {avg:.2f} "
                  f"(range {min(gl):.1f}–{max(gl):.1f})")
    w("")

    w("### Score Distribution Changes")
    w("")
    w("| Problem | Direction | Gold Idea | Baseline Score | Retrieval Score | Score Δ |")
    w("|---|---|---|---|---|---|")
    for e in improved + degraded:
        p_id = e["problem"]
        p_gold = e["gold"]
        b_score = baseline_scores[p_id].get(p_gold, 0)
        r_score = retrieval_scores[p_id].get(p_gold, 0)
        direction = "🟢 Improved" if e in improved else "🔴 Degraded"
        delta = r_score - b_score
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        w(f"| {p_id} | {direction} | Idea {p_gold} | {b_score:.1f} | {r_score:.1f} | {delta_str} |")
    w("")

    w("### Retrieval Relevance Scores by Category")
    w("")
    w("| Category | Avg Relevance | Problems |")
    w("|---|---|---|")
    for label, entries in [
        ("🟢 Improved", improved), ("🔴 Degraded", degraded),
        ("⚪ Unchanged Correct", unchanged_correct), ("🟡 Unchanged Wrong", unchanged_wrong),
    ]:
        cat_confs = [
            d["avg_confidence"]
            for e in entries
            for d in retrieval_data.get(e["problem"], {}).values()
            if d.get("avg_confidence") is not None
        ]
        if cat_confs:
            w(f"| {label} | `{sum(cat_confs)/len(cat_confs):.4f}` | {len(entries)} |")
        else:
            w(f"| {label} | N/A | {len(entries)} |")
    w("")

    w("### Per-Problem Retrieval Relevance (Changed Cases Only)")
    w("")
    w("| Problem | Direction | Gold Rank Change | Avg Relevance | Gold Idea Relevance | Distractors Avg | Bad Distractors Avg | Good Distractors Avg |")
    w("|---|---|---|---|---|---|---|---|")
    for e in improved + degraded:
        p_id = e["problem"]
        p_gold = e["gold"]
        r_r = e["r_ranking"]
        r_rk = e["r_rank"]
        direction = "🟢 Improved" if e in improved else "🔴 Degraded"
        rank_ch = f"{e['b_rank']}→{e['r_rank']}"

        prob_ret = retrieval_data.get(p_id, {})
        all_c = [d["avg_confidence"] for d in prob_ret.values() if d.get("avg_confidence") is not None]
        prob_avg = f"`{sum(all_c)/len(all_c):.4f}`" if all_c else "N/A"

        gold_ret = prob_ret.get(p_gold, {})
        gold_c = f"`{gold_ret['avg_confidence']:.4f}`" if gold_ret.get("avg_confidence") is not None else "N/A"

        dist_c = [d["avg_confidence"] for iid, d in prob_ret.items() if iid != p_gold and d.get("avg_confidence") is not None]
        dist_avg = f"`{sum(dist_c)/len(dist_c):.4f}`" if dist_c else "N/A"

        bad_ids = set(r_r[:r_rk - 1])
        bad_c = [prob_ret[iid]["avg_confidence"] for iid in bad_ids if iid in prob_ret and prob_ret[iid].get("avg_confidence") is not None]
        bad_avg = f"`{sum(bad_c)/len(bad_c):.4f}`" if bad_c else "N/A"

        good_ids = set(r_r[r_rk:])
        good_c = [prob_ret[iid]["avg_confidence"] for iid in good_ids if iid in prob_ret and prob_ret[iid].get("avg_confidence") is not None]
        good_avg = f"`{sum(good_c)/len(good_c):.4f}`" if good_c else "N/A"

        w(f"| {p_id} | {direction} | {rank_ch} | {prob_avg} | {gold_c} | {dist_avg} | {bad_avg} | {good_avg} |")
    w("")


# ── Report generation: pairwise ──────────────────────────────────────────────────

def generate_pairwise_report(
    lines: list[str],
    baseline_acc: dict,
    retrieval_acc: dict,
    baseline_problems: dict[int, dict] | None = None,
    retrieval_problems: dict[int, dict] | None = None,
) -> None:
    """Generate the pairwise comparison section of the report."""
    w = lines.append

    b_acc = baseline_acc.get("acc_with_ties", float("nan"))
    r_acc = retrieval_acc.get("acc_with_ties", float("nan"))
    b_acc_nt = baseline_acc.get("acc_no_ties", float("nan"))
    r_acc_nt = retrieval_acc.get("acc_no_ties", float("nan"))
    support = baseline_acc.get("support", "?")

    delta = r_acc - b_acc if not (b_acc != b_acc or r_acc != r_acc) else float("nan")
    delta_nt = r_acc_nt - b_acc_nt if not (b_acc_nt != b_acc_nt or r_acc_nt != r_acc_nt) else float("nan")

    def _fmt_delta(d: float) -> str:
        if d != d:
            return "N/A"
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    w("## Summary")
    w("")
    w("| Metric | Baseline | Retrieval | Δ |")
    w("|---|---|---|---|")
    w(f"| Accuracy (with ties) | `{b_acc:.4f}` | `{r_acc:.4f}` | `{_fmt_delta(delta)}` |")
    w(f"| Accuracy (no ties)   | `{b_acc_nt:.4f}` | `{r_acc_nt:.4f}` | `{_fmt_delta(delta_nt)}` |")
    w(f"| Mean ties | `{baseline_acc.get('n_ties', '?')}` | `{retrieval_acc.get('n_ties', '?')}` | — |")
    w(f"| Support | {support} | {retrieval_acc.get('support', '?')} | — |")
    w("")

    if delta == delta:  # not NaN
        if delta > 0.01:
            w(f"✅ Retrieval **improved** pairwise accuracy by `{delta:+.4f}`.")
        elif delta < -0.01:
            w(f"❌ Retrieval **degraded** pairwise accuracy by `{delta:.4f}`.")
        else:
            w(f"↔ Retrieval had **minimal effect** on pairwise accuracy (`{delta:+.4f}`).")
    w("")

    if baseline_problems is not None and retrieval_problems is not None:
        # Find problems where the outcome flipped
        flipped_to_correct, flipped_to_wrong = [], []
        common = set(baseline_problems) & set(retrieval_problems)
        for pid in sorted(common):
            b_correct = baseline_problems[pid].get("correct", False)
            r_correct = retrieval_problems[pid].get("correct", False)
            if not b_correct and r_correct:
                flipped_to_correct.append(pid)
            elif b_correct and not r_correct:
                flipped_to_wrong.append(pid)

        w("## Problem-Level Outcome Changes")
        w("")
        w(f"| Category | Count |")
        w("|---|---|")
        w(f"| Baseline wrong → Retrieval correct (🟢 fixed) | {len(flipped_to_correct)} |")
        w(f"| Baseline correct → Retrieval wrong (🔴 broken) | {len(flipped_to_wrong)} |")
        w(f"| No change | {len(common) - len(flipped_to_correct) - len(flipped_to_wrong)} |")
        w("")

        def _transition_table(pids: list[int], title: str) -> None:
            """Render a baseline→retrieval prediction transition matrix for a set of problem IDs."""
            if not pids:
                return
            # Collect all labels that appear, sorted (numbers first, then TIE)
            labels: set[str] = set()
            for pid in pids:
                b_pred = str(baseline_problems[pid].get("predicted") or "?")
                r_pred = str(retrieval_problems[pid].get("predicted") or "?")
                labels.add(b_pred)
                labels.add(r_pred)

            def _label_sort_key(lbl: str) -> tuple:
                try:
                    return (0, int(lbl))
                except ValueError:
                    return (1, lbl)

            sorted_labels = sorted(labels, key=_label_sort_key)

            # Build counts[b_pred][r_pred]
            counts: dict[str, dict[str, int]] = {lbl: {c: 0 for c in sorted_labels} for lbl in sorted_labels}
            for pid in pids:
                b_pred = str(baseline_problems[pid].get("predicted") or "?")
                r_pred = str(retrieval_problems[pid].get("predicted") or "?")
                counts[b_pred][r_pred] += 1

            w(f"#### {title}")
            w("")
            w("Rows = baseline prediction · Columns = retrieval prediction")
            w("")
            header = "| Baseline \\ Retrieval | " + " | ".join(f"`{c}`" for c in sorted_labels) + " |"
            sep    = "|---|" + "---|" * len(sorted_labels)
            w(header)
            w(sep)
            for b_lbl in sorted_labels:
                row_cells = " | ".join(str(counts[b_lbl][r_lbl]) for r_lbl in sorted_labels)
                w(f"| `{b_lbl}` | {row_cells} |")
            w("")

        _transition_table(flipped_to_correct, "🟢 Fixed — prediction transitions")
        _transition_table(flipped_to_wrong,   "🔴 Broken — prediction transitions")

        def _render_pairwise_problem(pid: int, baseline_info: dict, retrieval_info: dict) -> None:
            gold   = retrieval_info.get("gold_answer", "?")
            b_pred = baseline_info.get("predicted", "?")
            r_pred = retrieval_info.get("predicted", "?")
            # Use a block-level paragraph (not a list item) so the <details> below
            # starts at column 0 and is recognised as a raw HTML block by markdown
            # parsers. Indented (list-item) <details> break nested toggle rendering.
            w(f"**Problem {pid}** — GT: `{gold}` | Baseline predicted: `{b_pred}` | Retrieval predicted: `{r_pred}`")
            w("")
            raw = retrieval_info.get("raw_block", "")
            if raw:
                w("<details>")
                w("<summary>Problem details (click to expand)</summary>")
                w("")
                for line in raw.rstrip().split("\n"):
                    w(line)
                w("")
                w("</details>")
            w("")

        if flipped_to_correct:
            w("### 🟢 Problems Fixed by Retrieval")
            w("")
            for pid in flipped_to_correct:
                _render_pairwise_problem(pid, baseline_problems[pid], retrieval_problems[pid])

        if flipped_to_wrong:
            w("### 🔴 Problems Broken by Retrieval")
            w("")
            for pid in flipped_to_wrong:
                _render_pairwise_problem(pid, baseline_problems[pid], retrieval_problems[pid])


# ── Report generation: pointwise ─────────────────────────────────────────────────

def generate_pointwise_report(
    lines: list[str],
    baseline_metrics: dict[str, float],
    retrieval_metrics: dict[str, float],
    baseline_problems: dict[int, dict] | None = None,
    retrieval_problems: dict[int, dict] | None = None,
) -> None:
    """Generate the pointwise comparison section of the report."""
    w = lines.append

    metric_labels = [
        ("accuracy",      "Accuracy"),
        ("f1_macro",      "F1 (macro)"),
        ("precision_pos", "Precision (POSITIVE)"),
        ("precision_neg", "Precision (NEGATIVE)"),
        ("recall_pos",    "Recall (POSITIVE)"),
        ("recall_neg",    "Recall (NEGATIVE)"),
        ("f1_pos",        "F1 (POSITIVE)"),
        ("f1_neg",        "F1 (NEGATIVE)"),
    ]

    def _delta_cell(key: str) -> str:
        b = baseline_metrics.get(key)
        r = retrieval_metrics.get(key)
        if b is None or r is None:
            return "N/A"
        d = r - b
        sign = "+" if d >= 0 else ""
        emoji = " ✅" if d > 0.005 else (" ❌" if d < -0.005 else "")
        return f"`{sign}{d:.4f}`{emoji}"

    w("## Summary")
    w("")
    w("| Metric | Baseline | Retrieval | Δ |")
    w("|---|---|---|---|")
    for key, label in metric_labels:
        b_val = baseline_metrics.get(key)
        r_val = retrieval_metrics.get(key)
        b_str = f"`{b_val:.4f}`" if b_val is not None else "N/A"
        r_str = f"`{r_val:.4f}`" if r_val is not None else "N/A"
        w(f"| {label} | {b_str} | {r_str} | {_delta_cell(key)} |")
    w("")

    acc_b = baseline_metrics.get("accuracy")
    acc_r = retrieval_metrics.get("accuracy")
    if acc_b is not None and acc_r is not None:
        delta = acc_r - acc_b
        if delta > 0.005:
            w(f"✅ Retrieval **improved** pointwise accuracy by `{delta:+.4f}`.")
        elif delta < -0.005:
            w(f"❌ Retrieval **degraded** pointwise accuracy by `{delta:.4f}`.")
        else:
            w(f"↔ Retrieval had **minimal effect** on pointwise accuracy (`{delta:+.4f}`).")
    w("")

    if baseline_problems is not None and retrieval_problems is not None:
        common = set(baseline_problems) & set(retrieval_problems)
        fixed_ids  = sorted(
            pid for pid in common
            if not baseline_problems[pid]["correct"] and retrieval_problems[pid]["correct"]
        )
        broken_ids = sorted(
            pid for pid in common
            if baseline_problems[pid]["correct"] and not retrieval_problems[pid]["correct"]
        )
        no_change = len(common) - len(fixed_ids) - len(broken_ids)

        w("## Instance-Level Outcome Changes")
        w("")
        w("| Category | Count |")
        w("|---|---|")
        w(f"| Baseline wrong → Retrieval correct (🟢 fixed) | {len(fixed_ids)} |")
        w(f"| Baseline correct → Retrieval wrong (🔴 broken) | {len(broken_ids)} |")
        w(f"| No change | {no_change} |")
        w("")

        def _render_instance(inst_id: int, baseline_info: dict, retrieval_info: dict) -> None:
            title = retrieval_info.get("title", "")
            gt    = retrieval_info.get("ground_truth", "?")
            b_pred = baseline_info.get("predicted", "?")
            r_pred = retrieval_info.get("predicted", "?")
            idea_text = retrieval_info.get("idea_text", "")
            header = f"**Instance {inst_id}**"
            if title:
                header += f": *{title}*"
            w(f"- {header}")
            w(f"  - GT: `{gt}` | Baseline predicted: `{b_pred}` | Retrieval predicted: `{r_pred}`")
            if idea_text:
                w("")
                w("  <details>")
                w("  <summary>Idea text (click to expand)</summary>")
                w("")
                w("  ```")
                import textwrap as _tw
                for wrapped_line in _tw.wrap(idea_text, width=100) or [idea_text]:
                    w(f"  {wrapped_line}")
                w("  ```")
                w("")
                w("  </details>")
            w("")

        if fixed_ids:
            w("### 🟢 Instances Fixed by Retrieval")
            w("")
            for iid in fixed_ids:
                _render_instance(iid, baseline_problems[iid], retrieval_problems[iid])

        if broken_ids:
            w("### 🔴 Instances Broken by Retrieval")
            w("")
            for iid in broken_ids:
                _render_instance(iid, baseline_problems[iid], retrieval_problems[iid])


# ── Top-level report runner ───────────────────────────────────────────────────────

def run_comparison(
    baseline_dir: Path,
    retrieval_dir: Path,
    output_path: Path,
    baseline_label: str | None = None,
    retrieval_label: str | None = None,
) -> None:
    """
    Run the full retrieval impact comparison and write the markdown report.
    Auto-detects the test mode (ranking / pairwise / pointwise).
    """
    baseline_txt, baseline_md = find_report_files(baseline_dir)
    retrieval_txt, retrieval_md = find_report_files(retrieval_dir)

    mode = detect_mode(baseline_txt)

    b_label = baseline_label or _infer_label(baseline_dir)
    r_label = retrieval_label or _infer_label(retrieval_dir)

    lines: list[str] = []
    w = lines.append
    w("# Retrieval Impact Comparison Report")
    w("")
    w(f"**Mode:** `{mode}`")
    w("")
    w(f"Comparing **baseline** (`{b_label}`) vs **retrieval-augmented** (`{r_label}`).")
    w("")
    w(f"- Baseline report: `{baseline_txt}`")
    w(f"- Retrieval report: `{retrieval_txt}`")
    w("")

    if mode == "ranking":
        with open(baseline_txt, encoding="utf-8") as fh:
            b_scores = parse_ranking_scores(fh.read())
        with open(retrieval_txt, encoding="utf-8") as fh:
            r_scores = parse_ranking_scores(fh.read())

        all_gold = {**parse_gold_from_md(baseline_md), **parse_gold_from_md(retrieval_md)}
        titles = {**parse_idea_titles_from_md(baseline_md), **parse_idea_titles_from_md(retrieval_md)}
        ratings = {**parse_iclr_ratings_from_md(baseline_md), **parse_iclr_ratings_from_md(retrieval_md)}
        reasoning_b = parse_comparison_reasoning_from_md(baseline_md)
        reasoning_r = parse_comparison_reasoning_from_md(retrieval_md)
        ret_data = parse_retrieval_data_from_md(retrieval_md)
        idea_texts = {
            **parse_idea_texts_from_md(baseline_md),
            **parse_idea_texts_from_md(retrieval_md),
        }
        retrieved_work = parse_retrieved_related_work_from_md(retrieval_md)

        generate_ranking_report(
            lines, b_scores, r_scores, all_gold, titles, ratings,
            reasoning_b, reasoning_r, ret_data,
            idea_texts=idea_texts, retrieved_work=retrieved_work,
        )

    elif mode == "pairwise":
        b_acc = parse_pairwise_accuracy(baseline_txt)
        r_acc = parse_pairwise_accuracy(retrieval_txt)
        b_problems = parse_pairwise_problems_from_md(baseline_md)
        r_problems = parse_pairwise_problems_from_md(retrieval_md)
        generate_pairwise_report(lines, b_acc, r_acc, b_problems, r_problems)

    elif mode == "pointwise":
        b_metrics = parse_pointwise_metrics(baseline_txt)
        r_metrics = parse_pointwise_metrics(retrieval_txt)
        b_problems = parse_pointwise_problems_from_md(baseline_md)
        r_problems = parse_pointwise_problems_from_md(retrieval_md)
        generate_pointwise_report(lines, b_metrics, r_metrics, b_problems, r_problems)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[{mode}] Report written to: {output_path}")


def _infer_label(d: Path) -> str:
    """Infer a human-readable label from a directory path."""
    parts = d.parts
    return parts[-3] if len(parts) >= 3 else d.name


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
        if not sweep_dir.is_dir():
            print(f"ERROR: sweep dir does not exist: {sweep_dir}", file=sys.stderr)
            sys.exit(1)

        pairs = find_retrieval_pairs(
            sweep_dir,
            track_filter=args.track,
            baseline_name=args.baseline_name,
            retrieval_name=args.retrieval_name,
        )
        if not pairs:
            print("No retrieval+baseline pairs found in sweep dir.", file=sys.stderr)
            sys.exit(1)

        for pair in pairs:
            out = Path(args.output) if args.output else pair["output_path"]
            print(f"Processing pair: track={pair['track']} model={pair['model']}")
            run_comparison(
                pair["baseline_dir"], pair["retrieval_dir"], out,
                baseline_label=args.baseline_label,
                retrieval_label=args.retrieval_label,
            )

    elif args.baseline_dir and args.retrieval_dir:
        baseline_dir = Path(args.baseline_dir)
        retrieval_dir = Path(args.retrieval_dir)
        output_path = Path(args.output) if args.output else retrieval_dir / "retrieval_impact_comparison.md"

        run_comparison(
            baseline_dir, retrieval_dir, output_path,
            baseline_label=args.baseline_label,
            retrieval_label=args.retrieval_label,
        )
    else:
        print(
            "ERROR: provide either --baseline-dir + --retrieval-dir, or --sweep-dir.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
