"""Shared instance selection for judge-error annotation.

Holds the single definition of "which scored instances are judge mistakes" and
the single paper-blocklist rule, so the annotation UI, the HTML export and the
batch preparer can never drift apart on either.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

import yaml

# One scored instance: (problem_id, instance, score).
ScoredInstance = tuple[str, dict, dict]


class Partition(NamedTuple):
    """Scored instances split by whether the judge agreed with the gold label."""

    mistakes: list[ScoredInstance]  # judge picked a non-gold idea, or tied
    controls: list[ScoredInstance]  # judge picked a gold winner outright
    n_blocked: int  # instances dropped by the paper blocklist


def find_scores_and_report(run_dir: Path) -> tuple[Path, Path | None]:
    """Locate scores.json and the accuracy_report.txt under a run/model dir."""
    scores = sorted(run_dir.rglob("run_pairwise_*/scores.json"))
    if not scores:
        scores = sorted(run_dir.rglob("scores.json"))
    if not scores:
        raise FileNotFoundError(f"No scores.json found under {run_dir}")
    if len(scores) > 1:
        raise SystemExit(
            f"Found {len(scores)} scores.json under {run_dir}; point --run-dir at a single run.\n"
            + "\n".join(f"  {s}" for s in scores)
        )
    scores_path = scores[0]
    reports = sorted(scores_path.parent.parent.rglob("accuracy_report.txt"))
    if not reports:
        reports = sorted(run_dir.rglob("accuracy_report.txt"))
    return scores_path, (reports[0] if reports else None)


def test_instances_path(report_path: Path | None, override: str | None, repo_root: Path) -> Path:
    """The test-instances YAML: an explicit override, else read from the accuracy report."""
    if override:
        return Path(override)
    if report_path is None:
        raise SystemExit("No accuracy_report.txt found; pass --test-instances explicitly.")
    text = report_path.read_text()
    m = re.search(r"Test Instances Path:\s*(.+)", text)
    if not m:
        raise SystemExit(f"Could not read 'Test Instances Path' from {report_path}; pass --test-instances.")
    rel = m.group(1).strip()
    p = Path(rel)
    return p if p.is_absolute() else repo_root / rel


def judge_pick(sc: dict) -> str | None:
    """The judge's pairwise winner as an idea id, or None when the judge tied.

    Deliberately not `elo_selected`: that is `elo_scores.index(max(...))`, and
    argmax returns the *first* maximum, so a tied instance silently reads as a
    confident vote for idea 0. Reading the comparison's winner code instead
    matches how calculate_pairwise_accuracy scores this run (winner == 2 is a
    tie, and a tie is never correct).
    """
    comps = sc.get("comparisons") or []
    if len(comps) == 1:
        try:
            w = int(comps[0].get("winner"))
        except (TypeError, ValueError):
            w = None
        if w == 2:
            return None
        if w in (0, 1):
            return str(comps[0][f"idea_{w}"])
    # No usable comparison: fall back to the aggregate scores, where an exact
    # draw is the same tie that produced winner == 2 in scoring.py.
    overall = sc.get("overall") or []
    if len(overall) == 2 and overall[0] == overall[1]:
        return None
    return str(sc["elo_selected"])


def is_judge_mistake(sc: dict, gold_list: list[str]) -> bool:
    """Whether the judge failed this instance. A tie counts as a failure."""
    pick = judge_pick(sc)
    return pick is None or pick not in gold_list


def novelty_margin(sc: dict) -> float:
    """How far apart the judge scored the two ideas on novelty (0 when it tied)."""
    novelty = sc.get("novelty") or []
    return abs(novelty[0] - novelty[1]) if len(novelty) == 2 else 0.0


def judge_novelty_winner(novelty: list[float]) -> str:
    """Argmax of the judge's per-idea novelty scores, or "tie"."""
    a, b = novelty[0], novelty[1]
    if a == b:
        return "tie"
    return "0" if a > b else "1"


def load_blocked_titles(repo_root: Path) -> set[str]:
    """Lower-cased paper titles from paper_blocklist.yaml (mirrors analysis.filtering)."""
    p = repo_root / "src" / "novelty_eval" / "benchmark_data" / "paper_blocklist.yaml"
    if not p.exists():
        return set()
    data = yaml.safe_load(p.read_text()) or {}
    return {t.lower() for t in data.get("blocked_titles", [])}


def instance_titles(inst: dict) -> list[str]:
    """Titles of the papers participating in an instance (from per-idea metadata)."""
    meta = inst.get("metadata") or {}
    if not isinstance(meta, dict):
        return []
    return [m["title"] for m in meta.values() if isinstance(m, dict) and m.get("title")]


def partition_scored(run_dir: Path, test_instances_override: str | None, repo_root: Path) -> Partition:
    """Split a judge run's scored instances into mistakes and correct-answer controls.

    A tie counts as a mistake: the judge failed to pick the gold winner, so it
    cannot serve as a control for "the judge got this right".

    Instances whose papers are blocklisted are dropped from both groups, so they
    never surface for annotation.
    """
    scores_path, report_path = find_scores_and_report(run_dir)
    scores = json.loads(scores_path.read_text())
    instances = yaml.safe_load(test_instances_path(report_path, test_instances_override, repo_root).read_text())
    blocked_titles = load_blocked_titles(repo_root)

    mistakes: list[ScoredInstance] = []
    controls: list[ScoredInstance] = []
    n_blocked = 0
    for pid, sc in scores.items():
        inst = instances.get(int(pid), instances.get(pid))
        if inst is None:
            continue
        if blocked_titles and any(t.lower() in blocked_titles for t in instance_titles(inst)):
            n_blocked += 1
            continue
        gold_list = [str(w) for w in inst["expected_winners"]]
        group = mistakes if is_judge_mistake(sc, gold_list) else controls
        group.append((str(pid), inst, sc))
    return Partition(mistakes, controls, n_blocked)
