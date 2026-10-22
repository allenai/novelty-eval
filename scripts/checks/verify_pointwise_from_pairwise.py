#!/usr/bin/env python3
"""Verify that a pointwise dataset contains exactly the same ideas as a pairwise dataset.

Usage:
    python scripts/verify_pointwise_from_pairwise.py <pairwise.yaml> <pointwise.yaml>
"""

import sys
import yaml
from pathlib import Path


def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(val)) for k, val in v.items()))
    if isinstance(v, list):
        return tuple(_freeze(item) for item in v)
    return v


def idea_key(text: str, metadata: dict) -> tuple:
    return (text.strip(), _freeze(metadata))


def extract_from_pairwise(path: Path) -> set:
    with open(path) as f:
        data = yaml.safe_load(f)
    ideas = set()
    for instance in data.values():
        for idx, text in instance["ideas"].items():
            ideas.add(idea_key(text, instance["metadata"][idx]))
    return ideas


def extract_from_pointwise(path: Path) -> tuple[set, dict]:
    """Returns (ideas, counts) where counts maps idea_key -> occurrence count."""
    with open(path) as f:
        data = yaml.safe_load(f)
    ideas: set = set()
    counts: dict = {}
    for instance in data.values():
        key = idea_key(instance["idea"], instance["metadata"])
        ideas.add(key)
        counts[key] = counts.get(key, 0) + 1
    return ideas, counts


def main():
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <pairwise.yaml> <pointwise.yaml>")

    pairwise_path = Path(sys.argv[1])
    pointwise_path = Path(sys.argv[2])

    for p in (pairwise_path, pointwise_path):
        if not p.exists():
            sys.exit(f"ERROR: file not found: {p}")

    print(f"Pairwise : {pairwise_path}")
    print(f"Pointwise: {pointwise_path}")
    print()

    pairwise_ideas = extract_from_pairwise(pairwise_path)
    pointwise_ideas, pointwise_counts = extract_from_pointwise(pointwise_path)

    all_pass = True

    # Check 1: idea sets are identical
    if pairwise_ideas == pointwise_ideas:
        print(f"PASS  All {len(pairwise_ideas)} ideas match between pairwise and pointwise.")
    else:
        all_pass = False
        only_pairwise = pairwise_ideas - pointwise_ideas
        only_pointwise = pointwise_ideas - pairwise_ideas
        print(f"FAIL  Idea sets differ (pairwise: {len(pairwise_ideas)}, pointwise: {len(pointwise_ideas)})")
        if only_pairwise:
            print(f"\n  In pairwise only ({len(only_pairwise)}):")
            for key in sorted(only_pairwise, key=lambda x: x[0]):
                title = dict(key[1]).get("title", key[0][:80])
                print(f"    - {title}")
        if only_pointwise:
            print(f"\n  In pointwise only ({len(only_pointwise)}):")
            for key in sorted(only_pointwise, key=lambda x: x[0]):
                title = dict(key[1]).get("title", key[0][:80])
                print(f"    - {title}")

    # Check 2: no duplicates in pointwise
    dupes = {k: n for k, n in pointwise_counts.items() if n > 1}
    if not dupes:
        print(f"PASS  No duplicate ideas in pointwise ({len(pointwise_counts)} unique).")
    else:
        all_pass = False
        print(f"FAIL  {len(dupes)} duplicate idea(s) in pointwise:")
        for key, n in sorted(dupes.items(), key=lambda x: x[0][0]):
            title = dict(key[1]).get("title", key[0][:80])
            print(f"    x{n}  {title}")

    print()
    if all_pass:
        print("All checks PASSED.")
        sys.exit(0)
    else:
        print("One or more checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
