#!/usr/bin/env python3
"""Validates consistency of baseline test data.

Checks:
  1. Positive ideas are identical across iclr-vs-generated and iclr-vs-iclr
     (both within each track's pairwise/pointwise files and across tracks).
  2. Negative ideas are identical within each track (pairwise vs pointwise),
     but may legitimately differ across tracks.
  3. No idea appears more than once in a pointwise file (duplicates are fine in pairwise).
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


def extract_from_pairwise(path: Path) -> tuple[set, set]:
    with open(path) as f:
        data = yaml.safe_load(f)
    positives, negatives = set(), set()
    for instance in data.values():
        for idx, text in instance["ideas"].items():
            m = instance["metadata"][idx]
            key = idea_key(text, m)
            if m["type"] == "POSITIVE":
                positives.add(key)
            elif m["type"] == "NEGATIVE":
                negatives.add(key)
    return positives, negatives


def extract_from_pointwise(path: Path) -> tuple[set, set, dict]:
    """Returns (positives, negatives, counts) where counts maps idea_key -> int."""
    with open(path) as f:
        data = yaml.safe_load(f)
    positives, negatives = set(), set()
    counts: dict = {}
    for instance in data.values():
        key = idea_key(instance["idea"], instance["metadata"])
        counts[key] = counts.get(key, 0) + 1
        if instance["label"] == "POSITIVE":
            positives.add(key)
        elif instance["label"] == "NEGATIVE":
            negatives.add(key)
    return positives, negatives, counts


def check_no_duplicates(label: str, counts: dict) -> bool:
    dupes = {k: n for k, n in counts.items() if n > 1}
    if not dupes:
        print(f"  PASS  no duplicates in {label}  ({len(counts)} ideas)")
        return True
    print(f"  FAIL  duplicates found in {label}:")
    for key, n in sorted(dupes.items(), key=lambda x: x[0][0]):
        title = dict(key[1]).get("title", key[0][:60])
        print(f"    x{n}  {title}")
    return False


def compare(label_a: str, set_a: set, label_b: str, set_b: set) -> bool:
    if set_a == set_b:
        print(f"  PASS  {label_a} == {label_b}  ({len(set_a)} ideas)")
        return True

    only_a = set_a - set_b
    only_b = set_b - set_a
    print(f"  FAIL  {label_a} != {label_b}")
    if only_a:
        print(f"    Only in '{label_a}' ({len(only_a)}):")
        for key in sorted(only_a, key=lambda x: x[0]):
            title = dict(key[1]).get("title", key[0][:60])
            print(f"      - {title}")
    if only_b:
        print(f"    Only in '{label_b}' ({len(only_b)}):")
        for key in sorted(only_b, key=lambda x: x[0]):
            title = dict(key[1]).get("title", key[0][:60])
            print(f"      - {title}")
    return False


def find_file(track_dir: Path, pattern: str) -> Path:
    matches = list(track_dir.glob(pattern))
    if not matches:
        sys.exit(f"ERROR: no file matching {track_dir / pattern}")
    if len(matches) > 1:
        print(f"WARNING: multiple matches for {pattern}, using most recent")
        matches.sort()
    return matches[-1]


def main():
    base = Path("output/novelty_benchmark_data")
    gen_dir = base / "human_prompted_llm"
    iclr_dir = base / "human-only"

    gen_pairwise = find_file(gen_dir, "pairwise/*/iclr_test_instances.yaml")
    gen_pointwise = find_file(gen_dir, "pointwise/*/iclr_pointwise_instances.yaml")
    iclr_pairwise = find_file(iclr_dir, "pairwise/*/iclr_test_instances.yaml")
    iclr_pointwise = find_file(iclr_dir, "pointwise/*/iclr_pointwise_instances.yaml")

    print("Files:")
    for label, path in [
        ("human_prompted_llm / pairwise ", gen_pairwise),
        ("human_prompted_llm / pointwise", gen_pointwise),
        ("human-only         / pairwise ", iclr_pairwise),
        ("human-only         / pointwise", iclr_pointwise),
    ]:
        print(f"  {label}  {path}")
    print()

    gen_pair_pos, gen_pair_neg = extract_from_pairwise(gen_pairwise)
    gen_pt_pos, gen_pt_neg, gen_pt_counts = extract_from_pointwise(gen_pointwise)
    iclr_pair_pos, iclr_pair_neg = extract_from_pairwise(iclr_pairwise)
    iclr_pt_pos, iclr_pt_neg, iclr_pt_counts = extract_from_pointwise(iclr_pointwise)

    all_pass = True

    print("=== Check 1: Positive ideas identical across all four files ===")
    all_pass &= compare("llm-pairwise positives", gen_pair_pos, "llm-pointwise positives", gen_pt_pos)
    all_pass &= compare("human-pairwise positives", iclr_pair_pos, "human-pointwise positives", iclr_pt_pos)
    all_pass &= compare("llm positives", gen_pair_pos, "human positives", iclr_pair_pos)
    print()

    print("=== Check 2: Negative ideas identical within each track (pairwise vs pointwise) ===")
    all_pass &= compare("llm-pairwise negatives", gen_pair_neg, "llm-pointwise negatives", gen_pt_neg)
    all_pass &= compare("human-pairwise negatives", iclr_pair_neg, "human-pointwise negatives", iclr_pt_neg)
    print()

    print("=== Check 3: No duplicate ideas in pointwise files ===")
    all_pass &= check_no_duplicates("llm-pointwise", gen_pt_counts)
    all_pass &= check_no_duplicates("human-pointwise", iclr_pt_counts)
    print()

    overlap = len(gen_pair_neg & iclr_pair_neg)
    print("=== Info: Cross-track negatives (allowed to differ) ===")
    print(f"  human_prompted_llm negatives : {len(gen_pair_neg)}")
    print(f"  human-only negatives         : {len(iclr_pair_neg)}")
    print(f"  Shared                       : {overlap}")
    print()

    if all_pass:
        print("All checks PASSED.")
        sys.exit(0)
    else:
        print("One or more checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
