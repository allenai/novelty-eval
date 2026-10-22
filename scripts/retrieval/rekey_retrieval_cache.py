#!/usr/bin/env python3
"""
Re-key a retrieval cache to align with a new (manipulated) test instances YAML.

When test instances are re-ordered / re-indexed, the existing retrieval cache
entries (keyed by old problem_id → old idea_id) no longer match the new YAML's
(problem_id → idea_id) mapping.  This script:

  1. Loads the OLD cache and the NEW YAML.
  2. For every problem in the new YAML, uses the metadata 'type' field to
     identify which idea is POSITIVE, then locates that idea's text in the
     old cache.
  3. Writes a NEW cache file keyed correctly for the new YAML:
       new_cache[str(new_problem_id)][str(positive_idea_id)] = <remapped entry>
  4. Copies retrieval debug .md files (retrieval_debug/{pid}/{iid}_retrieval.md)
     from the old layout into the new layout under the correct keys.
  5. Regenerates cache_status.md for the new cache / new YAML pair.
  6. Leaves negative ideas absent — the retrieval script will fill them in on
     the next run, skipping anything already present.

Usage:
    python3 scripts/rekey_retrieval_cache.py \\
        --old-cache output/iclr_test_instances/pairwise_data/20260318_155708/manipulated/retrieval_cache.json \\
        --new-yaml  data/human-plus-generated/pairwise.yaml \\
        --output    output/iclr_test_instances/pairwise_data/20260517_171954/manipulated/retrieval_cache.json
"""

import argparse
import json
import os
import shutil
import yaml
from pathlib import Path
from datetime import datetime


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_text_index(old_cache: dict, snippet_len: int = 100) -> dict:
    """Return mapping: first-N-chars-of-text → (old_prob_id, old_idea_id, entry)."""
    index: dict = {}
    for prob_id, ideas in old_cache.items():
        for idea_id, entry in ideas.items():
            text = entry.get("text", "")
            key = text[:snippet_len]
            if key:
                if key in index:
                    old_pid, old_iid, _ = index[key]
                    print(f"  WARNING: duplicate text snippet in old cache "
                          f"(existing: prob={old_pid}, idea={old_iid}; "
                          f"new: prob={prob_id}, idea={idea_id}). Overwriting.")
                index[key] = (prob_id, idea_id, entry)
    return index


def find_positive_idea(data: dict) -> tuple | None:
    """Return (idea_key, idea_text) for the POSITIVE idea in a problem, or None."""
    ideas = data.get("ideas", {})
    metadata = data.get("metadata", {})
    for idea_key, text in ideas.items():
        itype = metadata.get(idea_key, {}).get("type", "")
        if itype == "POSITIVE":
            return idea_key, str(text)
    return None


def copy_debug_file(old_debug_root: Path, new_debug_root: Path,
                    old_pid: str, old_iid: str,
                    new_pid: str, new_iid: str) -> bool:
    """Copy retrieval_debug/{pid}/{iid}_retrieval.md from old layout to new layout."""
    src = old_debug_root / old_pid / f"{old_iid}_retrieval.md"
    dst = new_debug_root / new_pid / f"{new_iid}_retrieval.md"
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def generate_cache_status(new_cache: dict, new_yaml: dict,
                           output_file: Path, new_yaml_path: str) -> None:
    """Write a cache_status.md next to the output cache file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = output_file.parent / "cache_status.md"

    total_problems = len(new_yaml)
    total_ideas_in_input = 0
    total_ideas_cached = 0
    total_ideas_missing = 0
    total_ideas_empty = 0
    total_candidates = 0
    candidate_counts: list[int] = []
    per_problem_lines: list[str] = []

    for pid, data in new_yaml.items():
        ideas = data.get("ideas", {})
        num_ideas = len(ideas)
        total_ideas_in_input += num_ideas

        cached_entry = new_cache.get(str(pid), {})
        cached_keys, missing_keys, empty_keys = [], [], []

        for key in ideas:
            str_key = str(key)
            if str_key in cached_entry:
                entry = cached_entry[str_key]
                cands = entry.get("candidates", [])
                if cands:
                    cached_keys.append(str_key)
                    n = len(cands)
                    total_candidates += n
                    candidate_counts.append(n)
                else:
                    empty_keys.append(str_key)
            else:
                missing_keys.append(str_key)

        total_ideas_cached += len(cached_keys)
        total_ideas_missing += len(missing_keys)
        total_ideas_empty += len(empty_keys)

        if not missing_keys and not empty_keys:
            status = "✅"
        elif cached_keys:
            status = "⚠️"
        else:
            status = "❌"

        per_problem_lines.append(
            f"| {status} `{pid}` | {num_ideas} | {len(cached_keys)} | {len(missing_keys)} | {len(empty_keys)} |"
        )

    coverage_pct = (total_ideas_cached / total_ideas_in_input * 100) if total_ideas_in_input else 0
    problems_fully_covered = sum(
        1 for pid, data in new_yaml.items()
        if all(
            str(k) in new_cache.get(str(pid), {}) and
            new_cache[str(pid)].get(str(k), {}).get("candidates")
            for k in data.get("ideas", {})
        )
    )
    avg_cands = (total_candidates / total_ideas_cached) if total_ideas_cached else 0
    min_cands = min(candidate_counts) if candidate_counts else 0
    max_cands = max(candidate_counts) if candidate_counts else 0

    lines = [
        "# 📦 Cache Status Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Cache file:** `{output_file.resolve()}`  ",
        f"**Input file:** `{os.path.abspath(new_yaml_path)}`  ",
        f"**Note:** Generated by rekey_retrieval_cache.py — positives remapped from old cache, negatives pending retrieval.",
        "",
        "---",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total problems in input | {total_problems} |",
        f"| Problems fully covered | {problems_fully_covered} / {total_problems} |",
        f"| Total ideas in input | {total_ideas_in_input} |",
        f"| Ideas cached (with candidates) | {total_ideas_cached} |",
        f"| Ideas missing from cache | {total_ideas_missing} |",
        f"| Ideas cached but 0 candidates | {total_ideas_empty} |",
        f"| **Coverage** | **{coverage_pct:.1f}%** |",
        "",
        "## Candidate Statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total candidates across all ideas | {total_candidates} |",
        f"| Avg candidates per idea | {avg_cands:.1f} |",
        f"| Min candidates per idea | {min_cands} |",
        f"| Max candidates per idea | {max_cands} |",
        "",
        "## Per-Problem Breakdown",
        "",
        "| Problem | Total Ideas | Cached | Missing | Empty (0 candidates) |",
        "|---|---|---|---|---|",
    ]
    lines.extend(per_problem_lines)
    lines.extend([
        "",
        "**Legend:** ✅ = fully covered | ⚠️ = partially covered | ❌ = no ideas cached",
    ])

    report_path.write_text("\n".join(lines) + "\n")
    print(f"  cache_status.md → {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-key a retrieval cache to match a new test-instances YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--old-cache", required=True,
                        help="Path to the existing retrieval_cache.json (source of positive data).")
    parser.add_argument("--new-yaml", required=True,
                        help="Path to the new iclr_test_instances.yaml.")
    parser.add_argument("--output", required=True,
                        help="Where to write the re-keyed cache JSON.")
    parser.add_argument("--snippet-len", type=int, default=100,
                        help="Number of leading characters used for text matching (default: 100).")
    parser.add_argument("--no-debug-copy", action="store_true",
                        help="Skip copying retrieval debug .md files.")
    parser.add_argument("--no-cache-status", action="store_true",
                        help="Skip regenerating cache_status.md.")
    args = parser.parse_args()

    output_path = Path(args.output)
    old_debug_root = Path(args.old_cache).parent / "retrieval_debug"
    new_debug_root = output_path.parent / "retrieval_debug"

    # ── Load inputs ────────────────────────────────────────────────────────────
    print(f"Loading old cache : {args.old_cache}")
    with open(args.old_cache) as f:
        old_cache = json.load(f)

    print(f"Loading new YAML  : {args.new_yaml}")
    with open(args.new_yaml) as f:
        new_yaml = yaml.safe_load(f)

    # ── Build text → (old_pid, old_iid, entry) lookup ─────────────────────────
    text_index = build_text_index(old_cache, snippet_len=args.snippet_len)
    old_problems = len(old_cache)
    old_ideas = sum(len(v) for v in old_cache.values())
    print(f"Old cache         : {old_problems} problems, {old_ideas} cached ideas, "
          f"{len(text_index)} unique text snippets indexed.")

    # ── Re-key cache + copy debug files ───────────────────────────────────────
    new_cache: dict = {}
    matched = 0
    unmatched: list = []
    no_positive_metadata: list = []
    debug_copied = 0
    debug_missing = 0

    for new_pid, data in new_yaml.items():
        str_new_pid = str(new_pid)
        new_cache[str_new_pid] = {}

        result = find_positive_idea(data)
        if result is None:
            no_positive_metadata.append(new_pid)
            continue

        pos_key, pos_text = result
        str_pos_key = str(pos_key)
        snippet = pos_text[:args.snippet_len]

        if snippet not in text_index:
            unmatched.append(new_pid)
            continue

        old_pid, old_iid, entry = text_index[snippet]
        new_cache[str_new_pid][str_pos_key] = entry
        matched += 1

        # Copy debug file
        if not args.no_debug_copy:
            copied = copy_debug_file(
                old_debug_root, new_debug_root,
                old_pid, old_iid,
                str_new_pid, str_pos_key,
            )
            if copied:
                debug_copied += 1
            else:
                debug_missing += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nRe-keying results:")
    print(f"  Positives remapped successfully   : {matched} / {len(new_yaml)}")
    if unmatched:
        print(f"  Positives NOT found in old cache  : {len(unmatched)}")
        print(f"  Unmatched problem IDs             : {unmatched}")
    if no_positive_metadata:
        print(f"  Problems with no POSITIVE metadata: {no_positive_metadata}")
    if not args.no_debug_copy:
        print(f"  Debug .md files copied            : {debug_copied}")
        if debug_missing:
            print(f"  Debug .md files not found (src)   : {debug_missing}")

    # ── Save cache ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(new_cache, f, indent=2)
    print(f"\nNew cache written to: {output_path}")

    # ── Regenerate cache_status.md ────────────────────────────────────────────
    if not args.no_cache_status:
        print("Generating cache_status.md ...")
        generate_cache_status(new_cache, new_yaml, output_path, args.new_yaml)

    print()
    print("Next step: run retrieve_candidates.py on the new YAML using this cache as output_file.")
    print("  → It will SKIP positive ideas (already cached under their correct keys).")
    print(f"  → It will RETRIEVE the negatives for all {len(new_yaml)} problems.")


if __name__ == "__main__":
    main()
