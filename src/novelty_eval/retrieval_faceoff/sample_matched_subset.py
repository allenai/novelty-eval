#!/usr/bin/env python3
"""sample_matched_subset.py — draw a matched, balanced, blocklist-clean subset of
ideas that exists identically in the pointwise and pairwise datasets.

The pointwise dataset (one idea + label per instance) and the pairwise dataset
(one POSITIVE + one NEGATIVE idea per pair) share the *same* underlying ideas.
This script samples whole PAIRS: each drawn pair contributes exactly one positive
and one negative pointwise instance, so the pointwise subset is automatically
balanced and the pairwise subset covers the identical items — the whole point of
"matching" the two framings.

Blocklisted papers (``benchmark_data/paper_blocklist.yaml``) are dropped *before*
sampling, at the pair level. Because a pair is only eligible when NEITHER of its
two component titles is blocklisted, the subset never contains a blocked paper and
never leaves an orphaned pairwise partner — strictly cleaner than the ablation
pipeline's post-hoc partner derivation.

Outputs (into --out-dir):
  manifest.json          — the join table (pair id ↔ pos/neg pointwise ids ↔ titles
                           ↔ area), the sampled id sets, seed, and blocklist audit.
  pointwise_subset.yaml  — the 2N sampled pointwise instances, ORIGINAL keys preserved.
  pairwise_subset.yaml   — the N sampled pairs, ORIGINAL keys preserved.
  ws_subset.yaml         — ready-to-run web_search_novelty_judge.py config.
  self_judge_subset.yaml — ready-to-run run_benchmark.py cached_self_judgement config.

Preserving the original problem ids matters: the subset scorer (subset_reports.py)
expresses the subset as an exclude set over those ids and the merge pipeline pairs
runs by them.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Allow `from novelty_eval...` imports when run as a script.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils import LOGGER as LOGGER  # reassigned to a file logger in main() once out_dir is known
from logging_utils import setup_logger
from novelty_eval.analysis.filtering import _load_paper_blocklist
from novelty_eval.retrieval.retrieval_common import normalize_idea_text


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _idea_text(s) -> str:
    """Join key for matching idea texts across the pointwise/pairwise datasets.

    Routes through `normalize_idea_text` so the join key is byte-identical to what
    `idea_identity` hashes — otherwise the join could pair ideas the content-addressed
    cache treats as distinct (or vice versa), silently breaking projection.
    """
    return normalize_idea_text(s)


def _build_pairs(pointwise: dict, pairwise: dict) -> Tuple[List[dict], List[str]]:
    """Join pairwise pairs to their pointwise counterparts by idea text.

    Returns (pairs, warnings). Each pair dict carries original ids as strings:
      {pair_id, pos_pw_id, neg_pw_id, pos_title, neg_title, context}
    Pairs whose components can't be resolved to a pointwise id are skipped (warned).
    """
    # idea text -> pointwise problem id (str). Text overlap is exact (verified).
    text_to_pw: Dict[str, str] = {}
    for pw_id, entry in pointwise.items():
        if isinstance(entry, dict) and "idea" in entry:
            text_to_pw[_idea_text(entry["idea"])] = str(pw_id)

    pairs: List[dict] = []
    warnings: List[str] = []
    for pair_id, inst in pairwise.items():
        if not isinstance(inst, dict):
            continue
        ideas = inst.get("ideas") or {}
        meta = inst.get("metadata") or {}
        context = inst.get("context", "")

        pos = neg = None  # (title, pointwise_id)
        for k, idea_meta in meta.items():
            if not isinstance(idea_meta, dict):
                continue
            title = idea_meta.get("title", "")
            itype = idea_meta.get("type")
            # idea text lives under the matching key in `ideas` (int or str key).
            text = _idea_text(ideas.get(k, ideas.get(str(k), ideas.get(_maybe_int(k)))))
            pw_id = text_to_pw.get(text)
            if itype == "POSITIVE":
                pos = (title, pw_id)
            elif itype == "NEGATIVE":
                neg = (title, pw_id)

        if not pos or not neg:
            warnings.append(f"pair {pair_id}: missing POSITIVE/NEGATIVE metadata — skipped")
            continue
        if pos[1] is None or neg[1] is None:
            warnings.append(f"pair {pair_id}: idea text not found in pointwise dataset — skipped")
            continue

        pairs.append({
            "pair_id": str(pair_id),
            "pos_pw_id": pos[1],
            "neg_pw_id": neg[1],
            "pos_title": pos[0],
            "neg_title": neg[0],
            "context": context,
        })
    return pairs, warnings


def _maybe_int(k):
    try:
        return int(k)
    except (ValueError, TypeError):
        return k


def _eligible_pairs(pairs: List[dict], blocked: set) -> Tuple[List[dict], List[dict]]:
    """Split pairs into (eligible, dropped) by the lower-cased title blocklist."""
    eligible, dropped = [], []
    for p in pairs:
        if p["pos_title"].lower() in blocked or p["neg_title"].lower() in blocked:
            dropped.append(p)
        else:
            eligible.append(p)
    return eligible, dropped


def _pair_sort_key(p: dict):
    pid = p["pair_id"]
    return int(pid) if pid.lstrip("-").isdigit() else pid


def _sample(pairs: List[dict], n: int, seed: int) -> List[dict]:
    """Deterministically sample n pairs (or all of them when n >= len(pairs))."""
    rng = random.Random(seed)
    picked = pairs if n >= len(pairs) else rng.sample(pairs, n)
    return sorted(picked, key=_pair_sort_key)


def _subset_by_ids(data: dict, keep_ids: set) -> dict:
    """Return {orig_key: entry} for entries whose str(key) is in keep_ids (types preserved)."""
    return {k: v for k, v in data.items() if str(k) in keep_ids}


def _dump_yaml(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=88)


def _write_ws_config(path: str, out_dir: str, pointwise_subset: str, engine: str,
                     cutoff: str, reasoning: str, top_k: int, min_relevance_score: float,
                     allowed_domains: List[str]) -> None:
    cfg = {
        "test_inputs": pointwise_subset,
        "output_file": os.path.join(out_dir, "web_search_master.json"),
        "content_addressed": True,
        "llm_engine": engine,
        "reasoning_effort": reasoning,
        "cutoff_date": cutoff,
        "top_k_candidates": top_k,
        "min_relevance_score": min_relevance_score,
        "allowed_domains": allowed_domains,
        "max_workers": 4,
        "chunk_size": 10,
    }
    header = (
        "# ws_subset.yaml — web_search_novelty_judge.py config (auto-generated by\n"
        "# sample_matched_subset.py). Content-addressed master store over the sampled\n"
        "# pointwise ideas. Run:\n"
        "#   python -m novelty_eval.retrieval.web_search_novelty_judge \\\n"
        "#       --config {cfg} --mode submit   # then --mode collect\n".format(cfg=path)
    )
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _write_self_judge_config(path: str, out_dir: str, pointwise_subset: str,
                             engine: str, cutoff: str) -> None:
    cfg = {
        "test_inputs": pointwise_subset,
        "retrieval_cache_file": os.path.join(out_dir, "cache_pointwise.json"),
        "output_file": "accuracy_report.txt",
        "output_dir": os.path.join(out_dir, "web_search_self_judge"),
        "n": 1,
        "test_mode": "pointwise",
        "judge_backend": "cached_self_judgement",
        "llm_engine": engine,
        "cutoff_date": cutoff,
        "use_batch_api": False,
    }
    header = (
        "# self_judge_subset.yaml — run_benchmark.py config (auto-generated). Scores the\n"
        "# web-search judge's OWN cached verdict (no LLM call). Requires cache_pointwise.json,\n"
        "# produced by project_retrieval_cache.py from the web_search master. Run:\n"
        "#   python -m novelty_eval.run_benchmark --config {cfg}\n".format(cfg=path)
    )
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pointwise", required=True, help="Pointwise instances YAML.")
    parser.add_argument("--pairwise", required=True, help="Pairwise instances YAML.")
    parser.add_argument("--out-dir", required=True, help="Where to write the subset + configs.")
    parser.add_argument("--n-pairs", type=int, default=25,
                        help="Number of pairs to sample (→ 2N balanced pointwise ideas). Default 25.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm-engine", default="gpt-5.5-pro",
                        help="Strong judge model for the web-search retrieval config.")
    parser.add_argument("--cutoff-date", default="2025-03-01",
                        help="Retrieval cutoff; MUST match every downstream step (it is part "
                             "of the content-addressed cache id).")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--top-k-candidates", type=int, default=5)
    parser.add_argument("--min-relevance-score", type=float, default=0.0,
                        help="Minimum model-returned relevance_score to keep a cited paper.")
    parser.add_argument("--allowed-domains", nargs="+", default=["arxiv.org"],
                        help="Domains the web_search tool is allowed to use.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    global LOGGER
    LOGGER = setup_logger(output_dir=args.out_dir, console_level="INFO")

    pointwise = _load_yaml(args.pointwise)
    pairwise = _load_yaml(args.pairwise)

    pairs, warnings = _build_pairs(pointwise, pairwise)
    for w in warnings:
        LOGGER.warning(w)

    blocked = _load_paper_blocklist()
    eligible, dropped = _eligible_pairs(pairs, blocked)

    if args.n_pairs > len(eligible):
        LOGGER.warning(f"requested {args.n_pairs} pairs but only {len(eligible)} eligible; "
                       f"using all eligible.")

    sampled = _sample(eligible, args.n_pairs, args.seed)

    keep_pairwise = {p["pair_id"] for p in sampled}
    keep_pointwise = {p["pos_pw_id"] for p in sampled} | {p["neg_pw_id"] for p in sampled}

    # Subset YAMLs — preserve original keys/types so downstream keying matches.
    pw_subset = _subset_by_ids(pointwise, keep_pointwise)
    pr_subset = _subset_by_ids(pairwise, keep_pairwise)
    pw_subset_path = os.path.join(args.out_dir, "pointwise_subset.yaml")
    pr_subset_path = os.path.join(args.out_dir, "pairwise_subset.yaml")
    _dump_yaml(pw_subset, pw_subset_path)
    _dump_yaml(pr_subset, pr_subset_path)

    # Sanity: balance.
    n_pos = sum(1 for k in keep_pointwise
                if str(pointwise.get(k, pointwise.get(_maybe_int(k), {})).get("label")) == "POSITIVE")
    n_neg = len(keep_pointwise) - n_pos

    manifest = {
        "seed": args.seed,
        "n_pairs_requested": args.n_pairs,
        "n_pairs_sampled": len(sampled),
        "cutoff_date": args.cutoff_date,
        "llm_engine": args.llm_engine,
        "sources": {
            "pointwise": os.path.abspath(args.pointwise),
            "pairwise": os.path.abspath(args.pairwise),
        },
        "blocklist": {
            "n_blocked_titles": len(blocked),
            "n_pairs_dropped": len(dropped),
            "dropped_pair_ids": sorted((p["pair_id"] for p in dropped),
                                       key=lambda x: int(x) if x.isdigit() else x),
        },
        "balance": {"positives": n_pos, "negatives": n_neg},
        "keep_pointwise_ids": sorted(keep_pointwise, key=lambda x: int(x) if x.lstrip("-").isdigit() else x),
        "keep_pairwise_ids": sorted(keep_pairwise, key=lambda x: int(x) if x.lstrip("-").isdigit() else x),
        "pairs": sampled,
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Materialize the two downstream configs with correct paths.
    ws_cfg = os.path.join(args.out_dir, "ws_subset.yaml")
    sj_cfg = os.path.join(args.out_dir, "self_judge_subset.yaml")
    _write_ws_config(ws_cfg, args.out_dir, pw_subset_path, args.llm_engine,
                     args.cutoff_date, args.reasoning_effort, args.top_k_candidates,
                     args.min_relevance_score, args.allowed_domains)
    _write_self_judge_config(sj_cfg, args.out_dir, pw_subset_path, args.llm_engine, args.cutoff_date)

    LOGGER.info(f"Sampled {len(sampled)} pairs → {len(keep_pointwise)} pointwise ideas "
                f"({n_pos} POS / {n_neg} NEG).")
    LOGGER.info(f"Eligible pool: {len(eligible)} pairs ({len(dropped)} dropped by blocklist "
                f"of {len(blocked)} titles).")
    LOGGER.info("Wrote:\n  %s\n  %s\n  %s\n  %s\n  %s",
                manifest_path, pw_subset_path, pr_subset_path, ws_cfg, sj_cfg)


if __name__ == "__main__":
    main()
