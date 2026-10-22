#!/usr/bin/env python3
"""project_retrieval_cache.py — re-key a content-addressed retrieval master store
into a dataset-specific retrieval cache that run_benchmark.py can consume.

The web-search judge (``web_search_novelty_judge.py`` with ``content_addressed: true``)
writes a *master store* keyed by ``idea_identity(idea_text, cutoff_date)`` — one entry
per distinct idea text, independent of any dataset's positional ``problem_id``. This
script reads such a master plus a target benchmark YAML and emits a
``retrieval_cache.json`` keyed the way the sweep expects:

  pointwise -> flat:    cache[problem_id]           = {self_judgement, candidates, related_work, ...}
  pairwise  -> nested:  cache[problem_id][idea_key] = {candidates, related_work, ...}

Because every payload is recovered by idea-text hash, one master serves the pairwise
dataset and every pointwise dataset that shares its positives — only the keying/shape
differs. Ideas absent from the master are reported (not invented) so you can run the
judge for just those (e.g. a newly added negative set):

    # 1. build/extend the master store (only unseen idea texts get retrieved)
    python web_search_novelty_judge.py --config ws.yaml --mode submit   # then --mode collect
    #    (ws.yaml sets content_addressed: true and output_file: <master.json>)

    # 2. project the master onto each dataset's own keys/shape
    python project_retrieval_cache.py \
        --master   output/.../web_search_master.json \
        --test-inputs output/.../iclr_pointwise_instances.yaml \
        --output   output/.../retrieval_cache.json \
        --cutoff-date 2025-03-01

IMPORTANT: ``--cutoff-date`` must match the value the judge ran with, because it is
part of the content-addressed id. A mismatch makes every lookup miss.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from .retrieval_common import idea_identity, load_retrieval_cache, save_retrieval_cache
except ImportError:
    from retrieval_common import idea_identity, load_retrieval_cache, save_retrieval_cache


def detect_mode(inputs: Dict) -> str:
    """Infer 'pairwise' (entries carry `ideas`) or 'pointwise' (entries carry `idea`)."""
    for v in inputs.values():
        if isinstance(v, dict):
            if 'ideas' in v:
                return 'pairwise'
            if 'idea' in v:
                return 'pointwise'
    raise ValueError("Could not detect mode: no instance has an 'ideas' or 'idea' field.")


def _iter_dataset_ideas(inputs: Dict, mode: str):
    """Yield (problem_id, idea_key, idea_text, title) for every idea in the dataset.

    For pointwise, idea_key is None (the cache entry is flat). For pairwise, idea_key is
    the per-pair key ('0'/'1') the sweep looks up via cache[problem_id][idea_key].
    """
    for problem_id, data in inputs.items():
        if not isinstance(data, dict):
            continue
        if mode == 'pointwise':
            if 'idea' not in data:
                continue
            title = (data.get('metadata') or {}).get('title') if isinstance(data.get('metadata'), dict) else None
            yield str(problem_id), None, data['idea'], title
        else:
            ideas = data.get('ideas') or {}
            meta = data.get('metadata') or {}
            for idea_key, text in ideas.items():
                title = None
                if isinstance(meta, dict):
                    per = meta.get(idea_key) if idea_key in meta else meta.get(str(idea_key))
                    if isinstance(per, dict):
                        title = per.get('title')
                yield str(problem_id), str(idea_key), text, title


def project(master: Dict, inputs: Dict, mode: str, cutoff_date: Optional[str]) -> Tuple[Dict, int, int, List[Dict]]:
    """Re-key `master` (idea_identity -> payload) onto the dataset's problem_ids/shape."""
    out: Dict = {}
    n_ideas = n_hit = 0
    misses: List[Dict] = []
    for problem_id, idea_key, text, title in _iter_dataset_ideas(inputs, mode):
        n_ideas += 1
        key = idea_identity(text, cutoff_date)
        entry = master.get(key)
        if not isinstance(entry, dict):
            misses.append({"problem_id": problem_id, "idea_key": idea_key,
                           "idea_hash": key, "title": title})
            continue
        n_hit += 1
        if mode == 'pointwise':
            out[problem_id] = entry
        else:
            out.setdefault(problem_id, {})[idea_key] = entry
    return out, n_ideas, n_hit, misses


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--master', required=True, help="Content-addressed master store (idea_identity -> payload).")
    parser.add_argument('--test-inputs', required=True, help="Target dataset YAML to project onto.")
    parser.add_argument('--output', required=True, help="Where to write the dataset-keyed retrieval_cache.json.")
    parser.add_argument('--mode', choices=['auto', 'pointwise', 'pairwise'], default='auto')
    parser.add_argument('--cutoff-date', default='2025-03-01',
                        help="Must match the cutoff_date the judge ran with (default: 2025-03-01).")
    parser.add_argument('--max-misses-shown', type=int, default=15)
    args = parser.parse_args()

    master = load_retrieval_cache(args.master)
    if not master:
        print(f"ERROR: master store at {args.master} is empty or missing.", file=sys.stderr)
        sys.exit(1)
    with open(args.test_inputs) as f:
        inputs = yaml.safe_load(f) or {}

    mode = detect_mode(inputs) if args.mode == 'auto' else args.mode
    out, n_ideas, n_hit, misses = project(master, inputs, mode, args.cutoff_date)

    save_retrieval_cache(args.output, out)

    print(f"[project] mode={mode}  master_entries={len(master)}  "
          f"dataset_ideas={n_ideas}  hit={n_hit}  miss={len(misses)}")
    print(f"[project] wrote {len(out)} instance key(s) -> {args.output}")
    if misses:
        print(f"[project] {len(misses)} idea(s) NOT in master (run the judge for these). "
              f"Showing up to {args.max_misses_shown}:")
        for m in misses[:args.max_misses_shown]:
            loc = m['problem_id'] + (f"/{m['idea_key']}" if m['idea_key'] is not None else "")
            print(f"    - {loc}  hash={m['idea_hash']}  title={m['title']!r}")
        if len(misses) > args.max_misses_shown:
            print(f"    ... and {len(misses) - args.max_misses_shown} more.")


if __name__ == '__main__':
    main()
