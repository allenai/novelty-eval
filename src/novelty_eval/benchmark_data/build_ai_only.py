#!/usr/bin/env python3
"""Build a pairwise AI-only dataset: a stronger model's idea vs. a weaker model's idea.

Both inputs must be pairwise datasets over the same instances whose negative slot
holds a generated idea (e.g. data/human-plus-generated/backbone-*/pairwise.yaml).
The strong model's idea takes the positive slot, the weak model's the negative one;
instance ids, contexts and slot positions are kept.

Usage:
    python src/novelty_eval/benchmark_data/build_ai_only.py \
        --strong data/human-plus-generated/backbone-opus-4-5/pairwise.yaml --strong-model claude-opus-4-5 \
        --weak data/human-plus-generated/pairwise.yaml --weak-model claude-sonnet-4-5 \
        --out data/ai-only/claude/pairwise.yaml
"""

import argparse
from pathlib import Path

import yaml


def generated_idea(instance: dict, model: str, label: str) -> tuple[str, dict]:
    """Return the instance's generated (negative-slot) idea, relabelled."""
    (idx,) = {0, 1} - set(instance["expected_winners"])
    # Prefix the title so the two ideas of an instance stay distinct.
    meta = {**instance["metadata"][idx], "type": label, "generator": model,
            "title": f"{model}/{instance['metadata'][idx]['title']}"}
    return instance["ideas"][idx], meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strong", required=True, help="Dataset with the strong model's ideas.")
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--weak", required=True, help="Dataset with the weak model's ideas.")
    parser.add_argument("--weak-model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    strong = yaml.safe_load(open(args.strong))
    weak = yaml.safe_load(open(args.weak))
    assert strong.keys() == weak.keys(), "datasets cover different instances"

    out = {}
    for inst_id, s in strong.items():
        w = weak[inst_id]
        assert s["context"] == w["context"], f"context mismatch in instance {inst_id}"
        pos = s["expected_winners"][0]
        neg = 1 - pos
        out[inst_id] = {"context": s["context"], "expected_winners": [pos], "ideas": {}, "metadata": {}}
        for slot, src, model, label in [(pos, s, args.strong_model, "POSITIVE"),
                                        (neg, w, args.weak_model, "NEGATIVE")]:
            out[inst_id]["ideas"][slot], out[inst_id]["metadata"][slot] = generated_idea(src, model, label)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.dump(out, f, sort_keys=True, allow_unicode=True, width=float("inf"))
    print(f"Wrote {len(out)} instances to {args.out}")


if __name__ == "__main__":
    main()
