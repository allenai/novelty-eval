"""Sample novelty signals for annotation and write the blind form plus its key.

One random signal per idea, N ideas per label. The form shows only the area,
idea and signal; key.csv holds the model polarity and is not shown to the annotator.

Usage:
    python -m novelty_eval.signal_annotation.sample_signals
"""

import argparse
import csv
import json
import random
from pathlib import Path

import yaml

FORM_TEMPLATE = Path(__file__).with_name("annotation_form.html")


def sample(data: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for label in ("POSITIVE", "NEGATIVE"):
        idea_ids = sorted(k for k, v in data.items() if v["label"] == label)
        for idea_id in rng.sample(idea_ids, n):
            idea = data[idea_id]
            polarity = label.lower()
            items.append({
                "idea_id": idea_id,
                "model_polarity": polarity,
                "area": idea["context"],
                "idea": idea["idea"],
                "signal": rng.choice(idea["metadata"][f"{polarity}_signals"]),
            })
    rng.shuffle(items)
    for i, item in enumerate(items, 1):
        item["item_id"] = f"s{i:02d}"
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/human-only/pointwise.yaml"))
    parser.add_argument("--n", type=int, default=25, help="signals per polarity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("output/signal_annotation"))
    args = parser.parse_args()

    items = sample(yaml.safe_load(args.data.read_text()), args.n, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "key.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "idea_id", "model_polarity", "signal"])
        writer.writeheader()
        writer.writerows({k: it[k] for k in writer.fieldnames} for it in items)

    shown = [{k: it[k] for k in ("item_id", "area", "idea", "signal")} for it in items]
    payload = json.dumps(shown, ensure_ascii=False).replace("</", "<\\/")
    form = FORM_TEMPLATE.read_text().replace("__ITEMS__", payload)
    (args.out / "form.html").write_text(form)
    print(f"Wrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
