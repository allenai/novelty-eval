#!/usr/bin/env python3
"""Look up the OpenReview forum ID of every human paper in data/ and write them to a CSV.

The released data refers to papers by forum ID, so authors can find (and ask us to
remove) their paper without any author names being stored. Matching is by title,
which the YAML copies verbatim from OpenReview; every title must match exactly one
ICLR 2026 submission.

Needs OpenReview credentials (anonymous API access is behind a browser challenge):

    export OPENREVIEW_USERNAME=... OPENREVIEW_PASSWORD=...
    python scripts/hf/fetch_forum_ids.py
"""

import argparse
import csv
import os
from pathlib import Path

import openreview
import yaml

VENUE_IDS = [
    "ICLR.cc/2026/Conference",
    "ICLR.cc/2026/Conference/Rejected_Submission",
    "ICLR.cc/2026/Conference/Withdrawn_Submission",
]

# Titles changed on OpenReview after our data was fetched (title in data/ -> current title).
RENAMED = {
    "Native Adaptive Solution Expansion for Diffusion-based Combinatorial Optimization":
        "NExCO: Native Solution Expansion for Diffusion-based Combinatorial Optimization",
}


def normalize(title: str) -> str:
    return " ".join(RENAMED.get(title, title).split()).lower()


def human_titles(data_dir: Path) -> set[str]:
    """Generated ideas have no reviewer rating; everything else is an ICLR paper."""
    titles = set()
    for path in data_dir.rglob("*.yaml"):
        for instance in yaml.safe_load(path.read_text()).values():
            metadata = instance["metadata"]
            for m in [metadata] if "idea" in instance else metadata.values():
                if m["rating"] != "N/A":
                    titles.add(m["title"])
    return titles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/openreview_forum_ids.csv"))
    args = parser.parse_args()

    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=os.getenv("OPENREVIEW_USERNAME"),
        password=os.getenv("OPENREVIEW_PASSWORD"),
    )
    forums: dict[str, set[str]] = {}
    for venue_id in VENUE_IDS:
        for note in client.get_all_notes(content={"venueid": venue_id}):
            forums.setdefault(normalize(note.content["title"]["value"]), set()).add(note.forum)

    titles = sorted(human_titles(args.data_dir))
    missing = [t for t in titles if normalize(t) not in forums]
    ambiguous = [t for t in titles if len(forums.get(normalize(t), ())) > 1]
    assert not missing and not ambiguous, f"unmatched: {missing}\nambiguous: {ambiguous}"

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "forum_id"])
        for title in titles:
            writer.writerow([title, next(iter(forums[normalize(title)]))])
    print(f"{len(titles)} papers -> {args.out}")


if __name__ == "__main__":
    main()
