#!/usr/bin/env python3
"""Check that every paper in the release, and its official reviews, are public on OpenReview.

Reads the forum IDs from data/openreview_forum_ids.csv and checks that the submission
and every official review in its forum list 'everyone' among their readers. Fetches
whole venues with their replies (a few dozen requests), staying under OpenReview's
rate limit of 500 requests per hour.

Needs OpenReview credentials (anonymous API access is behind a browser challenge):

    export OPENREVIEW_USERNAME=... OPENREVIEW_PASSWORD=...
    python scripts/hf/check_openreview_visibility.py
"""

import csv
import os

import openreview

VENUE_IDS = [
    "ICLR.cc/2026/Conference",
    "ICLR.cc/2026/Conference/Rejected_Submission",
    "ICLR.cc/2026/Conference/Withdrawn_Submission",
]

client = openreview.api.OpenReviewClient(
    baseurl="https://api2.openreview.net",
    username=os.getenv("OPENREVIEW_USERNAME"),
    password=os.getenv("OPENREVIEW_PASSWORD"),
)

with open("data/openreview_forum_ids.csv") as f:
    forum_ids = {row["forum_id"] for row in csv.DictReader(f)}

found, n_reviews, not_public = set(), 0, []
for venue_id in VENUE_IDS:
    for note in client.get_all_notes(content={"venueid": venue_id}, details="replies"):
        if note.id not in forum_ids:
            continue
        found.add(note.id)
        if "everyone" not in note.readers:
            not_public.append((note.id, "submission"))
        for reply in note.details["replies"]:
            if any("Official_Review" in i for i in reply["invitations"]):
                n_reviews += 1
                if "everyone" not in reply["readers"]:
                    not_public.append((note.id, reply["id"]))

assert found == forum_ids, f"submissions not found: {sorted(forum_ids - found)}"
print(f"Checked {len(found)} submissions and {n_reviews} official reviews.")
print(f"NOT PUBLIC: {not_public}" if not_public else "All are readable by everyone.")
