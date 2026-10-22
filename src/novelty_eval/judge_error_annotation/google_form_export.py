"""Export one annotator's batch as a Google Form, for annotating on a phone.

The HTML page saves answers to a file the annotator picks, which needs an API no
mobile browser has; the xlsx needs a spreadsheet app. This route needs neither:
the annotator taps through a form and the answers live in Google's copy, so
nothing rides on browser storage or on a file surviving the trip back.

Google exposes no API for creating a form from outside, so this writes an Apps
Script the batch owner pastes into script.google.com and runs once. The script
is the export; the form it builds is disposable and can be rebuilt from it.

The reverse direction lives here too: `--responses` folds the form's answers
back into the same `item_id,your_choice,notes` file the page exports, so both
routes hand the batch owner the same artifact. It takes either a downloaded
response CSV or the URL of the form's response spreadsheet, which saves a
download step when the sheet is readable by link.

Presentation-only, like annotation_page.py: it reads a rendered page and shows
the four whitelisted fields, so nothing that would de-blind an annotator can
reach the form even if the source page is regenerated with richer rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from novelty_eval.judge_error_annotation.annotation_page import VISIBLE_FIELDS

# How an item is tagged in a question title. Every question a response CSV must
# be matched on carries one, because the CSV names its columns by question title
# and nothing else in a response ties an answer to a row.
ITEM_TAG = "[item {item_id}]"
CHOICE_TITLE = "Which idea is more novel? " + ITEM_TAG
NOTES_TITLE = "Notes " + ITEM_TAG
_TAGGED = re.compile(r"\[item ([^\]]+)\]\s*$")

ANSWER_HEADERS = ["item_id", "your_choice", "notes"]

# A form's answers live in a Google Sheet, and Sheets will hand out any one tab
# as CSV. The id says which spreadsheet, the gid which tab: a form's responses
# are usually its own tab, so a URL without a gid would export the wrong one.
_SHEET_ID = re.compile(r"/spreadsheets/d/([A-Za-z0-9_-]+)")
_SHEET_GID = re.compile(r"[#?&]gid=([0-9]+)")
_SHEET_EXPORT = "https://docs.google.com/spreadsheets/d/{id}/export?format=csv&gid={gid}"


def _choice_label(choice: str) -> str:
    """The wording an annotator taps. Matches the page's buttons."""
    return f"Idea {choice} is more novel" if choice in ("A", "B") else str(choice).capitalize()


def read_page_rows(path: Path) -> tuple[str, list[dict], list[str]]:
    """Pull the annotator, rows and choices back out of a rendered annotation page.

    The page is the source of truth for what this annotator was actually shown:
    it carries their row order and their A/B blinding, which the batch config
    alone does not determine.
    """
    html = path.read_text(encoding="utf-8")
    annotator = re.search(r'data-annotator="([^"]*)"', html)
    if not annotator:
        raise SystemExit(f"{path} is not an annotation page (no data-annotator).")
    blocks = {}
    for name in ("items", "choices"):
        found = re.search(rf'<script type="application/json" id="{name}">(.*?)</script>', html, re.S)
        if not found:
            raise SystemExit(f"{path} is not an annotation page (no embedded {name}).")
        blocks[name] = json.loads(found.group(1).replace("\\u003c", "<"))
    rows = [{k: row.get(k, "") for k in VISIBLE_FIELDS} for row in blocks["items"]]
    return annotator.group(1), rows, [str(c) for c in blocks["choices"]]


def build_apps_script(
    annotator: str,
    rows: Sequence[dict],
    choices: Sequence[str],
    instructions: Sequence[str],
    require_answers: bool = False,
    pairs_per_page: int = 1,
) -> str:
    """The Apps Script that builds this annotator's form.

    With `require_answers` the form refuses to submit until every pair is
    answered, which also means nothing reaches the responses CSV until the whole
    batch is done. Left off, an annotator can submit part-way and reopen their
    response to carry on, and each submit updates the CSV.
    """
    items = [
        {
            "item_id": str(r["item_id"]),
            "area": str(r.get("area") or ""),
            "A_text": str(r["A_text"]),
            "B_text": str(r["B_text"]),
        }
        for r in rows
    ]
    intro = "\n".join(line for line in instructions if line.strip() and not line.strip().startswith("Judge novelty"))
    payload = {
        "annotator": annotator,
        "title": f"Novelty annotation — {annotator}",
        "intro": intro,
        "labels": [_choice_label(c) for c in choices],
        "unclear": _choice_label(choices[-1]),
        "required": bool(require_answers),
        "pairsPerPage": max(0, int(pairs_per_page)),
        "items": items,
    }
    # ensure_ascii keeps the file free of the line separators that break a JS
    # literal, and of anything the Apps Script editor might re-encode.
    blob = json.dumps(payload, ensure_ascii=True, indent=2)
    # The idea texts must reach the form exactly as the page shows them. Read the
    # blob back and compare, so a change to how it is built can never quietly
    # alter what an annotator reads.
    for shown, source in zip(json.loads(blob)["items"], rows):
        for field in ("item_id", "area", "A_text", "B_text"):
            if shown[field] != str(source[field] if field != "area" else (source.get("area") or "")):
                raise SystemExit(f"item {source['item_id']}: {field} changed while building the script.")
    js_id = "' + it.item_id + '"
    choice_title = CHOICE_TITLE.replace("{item_id}", js_id)
    notes_title = NOTES_TITLE.replace("{item_id}", js_id)
    return f"""/**
 * Builds the novelty-annotation form for {annotator}.
 *
 * Generated by google_form_export.py -- edit that, not this file.
 *
 * To use:
 *   1. Open script.google.com and start a new project.
 *   2. Replace the editor's contents with this whole file and save.
 *   3. Run buildForm(), and grant the permission it asks for.
 *   4. The execution log prints two links. Send the annotator the "Fill in" one.
 *
 * Rerunning buildForm() creates a second, independent form; it never edits the
 * first. Delete the old one first if that is not what you want.
 */

var BATCH = {blob};

function buildForm() {{
  var form = FormApp.create(BATCH.title);
  form.setDescription(BATCH.intro);
  // The progress bar only means anything when the form has pages.
  form.setProgressBar(BATCH.pairsPerPage > 0);
  form.setShuffleQuestions(false);
  form.setCollectEmail(false);
  form.setPublishingSummary(false);
  // Lets the annotator reopen their submitted answers and change them, which is
  // the only way back into a form once it has been sent.
  form.setAllowResponseEdits(true);
  form.setConfirmationMessage(
      'Thank you. If you have not finished, use the "Edit your response" link on this page to '
      + 'carry on later -- keep that link, it reopens the answers you have already given. '
      + 'Submitting again as often as you like is fine and never loses an earlier answer.');

  for (var n = 0; n < BATCH.items.length; n++) {{
    var it = BATCH.items[n];
    // Every page break is one more Next the annotator must tap before Submit is
    // reachable at all, so how many pairs share a page is a setting.
    if (BATCH.pairsPerPage > 0 && n % BATCH.pairsPerPage === 0) {{
      var last = Math.min(n + BATCH.pairsPerPage, BATCH.items.length);
      form.addPageBreakItem().setTitle(
          last > n + 1 ? 'Pairs ' + (n + 1) + '-' + last + ' of ' + BATCH.items.length
                       : 'Pair ' + (n + 1) + ' of ' + BATCH.items.length);
    }}
    form.addSectionHeaderItem()
        .setTitle('Pair ' + (n + 1) + ' of ' + BATCH.items.length)
        .setHelpText(it.area ? 'Area: ' + it.area : '');
    form.addSectionHeaderItem().setTitle('Idea A').setHelpText(it.A_text);
    form.addSectionHeaderItem().setTitle('Idea B').setHelpText(it.B_text);
    form.addMultipleChoiceItem()
        .setTitle('{choice_title}')
        .setChoiceValues(BATCH.labels)
        .setRequired(BATCH.required);
    form.addParagraphTextItem()
        .setTitle('{notes_title}')
        .setHelpText('Required whenever you answer "' + BATCH.unclear + '".')
        .setRequired(false);
  }}

  PropertiesService.getScriptProperties().setProperty('FORM_ID', form.getId());
  Logger.log('Fill in: %s', form.getPublishedUrl());
  Logger.log('Edit:    %s', form.getEditUrl());
  Logger.log('Now run verifyForm() to confirm the idea texts came through unchanged.');
}}

/**
 * Reads every idea text back off the built form and compares it, character for
 * character, with the batch. Run it after buildForm().
 */
function verifyForm() {{
  var id = PropertiesService.getScriptProperties().getProperty('FORM_ID');
  if (!id) {{
    Logger.log('No form has been built from this script yet -- run buildForm() first.');
    return;
  }}
  var all = FormApp.openById(id).getItems(FormApp.ItemType.SECTION_HEADER);
  var headers = [];
  for (var h = 0; h < all.length; h++) {{
    // Each pair also carries a "Pair k of N" header, which holds no idea text.
    if (all[h].getTitle() === 'Idea A' || all[h].getTitle() === 'Idea B') headers.push(all[h]);
  }}
  var expected = [];
  for (var k = 0; k < BATCH.items.length; k++) {{
    expected.push([BATCH.items[k].item_id, 'A', BATCH.items[k].A_text]);
    expected.push([BATCH.items[k].item_id, 'B', BATCH.items[k].B_text]);
  }}
  if (headers.length !== expected.length) {{
    Logger.log('FAIL: form holds %s idea texts, batch holds %s.', headers.length, expected.length);
    return;
  }}
  var bad = 0;
  for (var n = 0; n < expected.length; n++) {{
    var want = expected[n][2], got = headers[n].getHelpText();
    if (want === got) continue;
    bad++;
    var at = 0;
    while (at < want.length && at < got.length && want.charAt(at) === got.charAt(at)) at++;
    Logger.log('FAIL item %s idea %s: differs at character %s (expected %s, form has %s)',
               expected[n][0], expected[n][1], at,
               JSON.stringify(want.substr(at, 20)), JSON.stringify(got.substr(at, 20)));
  }}
  Logger.log(bad ? bad + ' of ' + expected.length + ' idea texts were altered by Forms.'
                 : 'OK: all ' + expected.length + ' idea texts are identical to the page.');
}}
"""


def is_sheet_url(source: str) -> bool:
    """Whether `--responses` names a Google Sheet rather than a local file."""
    return str(source).startswith(("http://", "https://"))


def fetch_sheet_csv(url: str) -> str:
    """Download one tab of a Google Sheet as CSV text.

    Only works on a sheet readable without signing in: the export endpoint has
    no way to ask for credentials, so a private sheet comes back as the sign-in
    page rather than as an error a caller could act on. That is worth catching
    here, because the HTML would otherwise parse as a headerless CSV and fail
    much further along.
    """
    found = _SHEET_ID.search(url)
    if not found:
        raise SystemExit(f"{url} is not a Google Sheets URL (no /spreadsheets/d/<id>).")
    gid = _SHEET_GID.search(url)
    export = _SHEET_EXPORT.format(id=found.group(1), gid=gid.group(1) if gid else "0")
    try:
        with urllib.request.urlopen(export) as response:
            body = response.read().decode("utf-8-sig")
            final = response.geturl()
    except urllib.error.HTTPError as err:
        body, final = "", ""
        if err.code not in (401, 403, 404):
            raise SystemExit(f"{export}: {err}") from err
    except urllib.error.URLError as err:
        raise SystemExit(f"{export}: {err.reason}") from err
    # Sheets answers a request it will not serve with the sign-in page, at 401
    # or at 200 after a redirect to accounts.google.com.
    if not body or body.lstrip().startswith("<") or "accounts.google.com" in final:
        raise SystemExit(
            f"{url} is not readable without signing in.\n"
            "Either share it (Share > General access > Anyone with the link > Viewer) "
            "and rerun, or download it (File > Download > Comma-separated values) and "
            "pass the downloaded file to --responses instead."
        )
    return body


def read_response_table(source: str) -> list[list[str]]:
    """Read a response CSV off disk, or off the sheet it lives in."""
    if is_sheet_url(source):
        return list(csv.reader(fetch_sheet_csv(source).splitlines()))
    with Path(source).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.reader(fh))


def answers_from_responses(
    responses: str | Path,
    rows: Sequence[dict],
    choices: Sequence[str],
) -> tuple[list[dict], list[str]]:
    """Fold a form's responses into answer rows, plus what is still missing.

    `responses` is a downloaded response CSV or the URL of the sheet holding it.

    A form can hold more than one submission -- an edited response, or a second
    sitting. Later rows win per item, and a blank answer never overwrites one
    already given, so a partial resubmission can only add.
    """
    table = read_response_table(str(responses))
    if not table:
        raise SystemExit(f"{responses} is empty.")
    header, *body = table

    by_choice = {_choice_label(c): str(c) for c in choices}
    columns = {}  # index -> (item_id, "choice" | "notes")
    for idx, name in enumerate(header):
        tagged = _TAGGED.search(name.strip())
        if tagged:
            columns[idx] = (tagged.group(1), "notes" if name.strip().startswith("Notes") else "choice")
    if not columns:
        raise SystemExit(f"{responses} has no '{ITEM_TAG.format(item_id='...')}' columns; is it this form's responses?")

    known = {str(r["item_id"]) for r in rows}
    answers: dict[str, dict] = {}
    for line in body:
        for idx, (item_id, kind) in columns.items():
            if item_id not in known or idx >= len(line):
                continue
            value = line[idx].strip()
            if not value:
                continue
            slot = answers.setdefault(item_id, {"choice": "", "notes": ""})
            slot[kind] = by_choice.get(value, value) if kind == "choice" else line[idx].strip()

    out, problems = [], []
    for row in rows:
        item_id = str(row["item_id"])
        got = answers.get(item_id, {})
        out.append({"item_id": item_id, "your_choice": got.get("choice", ""), "notes": got.get("notes", "")})
        if not got.get("choice"):
            problems.append(f"item {item_id}: unanswered")
        elif got.get("choice") == str(choices[-1]) and not got.get("notes", "").strip():
            problems.append(f"item {item_id}: answered '{choices[-1]}' with no note saying why")
    return out, problems


def write_answers_csv(path: Path, answers: Sequence[dict]) -> None:
    """Write the answers file byte-for-byte as the annotation page writes it:
    a bare header, every value quoted, LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(ANSWER_HEADERS) + "\n")
        writer = csv.DictWriter(
            fh, fieldnames=ANSWER_HEADERS, quoting=csv.QUOTE_ALL, lineterminator="\n"
        )
        writer.writerows(answers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("page", type=Path, help="a rendered <annotator>.html from a batch plan")
    parser.add_argument(
        "--responses",
        metavar="CSV_OR_SHEET_URL",
        help="the form's answers -- a downloaded response CSV, or the URL of the response "
        "spreadsheet if it is shared with anyone who has the link. Converts them into an "
        "answers CSV instead of building the script.",
    )
    parser.add_argument("--out", type=Path, help="where to write (default: alongside the page)")
    parser.add_argument(
        "--pairs-per-page",
        type=int,
        default=0,
        metavar="N",
        help="how many pairs share a page (default 0: one scrolling page, so Submit is always "
        "reachable). N>0 splits the form into pages, each needing a Next to pass.",
    )
    parser.add_argument(
        "--require-answers",
        action="store_true",
        help="refuse to submit until every pair is answered. Off by default, so an annotator can "
        "submit part-way and keep adding: only a submitted response reaches the responses CSV.",
    )
    args = parser.parse_args(argv)

    annotator, rows, choices = read_page_rows(args.page)

    if args.responses:
        answers, problems = answers_from_responses(args.responses, rows, choices)
        out = args.out or args.page.with_name(f"answers_{annotator}.csv")
        write_answers_csv(out, answers)
        done = sum(1 for a in answers if a["your_choice"])
        source = "sheet" if is_sheet_url(args.responses) else "file"
        print(f"{out}: {done} of {len(rows)} answered, from the {source}")
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 0

    from novelty_eval.judge_error_annotation.prepare_annotation_batch import INSTRUCTIONS

    out = args.out or args.page.with_name(f"{annotator}_form.gs")
    script = build_apps_script(
        annotator, rows, choices, INSTRUCTIONS, args.require_answers, args.pairs_per_page
    )
    out.write_text(script, encoding="utf-8")
    mode = "every pair required" if args.require_answers else "partial submissions allowed"
    if args.pairs_per_page:
        pages = -(-len(rows) // args.pairs_per_page)
        layout = f"{pages} pages of {args.pairs_per_page}, so {pages - 1} Next taps to reach Submit"
    else:
        layout = "one scrolling page, Submit always reachable"
    print(f"{out}: Apps Script for {len(rows)} pairs, {mode}, {layout}.")
    print("Paste it into script.google.com, run buildForm(), then verifyForm().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
