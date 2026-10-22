"""Render an annotation batch as a standalone HTML page.

The browser-facing twin of the xlsx: same rows, same blinding, nicer to read.
An annotator opens the file offline, answers with the keyboard, and downloads a
small answers CSV to send back. Progress autosaves to the browser, and the same
CSV can be loaded back in to resume, so nothing depends on browser storage
surviving.

This module is deliberately presentation-only. It knows nothing about judge
runs, gold labels or sampling: it takes finished rows and whitelists the four
fields a page may show, so nothing that would de-blind an annotator can reach
the markup even if a caller passes richer rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

# The only fields a page may render. Anything else in a row -- group, source,
# gold_idea, is_mistake -- is dropped here rather than trusted not to be used.
VISIBLE_FIELDS = ("item_id", "area", "A_text", "B_text")

PAGE_CSS = """
:root { color-scheme: light dark;
  --fg:#1a1a1a; --bg:#fff; --muted:#5b6570; --line:#d8dee5; --card:#fff;
  --accent:#2f6fb5; --accent-fg:#fff; --chip:#eef1f5; --warn:#8a5b00; --warnbg:#fff6e5; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e8ea; --bg:#16181a; --muted:#9aa4ae; --line:#333a41; --card:#1d2023;
    --accent:#7fb0e6; --accent-fg:#10151b; --chip:#262b30; --warn:#f0c67a; --warnbg:#2b2313; }
}
* { box-sizing:border-box; }
body { font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  color:var(--fg); background:var(--bg); line-height:1.6; margin:0; padding:1.5rem 1.25rem 4rem; }
main { max-width:1180px; margin:0 auto; }
h1 { font-size:1.25rem; margin:0; }
.top { display:flex; justify-content:space-between; align-items:baseline; gap:1rem; flex-wrap:wrap; }
.who { color:var(--muted); font-size:.9rem; }
.bar { height:6px; background:var(--chip); border-radius:3px; margin:.8rem 0 1.2rem; overflow:hidden; }
.bar > div { height:100%; background:var(--accent); width:0; transition:width .2s; }
.area { background:var(--chip); border-left:3px solid var(--accent); padding:.55rem .9rem;
  border-radius:4px; margin-bottom:1rem; font-size:.9rem; color:var(--muted); }
.ideas { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
@media (max-width:860px) { .ideas { grid-template-columns:1fr; } }
.idea { border:1px solid var(--line); border-radius:10px; padding:1rem 1.1rem; background:var(--card); }
.idea.sel { border-color:var(--accent); box-shadow:0 0 0 2px color-mix(in srgb, var(--accent) 35%, transparent); }
.idea h2 { font-size:.8rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
  margin:0 0 .6rem; font-weight:700; }
.idea p { margin:0; white-space:pre-wrap; font-size:.95rem; }
.choices { display:flex; gap:.6rem; margin:1.2rem 0 .4rem; flex-wrap:wrap; align-items:center; }
button { font:inherit; padding:.55rem 1.1rem; border-radius:8px; border:1px solid var(--line);
  background:var(--card); color:var(--fg); cursor:pointer; }
button:hover { border-color:var(--accent); }
button.on { background:var(--accent); color:var(--accent-fg); border-color:var(--accent); }
button.primary { background:var(--accent); color:var(--accent-fg); border-color:var(--accent); }
button:disabled { opacity:.4; cursor:default; }
label.notes { display:block; margin-top:.8rem; font-size:.9rem; color:var(--muted); }
textarea { width:100%; min-height:3.2rem; font:inherit; margin-top:.3rem; padding:.5rem .6rem;
  border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--fg); }
textarea.needed { border-color:var(--warn); background:var(--warnbg); }
.nav { display:flex; justify-content:space-between; margin-top:1.4rem; gap:.6rem; }
.hint { font-size:.85rem; color:var(--muted); }
kbd { background:var(--chip); border-radius:4px; padding:.05rem .35rem; font-size:.85em;
  border:1px solid var(--line); }
.saved { color:var(--accent); font-size:.85rem; font-weight:600; }
.warn { background:var(--warnbg); color:var(--warn); border:1px solid var(--warn); border-radius:8px;
  padding:.6rem .9rem; margin:1rem 0; font-size:.9rem; }
.warn:empty { display:none; }
.done { border:1px solid var(--line); border-radius:10px; padding:1.1rem; margin-top:1.5rem;
  background:var(--card); }
.done h2 { margin:0 0 .5rem; font-size:1rem; }
.done .row { display:flex; gap:.6rem; flex-wrap:wrap; align-items:center; margin-top:.7rem; }
details.help { margin-bottom:1.2rem; }
details.help summary { cursor:pointer; color:var(--accent); font-size:.9rem; }
details.help ol { margin:.5rem 0 0; padding-left:1.2rem; font-size:.9rem; color:var(--muted); }
"""

PAGE_JS = r"""
const ITEMS = JSON.parse(document.getElementById('items').textContent);
const CHOICES = JSON.parse(document.getElementById('choices').textContent);
const NEEDS_NOTE = CHOICES[CHOICES.length - 1];   // the "can't judge" option
// Item ids restart at 001 in every batch, so the store key has to depend on the
// contents too: without this a second batch of the same size would read the
// previous one's answers as its own.
const FINGERPRINT = (() => {
  let h = 2166136261;
  const s = JSON.stringify(ITEMS);
  for (let p = 0; p < s.length; p++) { h ^= s.charCodeAt(p); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(36);
})();
const STORE = 'annotation:' + document.body.dataset.annotator + ':' + FINGERPRINT;
let answers = {}, i = 0, notice = '';

const $ = id => document.getElementById(id);
const load = () => { try { answers = JSON.parse(localStorage.getItem(STORE)) || {}; } catch (e) { answers = {}; } };
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(answers)); } catch (e) {} };

function counts() {
  const done = ITEMS.filter(t => answers[t.item_id] && answers[t.item_id].choice).length;
  const missing = ITEMS.filter(t => {
    const a = answers[t.item_id];
    return a && a.choice === NEEDS_NOTE && !(a.notes || '').trim();
  }).length;
  return { done, missing };
}

function render() {
  const t = ITEMS[i], a = answers[t.item_id] || {};
  const { done, missing } = counts();
  $('counter').textContent = `Pair ${i + 1} of ${ITEMS.length} — ${done} answered`;
  $('progress').style.width = (100 * done / ITEMS.length) + '%';
  $('area').textContent = t.area || '(no area given)';
  $('textA').textContent = t.A_text;
  $('textB').textContent = t.B_text;
  $('itemId').textContent = 'item ' + t.item_id;
  CHOICES.forEach(c => $('btn-' + c).classList.toggle('on', a.choice === c));
  $('ideaA').classList.toggle('sel', a.choice === 'A');
  $('ideaB').classList.toggle('sel', a.choice === 'B');
  const notes = $('notes');
  notes.value = a.notes || '';
  notes.classList.toggle('needed', a.choice === NEEDS_NOTE && !(a.notes || '').trim());
  $('savedMark').textContent = a.choice ? 'saved' : '';
  $('prev').disabled = i === 0;
  $('next').disabled = i === ITEMS.length - 1;
  const left = ITEMS.length - done;
  $('warn').textContent = [
    notice,
    left ? `${left} pair(s) still unanswered.` : '',
    missing ? `${missing} "${NEEDS_NOTE}" answer(s) still need a note saying why.` : ''
  ].filter(Boolean).join(' ');
  $('where').innerHTML = whereLine();
  $('pick').textContent = handle ? 'Save somewhere else' : 'Choose where to save my answers';
  $('pick').hidden = !CAN_WRITE_FILES;
}

function choose(c) {
  const t = ITEMS[i];
  answers[t.item_id] = Object.assign({}, answers[t.item_id], { choice: c });
  save(); persist(); render();
  if (c !== NEEDS_NOTE) setTimeout(() => go(1), 120);   // keep moving; notes can wait
}
function note(v) {
  const t = ITEMS[i];
  answers[t.item_id] = Object.assign({}, answers[t.item_id], { notes: v });
  save(); persist(); render();
}
function go(d) { i = Math.min(ITEMS.length - 1, Math.max(0, i + d)); render(); window.scrollTo(0, 0); }
function nextUnanswered() {
  for (let k = 1; k <= ITEMS.length; k++) {
    const j = (i + k) % ITEMS.length;
    if (!(answers[ITEMS[j].item_id] || {}).choice) { i = j; render(); window.scrollTo(0, 0); return; }
  }
}

const q = s => '"' + String(s == null ? '' : s).replace(/"/g, '""') + '"';
const FILENAME = 'answers_' + document.body.dataset.annotator + '.csv';

function csvText() {
  const lines = ['item_id,your_choice,notes'];
  ITEMS.forEach(t => {
    const a = answers[t.item_id] || {};
    lines.push([q(t.item_id), q(a.choice || ''), q(a.notes || '')].join(','));
  });
  return lines.join('\n') + '\n';
}

function download() {
  const blob = new Blob([csvText()], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const el = document.createElement('a');
  el.href = url;
  el.download = FILENAME;
  el.click();
  URL.revokeObjectURL(url);
}

// --- keeping the answers on disk -------------------------------------------
// The answers live in a real file the annotator picks once. Every answer
// rewrites it, so finishing the batch requires no closing action and nothing
// is riding on browser storage. Where the API is missing (Safari, Firefox) the
// page falls back to re-downloading the file every few answers.
const CAN_WRITE_FILES = 'showSaveFilePicker' in window;
let handle = null, savedAt = null, sinceDownload = 0, pickerBusy = false, declined = false;

function whereLine() {
  if (handle) {
    return savedAt
      ? `Saved to <b>${handle.name}</b>, in the folder you chose &middot; last saved ${savedAt}`
      : `Answers will be saved to <b>${handle.name}</b>, in the folder you chose.`;
  }
  if (!CAN_WRITE_FILES) {
    return `This browser cannot save straight to a file, so <b>${FILENAME}</b> is re-downloaded to your `
         + `Downloads folder every few answers. Chrome saves continuously instead, if you have it.`;
  }
  if (declined) {
    return `<b>Your answers are not being written to a file.</b> A copy of <b>${FILENAME}</b> lands in your `
         + `Downloads folder every few answers, so nothing is lost, but choosing a file below is tidier.`;
  }
  return `<b>Not saving to a file yet.</b> Choose where your answers should be kept &mdash; after that `
       + `every answer is written there automatically.`;
}

async function chooseDestination() {
  if (!CAN_WRITE_FILES || pickerBusy) return false;
  pickerBusy = true;
  try {
    handle = await window.showSaveFilePicker({
      suggestedName: FILENAME,
      startIn: 'desktop',
      types: [{ description: 'CSV', accept: { 'text/csv': ['.csv'] } }]
    });
    // Picking a file that already holds answers must never wipe them: read it
    // back in first, then write the union.
    await adoptExisting();
    await writeFile();
    declined = false;
    return true;
  } catch (e) {
    return false;                       // cancelled: the fallback covers it
  } finally {
    pickerBusy = false;
    render();
  }
}

// Fold answers already sitting in a CSV into this session, keeping the file's
// version wherever the two disagree, so choosing an existing file can only ever
// add answers back -- never silently replace them with a blank screen.
function mergeAnswers(text, source) {
  const rows = parseCsv(text);
  const head = rows.shift() || [];
  const ci = head.indexOf('item_id'), cc = head.indexOf('your_choice'), cn = head.indexOf('notes');
  if (ci < 0 || cc < 0) { notice = `${source} is not an answers file (no item_id column); it was left alone.`; return false; }
  let added = 0, replaced = 0;
  rows.forEach(r => {
    const id = r[ci];
    if (!id || !ITEMS.some(t => t.item_id === id)) return;
    const choice = (r[cc] || '').trim();
    const notes = cn >= 0 ? (r[cn] || '') : '';
    if (!choice && !notes.trim()) return;                 // a blank row is not an answer
    const was = answers[id];
    const val = { choice: CHOICES.includes(choice) ? choice : '', notes: notes };
    if (!was || !was.choice) added++;
    else if (was.choice !== val.choice || (was.notes || '') !== val.notes) replaced++;
    else return;
    answers[id] = val;
  });
  save();
  notice = added || replaced
    ? `Loaded ${added + replaced} answer(s) from ${source}.`
      + (replaced ? ` ${replaced} differed from this screen and the file's version was kept.` : '')
    : `${source} held nothing new.`;
  return true;
}

async function adoptExisting() {
  let text = '';
  try { text = await (await handle.getFile()).text(); } catch (e) { return; }
  if (text.trim()) mergeAnswers(text, 'that file');
}

async function writeFile() {
  if (!handle) return false;
  try {
    const w = await handle.createWritable();
    await w.write(csvText());
    await w.close();
    savedAt = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return true;
  } catch (e) {
    handle = null;                      // permission lost: fall back rather than fail quietly
    return false;
  }
}

// Called after every answer. Writes the file if we have one, and otherwise
// keeps a recent copy in Downloads so a lost tab costs a few answers at most.
async function persist() {
  if (handle) { await writeFile(); render(); return; }
  // Offer the picker once. Someone who dismisses it must not be asked again on
  // every single answer -- the button in the panel is there if they reconsider.
  if (CAN_WRITE_FILES && !declined && !pickerBusy) {
    if (!(await chooseDestination())) declined = true;
  }
  if (!handle) {
    sinceDownload++;
    if (sinceDownload >= 5) { sinceDownload = 0; download(); }
  }
  render();
}

// Minimal RFC4180 reader: enough for the file this page writes.
function parseCsv(text) {
  const rows = []; let row = [], cell = '', quoted = false;
  for (let p = 0; p < text.length; p++) {
    const ch = text[p];
    if (quoted) {
      if (ch === '"' && text[p + 1] === '"') { cell += '"'; p++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(cell); cell = ''; }
    else if (ch === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
    else if (ch !== '\r') cell += ch;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}
function importCsv(file) {
  const reader = new FileReader();
  reader.onload = () => { mergeAnswers(reader.result, 'that file'); i = 0; render(); persist(); };
  reader.readAsText(file);
}

CHOICES.forEach(c => $('btn-' + c).onclick = () => choose(c));
$('prev').onclick = () => go(-1);
$('next').onclick = () => go(1);
$('skip').onclick = nextUnanswered;
$('notes').oninput = e => note(e.target.value);
$('download').onclick = download;
$('pick').onclick = chooseDestination;
$('importer').onchange = e => { if (e.target.files[0]) importCsv(e.target.files[0]); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.metaKey || e.ctrlKey) return;
  const k = e.key.toLowerCase();
  const hit = CHOICES.find(c => c[0].toLowerCase() === k);
  if (hit) { choose(hit); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') go(-1);
  else if (e.key === 'ArrowRight') go(1);
});
// Only nag when the answers are not already on disk, so the warning stays
// meaningful instead of firing on every close.
window.addEventListener('beforeunload', e => {
  if (counts().done && !handle) { e.preventDefault(); e.returnValue = ''; }
});
load(); render();
"""


def _page_items(rows: Sequence[dict]) -> list[dict]:
    """Strip every row down to the fields a page is allowed to show."""
    return [{k: row[k] for k in VISIBLE_FIELDS} for row in rows]


def _embed(value) -> str:
    """JSON for a <script> block, with < escaped so a stray </script> cannot break out."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def write_annotation_page(
    path: Path,
    annotator: str,
    rows: Sequence[dict],
    choices: Sequence[str],
    instructions: Sequence[str],
) -> None:
    """Write one annotator's batch as a self-contained HTML page."""
    steps = "".join(f"<li>{line.lstrip('1234567890. ')}</li>" for line in instructions if line[:1].isdigit())
    buttons = "".join(
        f"<button id='btn-{c}'>{'Idea ' + c + ' is more novel' if c in ('A', 'B') else c.capitalize()}"
        f" <kbd>{c[0].upper()}</kbd></button>"
        for c in choices
    )
    return path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Novelty annotation — {annotator}</title>
<style>{PAGE_CSS}</style></head>
<body data-annotator="{annotator}"><main>
<div class="top">
  <h1>Which idea is more <em>novel</em>?</h1>
  <div class="who">{annotator} · <span id="itemId"></span></div>
</div>
<div class="bar"><div id="progress"></div></div>
<div id="counter" class="hint"></div>

<details class="help"><summary>Instructions</summary><ol>{steps}</ol>
<p class="hint">Keys: {" ".join(f"<kbd>{c[0].upper()}</kbd>" for c in choices)} to answer,
<kbd>&larr;</kbd> <kbd>&rarr;</kbd> to move. Your first answer asks where to keep the answers file;
after that every answer is saved there automatically and there is nothing to remember at the
end.</p></details>

<div id="warn" class="warn"></div>
<div class="area" id="area"></div>
<div class="ideas">
  <div class="idea" id="ideaA"><h2>Idea A</h2><p id="textA"></p></div>
  <div class="idea" id="ideaB"><h2>Idea B</h2><p id="textB"></p></div>
</div>
<div class="choices">{buttons}<span id="savedMark" class="saved"></span></div>
<label class="notes">Notes (required when you answer {choices[-1]})
  <textarea id="notes" placeholder="What made this hard?"></textarea></label>
<div class="nav">
  <button id="prev">&larr; Previous</button>
  <button id="skip">Next unanswered</button>
  <button id="next">Next &rarr;</button>
</div>

<div class="done">
  <h2>Where your answers are saved</h2>
  <p class="hint" id="where"></p>
  <p class="hint">There is nothing to do at the end &mdash; every answer is written as you go. To
  carry on later, reopen this page; to continue on another machine, load your answers file below.
  Loading a file only ever adds answers back, it never blanks the ones you have.</p>
  <div class="row">
    <button id="pick" class="primary">Choose where to save my answers</button>
    <button id="download">Save a copy now</button>
    <label class="hint">Continue from a file: <input type="file" id="importer" accept=".csv"></label>
  </div>
</div>
</main>
<script type="application/json" id="items">{_embed(_page_items(rows))}</script>
<script type="application/json" id="choices">{_embed(list(choices))}</script>
<script>{PAGE_JS}</script>
</body></html>""",
        encoding="utf-8",
    )
