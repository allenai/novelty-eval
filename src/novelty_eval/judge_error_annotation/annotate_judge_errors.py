"""Blind human annotation of judge novelty mistakes.

Given a pairwise judge run (e.g. .../pairwise-vanilla-ai_current-claude-opus-4-6),
this finds the instances where the judge's predicted winner disagrees with the
gold label, then serves a small local web UI that shows both ideas (blind: no
scores, no gold, no "this was an error" hint, with left/right randomized) and
lets a human pick which idea is more novel. Every answer is written to a local
CSV so results can be compared against the judge and the ground truth.

Usage:
    python -m novelty_eval.judge_error_annotation.annotate_judge_errors \
        --run-dir output/ablation_sweeps/20260518_110534/pairwise-vanilla-ai_current/pairwise-vanilla-ai_current-claude-opus-4-6 \
        --out-csv output/judge_error_annotations/opus-4-6.csv

Then open http://127.0.0.1:8000 in a browser. No external dependencies.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from novelty_eval.judge_error_annotation.selection import (
    ScoredInstance,
    judge_pick,
    judge_novelty_winner,
    partition_scored,
)

CSV_FIELDS = [
    "problem_id",
    "context",
    "human_choice_ab",  # "A" | "B" | "tie" (what the annotator clicked)
    "human_choice_idea",  # decoded idea id "0"/"1", or "tie"
    "judge_choice_idea",  # judge's pairwise winner, or "tie"
    "gold_idea",  # ground-truth winner (expected_winners[0])
    "judge_novelty_winner",  # argmax of judge novelty scores (or "tie")
    "human_correct",  # human_choice_idea == gold_idea
    "judge_correct",  # judge_choice_idea == gold_idea (should be False: this is the mistake set)
    "human_agrees_judge",  # human_choice_idea == judge_choice_idea
    "notes",
    "shown_A_idea",  # which idea was displayed as "A" (blinding record)
    "shown_B_idea",
    "annotated_at",
]


def _iter_mistakes(run_dir: Path, test_instances_override: str | None, repo_root: Path) -> list[ScoredInstance]:
    """The mistake set: (problem_id, instance, score) where the judge's pick is not gold."""
    part = partition_scored(run_dir, test_instances_override, repo_root)
    if part.n_blocked:
        print(f"Excluded {part.n_blocked} blocklisted instance(s) from the pool.", flush=True)
    return part.mistakes


def build_items(run_dir: Path, test_instances_override: str | None, seed: int, repo_root: Path) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    for pid, inst, sc in _iter_mistakes(run_dir, test_instances_override, repo_root):
        gold_list = [str(w) for w in inst["expected_winners"]]
        # Idea keys may load from YAML as ints; keep everything as str so the
        # human/judge/gold equality checks below compare like-typed values.
        ideas = {str(k): v for k, v in inst["ideas"].items()}
        # Randomize which idea is shown as A vs B (blinding).
        idea_ids = list(ideas.keys())
        rng.shuffle(idea_ids)
        a_idea, b_idea = idea_ids[0], idea_ids[1]
        items.append(
            {
                "problem_id": pid,
                "context": inst.get("context", ""),
                "A_idea": a_idea,
                "B_idea": b_idea,
                "A_text": ideas[a_idea],
                "B_text": ideas[b_idea],
                # server-side truth, never sent to the browser:
                "_gold_idea": gold_list[0],
                "_judge_choice_idea": judge_pick(sc) or "tie",
                "_judge_novelty_winner": judge_novelty_winner(sc["novelty"]),
            }
        )
    # Stable, shuffled order (already independent of judge/gold).
    rng.shuffle(items)
    return items


def _client_items(items: list[dict]) -> list[dict]:
    """Strip everything that would de-blind the annotator."""
    return [
        {
            "problem_id": it["problem_id"],
            "context": it["context"],
            "A_text": it["A_text"],
            "B_text": it["B_text"],
        }
        for it in items
    ]


class AnnotationStore:
    def __init__(self, items: list[dict], out_csv: Path):
        self.items = {it["problem_id"]: it for it in items}
        self.order = [it["problem_id"] for it in items]
        self.out_csv = out_csv
        self.answers: dict[str, dict] = {}
        self.lock = threading.Lock()
        self._load_existing()

    def _load_existing(self) -> None:
        if self.out_csv.exists():
            with self.out_csv.open(newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("problem_id") in self.items:
                        self.answers[row["problem_id"]] = row

    def record(self, problem_id: str, choice_ab: str, notes: str) -> dict:
        it = self.items[problem_id]
        if choice_ab == "A":
            human_idea = it["A_idea"]
        elif choice_ab == "B":
            human_idea = it["B_idea"]
        else:
            human_idea = "tie"
        gold = it["_gold_idea"]
        judge = it["_judge_choice_idea"]
        row = {
            "problem_id": problem_id,
            "context": it["context"],
            "human_choice_ab": choice_ab,
            "human_choice_idea": human_idea,
            "judge_choice_idea": judge,
            "gold_idea": gold,
            "judge_novelty_winner": it["_judge_novelty_winner"],
            "human_correct": str(human_idea == gold),
            "judge_correct": str(judge == gold),
            "human_agrees_judge": str(human_idea == judge),
            "notes": notes,
            "shown_A_idea": it["A_idea"],
            "shown_B_idea": it["B_idea"],
            "annotated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with self.lock:
            self.answers[problem_id] = row
            self._flush()
        return row

    def _flush(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_csv.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for pid in self.order:
                if pid in self.answers:
                    w.writerow({k: self.answers[pid].get(k, "") for k in CSV_FIELDS})
        tmp.replace(self.out_csv)

    def done_ids(self) -> list[str]:
        return list(self.answers.keys())


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Judge Novelty Mistakes — Blind Annotation</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 0 auto;
         padding: 1.5rem; line-height: 1.5; }
  header { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
  .bar { height: 6px; background: #8883; border-radius: 3px; margin: .6rem 0 1.2rem; }
  .bar > div { height: 100%; background: #4a90d9; border-radius: 3px; transition: width .2s; }
  .context { background: #8881; border-left: 3px solid #4a90d9; padding: .6rem .9rem;
             border-radius: 4px; margin-bottom: 1rem; font-size: .95rem; }
  .ideas { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .idea { border: 1px solid #8884; border-radius: 8px; padding: 1rem; }
  .idea.sel { border-color: #4a90d9; box-shadow: 0 0 0 2px #4a90d955; }
  .idea h3 { margin: 0 0 .6rem; }
  .choices { display: flex; gap: .6rem; margin: 1.2rem 0; flex-wrap: wrap; }
  button { font: inherit; padding: .55rem 1rem; border-radius: 6px; border: 1px solid #8886;
           background: #8881; cursor: pointer; }
  button.primary { background: #4a90d9; color: #fff; border-color: #4a90d9; }
  button:disabled { opacity: .4; cursor: default; }
  .sel-A #btnA, .sel-B #btnB, .sel-tie #btnTie { background: #4a90d9; color:#fff; border-color:#4a90d9; }
  textarea { width: 100%; min-height: 3rem; font: inherit; margin-top: .3rem; box-sizing: border-box; }
  .nav { display: flex; justify-content: space-between; margin-top: 1.2rem; }
  .done { color: #3a3; font-weight: 600; }
  kbd { background:#8883; border-radius:3px; padding:0 .3rem; font-size:.85em; }
  .hint { font-size:.85rem; opacity:.7; }
  .areas { list-style:none; padding:0; margin:1rem 0; display:grid;
           grid-template-columns:1fr; gap:.4rem; }
  .areas li { border:1px solid #8884; border-radius:6px; }
  .areas label { display:flex; align-items:center; gap:.6rem; padding:.55rem .8rem; cursor:pointer; }
  .areas .cnt { margin-left:auto; opacity:.6; font-size:.85rem; white-space:nowrap; }
  .gate-actions { display:flex; gap:.6rem; align-items:center; margin-top:.4rem; flex-wrap:wrap; }
  #changeAreas { font-size:.8rem; padding:.25rem .6rem; }
</style></head>
<body>
<!-- Area selection gate -->
<section id="gate">
  <h2 style="margin:.2rem 0">Pick the areas you're comfortable annotating</h2>
  <p class="hint">Only instances from the checked areas will be shown. You can come back and change this at any time.</p>
  <div class="gate-actions">
    <button id="selAll">Select all</button>
    <button id="selNone">Clear</button>
    <span id="gateCount" class="hint"></span>
  </div>
  <ul class="areas" id="areaList"></ul>
  <button id="startBtn" class="primary">Start annotating →</button>
</section>

<!-- Annotation view (hidden until areas are chosen) -->
<div id="annotate" style="display:none">
<header>
  <h2 style="margin:.2rem 0">Which idea is more <em>novel</em>?</h2>
  <div style="display:flex;gap:.8rem;align-items:baseline">
    <button id="changeAreas">← Change areas</button>
    <div id="counter"></div>
  </div>
</header>
<div class="bar"><div id="progress"></div></div>
<p class="hint">Blind task: order is randomized; you are not shown any scores or "correct" answer.
  Keys: <kbd>A</kbd> / <kbd>B</kbd> pick, <kbd>T</kbd> tie, <kbd>←</kbd>/<kbd>→</kbd> navigate.</p>
<div class="context" id="context"></div>
<div class="ideas" id="ideasBox">
  <div class="idea" id="ideaA"><h3>Idea A</h3><div id="textA"></div></div>
  <div class="idea" id="ideaB"><h3>Idea B</h3><div id="textB"></div></div>
</div>
<div class="choices" id="choices">
  <button id="btnA">A is more novel</button>
  <button id="btnB">B is more novel</button>
  <button id="btnTie">Tie / can't tell</button>
  <span id="savedMark" class="done"></span>
</div>
<label>Notes (optional): <textarea id="notes"></textarea></label>
<div class="nav">
  <button id="prev">← Prev</button>
  <button id="next" class="primary">Next unanswered →</button>
</div>
</div><!-- /#annotate -->

<script>
let ALL_ITEMS = [], ITEMS = [], ANSWERS = {}, i = 0;
const LS_KEY = 'judgeAnnoAreas';

async function boot() {
  const r = await fetch('/data'); const d = await r.json();
  ALL_ITEMS = d.items; d.done.forEach(a => ANSWERS[a.problem_id] = a);
  renderGate();
}

function areaCounts() {
  const m = new Map();
  ALL_ITEMS.forEach(it => {
    const a = it.context || '(no area)';
    if (!m.has(a)) m.set(a, {total:0, done:0});
    m.get(a).total++;
    if (ANSWERS[it.problem_id]) m.get(a).done++;
  });
  return [...m.entries()].sort((x,y)=>y[1].total-x[1].total);
}
function renderGate() {
  const saved = new Set(JSON.parse(localStorage.getItem(LS_KEY) || 'null') || []);
  const rows = areaCounts();
  const ul = document.getElementById('areaList');
  ul.innerHTML = '';
  rows.forEach(([area, c]) => {
    const checked = saved.size ? saved.has(area) : true;  // default: all
    const li = document.createElement('li');
    li.innerHTML = `<label><input type="checkbox" value="${encodeURIComponent(area)}" ${checked?'checked':''}>`
      + `<span>${area}</span><span class="cnt">${c.done}/${c.total} done</span></label>`;
    ul.appendChild(li);
  });
  updateGateCount();
  ul.querySelectorAll('input').forEach(cb => cb.onchange = updateGateCount);
}
function selectedAreas() {
  return [...document.querySelectorAll('#areaList input:checked')].map(cb => decodeURIComponent(cb.value));
}
function updateGateCount() {
  const sel = new Set(selectedAreas());
  const n = ALL_ITEMS.filter(it => sel.has(it.context || '(no area)')).length;
  document.getElementById('gateCount').textContent = `${n} instance(s) selected`;
}
function startAnnotating() {
  const sel = new Set(selectedAreas());
  if (!sel.size) { updateGateCount(); return; }
  localStorage.setItem(LS_KEY, JSON.stringify([...sel]));
  ITEMS = ALL_ITEMS.filter(it => sel.has(it.context || '(no area)'));
  i = ITEMS.findIndex(it => !ANSWERS[it.problem_id]); if (i < 0) i = 0;
  document.getElementById('gate').style.display = 'none';
  document.getElementById('annotate').style.display = '';
  render();
}
function showGate() {
  document.getElementById('annotate').style.display = 'none';
  document.getElementById('gate').style.display = '';
  renderGate();
}
function render() {
  const it = ITEMS[i];
  const doneHere = ITEMS.filter(x => ANSWERS[x.problem_id]).length;
  document.getElementById('counter').textContent =
    `Item ${i+1} / ${ITEMS.length} — ${doneHere} annotated in selected areas`;
  document.getElementById('progress').style.width =
    (ITEMS.length ? 100*doneHere/ITEMS.length : 0) + '%';
  document.getElementById('context').textContent = it.context || '(no context)';
  document.getElementById('textA').textContent = it.A_text;
  document.getElementById('textB').textContent = it.B_text;
  document.getElementById('notes').value = (ANSWERS[it.problem_id]||{}).notes || '';
  const box = document.getElementById('ideasBox');
  const cur = ANSWERS[it.problem_id];
  document.getElementById('choices').className = 'choices' +
    (cur ? ' sel-' + cur.human_choice_ab : '');
  document.getElementById('ideaA').classList.toggle('sel', cur && cur.human_choice_ab==='A');
  document.getElementById('ideaB').classList.toggle('sel', cur && cur.human_choice_ab==='B');
  document.getElementById('savedMark').textContent = cur ? '✓ saved' : '';
  document.getElementById('prev').disabled = i===0;
}
async function choose(ab) {
  const it = ITEMS[i];
  const body = { problem_id: it.problem_id, choice: ab, notes: document.getElementById('notes').value };
  const r = await fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  ANSWERS[it.problem_id] = await r.json();
  render();
}
function go(d) { i = Math.min(ITEMS.length-1, Math.max(0, i+d)); render(); }
function nextUnanswered() {
  for (let k=1;k<=ITEMS.length;k++){ const j=(i+k)%ITEMS.length; if(!ANSWERS[ITEMS[j].problem_id]){i=j;render();return;} }
  go(1);
}
document.getElementById('btnA').onclick = ()=>choose('A');
document.getElementById('btnB').onclick = ()=>choose('B');
document.getElementById('btnTie').onclick = ()=>choose('tie');
document.getElementById('prev').onclick = ()=>go(-1);
document.getElementById('next').onclick = nextUnanswered;
document.getElementById('startBtn').onclick = startAnnotating;
document.getElementById('changeAreas').onclick = showGate;
document.getElementById('selAll').onclick = ()=>{ document.querySelectorAll('#areaList input').forEach(cb=>cb.checked=true); updateGateCount(); };
document.getElementById('selNone').onclick = ()=>{ document.querySelectorAll('#areaList input').forEach(cb=>cb.checked=false); updateGateCount(); };
document.addEventListener('keydown', e=>{
  if (e.target.tagName==='TEXTAREA') return;
  if (document.getElementById('gate').style.display !== 'none') return;
  if (e.key==='a'||e.key==='A') choose('A');
  else if (e.key==='b'||e.key==='B') choose('B');
  else if (e.key==='t'||e.key==='T') choose('tie');
  else if (e.key==='ArrowLeft') go(-1);
  else if (e.key==='ArrowRight') go(1);
});
boot();
</script>
</body></html>"""


EXPORT_CSS = """
:root { color-scheme: light dark; --fg:#1a1a1a; --bg:#fff; --muted:#5b6570; --line:#d8dee5;
        --card:#fff; --gold:#1a7f45; --goldbg:#eaf6ee; --pick:#b3261e; --pickbg:#fdecea;
        --chip:#eef1f5; --accent:#2f6fb5; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e8ea; --bg:#16181a; --muted:#9aa4ae; --line:#333a41; --card:#1d2023;
          --gold:#5cc98a; --goldbg:#12301f; --pick:#f2938c; --pickbg:#331917;
          --chip:#262b30; --accent:#7fb0e6; }
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       color: var(--fg); background: var(--bg); line-height: 1.55; margin: 0;
       padding: 2rem 1.25rem 4rem; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .35rem; }
.sub { color: var(--muted); font-size: .9rem; margin: 0 0 1.5rem; }
.sub code { background: var(--chip); padding: .1rem .35rem; border-radius: 4px; font-size: .9em; }
.stats { display: flex; flex-wrap: wrap; gap: .75rem; margin-bottom: 1.25rem; }
.stat { background: var(--chip); border-radius: 8px; padding: .55rem .9rem; }
.stat b { font-size: 1.25rem; display: block; line-height: 1.2; }
.stat span { color: var(--muted); font-size: .8rem; }
.controls { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
            margin-bottom: 1.5rem; position: sticky; top: 0; background: var(--bg);
            padding: .6rem 0; border-bottom: 1px solid var(--line); z-index: 5; }
.controls select, .controls input { font: inherit; padding: .4rem .6rem; border-radius: 6px;
       border: 1px solid var(--line); background: var(--card); color: var(--fg); }
.controls input { flex: 1; min-width: 12rem; }
.count { color: var(--muted); font-size: .85rem; }
.card { border: 1px solid var(--line); border-radius: 10px; background: var(--card);
        padding: 1.1rem; margin-bottom: 1.25rem; }
.card > header { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
                 margin-bottom: .5rem; }
.pid { font-weight: 700; font-size: 1.05rem; }
.area { color: var(--muted); font-size: .85rem; }
.tags { display: flex; flex-wrap: wrap; gap: .4rem; margin-left: auto; }
.tag { font-size: .75rem; font-weight: 600; padding: .2rem .55rem; border-radius: 999px;
       border: 1px solid transparent; white-space: nowrap; }
.tag.gold { color: var(--gold); background: var(--goldbg); border-color: var(--gold); }
.tag.pick { color: var(--pick); background: var(--pickbg); border-color: var(--pick); }
.tag.plain { color: var(--muted); background: var(--chip); }
.ideas { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: .8rem; }
@media (max-width: 820px) { .ideas { grid-template-columns: 1fr; } }
.idea { border: 1px solid var(--line); border-radius: 8px; padding: .9rem; }
.idea h3 { margin: 0 0 .1rem; font-size: .95rem; }
.scores { display: flex; flex-wrap: wrap; gap: .8rem; font-size: .8rem; color: var(--muted);
          margin: .4rem 0 .6rem; padding-bottom: .5rem; border-bottom: 1px dashed var(--line); }
.scores b { color: var(--fg); }
.text { font-size: .9rem; white-space: pre-wrap; }
details { margin-top: .6rem; font-size: .85rem; }
summary { cursor: pointer; color: var(--accent); }
details ul { margin: .4rem 0 0; padding-left: 1.1rem; }
details li { margin-bottom: .3rem; }
.empty { color: var(--muted); font-style: italic; }
@media print { .controls { position: static; } .card { break-inside: avoid; } }
"""

EXPORT_JS = """
const cards = Array.from(document.querySelectorAll('.card'));
const areaSel = document.getElementById('areaSel');
const q = document.getElementById('q');
const count = document.getElementById('count');
function apply() {
  const a = areaSel.value, t = q.value.trim().toLowerCase();
  let n = 0;
  cards.forEach(c => {
    const okArea = (a === '*' || c.dataset.area === a);
    const okText = (!t || c.dataset.haystack.includes(t));
    const show = okArea && okText;
    c.style.display = show ? '' : 'none';
    if (show) n++;
  });
  count.textContent = n + ' of ' + cards.length + ' shown';
}
areaSel.onchange = apply; q.oninput = apply; apply();
"""


def _signals_block(label: str, signals) -> str:
    if not signals:
        return ""
    lis = "".join(f"<li>{html.escape(str(s))}</li>" for s in signals)
    return f"<details><summary>{html.escape(label)} ({len(signals)})</summary><ul>{lis}</ul></details>"


def _idea_html(idea_id: str, text: str, meta: dict) -> str:
    """One idea column. Deliberately neutral: which idea is gold and which the judge
    picked is stated once, in the card header tags."""
    title = meta.get("title") or f"Idea {idea_id}"
    scores = (
        f"<div class='scores'>"
        f"<span>idea id <b>{html.escape(idea_id)}</b></span>"
        f"<span>contribution <b>{html.escape(str(meta.get('contribution', '—')))}</b></span>"
        f"<span>rating <b>{html.escape(str(meta.get('rating', '—')))}</b></span>"
        f"</div>"
    )
    body = html.escape(text).strip() or "<span class='empty'>(no text)</span>"
    return (
        f"<div class='idea'>"
        f"<h3>{html.escape(str(title))}</h3>"
        f"{scores}"
        f"<div class='text'>{body}</div>"
        f"{_signals_block('Positive signals', meta.get('positive_signals'))}"
        f"{_signals_block('Negative signals', meta.get('negative_signals'))}"
        f"</div>"
    )


def _card_html(pid: str, inst: dict, sc: dict) -> tuple[str, str]:
    """Render one mistake as a card. Returns (area, html)."""
    area = inst.get("context", "") or "(no area)"
    gold = str(inst["expected_winners"][0])
    pick = judge_pick(sc)
    metadata = inst.get("metadata") or {}
    ideas = {str(k): v for k, v in inst["ideas"].items()}

    cols = []
    haystack = [pid, area]
    for idea_id in sorted(ideas):
        meta = metadata.get(idea_id) or metadata.get(int(idea_id)) or {}
        if not isinstance(meta, dict):
            meta = {}
        cols.append(_idea_html(idea_id, ideas[idea_id], meta))
        haystack += [str(meta.get("title") or ""), ideas[idea_id]]

    pick_label = "judge scored them equal" if pick is None else f"judge picked: idea {html.escape(pick)}"
    tags = (
        f"<span class='tag gold'>gold: idea {html.escape(gold)}</span>"
        f"<span class='tag pick'>{pick_label}</span>"
    )
    card = (
        f"<article class='card' data-area=\"{html.escape(area, quote=True)}\" "
        f"data-haystack=\"{html.escape(' '.join(haystack).lower(), quote=True)}\">"
        f"<header><span class='pid'>Instance {html.escape(pid)}</span>"
        f"<span class='area'>{html.escape(area)}</span>"
        f"<span class='tags'>{tags}</span></header>"
        f"<div class='ideas'>{''.join(cols)}</div>"
        f"</article>"
    )
    return area, card


def export_html(run_dir: Path, test_instances_override: str | None, repo_root: Path, out_path: Path) -> int:
    """Write the full mistake set as a standalone, self-contained HTML report."""
    mistakes = _iter_mistakes(run_dir, test_instances_override, repo_root)
    if not mistakes:
        raise SystemExit("No judge mistakes found for this run (predicted == gold everywhere).")

    # Group by area, then by numeric problem id, so related cases read together.
    rendered = [_card_html(pid, inst, sc) for pid, inst, sc in mistakes]
    order = {pid: i for i, (pid, _, _) in enumerate(mistakes)}
    pairs = sorted(zip(mistakes, rendered), key=lambda t: (t[1][0], int(t[0][0]) if t[0][0].isdigit() else order[t[0][0]]))
    cards = [card for _, (_, card) in pairs]
    areas = sorted({area for _, (area, _) in pairs})

    counts: dict[str, int] = {}
    for _, (area, _) in pairs:
        counts[area] = counts.get(area, 0) + 1
    options = "".join(
        f"<option value=\"{html.escape(a, quote=True)}\">{html.escape(a)} ({counts[a]})</option>" for a in areas
    )

    run_label = run_dir.name
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judge novelty mistakes — {html.escape(run_label)}</title>
<style>{EXPORT_CSS}</style></head>
<body><main>
<h1>Judge novelty mistakes</h1>
<p class="sub">Pairwise-novelty instances where the judge's predicted winner disagrees with the
gold label. Run: <code>{html.escape(run_label)}</code> · generated {html.escape(generated)}</p>
<div class="stats">
  <div class="stat"><b>{len(cards)}</b><span>mistakes</span></div>
  <div class="stat"><b>{len(areas)}</b><span>areas</span></div>
</div>
<div class="controls">
  <select id="areaSel"><option value="*">All areas ({len(cards)})</option>{options}</select>
  <input id="q" type="search" placeholder="Search titles, ideas, instance ids…">
  <span class="count" id="count"></span>
</div>
{''.join(cards)}
</main>
<script>{EXPORT_JS}</script>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return len(cards)


def make_handler(store: AnnotationStore, client_items: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/data":
                done = [store.answers[p] for p in store.done_ids()]
                self._send(200, json.dumps({"items": client_items, "done": done}))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if self.path != "/save":
                self._send(404, json.dumps({"error": "not found"}))
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            pid = str(payload.get("problem_id"))
            if pid not in store.items:
                self._send(400, json.dumps({"error": "unknown problem_id"}))
                return
            row = store.record(pid, payload.get("choice", "tie"), payload.get("notes", ""))
            self._send(200, json.dumps(row))

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="Judge run/model dir containing scores.json")
    ap.add_argument("--out-csv", default=None, help="CSV file to write annotations to (created/updated)")
    ap.add_argument("--test-instances", default=None, help="Override path to the test-instances YAML")
    ap.add_argument(
        "--export-html",
        default=None,
        metavar="PATH",
        help="Write the mistake set to a standalone HTML report (unblinded: shows gold and the "
        "judge's pick) and exit, instead of serving the annotation UI.",
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seed", type=int, default=13, help="Shuffle seed for reproducible A/B blinding")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = Path(args.run_dir)

    if args.export_html:
        out_path = Path(args.export_html)
        n = export_html(run_dir, args.test_instances, repo_root, out_path)
        print(f"Wrote {n} judge mistakes to {out_path}")
        return

    if not args.out_csv:
        ap.error("--out-csv is required when serving the annotation UI")
    items = build_items(run_dir, args.test_instances, args.seed, repo_root)
    if not items:
        raise SystemExit("No judge mistakes found for this run (predicted == gold everywhere).")

    store = AnnotationStore(items, Path(args.out_csv))
    handler = make_handler(store, _client_items(items))
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Found {len(items)} judge mistakes to annotate.")
    print(f"Writing annotations to: {store.out_csv}")
    print(f"Already annotated: {len(store.done_ids())}/{len(items)}")
    print(f"\n  Open  http://{args.host}:{args.port}  in your browser.\n  Ctrl-C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. CSV saved at", store.out_csv)


if __name__ == "__main__":
    main()
