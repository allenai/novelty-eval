 #!/usr/bin/env python3
"""web_search_novelty_judge.py — GPT web-search novelty judge that emits its citations.

Given a research idea, ask a GPT model (default gpt-5.5-pro) with the hosted
`web_search` tool to (1) judge the idea's novelty using the *verbatim* web_search
pointwise-judge prompt, then (2) list the arXiv papers it used to reach that
verdict. Each idea therefore produces TWO artifacts, stored side-by-side in one
retrieval-cache entry:

  - `self_judgement` — the judge's novelty verdict, scored later by
    run_benchmark.py's `cached_self_judgement` backend through the exact same
    pointwise metrics path as every other judge.
  - `candidates`     — the cited papers, grounded via the Semantic Scholar batch
    API (ARXIV:<id>) and filtered (cutoff date, dedup, top-k) identically to the
    paper-finder backend in retrieve_candidates.py, so independent downstream
    judges can consume them as a neutral paper cache.

Four run modes (--mode):
  live     — synchronous Responses-API calls (debug / small runs).
  submit   — emit one /v1/responses batch line per uncached idea and submit it.
  collect  — download a finished batch, ground + filter, and merge into the cache.
  resubmit — re-submit only the previous batch's ideas still missing a verdict
             (provider errors / parse failures), then collect again to fill the gaps.

The web_search batch path runs on the OpenAI Responses endpoint, which is
confirmed to execute web_search inside batch jobs.
"""
import argparse
import asyncio
import ast
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple

import yaml

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import LOGGER as _BASE_LOGGER, prompt_openai_client, extract_json_choice, extract_thinking_process
from semantic_scholar import SEMANTIC_API
from batch_api import (
    prepare_web_search_responses_request,
    submit_batch_job,
    retrieve_raw_batch_content,
    check_batch_status,
)

try:
    from logging_utils import setup_logger as _setup_logger
except ImportError:
    try:
        from src.logging_utils import setup_logger as _setup_logger
    except ImportError:
        _setup_logger = None

LOGGER = _BASE_LOGGER

try:
    from .retrieval_common import (
        RetrievalConfigBase,
        render_prompt,
        load_retrieval_cache,
        save_retrieval_cache,
        _filter_and_select_candidates,
        select_and_format_candidates,
        read_inputs,
        run_in_chunks,
        idea_identity,
        set_logger as _set_common_logger,
    )
except ImportError:
    from retrieval_common import (
        RetrievalConfigBase,
        render_prompt,
        load_retrieval_cache,
        save_retrieval_cache,
        _filter_and_select_candidates,
        select_and_format_candidates,
        read_inputs,
        run_in_chunks,
        idea_identity,
        set_logger as _set_common_logger,
    )

try:
    from .retrieval_reports import format_retrieval_debug_info, write_skipped_duplicates_log
except ImportError:
    from retrieval_reports import format_retrieval_debug_info, write_skipped_duplicates_log

try:
    from src.cost_tracker import cost_stage, GLOBAL_COST_TRACKER
except ImportError:
    try:
        from cost_tracker import cost_stage, GLOBAL_COST_TRACKER
    except ImportError:
        from contextlib import nullcontext as cost_stage
        GLOBAL_COST_TRACKER = None

try:
    from novelty_eval.cost_report_writer import write_cost_report_md as _write_cost_report_md
except ImportError:
    try:
        from cost_report_writer import write_cost_report_md as _write_cost_report_md
    except ImportError:
        _write_cost_report_md = None

# EVALUATION_CRITERIA — imported so the judge-with-papers template renders the
# *exact* novelty criterion text used by the web_search pointwise judge.
try:
    from novelty_eval.judge import EVALUATION_CRITERIA as _EVALUATION_CRITERIA
except ImportError:
    try:
        from judge import EVALUATION_CRITERIA as _EVALUATION_CRITERIA
    except ImportError:
        _EVALUATION_CRITERIA = {"novelty": ""}


# Judge-prompt-verbatim template: judge novelty first, then list the evidence papers.
_JUDGE_WITH_PAPERS_TEMPLATE = "pointwise_novelty_web_search_with_papers.jinja2"
_SS_FIELDS = "title,abstract,publicationDate,year,externalIds,corpusId,url,paperId"


@dataclass
class WebSearchNoveltyJudgeConfig(RetrievalConfigBase):
    llm_engine: str = "gpt-5.5-pro"
    reasoning_effort: str = "medium"
    allowed_domains: List[str] = field(default_factory=lambda: ["arxiv.org"])
    # Concurrency for the synchronous 'live' mode.
    max_workers: int = 4
    # When True, key the cache by `idea_identity(idea_text, cutoff_date)` instead of
    # the dataset's positional problem_id, turning `output_file` into a reusable
    # content-addressed master store. Point several datasets' runs at one master file
    # and only unseen idea texts (e.g. a new negative set) get retrieved; re-key into
    # each dataset's expected shape with project_retrieval_cache.py.
    content_addressed: bool = False


# ---------------------------------------------------------------------------
# Prompt + tool construction
# ---------------------------------------------------------------------------

def build_idea_prompt(idea_text: str, cutoff_date: Optional[str]) -> str:
    """Render the verbatim web_search pointwise-judge prompt (verdict, then papers)."""
    return render_prompt(
        _JUDGE_WITH_PAPERS_TEMPLATE,
        idea={"text": idea_text},
        evaluation_criteria=_EVALUATION_CRITERIA,
        cutoff_date=cutoff_date,
    )


def web_search_tools(allowed_domains: List[str]) -> List[Dict]:
    tool: Dict = {"type": "web_search"}
    if allowed_domains:
        tool["filters"] = {"allowed_domains": list(allowed_domains)}
    return [tool]


# ---------------------------------------------------------------------------
# Parsing model output → raw arxiv items
# ---------------------------------------------------------------------------

_PAPERS_BLOCK_RE = re.compile(r'\[PAPERS_START\](.*?)\[PAPERS_END\]', re.DOTALL)


def parse_web_search_items(text: str) -> List[Dict]:
    """Extract a JSON array of {arxiv_id, why_retrieve, relevance_score} from model text.

    The cited papers live inside a [PAPERS_START]...[PAPERS_END] block, kept separate
    from the verdict JSON object; prefer that block when present so the verdict object
    is never mistaken for the papers array.
    """
    if not text:
        return []
    block = _PAPERS_BLOCK_RE.search(text)
    t = block.group(1) if block else text
    t = re.sub(r'<thinking>.*?</thinking>', '', t, flags=re.DOTALL)

    blob = None
    m = re.search(r'```(?:json)?\s*(\[.*\])\s*```', t, re.DOTALL)
    if m:
        blob = m.group(1)
    else:
        start, end = t.find('['), t.rfind(']')
        if start != -1 and end != -1 and end > start:
            blob = t[start:end + 1]

    data = None
    if blob is not None:
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(blob)
                break
            except Exception:
                continue

    if data is None:
        # Fallback: an object wrapping the list under a known key.
        obj = extract_json_choice(t)
        if isinstance(obj, dict):
            for k in ("papers", "results", "candidates"):
                if isinstance(obj.get(k), list):
                    data = obj[k]
                    break

    return data if isinstance(data, list) else []


def parse_self_judgement(text: str) -> Optional[Dict]:
    """Extract the judge's novelty verdict from a judge-with-papers response.

    Mirrors evaluate_idea_pointwise's aggregation: parse the verdict JSON object
    (the part before [PAPERS_START]) and collapse its per-criterion 0/1 labels into a
    single `novelty` label by majority (usually just one criterion: "novelty"). Keeps
    the raw per-criterion dict + reasoning. Returns None when no valid verdict is present.
    """
    if not text:
        return None
    verdict_text = text.split('[PAPERS_START]', 1)[0]
    criterion_labels_by_name = extract_json_choice(verdict_text)
    if not isinstance(criterion_labels_by_name, dict):
        return None

    criterion_labels = [v for v in criterion_labels_by_name.values() if isinstance(v, int) and v in (0, 1)]
    if not criterion_labels:
        return None
    novelty = 1 if sum(criterion_labels) > len(criterion_labels) / 2 else 0
    return {
        "novelty": novelty,
        "criteria": criterion_labels_by_name,
        "reasoning": extract_thinking_process(verdict_text) or "",
    }


def _normalize_arxiv_id(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r'(?i)^arxiv:', '', s).strip()
    s = re.sub(r'v\d+$', '', s)  # strip version suffix (e.g. 2106.15928v2)
    return s.lower()


# ---------------------------------------------------------------------------
# Grounding via Semantic Scholar
# ---------------------------------------------------------------------------

def ground_arxiv_candidates(raw_items: List[Dict], source_query: str = "web_search") -> Tuple[List[Dict], List[str]]:
    """Resolve model-returned arXiv ids into full paper records via Semantic Scholar.

    Uses the model-returned relevance_score directly (clamped to [0, 1]) so downstream
    top-k selection reflects the model's own quality judgement rather than list position.
    Returns (candidates, unresolved_arxiv_ids) — the latter for debug reporting on which
    cited ids Semantic Scholar failed to match.
    """
    id_to_item: Dict[str, Dict] = {}
    query_ids: List[str] = []
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        aid = _normalize_arxiv_id(it.get("arxiv_id", ""))
        if not aid or aid in id_to_item:
            continue
        id_to_item[aid] = it
        query_ids.append(f"ARXIV:{aid}")

    if not query_ids:
        return [], []

    try:
        papers = SEMANTIC_API.get_paper_details_batch(query_ids, fields=_SS_FIELDS)
    except Exception as e:
        LOGGER.error(f"Semantic Scholar grounding failed for {len(query_ids)} ids: {e}")
        return [], list(id_to_item.keys())

    candidates: List[Dict] = []
    matched_ids = set()
    for p in papers:
        if not p:
            continue
        ext = p.get("externalIds") or {}
        aid = _normalize_arxiv_id(ext.get("ArXiv") or ext.get("arXiv") or ext.get("arxiv") or "")
        item = id_to_item.get(aid, {})
        if aid:
            matched_ids.add(aid)

        raw_score = item.get("relevance_score", 0)
        score = round(max(0.0, min(1.0, float(raw_score))), 4)

        abstract = p.get("abstract") or ""
        corpus_id = p.get("corpusId")
        candidates.append({
            "title": p.get("title") or "",
            "abstract": abstract,
            "text": abstract,
            "url": p.get("url") or (f"https://arxiv.org/abs/{aid}" if aid else ""),
            "corpus_id": str(corpus_id) if corpus_id is not None else None,
            "paperId": p.get("paperId"),
            "publication_date": p.get("publicationDate"),
            "year": p.get("year"),
            "arxiv_id": aid,
            "source_query": source_query,
            "relevance_judgement": {"relevance_score": score, "relevance_summary": item.get("why_retrieve", "")},
        })

    unresolved = [aid for aid in id_to_item if aid not in matched_ids]
    if unresolved:
        LOGGER.debug(f"{len(unresolved)} arXiv id(s) not resolved by Semantic Scholar: {unresolved[:5]}")
    return candidates, unresolved


def ground_and_filter(raw_items: List[Dict], cfg: WebSearchNoveltyJudgeConfig,
                      idea_text: str, idea_title: Optional[str]) -> Tuple[List[Dict], List[Dict], List[str]]:
    candidates, unresolved = ground_arxiv_candidates(raw_items)
    final, skipped, _stats = _filter_and_select_candidates(
        candidates,
        cfg.top_k_candidates,
        idea_text=idea_text,
        idea_title=idea_title,
        min_relevance_score=cfg.min_relevance_score,
        cutoff_date=cfg.cutoff_date,
    )
    return final, skipped, unresolved


# ---------------------------------------------------------------------------
# Idea iteration + cache helpers (shared by all three modes)
# ---------------------------------------------------------------------------

def iter_idea_records(inputs: Dict, cfg: "WebSearchNoveltyJudgeConfig"):
    """Yield one record per pointwise instance ({problem_id: {context, idea, metadata}}).

    This judge is pointwise-only: each instance carries a single `idea`. Each record
    carries a `cache_key` — the dataset's positional problem_id by default, or the
    content-addressed `idea_identity` when `cfg.content_addressed` is set, so the same
    cache file can be recycled across datasets. Entries that don't look pointwise (no
    `idea` field) are skipped.
    """
    for i, (problem_id, data) in enumerate(inputs.items()):
        if cfg.nr_examples is not None and i >= cfg.nr_examples:
            break
        if not isinstance(data, dict) or 'idea' not in data:
            LOGGER.warning(f"Skipping {problem_id!r}: not a pointwise instance (no 'idea' field).")
            continue
        metadata = data.get('metadata') or {}
        title = metadata.get('title') if isinstance(metadata, dict) else None
        idea_text = data['idea']
        idea_hash = idea_identity(idea_text, cfg.cutoff_date)
        yield {
            "problem_id": problem_id,
            "idea_key": "idea",
            "idea_text": idea_text,
            "idea_title": title,
            "topic": data.get('context'),
            "idea_hash": idea_hash,
            "cache_key": idea_hash if cfg.content_addressed else str(problem_id),
            "gold_label": data.get('label'),
        }


def _entry_is_complete(entry) -> bool:
    # An entry counts as cached only once it has a stored verdict. This judge always
    # produces a verdict, so an entry with candidates but no self_judgement means the
    # verdict failed to parse — treat it as incomplete so the next run retries it,
    # rather than freezing it as a permanent no-verdict (scored wrong) entry.
    return isinstance(entry, dict) and entry.get("self_judgement") is not None


def is_cached(cache: Dict, rec: Dict) -> bool:
    return _entry_is_complete(cache.get(rec["cache_key"]))


def write_cache_entry(cache: Dict, rec: Dict, candidates: List[Dict], raw_items: List[Dict],
                      self_judgement: Optional[Dict] = None):
    val = {
        "title": str(rec["idea_key"]),
        "text": rec["idea_text"],
        "related_work": select_and_format_candidates(candidates),
        "candidates": candidates,
        "retrieval_method": "web_search",
        "raw_web_search_items": raw_items,
        # Traceability when the cache is content-addressed (key != problem_id).
        "idea_hash": rec.get("idea_hash"),
        "source_problem_id": str(rec["problem_id"]),
    }
    if self_judgement is not None:
        val["self_judgement"] = self_judgement
    cache[rec["cache_key"]] = val


def _log_prompt_and_answer(rec: Dict, prompt: str, raw_text: str) -> None:
    """Emit the compiled prompt + raw model answer to the debug log (readable, multi-line)."""
    tag = f"{rec.get('problem_id')}/{rec.get('idea_key')}"
    LOGGER.debug(
        f"\n===== [{tag}] COMPILED PROMPT =====\n{prompt}\n"
        f"===== [{tag}] RAW MODEL ANSWER =====\n{raw_text}\n"
        f"===== [{tag}] END =====")


def _save_debug(output_path: str, rec: Dict, raw_items: List[Dict], raw_text: str,
                candidates: List[Dict], skipped: List[Dict], self_judgement: Optional[Dict] = None,
                prompt: Optional[str] = None, unresolved_arxiv_ids: Optional[List[str]] = None):
    idea_key = str(rec["idea_key"])
    debug_dir = os.path.join(output_path, "retrieval_debug", str(rec["problem_id"]))
    os.makedirs(debug_dir, exist_ok=True)

    debug_data = {
        "contributions": {},
        "search_queries_dict": {},
        "candidates": candidates,
        "skipped_same_title": skipped,
    }
    judge_debug = {
        "self_judgement": self_judgement,
        "raw_items": raw_items,
        "unresolved_arxiv_ids": unresolved_arxiv_ids or [],
        "prompt": prompt,
        "raw_text": raw_text,
        "gold_label": rec.get("gold_label"),
    }
    md = format_retrieval_debug_info(idea_key, rec["idea_text"], rec.get("topic") or "N/A", debug_data,
                                     judge_debug=judge_debug)
    with open(os.path.join(debug_dir, f"{idea_key}_retrieval.md"), "w") as f:
        f.write(md)

    with open(os.path.join(debug_dir, f"{idea_key}_web_search.json"), "w") as f:
        json.dump({
            "idea_title": rec.get("idea_title"),
            "raw_model_text": raw_text,
            "raw_web_search_items": raw_items,
            "self_judgement": self_judgement,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Responses output / usage helpers (batch)
# ---------------------------------------------------------------------------

def _responses_body_text(body: Dict) -> str:
    text = ""
    for item in body.get("output", []):
        if isinstance(item, dict) and item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text += content.get("text", "")
    if not text:
        text = body.get("output_text") or ""
    return text


def _responses_did_search(body: Dict) -> bool:
    return any(isinstance(it, dict) and it.get("type") == "web_search_call" for it in body.get("output", []))


def _record_responses_cost_dict(engine: str, usage: Optional[Dict], is_batch: bool = False):
    if GLOBAL_COST_TRACKER is None or not usage:
        return
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0) or 0
    total_input = usage.get("input_tokens", 0) or 0
    GLOBAL_COST_TRACKER.record(
        engine,
        input_tokens=max(total_input - cached, 0),
        output_tokens=usage.get("output_tokens", 0) or 0,
        cached_input_tokens=cached,
        is_batch=is_batch,
    )


# ---------------------------------------------------------------------------
# Mode: live
# ---------------------------------------------------------------------------

def _retrieve_live_sync(rec: Dict, cfg: WebSearchNoveltyJudgeConfig) -> Tuple[List[Dict], List[Dict], List[Dict], str, Optional[Dict], str, List[str]]:
    prompt = build_idea_prompt(rec["idea_text"], cfg.cutoff_date)
    tools = web_search_tools(cfg.allowed_domains)
    with cost_stage("retrieval"):
        # prompt_openai_client records Responses-API usage into GLOBAL_COST_TRACKER itself.
        raw_text = prompt_openai_client(prompt, engine=cfg.llm_engine,
                                        max_completion_tokens=None,
                                        reasoning={"effort": cfg.reasoning_effort}, tools=tools)
    _log_prompt_and_answer(rec, prompt, raw_text)
    raw_items = parse_web_search_items(raw_text)
    candidates, skipped, unresolved = ground_and_filter(raw_items, cfg, rec["idea_text"], rec["idea_title"])
    self_judgement = parse_self_judgement(raw_text)
    return candidates, skipped, raw_items, raw_text, self_judgement, prompt, unresolved


async def run_live(cfg: WebSearchNoveltyJudgeConfig, inputs: Dict, cache: Dict, output_path: str):
    records = [r for r in iter_idea_records(inputs, cfg) if not is_cached(cache, r)]
    LOGGER.info(f"[live] {len(records)} idea(s) to retrieve (web_search, {cfg.llm_engine}).")
    if not records:
        return

    skipped_duplicates_log: List[Dict] = []
    sem = asyncio.Semaphore(cfg.max_workers)

    async def one(rec):
        async with sem:
            candidates, skipped, raw_items, raw_text, self_judgement, prompt, unresolved = await asyncio.to_thread(_retrieve_live_sync, rec, cfg)
        _save_debug(output_path, rec, raw_items, raw_text, candidates, skipped, self_judgement,
                   prompt=prompt, unresolved_arxiv_ids=unresolved)
        for c in skipped:
            skipped_duplicates_log.append({
                "problem_id": rec["problem_id"], "idea_key": rec["idea_key"],
                "candidate_title": c.get("title", "Unknown"),
                "candidate_url": c.get("url", "#"),
                "relevance_score": (c.get("relevance_judgement") or {}).get("relevance_score", "N/A"),
                "year": c.get("year", "N/A"),
                "source_query": c.get("source_query", ""),
            })
        # Only cache complete entries (verdict present). A parsed paper list without a
        # verdict is discarded so the cache invariant stays "every entry has a verdict"
        # and the idea is retried on the next run instead of persisting a partial.
        if self_judgement is None:
            LOGGER.warning(f"No verdict parsed for problem {rec['problem_id']} idea {rec['idea_key']}.")
            return
        write_cache_entry(cache, rec, candidates, raw_items, self_judgement)

    def save_progress():
        save_retrieval_cache(cfg.output_file, cache)

    await run_in_chunks([one(r) for r in records], cfg.chunk_size, on_chunk_done=save_progress)
    save_retrieval_cache(cfg.output_file, cache)
    write_skipped_duplicates_log(skipped_duplicates_log, output_path)


# ---------------------------------------------------------------------------
# Mode: submit
# ---------------------------------------------------------------------------

def _submit_records(cfg: WebSearchNoveltyJudgeConfig, records: List[Dict], output_path: str) -> None:
    """Build one /v1/responses batch line per record, (re)write the batch map, and submit.

    The custom_id → idea map is rewritten from scratch to describe exactly the ideas in
    this batch (renumbered ws__0..N). `collect` reads it to look up each returned line and
    merges verdicts into the existing master cache, so a partial resubmit map is sufficient:
    ideas that already succeeded stay in the master untouched.
    """
    tools = web_search_tools(cfg.allowed_domains)
    requests: List[Dict] = []
    batch_map: Dict[str, Dict] = {}
    for n, rec in enumerate(records):
        custom_id = f"ws__{n}"
        prompt = build_idea_prompt(rec["idea_text"], cfg.cutoff_date)
        _log_prompt_and_answer(rec, prompt, "(batch — answer arrives at collect time)")
        requests.append(prepare_web_search_responses_request(
            custom_id, prompt, cfg.llm_engine, tools, effort=cfg.reasoning_effort))
        batch_map[custom_id] = {
            "problem_id": rec["problem_id"],
            "idea_key": rec["idea_key"],
            "idea_text": rec["idea_text"],
            "idea_title": rec["idea_title"],
            "topic": rec["topic"],
            "idea_hash": rec["idea_hash"],
            "cache_key": rec["cache_key"],
            "gold_label": rec.get("gold_label"),
        }

    map_path = os.path.join(output_path, "web_search_batch_map.json")
    with open(map_path, "w") as f:
        json.dump({"engine": cfg.llm_engine, "config": {
            "cutoff_date": cfg.cutoff_date, "top_k_candidates": cfg.top_k_candidates,
            "min_relevance_score": cfg.min_relevance_score,
        }, "map": batch_map}, f, indent=2)
    LOGGER.info(f"Saved custom_id → idea map ({len(batch_map)} entries) to {map_path}")

    submit_batch_job(requests, output_path, description="web_search_retrieval", endpoint="/v1/responses")


def run_submit(cfg: WebSearchNoveltyJudgeConfig, inputs: Dict, cache: Dict, output_path: str):
    records = [r for r in iter_idea_records(inputs, cfg) if not is_cached(cache, r)]
    LOGGER.info(f"[submit] {len(records)} idea(s) to submit (web_search batch, {cfg.llm_engine}).")
    if not records:
        LOGGER.info("Nothing to submit — all ideas already cached.")
        return
    _submit_records(cfg, records, output_path)
    LOGGER.info("Submitted web_search batch. Run with --mode collect once it completes.")


# ---------------------------------------------------------------------------
# Mode: resubmit
# ---------------------------------------------------------------------------

def run_resubmit(cfg: WebSearchNoveltyJudgeConfig, cache: Dict, output_path: str):
    """Resubmit only the previous batch's ideas that are still missing a verdict.

    Reads the last `web_search_batch_map.json` and retries every mapped idea whose
    cache_key has no complete entry in the master cache — i.e. the ones the provider
    dropped into its error file (server overload, etc.) or whose response failed to
    parse a verdict. A fresh batch is submitted and the map rewritten to match; run
    `--mode collect` once it finishes to merge the recovered verdicts into the master
    alongside the ideas that already succeeded.
    """
    map_path = os.path.join(output_path, "web_search_batch_map.json")
    if not os.path.exists(map_path):
        LOGGER.error(f"No batch map found at {map_path}. Run --mode submit first.")
        return
    with open(map_path) as f:
        prev_map = json.load(f).get("map", {})

    records: List[Dict] = []
    seen_keys = set()
    for rec in prev_map.values():
        # Tolerate maps written before content-addressing existed.
        rec.setdefault("idea_hash", idea_identity(rec["idea_text"], cfg.cutoff_date))
        rec.setdefault("cache_key", rec["idea_hash"] if cfg.content_addressed else str(rec["problem_id"]))
        if _entry_is_complete(cache.get(rec["cache_key"])):
            continue  # already has a verdict in the master — nothing to retry
        if rec["cache_key"] in seen_keys:
            continue  # same idea mapped twice — retry once
        seen_keys.add(rec["cache_key"])
        records.append(rec)

    LOGGER.info(
        f"[resubmit] {len(records)} of {len(prev_map)} mapped idea(s) still missing a "
        f"verdict — resubmitting (web_search batch, {cfg.llm_engine})."
    )
    if not records:
        LOGGER.info("Nothing to resubmit — every mapped idea already has a cached verdict.")
        return
    _submit_records(cfg, records, output_path)
    LOGGER.info("Resubmitted the missing ideas. Run with --mode collect once the new batch completes.")


# ---------------------------------------------------------------------------
# Mode: collect
# ---------------------------------------------------------------------------

def _find_latest_batch_info(output_path: str) -> Optional[Dict]:
    infos = glob.glob(os.path.join(output_path, "batch_info_*.json"))
    if not infos:
        return None
    latest = max(infos, key=lambda p: os.path.getmtime(p))
    try:
        with open(latest) as f:
            return json.load(f)
    except Exception as e:
        LOGGER.error(f"Failed to read batch info {latest}: {e}")
        return None


def run_collect(cfg: WebSearchNoveltyJudgeConfig, cache: Dict, output_path: str, batch_id: Optional[str]):
    map_path = os.path.join(output_path, "web_search_batch_map.json")
    if not os.path.exists(map_path):
        LOGGER.error(f"No batch map found at {map_path}. Run --mode submit first.")
        return
    with open(map_path) as f:
        batch_map = json.load(f).get("map", {})

    provider = "openai"
    if not batch_id:
        info = _find_latest_batch_info(output_path)
        if not info:
            LOGGER.error("No batch_id given and no batch_info_*.json found in output dir.")
            return
        batch_id = info.get("batch_id")
        provider = info.get("provider", "openai")
        LOGGER.info(f"Using batch {batch_id} (provider={provider}) from batch_info.")

    status = check_batch_status(batch_id, provider)
    if status != "completed":
        LOGGER.warning(f"Batch {batch_id} status is '{status}', not 'completed'. Try again later.")
        return

    content = retrieve_raw_batch_content(batch_id, output_path, provider)
    if not content:
        LOGGER.error("No batch content downloaded.")
        return

    n_written = n_empty = n_searched = 0
    skipped_duplicates_log: List[Dict] = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            LOGGER.warning("Skipping unparsable batch output line.")
            continue

        custom_id = obj.get("custom_id")
        rec = batch_map.get(custom_id)
        if rec is None:
            LOGGER.warning(f"No map entry for custom_id {custom_id!r}; skipping.")
            continue
        # Tolerate batch maps written before content-addressing existed.
        rec.setdefault("idea_hash", idea_identity(rec["idea_text"], cfg.cutoff_date))
        rec.setdefault("cache_key", rec["idea_hash"] if cfg.content_addressed else str(rec["problem_id"]))

        body = ((obj.get("response") or {}).get("body")) or {}
        if obj.get("error"):
            LOGGER.warning(f"Batch error for {custom_id}: {obj['error']}")
            continue

        _record_responses_cost_dict(cfg.llm_engine, body.get("usage"), is_batch=True)
        if _responses_did_search(body):
            n_searched += 1

        raw_text = _responses_body_text(body)
        prompt = build_idea_prompt(rec["idea_text"], cfg.cutoff_date)
        _log_prompt_and_answer(rec, prompt, raw_text)
        raw_items = parse_web_search_items(raw_text)
        candidates, skipped, unresolved = ground_and_filter(raw_items, cfg, rec["idea_text"], rec["idea_title"])
        self_judgement = parse_self_judgement(raw_text)
        _save_debug(output_path, rec, raw_items, raw_text, candidates, skipped, self_judgement,
                   prompt=prompt, unresolved_arxiv_ids=unresolved)

        for c in skipped:
            skipped_duplicates_log.append({
                "problem_id": rec["problem_id"], "idea_key": rec["idea_key"],
                "candidate_title": c.get("title", "Unknown"),
                "candidate_url": c.get("url", "#"),
                "relevance_score": (c.get("relevance_judgement") or {}).get("relevance_score", "N/A"),
                "year": c.get("year", "N/A"),
                "source_query": c.get("source_query", ""),
            })

        # Only cache complete entries (verdict present); a paper list without a verdict
        # is discarded so every cached entry has a verdict and the idea is retried later.
        if self_judgement is None:
            n_empty += 1
            LOGGER.warning(f"No verdict parsed for problem {rec['problem_id']} idea {rec['idea_key']} ({custom_id}).")
            continue
        write_cache_entry(cache, rec, candidates, raw_items, self_judgement)
        n_written += 1

    save_retrieval_cache(cfg.output_file, cache)
    write_skipped_duplicates_log(skipped_duplicates_log, output_path)
    LOGGER.info(f"[collect] wrote {n_written} idea(s); {n_empty} empty; {n_searched} actually ran web_search.")


# ---------------------------------------------------------------------------
# Cost report
# ---------------------------------------------------------------------------

def _write_cost_report(output_path: str, engine: str, n_instances: int):
    if GLOBAL_COST_TRACKER is None:
        return
    report = GLOBAL_COST_TRACKER.get_report()
    report["instances_processed"] = n_instances
    with open(os.path.join(output_path, "retrieval_cost_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    if _write_cost_report_md is not None:
        md_path = os.path.join(output_path, "retrieval_cost_report.md")
        _write_cost_report_md(report, md_path, title=f"Cost Report — Web-Search Retrieval (`{engine}`)")
    LOGGER.info(
        f"Estimated web_search retrieval cost: ${report['total_cost_usd']:.4f} "
        f"({report['total_calls']} calls, {report['total_input_tokens']:,} in + "
        f"{report['total_output_tokens']:,} out tokens)."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Web-search (GPT + arXiv) novelty judge that emits its cited papers.")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file.")
    parser.add_argument('--mode', choices=["live", "submit", "collect", "resubmit"], default="live")
    parser.add_argument('--batch-id', type=str, default=None,
                        help="(collect) batch id to download; defaults to latest batch_info in output dir.")
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        raw = yaml.safe_load(f) or {}
    if 'test_inputs' not in raw:
        raise ValueError("Config must specify 'test_inputs'.")

    cfg_field_names = {fld.name for fld in fields(WebSearchNoveltyJudgeConfig)}
    cfg = WebSearchNoveltyJudgeConfig(**{k: v for k, v in raw.items() if k in cfg_field_names})

    global LOGGER
    output_path = os.path.dirname(os.path.abspath(cfg.output_file))
    os.makedirs(output_path, exist_ok=True)
    if _setup_logger is not None:
        LOGGER = _setup_logger(output_dir=output_path, console_level="INFO")
        _set_common_logger(LOGGER)
    LOGGER.info(f"Loaded config from {cli_args.config} (mode={cli_args.mode})")

    if GLOBAL_COST_TRACKER is not None:
        GLOBAL_COST_TRACKER.reset()

    cache = load_retrieval_cache(cfg.output_file) if os.path.exists(cfg.output_file) else {}

    if cli_args.mode == "collect":
        run_collect(cfg, cache, output_path, cli_args.batch_id)
        _write_cost_report(output_path, cfg.llm_engine, len(cache))
        return

    if cli_args.mode == "resubmit":
        run_resubmit(cfg, cache, output_path)
        return

    inputs = read_inputs(cfg.test_inputs)
    if cli_args.mode == "submit":
        run_submit(cfg, inputs, cache, output_path)
        return

    # live
    await run_live(cfg, inputs, cache, output_path)
    _write_cost_report(output_path, cfg.llm_engine, len(inputs))


if __name__ == '__main__':
    asyncio.run(main())
