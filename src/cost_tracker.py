"""
cost_tracker.py — Thread-safe global LLM cost accumulator with stage attribution.

All prompt functions in utils.py call GLOBAL_COST_TRACKER.record() after each
successful API response.  Call get_report() at any point to get a JSON-serialisable
summary, or reset() to start a fresh accounting period.

Stage tagging: use the cost_stage() context manager to label LLM calls with a
logical pipeline stage (e.g. "judge_pairwise", "retrieval").  Works with threads
and asyncio — contextvars.ContextVar is inherited by asyncio.to_thread() workers.

Pricing is in USD per 1,000,000 tokens.  Update MODEL_PRICING with the actual
figures from your reference project — entries marked # PLACEHOLDER are estimates.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Stage context variable
# ---------------------------------------------------------------------------

_current_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    'cost_stage', default='unknown'
)


@contextmanager
def cost_stage(stage: str):
    """Context manager that tags all LLM calls within its block with *stage*.

    Works transparently with threads and asyncio.to_thread():
    the ContextVar value is inherited by worker threads, so any
    prompt_openai_client() call made inside the block is tagged correctly.

    Example
    -------
    with cost_stage("judge_pairwise"):
        response = await asyncio.to_thread(prompt_openai_client, ...)
    """
    token = _current_stage.set(stage)
    try:
        yield
    finally:
        _current_stage.reset(token)


# ---------------------------------------------------------------------------
# Pricing table  (USD / 1M tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Prices are USD per 1M tokens.
    # Keys:
    #   input          — standard (non-cached) input tokens
    #   cached_input   — cache-read tokens (prompt cache hit)
    #   cache_creation — cache-write tokens (prompt cache population, 5-min TTL)
    #   output         — output / completion tokens
    #
    # Sources:
    #   OpenAI    — https://developers.openai.com/api/docs/pricing
    #   Anthropic — https://platform.claude.com/docs/en/about-claude/pricing

    # ── OpenAI ────────────────────────────────────────────────────────────────
    # Confirmed from pricing page (cached_input = 10% of input for gpt-5.x series)
    # gpt-5.6 family: Sol (flagship), Terra (balanced), Luna (cost-optimized),
    # Cyber (security). Sol/Terra/Luna are on promotional pricing, listed as
    # available at least through 2026-11-21 — recheck the pricing page after that.
    # Specific -sol/-terra/-luna/-cyber keys kept BEFORE the "gpt-5.6" alias so a
    # future dated snapshot (e.g. "gpt-5.6-terra-2026-08-01") prefix-matches its
    # own variant rather than falling through to the alias (which mirrors -sol).
    # Rates below are the standard tier; sol/terra/luna also have a long-context
    # tier (>272K input tokens) that this table does not model.
    "gpt-5.6-sol":       {"input":  4.00, "cached_input":  0.400, "cache_creation":   None, "output": 20.00},
    "gpt-5.6-terra":     {"input":  2.00, "cached_input":  0.200, "cache_creation":   None, "output": 12.00},
    "gpt-5.6-luna":      {"input":  0.20, "cached_input":  0.020, "cache_creation":   None, "output":  1.20},
    "gpt-5.6-cyber":     {"input": 12.50, "cached_input":  1.250, "cache_creation": 15.625, "output": 75.00},
    "gpt-5.6":           {"input":  4.00, "cached_input":  0.400, "cache_creation":   None, "output": 20.00},  # alias -> gpt-5.6-sol
    "gpt-5.4":           {"input":  2.50, "cached_input":  0.250, "cache_creation": None, "output": 15.00},
    "gpt-5.4-mini":      {"input":  0.75, "cached_input":  0.075, "cache_creation": None, "output":  4.50},
    "gpt-5.4-nano":      {"input":  0.20, "cached_input":  0.020, "cache_creation": None, "output":  1.25},
    "gpt-5.4-pro":       {"input": 30.00, "cached_input":   None, "cache_creation": None, "output": 180.00},
    "gpt-5.5-pro":       {"input": 30.00, "cached_input":   None, "cache_creation": None, "output": 180.00},
    # Dated snapshot the gpt-5.5-pro alias currently resolves to — priced identically.
    "gpt-5.5-2026-04-23":{"input": 30.00, "cached_input":   None, "cache_creation": None, "output": 180.00},
    # Standard tier (short-context). Kept AFTER the -pro keys so _lookup_pricing's
    # prefix fallback matches a future "gpt-5.5-pro-<date>" snapshot to -pro, not this.
    "gpt-5.5":           {"input":  5.00, "cached_input":  0.500, "cache_creation": None, "output":  30.00},
    "gpt-5.3-chat-latest":{"input": 1.75, "cached_input":  0.175, "cache_creation": None, "output": 14.00},
    "gpt-5.3-codex":     {"input":  1.75, "cached_input":  0.175, "cache_creation": None, "output": 14.00},
    # gpt-5.1 / gpt-5.2 not listed on pricing page; assume 10% cached discount
    "gpt-5.1":           {"input":  1.75, "cached_input":  0.175, "cache_creation": None, "output": 10.00},
    "gpt-5.2":           {"input":  1.75, "cached_input":  0.175, "cache_creation": None, "output": 14.00},
    # Older OpenAI models — cached_input = 50% of input (historical OpenAI rate)
    "gpt-4o":            {"input":  2.50, "cached_input":  1.250, "cache_creation": None, "output": 10.00},
    "gpt-4o-mini":       {"input":  0.15, "cached_input":  0.075, "cache_creation": None, "output":  0.60},
    "gpt-4-turbo":       {"input": 10.00, "cached_input":  5.000, "cache_creation": None, "output": 30.00},
    # o-series — cached_input = 50% of input (OpenAI reasoning models)
    "o3":                {"input":  2.00, "cached_input":  1.000, "cache_creation": None, "output":  8.00},
    "o3-mini":           {"input":  1.10, "cached_input":  0.550, "cache_creation": None, "output":  4.40},
    "o1":                {"input": 15.00, "cached_input":  7.500, "cache_creation": None, "output": 60.00},
    "o1-mini":           {"input": 15.00, "cached_input":  7.500, "cache_creation": None, "output":  4.40},

    # ── Anthropic ─────────────────────────────────────────────────────────────
    # cache_read  = 0.10× input  (confirmed from pricing page)
    # cache_write = 1.25× input  (5-min TTL; confirmed from pricing page)
    "claude-opus-4-7":            {"input":  5.00, "cached_input": 0.50, "cache_creation":  6.25, "output": 25.00},
    "claude-opus-4-6":            {"input":  5.00, "cached_input": 0.50, "cache_creation":  6.25, "output": 25.00},
    "claude-opus-4-5":            {"input":  5.00, "cached_input": 0.50, "cache_creation":  6.25, "output": 25.00},
    "claude-opus-4-1":            {"input": 15.00, "cached_input": 1.50, "cache_creation": 18.75, "output": 75.00},
    "claude-opus-4":              {"input": 15.00, "cached_input": 1.50, "cache_creation": 18.75, "output": 75.00},
    "claude-sonnet-4-6":          {"input":  3.00, "cached_input": 0.30, "cache_creation":  3.75, "output": 15.00},
    "claude-sonnet-4-5":          {"input":  3.00, "cached_input": 0.30, "cache_creation":  3.75, "output": 15.00},
    "claude-sonnet-4":            {"input":  3.00, "cached_input": 0.30, "cache_creation":  3.75, "output": 15.00},
    "claude-sonnet-3-7":          {"input":  3.00, "cached_input": 0.30, "cache_creation":  3.75, "output": 15.00},
    "claude-haiku-4-5":           {"input":  1.00, "cached_input": 0.10, "cache_creation":  1.25, "output":  5.00},
    "claude-haiku-3-5":           {"input":  0.80, "cached_input": 0.08, "cache_creation":  1.00, "output":  4.00},
    "claude-3-5-sonnet-20241022": {"input":  3.00, "cached_input": 0.30, "cache_creation":  3.75, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input":  3.00, "cached_input": 0.30, "cache_creation":  3.75, "output": 15.00},
    "claude-3-opus-20240229":     {"input": 15.00, "cached_input": 1.50, "cache_creation": 18.75, "output": 75.00},
    "claude-3-haiku-20240307":    {"input":  0.25, "cached_input": 0.03, "cache_creation":  0.30, "output":  1.25},
}

_PER_MILLION = 1_000_000.0


def _lookup_pricing(model: str) -> dict[str, float] | None:
    """Return pricing dict for *model*, trying exact then prefix match."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Prefix match: e.g. "gpt-4o-2024-08-06" → "gpt-4o"
    for key in MODEL_PRICING:
        if model.startswith(key):
            return MODEL_PRICING[key]
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelCostEntry:
    model: str
    input_tokens: int = 0            # non-cached input tokens
    cached_input_tokens: int = 0     # cache-read tokens (0.1x input price)
    cache_creation_tokens: int = 0   # cache-write tokens (1.25x input price, Anthropic only)
    output_tokens: int = 0
    call_count: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "output_tokens": self.output_tokens,
            "call_count": self.call_count,
            "cost_usd": round(self.cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Thread-safe accumulator for LLM token usage and cost, with stage attribution.

    Usage
    -----
    from cost_tracker import GLOBAL_COST_TRACKER, cost_stage
    with cost_stage("judge_pairwise"):
        response = await asyncio.to_thread(prompt_openai_client, ...)
    report = GLOBAL_COST_TRACKER.get_report()
    # report["by_stage"]["judge_pairwise"]["total_cost_usd"] -> float
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Key: (model, stage) — supports both model-level and stage-level aggregation
        self._entries: dict[tuple[str, str], ModelCostEntry] = {}
        self._unknown_models: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_creation_tokens: int = 0,
        is_batch: bool = False,
    ) -> None:
        """Accumulate token usage for *model*, tagged with the current stage.

        The stage is read from the ambient _current_stage ContextVar, which is
        set by the cost_stage() context manager.  Defaults to 'unknown'.

        Parameters
        ----------
        input_tokens:
            Non-cached input tokens (regular price).
        output_tokens:
            Output tokens (regular price).
        cached_input_tokens:
            Tokens served from the prompt cache (cache-read price per MODEL_PRICING).
            OpenAI gpt-5.x: 10% of input price. Older OpenAI/o-series: 50%.
            Anthropic: 10% of input price (0.1× multiplier).
            OpenAI: ``usage.input_tokens_details.cached_tokens``
            Anthropic: ``usage.cache_read_input_tokens``
        cache_creation_tokens:
            Tokens written into the prompt cache (cache-write price per MODEL_PRICING).
            Anthropic: 1.25× input price (5-min TTL). OpenAI: not applicable (None).
            Anthropic: ``usage.cache_creation_input_tokens``
        is_batch:
            Whether this call was made via a Batch API (OpenAI/Anthropic).
            Applies a 50% discount to the total cost.
        """
        if not input_tokens and not output_tokens and not cached_input_tokens and not cache_creation_tokens:
            return

        stage = _current_stage.get()

        pricing = _lookup_pricing(model)
        if pricing is None:
            if model not in self._unknown_models:
                self._unknown_models.add(model)
                import logging
                logging.getLogger(__name__).warning(
                    f"[CostTracker] Unknown model '{model}' — tokens recorded but cost set to $0. "
                    "Add pricing to MODEL_PRICING in cost_tracker.py."
                )
            cost = 0.0
        else:
            input_price    = pricing["input"]
            # Use explicit per-model rates; fall back to standard multipliers if not set.
            cached_price   = pricing["cached_input"]   if pricing.get("cached_input")   is not None else input_price * 0.10
            creation_price = pricing["cache_creation"] if pricing.get("cache_creation") is not None else input_price * 1.25
            cost = (
                input_tokens          / _PER_MILLION * input_price    +
                cached_input_tokens   / _PER_MILLION * cached_price   +
                cache_creation_tokens / _PER_MILLION * creation_price +
                output_tokens         / _PER_MILLION * pricing["output"]
            )

        # Apply 50% batch discount if applicable
        if is_batch:
            cost *= 0.5

        with self._lock:
            key = (model, stage)
            if key not in self._entries:
                self._entries[key] = ModelCostEntry(model=model)
            entry = self._entries[key]
            entry.input_tokens          += input_tokens
            entry.cached_input_tokens   += cached_input_tokens
            entry.cache_creation_tokens += cache_creation_tokens
            entry.output_tokens         += output_tokens
            entry.call_count            += 1
            entry.cost_usd              += cost

    def reset(self) -> None:
        """Clear all accumulated data."""
        with self._lock:
            self._entries.clear()
            self._unknown_models.clear()

    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(e.cost_usd for e in self._entries.values())

    def total_calls(self) -> int:
        with self._lock:
            return sum(e.call_count for e in self._entries.values())

    def merge_report(self, report: dict) -> None:
        """Merge an existing JSON report into this tracker."""
        with self._lock:
            # Models data
            for model_name, data in report.get("models", {}).items():
                # We use a special stage 'imported' to avoid double-counting 
                # if we later attribute by stage.
                key = (model_name, "imported")
                if key not in self._entries:
                    self._entries[key] = ModelCostEntry(model=model_name)
                entry = self._entries[key]
                entry.input_tokens          += data.get("input_tokens", 0)
                entry.cached_input_tokens   += data.get("cached_input_tokens", 0)
                entry.cache_creation_tokens += data.get("cache_creation_tokens", 0)
                entry.output_tokens         += data.get("output_tokens", 0)
                entry.call_count            += data.get("call_count", 0)
                entry.cost_usd              += data.get("cost_usd", 0.0)

            # Stage data - only merge if they don't overlap with models
            # For simplicity, we mostly rely on models data for grand totals.
            # If the report has 'by_stage', we could try to merge that too, 
            # but it's tricky to avoid double counting if we also merge 'models'.
            # Since 'models' is the source of truth for grand totals in get_report,
            # merging 'models' is sufficient for total_cost_usd().

    def get_report(self) -> dict:
        """Return a JSON-serialisable cost report with both model and stage breakdowns."""
        with self._lock:
            entries_snapshot = dict(self._entries)

        # --- Model-level aggregate (backward-compat) ---
        model_agg: dict[str, ModelCostEntry] = {}
        for (model, _stage), entry in entries_snapshot.items():
            if model not in model_agg:
                model_agg[model] = ModelCostEntry(model=model)
            agg = model_agg[model]
            agg.input_tokens          += entry.input_tokens
            agg.cached_input_tokens   += entry.cached_input_tokens
            agg.cache_creation_tokens += entry.cache_creation_tokens
            agg.output_tokens         += entry.output_tokens
            agg.call_count            += entry.call_count
            agg.cost_usd              += entry.cost_usd

        models_data = {
            name: agg.to_dict()
            for name, agg in sorted(model_agg.items(), key=lambda kv: kv[1].cost_usd, reverse=True)
        }

        # --- Stage-level aggregate (new) ---
        stage_agg: dict[str, dict] = {}
        for (_model, stage), entry in entries_snapshot.items():
            if stage not in stage_agg:
                stage_agg[stage] = {
                    "total_cost_usd": 0.0,
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_cached_input_tokens": 0,
                    "total_cache_creation_tokens": 0,
                    "total_output_tokens": 0,
                }
            sb = stage_agg[stage]
            sb["total_cost_usd"]              += entry.cost_usd
            sb["total_calls"]                 += entry.call_count
            sb["total_input_tokens"]          += entry.input_tokens
            sb["total_cached_input_tokens"]   += entry.cached_input_tokens
            sb["total_cache_creation_tokens"] += entry.cache_creation_tokens
            sb["total_output_tokens"]         += entry.output_tokens

        # Round cost values in stage_agg
        for sb in stage_agg.values():
            sb["total_cost_usd"] = round(sb["total_cost_usd"], 6)

        # Sort stages by cost descending
        stage_agg = dict(
            sorted(stage_agg.items(), key=lambda kv: kv[1]["total_cost_usd"], reverse=True)
        )

        # --- Totals ---
        total_input          = sum(e.input_tokens          for e in entries_snapshot.values())
        total_cached_input   = sum(e.cached_input_tokens   for e in entries_snapshot.values())
        total_cache_creation = sum(e.cache_creation_tokens for e in entries_snapshot.values())
        total_output         = sum(e.output_tokens         for e in entries_snapshot.values())
        total_calls          = sum(e.call_count            for e in entries_snapshot.values())
        total_cost           = sum(e.cost_usd              for e in entries_snapshot.values())

        return {
            "models": models_data,
            "by_stage": stage_agg,
            "total_input_tokens":          total_input,
            "total_cached_input_tokens":   total_cached_input,
            "total_cache_creation_tokens": total_cache_creation,
            "total_output_tokens":         total_output,
            "total_calls":                 total_calls,
            "total_cost_usd":              round(total_cost, 6),
            "pricing_note": (
                "Prices are USD/1M tokens. Entries marked PLACEHOLDER in cost_tracker.py "
                "are estimates — update MODEL_PRICING with actual rates."
            ),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

GLOBAL_COST_TRACKER = CostTracker()
