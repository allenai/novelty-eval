"""
search_query_monitor.py — Validates web search queries and citations stay within the knowledge cutoff.

Three checks:
  - Text queries: must contain a date constraint (before:YYYY-MM-DD, until:YYYY-MM-DD, year:YYYY).
  - ArXiv URL queries: the paper's submission month (inferred from its YYMM ID prefix) must be
    strictly before the configured cutoff date.
  - url_citation annotations: same YYMM check applied to every paper the model actually cites.

Usage (called once at run startup, web_search backend only):
    from novelty_eval import search_query_monitor
    search_query_monitor.configure(cutoff_date="2025-06-01", logger=LOGGER)

The monitor registers itself into utils._search_query_check_hook and
utils._search_citation_check_hook so it is called automatically inside
_parse_responses_output on every web_search_call and url_citation item.
"""
import re
import datetime
import logging
from typing import Optional, List

_cutoff_date: Optional[datetime.date] = None
_logger: Optional[logging.Logger] = None

_DATE_CONSTRAINT_RE = re.compile(
    r'\b(before|until|after|since):\d{4}-\d{2}-\d{2}\b', re.IGNORECASE
)
_YEAR_CONSTRAINT_RE = re.compile(r'\byear:\d{4}\b', re.IGNORECASE)
_ARXIV_ID_RE = re.compile(
    r'(?:arxiv\.org/abs/)?(\d{4}\.\d{4,5})(?:v\d+)?', re.IGNORECASE
)


def configure(cutoff_date: str, logger: logging.Logger) -> None:
    """Configure the monitor and register both check hooks into utils."""
    global _cutoff_date, _logger
    _logger = logger
    try:
        _cutoff_date = datetime.date.fromisoformat(cutoff_date)
    except (ValueError, TypeError):
        logger.warning(
            f"[search_query_monitor] invalid cutoff_date={cutoff_date!r} — monitor disabled"
        )
        _cutoff_date = None
        return

    import utils as _utils
    _utils.register_search_query_hook(check_queries)
    _utils.register_search_citation_hook(check_citation)
    logger.info(f"[search_query_monitor] active, cutoff={cutoff_date}")


def check_queries(queries: List[str]) -> None:
    """Entry-point called for every web_search_call output item."""
    if _cutoff_date is None or _logger is None:
        return
    for q in queries:
        if not q:
            continue
        if _is_arxiv_url_query(q):
            _check_arxiv_id(_ARXIV_ID_RE.search(q), context=f"query: {q!r}")
        else:
            _check_text_query(q)


def check_citation(url: str) -> None:
    """Entry-point called for every url_citation annotation URL."""
    if _cutoff_date is None or _logger is None or not url:
        return
    match = _ARXIV_ID_RE.search(url)
    if match:
        _check_arxiv_id(match, context=f"citation: {url!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_arxiv_url_query(query: str) -> bool:
    return bool(re.search(r'arxiv\.org/abs/', query, re.IGNORECASE))


def _check_text_query(query: str) -> None:
    if _DATE_CONSTRAINT_RE.search(query) or _YEAR_CONSTRAINT_RE.search(query):
        return
    _logger.warning(
        f"[search_query_monitor] text query missing date constraint: {query!r}"
    )


def _check_arxiv_id(match, context: str) -> None:
    """Warn if the arXiv paper encoded in *match* (group 1) is at or after the cutoff."""
    if not match:
        _logger.warning(
            f"[search_query_monitor] could not extract arXiv ID — {context}"
        )
        return

    arxiv_id = match.group(1)  # e.g. "2305.19118"
    yymm = arxiv_id[:4]
    try:
        year = 2000 + int(yymm[:2])
        month = int(yymm[2:])
        paper_month = datetime.date(year, month, 1)
    except ValueError:
        _logger.warning(
            f"[search_query_monitor] could not parse date from arXiv ID {arxiv_id!r} — {context}"
        )
        return

    if paper_month >= _cutoff_date:
        _logger.warning(
            f"[search_query_monitor] arXiv {arxiv_id} ({year}-{month:02d}) "
            f">= cutoff {_cutoff_date} — {context}"
        )
