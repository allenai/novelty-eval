from typing import List, Dict, Tuple
import asyncio
import os
import threading
import time
import requests
from tqdm import tqdm
from utils import SECRETS, LOGGER

MAX_BATCH = 1000  # Maximum items for batch endpoints; not used for per-page search default


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class S2RateLimiter:
    """Process-wide pacing for the S2 search/snippet endpoints (~1 RPS).

    Shared state so sync threads and asyncio coroutines can't blow past the
    limit together. The internal critical section is microseconds; the wait
    happens outside the lock. Tune via the `S2_MIN_INTERVAL` env var.
    """

    def __init__(self, min_interval: float = 1.05):
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def _reserve(self) -> float:
        with self._lock:
            now = time.monotonic()
            wait_for = self._next_at - now
            self._next_at = max(now, self._next_at) + self.min_interval
        return wait_for

    def wait(self) -> None:
        wait_for = self._reserve()
        if wait_for > 0:
            time.sleep(wait_for)

    async def wait_async(self) -> None:
        wait_for = self._reserve()
        if wait_for > 0:
            await asyncio.sleep(wait_for)


_S2_MIN_INTERVAL = float(os.getenv("S2_MIN_INTERVAL", "1.05"))
S2_SEARCH_LIMITER = S2RateLimiter(min_interval=_S2_MIN_INTERVAL)


def sanitize_s2_query(query: str) -> str:
    """Collapse whitespace, drop control chars, truncate to 256 chars.

    S2's search endpoints (especially /snippet/search) reject long or
    unprintable-character queries with a 400. Use this on any free-form text
    (idea bodies, abstracts) before sending it as a `query` param.
    """
    if not isinstance(query, str):
        return ""
    q = " ".join(query.strip().split())
    q = "".join(ch for ch in q if ch.isprintable())
    return q[:256]


class SemanticScholarAPI:
    """Interact with the Semantic Scholar API to get information about papers."""

    def __init__(self, request_timeout_sec: int = 200, backoff_time_sec: int = 5, request_items_limit: int = MAX_BATCH):
        self.api_key = SECRETS.get('semantic_scholar_key', None)
        self.request_timeout_sec = request_timeout_sec
        self.backoff_time_sec = backoff_time_sec
        self.page_size = request_items_limit
        self._session = requests.Session()
        self._headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}

    def _sanitize_query(self, query: str) -> str:
        # Kept as a method for backward-compatibility; delegates to the
        # module-level `sanitize_s2_query` so the aiohttp clients can share it.
        return sanitize_s2_query(query)

    def search_papers(self, query: str, max_results: int = 5, max_retries_per_page: int = 3,
                      fields: str = "title,authors,abstract,paperId,year",
                      publication_date_or_year: str = None) -> Dict[str, Dict]:
        """
        Search papers with safer defaults and bounded retries.

        - Sanitizes the query.
        - Limits retries per page/offset and degrades page size on repeated failures.
        - Returns what was collected so far rather than stalling indefinitely.
        - publication_date_or_year (optional): forwarded to S2's `publicationDateOrYear`
          query parameter. Examples: ":2025-06-01" (before this date), "2020-2024"
          (year range), "2018-03-23:" (since this date). See S2 API docs.
        """
        q = self._sanitize_query(query)
        if not q:
            LOGGER.warning("semantic_scholar_search: empty/invalid query after sanitization; returning empty result.")
            return {}

        collected: Dict[str, Dict] = {}
        offset = 0
        pbar = tqdm(total=max_results, desc=f'Collecting papers for query: {q}', disable=True)
        exhausted = False  # set when S2 returned fewer than `limit` -> no more pages

        while len(collected) < max_results and not exhausted:
            need = max_results - len(collected)
            limit = min(need, self.page_size)
            params = {
                "query": q,
                "fields": fields,
                "limit": limit,
                "offset": offset
            }
            if publication_date_or_year:
                params["publicationDateOrYear"] = publication_date_or_year

            attempt = 0
            backoff = self.backoff_time_sec
            while attempt < max_retries_per_page:
                attempt += 1
                try:
                    S2_SEARCH_LIMITER.wait()
                    rsp = self._session.get(
                        'https://api.semanticscholar.org/graph/v1/paper/search',
                        params=params, headers=self._headers, timeout=self.request_timeout_sec
                    )
                    if rsp.status_code == 400:
                        LOGGER.warning(
                            f'semantic_scholar_search: 400 for query "{q}" (params={params}). Aborting page.')
                        exhausted = True
                        break
                    rsp.raise_for_status()
                    data = rsp.json()
                    papers = data.get('data', []) or []

                    before = len(collected)
                    for p in papers:
                        pid = p.get('paperId')
                        if pid and pid not in collected:
                            collected[pid] = p
                    pbar.update(len(collected) - before)

                    # stop if fewer items than requested -> no more pages
                    if len(papers) < limit:
                        exhausted = True
                        offset += len(papers)
                        break

                    offset += len(papers)
                    break  # page succeeded
                except requests.exceptions.RequestException as e:
                    LOGGER.warning(f'Error fetching papers for "{q}" (attempt {attempt}/{max_retries_per_page}): {e}')
                    if attempt >= max_retries_per_page:
                        LOGGER.debug('Giving up on this page after max retries; moving on.')
                        break
                    # Exponential backoff and degrade page size to reduce load
                    LOGGER.debug(f'Retrying in {backoff} seconds.')
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 64)

            # First page completely failed: bail out early to avoid stalling
            if attempt >= max_retries_per_page and len(collected) == 0 and offset == 0 and not exhausted:
                LOGGER.warning('semantic_scholar_search: first page failed; returning empty result set.')
                break

            # Safety: if nothing new was collected during this outer cycle, break
            if need == max_results - len(collected):
                break

        return collected

    def search_snippets(self, query: str, paper_ids: list = None, max_results: int = 5, max_retries_per_page: int = 3,
                        fields: str = "snippet.text,snippet.snippetKind,snippet.section", is_sorted: bool = True) -> Dict[str, Dict]:
        """
        Search snippets with optional restriction to specific paper IDs.

        - Supports batching paper_ids in chunks of 30 (API limit) when provided.
        - Aggregates results across chunks until max_results is reached.
        """

        q = self._sanitize_query(query)
        if not q:
            LOGGER.warning("Empty/invalid query after sanitization; returning empty result.")
            return {}

        # Deduplicate and sanitize provided paper IDs
        id_chunks: List[List[str]] | None = None
        if paper_ids:
            clean_ids = [pid for pid in dict.fromkeys(paper_ids) if isinstance(pid, str) and pid.strip()]
            if not clean_ids:
                LOGGER.warning("All provided paper_ids were empty/invalid; proceeding without paper filter.")
            else:
                id_chunks = list(_chunks(clean_ids, 30))

        collected: List[Dict] = []
        pbar = tqdm(total=max_results, desc=f'Collecting snippets for query: {q}', disable=True)

        def fetch_for_ids(ids: List[str] | None):
            nonlocal collected
            offset = 0
            while len(collected) < max_results:
                need = max_results - len(collected)
                limit = min(need, self.page_size)
                params = {
                    "query": q,
                    "fields": fields,
                    "limit": limit,
                    "offset": offset
                }
                if ids:
                    params["paperIds"] = ",".join(ids)

                attempt = 0
                backoff = self.backoff_time_sec
                while attempt < max_retries_per_page:
                    attempt += 1
                    try:
                        S2_SEARCH_LIMITER.wait()
                        rsp = self._session.get(
                            'https://api.semanticscholar.org/graph/v1/snippet/search',
                            params=params, headers=self._headers, timeout=self.request_timeout_sec
                        )
                        if rsp.status_code == 400:
                            LOGGER.warning(f'400 for query "{q}" (params={params}). Aborting page.')
                            break
                        rsp.raise_for_status()
                        data = rsp.json()
                        snippets = data.get('data', []) or []
                        # LOGGER.debug(f"Snippets: {snippets}")
                        snippets = [s for s in snippets if s['snippet']['snippetKind'] not in ['title']]

                        before = len(collected)
                        collected.extend(snippets)
                        pbar.update(len(collected) - before)

                        # stop if fewer items than requested -> no more pages for this ids set
                        if len(snippets) < limit:
                            attempt = max_retries_per_page  # break inner loop
                            offset += len(snippets)
                            break

                        offset += len(snippets)
                        break  # page succeeded
                    except requests.exceptions.RequestException as e:
                        LOGGER.debug(
                            f'Error fetching snippets for "{q}" (attempt {attempt}/{max_retries_per_page}): {e}')
                        if attempt >= max_retries_per_page:
                            LOGGER.debug('Giving up on this page after max retries; moving on.')
                            break
                        # Exponential backoff
                        LOGGER.debug(f'Retrying in {backoff} seconds.')
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 64)

                # If first page for this ids set completely failed
                if attempt >= max_retries_per_page and offset == 0 and len(collected) == 0:
                    LOGGER.debug('First page failed; returning empty result set for this ids chunk.')
                    break

                # Safety: if no progress made this outer cycle, break to avoid infinite loop
                if need == max_results - len(collected):
                    break

        if id_chunks:
            for ids in id_chunks:
                if len(collected) >= max_results:
                    break
                fetch_for_ids(ids)
        else:
            fetch_for_ids(None)

        if is_sorted:
            collected = sorted(collected, key=lambda x: x['score'], reverse=True)[:max_results]

        return collected

    def get_author_details_batch(self, author_ids: List[str], fields: str = 'name,paperCount') -> List[Dict]:
        clean_ids = [aid for aid in set(author_ids) if isinstance(aid, str) and aid.strip()]
        bad_ids = [aid for aid in author_ids if aid not in clean_ids]
        if bad_ids:
            LOGGER.warning(f"Filtered out {len(bad_ids)} empty/malformed author IDs before calling the API.")

        chunk_size = min(self.page_size or MAX_BATCH, MAX_BATCH)
        collected: List[Dict] = []
        backoff_base = self.backoff_time_sec or 5

        def fetch_chunk(ids: List[str]) -> Tuple[List[Dict], List[str]]:
            backoff = backoff_base
            while True:
                try:
                    rsp = self._session.post(
                        'https://api.semanticscholar.org/graph/v1/author/batch',
                        headers=self._headers,
                        params={'fields': fields},
                        json={'ids': ids},
                        timeout=self.request_timeout_sec,
                    )
                    if rsp.status_code == 400:
                        # at least one bad id; bisect to isolate
                        if len(ids) == 1:
                            LOGGER.error(f"Bad author id rejected by API: {ids[0]} | {rsp.text[:200]}")
                            return [], ids
                        mid = len(ids) // 2
                        left_res, left_bad = fetch_chunk(ids[:mid])
                        right_res, right_bad = fetch_chunk(ids[mid:])
                        return left_res + right_res, left_bad + right_bad

                    rsp.raise_for_status()
                    return rsp.json(), []
                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    if status in (429, 500, 502, 503, 504):
                        LOGGER.debug(f"Transient {status}. Retrying in {backoff}s (chunk size {len(ids)}).")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 64)
                        continue
                    LOGGER.error(
                        f"HTTP {status} on author chunk {len(ids)}: {e} | {getattr(e.response, 'text', '')[:200]}")
                    return [], ids
                except requests.exceptions.RequestException as e:
                    LOGGER.info(f"Network error: {e}. Retrying in {backoff}s.")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 64)

        all_bad: List[str] = []
        for chunk in _chunks(clean_ids, chunk_size):
            res, bad = fetch_chunk(chunk)
            collected.extend(res)
            all_bad.extend(bad)

        if all_bad:
            LOGGER.warning(f"{len(all_bad)} author IDs failed validation. Examples: {all_bad[:5]}")
        return collected

    def map_corpus_to_paper_ids(self, corpus_ids: List[str]) -> Dict[str, str]:
        id_map: Dict[str, str] = {}
        formatted_ids = [f"CorpusId:{cid}" for cid in corpus_ids]
        papers = self.get_paper_details_batch(formatted_ids, fields='paperId,externalIds')
        for p in papers:
            ext = p.get('externalIds', {})
            cid = ext.get('CorpusId')
            LOGGER.debug(f"Mapping CorpusId {cid} to paperId {p.get('paperId')}")
            pid = p.get('paperId')
            if cid and pid:
                id_map[str(cid)] = str(pid)
        return id_map

    def get_paper_details_batch(self, paper_ids: List[str], fields: str = 'title,abstract') -> List[Dict]:
        """
        Fetch details for paper_ids using Semantic Scholar /paper/batch.
        - Retries with exponential backoff on transient errors (429/5xx).
        - If a 400 occurs, splits the chunk to isolate bad ids and logs them.
        """
        # Sanity filter for obviously bad IDs
        clean_ids = [pid for pid in set(paper_ids) if isinstance(pid, str) and pid.strip()]
        bad_ids = set(pid for pid in paper_ids if pid not in clean_ids)
        if bad_ids:
            LOGGER.warning(f"Filtered out {len(bad_ids)} empty/malformed IDs before calling the API.")

        chunk_size = min(self.page_size or MAX_BATCH, MAX_BATCH)
        collected: List[Dict] = []
        backoff_base = self.backoff_time_sec or 5

        session = requests.Session()
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json',
        }
        params = {'fields': fields}

        def fetch_chunk(ids: List[str]) -> Tuple[List[Dict], List[str]]:
            """Returns (results, bad_ids). Splits on 400 to isolate bad IDs."""
            backoff = backoff_base
            while True:
                try:
                    rsp = session.post(
                        'https://api.semanticscholar.org/graph/v1/paper/batch',
                        headers=headers,
                        params=params,
                        json={'ids': ids},
                        timeout=self.request_timeout_sec,
                    )
                    if rsp.status_code == 400:
                        # Likely one or more bad IDs. Split to find them.
                        if len(ids) == 1:
                            LOGGER.error(f"Bad paper id rejected by API: {ids[0]}. Details: {rsp.text[:200]}")
                            return [], ids  # this single id is bad
                        mid = len(ids) // 2
                        left_res, left_bad = fetch_chunk(ids[:mid])
                        right_res, right_bad = fetch_chunk(ids[mid:])
                        return left_res + right_res, left_bad + right_bad

                    rsp.raise_for_status()
                    data = rsp.json()
                    return data, []
                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    if status in (429, 500, 502, 503, 504):
                        LOGGER.debug(f"Transient {status}. Retrying in {backoff}s (chunk size {len(ids)}).")
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 64)
                        continue
                    # Non-retriable
                    LOGGER.error(
                        f"HTTP error {status} on chunk of size {len(ids)}: {e} | Body: {getattr(e.response, 'text', '')[:200]}")
                    return [], ids
                except requests.exceptions.RequestException as e:
                    LOGGER.info(f"Network error: {e}. Retrying in {backoff}s.")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 64)

        total = len(clean_ids)
        processed = 0
        all_bad_ids: List[str] = []

        for chunk in _chunks(clean_ids, chunk_size):
            LOGGER.debug(f"Requesting {len(chunk)} papers (processed {processed}/{total}).")
            results, bad = fetch_chunk(chunk)
            collected.extend(results)
            all_bad_ids.extend(bad)
            processed += len(chunk)
            LOGGER.debug(f"Collected {len(collected)} papers so far (bad in this chunk: {len(bad)}).")

        if all_bad_ids:
            LOGGER.warning(f"{len(all_bad_ids)} IDs failed validation. Examples: {all_bad_ids[:5]}")

        return collected

    def get_author_details(self, author_id: str, fields: str = 'name,affiliations,paperCount,hIndex') -> Dict:
        """Fetch author *details only* (no pagination here)."""
        backoff = self.backoff_time_sec
        params = {'fields': fields}
        while True:
            try:
                rsp = self._session.get(
                    f'https://api.semanticscholar.org/graph/v1/author/{author_id}',
                    headers=self._headers, params=params, timeout=self.request_timeout_sec
                )
                rsp.raise_for_status()
                return rsp.json()
            except requests.exceptions.RequestException as e:
                LOGGER.warning(f'Error fetching author {author_id}: {e}')
                LOGGER.info(f'Retrying in {backoff} seconds')
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)


SEMANTIC_API = SemanticScholarAPI()
