import requests
import time
from utils import SECRETS, LOGGER
import threading


class PaperFinderAPIError(Exception):
    pass


# Global stats tracker
class PaperFinderStats:
    def __init__(self):
        self.total_calls = 0
        self.current_parallel_calls = 0
        self.max_parallel_calls = 0
        self.lock = threading.Lock()
        self.call_times = []
        # Per-method HTTP request tracking
        self.current_parallel_gets = 0
        self.max_parallel_gets = 0
        self.current_parallel_posts = 0
        self.max_parallel_posts = 0
        self.current_parallel_requests = 0
        self.max_parallel_requests = 0

    def _get_stats_internal(self):
        avg_time = sum(self.call_times) / len(self.call_times) if self.call_times else 0
        min_time = min(self.call_times) if self.call_times else 0
        max_time = max(self.call_times) if self.call_times else 0
        return {
            "total_calls": self.total_calls,
            "max_parallel_calls": self.max_parallel_calls,
            "average_call_duration": avg_time,
            "min_call_duration": min_time,
            "max_call_duration": max_time,
            "current_parallel_gets": self.current_parallel_gets,
            "max_parallel_gets": self.max_parallel_gets,
            "current_parallel_posts": self.current_parallel_posts,
            "max_parallel_posts": self.max_parallel_posts,
            "current_parallel_requests": self.current_parallel_requests,
            "max_parallel_requests": self.max_parallel_requests,
        }

    def start_call(self):
        with self.lock:
            self.total_calls += 1
            self.current_parallel_calls += 1
            if self.current_parallel_calls > self.max_parallel_calls:
                self.max_parallel_calls = self.current_parallel_calls

            if self.total_calls % 100 == 0:
                LOGGER.info(f"PaperFinder Stats: {self._get_stats_internal()}")

            return time.time()

    def end_call(self, start_time):
        with self.lock:
            self.current_parallel_calls -= 1
            duration = time.time() - start_time
            self.call_times.append(duration)

    def start_http(self, method: str) -> float:
        with self.lock:
            if method == "GET":
                self.current_parallel_gets += 1
                if self.current_parallel_gets > self.max_parallel_gets:
                    self.max_parallel_gets = self.current_parallel_gets
            elif method == "POST":
                self.current_parallel_posts += 1
                if self.current_parallel_posts > self.max_parallel_posts:
                    self.max_parallel_posts = self.current_parallel_posts
            self.current_parallel_requests += 1
            if self.current_parallel_requests > self.max_parallel_requests:
                self.max_parallel_requests = self.current_parallel_requests
        return time.time()

    def end_http(self, method: str) -> None:
        with self.lock:
            if method == "GET":
                self.current_parallel_gets -= 1
            elif method == "POST":
                self.current_parallel_posts -= 1
            self.current_parallel_requests -= 1

    def get_stats(self):
        with self.lock:
            return self._get_stats_internal()

    def reset(self):
        with self.lock:
            self.total_calls = 0
            self.current_parallel_calls = 0
            self.max_parallel_calls = 0
            self.call_times = []
            self.current_parallel_gets = 0
            self.max_parallel_gets = 0
            self.current_parallel_posts = 0
            self.max_parallel_posts = 0
            self.current_parallel_requests = 0
            self.max_parallel_requests = 0

STATS = PaperFinderStats()

_MAX_PARALLEL_REQUESTS = 10
ENDPOINT = 'https://main-web.ai2i-agents.pandajungle.org'
# (connect, read) seconds. Without this requests blocks forever when the server accepts
# the connection and then never answers, which parks a worker and stalls the whole run —
# the retry budgets below only count requests that returned, so they never fire.
_REQUEST_TIMEOUT = (10, 60)
_request_semaphore = threading.Semaphore(_MAX_PARALLEL_REQUESTS)


def set_max_parallel_requests(max_parallel: int) -> None:
    global _MAX_PARALLEL_REQUESTS, _request_semaphore
    _MAX_PARALLEL_REQUESTS = max_parallel
    _request_semaphore = threading.Semaphore(max_parallel)


def poll_response(location: str, is_enriched: bool = True) -> requests.Response:
    url = f"{ENDPOINT}{location}"
    if is_enriched:
        url += "/enriched"
    STATS.start_http("GET")
    try:
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
    finally:
        STATS.end_http("GET")
    return response


def search_papers(paper_description: str, poll_interval: int = 3) -> requests.Response:
    tid = threading.get_ident()
    semaphore_wait_start = time.time()
    LOGGER.debug(f"[thread={tid}] Waiting to acquire semaphore (slots in use: {_MAX_PARALLEL_REQUESTS - _request_semaphore._value})")
    with _request_semaphore:
        semaphore_wait_s = time.time() - semaphore_wait_start
        if semaphore_wait_s > 1.0:
            LOGGER.warning(f"[thread={tid}] Semaphore acquired after {semaphore_wait_s:.1f}s wait — possible semaphore exhaustion")
        else:
            LOGGER.debug(f"[thread={tid}] Semaphore acquired after {semaphore_wait_s:.2f}s")
        result = _search_papers_impl(paper_description, poll_interval, tid)
    LOGGER.debug(f"[thread={tid}] Semaphore released")
    return result


def _search_papers_impl(paper_description: str, poll_interval: int, tid: int = 0) -> dict:
    start_time = STATS.start_call()
    try:
        LOGGER.debug(f'[thread={tid}] Searching papers with description: {paper_description}')
        url = f"{ENDPOINT}/api/3/tasks"
        data = {
            "paper_description": paper_description,
            "caller_actor_id": SECRETS.get('mabool_actor_id'),
            "generate_widget_state": False,
            "worker_profile": "standard"
        }

        max_504_retries = 3
        for attempt in range(max_504_retries + 1):
            STATS.start_http("POST")
            post_start = time.time()
            LOGGER.debug(f"[thread={tid}] Sending POST to {url} (attempt {attempt + 1})")
            try:
                response = requests.post(url, json=data, timeout=_REQUEST_TIMEOUT)
            finally:
                post_elapsed = time.time() - post_start
                STATS.end_http("POST")
            LOGGER.debug(f"[thread={tid}] POST completed in {post_elapsed:.2f}s — status={response.status_code}")
            if response.status_code != 504:
                response.raise_for_status()
                break
            LOGGER.warning(f"[thread={tid}] Got 504, retrying POST ({attempt + 1}/{max_504_retries})...")
            time.sleep(poll_interval)
        else:
            LOGGER.error(f"[thread={tid}] Exceeded max 504 retries ({max_504_retries})")

        try:
            location = response.headers['location']
            LOGGER.debug(f"[thread={tid}] Polling location: {location}")
            result = poll_response(location)

            # ~3 min at poll_interval=3s. A healthy search returns in ~40s, but under
            # load it overruns a shorter budget, and giving up looks like an empty
            # result rather than an error (the 404 body is valid JSON).
            max_404_retries = 60
            max_204_retries = 3
            retries_404 = 0
            retries_204 = 0
            poll_iteration = 0
            poll_loop_start = time.time()
            while result.status_code in {404, 204}:
                poll_iteration += 1
                elapsed = time.time() - poll_loop_start
                if result.status_code == 404:
                    retries_404 += 1
                    if retries_404 > max_404_retries:
                        LOGGER.error(f"[thread={tid}] Exceeded max 404 retries ({max_404_retries}) after {elapsed:.1f}s total polling")
                        break
                elif result.status_code == 204:
                    retries_204 += 1
                    if retries_204 > max_204_retries:
                        LOGGER.error(f"[thread={tid}] Exceeded max 204 retries ({max_204_retries}) after {elapsed:.1f}s total polling")
                        break
                if poll_iteration % 10 == 0:
                    LOGGER.warning(f"[thread={tid}] Still polling after {elapsed:.1f}s ({poll_iteration} iterations, last status={result.status_code})")
                else:
                    LOGGER.debug(f"[thread={tid}] Poll iteration {poll_iteration} — status={result.status_code}, elapsed={elapsed:.1f}s")
                time.sleep(poll_interval)
                result = poll_response(location)

            total_poll_s = time.time() - poll_loop_start
            LOGGER.debug(f"[thread={tid}] Polling finished after {total_poll_s:.1f}s ({poll_iteration} iterations), final status={result.status_code}")
            json = result.json()
        except Exception as e:
            LOGGER.error(f"[thread={tid}] PaperFinder error: {e}, status: {response.status_code}, response: {response.text}")
            raise PaperFinderAPIError(str(e)) from e

        return json
    finally:
        STATS.end_call(start_time)