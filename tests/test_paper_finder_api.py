import sys
import os
import time
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import paper_finder_api as api
from paper_finder_api import search_papers, set_max_parallel_requests, STATS, _MAX_PARALLEL_REQUESTS


def _make_mock_response(location="/api/result/1", status=200):
    post_resp = MagicMock()
    post_resp.status_code = status
    post_resp.headers = {"location": location}

    get_resp = MagicMock()
    get_resp.status_code = 200
    get_resp.json.return_value = {"papers": []}
    return post_resp, get_resp


class TestParallelRequestCap(unittest.TestCase):

    def setUp(self):
        STATS.reset()
        # Reset to a known cap before each test
        set_max_parallel_requests(5)

    def tearDown(self):
        set_max_parallel_requests(5)

    # ------------------------------------------------------------------
    # Test 1: Peak concurrency never exceeds the configured cap
    # ------------------------------------------------------------------
    def test_peak_concurrency_never_exceeds_cap(self):
        cap = 3
        set_max_parallel_requests(cap)

        post_resp, get_resp = _make_mock_response()

        def slow_post(*args, **kwargs):
            time.sleep(0.05)
            return post_resp

        with patch("paper_finder_api.requests.post", side_effect=slow_post), \
             patch("paper_finder_api.requests.get", return_value=get_resp):

            threads = [threading.Thread(target=search_papers, args=("test query",)) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        self.assertLessEqual(STATS.max_parallel_calls, cap,
                             f"max_parallel_calls={STATS.max_parallel_calls} exceeded cap={cap}")

    # ------------------------------------------------------------------
    # Test 2: Semaphore slot is released after a successful call
    # ------------------------------------------------------------------
    def test_semaphore_released_on_success(self):
        cap = 4
        set_max_parallel_requests(cap)

        post_resp, get_resp = _make_mock_response()

        with patch("paper_finder_api.requests.post", return_value=post_resp), \
             patch("paper_finder_api.requests.get", return_value=get_resp):
            search_papers("query")

        # After the call the semaphore should be fully available again
        self.assertEqual(api._request_semaphore._value, cap)

    # ------------------------------------------------------------------
    # Test 3: Semaphore slot is released even when an exception occurs
    # ------------------------------------------------------------------
    def test_semaphore_released_on_exception(self):
        cap = 4
        set_max_parallel_requests(cap)

        with patch("paper_finder_api.requests.post", side_effect=RuntimeError("network failure")):
            with self.assertRaises(RuntimeError):
                search_papers("query")

        self.assertEqual(api._request_semaphore._value, cap,
                         "Semaphore slot leaked after exception")

    # ------------------------------------------------------------------
    # Test 4: All callers complete (requests are queued, never dropped)
    # ------------------------------------------------------------------
    def test_all_callers_complete_when_queued(self):
        set_max_parallel_requests(1)

        post_resp, get_resp = _make_mock_response()

        def slow_post(*args, **kwargs):
            time.sleep(0.02)
            return post_resp

        results = []
        lock = threading.Lock()

        def call_and_collect():
            r = search_papers("query")
            with lock:
                results.append(r)

        with patch("paper_finder_api.requests.post", side_effect=slow_post), \
             patch("paper_finder_api.requests.get", return_value=get_resp):
            threads = [threading.Thread(target=call_and_collect) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        self.assertEqual(len(results), 5, "Not all callers completed — some may have been dropped")

    # ------------------------------------------------------------------
    # Test 5: set_max_parallel_requests reconfigures the cap at runtime
    # ------------------------------------------------------------------
    def test_reconfigure_cap_at_runtime(self):
        new_cap = 2
        set_max_parallel_requests(new_cap)
        STATS.reset()

        post_resp, get_resp = _make_mock_response()

        def slow_post(*args, **kwargs):
            time.sleep(0.05)
            return post_resp

        with patch("paper_finder_api.requests.post", side_effect=slow_post), \
             patch("paper_finder_api.requests.get", return_value=get_resp):

            threads = [threading.Thread(target=search_papers, args=("q",)) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        self.assertLessEqual(STATS.max_parallel_calls, new_cap,
                             f"max_parallel_calls={STATS.max_parallel_calls} exceeded new cap={new_cap}")


class TestErrorHandling(unittest.TestCase):

    @patch("paper_finder_api.requests.post")
    def test_http_500_response(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500, text="Internal Server Error")
        mock_post.return_value.raise_for_status.side_effect = Exception("500 Server Error")
        
        with self.assertRaises(Exception):
            search_papers("test query")

    @patch("paper_finder_api.requests.post")
    def test_timeout_handling(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        
        with self.assertRaises(Exception):
            search_papers("test query")

    @patch("paper_finder_api.requests.post")
    def test_http_429_too_many_requests(self, mock_post):
        mock_post.return_value = MagicMock(status_code=429, text="Too Many Requests")
        mock_post.return_value.raise_for_status.side_effect = Exception("429 Too Many Requests")
        
        with self.assertRaises(Exception):
            search_papers("test query")


class TestPaperFinderAPIError(unittest.TestCase):

    def setUp(self):
        STATS.reset()
        set_max_parallel_requests(5)

    def tearDown(self):
        set_max_parallel_requests(5)

    def test_raised_on_json_parse_failure(self):
        """A broken JSON response from the poll endpoint raises PaperFinderAPIError."""
        from paper_finder_api import PaperFinderAPIError

        post_resp, get_resp = _make_mock_response()
        get_resp.json.side_effect = ValueError("no JSON")

        with patch("paper_finder_api.requests.post", return_value=post_resp), \
             patch("paper_finder_api.requests.get", return_value=get_resp):
            with self.assertRaises(PaperFinderAPIError):
                search_papers("test query")

    def test_raised_on_missing_location_header(self):
        """A 202 POST response with no Location header raises PaperFinderAPIError."""
        from paper_finder_api import PaperFinderAPIError

        post_resp = MagicMock()
        post_resp.status_code = 202
        post_resp.headers = {}  # no 'location' key

        with patch("paper_finder_api.requests.post", return_value=post_resp):
            with self.assertRaises(PaperFinderAPIError):
                search_papers("test query")

    def test_error_message_preserved_in_exception(self):
        """The original error detail is accessible in the raised PaperFinderAPIError."""
        from paper_finder_api import PaperFinderAPIError

        post_resp, get_resp = _make_mock_response()
        get_resp.json.side_effect = ValueError("unexpected EOF")

        with patch("paper_finder_api.requests.post", return_value=post_resp), \
             patch("paper_finder_api.requests.get", return_value=get_resp):
            with self.assertRaises(PaperFinderAPIError) as ctx:
                search_papers("test query")

        self.assertIn("unexpected EOF", str(ctx.exception))

    def test_semaphore_released_after_api_error(self):
        """The global semaphore slot must be released even when PaperFinderAPIError is raised."""
        from paper_finder_api import PaperFinderAPIError

        cap = 4
        set_max_parallel_requests(cap)

        post_resp = MagicMock()
        post_resp.status_code = 202
        post_resp.headers = {}  # triggers KeyError → PaperFinderAPIError

        with patch("paper_finder_api.requests.post", return_value=post_resp):
            with self.assertRaises(PaperFinderAPIError):
                search_papers("test query")

        self.assertEqual(api._request_semaphore._value, cap,
                         "Semaphore leaked after PaperFinderAPIError")


if __name__ == "__main__":
    unittest.main()
