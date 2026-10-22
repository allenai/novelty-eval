import json
import os
import time
import re
import ast
from typing import Any, Optional, Dict, Tuple

import toml
from openai import OpenAI, BadRequestError

try:
    import anthropic
except ImportError:
    anthropic = None

import logging

try:
    from src.cost_tracker import GLOBAL_COST_TRACKER
except ImportError:
    try:
        from cost_tracker import GLOBAL_COST_TRACKER
    except ImportError:
        GLOBAL_COST_TRACKER = None

# Use the shared logger configured by the entry-point (via setup_logger / _attach_run_log).
# Do NOT call setup_logger here — it clears existing handlers and breaks the
# entry-point's file handler that points to the run's artifacts directory.
LOGGER = logging.getLogger("logging_utils")

# Optional hooks set by search_query_monitor.configure() — only active in web_search backend runs.
_search_query_check_hook = None    # called with List[str] on every web_search_call
_search_citation_check_hook = None  # called with a URL str on every url_citation annotation


def register_search_query_hook(hook) -> None:
    global _search_query_check_hook
    _search_query_check_hook = hook


def register_search_citation_hook(hook) -> None:
    global _search_citation_check_hook
    _search_citation_check_hook = hook


def init_secrets():
    secrets_path = os.getenv("SECRETS")
    if not secrets_path:
        # Try to find secrets.toml in the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_path = os.path.join(project_root, "secrets.toml")
        if os.path.exists(possible_path):
            secrets_path = possible_path
        else:
            LOGGER.warning("SECRETS environment variable not set and secrets.toml not found. Using empty secrets.")
            return {}

    try:
        with open(secrets_path, "r") as f:
            secrets = toml.load(f)
    except Exception as e:
        LOGGER.warning(f"Failed to load secrets from {secrets_path}: {e}. Using empty secrets.")
        return {}

    return secrets


SECRETS = init_secrets()


class _LazyOpenAIClient:
    """An OpenAI client built on first use, not on import.

    Importing this module must not need credentials: the analysis and figure code
    reaches it transitively (novelty_eval.judge -> utils) for helpers that never
    call the API, and OpenAI() raises when no key is configured. Anything that does
    call the API still gets the same error, at the first call instead of at import.
    """

    _client = None

    def __getattr__(self, name):
        if self._client is None:
            self._client = OpenAI(api_key=SECRETS.get('openai_key', None))
        return getattr(self._client, name)


OPENAI_CLIENT = _LazyOpenAIClient()

# Initialize Anthropic client lazily or if available
ANTHROPIC_CLIENT = None
if anthropic:
    # Try 'anthropic_key' first, then 'anthropic' (as used in other scripts)
    key = SECRETS.get('anthropic_key', SECRETS.get('anthropic', None))
    if key:
        ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=key, timeout=120.0)


# ---------------------------------------------------------------------------
# Provider-agnostic param builders (shared by live calls and batch requests)
# ---------------------------------------------------------------------------

# effort parameter is supported by claude-opus-4-6, claude-sonnet-4-6, and claude-opus-4-5.
# claude-sonnet-4-5 and older models do NOT support it.
_ANTHROPIC_EFFORT_SUPPORTED_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5"}


def build_anthropic_message_params(
    model: str,
    max_tokens: int,
    messages: list,
    effort: Optional[str] = None,
    system: Optional[str] = None,
) -> dict:
    """Return the params dict for ANTHROPIC_CLIENT.messages.create() or a batch request.

    Handles the effort → output_config mapping and model-support check so
    both the live path and batch path stay in sync.

    `system` is a top-level parameter for Anthropic, unlike OpenAI where it is
    an entry in the message array.
    """
    supports_effort = any(m in model for m in _ANTHROPIC_EFFORT_SUPPORTED_MODELS)
    effective_effort = ("low" if effort == "none" else effort) if supports_effort else None # no effort = none in anthropic
    params: Dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        params["system"] = system
    if effective_effort:
        params["output_config"] = {"effort": effective_effort}
    return params


def to_anthropic_messages(prompt) -> Tuple[Optional[str], list]:
    """Split an OpenAI-shaped prompt into Anthropic's (system, messages) form.

    Anthropic takes the system prompt as a top-level parameter rather than a
    message. A plain string keeps today's behaviour exactly — one user message,
    no system — so the in-house judge is unaffected.
    """
    if not isinstance(prompt, list):
        return None, [{"role": "user", "content": prompt}]

    system_parts = [
        str(m.get("content", "")) for m in prompt if m.get("role") == "system"
    ]
    messages = [
        {"role": m.get("role") or "user", "content": m.get("content", "")}
        for m in prompt if m.get("role") != "system"
    ]
    return ("\n\n".join(p for p in system_parts if p) or None), messages


def build_openai_chat_body(
    model: str,
    max_tokens: int,
    prompt: str,
    effort: Optional[str] = None,
) -> dict:
    """Return the body dict for a /v1/chat/completions request.

    Used by both the live chat-completions path and OpenAI batch requests so
    the reasoning parameter format is always consistent.
    """
    body: Dict[str, Any] = {
        "model": model,
        # `prompt` may already be a role-tagged message array — the baselines send
        # real multi-turn conversations rather than a flattened string.
        "messages": prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    if effort and effort != "none":
        body["reasoning_effort"] = effort
    return body


def build_responses_body(
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    tools: Optional[list] = None,
) -> dict:
    """Return the body dict for a /v1/responses request (Responses API).

    Used by both the live responses path and Responses API batch requests so
    the parameter format is always consistent. Omits max_output_tokens when
    max_tokens is None, letting the model generate as many tokens as needed.
    """
    body: Dict[str, Any] = {"model": model, "input": prompt}
    if max_tokens is not None:
        body["max_output_tokens"] = max_tokens
    if reasoning:
        body["reasoning"] = {**reasoning, "summary": "auto"}
    if tools:
        body["tools"] = tools
    return body


# ---------------------------------------------------------------------------
# OpenAI Responses API helpers (shared by prompt_openai_client and prompt_openai_web_search)
# ---------------------------------------------------------------------------

def _parse_responses_output(completion) -> str:
    """Extract plain text from an OpenAI Responses API completion object."""
    text = ""
    for item in completion.output:
        if item.type == "web_search_call":
            action = getattr(item, "action", None)
            queries = getattr(action, "queries", None) or []
            query = getattr(action, "query", None)
            LOGGER.debug(
                f"[web_search_call] id={getattr(item, 'id', '?')} status={getattr(item, 'status', '?')} "
                f"query={query!r} queries={queries}"
            )
            if _search_query_check_hook is not None:
                all_queries = list(queries)
                if query and query not in all_queries:
                    all_queries.append(query)
                _search_query_check_hook(all_queries)
        elif item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    text += content.text
                    annotations = getattr(content, "annotations", None) or []
                    for ann in annotations:
                        if getattr(ann, "type", None) == "url_citation":
                            url = getattr(ann, "url", "?")
                            LOGGER.debug(
                                f"[web_search_citation] title={getattr(ann, 'title', '?')!r} "
                                f"url={url} "
                                f"start={getattr(ann, 'start_index', '?')} end={getattr(ann, 'end_index', '?')}"
                            )
                            if _search_citation_check_hook is not None:
                                _search_citation_check_hook(url)
        elif item.type == "reasoning":
            summary_text = " ".join(getattr(part, "text", "") for part in (item.summary or []))
            LOGGER.debug(f"[reasoning] id={getattr(item, 'id', '?')} summary={summary_text!r}")
        else:
            LOGGER.debug(f"[responses_output_item] type={item.type} data={item}")
    return text


def _record_responses_cost(engine: str, usage) -> None:
    """Record token usage from a Responses API completion into GLOBAL_COST_TRACKER."""
    if GLOBAL_COST_TRACKER is None or usage is None:
        return
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    total_input = getattr(usage, "input_tokens", 0) or 0
    GLOBAL_COST_TRACKER.record(
        engine,
        input_tokens=total_input - cached,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cached_input_tokens=cached,
    )


def _record_chat_cost(engine: str, usage) -> None:
    """Record token usage from a Chat Completions API completion into GLOBAL_COST_TRACKER."""
    if GLOBAL_COST_TRACKER is None or usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    total_input = getattr(usage, "prompt_tokens", 0) or 0
    GLOBAL_COST_TRACKER.record(
        engine,
        input_tokens=total_input - cached,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_input_tokens=cached,
    )


# ---------------------------------------------------------------------------
# Live LLM callers
# ---------------------------------------------------------------------------

def prompt_openai_client(
    prompt: str,
    engine: str = 'o3',
    max_completion_tokens: Optional[int] = 8192,
    max_attempts: int = 3,
    reasoning=None,
    error_out: list | None = None,
    use_prompt_caching: bool = False,
    tools: Optional[list] = None,
) -> str:
    if "claude" in engine.lower():
        effort_val = reasoning.get("effort") if reasoning else None
        return prompt_anthropic_client(prompt, engine, max_completion_tokens, max_attempts, effort=effort_val, error_out=error_out, use_prompt_caching=use_prompt_caching)

    response = ""
    nr_attempts = 0
    while not response:
        LOGGER.debug(f"Prompting OpenAI with engine {engine}, reasoning={reasoning}, tools={tools}, attempt {nr_attempts + 1}/{max_attempts}")
        try:
            if tools or reasoning:
                try:
                    completion = OPENAI_CLIENT.responses.create(
                        **build_responses_body(engine, prompt, max_tokens=max_completion_tokens, reasoning=reasoning, tools=tools)
                    )
                    response = _parse_responses_output(completion)
                    _record_responses_cost(engine, getattr(completion, "usage", None))
                except AttributeError:
                    # Fallback to chat completions if responses.create doesn't exist (no tools support)
                    effort_val = reasoning.get("effort") if reasoning else None
                    completion = OPENAI_CLIENT.chat.completions.create(
                        **build_openai_chat_body(engine, max_completion_tokens or 8192, prompt, effort=effort_val)
                    )
                    response = completion.choices[0].message.content
                    _record_chat_cost(engine, getattr(completion, "usage", None))
            else:
                completion = OPENAI_CLIENT.chat.completions.create(
                    **build_openai_chat_body(engine, max_completion_tokens or 8192, prompt)
                )
                response = completion.choices[0].message.content
                _record_chat_cost(engine, getattr(completion, "usage", None))

        except BadRequestError as e:
            if e.code == "invalid_prompt":
                LOGGER.warning(f"OpenAI safety filter blocked this prompt (skipping): {e}")
            LOGGER.error(f"OpenAI BadRequestError (non-retryable): {e}")
            if error_out is not None:
                error_out.append({"code": e.code, "type": type(e).__name__, "msg": str(e)})
            return ""
        except Exception as e:
            LOGGER.error(f"OpenAI Error: {e}")
            if error_out is not None:
                code = getattr(e, "status_code", None)
                error_out.append({"code": code, "type": type(e).__name__, "msg": str(e)})
            nr_attempts += 1
            if nr_attempts >= max_attempts:
                LOGGER.error("Max attempts reached without receiving a valid response from OpenAI.")
                break
            time.sleep(5)
            continue

        if not response:
            nr_attempts += 1
            if nr_attempts >= max_attempts:
                LOGGER.error("Max attempts reached without receiving a valid response from OpenAI.")
                break
            LOGGER.warning("Received empty response from OpenAI, retrying in 5 seconds...")
            time.sleep(5)
    return response


_ANTHROPIC_MAX_TOKENS: Dict[str, int] = {}


def _anthropic_max_tokens(model: str) -> int:
    """The model's own output ceiling, cached per model."""
    if model not in _ANTHROPIC_MAX_TOKENS:
        try:
            _ANTHROPIC_MAX_TOKENS[model] = ANTHROPIC_CLIENT.models.retrieve(model).max_tokens
        except Exception as e:
            LOGGER.warning(f"Could not read max_tokens for {model} ({e}); using 8192.")
            _ANTHROPIC_MAX_TOKENS[model] = 8192
    return _ANTHROPIC_MAX_TOKENS[model]


def prompt_anthropic_client(prompt: str, model='claude-3-5-sonnet-20240620', max_tokens=8192, max_attempts=3, effort: Optional[str] = None, error_out: list | None = None, use_prompt_caching: bool = False) -> str:
    if not anthropic:
        LOGGER.error("Anthropic library not installed. Please install it to use Claude models.")
        return ""

    if not ANTHROPIC_CLIENT:
        LOGGER.error("Anthropic client not initialized. Check your secrets.toml for 'anthropic_key' or 'anthropic'.")
        return ""

    # max_tokens=None means "no cap": use the model's own ceiling and stream, since
    # the SDK refuses non-streaming requests above ~21k output tokens. Callers that
    # pass a value keep the create() path unchanged.
    stream = max_tokens is None
    if stream:
        max_tokens = _anthropic_max_tokens(model)

    response_text = ""
    nr_attempts = 0
    while not response_text:
        LOGGER.debug(f"Prompting Anthropic with model {model}, effort={effort or 'default'}, attempt {nr_attempts + 1}/{max_attempts}")
        try:
            system_text, messages = to_anthropic_messages(prompt)
            create_kwargs = build_anthropic_message_params(
                model, max_tokens, messages, effort=effort, system=system_text
            )
            if use_prompt_caching:
                create_kwargs["cache_control"] = {"type": "ephemeral"}
            if stream:
                # 10 min, matching what create() allows itself; the client default
                # of 120s is too short for a full-length response.
                with ANTHROPIC_CLIENT.messages.stream(**create_kwargs, timeout=600.0) as s:
                    message = s.get_final_message()
            else:
                message = ANTHROPIC_CLIENT.messages.create(**create_kwargs)
            if message.content:
                response_text = message.content[0].text
                if GLOBAL_COST_TRACKER is not None:
                    usage = getattr(message, "usage", None)
                    if usage is not None:
                        GLOBAL_COST_TRACKER.record(
                            model,
                            input_tokens=getattr(usage, "input_tokens", 0) or 0,
                            output_tokens=getattr(usage, "output_tokens", 0) or 0,
                            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                        )
            else:
                stop_reason = message.stop_reason
                LOGGER.warning(f"Anthropic returned empty content (stop_reason={stop_reason!r}). "
                               "This is likely caused by content safety filtering — skipping retries.")
                if error_out is not None:
                    error_out.append({
                        "code": stop_reason,
                        "type": "ContentRefusal",
                        "msg": f"Anthropic returned empty content (stop_reason={stop_reason!r})",
                    })
                break
        except Exception as e:
            nr_attempts += 1
            is_rate_limit = anthropic and isinstance(e, anthropic.RateLimitError)
            is_server_error = anthropic and isinstance(e, anthropic.APIStatusError) and e.status_code >= 500
            use_backoff = is_rate_limit or is_server_error
            wait = min(2 ** nr_attempts * 10, 120) if use_backoff else 5
            if is_rate_limit:
                LOGGER.warning(f"Anthropic rate limit hit, retrying in {wait}s (attempt {nr_attempts}/{max_attempts}): {e}")
            elif is_server_error:
                LOGGER.warning(f"Anthropic server error ({e.status_code}), retrying in {wait}s (attempt {nr_attempts}/{max_attempts}): {e}")
            else:
                LOGGER.error(f"Anthropic Error: {e}")
            if error_out is not None:
                if is_rate_limit:
                    code = 429
                else:
                    code = getattr(e, "status_code", None)
                error_out.append({"code": code, "type": type(e).__name__, "msg": str(e)})
            if nr_attempts >= max_attempts:
                LOGGER.error("Max attempts reached without receiving a valid response from Anthropic.")
                break
            time.sleep(wait)
            continue

        if not response_text:
            nr_attempts += 1
            if nr_attempts >= max_attempts:
                break
            time.sleep(5)
            
    return response_text


def save_json_artifact(output_dir: str, data: Any, filename: str) -> None:
    if not filename.endswith(".json"):
        filename += ".json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        try:
            json.dump(data, f, ensure_ascii=False, indent=4)
            LOGGER.info(f"Saved artifact to {output_path}")
        except json.decoder.JSONDecodeError:
            LOGGER.warning(f"Unable to save to {output_path}, skipping...")


def extract_json_choice(text):
    """
    Extracts and parses a JSON object from the model's response.
    """
    if not text:
        return None

    # Strip <thinking>...</thinking> blocks first — they often contain stray
    # braces that confuse the fallback {…} extractor.
    search_text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()

    json_candidates = []

    # Use regex to find content inside ```json ... ``` or just ``` ... ```
    # Updated regex to be more robust
    matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', search_text, re.DOTALL)
    if matches:
        json_candidates.extend(matches)

    # Fallback: try to find the first '{' and the last '}'
    start = search_text.find('{')
    end = search_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = search_text[start:end+1]
        # Avoid duplicates
        if candidate not in json_candidates:
            json_candidates.append(candidate)

    for json_str in json_candidates:
        # Attempt 1: Try parsing as standard JSON first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Try parsing as Python literal (handles single quotes for keys/values)
        try:
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            pass

        # Attempt 3: Replace single quotes with double quotes (legacy fix)
        # This is kept as a fallback but is risky for text containing apostrophes
        json_str_fixed = json_str.replace("'", '"')
        try:
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            pass

        # Attempt 4: Fix simple lists closed with } (e.g. ["a", "b"} -> ["a", "b"])
        # This regex looks for a list start [, content without { or [, and a closing }
        json_str_fixed_list = re.sub(r'\[([^\{\[]*?)\}', r'[\1]', json_str, flags=re.DOTALL)
        try:
            return json.loads(json_str_fixed_list)
        except json.JSONDecodeError:
            pass

    LOGGER.warning(f"Could not parse JSON as JSON: {text}")

    return None


def extract_thinking_process(text):
    """
    Extracts the thinking process from the model's response.
    """
    if not text:
        return None
    match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
