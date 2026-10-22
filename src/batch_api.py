"""batch_api.py — Provider-agnostic batch submission and retrieval primitives.

Supports both OpenAI and Anthropic batch APIs. Domain-specific response parsing
is left to the caller — this module handles only submit, status-check, and raw download.
"""
import json
import os
import time
from typing import Dict, List

from utils import LOGGER, OPENAI_CLIENT, save_json_artifact, build_openai_chat_body, build_anthropic_message_params, build_responses_body


def is_anthropic_model(model_name: str) -> bool:
    return "claude" in model_name.lower()


# ---------------------------------------------------------------------------
# Request preparation
# ---------------------------------------------------------------------------

def _prepare_openai_batch_request(custom_id: str, prompt: str, model_name: str, effort: str = "none") -> Dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": build_openai_chat_body(model_name, 8192, prompt, effort=effort),
    }


def _prepare_anthropic_batch_request(custom_id: str, prompt: str, model_name: str, effort: str = "none") -> Dict:
    params = build_anthropic_message_params(
        model_name, 8192, [{"role": "user", "content": prompt}], effort=effort
    )
    return {"custom_id": custom_id, "params": params}


def prepare_web_search_responses_request(
    custom_id: str,
    prompt: str,
    model_name: str,
    tools: List[Dict],
    effort: str = "medium",
) -> Dict:
    """Build a `/v1/responses` batch line that runs the hosted web_search tool.

    Unlike prepare_batch_request (which emits a tool-less /v1/chat/completions body),
    this targets the Responses API so the model can actually invoke web_search inside
    the batch job. Submit with submit_batch_job(..., endpoint="/v1/responses").
    """
    reasoning = {"effort": effort} if effort and effort != "none" else None
    body = build_responses_body(model_name, prompt, reasoning=reasoning, tools=tools)
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}


def prepare_batch_request(custom_id: str, prompt: str, model_name: str, effort: str = "none") -> Dict:
    if is_anthropic_model(model_name):
        return _prepare_anthropic_batch_request(custom_id, prompt, model_name, effort)
    return _prepare_openai_batch_request(custom_id, prompt, model_name, effort)


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def _submit_openai_batch_job(requests: List[Dict], output_path: str, description: str = "",
                             endpoint: str = "/v1/chat/completions") -> None:
    batch_file_path = os.path.join(output_path, f"batch_requests_{int(time.time())}.jsonl")
    with open(batch_file_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    LOGGER.info(f"Created batch file at {batch_file_path}")

    batch_input_file = OPENAI_CLIENT.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch",
    )
    batch_input_file_id = batch_input_file.id
    LOGGER.info(f"Uploaded batch file with ID: {batch_input_file_id}")

    metadata = {"description": description} if description else {}
    batch_job = OPENAI_CLIENT.batches.create(
        input_file_id=batch_input_file_id,
        endpoint=endpoint,
        completion_window="24h",
        metadata=metadata,
    )
    LOGGER.info(f"Created OpenAI batch job with ID: {batch_job.id}")

    batch_info = {
        "provider": "openai",
        "batch_id": batch_job.id,
        "input_file_id": batch_input_file_id,
        "file_path": batch_file_path,
        "timestamp": time.time(),
    }
    save_json_artifact(output_path, batch_info, f"batch_info_{batch_job.id}")


def _submit_anthropic_batch_job(requests: List[Dict], output_path: str) -> None:
    from utils import ANTHROPIC_CLIENT
    if ANTHROPIC_CLIENT is None:
        raise RuntimeError("Anthropic client not initialized — check 'anthropic_key' in secrets.toml")

    batch_file_path = os.path.join(output_path, f"batch_requests_{int(time.time())}.jsonl")
    with open(batch_file_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    LOGGER.info(f"Created batch file at {batch_file_path}")

    batch = ANTHROPIC_CLIENT.messages.batches.create(requests=requests)
    LOGGER.info(f"Created Anthropic batch job with ID: {batch.id}")

    batch_info = {
        "provider": "anthropic",
        "batch_id": batch.id,
        "file_path": batch_file_path,
        "timestamp": time.time(),
    }
    save_json_artifact(output_path, batch_info, f"batch_info_{batch.id}")


def submit_batch_job(requests: List[Dict], output_path: str, description: str = "",
                     endpoint: str = "/v1/chat/completions") -> None:
    """Submit requests to the appropriate provider batch API.

    Provider is auto-detected from the first request's format (Anthropic requests
    have a 'params' key; OpenAI requests have 'body'/'method'/'url'). For OpenAI,
    pass endpoint="/v1/responses" to submit web-search/Responses-API batch lines.
    """
    if not requests:
        LOGGER.warning("No requests to submit for batch job.")
        return

    try:
        if requests[0].get("params") is not None:
            _submit_anthropic_batch_job(requests, output_path)
        else:
            req_endpoint = requests[0].get("url") or endpoint
            _submit_openai_batch_job(requests, output_path, description=description, endpoint=req_endpoint)
    except Exception as e:
        LOGGER.error(f"Failed to submit batch job: {e}")


# ---------------------------------------------------------------------------
# Status polling
# ---------------------------------------------------------------------------

_OPENAI_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def check_batch_status(batch_id: str, provider: str) -> str:
    """Poll the provider API for batch_id. Returns 'pending', 'completed', or 'failed'."""
    try:
        if provider == "anthropic":
            from utils import ANTHROPIC_CLIENT
            if ANTHROPIC_CLIENT is None:
                raise RuntimeError("Anthropic client not initialized")
            batch = ANTHROPIC_CLIENT.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                counts = batch.request_counts
                return "completed" if counts.succeeded > 0 else "failed"
            return "pending"
        else:
            batch = OPENAI_CLIENT.batches.retrieve(batch_id)
            if batch.status == "completed":
                return "completed"
            if batch.status in _OPENAI_TERMINAL:
                return "failed"
            return "pending"
    except Exception as e:
        LOGGER.error(f"Error checking batch status for {batch_id}: {e}")
        return "error"


# ---------------------------------------------------------------------------
# Raw content retrieval
# ---------------------------------------------------------------------------

def retrieve_raw_batch_content(batch_id: str, output_dir: str, provider: str) -> "str | None":
    """Download the completed batch output and save it as batch_output_{batch_id}.jsonl.

    Returns the raw JSONL content string, or None if the batch is not yet complete
    or an error occurred. Domain-specific parsing is left to the caller.
    """
    try:
        if provider == "anthropic":
            from utils import ANTHROPIC_CLIENT
            if ANTHROPIC_CLIENT is None:
                LOGGER.error("Anthropic client not initialized — check 'anthropic_key' in secrets.toml")
                return None

            LOGGER.info(f"[Anthropic] Checking status for batch {batch_id}...")
            batch = ANTHROPIC_CLIENT.messages.batches.retrieve(batch_id)
            LOGGER.info(f"Processing status: {batch.processing_status}")

            if batch.processing_status != "ended":
                LOGGER.info("Batch not yet completed. Please try again later.")
                return None

            lines = []
            for result in ANTHROPIC_CLIENT.messages.batches.results(batch_id):
                try:
                    if hasattr(result, "model_dump_json"):
                        lines.append(result.model_dump_json())
                    elif hasattr(result, "to_json"):
                        lines.append(result.to_json())
                    else:
                        lines.append(json.dumps(result, default=lambda o: o.__dict__))
                except Exception as e:
                    LOGGER.warning(f"Failed to serialize raw Anthropic result: {e}")
            content = "\n".join(lines)

        else:
            LOGGER.info(f"[OpenAI] Checking status for batch {batch_id}...")
            batch = OPENAI_CLIENT.batches.retrieve(batch_id)
            LOGGER.info(f"Status: {batch.status}")

            if batch.status == "completed":
                # A "completed" OpenAI batch still routes any errored requests to a
                # SEPARATE error file that the output file omits entirely. Download it
                # so failures are visible (and re-runnable) instead of silently lost.
                error_file_id = getattr(batch, "error_file_id", None)
                if error_file_id:
                    try:
                        err_text = OPENAI_CLIENT.files.content(error_file_id).text
                        err_path = os.path.join(output_dir, f"batch_errors_{batch_id}.jsonl")
                        with open(err_path, "w") as f:
                            f.write(err_text)
                        n_err = sum(1 for ln in err_text.splitlines() if ln.strip())
                        LOGGER.warning(
                            f"Batch {batch_id} completed with {n_err} FAILED request(s), "
                            f"absent from the output file (saved to {err_path}). "
                            f"Re-run those ideas to fill the gaps."
                        )
                    except Exception as e:
                        LOGGER.error(f"Failed to download batch error file {error_file_id}: {e}")

                output_file_id = batch.output_file_id
                if not output_file_id:
                    LOGGER.warning("Batch completed but no output file ID found.")
                    return None
                LOGGER.info(f"Downloading output file {output_file_id}...")
                file_response = OPENAI_CLIENT.files.content(output_file_id)
                content = file_response.text
            elif batch.status == "failed":
                LOGGER.error(f"Batch failed. Errors: {batch.errors}")
                return None
            else:
                LOGGER.info("Batch not yet completed. Please try again later.")
                return None

        raw_output_path = os.path.join(output_dir, f"batch_output_{batch_id}.jsonl")
        with open(raw_output_path, "w") as f:
            f.write(content)
        LOGGER.info(f"Saved raw output to {raw_output_path}")
        return content

    except Exception as e:
        LOGGER.error(f"Error retrieving batch content for {batch_id}: {e}")
        return None
