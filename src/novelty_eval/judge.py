import asyncio
import datetime
import re
from dataclasses import dataclass
from typing import Dict, Any, List
import yaml
import json
import os
import sys
from jinja2 import Environment, FileSystemLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import LOGGER, save_json_artifact, prompt_openai_client, extract_json_choice, \
    extract_thinking_process
from batch_api import prepare_batch_request, submit_batch_job
from novelty_eval.retrieval.retrieval_common import load_retrieval_cache

try:
    from cost_tracker import cost_stage
except ImportError:
    from contextlib import nullcontext as cost_stage
from novelty_eval.comparison_logger import write_comparison_to_file, write_pointwise_to_file, format_candidates_readable
from novelty_eval.scoring import change_winner_to_score

# ---------------------------------------------------------------------------
# Judge failure logging
# ---------------------------------------------------------------------------

@dataclass
class _JudgeFailure:
    problem_name: str
    idea_i: str
    idea_j: str
    kind: str          # "llm_empty" | "json_parse_fail"
    prompt: str        # full prompt text
    raw_response: str  # LLM response (first 500 chars); empty for llm_empty
    error_code: str    # API error code if available (e.g. "refusal", "invalid_prompt")
    error_type: str    # exception class name
    error_msg: str
    run_label: str     # e.g. "rr_0", "pairwise_1"


class JudgeFailureLog:
    """Collector of judge LLM failures.

    All methods are called from async coroutines within the same event loop
    (after ``await asyncio.to_thread()`` returns), so no lock is needed.
    Call ``write_md(path)`` once all experiments finish.
    """

    def __init__(self):
        self.failures: list[_JudgeFailure] = []
        self.total: int = 0
        self.successes: int = 0

    def record_success(self) -> None:
        self.total += 1
        self.successes += 1

    def record_failure(self, f: _JudgeFailure) -> None:
        self.total += 1
        self.failures.append(f)

    def write_md(self, path: str) -> None:
        """Write a human-readable Markdown failure report to *path*."""
        llm_empty = [f for f in self.failures if f.kind == "llm_empty"]
        json_fail = [f for f in self.failures if f.kind == "json_parse_fail"]

        lines = [
            "# Novelty Judge LLM Failure Debug Log",
            "",
            f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total comparisons attempted | {self.total} |",
        ]
        pct = f"{100 * self.successes / self.total:.1f}%" if self.total else "—"
        lines.append(f"| Successful | {self.successes} ({pct}) |")
        lines.append(f"| LLM empty / None response | {len(llm_empty)} |")
        lines.append(f"| JSON parse failures | {len(json_fail)} |")

        # Error code breakdown
        if self.failures:
            from collections import Counter
            code_type_counts: Counter = Counter()
            for f in llm_empty:
                key = (f.error_code or "—", f.error_type or "—")
                code_type_counts[key] += 1
            if json_fail:
                code_type_counts[("—", "json_parse_fail")] += len(json_fail)

            lines += [
                "",
                "### Error Code Breakdown",
                "",
                "| Code | Type | Count |",
                "|---|---|---|",
            ]
            for (code, etype), cnt in sorted(code_type_counts.items()):
                lines.append(f"| `{code}` | `{etype}` | {cnt} |")

        if not self.failures:
            lines += ["", "---", "", "*All judge calls succeeded — no failures recorded.*", ""]
        else:
            lines += ["", "---", "", "## Failed Comparisons", ""]
            for idx, f in enumerate(self.failures, 1):
                lines.append(
                    f"### {idx} · Problem `{f.problem_name}` · "
                    f"Ideas `{f.idea_i} vs {f.idea_j}` · `{f.kind}` · run `{f.run_label}`"
                )
                lines.append("")
                raw_cell = f"`{f.raw_response[:500]}`" if f.raw_response else "*(empty)*"
                rows = []
                if f.error_type:
                    rows.append(("Error type", f"`{f.error_type}`"))
                if f.error_code:
                    rows.append(("Error code", f"`{f.error_code}`"))
                if f.error_msg:
                    rows.append(("Error message", f"`{f.error_msg}`"))
                rows.append(("Raw response", raw_cell))
                lines += ["| Field | Value |", "|---|---|"]
                lines += [f"| {k} | {v} |" for k, v in rows]
                lines += [
                    "",
                    "<details>",
                    "<summary>Full prompt</summary>",
                    "",
                    f.prompt,
                    "",
                    "</details>",
                    "",
                    "---",
                    "",
                ]

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        LOGGER.info(f"LLM failures debug log written to: {path}")


NOVELTY_PAIRWISE_TIP = (
    "**Tip**: Think about the main contributions of each idea. Which makes more novel contributions? Which makes relatively more incremental contributions?"
)

NOVELTY_RETRIEVAL_PAIRWISE_TIP = (
    "**Tip**: Think about the main contributions of each idea. Which makes more novel contributions over prior work? Which makes relatively more incremental contributions over prior work?"
)

EVALUATION_CRITERIA = {
    "novelty": (
        """
         The substantive originality of the idea, characterized by:
         - The introduction of new frameworks, concepts, evaluations, resources or approaches absent from existing knowledge.
         - The non-trivial synthesis of existing frameworks, concepts, evaluations, resources or approaches.
         - The extension of established frameworks, concepts, evaluations, resources or approaches into previously unexplored problem domains.

         In general, composite methods / multi-stage pipelines built from several existing motifs in haphazard manner, stringing together all kinds of common methods, are NOT to be considered novel, except if the composition is highly different than any existing work, involves substantial creativity, and the combination is highly non-trivial.

         Do NOT rely on assumptions of all kinds of unknown hypotheticals. For example, "this could be novel IF formulated as..." is NOT acceptable as a reason to judge an idea as novel. Only use what is explicitly stated.

         Do NOT assume a new model/training method/technical algorithmic/mathematical contribution is more novel than a new analysis, new evaluation, new task, or new resource. All types of contributions could either be novel, or not, depending on context and the state of scientific knowledge.
        """
    ),
}

EVALUATION_CRITERIA_RETRIEVAL = {
    "novelty": (
        """
        The substantive originality of the idea in relation to the provided related work, characterized by:
         - The introduction of new frameworks, concepts, evaluations, resources or approaches absent from existing knowledge.
         - The non-trivial synthesis of existing frameworks, concepts, evaluations, resources or approaches.
         - The extension of established frameworks, concepts, evaluations, resources or approaches into previously unexplored problem domains.

          In general, composite methods / multi-stage pipelines built from several existing motifs in haphazard manner, stringing together all kinds of common methods, are NOT to be considered novel, except if the composition is highly different than any existing work, involves substantial creativity, and the combination is highly non-trivial.

         Do NOT rely on assumptions of all kinds of unknown hypotheticals. For example, "this could be novel IF formulated as..." is NOT acceptable as a reason to judge an idea as novel. Only use what is explicitly stated.

         Do NOT assume a new model/training method/technical algorithmic/mathematical contribution is more novel than a new analysis, new evaluation, new task, or new resource. All types of contributions could either be novel, or not, depending on context and the state of scientific knowledge.
        """
    ),
}

# Initialize Jinja2 environment
current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, 'templates')
env = Environment(loader=FileSystemLoader(templates_dir))

# Template selection — overridden by run_benchmark.py when judge_template is set in config
_PAIRWISE_JUDGE_TEMPLATE: str = "pairwise_novelty.jinja2"
_POINTWISE_JUDGE_TEMPLATE: str = "pointwise_novelty.jinja2"


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def get_pairwise_novelty_prompt(idea0, idea1, evaluation_criteria,
                                pairwise_tip: str | None = None,
                                cutoff_date: str | None = None):
    template = env.get_template(_PAIRWISE_JUDGE_TEMPLATE)
    criteria = dict(evaluation_criteria)
    if pairwise_tip and "novelty" in criteria:
        novelty_text = criteria["novelty"]
        first_content_line = next((l for l in novelty_text.splitlines() if l.strip()), "")
        indent = " " * (len(first_content_line) - len(first_content_line.lstrip()))
        criteria["novelty"] = novelty_text.rstrip() + "\n\n" + indent + pairwise_tip
    return template.render(
        idea0=idea0,
        idea1=idea1,
        evaluation_criteria=criteria,
        cutoff_date=cutoff_date,
    )


def get_pointwise_novelty_prompt(idea_text: str, evaluation_criteria: Dict,
                                  related_work: str = "",
                                  cutoff_date: str | None = None) -> str:
    template = env.get_template(_POINTWISE_JUDGE_TEMPLATE)
    return template.render(
        idea={"text": idea_text, "related_work": related_work},
        evaluation_criteria=evaluation_criteria,
        cutoff_date=cutoff_date,
    )


async def evaluate_idea_pointwise(
    problem_name,
    idea_text: str,
    llm_engine: str,
    semaphore: asyncio.Semaphore,
    evaluation_criteria: Dict,
    effort: str = "high",
    related_work: str = "",
    failure_log: "JudgeFailureLog | None" = None,
    run_label: str = "",
    mec_k: int = 1,
    output_path: str = "",
    use_batch_api: bool = False,
    use_prompt_caching: bool = False,
    tools: list | None = None,
    cutoff_date: str | None = None,
) -> int | list | None:
    """Call the LLM mec_k times to classify a single idea as novel (1) or not novel (0).

    Runs mec_k independent calls in parallel and aggregates via majority vote across
    all successful responses. Returns the integer prediction (0 or 1), or None if all
    calls failed.
    """
    prompt = get_pointwise_novelty_prompt(idea_text, evaluation_criteria, related_work, cutoff_date=cutoff_date)
    LOGGER.debug(f"[pointwise prompt] instance={problem_name}\n{prompt}\n{'=' * 80}")

    if use_batch_api:
        batch_requests = []
        for k in range(mec_k):
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", str(problem_name))[:40]
            custom_id = f"ptw__{safe_name}__{k}"
            req = prepare_batch_request(custom_id, prompt, llm_engine, effort=effort)
            batch_requests.append(req)
        return batch_requests

    call_counter = [0]

    async def _single_call() -> int | None:
        call_counter[0] += 1
        call_index = call_counter[0]
        async with semaphore:
            error_out: list = []
            with cost_stage("judge_pointwise"):
                response = await asyncio.to_thread(
                    prompt_openai_client, prompt, engine=llm_engine, max_attempts=3,
                    reasoning={"effort": effort}, error_out=error_out,
                    use_prompt_caching=use_prompt_caching,
                    tools=tools,
                )

        if not response:
            LOGGER.warning(f"Pointwise LLM returned empty response for problem {problem_name}.")
            if failure_log is not None:
                err = error_out[0] if error_out else {}
                failure_log.record_failure(_JudgeFailure(
                    problem_name=str(problem_name), idea_i=str(problem_name), idea_j="",
                    kind="llm_empty", prompt=prompt, raw_response="",
                    error_code=str(err["code"]) if err.get("code") is not None else "",
                    error_type=err.get("type", ""), error_msg=err.get("msg", ""),
                    run_label=run_label,
                ))
            return None

        parsed = extract_json_choice(response)
        thinking_process = extract_thinking_process(response)

        if output_path:
            write_pointwise_to_file(
                output_path, str(problem_name), idea_text, related_work,
                thinking_process, parsed, call_index,
            )

        if not parsed:
            LOGGER.warning(f"Pointwise JSON parse failed for problem {problem_name}. Response: {response[:200]}")
            if failure_log is not None:
                failure_log.record_failure(_JudgeFailure(
                    problem_name=str(problem_name), idea_i=str(problem_name), idea_j="",
                    kind="json_parse_fail", prompt=prompt, raw_response=response[:500],
                    error_code="", error_type="", error_msg="json parse failed",
                    run_label=run_label,
                ))
            return None

        if failure_log is not None:
            failure_log.record_success()

        # The template returns one key per criterion (e.g. {"novelty": 0|1}).
        # We aggregate by majority vote across criteria (usually just one: "novelty").
        votes = [v for v in parsed.values() if isinstance(v, int) and v in (0, 1)]
        if not votes:
            LOGGER.warning(f"No valid 0/1 votes found for problem {problem_name}.")
            return None

        return 1 if sum(votes) > len(votes) / 2 else 0

    results = await asyncio.gather(*[_single_call() for _ in range(mec_k)])
    valid_votes = [r for r in results if r is not None]

    if not valid_votes:
        return None

    return 1 if sum(valid_votes) > len(valid_votes) / 2 else 0


def call_llm(full_prompt: str, system_prompt: str, model_name: str) -> str:
    combined_prompt = f"{system_prompt}\n\n{full_prompt}"
    return prompt_openai_client(combined_prompt, engine=model_name)


def call_llm_json(full_prompt: str, system_prompt: str, model_name: str) -> Dict:
    response_text = call_llm(full_prompt, system_prompt, model_name)
    return extract_json_choice(response_text)





# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_judge_response(response: str, evaluation_criteria: Dict) -> Dict | None:
    return extract_json_choice(response)


# ---------------------------------------------------------------------------
# Core async comparison
# ---------------------------------------------------------------------------

async def judge_idea(i, j, idea0, idea1, llm_engine: str, semaphore: asyncio.Semaphore,
                     output_path: str, problem_name: str, evaluation_criteria: Dict, effort: str = "none",
                     failure_log: "JudgeFailureLog | None" = None, run_label: str = "",
                     use_prompt_caching: bool = False, use_batch_api: bool = False,
                     call_index: int = 0, pairwise_tip: str | None = None,
                     tools: list | None = None, cutoff_date: str | None = None):
    prompt = get_pairwise_novelty_prompt(idea0, idea1, evaluation_criteria, pairwise_tip, cutoff_date=cutoff_date)

    if use_batch_api:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", str(problem_name))[:40]
        custom_id = f"cmp__{safe_name}__{i}__{j}__{call_index}"
        return i, j, prepare_batch_request(custom_id, prompt, llm_engine, effort=effort)

    async with semaphore:
        LOGGER.debug(f"[pairwise prompt] problem={problem_name} ideas={i} vs {j}\n{prompt}\n{'=' * 80}")
        LOGGER.debug(f"Starting LLM call for ideas {i} vs {j}")
        error_out: list = []
        with cost_stage("judge_pairwise"):
            response = await asyncio.to_thread(
                prompt_openai_client, prompt, engine=llm_engine, max_attempts=3,
                reasoning={"effort": effort}, error_out=error_out,
                use_prompt_caching=use_prompt_caching,
                tools=tools,
            )
        LOGGER.debug(f"Finished LLM call for ideas {i} vs {j}")

        if not response:
            def _idea_snippet(idea) -> str:
                text = idea.get("text", "") if isinstance(idea, dict) else str(idea)
                first_sentence = text.split(".")[0].strip()
                return first_sentence[:120] if len(first_sentence) > 120 else first_sentence

            LOGGER.warning(
                f"LLM returned None for ideas {i} vs {j} (blocked or failed). Skipping comparison.\n"
                f"  idea[{i}]: {_idea_snippet(idea0)}\n"
                f"  idea[{j}]: {_idea_snippet(idea1)}"
            )
            if failure_log is not None:
                err = error_out[0] if error_out else {}
                failure_log.record_failure(_JudgeFailure(
                    problem_name=str(problem_name), idea_i=str(i), idea_j=str(j),
                    kind="llm_empty", prompt=prompt, raw_response="",
                    error_code=str(err["code"]) if err.get("code") is not None else "",
                    error_type=err.get("type", ""), error_msg=err.get("msg", ""),
                    run_label=run_label,
                ))
            return i, j, {}

        json_response = parse_judge_response(response, evaluation_criteria)
        thinking_process = extract_thinking_process(response)

        write_comparison_to_file(
            output_path, problem_name, i, j,
            idea0, idea1, thinking_process, json_response,
        )

        if json_response is None:
            LOGGER.warning(f"Could not parse JSON for evaluation of idea {i} and {j}. Response was: {response}")
            if failure_log is not None:
                failure_log.record_failure(_JudgeFailure(
                    problem_name=str(problem_name), idea_i=str(i), idea_j=str(j),
                    kind="json_parse_fail", prompt=prompt, raw_response=response[:500],
                    error_code="", error_type="", error_msg="",
                    run_label=run_label,
                ))
            return i, j, {}

        if failure_log is not None:
            failure_log.record_success()
        scores = {key: json_response.get(key, '') for key in evaluation_criteria.keys()}
        return i, j, scores


async def judge_idea_mec(
    i, j, idea0, idea1, llm_engine: str, semaphore: asyncio.Semaphore,
    output_path: str, problem_name: str, evaluation_criteria: Dict,
    effort: str = "none", mec_k: int = 1, bidirectional: bool = True,
    failure_log: "JudgeFailureLog | None" = None, run_label: str = "",
    use_prompt_caching: bool = False, use_batch_api: bool = False,
    pairwise_tip: str | None = None,
    tools: list | None = None,
    cutoff_date: str | None = None,
):
    """Run k forward + k reverse calls and pool binary scores (BPC + MEC).

    When mec_k=1 this is identical in result to the previous
    aggregate_bidirectional_winner logic (1 fwd + 1 rev, tie on disagreement).
    When bidirectional=False, only forward calls are run (no position debiasing).
    Pass tools to enable tool-augmented judging (e.g. web search).
    """
    fwd_tasks = [
        judge_idea(i, j, idea0, idea1, llm_engine, semaphore,
                   output_path, problem_name, evaluation_criteria, effort,
                   failure_log=failure_log, run_label=run_label,
                   use_prompt_caching=use_prompt_caching, use_batch_api=use_batch_api,
                   call_index=k, pairwise_tip=pairwise_tip, tools=tools,
                   cutoff_date=cutoff_date)
        for k in range(mec_k)
    ]
    fwd_results = await asyncio.gather(*fwd_tasks)
    rev_results = []
    if bidirectional:
        rev_tasks = [
            judge_idea(j, i, idea1, idea0, llm_engine, semaphore,
                       output_path, problem_name, evaluation_criteria, effort,
                       failure_log=failure_log, run_label=run_label,
                       use_prompt_caching=use_prompt_caching, use_batch_api=use_batch_api,
                       call_index=k, pairwise_tip=pairwise_tip, tools=tools,
                       cutoff_date=cutoff_date)
            for k in range(mec_k)
        ]
        rev_results = await asyncio.gather(*rev_tasks)

    if use_batch_api:
        # Collect all requests from (i, j, request) tuples
        all_requests = [r for _, _, r in fwd_results] + [r for _, _, r in rev_results]
        return i, j, all_requests

    aggregated_scores = {}
    mec_details = {}
    for dim in evaluation_criteria:
        scores_i, scores_j = [], []
        fwd_votes, rev_votes = [], []

        # Forward calls: winner=0 → i wins
        for _, _, s in fwd_results:
            if not s or dim not in s:
                continue
            try:
                w = int(s[dim])
            except (TypeError, ValueError):
                continue
            fwd_votes.append(w)
            scores_i.append(1.0 if w == 0 else (0.0 if w == 1 else 0.5))
            scores_j.append(0.0 if w == 0 else (1.0 if w == 1 else 0.5))

        # Reverse calls: in (j, i) orientation, winner=0 means j wins → i gets 0
        for _, _, s in rev_results:
            if not s or dim not in s:
                continue
            try:
                w = int(s[dim])
            except (TypeError, ValueError):
                continue
            rev_votes.append(w)
            scores_i.append(0.0 if w == 0 else (1.0 if w == 1 else 0.5))  # flipped
            scores_j.append(1.0 if w == 0 else (0.0 if w == 1 else 0.5))  # flipped

        if not scores_i:
            total_calls = mec_k * (2 if bidirectional else 1)
            def _snip(idea) -> str:
                text = idea.get("text", "") if isinstance(idea, dict) else str(idea)
                s = text.split(".")[0].strip()
                return s[:120] if len(s) > 120 else s

            LOGGER.warning(
                f"All {total_calls} MEC calls failed for dimension '{dim}' "
                f"on ideas {i} vs {j}. Scoring as tie.\n"
                f"  idea[{i}]: {_snip(idea0)}\n"
                f"  idea[{j}]: {_snip(idea1)}"
            )
            aggregated_scores[dim] = 2  # tie
            mec_details[dim] = {"fwd_votes": [], "rev_votes": [], "idea_0_scores": [], "idea_1_scores": []}
            continue

        avg_i = sum(scores_i) / len(scores_i)
        avg_j = sum(scores_j) / len(scores_j)
        if avg_i == avg_j:
            LOGGER.warning(
                f"MEC produced a tie (avg_0={avg_i:.3f} == avg_1={avg_j:.3f}) "
                f"for dim='{dim}', ideas {i} vs {j}. "
                f"Raw votes (fwd+rev): {fwd_votes + rev_votes}. "
                f"Increase mec_k (currently {mec_k}) to reduce ties."
            )
        decision = "0>1" if avg_i > avg_j else ("1>0" if avg_j > avg_i else "0=1")
        aggregated_scores[dim] = 0 if avg_i > avg_j else (1 if avg_j > avg_i else 2)
        mec_details[dim] = {
            "fwd_votes": fwd_votes,
            "rev_votes": rev_votes,
            "idea_0_scores": scores_i,
            "idea_1_scores": scores_j,
            "avg_0": round(avg_i, 4),
            "avg_1": round(avg_j, 4),
            "decision": decision,
        }

    aggregated_scores["_mec_details"] = mec_details
    return i, j, aggregated_scores


# ---------------------------------------------------------------------------
# Main entry point — delegates to tournament runners
# ---------------------------------------------------------------------------

async def evaluate_ideas(ideas, llm_engine: str, semaphore: asyncio.Semaphore, output_path: str,
                         problem_name: str, swiss_tournament: bool = False,
                         evaluation_criteria: Dict = EVALUATION_CRITERIA,
                         use_batch_api: bool = False,
                         effort: str = "none",
                         mec_k: int = 1,
                         bidirectional: bool = True,
                         failure_log: "JudgeFailureLog | None" = None,
                         run_label: str = "",
                         use_prompt_caching: bool = False,
                         pairwise_tip: str | None = NOVELTY_PAIRWISE_TIP,
                         tools: list | None = None,
                         cutoff_date: str | None = None):
    # Deferred import avoids circular imports (tournament.py imports judge_idea from here)
    from novelty_eval.tournament import (
        run_rr_tournament,
        run_swiss_tournament,
    )

    shared = dict(
        ideas=ideas,
        llm_engine=llm_engine,
        semaphore=semaphore,
        output_path=output_path,
        problem_name=problem_name,
        evaluation_criteria=evaluation_criteria,
        effort=effort,
        mec_k=mec_k,
        bidirectional=bidirectional,
        failure_log=failure_log,
        run_label=run_label,
        use_prompt_caching=use_prompt_caching,
        use_batch_api=use_batch_api,
        pairwise_tip=pairwise_tip,
        tools=tools,
        cutoff_date=cutoff_date,
    )

    if use_batch_api and swiss_tournament:
        LOGGER.warning("Batch API is not supported with Swiss Tournament. Falling back to standard API.")
        shared["use_batch_api"] = False

    if swiss_tournament:
        return await run_swiss_tournament(**shared)
    return await run_rr_tournament(**shared)


# ---------------------------------------------------------------------------
# Config / input helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    import tomllib
    LOGGER.info(f"Reading config from {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def read_inputs(input_path: str) -> Dict[str, Any]:
    LOGGER.info(f"Reading ideas from {input_path}")
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

