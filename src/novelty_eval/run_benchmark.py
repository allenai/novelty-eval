#!/usr/bin/env python3
"""
run_benchmark.py — Entry point for accuracy experiments.

Two test modes:
  - ranking  : instances each contain multiple ideas to rank.
                Computes NDCG, MRR, MR, Hits@k, ranking accuracy.
                Tournament modes: random, rr, swiss.
  - pairwise : instances are idea pairs.
                Computes LLM pairwise accuracy only.
                Tournament modes are not applicable; a single direct comparison
                is run per pair.
"""
import argparse
import asyncio
import yaml
import os
import sys
import random
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from novelty_eval.judge import (
    evaluate_ideas, evaluate_idea_pointwise,
    EVALUATION_CRITERIA, EVALUATION_CRITERIA_RETRIEVAL,
    NOVELTY_PAIRWISE_TIP, NOVELTY_RETRIEVAL_PAIRWISE_TIP,
    JudgeFailureLog,
)
from novelty_eval.retrieval.retrieval_common import load_retrieval_cache, select_and_format_candidates
from novelty_eval.experiment_stats import ExperimentStats
from novelty_eval.metrics import (
    compute_ranking_metrics,
    compute_pairwise_metrics,
    compute_pairwise_wrong_good,
    compute_pointwise_metrics,
    collect_run_scores,
    calculate_ranking_accuracy,
)
from novelty_eval.report_writer import save_report
from novelty_eval.md_report_writer import save_md_debug_report
from novelty_eval.retrieve_batch_results import retrieve_batch_results
from novelty_eval.comparison_logger import (
    write_comparison_logs_from_batch_results,
    write_pointwise_logs_from_batch_results,
)
from logging_utils import setup_logger, attach_run_log

try:
    from cost_tracker import GLOBAL_COST_TRACKER
except ImportError:
    GLOBAL_COST_TRACKER = None

try:
    from novelty_eval.cost_report_writer import write_cost_report_md as _write_cost_report_md
except ImportError:
    from cost_report_writer import write_cost_report_md as _write_cost_report_md

LOGGER = setup_logger(output_dir=os.getenv("OUTPUT_DIR", "."))


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def read_inputs(input_path: str) -> Dict[str, Any]:
    if not input_path.endswith(('.yaml', '.yml')):
        LOGGER.warning(
            f"test_inputs file '{input_path}' is not a YAML file. "
            "Make sure test_inputs and retrieval_cache_file are not swapped."
        )
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected test inputs to be a dict, got {type(data).__name__}. "
            f"Check that '{input_path}' is a valid test instances file."
        )
    has_expected_winners = any('expected_winners' in v for v in data.values() if isinstance(v, dict))
    has_labels = any('label' in v for v in data.values() if isinstance(v, dict))
    if not has_expected_winners and not has_labels:
        LOGGER.warning(
            f"No instances with 'expected_winners' or 'label' found in '{input_path}'. "
            "Results will be all zeros. Check that this is the correct test instances file."
        )
    return data


# ---------------------------------------------------------------------------
# Metric accumulation helpers
# ---------------------------------------------------------------------------

def _accumulate_run_metrics(
    stats: ExperimentStats,
    results: Dict[str, Any],
    test_inputs: Dict[str, Any],
    test_mode: str,
) -> Dict[str, Any]:
    """Compute metrics for one run and append to stats.

    ranking   → NDCG, MRR, MR, Hits@k
    pairwise  → LLM pairwise accuracy only
    pointwise → binary classification accuracy, precision, recall, F1

    Returns a dict of metric values for use as tqdm postfix.
    """
    if test_mode == 'pointwise':
        pw_metrics = compute_pointwise_metrics(results, test_inputs)
        stats.pointwise_accuracies.append(pw_metrics['accuracy'])
        stats.pointwise_precision_pos.append(pw_metrics['precision_pos'])
        stats.pointwise_precision_neg.append(pw_metrics['precision_neg'])
        stats.pointwise_recall_pos.append(pw_metrics['recall_pos'])
        stats.pointwise_recall_neg.append(pw_metrics['recall_neg'])
        stats.pointwise_f1_pos.append(pw_metrics['f1_pos'])
        stats.pointwise_f1_neg.append(pw_metrics['f1_neg'])
        stats.pointwise_f1_macro.append(pw_metrics['f1_macro'])
        stats.pointwise_support_pos.append(pw_metrics['support_pos'])
        stats.pointwise_support_neg.append(pw_metrics['support_neg'])
        LOGGER.debug(
            f"  pointwise_accuracy={pw_metrics['accuracy']:.4f}, "
            f"f1_macro={pw_metrics['f1_macro']:.4f}, "
            f"f1_pos={pw_metrics['f1_pos']:.4f}, f1_neg={pw_metrics['f1_neg']:.4f}"
        )
        return {
            "acc": f"{pw_metrics['accuracy']:.3f}",
            "f1_macro": f"{pw_metrics['f1_macro']:.3f}",
            "f1+": f"{pw_metrics['f1_pos']:.3f}",
            "f1-": f"{pw_metrics['f1_neg']:.3f}",
        }
    elif test_mode == 'pairwise':
        pairwise_metrics = compute_pairwise_metrics(results, test_inputs)
        acc_with_ties = pairwise_metrics["accuracy_with_ties"]
        acc_strict = pairwise_metrics["accuracy_strict"]
        acc_no_ties = pairwise_metrics["accuracy_without_ties"]
        n_ties = pairwise_metrics["n_ties"]
        support = pairwise_metrics["support"]
        support_no_ties = pairwise_metrics["support_without_ties"]
        stats.pairwise_accuracies.append(acc_with_ties)
        stats.pairwise_accuracies_strict.append(acc_strict)
        stats.pairwise_accuracies_no_ties.append(acc_no_ties)
        stats.pairwise_n_ties.append(n_ties)
        stats.pairwise_support.append(support)
        stats.pairwise_support_no_ties.append(support_no_ties)
        wrong, good = compute_pairwise_wrong_good(results, test_inputs)
        stats.wrong_pairs.append(wrong)
        stats.good_pairs.append(good)
        stats.all_scores.append(collect_run_scores(results))
        LOGGER.debug(
            f"  pairwise_accuracy_with_ties={acc_with_ties:.4f} (support={support}), "
            f"pairwise_accuracy_without_ties={acc_no_ties:.4f} (support={support_no_ties}), "
            f"n_ties={n_ties}"
        )
        return {"acc": f"{acc_with_ties:.3f}", "acc_no_ties": f"{acc_no_ties:.3f}", "ties": n_ties, "support": support}
    else:  # ranking
        ndcg, mrr, mr, hits = compute_ranking_metrics(results, test_inputs)
        stats.ndcg_scores.append(ndcg)
        stats.mrr_scores.append(mrr)
        stats.mr_scores.append(mr)
        for k in [1, 2, 3]:
            stats.hits_scores[k].append(hits[k])
        acc, wrong, good = calculate_ranking_accuracy(results, test_inputs)
        stats.accuracies.append(acc)
        stats.wrong_pairs.append(wrong)
        stats.good_pairs.append(good)
        stats.all_scores.append(collect_run_scores(results))
        LOGGER.debug(
            f"  NDCG={ndcg:.4f}, MRR={mrr:.4f}, MR={mr:.4f}, "
            f"Hits@1={hits[1]:.4f}, Hits@2={hits[2]:.4f}, Hits@3={hits[3]:.4f}, "
            f"Accuracy={acc:.4f}"
        )
        return {"NDCG": f"{ndcg:.3f}", "MRR": f"{mrr:.3f}", "Hits@1": f"{hits[1]:.3f}"}


def _finalize_stats(stats: ExperimentStats) -> None:
    """Populate mean_* fields once all runs are collected."""
    if stats.accuracies:
        stats.mean_accuracy = statistics.mean(stats.accuracies)
    if stats.ndcg_scores:
        stats.mean_ndcg = statistics.mean(stats.ndcg_scores)
    if stats.mrr_scores:
        stats.mean_mrr = statistics.mean(stats.mrr_scores)
    if stats.mr_scores:
        stats.mean_mr = statistics.mean(stats.mr_scores)
    stats.mean_hits = {
        k: statistics.mean(stats.hits_scores[k]) if stats.hits_scores[k] else 0.0
        for k in [1, 2, 3]
    }
    if stats.pairwise_accuracies:
        stats.mean_pairwise_accuracy = statistics.mean(stats.pairwise_accuracies)
    if stats.pairwise_accuracies_strict:
        stats.mean_pairwise_accuracy_strict = statistics.mean(stats.pairwise_accuracies_strict)
    if stats.pairwise_accuracies_no_ties:
        stats.mean_pairwise_accuracy_no_ties = statistics.mean(stats.pairwise_accuracies_no_ties)
    if stats.pairwise_n_ties:
        stats.mean_pairwise_n_ties = statistics.mean(stats.pairwise_n_ties)
    if stats.pairwise_support:
        stats.mean_pairwise_support = statistics.mean(stats.pairwise_support)
    if stats.pairwise_support_no_ties:
        stats.mean_pairwise_support_no_ties = statistics.mean(stats.pairwise_support_no_ties)
    if stats.pointwise_accuracies:
        stats.mean_pointwise_accuracy = statistics.mean(stats.pointwise_accuracies)
    if stats.pointwise_precision_pos:
        stats.mean_pointwise_precision_pos = statistics.mean(stats.pointwise_precision_pos)
    if stats.pointwise_precision_neg:
        stats.mean_pointwise_precision_neg = statistics.mean(stats.pointwise_precision_neg)
    if stats.pointwise_recall_pos:
        stats.mean_pointwise_recall_pos = statistics.mean(stats.pointwise_recall_pos)
    if stats.pointwise_recall_neg:
        stats.mean_pointwise_recall_neg = statistics.mean(stats.pointwise_recall_neg)
    if stats.pointwise_f1_pos:
        stats.mean_pointwise_f1_pos = statistics.mean(stats.pointwise_f1_pos)
    if stats.pointwise_f1_neg:
        stats.mean_pointwise_f1_neg = statistics.mean(stats.pointwise_f1_neg)
    if stats.pointwise_f1_macro:
        stats.mean_pointwise_f1_macro = statistics.mean(stats.pointwise_f1_macro)


# ---------------------------------------------------------------------------
# Core evaluation orchestrator (moved from main.py)
# ---------------------------------------------------------------------------

def _annotate_gt_winner(comparisons: List[Dict[str, Any]], expected_winners: set) -> None:
    """Add a 'gt_winner' field to each comparison in-place.

    gt_winner = 0  → idea_0 is the expected winner
    gt_winner = 1  → idea_1 is the expected winner
    gt_winner = None → both or neither are expected winners (no clear GT)
    """
    for comp in comparisons:
        i_is_gt = str(comp['idea_0']) in expected_winners
        j_is_gt = str(comp['idea_1']) in expected_winners
        if i_is_gt and not j_is_gt:
            comp['gt_winner'] = 0
        elif j_is_gt and not i_is_gt:
            comp['gt_winner'] = 1
        else:
            comp['gt_winner'] = None  # both winners, both losers, or no GT available

async def run_comparative_evaluation(
    inputs: Dict[str, Any],
    llm_engine: str,
    max_workers: int,
    output_path: str,
    swiss_tournament: bool = False,
    retrieve_related_work: bool = False,
    retrieval_cache: Optional[Dict[str, Any]] = None,
    retrieval_cache_path: Optional[str] = None,
    use_batch_api: bool = False,
    effort: str = "none",
    mec_k: int = 1,
    bidirectional: bool = True,
    failure_log: Optional["JudgeFailureLog"] = None,
    run_label: str = "",
    top_k_candidates: Optional[int] = None,
    use_prompt_caching: bool = False,
    tools: Optional[list] = None,
    cutoff_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Run comparative evaluation over all problems in inputs.

    Retrieval is cache-only: pass retrieval_cache_path to enable it.
    No online search calls are made.
    """
    if use_batch_api and swiss_tournament:
        LOGGER.warning("Batch API not supported with Swiss Tournament. Falling back to standard API calls.")
        use_batch_api = False

    semaphore = asyncio.Semaphore(max_workers)
    evaluation_criteria = EVALUATION_CRITERIA_RETRIEVAL if retrieve_related_work else EVALUATION_CRITERIA
    pairwise_tip = NOVELTY_RETRIEVAL_PAIRWISE_TIP if retrieve_related_work else NOVELTY_PAIRWISE_TIP

    async def evaluate_single_problem(problem, data):
        ideas = data['ideas']

        if retrieve_related_work:
            if retrieval_cache is None:
                raise ValueError(
                    "retrieve_related_work=True but no retrieval_cache provided. "
                    "Pass a pre-computed cache via retrieval_cache_path."
                )
            cached_problem = retrieval_cache.get(str(problem), {})
            enriched_ideas = {k: cached_problem[str(k)] for k in ideas if str(k) in cached_problem}
            if top_k_candidates is not None:
                for key, entry in enriched_ideas.items():
                    if entry.get('candidates') is not None:
                        enriched_ideas[key] = {**entry, 'related_work': select_and_format_candidates(entry['candidates'], top_k=top_k_candidates)}
            missing = [k for k in ideas if str(k) not in cached_problem]
            for k in missing:
                LOGGER.warning(f"No cache entry for idea {k} in problem {problem}. Skipping.")
            if not enriched_ideas:
                LOGGER.warning(f"No cached ideas for problem {problem}. Skipping evaluation.")
                return problem, {}
        else:
            enriched_ideas = {k: {"text": v, "related_work": ""} for k, v in ideas.items()}

        problem_results = await evaluate_ideas(
            enriched_ideas, llm_engine, semaphore, output_path, problem,
            swiss_tournament, evaluation_criteria,
            use_batch_api=use_batch_api, effort=effort,
            mec_k=mec_k, bidirectional=bidirectional,
            failure_log=failure_log, run_label=run_label,
            use_prompt_caching=use_prompt_caching,
            pairwise_tip=pairwise_tip,
            tools=tools,
            cutoff_date=cutoff_date,
        )

        if use_batch_api:
            return problem, problem_results

        LOGGER.debug(
            f"Problem: {problem}, elo_scores: {problem_results['elo_scores']}, "
            f"elo_selected: {problem_results['elo_selected']}, ideas: {problem_results['ideas']}"
        )
        problem_data = problem_results['dimensional_elo_scores']
        problem_data['overall'] = problem_results['elo_scores']
        problem_data['elo_selected'] = problem_results['elo_selected']
        problem_data['ideas'] = problem_results['ideas']
        problem_data['comparisons'] = problem_results.get('comparisons', [])

        expected_winners = set(str(w) for w in data.get('expected_winners', []))
        _annotate_gt_winner(problem_data['comparisons'], expected_winners)

        return problem, problem_data

    pbar_problems = tqdm(total=len(inputs), desc="  Problems", leave=False)

    async def tracked_problem(p, d):
        result = await evaluate_single_problem(p, d)
        pbar_problems.update(1)
        return result

    completed = await asyncio.gather(*[tracked_problem(p, d) for p, d in inputs.items()])
    pbar_problems.close()

    if use_batch_api:
        from novelty_eval.judge import submit_batch_job
        all_requests = [req for _, data in completed if data for req in data]
        submit_batch_job(all_requests, output_path)
        return {}

    return {problem: data for problem, data in completed if data}


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

async def run_pairwise_experiment(
    n_runs: int,
    llm_engine: str,
    output_dir: str,
    inputs: Dict[str, Any],
    max_workers: int,
    timestamp: str,
    retrieval_cache: Dict[str, Any],
    retrieval_cache_path: Optional[str],
    use_batch_api: bool,
    effort: str,
    mec_k: int = 1,
    bidirectional: bool = True,
    failure_log: Optional[JudgeFailureLog] = None,
    top_k_candidates: Optional[int] = None,
    use_prompt_caching: bool = False,
    tools: Optional[list] = None,
    cutoff_date: Optional[str] = None,
    **_ignored,
) -> ExperimentStats:
    """Pairwise test mode: each instance is a pair of ideas.

    Calls evaluate_ideas directly (no tournament flags) to guarantee a single
    direct comparison per pair.  Computes LLM pairwise accuracy only.
    """
    test_inputs = {k: v for k, v in inputs.items() if 'expected_winners' in v}
    evaluation_criteria = EVALUATION_CRITERIA_RETRIEVAL if retrieval_cache_path else EVALUATION_CRITERIA
    pairwise_tip = NOVELTY_RETRIEVAL_PAIRWISE_TIP if retrieval_cache_path else NOVELTY_PAIRWISE_TIP

    LOGGER.info(f"Running Pairwise Experiment: n={n_runs}, instances={len(test_inputs)}, max_workers={max_workers}")
    stats = ExperimentStats()

    pbar_runs = tqdm(range(n_runs), desc="Runs (Pairwise)")
    for i in pbar_runs:
        LOGGER.debug(f"Run {i + 1}/{n_runs}...")
        run_output_path = os.path.join(output_dir, "accuracy_test_artifacts", timestamp, f"run_pairwise_{i}")
        stats.run_paths.append(run_output_path)
        os.makedirs(run_output_path, exist_ok=True)

        semaphore = asyncio.Semaphore(max_workers)

        async def evaluate_problem(problem_id, data):
            ideas = data['ideas']

            if retrieval_cache_path:
                cached_problem = retrieval_cache.get(str(problem_id), {})
                enriched = {k: cached_problem[str(k)] for k in ideas if str(k) in cached_problem}
                if top_k_candidates is not None:
                    for key, entry in enriched.items():
                        if entry.get('candidates') is not None:
                            enriched[key] = {**entry, 'related_work': select_and_format_candidates(entry['candidates'], top_k=top_k_candidates)}
                if not enriched:
                    LOGGER.warning(f"No cache entries for problem {problem_id}. Skipping.")
                    return problem_id, None
            else:
                enriched = {k: {"text": v, "related_work": ""} for k, v in ideas.items()}

            # Direct comparison — swiss_tournament=False hardcoded
            problem_results = await evaluate_ideas(
                enriched, llm_engine, semaphore, run_output_path, problem_id,
                swiss_tournament=False,
                evaluation_criteria=evaluation_criteria,
                use_batch_api=use_batch_api,
                effort=effort,
                mec_k=mec_k,
                bidirectional=bidirectional,
                failure_log=failure_log,
                run_label=f"pairwise_{i}",
                use_prompt_caching=use_prompt_caching,
                pairwise_tip=pairwise_tip,
                tools=tools,
                cutoff_date=cutoff_date,
            )

            if use_batch_api:
                return problem_id, problem_results

            problem_data = problem_results['dimensional_elo_scores']
            problem_data['overall'] = problem_results['elo_scores']
            problem_data['elo_selected'] = problem_results['elo_selected']
            problem_data['ideas'] = problem_results['ideas']
            problem_data['comparisons'] = problem_results.get('comparisons', [])

            expected_winners = set(str(w) for w in data.get('expected_winners', []))
            _annotate_gt_winner(problem_data['comparisons'], expected_winners)

            return problem_id, problem_data

        pbar_problems = tqdm(total=len(test_inputs), desc="  Problems", leave=False)

        async def tracked_problem(pid, d):
            result = await evaluate_problem(pid, d)
            pbar_problems.update(1)
            return result

        completed = await asyncio.gather(
            *[tracked_problem(pid, d) for pid, d in test_inputs.items()]
        )
        pbar_problems.close()

        if use_batch_api:
            from novelty_eval.judge import submit_batch_job
            all_requests = [req for _, data in completed if data for req in data]
            submit_batch_job(all_requests, run_output_path)
            continue

        run_results: Dict[str, Any] = {
            pid: data for pid, data in completed if data is not None
        }

        scores_path = os.path.join(run_output_path, "scores.json")
        with open(scores_path, 'w') as f:
            json.dump(run_results, f, indent=2)

        LOGGER.debug(f"Run {i + 1} metrics:")
        postfix = _accumulate_run_metrics(stats, run_results, test_inputs, 'pairwise')
        pbar_runs.set_postfix(postfix)

    _finalize_stats(stats)
    return stats


async def run_pointwise_experiment(
    n_runs: int,
    llm_engine: str,
    output_dir: str,
    inputs: Dict[str, Any],
    max_workers: int,
    timestamp: str,
    retrieval_cache: Dict[str, Any],
    retrieval_cache_path: Optional[str],
    effort: str,
    use_batch_api: bool = False,
    failure_log: Optional[JudgeFailureLog] = None,
    top_k_candidates: Optional[int] = None,
    mec_k: int = 1,
    use_prompt_caching: bool = False,
    tools: Optional[list] = None,
    cutoff_date: Optional[str] = None,
    judge_backend: str = "standard",
    baseline_adapter: Optional[Any] = None,
    **_ignored,  # absorbs unused shared_kwargs (bidirectional, etc.)
) -> ExperimentStats:
    """Pointwise test mode: classify each single idea as novel (1) or not (0).

    Each instance must have 'idea' (text) and 'label' ('POSITIVE'|'NEGATIVE').
    When retrieval_cache_path is set, related work is injected into the prompt.
    Computes per-class and aggregated accuracy, precision, recall, and F1.
    """
    from novelty_eval.judge import EVALUATION_CRITERIA as _EC, EVALUATION_CRITERIA_RETRIEVAL as _ECR

    retrieve = retrieval_cache_path is not None
    evaluation_criteria = _ECR if retrieve else _EC

    test_inputs = {k: v for k, v in inputs.items() if 'label' in v}
    LOGGER.info(
        f"Running Pointwise Experiment: n={n_runs}, instances={len(test_inputs)}, "
        f"max_workers={max_workers}, retrieve_from_cache={retrieve}, use_batch_api={use_batch_api}"
    )

    stats = ExperimentStats()

    pbar_runs = tqdm(range(n_runs), desc="Runs (Pointwise)")
    for i in pbar_runs:
        LOGGER.debug(f"Run {i + 1}/{n_runs}...")
        run_output_path = os.path.join(output_dir, "accuracy_test_artifacts", timestamp, f"run_pointwise_{i}")
        stats.run_paths.append(run_output_path)
        os.makedirs(run_output_path, exist_ok=True)

        semaphore = asyncio.Semaphore(max_workers)

        async def evaluate_instance(instance_id, data):
            idea_text = data.get('idea', '')

            # External baseline stands in for the judge. It does its own retrieval,
            # so the retrieval cache below does not apply. Returns 0/1, or None when
            # it could not decide — omitted from scores.json so the ablation layer
            # injects it as a wrong prediction.
            if baseline_adapter is not None:
                pred = await baseline_adapter.pointwise(
                    instance_id, data,
                    llm_engine=llm_engine,
                    semaphore=semaphore,
                    effort=effort,
                    failure_log=failure_log,
                    run_label=f"pointwise_{i}",
                    output_path=run_output_path,
                )
                return instance_id, pred

            # Pseudo-judge: read the verdict produced at retrieval time (judge-with-papers
            # mode) straight from the cache. No LLM call — the verdict was already made.
            if judge_backend == "cached_self_judgement":
                cached = retrieval_cache.get(str(instance_id), {})
                sj = cached.get('self_judgement') if isinstance(cached, dict) else None
                novelty = sj.get('novelty') if isinstance(sj, dict) else None
                if novelty is None:
                    LOGGER.warning(f"No cached self_judgement for instance {instance_id}.")
                    return instance_id, None
                return instance_id, int(novelty)

            related_work = ""
            if retrieve:
                cached = retrieval_cache.get(str(instance_id), {})
                candidates = cached.get('candidates')
                if candidates is not None and top_k_candidates is not None:
                    related_work = select_and_format_candidates(candidates, top_k=top_k_candidates)
                else:
                    # Pointwise cache entry: either a plain string or a dict with a 'text' key
                    related_work = cached.get('related_work') or cached.get('text') or ""
                if not related_work:
                    LOGGER.warning(f"No retrieval cache entry for instance {instance_id}.")

            pred = await evaluate_idea_pointwise(
                instance_id, idea_text, llm_engine, semaphore, evaluation_criteria,
                effort, related_work=related_work,
                failure_log=failure_log, run_label=f"pointwise_{i}",
                mec_k=mec_k,
                output_path=run_output_path,
                use_batch_api=use_batch_api,
                use_prompt_caching=use_prompt_caching,
                tools=tools,
                cutoff_date=cutoff_date,
            )
            return instance_id, pred

        pbar_problems = tqdm(total=len(test_inputs), desc="  Instances", leave=False)

        async def tracked_instance(iid, d):
            result = await evaluate_instance(iid, d)
            pbar_problems.update(1)
            return result

        completed = await asyncio.gather(
            *[tracked_instance(iid, d) for iid, d in test_inputs.items()]
        )
        pbar_problems.close()

        if use_batch_api:
            from novelty_eval.judge import submit_batch_job
            all_requests = [req for _, data in completed if data for req in data]
            submit_batch_job(all_requests, run_output_path)
            continue

        n_failed = sum(1 for _, pred in completed if pred is None)
        if n_failed:
            LOGGER.warning(
                f"{n_failed}/{len(completed)} pointwise instances excluded from metrics "
                f"(pred=None — LLM failure). Check llm_failures_debug.md for details."
            )
        run_results: Dict[str, Any] = {
            str(iid): {
                'prediction': pred,
                'label': test_inputs[iid]['label'],
            }
            for iid, pred in completed
            if pred is not None
        }

        scores_path = os.path.join(run_output_path, "scores.json")
        with open(scores_path, 'w') as f:
            json.dump(run_results, f, indent=2)

        LOGGER.debug(f"Run {i + 1} metrics:")
        postfix = _accumulate_run_metrics(stats, run_results, test_inputs, 'pointwise')
        pbar_runs.set_postfix(postfix)

    _finalize_stats(stats)
    return stats


async def run_ranking_experiment(
    n_runs: int,
    llm_engine: str,
    output_dir: str,
    inputs: Dict[str, Any],
    swiss_tournament: bool,
    max_workers: int,
    timestamp: str,
    retrieval_cache: Dict[str, Any],
    retrieval_cache_path: Optional[str],
    use_batch_api: bool,
    effort: str,
    mec_k: int = 1,
    bidirectional: bool = True,
    failure_log: Optional[JudgeFailureLog] = None,
    top_k_candidates: Optional[int] = None,
    use_prompt_caching: bool = False,
    tools: Optional[list] = None,
    cutoff_date: Optional[str] = None,
    **_ignored,
) -> ExperimentStats:
    """Ranking test mode: tournament over multiple-idea instances.

    Retrieval is enabled automatically when retrieval_cache_path is provided.
    """
    test_inputs = {k: v for k, v in inputs.items() if 'expected_winners' in v}
    retrieve_related_work = retrieval_cache_path is not None

    label = 'swiss' if swiss_tournament else 'rr'

    LOGGER.info(
        f"Running experiment mode={label}, n={n_runs}, instances={len(test_inputs)}, "
        f"max_workers={max_workers}, retrieve_from_cache={retrieve_related_work}"
    )

    stats = ExperimentStats()

    pbar_runs = tqdm(range(n_runs), desc=f"Runs ({label})")
    for i in pbar_runs:
        LOGGER.debug(f"Run {i + 1}/{n_runs}...")
        run_output_path = os.path.join(output_dir, "accuracy_test_artifacts", timestamp, f"run_{label}_{i}")
        stats.run_paths.append(run_output_path)
        os.makedirs(run_output_path, exist_ok=True)

        results = await run_comparative_evaluation(
            test_inputs,
            llm_engine,
            max_workers,
            run_output_path,
            swiss_tournament,
            retrieve_related_work=retrieve_related_work,
            retrieval_cache=retrieval_cache,
            retrieval_cache_path=retrieval_cache_path,
            use_batch_api=use_batch_api,
            effort=effort,
            mec_k=mec_k,
            bidirectional=bidirectional,
            failure_log=failure_log,
            run_label=f"{label}_{i}",
            top_k_candidates=top_k_candidates,
            use_prompt_caching=use_prompt_caching,
            tools=tools,
            cutoff_date=cutoff_date,
        )

        LOGGER.debug(f"Run {i + 1} metrics:")
        postfix = _accumulate_run_metrics(stats, results, test_inputs, 'ranking')
        pbar_runs.set_postfix(postfix)

    _finalize_stats(stats)
    return stats


async def run_random_experiment(n_runs: int, inputs: Dict[str, Any]) -> ExperimentStats:
    """Ranking baseline: assign random scores and compute ranking metrics."""
    test_inputs = {k: v for k, v in inputs.items() if 'expected_winners' in v}

    LOGGER.info(f"Running Random Ranking Experiment, n={n_runs}, instances={len(test_inputs)}")
    stats = ExperimentStats()

    pbar_runs = tqdm(range(n_runs), desc="Runs (Random)")
    for i in pbar_runs:
        results = {
            problem_id: {
                'overall': [random.random() for _ in input_data['ideas']],
                'ideas': list(input_data['ideas'].keys()),
            }
            for problem_id, input_data in test_inputs.items()
        }
        LOGGER.debug(f"Run {i + 1} metrics:")
        postfix = _accumulate_run_metrics(stats, results, test_inputs, 'ranking')
        pbar_runs.set_postfix(postfix)

    _finalize_stats(stats)
    return stats


async def reprocess_experiment_results(
    run_prefix: str,
    artifacts_dir: str,
    inputs: Dict[str, Any],
    test_mode: str,
) -> ExperimentStats:
    """Re-read saved run artifacts and recompute metrics without re-running the LLM."""
    if test_mode == 'pointwise':
        test_inputs = {k: v for k, v in inputs.items() if 'label' in v}
    else:
        test_inputs = {k: v for k, v in inputs.items() if 'expected_winners' in v}

    run_dirs = sorted(
        [
            os.path.join(artifacts_dir, d)
            for d in os.listdir(artifacts_dir)
            if d.startswith(f"run_{run_prefix}_")
        ],
        key=lambda p: int(p.split('_')[-1]) if p.split('_')[-1].isdigit() else -1,
    ) if os.path.exists(artifacts_dir) else []

    if not run_dirs:
        return ExperimentStats.empty()

    stats = ExperimentStats()
    for run_path in tqdm(run_dirs, desc="Reprocessing runs"):
        LOGGER.debug(f"Processing {run_path}...")
        results = _load_run_results(run_path)
        if not results:
            LOGGER.warning(f"No results found for {run_path}")
            continue
        # For batch runs, the LLM reasoning lives in the JSONL but was never written
        # to comparisons/*.txt (that only happens in the live non-batch path).
        # Write them now so the markdown debug report can display them.
        jsonl_files = [
            os.path.join(run_path, fn)
            for fn in os.listdir(run_path)
            if fn.startswith("batch_output_") and fn.endswith(".jsonl")
        ] if os.path.isdir(run_path) else []
        for jsonl_path in jsonl_files:
            n_cmp = write_comparison_logs_from_batch_results(jsonl_path, run_path)
            if n_cmp:
                LOGGER.debug(f"Wrote {n_cmp} comparison log entries from {os.path.basename(jsonl_path)}")
            n_ptw = write_pointwise_logs_from_batch_results(jsonl_path, run_path)
            if n_ptw:
                LOGGER.debug(f"Wrote {n_ptw} pointwise log entries from {os.path.basename(jsonl_path)}")
        # Batch-retrieved results don't go through _annotate_gt_winner at write time,
        # so comparisons may be missing gt_winner. Annotate now using test_inputs.
        if test_mode == 'pairwise':
            for problem_id, result_data in results.items():
                comps = result_data.get('comparisons', [])
                if comps and 'gt_winner' not in comps[0]:
                    input_data = test_inputs.get(problem_id)
                    if input_data is None:
                        try:
                            input_data = test_inputs.get(int(problem_id))
                        except (ValueError, TypeError):
                            input_data = test_inputs.get(str(problem_id))
                    if input_data:
                        expected_winners = set(str(w) for w in input_data.get('expected_winners', []))
                        _annotate_gt_winner(comps, expected_winners)
        elif test_mode == 'pointwise':
            for instance_id, result_data in results.items():
                if 'label' not in result_data:
                    input_data = test_inputs.get(instance_id)
                    if input_data is None:
                        try:
                            input_data = test_inputs.get(int(instance_id))
                        except (ValueError, TypeError):
                            input_data = test_inputs.get(str(instance_id))
                    if input_data and 'label' in input_data:
                        result_data['label'] = input_data['label']
                    elif input_data is None:
                        LOGGER.warning(
                            "Pointwise label annotation: instance_id %r not found in "
                            "test_inputs. This can happen when the batch custom_id was "
                            "sanitized/truncated (special chars replaced, key capped at "
                            "40 chars). The instance will be excluded from metrics.",
                            instance_id,
                        )
        stats.run_paths.append(run_path)
        _accumulate_run_metrics(stats, results, test_inputs, test_mode)

    if not stats.accuracies and not stats.pairwise_accuracies and not stats.pointwise_accuracies:
        return ExperimentStats.empty()

    _finalize_stats(stats)
    return stats


def _load_run_results(run_path: str) -> Dict[str, Any]:
    """Load results from scores.json, falling back to a saved batch API response."""
    scores_path = os.path.join(run_path, "scores.json")
    if os.path.exists(scores_path):
        with open(scores_path, 'r') as f:
            return json.load(f)

    batch_files = [
        fn for fn in os.listdir(run_path)
        if fn.startswith("batch_info_") and fn.endswith(".json")
    ]
    if batch_files:
        with open(os.path.join(run_path, batch_files[0]), 'r') as f:
            batch_info = json.load(f)
        batch_id = batch_info.get('batch_id')
        if batch_id:
            LOGGER.info(f"Found batch ID {batch_id}, attempting retrieval...")
            batch_results = retrieve_batch_results(batch_id, run_path)
            if batch_results:
                with open(scores_path, 'w') as f_out:
                    json.dump(batch_results, f_out, indent=2)
                return batch_results

    return {}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS: Dict[str, Any] = {
    "llm_engine": "gpt-5.2",
    "reasoning_effort": "medium",
    "test_mode": "ranking",
    # ranking-mode only
    "modes": ["random", "rr", "swiss"],
    "max_workers": 10,
    # Lower per-model concurrency cap applied automatically for Anthropic models to avoid
    # concurrent-connection rate limits (429). Set explicitly in config to override.
    "anthropic_max_workers": 3,
    "use_batch_api": False,
    "output_dir": ".",
    "num_instances": None,
    "retrieval_cache_file": None,
    "update_report": None,
    "mec_k": 1,
    "bidirectional": True,
    "novelty_criterion_override": None,
    "pairwise_judge_template": None,
    "pointwise_judge_template": None,
    "use_prompt_caching": False,
    # Judge backend: "standard" (default) or "web_search" (GPT Responses API + ArXiv search)
    "judge_backend": "standard",
    # cutoff_date: passed to web_search templates to restrict paper searches (e.g. "2024-01-01")
    "cutoff_date": None,
    # judge_backend "baseline" only: which external baseline to run (see
    # baselines/registry.py for the available names), and the kwargs handed
    # verbatim to its runner.
    "baseline": None,
    "baseline_kwargs": {},
}

_VALID_JUDGE_BACKENDS: List[str] = ["standard", "web_search", "cached_self_judgement", "baseline"]

_VALID_RANKING_MODES: List[str] = ["random", "rr", "swiss"]


def _parse_override(item: str) -> tuple:
    """'num_instances=5' -> ('num_instances', 5). Values are parsed as YAML."""
    key, sep, value = item.partition("=")
    if not sep:
        raise ValueError(f"Invalid --set '{item}'. Expected KEY=VALUE.")
    return key.strip(), yaml.safe_load(value)


def _load_config(config_path: str, overrides: List[str] = ()) -> argparse.Namespace:
    """Load a YAML config file, apply defaults, then apply --set overrides."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    merged = {**_CONFIG_DEFAULTS, **cfg, **dict(_parse_override(o) for o in overrides)}

    # For Anthropic models, use anthropic_max_workers instead of max_workers to avoid
    # concurrent-connection rate limits (429). The sweep base config sets max_workers: 100
    # which is too high for Anthropic. Override anthropic_max_workers in your config to change.
    engine = merged.get("llm_engine", "")
    if engine.startswith("claude-"):
        merged["max_workers"] = merged["anthropic_max_workers"]

    for required in ("test_inputs", "output_file", "n"):
        if required not in merged:
            raise ValueError(f"Required config field '{required}' is missing in {config_path}.")

    if merged.get("test_mode") not in ("ranking", "pairwise", "pointwise"):
        raise ValueError(
            f"Invalid test_mode '{merged.get('test_mode')}' in {config_path}. "
            "Must be 'ranking', 'pairwise', or 'pointwise'."
        )

    if merged["test_mode"] == "ranking":
        bad = [m for m in merged.get("modes", []) if m not in _VALID_RANKING_MODES]
        if bad:
            raise ValueError(f"Invalid ranking modes {bad}. Valid: {_VALID_RANKING_MODES}")

    if merged.get("judge_backend", "standard") not in _VALID_JUDGE_BACKENDS:
        raise ValueError(
            f"Invalid judge_backend '{merged.get('judge_backend')}' in {config_path}. "
            f"Must be one of {_VALID_JUDGE_BACKENDS}."
        )

    return argparse.Namespace(**merged)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main():
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=256))

    parser = argparse.ArgumentParser(description="Test accuracy of comparative evaluator")
    parser.add_argument(
        '--config', type=str,
        default=os.path.join(os.path.dirname(__file__), 'config', 'accuracy_test.yaml'),
        help="Path to the YAML config file (default: config/accuracy_test.yaml)",
    )
    parser.add_argument(
        '--set', action='append', default=[], metavar='KEY=VALUE',
        help="Override a config field, e.g. --set num_instances=5. Repeatable.",
    )
    cli = parser.parse_args()
    args = _load_config(cli.config, cli.set)

    if getattr(args, 'novelty_criterion_override', None):
        from novelty_eval import judge as _eval_main
        _eval_main.EVALUATION_CRITERIA["novelty"] = args.novelty_criterion_override

    if getattr(args, 'pairwise_judge_template', None):
        from novelty_eval import judge as _eval_main
        _eval_main._PAIRWISE_JUDGE_TEMPLATE = args.pairwise_judge_template
    if getattr(args, 'pointwise_judge_template', None):
        from novelty_eval import judge as _eval_main
        _eval_main._POINTWISE_JUDGE_TEMPLATE = args.pointwise_judge_template

    existing_cost_report: Dict[str, Any] = {}
    if GLOBAL_COST_TRACKER is not None:
        GLOBAL_COST_TRACKER.reset()
        if args.update_report:
            existing_cost_path = os.path.join(args.update_report, "cost_report.json")
            if os.path.exists(existing_cost_path):
                try:
                    with open(existing_cost_path) as f:
                        existing_cost_report = json.load(f)
                except Exception as e:
                    LOGGER.warning(f"Could not load existing cost report: {e}")

    inputs = read_inputs(args.test_inputs)
    total_instances_available = len(inputs)
    if args.num_instances is not None:
        inputs = dict(list(inputs.items())[:args.num_instances])
        LOGGER.info(f"Limiting to first {args.num_instances} instances.")

    num_instances = len(inputs)

    if args.update_report:
        artifacts_dir = args.update_report
        timestamp = os.path.basename(artifacts_dir.rstrip('/'))
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        artifacts_dir = os.path.join(args.output_dir, "accuracy_test_artifacts", timestamp)

    os.makedirs(artifacts_dir, exist_ok=True)
    attach_run_log(LOGGER, artifacts_dir)

    # Record what this run actually used, overrides included. Skipped when
    # updating a report, so the original run's config survives.
    if not args.update_report:
        with open(os.path.join(artifacts_dir, "effective_config.yaml"), "w") as f:
            yaml.safe_dump(vars(args), f, sort_keys=False)

    full_report_path = os.path.join(artifacts_dir, args.output_file)
    debug_report_path = os.path.join(artifacts_dir, f"debug_{args.output_file}")
    md_debug_report_path = os.path.join(artifacts_dir, f"debug_{args.output_file}.md")

    LOGGER.info(f"Starting experiments. test_mode={args.test_mode}, artifacts={artifacts_dir}")

    retrieval_cache: Dict[str, Any] = {}
    if args.retrieval_cache_file:
        retrieval_cache = load_retrieval_cache(args.retrieval_cache_file)
        LOGGER.info(f"Loaded retrieval cache from {args.retrieval_cache_file}")

    empty = ExperimentStats.empty()

    failure_log = JudgeFailureLog()

    # Shared kwargs passed to every LLM runner
    shared_kwargs = dict(
        n_runs=args.n,
        llm_engine=args.llm_engine,
        output_dir=args.output_dir,
        inputs=inputs,
        max_workers=args.max_workers,
        timestamp=timestamp,
        retrieval_cache=retrieval_cache,
        retrieval_cache_path=args.retrieval_cache_file,
        use_batch_api=args.use_batch_api,
        effort=args.reasoning_effort,
        mec_k=args.mec_k,
        bidirectional=args.bidirectional,
        failure_log=failure_log,
        top_k_candidates=getattr(args, 'top_k_candidates', None),
        use_prompt_caching=getattr(args, 'use_prompt_caching', False),
        cutoff_date=getattr(args, 'cutoff_date', None),
        judge_backend=getattr(args, 'judge_backend', 'standard'),
    )

    # ------------------------------------------------------------------
    # Judge backend selection
    # ------------------------------------------------------------------
    judge_backend = getattr(args, 'judge_backend', 'standard')
    if judge_backend == 'web_search':
        from novelty_eval import judge as _judge_mod
        shared_kwargs['tools'] = [{"type": "web_search", "filters": {"allowed_domains": ["arxiv.org"]}}]
        if not getattr(args, 'pairwise_judge_template', None):
            _judge_mod._PAIRWISE_JUDGE_TEMPLATE = "pairwise_novelty_web_search.jinja2"
        if not getattr(args, 'pointwise_judge_template', None):
            _judge_mod._POINTWISE_JUDGE_TEMPLATE = "pointwise_novelty_web_search.jinja2"
        LOGGER.info("Judge backend: web_search (OpenAI Responses API + ArXiv web search tool)")
        if getattr(args, 'cutoff_date', None):
            from novelty_eval import search_query_monitor as _sqm
            _sqm.configure(cutoff_date=args.cutoff_date, logger=LOGGER)
    elif judge_backend == 'baseline':
        # External baseline (AI-Scientist, Scideator) stands in for the judge. It
        # returns a prediction and nothing else; the run loop, scores.json, metrics
        # and reports below are unchanged.
        from novelty_eval.baselines.adapter import BaselineAdapter
        if args.test_mode != 'pointwise':
            raise ValueError("judge_backend 'baseline' currently supports pointwise test_mode only.")
        shared_kwargs['baseline_adapter'] = BaselineAdapter.from_args(args)
    elif judge_backend == 'cached_self_judgement':
        # No LLM call: the verdict is read from the retrieval cache's `self_judgement`
        # field (produced by retrieval/web_search_novelty_judge.py).
        if args.test_mode != 'pointwise':
            raise ValueError("judge_backend 'cached_self_judgement' is only supported in pointwise test_mode.")
        if not args.retrieval_cache_file:
            raise ValueError("judge_backend 'cached_self_judgement' requires 'retrieval_cache_file'.")
        LOGGER.info("Judge backend: cached_self_judgement (reads self_judgement from retrieval cache, no LLM call)")
    else:
        LOGGER.info("Judge backend: standard")

    # ------------------------------------------------------------------
    # Pointwise mode — binary classification of a single idea
    # ------------------------------------------------------------------
    if args.test_mode == 'pointwise':
        if args.update_report:
            pointwise_stats = await reprocess_experiment_results('pointwise', artifacts_dir, inputs, 'pointwise')
        else:
            pointwise_stats = await run_pointwise_experiment(**shared_kwargs)

        save_report(
            full_report_path, debug_report_path,
            empty, empty, empty,
            args.n, args.test_inputs, num_instances,
            modes=['pointwise'], test_mode='pointwise',
            pointwise_stats=pointwise_stats,
        )
        save_md_debug_report(
            md_debug_report_path,
            empty, empty, inputs, ['pointwise'], retrieval_cache, args,
            pointwise_stats=pointwise_stats,
        )

    # ------------------------------------------------------------------
    # Pairwise mode — one direct comparison per pair, no tournaments
    # ------------------------------------------------------------------
    elif args.test_mode == 'pairwise':
        if args.update_report:
            pairwise_stats = await reprocess_experiment_results('pairwise', artifacts_dir, inputs, 'pairwise')
        else:
            pairwise_stats = await run_pairwise_experiment(**shared_kwargs)

        save_report(
            full_report_path, debug_report_path,
            empty, empty, empty,
            args.n, args.test_inputs, num_instances,
            modes=['pairwise'], test_mode='pairwise',
            pairwise_stats=pairwise_stats,
        )
        save_md_debug_report(
            md_debug_report_path,
            empty, empty, inputs, ['pairwise'], retrieval_cache, args,
            pairwise_stats=pairwise_stats,
        )

    # ------------------------------------------------------------------
    # Ranking mode — tournament structures over multi-idea instances
    # ------------------------------------------------------------------
    else:
        rr_stats = swiss_stats = random_stats = empty

        if args.update_report:
            if 'rr' in args.modes:
                rr_stats = await reprocess_experiment_results('rr', artifacts_dir, inputs, 'ranking')
            if 'swiss' in args.modes:
                swiss_stats = await reprocess_experiment_results('swiss', artifacts_dir, inputs, 'ranking')
            if 'random' in args.modes:
                random_stats = await run_random_experiment(args.n, inputs)
        else:
            if 'random' in args.modes:
                random_stats = await run_random_experiment(args.n, inputs)
            if 'rr' in args.modes:
                rr_stats = await run_ranking_experiment(swiss_tournament=False, **shared_kwargs)
            if 'swiss' in args.modes:
                swiss_stats = await run_ranking_experiment(swiss_tournament=True, **shared_kwargs)

        save_report(
            full_report_path, debug_report_path,
            rr_stats, swiss_stats, random_stats,
            args.n, args.test_inputs, num_instances, args.modes, 'ranking',
        )
        save_md_debug_report(
            md_debug_report_path,
            rr_stats, swiss_stats, inputs, args.modes, retrieval_cache, args,
        )

    llm_failures_path = os.path.join(artifacts_dir, "llm_failures_debug.md")
    failure_log.write_md(llm_failures_path)

    if GLOBAL_COST_TRACKER is not None:
        cost_report = GLOBAL_COST_TRACKER.get_report()
        # In update_report mode, if no new costs were recorded (scores.json was already
        # present so no JSONL re-download happened), fall back to the saved report so
        # the cost isn't reported as zero. If new costs were recorded, use those —
        # merging the old report would double-count every re-poll.
        if args.update_report and existing_cost_report and cost_report.get("total_calls", 0) == 0:
            cost_report = existing_cost_report
        cost_report["test_mode"] = args.test_mode
        cost_report["instances_processed"] = num_instances
        cost_report["instances_total"] = total_instances_available
        if num_instances > 0:
            cost_report["estimated_full_cost_usd"] = round(
                cost_report["total_cost_usd"] * total_instances_available / num_instances, 6
            )
        else:
            cost_report["estimated_full_cost_usd"] = cost_report["total_cost_usd"]

        # JSON — consumed by run_ablations.py for cross-ablation aggregation
        cost_json_path = os.path.join(artifacts_dir, "cost_report.json")
        with open(cost_json_path, "w") as _f:
            json.dump(cost_report, _f, indent=2)

        # Markdown — human-readable report
        cost_md_path = os.path.join(artifacts_dir, "cost_report.md")
        _write_cost_report_md(cost_report, cost_md_path, title=f"Cost Report — `{args.llm_engine}`")

        LOGGER.info(
            f"Estimated cost: ${cost_report['total_cost_usd']:.4f} "
            f"({cost_report['total_calls']} LLM calls, "
            f"{cost_report['total_input_tokens']:,} input + "
            f"{cost_report['total_output_tokens']:,} output tokens) — "
            f"saved to {cost_md_path}"
        )

    LOGGER.info(f"Report saved to {full_report_path}")
    LOGGER.info(f"Debug report saved to {debug_report_path}")
    LOGGER.info(f"MD debug report saved to {md_debug_report_path}")


if __name__ == '__main__':
    asyncio.run(main())

