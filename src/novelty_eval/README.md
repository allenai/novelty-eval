# Novelty Evaluation Benchmarking

A map of the codebase: what each subsystem does and where to start reading. 
The pipeline runs in one direction: build the benchmark → (optionally) retrieve related work → judge → analyse → plot.

### `benchmark_data/` — building the benchmark
`fetch_iclr_data.py` pulls ICLR submissions from OpenReview and extracts novelty signals from their reviews; `create_benchmark_instances.py` assembles the result into the pointwise and pairwise instance YAMLs. Walkthrough in [benchmark_data/README.md](benchmark_data/README.md).

`paper_blocklist.yaml` lists papers excluded from both instance creation and metric computation because they trip LLM safety filters. Reported numbers are always post-filter.

### `retrieval/` — related-work context for the judge
`retrieve_candidates.py` writes a disk-backed cache of retrieved papers per idea, via LLM-generated search queries against Paper Finder / Semantic Scholar. Retrieval is a precompute step: it runs once per track, before any judging.

`flatten_retrieval_cache.py` derives the pointwise cache from the pairwise one. `web_search_novelty_judge.py` is the separate, much stronger judge that searches the live web and reports its own citations.

### Judging (root of this package)
`run_benchmark.py` is the entry point for a single judge configuration; `judge.py` makes the calls (async, or via the OpenAI Batch API, whose results `retrieve_batch_results.py` collects). `metrics.py` computes every metric, `tournament.py` schedules which pairs get compared, and `scoring.py` turns raw winner verdicts into scores.

`run_benchmark_sweep.py` runs many configurations and writes a cross-config comparison report (see `config/example_sweep.yaml`). The `*_report_writer.py` and `comparison_logger.py` modules write the human-readable reports and per-comparison debug traces.

### `ablation/` — the controlled evaluation study
`ablations.yaml` is the single source of truth for every condition. `run_ablations.py` reads it and launches instance creation plus a sweep per condition; `merge_ablation_runs.py` combines several runs into one comparison report and writes the chart data the figures consume (configs in `configs/merge/`). `batch_manager.py` handles the asynchronous Batch API path.

`derive_subrun_artifacts.py` carves `mec_k_1` and `unidirectional` out of a `current` sweep — they are not separate runs.

### `retrieval_faceoff/`
Scores a strong web-search judge against the ordinary retrieval-aided sweeps on an identical subset. `run.py` is the driver (`sample` → `submit` → `poll`); `sample_matched_subset.py` draws whole pairs, so the pointwise subset comes out balanced and covers the same items as the pairwise one. `subset_reports.py` scores a subset by recomputing from each run's `scores.json` rather than re-running any judge, and `divergence_report.py` shows the individual ideas the methods disagree on.

### `analysis/`
Reads the artifacts a run writes and turns them into metrics. `artifacts.py` discovers run directories, `filtering.py` applies a blocklist and recomputes, `stats_one_pass.py` is the paired bootstrap behind every significance marker, and `generate_report.py` aggregates conditions into one Markdown table.
