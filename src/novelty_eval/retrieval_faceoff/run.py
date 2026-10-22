#!/usr/bin/env python3
"""run.py — driver for the retrieval face-off, configured from a YAML file.

Builds a matched, balanced, blocklist-clean subset of ideas shared by the pointwise
and pairwise datasets, runs the strong web-search novelty judge over them
(pointwise), and scores it + the existing paper-finder retrieval sweeps on the
identical subset — reusing the ablation pipeline's recompute machinery.

Four user-facing commands (all knobs come from --config; see config.yaml):

  sample    — draw a fresh matched, balanced, blocklist-clean subset only. No
                retrieval, no LLM calls. Inspect manifest.json / pointwise_subset.yaml /
                pairwise_subset.yaml under the printed ROOT before spending on retrieval.

  submit    — kick off web-search retrieval against an already-sampled ROOT.
                use_batch_api: false → runs the rest of the pipeline synchronously in
                  one shot (retrieve → project → self-judge → score). Best for small
                  debug samples. `poll` is then a no-op.
                use_batch_api: true  → submits the OpenAI batch and stops. Come back
                  and run `poll` (as many times as you like) until it's done.

  poll      — (batch mode only) check the submitted batch's status. If it's finished,
                collect it and run the rest of the pipeline (project → self-judge →
                score). If not, report the status and exit — nothing blocks or waits.

  recompute — re-run project → self-judge → score against an already-retrieved ROOT,
                with NO new LLM calls — reuses the existing web_search_master.json.
                Use after changing projection/self-judgement/scoring logic and wanting
                fresh numbers without paying for retrieval again.

  resubmit  — (batch mode only) re-submit just the ideas the last batch dropped
                (provider errors written to the batch error file, or responses whose
                verdict failed to parse) — i.e. every mapped idea still missing from
                web_search_master.json. Submits a fresh batch and stops; run `poll`
                afterwards to collect + score. Recovered verdicts merge into the
                existing master, so the reports come out as if nothing had failed.

  ./scripts/run_retrieval_faceoff.sh sample
  ./scripts/run_retrieval_faceoff.sh submit [ROOT]
  ./scripts/run_retrieval_faceoff.sh poll [ROOT]
  ./scripts/run_retrieval_faceoff.sh recompute [ROOT]
  ./scripts/run_retrieval_faceoff.sh resubmit [ROOT]

ROOT is the timestamped dir printed by `sample`; `submit`/`poll`/`recompute` default
to the latest run. `poll`/`recompute` write subset_comparison.md + subset_scores.json
+ subset_divergence.md (per-idea disagreements vs. the baseline sweeps, with links
into retrieval_debug/) under ROOT, and a subset_accuracy_report.txt into each scored
artifact dir.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
_M = "novelty_eval"

sys.path.insert(0, str(_SRC))
from logging_utils import setup_logger

# Module-level logger — re-pointed at the run's ROOT once known.
LOGGER = setup_logger(str(_REPO_ROOT / "output" / "retrieval_faceoff"), console_level="INFO")

_REQUIRED_KEYS = ("pointwise", "pairwise", "pointwise_sweep", "pairwise_sweep",
                  "n_pairs", "seed", "llm_engine", "cutoff", "top_k_candidates",
                  "use_batch_api", "output_root")


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"config {path} is missing required key(s): {', '.join(missing)}")
    return cfg


def _run(module_args: list[str]) -> None:
    """Invoke `python -m <module> ...` from the repo root with src on PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SRC), env.get("PYTHONPATH", "")]))
    cmd = [sys.executable, "-m", *module_args]
    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env, check=True)


def _selfjudge_artifact_dir(root: Path) -> str | None:
    """Newest run dir under the self-judge output, or None if not run yet."""
    base = root / "web_search_self_judge" / "accuracy_test_artifacts"
    if not base.is_dir():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    return str(runs[-1]) if runs else None


def _latest_root(output_root: str) -> Path | None:
    """Newest run dir (holding a manifest.json) under output_root, or None."""
    base = Path(output_root)
    if not base.is_dir():
        return None
    runs = sorted((d for d in base.iterdir() if d.is_dir() and (d / "manifest.json").exists()),
                  key=lambda d: d.name)
    return runs[-1] if runs else None


def _require_root(cfg: dict, root: str | None) -> Path:
    """Resolve ROOT (explicit arg or latest sampled run) and re-point logging at it."""
    global LOGGER
    if not root:
        p = _latest_root(cfg["output_root"])
        if p is None:
            LOGGER.error("no ROOT given and no sampled run found under %s — run `submit` first.",
                         cfg["output_root"])
            raise SystemExit(1)
        LOGGER.info("No ROOT given — using latest run: %s", p)
    else:
        p = Path(root)
        if not p.is_dir():
            LOGGER.error("ROOT not found: %s", root)
            raise SystemExit(1)
    LOGGER = setup_logger(str(p), console_level="INFO")
    return p


def _latest_batch_info(root: Path) -> dict | None:
    """Most-recently-written batch_info_*.json in ROOT (written by `submit`), or None."""
    infos = list(root.glob("batch_info_*.json"))
    if not infos:
        return None
    with open(max(infos, key=lambda p: p.stat().st_mtime)) as f:
        return json.load(f)


# --- Internal pipeline steps (not user-facing) ---------------------------------

def _do_sample(cfg: dict) -> Path:
    """Run the sampler into a fresh timestamped dir and return it."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{cfg['output_root'].rstrip('/')}/{ts}"
    _run([f"{_M}.retrieval_faceoff.sample_matched_subset",
          "--pointwise", cfg["pointwise"], "--pairwise", cfg["pairwise"],
          "--out-dir", out, "--n-pairs", str(cfg["n_pairs"]), "--seed", str(cfg["seed"]),
          "--llm-engine", cfg["llm_engine"], "--cutoff-date", cfg["cutoff"],
          "--top-k-candidates", str(cfg["top_k_candidates"]),
          "--reasoning-effort", str(cfg.get("reasoning_effort", "medium")),
          "--min-relevance-score", str(cfg.get("min_relevance_score", 0.0)),
          "--allowed-domains", *[str(d) for d in cfg.get("allowed_domains", ["arxiv.org"])]])
    return Path(out)


def _retrieve(p: Path, mode: str) -> None:
    _run([f"{_M}.retrieval.web_search_novelty_judge",
          "--config", str(p / "ws_subset.yaml"), "--mode", mode])


def _project(cfg: dict, p: Path) -> None:
    _run([f"{_M}.retrieval.project_retrieval_cache",
          "--master", str(p / "web_search_master.json"),
          "--test-inputs", str(p / "pointwise_subset.yaml"), "--mode", "pointwise",
          "--output", str(p / "cache_pointwise.json"), "--cutoff-date", cfg["cutoff"]])
    _run([f"{_M}.retrieval.project_retrieval_cache",
          "--master", str(p / "web_search_master.json"),
          "--test-inputs", str(p / "pairwise_subset.yaml"), "--mode", "pairwise",
          "--output", str(p / "cache_pairwise.json"), "--cutoff-date", cfg["cutoff"]])


def _selfjudge(p: Path) -> None:
    _run([f"{_M}.run_benchmark", "--config", str(p / "self_judge_subset.yaml")])


def _baseline_sweeps(cfg: dict) -> list[str]:
    """Paper-finder baseline sweep dirs: the two required ones plus any `extra_sweeps`.
    """
    return [cfg["pointwise_sweep"], cfg["pairwise_sweep"],
            *[str(d) for d in (cfg.get("extra_sweeps") or [])]]


def _score(cfg: dict, p: Path) -> None:
    baselines = _baseline_sweeps(cfg)
    dirs = list(baselines)
    sj = _selfjudge_artifact_dir(p)
    if sj:
        dirs.append(sj)
    else:
        LOGGER.info("No web-search self-judge run found yet — scoring baselines only.")
    _run([f"{_M}.retrieval_faceoff.subset_reports", *dirs, "--manifest", str(p / "manifest.json")])
    if sj:
        _run([f"{_M}.retrieval_faceoff.divergence_report", *baselines,
              "--reference", sj, "--manifest", str(p / "manifest.json"), "--root", str(p)])
    else:
        LOGGER.info("No web-search self-judge run found yet — skipping divergence report.")


def _finish_downstream(cfg: dict, p: Path) -> None:
    """The deterministic tail: project cache → self-judge → score. No external waits."""
    _project(cfg, p)
    _selfjudge(p)
    _score(cfg, p)
    LOGGER.info("Done — reports written under %s", p)


# --- User-facing commands ------------------------------------------------------

def cmd_sample(cfg: dict, root: str | None) -> None:
    """Draw a fresh matched subset only — no retrieval, no LLM calls."""
    global LOGGER
    if root:
        LOGGER.warning("`sample` does not take a ROOT argument — ignoring %r", root)
    p = _do_sample(cfg)
    LOGGER = setup_logger(str(p), console_level="INFO")
    LOGGER.info("Sampled subset written to %s — inspect it, then run `submit %s`.", p, p)
    print(f"ROOT={p}")


def cmd_submit(cfg: dict, root: str | None) -> None:
    """Start retrieval against an already-sampled ROOT. Runs the rest in live mode."""
    global LOGGER
    p = _require_root(cfg, root)
    if cfg["use_batch_api"]:
        _retrieve(p, "submit")
        LOGGER.info("Batch submitted. Run `poll` until it completes — poll will finish the pipeline.")
    else:
        _retrieve(p, "live")
        _finish_downstream(cfg, p)
    print(f"ROOT={p}")


def cmd_recompute(cfg: dict, root: str | None) -> None:
    """Re-run project → self-judge → score against an already-retrieved ROOT. No LLM calls."""
    p = _require_root(cfg, root)
    _finish_downstream(cfg, p)
    print(f"ROOT={p}")


def cmd_resubmit(cfg: dict, root: str | None) -> None:
    """Resubmit the ideas the last batch dropped (provider errors / parse failures).

    Submits a fresh batch containing only the ideas still missing a verdict in the
    master cache, then stops. Run `poll` afterwards to collect the new batch and
    re-score — the recovered verdicts merge into the existing master, so the reports
    come out as if the original batch had never failed.
    """
    p = _require_root(cfg, root)
    if not cfg["use_batch_api"]:
        LOGGER.info("use_batch_api=false — resubmit only applies to batch mode; nothing to do.")
        return
    _retrieve(p, "resubmit")
    LOGGER.info("Resubmitted the missing ideas. Run `poll` until the new batch completes.")
    print(f"ROOT={p}")


def cmd_poll(cfg: dict, root: str | None) -> None:
    """Check the submitted batch; if finished, collect it and score. Never blocks."""
    p = _require_root(cfg, root)
    if not cfg["use_batch_api"]:
        LOGGER.info("use_batch_api=false — the pipeline already ran during `submit`; nothing to poll.")
        return
    info = _latest_batch_info(p)
    if info is None:
        LOGGER.error("No batch_info_*.json under %s — run `submit` first.", p)
        raise SystemExit(1)
    batch_id, provider = info.get("batch_id"), info.get("provider", "openai")

    from batch_api import check_batch_status
    status = check_batch_status(batch_id, provider)
    if status != "completed":
        LOGGER.info("Batch %s status=%s — not ready. Re-run `poll` later.", batch_id, status)
        return

    LOGGER.info("Batch %s completed — collecting and scoring.", batch_id)
    _retrieve(p, "collect")
    _finish_downstream(cfg, p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the driver config YAML.")
    parser.add_argument("stage", choices=["sample", "submit", "poll", "recompute", "resubmit"],
                        help="sample (draw subset), submit (start retrieval), poll (finish "
                             "batch), recompute (re-run project/self-judge/score, no LLM calls), "
                             "or resubmit (retry ideas the last batch dropped, then poll).")
    parser.add_argument("root", nargs="?",
                        help="ROOT dir printed by `sample` (submit/poll/recompute/resubmit default to latest).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _COMMANDS = {"sample": cmd_sample, "submit": cmd_submit, "poll": cmd_poll,
                 "recompute": cmd_recompute, "resubmit": cmd_resubmit}
    _COMMANDS[args.stage](cfg, args.root)


if __name__ == "__main__":
    main()
