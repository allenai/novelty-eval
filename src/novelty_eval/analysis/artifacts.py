"""
Artifact directory discovery and data extraction helpers.

Handles reading ablation_summary.json / sweep_summary.json, parsing
accuracy_report.txt for instances paths and model/mode labels, and
extracting per-problem raw outcomes from scores.json files.
"""
import json
import re
import sys
import yaml
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from novelty_eval.ablation.config import (
    _get_test_inputs_path,
    _load_test_inputs,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SAFETY_ERROR_CODES = {"invalid_prompt", "refusal", "content_filter", "content_policy_violation"}
_RAW_OUTCOMES_CACHE: dict[tuple[Path, str], dict | None] = {}


def _is_pairwise_failure_tie(comp: dict) -> bool:
    """True if winner=2 because all MEC calls failed (empty fwd_votes for every dimension).

    A genuine tie has non-empty fwd_votes.  If mec_details is absent (old runs),
    returns False (can't distinguish → safe default, keeps existing tie behaviour).
    """
    if comp.get("winner") != 2:
        return False
    mec = comp.get("mec_details")
    if not mec or not isinstance(mec, dict):
        return False
    return all(
        isinstance(v, dict) and len(v.get("fwd_votes", [1])) == 0
        for v in mec.values()
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _relative_path_label(full_path: str) -> str:
    """Return a short display label for a path, relative to project root when possible."""
    try:
        return str(Path(full_path).relative_to(_PROJECT_ROOT))
    except ValueError:
        pass
    parts = Path(full_path).parts
    return "/".join(parts[-3:]) if len(parts) >= 3 else full_path


def _extract_instance_id_from_key(k: str) -> str | None:
    """
    Extract the bare problem-id from a scores.json key.

    Supported formats:
      - "cmp-{id}-{a}-{b}-{c}" (comparison): strips prefix and 3 trailing digit indices
      - "ptw-{id}-{n}" (pointwise): strips prefix and 1 trailing digit index
      - plain integer string (e.g. "42", "-5"): returned as-is
    Returns None for keys that don't match any recognized format.
    """
    s = str(k)

    if s.startswith("cmp-"):
        parts = s[4:].split("-")
        if len(parts) < 4 or not all(p.isdigit() for p in parts[-3:]):
            return None
        return "-".join(parts[:-3])

    if s.startswith("ptw-"):
        parts = s[4:].split("-")
        if len(parts) < 2 or not parts[-1].isdigit():
            return None
        return "-".join(parts[:-1])

    try:
        int(s)
        return s
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Artifact directory discovery
# ---------------------------------------------------------------------------

def _collect_artifact_dirs(run_dir: Path) -> list[str]:
    """
    Collect all successful artifact directories from an ablation or sweep
    output directory.  Checks ablation_summary.json first, then falls back
    to scanning for nested sweep_summary.json files.
    """
    # Ablation-level summary (produced by run_ablations.py)
    ablation_summary = run_dir / "ablation_summary.json"
    if ablation_summary.exists():
        with open(ablation_summary) as f:
            data = json.load(f)
        dirs = [
            d
            for row in data.get("runs", [])
            for d in row.get("artifact_dirs", [])
            if d
        ]
        if dirs:
            return dirs

    # Sweep-level summary (produced by run_accuracy_sweep.py)
    sweep_summary = run_dir / "sweep_summary.json"
    if sweep_summary.exists():
        with open(sweep_summary) as f:
            data = json.load(f)
        return [
            r["artifact_dir"]
            for r in data.get("runs", [])
            if r.get("success") and r.get("artifact_dir")
        ]

    # Fallback: walk all nested sweep_summary.json files
    dirs = []
    for path in sorted(run_dir.rglob("sweep_summary.json")):
        with open(path) as f:
            data = json.load(f)
        dirs += [
            r["artifact_dir"]
            for r in data.get("runs", [])
            if r.get("success") and r.get("artifact_dir")
        ]

    # Final fallback: if the directory itself contains an accuracy_report.txt, treat it as an artifact dir
    if not dirs and (run_dir / "accuracy_report.txt").exists():
        return [str(run_dir)]

    return dirs


def _read_ablation_rows(run_dir: Path, artifact_dirs: list[str]) -> list[dict]:
    """
    Try to read per-ablation metadata (name, description, status) from an
    ablation_summary.json.  Falls back to sweep_summary.json or the collected dirs.
    """
    summary_path = run_dir / "ablation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        rows = data.get("runs", [])
        if rows:
            return rows

    sweep_summary = run_dir / "sweep_summary.json"
    if sweep_summary.exists():
        with open(sweep_summary) as f:
            data = json.load(f)
        return [
            {
                "ablation": r.get("ablation", run_dir.name),
                "description": r.get("description", ""),
                "artifact_dirs": [r["artifact_dir"]] if r.get("artifact_dir") else [],
            }
            for r in data.get("runs", [])
            if r.get("success") and r.get("artifact_dir")
        ]

    # Synthesise a single row from the collected dirs
    if artifact_dirs:
        return [{"ablation": run_dir.name, "description": run_dir.name, "artifact_dirs": artifact_dirs}]
    return []


# ---------------------------------------------------------------------------
# Artifact directory inspection
# ---------------------------------------------------------------------------

def _extract_refusal_indices(artifact_dir: Path) -> list[int]:
    """Return sorted unique abstract indices refused for safety reasons in this artifact dir."""
    debug_path = artifact_dir / "llm_failures_debug.md"
    if not debug_path.exists():
        return []
    try:
        content = debug_path.read_text(encoding="utf-8")
    except Exception:
        return []
    section_re = re.compile(
        r"###\s+\d+\s+·\s+Problem\s+`([^`]+)`.*?(?=###\s+\d+\s+·|\Z)",
        re.DOTALL,
    )
    refused: set[int] = set()
    for m in section_re.finditer(content):
        problem_name = m.group(1)
        code_match = re.search(r"\|\s*Error code\s*\|\s*`([^`]+)`", m.group(0))
        if code_match and code_match.group(1) in _SAFETY_ERROR_CODES:
            try:
                refused.add(int(problem_name))
            except ValueError:
                pass
    return sorted(refused)


def _extract_instances_path(artifact_dir: Path) -> str:
    """Return the test instances path from accuracy_report.txt, or empty string."""
    report_path = artifact_dir / "accuracy_report.txt"
    if not report_path.exists():
        return ""
    try:
        content = report_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Test Instances Path:\s*(.+)", content)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""


def _instances_content_matches(
    path_a: str,
    path_b: str,
    setup: str,
) -> tuple[bool, list[str]]:
    """
    For two test-instances files that have different paths, check whether their
    content is identical for all shared pids.

    Compares idea text + label (pointwise) or ideas dict + expected_winners (pairwise).
    Returns (all_match, list_of_mismatched_pids).
    """
    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else _PROJECT_ROOT / pp

    try:
        data_a = _load_test_inputs(_resolve(path_a))
        data_b = _load_test_inputs(_resolve(path_b))
    except Exception:
        return True, []  # can't load — don't emit a false warning

    norm_a = {str(k): v for k, v in data_a.items()}
    norm_b = {str(k): v for k, v in data_b.items()}
    shared = set(norm_a) & set(norm_b)

    mismatched: list[str] = []
    for pid in sorted(shared):
        ea, eb = norm_a[pid], norm_b[pid]
        if not isinstance(ea, dict) or not isinstance(eb, dict):
            continue
        if setup == "pointwise":
            if str(ea.get("idea", "")).strip() != str(eb.get("idea", "")).strip():
                mismatched.append(pid)
            elif ea.get("label") != eb.get("label"):
                mismatched.append(pid)
        elif setup == "pairwise":
            ideas_a = {str(k): str(v).strip() for k, v in (ea.get("ideas") or {}).items()}
            ideas_b = {str(k): str(v).strip() for k, v in (eb.get("ideas") or {}).items()}
            ew_a = {str(w) for w in (ea.get("expected_winners") or [])}
            ew_b = {str(w) for w in (eb.get("expected_winners") or [])}
            if ideas_a != ideas_b or ew_a != ew_b:
                mismatched.append(pid)

    return len(mismatched) == 0, mismatched


def _extract_model_from_artifact_dir(artifact_dir: Path) -> str:
    """
    Extract the model name from an artifact directory.
    Looks for a debug_*.md file and parses the JSON arguments.
    """
    if not artifact_dir.exists():
        return "Unknown"
    for f in artifact_dir.glob("debug_*.md"):
        try:
            with open(f) as _f:
                content = _f.read()
                match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if match:
                    args_dict = json.loads(match.group(1))
                    return args_dict.get('llm_engine', 'Unknown')
        except Exception:
            pass
    return "Unknown"


def _extract_mode_from_artifact_dir(artifact_dir: Path) -> str:
    """
    Extract the test mode (pointwise, pairwise, ranking) from accuracy_report.txt.
    """
    report_path = artifact_dir / "accuracy_report.txt"
    if report_path.exists():
        try:
            with open(report_path) as f:
                content = f.read(1000)
                match = re.search(r'Test Mode:\s*(\w+)', content)
                if match:
                    mode = match.group(1).lower()
                    if mode in ("pointwise", "pairwise"):
                        return mode
        except Exception:
            pass
    return "ranking"


def _extract_ablation_from_artifact_dir(artifact_dir: Path) -> str:
    """
    Extract the ablation/config name an artifact dir belongs to.

    Sweep artifacts are laid out as
    ``<ablation>/<ablation>-<model>/accuracy_test_artifacts/<timestamp>``, so the
    ablation name sits two levels above the timestamp.  Needed whenever several
    configs of the *same* model are compared side by side (e.g. a k=3 sweep and
    the k=1 sub-runs derived from it), since the model name alone no longer
    identifies a row.  Falls back to the nearest named ancestor for dirs that do
    not follow the layout (e.g. ``web_search_self_judge/accuracy_test_artifacts/<ts>``).
    """
    if artifact_dir.parent.name != "accuracy_test_artifacts":
        return artifact_dir.name
    model_dir = artifact_dir.parent.parent
    ablation_dir = model_dir.parent
    if ablation_dir.name and model_dir.name.startswith(f"{ablation_dir.name}-"):
        return ablation_dir.name
    return model_dir.name


def _extract_exclude_set_from_filtered_report(
    artifact_dir: Path,
    report_filename: str = "filtered_accuracy_report.txt",
) -> set[str]:
    """Parse the excluded abstract indices written by _write_filtered_accuracy_report."""
    report_path = artifact_dir / report_filename
    if not report_path.exists():
        return set()
    try:
        content = report_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"# Filtered: excluded abstract indices (\[.*?\])", content)
        if m:
            import ast
            return {str(e) for e in ast.literal_eval(m.group(1))}
    except Exception:
        pass
    return set()


# ---------------------------------------------------------------------------
# Raw outcomes extraction
# ---------------------------------------------------------------------------

def _extract_raw_outcomes_for_dir(
    artifact_dir: Path,
    setup: str,
    ablation_name: str,
    logger_inst=None,
) -> dict[str, dict[str, float]] | None:
    """
    Extract raw problem-level outcomes for multiple metrics from scores.json.
    Averages across multiple runs if present.
    Returns: {metric_key -> {problem_id -> outcome}}
    """
    cache_key = (artifact_dir, setup)
    if cache_key in _RAW_OUTCOMES_CACHE:
        return _RAW_OUTCOMES_CACHE[cache_key]

    # Find scores.json files. 
    # We look for run_*/scores.json first. If found, we use those (multiple runs).
    # Otherwise, we look for a single scores.json in the root. 
    # This avoids rglob picking up both a merged root file and individual run files.
    scores_files = sorted(list(artifact_dir.glob("run_*/scores.json")))
    if not scores_files:
        scores_files = sorted(list(artifact_dir.glob("scores.json")))
    
    if not scores_files:
        _RAW_OUTCOMES_CACHE[cache_key] = None
        return None

    if len(scores_files) > 1 and logger_inst:
        logger_inst.warning(
            f"{artifact_dir.name}: {len(scores_files)} run-specific scores.json files found — "
            f"per-problem scores will be averaged across runs before bootstrap."
        )

    # We need ground truth for pairwise; labels for pointwise (often absent from batch scores.json)
    inputs = {}
    label_lookup: dict[str, str] = {}
    if setup in ("pairwise", "pointwise"):
        instances_path_str = _extract_instances_path(artifact_dir) if setup == "pointwise" else None
        full_inputs_path = None

        if setup == "pairwise":
            inputs_path_str = _get_test_inputs_path(ablation_name, setup)
            if inputs_path_str:
                full_inputs_path = _PROJECT_ROOT / inputs_path_str

        if setup == "pointwise" and instances_path_str:
            p = Path(instances_path_str)
            full_inputs_path = p if p.is_absolute() else _PROJECT_ROOT / p

        if not full_inputs_path or not full_inputs_path.exists():
            candidates = ["test_inputs.yaml", "benchmark_instances.yaml", "test_instances.yaml"]
            curr = artifact_dir
            found = False
            for _ in range(5):
                for c in candidates:
                    if (curr / c).exists():
                        full_inputs_path = curr / c
                        found = True
                        break
                if found: break
                curr = curr.parent

        if full_inputs_path and full_inputs_path.exists():
            inputs = _load_test_inputs(full_inputs_path)
            if setup == "pointwise":
                for k, v in inputs.items():
                    if isinstance(v, dict) and "label" in v:
                        label_lookup[str(k)] = v["label"]
        elif setup == "pairwise":
            _RAW_OUTCOMES_CACHE[cache_key] = None
            return None

    # Build gt_winner lookup from inputs for pairwise (used to backfill missing gt_winner).
    gt_lkp: dict[str, list] = {}
    if setup == "pairwise":
        for k, v in inputs.items():
            if isinstance(v, dict) and "expected_winners" in v:
                gt_lkp[str(k)] = [str(w) for w in v["expected_winners"]]

    all_run_outcomes = [] # list of {metric -> {pid -> score}}
    for sf in scores_files:
        try:
            with open(sf) as f:
                data = json.load(f)

            run_outcomes_by_metric = defaultdict(dict)

            if setup == "pointwise":
                for k, res in data.items():
                    pid = _extract_instance_id_from_key(k)
                    if pid is None: pid = str(k)

                    pred = res.get("prediction")
                    label = res.get("label")
                    # Back-fill label from instances YAML when absent (batch API runs)
                    if label is None and pid in label_lookup:
                        label = label_lookup[pid]
                    if pred is not None and label:
                        target = 1 if label == "POSITIVE" else 0
                        run_outcomes_by_metric["_pointwise_pred"][pid] = float(pred)
                        run_outcomes_by_metric["_pointwise_target"][pid] = float(target)

            elif setup == "pairwise":
                for k, res in data.items():
                    pid = _extract_instance_id_from_key(k)
                    if pid is None: pid = str(k)

                    comps = res.get("comparisons", [])
                    if not comps: continue

                    # Backfill gt_winner from instances YAML when absent (old-format scores.json).
                    if comps[0].get("gt_winner") is None and pid in gt_lkp:
                        from novelty_eval.run_benchmark import _annotate_gt_winner
                        _annotate_gt_winner(comps, set(gt_lkp[pid]))

                    gt = comps[0].get('gt_winner')
                    if gt is None:
                        continue  # no ground truth available — skip this problem

                    is_tie_raw = comps[0].get('winner') == 2
                    is_tie = is_tie_raw and not _is_pairwise_failure_tie(comps[0])
                    is_correct = (not is_tie) and (comps[0].get('winner') == gt)
                    if is_tie_raw and not is_tie and logger_inst:
                        _inst = _extract_instances_path(artifact_dir) or str(artifact_dir)
                        _model = _extract_model_from_artifact_dir(artifact_dir)
                        logger_inst.warning(
                            "[bootstrap / pairwise] Failure-tie reclassified as wrong "
                            "(all MEC calls failed) — ablation=%s, model=%s, artifact=%s, "
                            "pid=%s, instances=%s",
                            ablation_name, _model, artifact_dir.name, pid, _inst,
                        )

                    run_outcomes_by_metric["_pairwise_is_correct"][pid] = 0.0 if is_tie else float(is_correct)
                    run_outcomes_by_metric["_pairwise_is_tie"][pid] = 1.0 if is_tie else 0.0
                    run_outcomes_by_metric["_pairwise_gt_winner"][pid] = float(gt) if isinstance(gt, (int, float)) else 0.0

            if run_outcomes_by_metric:
                all_run_outcomes.append(run_outcomes_by_metric)
        except Exception:
            continue

    if not all_run_outcomes:
        _RAW_OUTCOMES_CACHE[cache_key] = None
        return None

    # Merge per-problem outcomes across runs and metrics
    metrics = set().union(*(r.keys() for r in all_run_outcomes))
    final_outcomes_by_metric = {}

    for m in metrics:
        all_pids = sorted(set().union(*(r[m].keys() for r in all_run_outcomes if m in r)))
        metric_outcomes = {}
        for pid in all_pids:
            scores = [r[m][pid] for r in all_run_outcomes if m in r and pid in r[m]]
            if scores:
                metric_outcomes[pid] = sum(scores) / len(scores)
        final_outcomes_by_metric[m] = metric_outcomes

    # Inject pids that are in the instances YAML but absent from all runs.
    # These are genuine failures (batch drops, all-call failures not captured above).
    # Treated as wrong predictions so they are included in support.
    if setup == "pairwise" and gt_lkp:
        present = set(final_outcomes_by_metric.get("_pairwise_is_correct", {}))
        _missing_pids: list[str] = []
        for pid, gt_list in gt_lkp.items():
            if pid not in present:
                gt_val = float(int(gt_list[0])) if gt_list else 0.0
                final_outcomes_by_metric.setdefault("_pairwise_is_correct", {})[pid] = 0.0
                final_outcomes_by_metric.setdefault("_pairwise_is_tie",     {})[pid] = 0.0
                final_outcomes_by_metric.setdefault("_pairwise_gt_winner",  {})[pid] = gt_val
                _missing_pids.append(pid)
        if _missing_pids and logger_inst:
            _inst = _extract_instances_path(artifact_dir) or str(artifact_dir)
            _model = _extract_model_from_artifact_dir(artifact_dir)
            # Suppress warnings for pids that are on the paper blocklist — they are
            # intentionally excluded from metrics and being absent is expected.
            _warn_pids = _missing_pids
            if _inst:
                try:
                    from novelty_eval.analysis.filtering import (
                        _load_paper_blocklist as _lbp,
                        _derive_exclude_indices as _dei,
                    )
                    _blocked = _lbp()
                    if _blocked:
                        _excl_by_bl = set(_dei(_inst, _blocked).keys())
                        _warn_pids = [p for p in _missing_pids if p not in _excl_by_bl]
                except Exception:
                    pass
            if _warn_pids:
                logger_inst.warning(
                    "[bootstrap / pairwise] %d pid(s) absent from all runs → injected as wrong — "
                    "ablation=%s, model=%s, artifact=%s, instances=%s, pids=%s",
                    len(_warn_pids), ablation_name, _model, artifact_dir.name, _inst, _warn_pids,
                )

    elif setup == "pointwise" and label_lookup:
        present = set(final_outcomes_by_metric.get("_pointwise_pred", {}))
        _missing_pids_ptw: list[str] = []
        for pid, label in label_lookup.items():
            if pid not in present:
                target = 1.0 if label == "POSITIVE" else 0.0
                final_outcomes_by_metric.setdefault("_pointwise_pred",   {})[pid] = 1.0 - target
                final_outcomes_by_metric.setdefault("_pointwise_target", {})[pid] = target
                _missing_pids_ptw.append(pid)
        if _missing_pids_ptw and logger_inst:
            _inst = _extract_instances_path(artifact_dir) or str(artifact_dir)
            _model = _extract_model_from_artifact_dir(artifact_dir)
            # Suppress warnings for pids that are on the paper blocklist — they are
            # intentionally excluded from metrics and being absent is expected.
            _warn_pids_ptw = _missing_pids_ptw
            if _inst:
                try:
                    from novelty_eval.analysis.filtering import (
                        _load_paper_blocklist as _lbp,
                        _derive_exclude_indices as _dei,
                    )
                    _blocked = _lbp()
                    if _blocked:
                        _excl_by_bl = set(_dei(_inst, _blocked).keys())
                        _warn_pids_ptw = [p for p in _missing_pids_ptw if p not in _excl_by_bl]
                except Exception:
                    pass
            if _warn_pids_ptw:
                logger_inst.warning(
                    "[bootstrap / pointwise] %d pid(s) absent from all runs → injected as wrong — "
                    "ablation=%s, model=%s, artifact=%s, instances=%s, pids=%s",
                    len(_warn_pids_ptw), ablation_name, _model, artifact_dir.name, _inst, _warn_pids_ptw,
                )

    _RAW_OUTCOMES_CACHE[cache_key] = final_outcomes_by_metric
    return final_outcomes_by_metric
