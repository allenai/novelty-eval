"""
Ablation configuration helpers: YAML loading, canonical-name resolution,
config diff table, and test-inputs lookup.

All functions here are stateless readers of ablations.yaml.  Import from this
module wherever you need ablation metadata; do not open ablations.yaml elsewhere.
"""
import functools
import json
import yaml
import threading
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ABLATIONS_FILE = _THIS_DIR / "ablations.yaml"
_ABLATIONS_CACHE: tuple[float, dict] | None = None
_TEST_INPUTS_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


def _load_ablations_yaml_data() -> dict:
    """Cached loader for ablations.yaml — single file read shared by all callers, with mtime validation."""
    global _ABLATIONS_CACHE
    if not _ABLATIONS_FILE.exists():
        return {}

    mtime = _ABLATIONS_FILE.stat().st_mtime
    with _CACHE_LOCK:
        if _ABLATIONS_CACHE and _ABLATIONS_CACHE[0] == mtime:
            return _ABLATIONS_CACHE[1]

    try:
        with open(_ABLATIONS_FILE, encoding="utf-8") as f:
            data = yaml.load(f, Loader=SafeLoader) or {}
    except Exception:
        data = {}

    with _CACHE_LOCK:
        _ABLATIONS_CACHE = (mtime, data)
    return data


def _expand_ablation_variants(name: str, entry: dict) -> list[str]:
    """Return all name variants for an ablation: canonical + track-prefixed + legacy + track-prefixed legacy."""
    tracks = entry.get("tracks", [])
    legacy_names = entry.get("legacy_names", [])
    variants = [name]
    for track in tracks:
        variants.append(f"{track}_{name}" if not name.startswith(f"{track}_") else name)
    for legacy in legacy_names:
        variants.append(legacy)
        for track in tracks:
            variants.append(f"{track}_{legacy}")
    return variants


def _load_descriptions_from_yaml() -> dict[str, str]:
    """
    Load ablation descriptions from ablations.yaml — the single source of truth.
    Returns a dict mapping both base names (e.g. 'current') and track-expanded
    names (e.g. 'pairwise_current') to their descriptions.
    """
    descriptions: dict[str, str] = {}
    for name, entry in _load_ablations_yaml_data().get("ablations", {}).items():
        desc = entry.get("description", "")
        for variant in _expand_ablation_variants(name, entry):
            descriptions[variant] = desc
    return descriptions


def _load_before_descriptions_from_yaml() -> dict[str, str]:
    """Load before_description fields from ablations.yaml (same expansion logic as _load_descriptions_from_yaml)."""
    before_descs: dict[str, str] = {}
    for name, entry in _load_ablations_yaml_data().get("ablations", {}).items():
        desc = entry.get("before_description", "")
        if not desc:
            continue
        for variant in _expand_ablation_variants(name, entry):
            before_descs[variant] = desc
    return before_descs


def _load_significance_alternatives_from_yaml() -> dict[str, str]:
    """Return ablation-name → significance_alternative ('two-sided'|'less'|'greater').

    Missing or invalid values fall back to 'two-sided'.
    """
    alternatives: dict[str, str] = {}
    for name, entry in _load_ablations_yaml_data().get("ablations", {}).items():
        alt = entry.get("significance_alternative", "two-sided")
        if alt not in ("two-sided", "less", "greater"):
            alt = "two-sided"
        for variant in _expand_ablation_variants(name, entry):
            alternatives[variant] = alt
    return alternatives


def _load_mismatch_suppressed_ablations() -> set[str]:
    """
    Return all ablation name variants (canonical + legacy + track-prefixed) that
    carry `suppress_instance_mismatch_warning: true` in ablations.yaml.

    Use this to silence the "mismatched content" bootstrap warning for ablations
    where instance differences from the baseline are intentional (e.g. the ablation
    generates its own test instances).
    """
    suppressed: set[str] = set()
    for name, entry in _load_ablations_yaml_data().get("ablations", {}).items():
        if entry.get("suppress_instance_mismatch_warning"):
            suppressed.update(_expand_ablation_variants(name, entry))
    return suppressed


def _build_canonical_key_map() -> dict[str, str]:
    """
    Build a reverse lookup from every known ablation name variant
    (canonical, legacy, track-prefixed) to the canonical YAML key.

    E.g. 'c1' → 'vague_criterion', 'pairwise_c1' → 'vague_criterion'.
    Used so that filenames and titles always use the authoritative YAML key/description.
    """
    reverse: dict[str, str] = {}
    for name, entry in _load_ablations_yaml_data().get("ablations", {}).items():
        for variant in _expand_ablation_variants(name, entry):
            reverse[variant] = name
    return reverse


def _load_legacy_name_aliases() -> dict[str, list[str]]:
    """
    Build a mapping from every known ablation name (canonical + legacy + track-prefixed)
    to the full list of aliases for that ablation.

    Used when looking up a baseline: if 'ranking_current' is not in the retrieval_index
    but 'c4' is, and 'c4' is a legacy name for 'current', the lookup should succeed.

    Returns: {name: [all_equivalent_names_for_same_ablation]}
    """
    aliases: dict[str, list[str]] = {}
    for name, entry in _load_ablations_yaml_data().get("ablations", {}).items():
        all_names = _expand_ablation_variants(name, entry)
        for n in all_names:
            aliases[n] = all_names
    return aliases


def _is_baseline_ablation(
    abl_name: str,
    canonical_key_map: dict,
    baseline_canonical_names: set,
    yaml_aliases: dict,
) -> bool:
    """Return True if abl_name resolves to a baseline ablation."""
    parts = abl_name.split("_", 1)
    stripped = parts[1] if len(parts) == 2 and parts[0] in ("pairwise", "ranking", "pointwise") else abl_name
    canonical = canonical_key_map.get(stripped, canonical_key_map.get(abl_name, stripped))
    if canonical in baseline_canonical_names:
        return True
    return any(canonical in yaml_aliases.get(b, []) for b in baseline_canonical_names)


def _get_test_inputs_path(ablation_name: str, setup: str) -> str | None:
    """Lookup the test_inputs path for an ablation in ablations.yaml, with track fallbacks.

    Honors legacy_names so a renamed ablation (e.g. an old run dir named
    'pairwise-vanilla-ai_ai_researcher_not_comparative') still resolves to its
    current entry, and falls back to the track named in the ablation_name's own
    prefix ('pairwise-vanilla-ai_...') before inferring a track from the setup —
    otherwise an unknown name silently resolves to the wrong track's instances.
    """
    data = _load_ablations_yaml_data()
    if not data:
        return None

    ablations = data.get("ablations", {})
    tracks_cfg = data.get("tracks", {})

    # 1. Find the ablation entry directly, or via track-expansion of its canonical
    #    name or any of its legacy_names.
    entry = ablations.get(ablation_name)
    track_name = None

    if not entry:
        for name, e in ablations.items():
            candidate_names = [name, *e.get("legacy_names", [])]
            for track in e.get("tracks", []):
                for cand in candidate_names:
                    expanded = cand if cand.startswith(f"{track}_") else f"{track}_{cand}"
                    if expanded == ablation_name:
                        entry = e
                        track_name = track
                        break
                if entry:
                    break
            if entry:
                break

    # 2. Extract test_inputs from the entry if it's not a placeholder
    if entry:
        path = entry.get("sweep", {}).get("test_inputs")
        if path and path != "__PLACEHOLDER__":
            return path

    # 3. Fallback to a track default. Prefer the track named in the ablation_name's
    #    own prefix (longest match, so 'pairwise-vanilla-ai' wins over 'pairwise'),
    #    then the setup name.
    if not track_name:
        track_name = next(
            (t for t in sorted(tracks_cfg, key=len, reverse=True)
             if ablation_name.startswith(f"{t}_")),
            None,
        )
    if not track_name and setup in tracks_cfg:
        track_name = setup

    if track_name:
        track_cfg = tracks_cfg.get(track_name, {})
        path = track_cfg.get("sweep", {}).get("test_inputs")
        if path and path != "__PLACEHOLDER__":
            return path

    return None


def _load_test_inputs(path: Path) -> dict:
    """Load JSON or YAML test inputs with caching and mtime validation."""
    path_str = str(path.resolve())
    if not path.exists():
        return {}
        
    mtime = path.stat().st_mtime
    with _CACHE_LOCK:
        if path_str in _TEST_INPUTS_CACHE:
            cached_mtime, data = _TEST_INPUTS_CACHE[path_str]
            if cached_mtime == mtime:
                return data

    try:
        with open(path, encoding="utf-8") as f:
            if path.suffix == ".json":
                data = json.load(f) or {}
            else:
                data = yaml.load(f, Loader=SafeLoader) or {}
    except Exception:
        data = {}

    with _CACHE_LOCK:
        _TEST_INPUTS_CACHE[path_str] = (mtime, data)
    return data


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, str]:
    """Recursively flatten a nested dict to dot-separated key → str(value) pairs."""
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, full_key))
        else:
            out[full_key] = str(v) if v is not None else "null"
    return out


def _build_config_diff_table(canonical_ablation: str) -> str:
    """
    Build a Markdown table comparing the effective sweep+instance parameters of
    the baseline ('current') vs the given ablation.

    A fixed set of key parameters is always shown; additional ablation-specific
    overrides are appended below a separator. Values that differ from the
    baseline are bolded in the Ablation column.
    """
    data = _load_ablations_yaml_data()
    if not data:
        return ""

    ablations = data.get("ablations", {})
    defaults = data.get("defaults", {})
    tracks_cfg = data.get("tracks", {})

    baseline_entry = ablations.get("current", {})
    ablation_entry = ablations.get(canonical_ablation, {})
    if not ablation_entry:
        return ""

    abl_sweep_raw = ablation_entry.get("sweep", {}) or {}
    abl_instance_raw = ablation_entry.get("instance", {}) or {}
    if isinstance(abl_instance_raw, str):
        abl_instance_raw = {}

    # Flatten ablation overrides (used to detect extra params)
    abl_flat: dict[str, str] = {}
    for k, v in abl_sweep_raw.items():
        abl_flat[f"sweep.{k}"] = str(v) if v is not None else "null"
    for k, v in _flatten_dict(abl_instance_raw).items():
        abl_flat[f"instance.{k}"] = v

    def _effective_value(key: str, entry: dict) -> str:
        """Resolve the effective value for a config key from an ablation entry,
        falling back to track configs and then global defaults."""
        section, *rest = key.split(".", 1)
        sub_key = rest[0] if rest else ""
        sources = [entry, defaults]
        for src in sources:
            section_dict = src.get(section, {}) or {}
            if isinstance(section_dict, dict):
                flat = _flatten_dict(section_dict)
                if sub_key in flat:
                    return flat[sub_key]
        for track in ablation_entry.get("tracks", []):
            track_section = tracks_cfg.get(track, {}).get(section, {}) or {}
            if isinstance(track_section, dict):
                flat = _flatten_dict(track_section)
                if sub_key in flat:
                    return flat[sub_key]
        return "—"

    def _abl_value(key: str) -> str:
        """Return ablation value: from explicit override if present, else same as baseline."""
        if key in abl_flat:
            return abl_flat[key]
        return _effective_value(key, baseline_entry)

    _PROMPT_LABELS: dict[str, str] = {
        "remove_eval_data":       "abstract",
        "extract_research_plan":  "plan",
        "extract_idea_summary":   "plan (verbatim)",
        "boring_idea_generation": "boring idea",
    }

    def _fmt_manipulation_prompt(val: str) -> str:
        """Convert a template file path to a short readable label."""
        if val == "—":
            return val
        stem = Path(val).stem
        return _PROMPT_LABELS.get(stem, stem.replace("_", " ").replace("-", " "))

    # Fixed key parameters always shown, in display order
    _ALWAYS_SHOW: list[tuple[str, str, bool]] = [
        # (config_key, display_name, apply_prompt_fmt)
        ("sweep.reasoning_effort",          "Reasoning effort",         False),
        ("sweep.mec_k",                     "Samples per direction",    False),
        ("sweep.retrieve_related_work",     "Retrieve related work",    False),
        ("sweep.novelty_criterion_override",   "Novelty criterion",          False),
        ("sweep.pairwise_judge_template",      "Pairwise judge template",    False),
        ("sweep.pointwise_judge_template",     "Pointwise judge template",   False),
        ("instance.manipulation_prompt",       "Idea format",                True),
    ]

    _SKIP_KEYS = {
        "sweep.test_inputs", "sweep.configs", "sweep.output_dir",
        "sweep.output_file", "sweep.retrieve_related_work",  # shown in always-show
        "instance.manipulation_prompt",                       # shown in always-show
    }

    rows = []

    # --- Always-shown rows ---
    for key, label, is_prompt in _ALWAYS_SHOW:
        base_val = _effective_value(key, baseline_entry)
        abl_val  = _abl_value(key)
        if is_prompt:
            base_val = _fmt_manipulation_prompt(base_val)
            abl_val  = _fmt_manipulation_prompt(abl_val)
        # Shorten long novelty criterion text
        if key == "sweep.novelty_criterion_override":
            base_val = "default" if base_val == "—" else "custom"
            abl_val  = "default" if abl_val  == "—" else "custom"
        abl_cell = f"**{abl_val}**" if abl_val != base_val else abl_val
        rows.append(f"| {label} | {base_val} | {abl_cell} |")

    # --- Extra override rows (params not already covered above) ---
    already_shown = {key for key, _, _ in _ALWAYS_SHOW}
    extra_rows = []
    for key, abl_val in sorted(abl_flat.items()):
        if key in already_shown or key in _SKIP_KEYS:
            continue
        base_val = _effective_value(key, baseline_entry)
        param_name = (
            key.replace("sweep.", "").replace("instance.", "")
               .replace("_", " ").replace(".", " › ")
        )
        abl_cell = f"**{abl_val}**" if abl_val != base_val else abl_val
        extra_rows.append(f"| {param_name} | {base_val} | {abl_cell} |")

    if extra_rows:
        rows += ["| | | |"] + extra_rows  # blank separator row

    lines = [
        "### Configuration",
        "",
        "| Parameter | Baseline | Ablation |",
        "| --- | --- | --- |",
    ] + rows + [""]
    return "\n".join(lines)
