"""Compare POSITIVE ideas across multiple iclr_test_instances YAML files.

Generates an HTML report showing which papers appear in all/some/one dataset,
with inline word-level colored diffs (vs the first file as baseline) for
papers that have textual differences across datasets.

Usage:
    python scripts/compare_datasets.py \
        --files path/to/run_a/iclr_test_instances.yaml \
                path/to/run_b/iclr_test_instances.yaml \
                path/to/run_c/iclr_test_instances.yaml \
        [--output compare_positives.html]
"""

import argparse
import difflib
import html
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml


# Palette for up to 8 datasets: (tag-bg, tag-fg, label-color)
_PALETTE = [
    ("#dbeafe", "#1d4ed8", "#1d4ed8"),  # blue
    ("#ede9fe", "#6d28d9", "#6d28d9"),  # purple
    ("#d1fae5", "#065f46", "#065f46"),  # green
    ("#fef3c7", "#92400e", "#b45309"),  # amber
    ("#fce7f3", "#9d174d", "#be185d"),  # pink
    ("#e0e7ff", "#3730a3", "#4338ca"),  # indigo
    ("#f0fdf4", "#166534", "#15803d"),  # lime
    ("#fff7ed", "#7c2d12", "#c2410c"),  # orange
]


@dataclass
class PositiveIdea:
    instance_idx: int
    idea_idx: int
    title: str
    text: str
    rating: str
    contribution: str
    positive_signals: list[str] = field(default_factory=list)
    context: str = ""


def _normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    return re.sub(r"\s+", " ", title).strip()


def _extract_positives(data: dict) -> dict[str, PositiveIdea]:
    """Returns {normalized_title: PositiveIdea}."""
    result = {}
    for instance_idx, instance in data.items():
        metadata = instance.get("metadata", {})
        ideas = instance.get("ideas", {})
        context = instance.get("context", "")
        for idea_idx, meta in metadata.items():
            if meta.get("type") != "POSITIVE":
                continue
            title = meta.get("title", "")
            norm = _normalize_title(title)
            result[norm] = PositiveIdea(
                instance_idx=instance_idx,
                idea_idx=idea_idx,
                title=title,
                text=str(ideas.get(idea_idx, "")),
                rating=str(meta.get("rating", "N/A")),
                contribution=str(meta.get("contribution", "N/A")),
                positive_signals=meta.get("positive_signals") or [],
                context=context,
            )
    return result


def _label_from_path(path: Path) -> str:
    parts = path.parts
    for i, p in enumerate(parts):
        if p == "ablation" and i + 1 < len(parts):
            return parts[i + 1]
    return path.parent.name


def _word_diff_html(text_base: str, text_other: str) -> str:
    tokens_a = re.split(r"(\s+)", text_base)
    tokens_b = re.split(r"(\s+)", text_other)
    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
    parts = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            parts.append(html.escape("".join(tokens_a[i1:i2])))
        elif opcode == "delete":
            parts.append(f'<span class="del">{html.escape("".join(tokens_a[i1:i2]))}</span>')
        elif opcode == "insert":
            parts.append(f'<span class="ins">{html.escape("".join(tokens_b[j1:j2]))}</span>')
        elif opcode == "replace":
            parts.append(
                f'<span class="del">{html.escape("".join(tokens_a[i1:i2]))}</span>'
                f'<span class="ins">{html.escape("".join(tokens_b[j1:j2]))}</span>'
            )
    return "".join(parts)


def _file_tag(label: str, idx: int) -> str:
    bg, fg, _ = _PALETTE[idx % len(_PALETTE)]
    return (
        f'<span style="display:inline-block;font-size:0.7rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.05em;padding:1px 6px;'
        f'border-radius:3px;background:{bg};color:{fg}">'
        f"{html.escape(label)}</span>"
    )


def _file_label_span(label: str, idx: int) -> str:
    _, _, color = _PALETTE[idx % len(_PALETTE)]
    return f'<span style="font-weight:600;color:{color}">{html.escape(label)}</span>'


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px; line-height: 1.6; color: #1a1a1a;
    background: #f5f5f5; padding: 2rem;
}
h1 { font-size: 1.6rem; margin-bottom: 0.5rem; }
h2 { font-size: 1.2rem; margin: 2rem 0 1rem; border-bottom: 2px solid #ccc; padding-bottom: 0.3rem; }
.meta-box {
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 1rem 1.25rem; margin-bottom: 1.5rem; font-size: 0.85rem; color: #555;
}
.meta-box strong { color: #222; }
.summary-table { border-collapse: collapse; margin: 1rem 0; }
.summary-table td, .summary-table th {
    padding: 6px 16px; border: 1px solid #ddd; text-align: left;
}
.summary-table th { background: #f0f0f0; }
.diff-card {
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    margin-bottom: 1.5rem; overflow: hidden;
}
.diff-card-header {
    background: #fafafa; border-bottom: 1px solid #ddd;
    padding: 0.6rem 1rem;
}
.diff-card-title { font-weight: 600; font-size: 0.95rem; display: block; margin-bottom: 0.2rem; }
.diff-card-meta { font-size: 0.78rem; color: #777; }
.diff-card-body { padding: 1rem 1.25rem; }
.diff-block { margin-bottom: 1rem; }
.diff-block:last-child { margin-bottom: 0; }
.diff-block-label {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: #888; margin-bottom: 0.3rem;
}
.diff-text { white-space: pre-wrap; font-family: inherit; line-height: 1.7; }
span.del {
    background: #ffd7d7; color: #a80000; text-decoration: line-through;
    border-radius: 2px; padding: 0 1px;
}
span.ins {
    background: #d4f5d4; color: #1a6e1a;
    border-radius: 2px; padding: 0 1px;
}
.tag-same { display:inline-block; font-size:0.7rem; font-weight:600; padding:1px 6px;
    border-radius:3px; background:#d1fae5; color:#065f46; }
.tag-diff { display:inline-block; font-size:0.7rem; font-weight:600; padding:1px 6px;
    border-radius:3px; background:#fef3c7; color:#92400e; }
.identical-msg { color: #666; font-style: italic; font-size: 0.85rem; }
.identical-list { color: #555; font-size: 0.85rem; }
.identical-list li { padding: 2px 0; }
details summary {
    cursor: pointer; font-weight: 600; font-size: 0.9rem;
    padding: 0.4rem 0; color: #444;
}
details summary:hover { color: #000; }
.partial-card {
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    margin-bottom: 0.75rem; overflow: hidden;
}
.partial-card-header {
    background: #fafafa; border-bottom: 1px solid #ddd;
    padding: 0.5rem 1rem; font-weight: 600; font-size: 0.88rem;
}
.partial-card-files { padding: 0.4rem 1rem; font-size: 0.82rem; color: #555; }
"""


def _build_html(
    labels: list[str],
    paths: list[Path],
    all_maps: list[dict[str, PositiveIdea]],
) -> str:
    n = len(labels)
    all_keys = set(all_maps[0])
    for m in all_maps[1:]:
        all_keys |= set(m)
    all_keys = sorted(all_keys)

    # For each key, which file indices contain it
    presence: dict[str, list[int]] = {
        k: [i for i, m in enumerate(all_maps) if k in m] for k in all_keys
    }

    keys_in_all = sorted(k for k, idxs in presence.items() if len(idxs) == n)
    keys_partial = sorted(k for k, idxs in presence.items() if 1 < len(idxs) < n)
    keys_unique = sorted(k for k, idxs in presence.items() if len(idxs) == 1)

    baseline_map = all_maps[0]

    # Within keys_in_all: which have any textual difference?
    def _any_diff(k: str) -> bool:
        base_text = all_maps[0][k].text
        return any(all_maps[i][k].text != base_text for i in range(1, n))

    diff_keys = [k for k in keys_in_all if _any_diff(k)]
    same_keys = [k for k in keys_in_all if not _any_diff(k)]

    body = []

    # ── Meta box ────────────────────────────────────────────────────────────
    body.append('<div class="meta-box">')
    for i, (label, path) in enumerate(zip(labels, paths)):
        body.append(
            f"<strong>File {i}</strong> {_file_tag(label, i)}<br>"
            f"<code>{html.escape(str(path))}</code><br>"
            + ("<br>" if i < n - 1 else "")
        )
    body.append(f"<br>Generated: {date.today().isoformat()}")
    body.append("</div>")

    # ── Summary ─────────────────────────────────────────────────────────────
    body.append("<h2>Summary</h2>")
    body.append('<table class="summary-table">')
    body.append("<tr><th>Dataset</th><th>Positives</th></tr>")
    for i, (label, m) in enumerate(zip(labels, all_maps)):
        body.append(f"<tr><td>{_file_tag(label, i)}</td><td>{len(m)}</td></tr>")
    body.append("<tr><td colspan='2' style='padding:0'></td></tr>")
    body.append(f"<tr><td><strong>In all {n} datasets</strong></td><td><strong>{len(keys_in_all)}</strong></td></tr>")
    body.append(
        f'<tr><td>&nbsp;&nbsp;↳ with <span class="tag-diff">differences</span></td>'
        f"<td><strong>{len(diff_keys)}</strong></td></tr>"
    )
    body.append(
        f'<tr><td>&nbsp;&nbsp;↳ <span class="tag-same">identical</span></td>'
        f"<td>{len(same_keys)}</td></tr>"
    )
    if keys_partial:
        body.append(f"<tr><td>In some (not all) datasets</td><td>{len(keys_partial)}</td></tr>")
    if keys_unique:
        body.append(f"<tr><td>Unique to one dataset</td><td>{len(keys_unique)}</td></tr>")
    body.append(f"<tr><td>Total unique papers</td><td>{len(all_keys)}</td></tr>")
    body.append("</table>")

    # ── Papers with differences (in all N) ─────────────────────────────────
    body.append(f"<h2>Papers in All {n} Datasets — with Differences ({len(diff_keys)})</h2>")
    if not diff_keys:
        body.append("<p><em>All matching papers have identical text.</em></p>")

    for key in diff_keys:
        base = all_maps[0][key]
        body.append('<div class="diff-card">')
        body.append('<div class="diff-card-header">')
        body.append(f'<span class="diff-card-title">{html.escape(base.title)}</span>')
        meta_parts = [f"context: {html.escape(base.context)}"]
        for i, m in enumerate(all_maps):
            idea = m[key]
            meta_parts.append(f"{_file_label_span(labels[i], i)} rating={html.escape(idea.rating)}")
        body.append(f'<span class="diff-card-meta">{" &nbsp;|&nbsp; ".join(meta_parts)}</span>')
        body.append("</div>")
        body.append('<div class="diff-card-body">')

        base_text = all_maps[0][key].text
        for i in range(1, n):
            other_text = all_maps[i][key].text
            body.append('<div class="diff-block">')
            body.append(
                f'<div class="diff-block-label">'
                f'{_file_tag(labels[0], 0)} baseline → {_file_tag(labels[i], i)}'
                f"</div>"
            )
            if other_text == base_text:
                body.append('<p class="identical-msg">identical to baseline</p>')
            else:
                diff_content = _word_diff_html(base_text, other_text)
                body.append(f'<p class="diff-text">{diff_content}</p>')
            body.append("</div>")

        body.append("</div>")  # diff-card-body
        body.append("</div>")  # diff-card

    # ── Identical papers (in all N) — collapsed ─────────────────────────────
    body.append(f"<h2>Papers in All {n} Datasets — Identical ({len(same_keys)})</h2>")
    body.append("<details>")
    body.append(f"<summary>Show {len(same_keys)} papers with no differences</summary>")
    body.append('<ul class="identical-list" style="margin-top:0.5rem;padding-left:1.5rem;">')
    for key in same_keys:
        body.append(f"<li>{html.escape(all_maps[0][key].title)}</li>")
    body.append("</ul></details>")

    # ── Papers in some but not all ──────────────────────────────────────────
    if keys_partial:
        body.append(f"<h2>Papers in Some Datasets (not all {n}) ({len(keys_partial)})</h2>")
        body.append("<details><summary>Show</summary><div style='margin-top:0.5rem'>")
        for key in keys_partial:
            idxs = presence[key]
            # Use any file that has it for title/context
            idea = all_maps[idxs[0]][key]
            body.append('<div class="partial-card">')
            body.append(f'<div class="partial-card-header">{html.escape(idea.title)}</div>')
            present_tags = " ".join(_file_tag(labels[i], i) for i in idxs)
            absent_labels = ", ".join(
                html.escape(labels[i]) for i in range(n) if i not in idxs
            )
            body.append(
                f'<div class="partial-card-files">'
                f"Present in: {present_tags} &nbsp;&nbsp; "
                f'<span style="color:#999">Absent from: {absent_labels}</span>'
                f"</div>"
            )
            body.append("</div>")
        body.append("</div></details>")

    # ── Unique papers (in only one dataset) — collapsed per file ────────────
    if keys_unique:
        body.append(f"<h2>Papers Unique to One Dataset ({len(keys_unique)})</h2>")
        for fi in range(n):
            mine = [k for k in keys_unique if presence[k] == [fi]]
            if not mine:
                continue
            body.append(
                f"<details><summary>{_file_tag(labels[fi], fi)} — {len(mine)} unique papers</summary>"
                f"<ul class='identical-list' style='margin-top:0.5rem;padding-left:1.5rem;'>"
            )
            for k in mine:
                body.append(f"<li>{html.escape(all_maps[fi][k].title)}</li>")
            body.append("</ul></details>")

    title_str = " vs ".join(html.escape(l) for l in labels)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Positive Ideas Comparison — {title_str}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Positive Ideas Comparison</h1>
{"".join(body)}
</body>
</html>"""


def _load_config(config_path: Path) -> tuple[list[Path], list[str | None], Path | None]:
    """Parse a YAML config file.

    Supported formats:

        # simple list of paths
        files:
          - path/to/run_a/iclr_test_instances.yaml
          - path/to/run_b/iclr_test_instances.yaml

        # list of dicts with optional label override
        files:
          - path: path/to/run_a/iclr_test_instances.yaml
            label: GPT-5.4          # optional
          - path: path/to/run_b/iclr_test_instances.yaml

        output: compare.html        # optional
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw = cfg.get("files", [])
    paths, labels = [], []
    cwd = Path.cwd()
    for entry in raw:
        if isinstance(entry, dict):
            paths.append(cwd / entry["path"])
            labels.append(entry.get("label"))
        else:
            paths.append(cwd / str(entry))
            labels.append(None)

    output = Path(cwd / cfg["output"]) if cfg.get("output") else None
    return paths, labels, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare POSITIVE ideas across multiple dataset YAMLs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--files", nargs="+", type=Path, metavar="YAML",
                        help="Two or more iclr_test_instances.yaml paths.")
    source.add_argument("--config", type=Path, metavar="CONFIG_YAML",
                        help="YAML config file listing files (and optional labels/output).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output HTML path (overrides config file value).")
    args = parser.parse_args()

    config_output: Path | None = None
    config_labels: list[str | None] = []

    if args.config:
        cfg_paths, config_labels, config_output = _load_config(args.config.resolve())
        raw_paths = cfg_paths
    else:
        raw_paths = args.files

    if len(raw_paths) < 2:
        parser.error("Provide at least 2 files.")

    paths = [p.resolve() for p in raw_paths]
    labels = [
        (config_labels[i] if i < len(config_labels) and config_labels[i] else _label_from_path(p))
        for i, p in enumerate(paths)
    ]
    all_maps = []
    for label, path in zip(labels, paths):
        with open(path) as f:
            data = yaml.safe_load(f)
        m = _extract_positives(data)
        all_maps.append(m)
        print(f"{label}: {len(m)} positive ideas")

    report = _build_html(labels, paths, all_maps)

    output_path = args.output or config_output or paths[0].parent / "compare_positives.html"
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
