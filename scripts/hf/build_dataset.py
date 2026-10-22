#!/usr/bin/env python3
"""Build the Hugging Face dataset repo from the YAML benchmark files in data/.

Emits a folder that can be uploaded as-is:

    <out>/README.md                     dataset card (front-matter generated from the manifest)
    <out>/assets/<image>.png            images the card shows
    <out>/data/<config>.parquet         one Parquet file per config, split 'test'

The YAML in data/ stays authoritative and is not shipped: the Parquet is lossless
with respect to it, so a second copy on the Hub would only be one more thing to keep
in sync. Every structural assumption is asserted while converting, so a change to the
YAML fails here rather than silently reshaping the released data.

Usage:
    python scripts/hf/build_dataset.py
    python scripts/hf/build_dataset.py --data-dir data --out output/hf_dataset

scripts/hf/upload.sh runs this and pushes the result to the Hub.
"""

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


CARD_TEMPLATE = Path(__file__).with_name("dataset_card.md")

# Shipped with the card so its images don't depend on any GitHub repo.
CARD_IMAGES = ["fig1-oct14-media.png", "data-creation-overview.png"]
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"

# Config name -> relative YAML path. Every file under --data-dir must land here,
# and this exact set must be present; both are checked.
EXPECTED_CONFIGS = {
    "human-only_pointwise": "human-only/pointwise.yaml",
    "human-only_pointwise_plan": "human-only/pointwise_plan.yaml",
    "human-only_pairwise": "human-only/pairwise.yaml",
    "human-only_pairwise_plan": "human-only/pairwise_plan.yaml",
    "human-plus-generated_pointwise": "human-plus-generated/pointwise.yaml",
    "human-plus-generated_pointwise_plan": "human-plus-generated/pointwise_plan.yaml",
    "human-plus-generated_pairwise": "human-plus-generated/pairwise.yaml",
    "human-plus-generated_pairwise_plan": "human-plus-generated/pairwise_plan.yaml",
    "human-plus-generated_pointwise_backbone-gpt-5": "human-plus-generated/backbone-gpt-5/pointwise.yaml",
    "human-plus-generated_pairwise_backbone-gpt-5": "human-plus-generated/backbone-gpt-5/pairwise.yaml",
    "human-plus-generated_pointwise_backbone-gpt-5.1": "human-plus-generated/backbone-gpt-5.1/pointwise.yaml",
    "human-plus-generated_pairwise_backbone-gpt-5.1": "human-plus-generated/backbone-gpt-5.1/pairwise.yaml",
    "human-plus-generated_pointwise_backbone-gpt-5.4": "human-plus-generated/backbone-gpt-5.4/pointwise.yaml",
    "human-plus-generated_pairwise_backbone-gpt-5.4": "human-plus-generated/backbone-gpt-5.4/pairwise.yaml",
    "human-plus-generated_pointwise_backbone-opus-4-5": "human-plus-generated/backbone-opus-4-5/pointwise.yaml",
    "human-plus-generated_pairwise_backbone-opus-4-5": "human-plus-generated/backbone-opus-4-5/pairwise.yaml",
}

DEFAULT_CONFIG = "human-plus-generated_pointwise"

# Stronger-vs-weaker ideator pairs, used only to count verdict flips. They have no
# gold label, so they stay out of the release.
UNRELEASED_DIR = "ai-only"

# Never made it past the experimental stage and is empty in every instance, so it is
# left out of the release rather than shipped half-built.
DROPPED_FIELD = "similar_papers_mentioned"

NOT_APPLICABLE = "N/A"
STR_LIST = pa.list_(pa.string())

POINTWISE_SCHEMA = pa.schema(
    [
        ("id", pa.int32()),
        ("iclr_area", pa.string()),
        ("idea", pa.string()),
        ("label", pa.string()),
        ("idea_source", pa.string()),
        ("title", pa.string()),
        ("rating", pa.float64()),
        ("contribution", pa.float64()),
        ("positive_signals", STR_LIST),
        ("negative_signals", STR_LIST),
    ]
)

PAIRWISE_SCHEMA = pa.schema(
    [("id", pa.int32()), ("iclr_area", pa.string())]
    + [(f"idea_{s}", pa.string()) for s in ("a", "b")]
    + [("expected_winner", pa.int8())]
    + [
        (f"{field}_{side}", dtype)
        for field, dtype in [
            ("label", pa.string()),
            ("idea_source", pa.string()),
            ("title", pa.string()),
            ("rating", pa.float64()),
            ("contribution", pa.float64()),
            ("positive_signals", STR_LIST),
            ("negative_signals", STR_LIST),
        ]
        for side in ("a", "b")
    ]
)


def task_of(config: str) -> str:
    return "pairwise" if "_pairwise" in config else "pointwise"


def to_float(value) -> float | None:
    """Reviewer scores are strings in the YAML, and 'N/A' for generated ideas."""
    if value is None or value == NOT_APPLICABLE or value == "":
        return None
    return float(value)


def idea_source(metadata: dict) -> str:
    """Generated ideas carry no reviewer scores; that is what marks them."""
    scored = metadata["rating"] != NOT_APPLICABLE
    assert scored == (metadata["contribution"] != NOT_APPLICABLE), (
        f"half-scored idea: rating={metadata['rating']!r} contribution={metadata['contribution']!r}"
    )
    return "human" if scored else "generated"


def check_metadata(metadata: dict, context: str, where: str) -> None:
    assert set(metadata) == {
        "area",
        "contribution",
        "negative_signals",
        "positive_signals",
        "rating",
        "similar_papers_mentioned",
        "title",
        "type",
    }, f"{where}: unexpected metadata fields {sorted(metadata)}"
    # 'area' is dropped on the way out; it is a duplicate of 'context' everywhere.
    assert metadata["area"] == context, f"{where}: metadata.area != context"
    assert metadata["type"] in ("POSITIVE", "NEGATIVE"), f"{where}: bad type {metadata['type']!r}"
    # 'similar_papers_mentioned' never got past the experimental stage and is empty in
    # every instance; it is stripped from the release. If that ever stops being true,
    # this fails rather than quietly shipping half-populated extraction output.
    assert not metadata["similar_papers_mentioned"], f"{where}: similar_papers_mentioned is populated"


def flatten_metadata(metadata: dict, suffix: str = "") -> dict:
    return {
        f"label{suffix}": metadata["type"],
        f"idea_source{suffix}": idea_source(metadata),
        f"title{suffix}": metadata["title"],
        f"rating{suffix}": to_float(metadata["rating"]),
        f"contribution{suffix}": to_float(metadata["contribution"]),
        f"positive_signals{suffix}": list(metadata["positive_signals"]),
        f"negative_signals{suffix}": list(metadata["negative_signals"]),
    }


def convert_pointwise(data: dict, path: Path) -> list[dict]:
    rows = []
    for key, instance in data.items():
        where = f"{path}:{key}"
        assert set(instance) == {"context", "idea", "label", "metadata"}, (
            f"{where}: unexpected fields {sorted(instance)}"
        )
        metadata = instance["metadata"]
        check_metadata(metadata, instance["context"], where)
        # 'type' is dropped on the way out; it never disagrees with 'label'.
        assert metadata["type"] == instance["label"], f"{where}: metadata.type != label"
        flat = flatten_metadata(metadata)
        flat.pop("label")
        rows.append(
            {
                "id": int(key),
                "iclr_area": instance["context"],
                "idea": instance["idea"],
                "label": instance["label"],
                **flat,
            }
        )
    return rows


def convert_pairwise(data: dict, path: Path) -> list[dict]:
    rows = []
    for key, instance in data.items():
        where = f"{path}:{key}"
        assert set(instance) == {"context", "expected_winners", "ideas", "metadata"}, (
            f"{where}: unexpected fields {sorted(instance)}"
        )
        assert set(instance["ideas"]) == {0, 1}, f"{where}: expected exactly two ideas"
        assert set(instance["metadata"]) == {0, 1}, f"{where}: metadata does not cover both ideas"
        winners = instance["expected_winners"]
        assert len(winners) == 1 and winners[0] in (0, 1), f"{where}: bad expected_winners {winners}"
        winner = winners[0]

        row = {"id": int(key), "iclr_area": instance["context"], "expected_winner": winner}
        for index, side in enumerate(("a", "b")):
            metadata = instance["metadata"][index]
            check_metadata(metadata, instance["context"], f"{where}[{index}]")
            row[f"idea_{side}"] = instance["ideas"][index]
            row.update(flatten_metadata(metadata, suffix=f"_{side}"))
        assert row[f"label_{'ab'[winner]}"] == "POSITIVE", f"{where}: winner is not the POSITIVE idea"
        assert row[f"label_{'ba'[winner]}"] == "NEGATIVE", f"{where}: loser is not the NEGATIVE idea"
        rows.append(row)
    return rows


def summarise(config: str, rows: list[dict]) -> dict:
    """Per-config numbers for the card's table."""
    if task_of(config) == "pointwise":
        positives = sum(r["label"] == "POSITIVE" for r in rows)
        composition = f"{positives} novel / {len(rows) - positives} not novel"
    else:
        first = sum(r["expected_winner"] == 0 for r in rows)
        composition = f"{first} pairs answer A / {len(rows) - first} answer B"
    return {"rows": len(rows), "composition": composition}


def build_configs_yaml(configs: list[str]) -> str:
    lines = []
    for config in configs:
        lines.append(f"- config_name: {config}")
        if config == DEFAULT_CONFIG:
            lines.append("  default: true")
        lines.append("  data_files:")
        lines.append("  - split: test")
        lines.append(f"    path: data/{config}.parquet")
    return "\n".join(lines)


def write_card(out_dir: Path, configs: list[str]) -> None:
    card = CARD_TEMPLATE.read_text()
    assert "{{CONFIGS_YAML}}" in card, "card template is missing {{CONFIGS_YAML}}"
    (out_dir / "README.md").write_text(card.replace("{{CONFIGS_YAML}}", build_configs_yaml(configs)))
    (out_dir / "assets").mkdir()
    for image in CARD_IMAGES:
        assert f"assets/{image}" in card, f"card does not use {image}"
        shutil.copy(ASSETS_DIR / image, out_dir / "assets" / image)


def resolve_configs(data_dir: Path) -> dict[str, Path]:
    found = {p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*.yaml")}
    found = {rel for rel in found if not rel.startswith(f"{UNRELEASED_DIR}/")}
    expected = set(EXPECTED_CONFIGS.values())
    assert found == expected, (
        "data/ does not match the manifest in this script — update EXPECTED_CONFIGS.\n"
        f"  only on disk:   {sorted(found - expected)}\n"
        f"  only in script: {sorted(expected - found)}"
    )
    return {config: data_dir / rel for config, rel in EXPECTED_CONFIGS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="source YAML tree")
    parser.add_argument("--out", type=Path, default=Path("output/hf_dataset"), help="folder to build")
    args = parser.parse_args()

    sources = resolve_configs(args.data_dir)

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "data").mkdir(parents=True)

    stats: dict[str, dict] = {}
    for config, path in sources.items():
        with open(path) as f:
            data = yaml.safe_load(f)
        assert len(set(data)) == len(data), f"{path}: duplicate instance ids"
        convert = convert_pairwise if task_of(config) == "pairwise" else convert_pointwise
        rows = convert(data, path)
        schema = PAIRWISE_SCHEMA if task_of(config) == "pairwise" else POINTWISE_SCHEMA
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, args.out / "data" / f"{config}.parquet", compression="zstd")
        stats[config] = summarise(config, rows)
        print(f"{config:<48} {stats[config]['rows']:>4} rows  ({stats[config]['composition']})")

    write_card(args.out, list(sources))

    total = sum(s["rows"] for s in stats.values())
    print(f"\n{len(stats)} configs, {total} rows -> {args.out}")
    print("Review README.md, then upload with scripts/hf/upload.sh")


if __name__ == "__main__":
    main()
