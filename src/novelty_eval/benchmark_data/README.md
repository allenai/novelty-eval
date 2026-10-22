# Rebuilding the benchmark

The files in [`data/`](../../../data) come out of the pipeline in this folder, which can be re-run on a newer conference cycle:

```
OpenReview ──► fetch_iclr_data.py ──► clean novelty dataset (JSON)
                                             │
                                             ▼
                        create_benchmark_instances.py ──► pairwise / pointwise YAML
                                             │
                                             ▼
                             apply_manipulation.sh ──► *_plan YAML
```

Run everything from the repository root with the environment from [Getting started](../../../README.md#getting-started) exported. Stage 1 also needs `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD`.

## Stage 1 — harvest novelty labels from ICLR reviews

```bash
python src/novelty_eval/benchmark_data/fetch_iclr_data.py \
    --year 2026 \
    --percentile 10 \
    --model claude-opus-4-6 \
    --output_dir src/novelty_eval/benchmark_data/iclr_data \
    --input_json ""
```

Submissions are grouped by primary area, the top and bottom `--percentile` of each area are kept, and the extraction model pulls verbatim novelty statements out of every review. 

`--input_json ""` forces a fresh pull from OpenReview; pass a path instead to re-process an earlier dump. The stage writes `iclr_<year>_clean_novelty_dataset_<p>_percent.json` (plus the raw and intermediate dumps) to `<output_dir>/<timestamp>/`.

## Stage 2 — build benchmark instances

Stage 2 turns the harvested papers into benchmark instances: it keeps the papers whose reviewer evidence is strong enough, decides where the low-novelty pool comes from (ICLR or generated), rewrites the abstracts, and emits pairwise or pointwise instances. One config file controls all of it:

```yaml
# what to build from: the clean dataset written by stage 1
iclr_data:
  - "…/iclr_2026_clean_novelty_dataset_10_percent.json"
output_dir: "output/benchmark_instances/example"

# pairwise (one D_+ idea against one D_- idea) or pointwise (one labeled idea)
pointwise: false
num_top_papers: 1
num_bottom_papers: 1

# which papers qualify: reviewer agreement, then score thresholds
strictness: majority          # null | majority | all
min_pos_rating: 6.0           # D_+
min_pos_contribution: 3.0
max_neg_rating: 3.5           # D_-
max_neg_contribution: 2.0

# where D_- comes from: false → low-rated papers, true → generated ideas
llm_negatives: false
llm_negatives_model: "claude-sonnet-4-5"
llm_negatives_prompt: "…/templates/idea_generation.jinja2"

# rewrite each abstract before it becomes an instance
model_name: claude-opus-4-6
manipulation_prompt: "…/templates/remove_eval_data.jinja2"
```

[config/example.yaml](config/example.yaml) is the same config in full, with every field commented. Copy it, point `iclr_data:` at your stage-1 output, and run:

```bash
python src/novelty_eval/benchmark_data/create_benchmark_instances.py \
    --config src/novelty_eval/benchmark_data/config/example.yaml
```

`--dry_run` prints the expected instance counts without spending an LLM call. A run writes a timestamped folder under `output_dir` with the instances, a markdown report, `data_summary.md` and a cost report.


### Keeping pointwise and pairwise on the same ideas

Building a pointwise set from scratch re-samples the pools, so it will not line up with a pairwise file you already have. To get exactly the same ideas, split the pairwise file instead of rebuilding:

```bash
python src/novelty_eval/benchmark_data/create_benchmark_instances.py \
    --config src/novelty_eval/benchmark_data/config/example.yaml \
    --derive_pointwise_from_pairwise data/human-only/pairwise.yaml \
    --output_dir output/benchmark_instances/example-pointwise
```


To confirm the two files really carry the same ideas:

```bash
python scripts/checks/verify_pointwise_from_pairwise.py \
    data/human-only/pairwise.yaml \
    data/human-only/pointwise.yaml
```

## Stage 3 — plan format

The `*_plan` files are stage-2 instances with each abstract rewritten into a *purpose* / *mechanism* plan:

```bash
./scripts/benchmark/apply_manipulation.sh \
    --dataset output/iclr_test_instances/pairwise_data/<timestamp>/benchmark_instances.yaml \
    --template src/novelty_eval/benchmark_data/templates/extract_research_plan.jinja2
```

The pointwise plan files are then derived from the pairwise plan file with the same `--derive_pointwise_from_pairwise` step as above.
