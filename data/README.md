# Benchmark instances

The evaluation instances used in the paper. Each file is one benchmark: a list of
instances, each holding the idea(s) to judge and the ground-truth label.

```
human-only/                     D_+ = validated novel ICLR papers
                                D_- = validated lower-novelty ICLR papers
  pairwise.yaml                   which of two ideas is more novel
  pairwise_plan.yaml              same pairs, ideas converted to plan form
  pointwise.yaml                  is this idea novel (binary)
  pointwise_plan.yaml             same ideas, plan form

human-plus-generated/           D_+ unchanged; D_- = ideas from a plain LLM ideator (sonnet-4-5)
  pairwise.yaml                   
  pairwise_plan.yaml
  pointwise.yaml
  pointwise_plan.yaml

  backbone-gpt-5/               D_- regenerated with a different ideation backbone
  backbone-gpt-5.1/
  backbone-gpt-5.4/             
  backbone-opus-4-5/
    pairwise.yaml
    pointwise.yaml

ai-only/                        stronger vs. weaker ideator, both LLM-generated (no gold label)
  claude/pairwise.yaml            claude-opus-4-5 vs. claude-sonnet-4-5
  gpt/pairwise.yaml               gpt-5.4 vs. gpt-5.1

expert-annotations/
  judge_errors.csv              judge mistakes re-judged blind by a domain expert
```

`_plan` variants hold the same ideas rewritten into a two-field plan (purpose,
mechanism) and are the idea-format ablation.

### 🤗 Hugging Face
These files are also released on the Hub as
[noystl/novelty-judge-bench](https://huggingface.co/datasets/noystl/novelty-judge-bench)

## Expert annotations

`expert-annotations/judge_errors.csv` has one row per pairwise instance a domain
expert judged blind. 28 are instances the LLM judge got wrong, and 2 are controls
it got right. The two ideas were shown in a random A/B order with no scores or
labels, and every verdict below is given in those A/B terms.

| Column | Meaning |
|---|---|
| `item_id` | Row id |
| `problem_id` | The instance's key in `source_file` |
| `source_file` | The benchmark file the pair comes from |
| `generation_model` | The model that generated that file's `D_-` ideas |
| `judge_model` | The LLM judge whose verdict is in `predicted` |
| `shown_A_idea`, `shown_B_idea` | The two ideas' texts, as shown to the expert |
| `gold` | The more novel idea by the benchmark label (`A`/`B`) |
| `predicted` | The judge's verdict (`A`/`B`/`tie`) |
| `annotation` | The expert's choice (`A`/`B`) |
| `annotation_reasoning` | The expert's explanation |

## Intended use

These files are for **evaluating novelty judges**: measuring how well an automatic judge
(e.g., a prompted LLM) tells more novel research ideas from less novel ones. They are a small
test set, not training data, and should not be used to train or fine-tune models.

## Model-generated content

Parts of these files were produced by proprietary models from Anthropic (Claude) and OpenAI (GPT).

### Models and their roles

| Model | Provider | Role in the paper | Output in these files | Volume |
|---|---|---|---|---|
| Claude Sonnet 4.5 | Anthropic | Idea generator; judge | `D_-` ideas of `human-plus-generated/`; 5 verdicts in `judge_errors.csv` | 154 ideas; 5 verdicts |
| Claude Opus 4.5 | Anthropic | Idea generator; judge | `D_-` ideas of `backbone-opus-4-5/` | 154 ideas |
| GPT-5 | OpenAI | Idea generator | `D_-` ideas of `backbone-gpt-5/` | 154 ideas |
| GPT-5.1 | OpenAI | Idea generator; judge | `D_-` ideas of `backbone-gpt-5.1/` | 154 ideas |
| GPT-5.4 | OpenAI | Idea generator; judge | `D_-` ideas of `backbone-gpt-5.4/` | 154 ideas |
| Claude Opus 4.6 | Anthropic | Data processing and labeling; judge | Edited human abstracts; plan-form rewrites; selection of reviewer excerpts; 25 verdicts in `judge_errors.csv` | 299 abstracts; 453 plans; 1,192 excerpts; 25 verdicts |
| GPT-5.2 | OpenAI | Judge | None | — |
| GPT-5.6 | OpenAI | Judge (web-search retrieval experiment) | None | — |

`ai-only/` reuses the generated ideas above and adds no new model outputs. The only judge
outputs released are the 30 final verdicts in `expert-annotations/judge_errors.csv`; no
prompt/response logs, reasoning traces, or retrieval transcripts are released.

**Authorization status:** none required; evaluation use.

### Collection method

All model outputs were collected through the providers' APIs (Anthropic API, OpenAI API),
using the prompt templates in
[benchmark_data/templates/](../src/novelty_eval/benchmark_data/templates/).

* **Generated ideas** — the generator receives only an ICLR primary area and is asked to propose
  a novel research idea as an abstract, with no scaffold, literature access, or tools. Each
  generated idea is one model response.
* **Edited abstracts** — each human abstract is passed once through a prompt that removes
  results, baselines, and evaluation datasets and otherwise keeps the original wording.
* **Plan-form ideas** (`_plan` files) — each abstract-form idea, human or generated, is
  rewritten into a two-field *purpose* / *mechanism* plan.
* **Reviewer excerpts** — the model is given the Strengths and Weaknesses sections of each
  official review and copies out, verbatim, the sentences that judge the paper's novelty. The
  model selects the excerpts; the text is the reviewer's.
* **Labels** — `POSITIVE` / `NEGATIVE` are assigned by fixed rules over reviewer scores,
  decisions, and the selected excerpts; generated ideas are `NEGATIVE` by construction. No
  model assigns labels directly.
* **Judge verdicts** — `predicted` in `judge_errors.csv` is the judge's final verdict only.

### Which fields are AI-generated

| Field | Origin | Model |
|---|---|---|
| `idea` / `ideas`, for generated ideas (`rating: N/A`) | Model-generated | Claude Sonnet 4.5, the model named in `backbone-*/`, or `metadata.generator` in `ai-only/` |
| `idea` / `ideas`, for human ideas (abstract files) | Human abstract, edited by a model | Claude Opus 4.6 |
| `idea` / `ideas` in `_plan` files | Model rewrite of the abstract-form idea | Claude Opus 4.6 |
| `metadata.positive_signals`, `metadata.negative_signals` | Verbatim reviewer text, selected by a model | Claude Opus 4.6 (selection only) |
| `shown_A_idea`, `shown_B_idea` in `judge_errors.csv` | Copied from the files above | As above |
| `predicted` in `judge_errors.csv` | Model verdict | The model in `judge_model` |
| `annotation`, `annotation_reasoning` in `judge_errors.csv` | Human expert | — |
| `metadata.title` of human ideas, `context`, `metadata.rating`, `metadata.contribution` | OpenReview | — |
| `metadata.title` of generated ideas, `label`, `metadata.type`, `expected_winners` | Assigned by our pipeline | — |