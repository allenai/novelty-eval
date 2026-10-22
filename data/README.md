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