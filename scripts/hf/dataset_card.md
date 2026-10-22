---
pretty_name: Novelty Judge Bench
license: odc-by
language:
- en
task_categories:
- text-classification
tags:
- novelty
- peer-review
- llm-as-a-judge
- ai-generated-ideas
- research-ideation
- benchmark
size_categories:
- n<1K
annotations_creators:
- expert-generated
- machine-generated
source_datasets:
- original
configs:
{{CONFIGS_YAML}}
---

# Old Ideas, Novel Problems:<br> The Instability of LLM-Based Novelty Evaluation

[![Arxiv](https://img.shields.io/badge/Arxiv-2610.02022-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.02022)
[![Code](https://img.shields.io/badge/Code-GitHub-black?style=flat-square&logo=github)](https://github.com/allenai/novelty-eval)
<!-- [![Project Page](https://img.shields.io/badge/Project%20Page-Here-green?style=flat-square&logo=github)](https://noy-sternlicht.github.io/Novelty-Evaluation-Web/) -->

<p align="center">
  <img src="assets/fig1-oct14-media.png" alt="Rewording the judge prompt changes GPT-5.4's and Opus 4.6's accuracy" width="100%" />
</p>

Benchmark instances for the paper
[**Old Ideas, Novel Problems: The Instability of LLM-Based Novelty Evaluation**](https://arxiv.org/abs/2610.02022).
Code for running judges on this benchmark and reproducing the paper's experiments is at
[allenai/novelty-eval](https://github.com/allenai/novelty-eval).

AI-based ideation systems are often evaluated on the novelty of their ideas, and that judgment is increasingly delegated to large language models. Such judges are typically built ad hoc and validated, if at all, on human-authored papers rather than on the generated ideas they are meant to score. So, how do novelty judges perform?

Not well. We present a systematic controlled study of novelty evaluation design choices. We build an evaluation set automatically, mining OpenReview for passages where reviewers explicitly affirm or dispute a paper's originality and keeping only submissions with unanimous agreement at the extremes of their research area; we pair these with ideas from a vanilla LLM generator. Across six judges, we find that seemingly small prompt design choices have large consequences; e.g., simply telling the judge that reviewers found one idea novel and the other not can change its verdict on more than half of the identical idea pairs it is shown, shifting pairwise accuracy by over 50 points and occasionally pushing it below chance. The same change helps one judge and hurts another. Retrieval and larger reasoning budgets help little, and two purpose-built novelty evaluators are outperformed by our cheapest prompted baseline. These results raise questions about reported novelty gains of automated ideation systems, and call for robust novelty evaluation methods.

## Setups

<p align="center">
  <img src="assets/data-creation-overview.png" alt="Novelty evaluation data creation pipeline" width="100%" />
</p>

The benchmark consists of two data setups: **Human-Only** and **Human+Generated**. Both setups
consist of two separate idea "pools": a **high-novelty pool `D_+`** and a **lower-novelty pool
`D_-`**, where ideas from `D_+` can be assumed to be more novel than those in `D_-`. The two
setups share the same `D_+` and differ in where `D_-` comes from.

| Setup | `D_+` (novel) | `D_-` (lower-novelty) |
|---|---|---|
| **Human-Only** (`human-only_*`) | 154 **validated-novel** | 145 **validated-lower-novelty** |
| **Human+Generated** (`human-plus-generated_*`) | 154 **validated-novel** | 154 **weakly-labeled-lower-novelty** |

Where:
* **validated-novel** (both setups) — ICLR 2026 submissions where a majority of reviewers explicitly praised the originality of the contribution itself and none disputed it; accepted, in the top rating decile of their primary area, with an average contribution score >= 3.0.
* **validated-lower-novelty** (Human-Only) — the mirror image: a majority of reviewers faulted the originality and none praised it; rejected, in the bottom rating decile of their primary area, with an average contribution score <= 2.0.
* **weakly-labeled-lower-novelty** (Human+Generated) — ideas from Claude Sonnet 4.5, prompted simply to propose a novel idea with no scaffold, literature access, or tools.

## Formats

Each setup is instantiated in two evaluation formats:

* **pointwise** — one idea per row, binary label: is this idea novel?
* **pairwise** — one `D_+` idea against one `D_-` idea; which of the two is more novel?

and in two idea formats:

* **abstract** (default) — the idea as a free-form abstract.
* **plan** (`*_plan` configs) — the same ideas normalized into a two-field
  *purpose* / *mechanism* research plan.

The `*_backbone-*` configs regenerate `D_-` with a different ideation backbone and are used
in the negatives-source experiment.

## Configs at a glance

| Config | Rows | Contents |
|---|---|---|
| `human-only_pointwise`, `human-only_pointwise_plan` | 299 | 154 novel, 145 lower-novelty |
| `human-only_pairwise`, `human-only_pairwise_plan` | 154 | Novel idea is `idea_a` in 84 pairs, `idea_b` in 70 |
| `human-plus-generated_pointwise` (default), `human-plus-generated_pointwise_plan` | 308 | 154 novel, 154 lower-novelty |
| `human-plus-generated_pairwise`, `human-plus-generated_pairwise_plan` | 154 | Novel idea is `idea_a` in 80 pairs, `idea_b` in 74 |
| `human-plus-generated_pointwise_backbone-{gpt-5,gpt-5.1,gpt-5.4,opus-4-5}` | 308 each | As above, with `D_-` from GPT-5, GPT-5.1, GPT-5.4, or Claude Opus 4.5 |
| `human-plus-generated_pairwise_backbone-{gpt-5,gpt-5.1,gpt-5.4,opus-4-5}` | 154 each | As above, with `D_-` from GPT-5, GPT-5.1, GPT-5.4, or Claude Opus 4.5 |

All configs have a single `test` split.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("allenai/novelty-judge-bench", "human-plus-generated_pointwise", split="test")
print(ds[0]["idea"], ds[0]["label"])
```

## Fields

### Pointwise configs

| Field | Type | Description |
|---|---|---|
| `id` | int32 | Instance id, unique within the config |
| `iclr_area` | string | The idea's ICLR primary area |
| `idea` | string | The idea to judge (abstract or plan form, depending on the config) |
| `label` | string | `POSITIVE` (`D_+`, novel) or `NEGATIVE` (`D_-`, lower-novelty) — the ground truth |
| `idea_source` | string | `human` (an ICLR submission) or `generated` (LLM ideator output) |
| `title` | string | Submission title, or a synthetic identifier for generated ideas |
| `rating` | float64 | Mean reviewer rating; `null` for generated ideas |
| `contribution` | float64 | Mean reviewer contribution score; `null` for generated ideas |
| `positive_signals` | list[string] | Reviewer excerpts praising originality |
| `negative_signals` | list[string] | Reviewer excerpts faulting originality |

### Pairwise configs

| Field | Type | Description |
|---|---|---|
| `id` | int32 | Instance id, unique within the config |
| `iclr_area` | string | The ICLR primary area shared by both ideas in the pair |
| `idea_a`, `idea_b` | string | The two ideas to compare |
| `expected_winner` | int8 | `0` if `idea_a` is the more novel one, `1` if `idea_b` |
| `label_a`, `label_b` | string | `POSITIVE` / `NEGATIVE` for each side |
| `idea_source_a`, `idea_source_b` | string | `human` or `generated` |
| `title_a`, `title_b` | string | Per-side titles |
| `rating_a`, `rating_b` | float64 | Per-side mean reviewer rating; `null` if generated |
| `contribution_a`, `contribution_b` | float64 | Per-side mean contribution score; `null` if generated |
| `positive_signals_a/_b` | list[string] | Per-side reviewer excerpts praising originality |
| `negative_signals_a/_b` | list[string] | Per-side reviewer excerpts faulting originality |

Idea order within each pair is randomized, so always picking the same position scores close
to chance (see the A/B split in [Configs at a glance](#configs-at-a-glance)).

## License

This dataset is licensed under ODC-BY 1.0. It is intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use).

## Authors

- [Noy Sternlicht](https://noy-sternlicht.github.io/) — Hebrew University of Jerusalem, Allen Institute for AI
- [Simra Shahid](https://sites.google.com/view/simra-shahid/home) — Microsoft
- [Peter Jansen](https://cognitiveai.org/) — Allen Institute for AI, University of Arizona
- [Daniel S. Weld](https://www.cs.washington.edu/people/faculty/weld/) — Allen Institute for AI, University of Washington
- [Pao Siangliulue](https://paoponder.com/) — Allen Institute for AI
- [Tom Hope](https://tomhoper.github.io/) — Hebrew University of Jerusalem, Allen Institute for AI

## Cite us!

```bibtex
@misc{sternlicht2026oldideasnovelproblems,
      title={Old Ideas, Novel Problems: The Instability of LLM-Based Novelty Evaluation}, 
      author={Noy Sternlicht and Simra Shahid and Peter Jansen and Daniel S. Weld and Pao Siangliulue and Tom Hope},
      year={2026},
      eprint={2610.02022},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.02022}, 
}
```
