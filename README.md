# Natural-BBQ

Code and data for **"Stable Scores, Unstable Decisions: Testing the Linguistic Invariance of Social Bias Measurements with Natural-BBQ"**, *Findings of the Association for Computational Linguistics: EACL 2027*.

> **Warning:** This repository contains examples of stereotypes and biases inherited from BBQ.

## Overview

Natural-BBQ is a controlled naturalization of BBQ that transforms 58,044 bias benchmark items toward the empirical style distribution of WildChat user prompts, while preserving each item's answer options, gold label, and bias structure. The paired Original-Natural resource enables testing whether bias measurements remain stable under meaning-preserving linguistic variation.

This repository contains the Natural-BBQ construction pipeline, the paired Original/Natural evaluation code, the human-validation sampling and checking scripts, and the released dataset.

## Repository Structure

```
natural_BBQ/
├── requirements.txt                # Python dependencies
├── run_pipeline_bon.sh             # Main construction pipeline
├── rerun_matched_body_eval.sh      # Full evaluation pipeline
├── rerun_matched_body_open_models.sh
├── rerun_matched_body_closed_models.sh
│
├── prompts/                        # Rewrite prompt templates
│   ├── rewrite_system_prompt.txt
│   ├── rewrite_user_prompt_template.txt
│   └── rewrite_retry_suffix.txt
│
└── dataset/
    ├── natural_bbq/                # Released paired dataset
    │   ├── natural_bbq.jsonl       # 58,044 Original-Natural pairs
    │   ├── DATA_CARD.md            # Datasheet
    │   └── export_release.py       # Pipeline output -> public JSONL
    │
    ├── wildchat/                   # WildChat preprocessing & clustering
    │   ├── wildchat_preprocess.py
    │   ├── encode_styledistance.py
    │   ├── cluster_styledistance.py
    │   └── build_reference_bank.py
    │
    └── bbq/
        ├── ANNOTATION_GUIDELINE.md # Human-audit annotation guideline
        ├── data/                   # Original BBQ data (11 categories)
        ├── parsing/                # Deterministic output parser
        └── scripts/                # Construction and evaluation
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

```bash
export OPENAI_API_KEY="..."      # Required for rewriting (GPT-5.4-mini) and evaluation (GPT-4.1)
export ANTHROPIC_API_KEY="..."   # Required for evaluation (Claude models)
```

## Reproduction

The pipeline reproduces the final construction and matched-body evaluation used
in the paper. Each stage below writes to `data/processed/`.

### 1. WildChat Style Modeling

Run the WildChat stages in order:

1. `dataset/wildchat/wildchat_preprocess.py` — retain the first user turn, filter
   prompts, and create the conversation-disjoint reference/held-out split.
2. `dataset/wildchat/encode_styledistance.py` — encode prompts with StyleDistance.
3. `dataset/wildchat/cluster_styledistance.py` — fit the six-cluster style model
   and export assignments and cluster statistics.
4. `dataset/wildchat/build_reference_bank.py` — construct disjoint generation and
   selection banks.
5. `dataset/bbq/scripts/build_heldout_cluster_samples.py` — assign held-out
   prompts to clusters for final style validation.
6. `dataset/bbq/scripts/assign_bbq_clusters.py` — assign BBQ items to style
   clusters with category-by-context stratified quotas.

The cluster ID is used only to sample a WildChat reference; it is never rendered
into the rewrite-model prompt.

### 2. Natural-BBQ Construction

```bash
bash run_pipeline_bon.sh
```

Builds rewrite inputs, sends one sampled WildChat reference per candidate to
GPT-5.4-mini through the OpenAI Batch API (best-of-seven), applies structural and
SBERT filters, scores surviving candidates against the disjoint selection bank,
and retains the best candidate per BBQ item. A second Batch API round handles
items with no surviving initial candidate.

Primary output:

```text
data/processed/rewrite_output_bon/bbq_bon_vfull_selected.parquet
```

### 3. Matched-Body Style Selection

The paper excludes answer choices from both sides of the StyleDistance
comparison. Recompute style scores and candidate selection with:

```bash
GPU_ID=0 bash dataset/bbq/scripts/recompute_matched_body_style.sh
```

Final paired dataset:

```text
data/processed/rewrite_output_bon/bbq_bon_vfull_selected_matched_body.parquet
```

### 4. Evaluation

Both conditions share the same answer options and digit-only response
instruction and differ only in the prompt body:

| Condition | Prompt body |
| --- | --- |
| A: Original-MCQ | Original BBQ context and question |
| B: Natural-MCQ | Selected Natural-BBQ rewrite |

```bash
GPU_ID=0 bash rerun_matched_body_open_models.sh    # Open-weight models
bash rerun_matched_body_closed_models.sh           # Proprietary API models
bash rerun_matched_body_eval.sh finalize           # Paired metrics
```

Open-weight inference uses physical GPU 0 by default (override with `GPU_ID`).
Proprietary-model inference requires `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` and
uses the providers' Batch APIs. Shared implementations:

- `dataset/bbq/scripts/run_bias_eval.py` — open-weight inference.
- `dataset/bbq/scripts/run_bias_eval_api.py` — proprietary inference.
- `dataset/bbq/parsing/controlled_parser.py` — digit-only output parsing.
- `dataset/bbq/scripts/compute_bias_metrics.py` — paired A/B metrics.

Matched-body outputs are isolated under `data/processed/bias_eval_v2_matched_body/`.

## Human Validation

The three-annotator semantic-preservation audit (2,640 pairs) is sampled with
`dataset/bbq/scripts/sample_human_validity_audit.py` and scored with
`dataset/bbq/scripts/validate_human_validity_annotations.py`. The annotation
protocol is documented in `dataset/bbq/ANNOTATION_GUIDELINE.md`.

## Data

- **Natural-BBQ paired dataset** (the 58,044 Original-Natural pairs) is released
  in `dataset/natural_bbq/natural_bbq.jsonl`; fields, coverage, and provenance are
  documented in [`dataset/natural_bbq/DATA_CARD.md`](dataset/natural_bbq/DATA_CARD.md).
  It is released under CC BY 4.0. To regenerate it from the pipeline output, run
  `python dataset/natural_bbq/export_release.py`.
- **BBQ original data** is included in `dataset/bbq/data/` (CC BY 4.0).
- **WildChat raw prompts** are NOT included. The preprocessing script downloads
  from the de-identified `allenai/WildChat-4.8M` release via the Hugging Face
  `datasets` library; raw prompts, identifiers, and user metadata are not
  redistributed.

## Models

### Construction
- GPT-5.4-mini (via OpenAI Batch API)

### Evaluation
- **Open-weight:** Qwen3-4B-Instruct, Gemma-3-4B-IT, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
- **Proprietary:** GPT-4.1, GPT-4.1-mini, Claude Sonnet 5, Claude Haiku 4.5

## Citation

```bibtex
@inproceedings{park-etal-2027-stable,
    title = "Stable Scores, Unstable Decisions: Testing the Linguistic Invariance of Social Bias Measurements with Natural-{BBQ}",
    author = "Park, Minhee and
      Kim, Daehan and
      Park, Seokwon and
      Xia, Lin and
      Kim, Eunchan",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2027",
    year = "2027",
    publisher = "Association for Computational Linguistics",
}
```

## License

Code is released under the [MIT License](LICENSE). See [NOTICE](NOTICE) for third-party data attribution.
