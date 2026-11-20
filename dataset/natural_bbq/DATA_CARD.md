# Natural-BBQ — Data Card

> **Warning:** This dataset contains examples of stereotypes and biases inherited
> from BBQ. It is intended to *measure* social bias, not to endorse or amplify it.

## Summary

Natural-BBQ is a controlled naturalization of the Bias Benchmark for Question
Answering (BBQ). Each item pairs the **Original** BBQ context/question with a
**Natural** rewrite whose surface form is moved toward the empirical style
distribution of WildChat user prompts, while the answer options, gold label,
bias annotations, and ambiguous/disambiguated structure are preserved unchanged.
The resource enables testing whether social bias measurements remain stable under
meaning-preserving linguistic variation.

## Composition

| Property | Value |
| --- | --- |
| Paired items | 58,044 |
| Coverage of BBQ construction set | 99.5% (58,044 / 58,360) |
| Categories | 11 (Age, Disability, Gender, Nationality, Physical appearance, Race/ethnicity, Race×SES, Race×gender, Religion, SES, Sexual orientation) |
| Context conditions | ambiguous (29,105) / disambiguated (28,939) |
| Human-audited pairs | 2,640 (3 annotators; separate release) |

Each pair differs **only** in the prompt body (Original vs. Natural); the answer
choices, their order, the gold label, and the digit-only response instruction are
identical across the two conditions.

## Schema

One row per BBQ item (`.jsonl`, one JSON object per line):

| Field | Type | Description |
| --- | --- | --- |
| `bbq_item_id` | str | Unique item identifier |
| `example_id` | int | Original BBQ `example_id` |
| `category` | str | BBQ social-bias category |
| `context_condition` | str | `ambig` or `disambig` |
| `question_polarity` | str | `neg` or `nonneg` |
| `question_index` | str | Original BBQ `question_index` |
| `original_context` | str | Original BBQ context |
| `original_question` | str | Original BBQ question |
| `natural_question` | str | Natural-BBQ rewrite of the concatenated context + question |
| `ans0`, `ans1`, `ans2` | str | Answer options (unchanged from BBQ, same order) |
| `label` | int | Gold answer index (0/1/2) |
| `target_cluster_id` | int | WildChat style cluster sampled to condition the rewrite (never rendered into the prompt) |
| `rewrite_model` | str | Model used for the rewrite (`gpt-5.4-mini`) |
| `fallback_used` | bool | Whether the item was produced by the second-round fallback |
| `sbert_similarity` | float | SBERT content-preservation score between Original and Natural bodies |
| `delta_style` | float | Matched-body held-out style shift toward WildChat |
| `answer_info` | object | BBQ per-option annotation (label word + group tag for `ans0`/`ans1`/`ans2`) |
| `additional_metadata` | object | BBQ metadata: `stereotyped_groups`, `subcategory`, `source`, `version` |

> Exact field names follow the pipeline output; the release is regenerated with
> `export_release.py`, which selects these public fields and drops internal
> pipeline diagnostics and any WildChat-derived text.

## Construction

Built with the pipeline in this repository: WildChat style modeling
(StyleDistance + six-cluster k-means), best-of-seven reference-conditioned
rewriting with GPT-5.4-mini, structural + SBERT filtering, matched-body
StyleDistance selection, and a held-out style-validation split. Full details are
in the paper and `README.md`.

## Validation

- **Automatic:** SBERT similarity (mean 0.982), held-out matched-body style shift.
- **Human:** three-annotator audit of 2,640 pairs — majority semantic validity
  99.5% (representative). See `dataset/bbq/ANNOTATION_GUIDELINE.md`.

## Intended Uses

Measuring the *linguistic invariance* of social-bias benchmarks: comparing model
accuracy, signed BBQ bias, response transitions, and rank stability across the
Original and Natural conditions. **Not** intended as a replacement for BBQ, a
simulation of deployment, or a claim about the inherent properties of social
groups.

## Provenance and Licensing

- Derived from **BBQ** (CC BY 4.0). Style references from the de-identified
  **WildChat-4.8M** release (ODC-By); raw WildChat prompts, identifiers, and user
  metadata are **not** redistributed.
- Natural-BBQ is released under **CC BY 4.0**, with attribution to the original
  BBQ and WildChat authors. See the repository `NOTICE`.

## Files

- `natural_bbq.jsonl` — the 58,044 paired items (one JSON object per line).
- `DATA_CARD.md` — this file.
- `export_release.py` — regenerates the public JSONL from the pipeline output parquet.
