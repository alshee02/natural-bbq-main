# Paired Prompt Human Evaluation Guideline

Version 1.1  
Last updated: 2026-08-03

## 1. Purpose

This study evaluates pairs of texts that are intended to describe the same
situation and ask the same question. For every row, independently assess:

1. the answer supported by Text A and Text B;
2. whether their answer-relevant meanings are equivalent;
3. whether the amount and role of evidence are preserved;
4. which semantic error types, if any, occur;
5. the linguistic naturalness of each text; and
6. which text is more natural as a user prompt.

The labels and provenance of Text A and Text B are intentionally withheld. Do
not attempt to identify how either text was produced. Judge only the text shown
in the assigned annotation file.

Some items contain stereotypes, discriminatory assumptions, or other
potentially uncomfortable social content inherited from the source benchmark.
These statements are presented only for evaluation. Annotators may pause or
take a break whenever needed.

## 2. Independent Annotation Protocol

- Complete all judgments independently. Do not discuss audit items or compare
  labels with another annotator before all three files have been submitted.
- Do not search the web or consult outside sources. Use only Text A, Text B, and
  their shared answer options.
- Do not use demographic stereotypes or common assumptions as evidence. Select
  an Unknown, Not enough information, or Cannot be determined option whenever
  the text does not provide enough evidence for a specific answer.
- Judge every row in its provided order. Do not sort, filter, delete, reorder,
  or modify the identifier, text, or answer-option columns.
- Enter values only in the unlocked response columns of the XLSX workbook.
- Every error field is initialized to `0`. Leave it at `0` when the error is
  absent and change it to `1` only when the error is present. Do not blank an
  error field.
- Work in manageable sessions. A recommended session contains 100--150 rows,
  followed by a break. Stop a session if attention or consistency declines.

The formatting of a text is part of its linguistic presentation. Labels such
as `Context:` and `Question:` may affect naturalness, but their presence alone
is not a semantic error and does not make a text malformed.

## 3. Required Order of Judgment

For each row, use the following order.

1. Read the three shared answer options.
2. Read Text A and record `answer_a` before evaluating Text B.
3. Rate `naturalness_a_1_to_5` based only on Text A.
4. Read Text B and record `answer_b`.
5. Rate `naturalness_b_1_to_5` based only on Text B.
6. Compare the texts and assign `semantic_validity` and
   `evidence_state_preserved`.
7. Review all error columns and change each applicable error from `0` to `1`.
8. Record `preferred_version` based on linguistic naturalness.
9. Add a note when required by Section 11.

This order reduces the chance that one text changes the answer or naturalness
judgment made for the other.

## 4. Answer Judgments

### `answer_a` and `answer_b`

Allowed values:

| Value | Meaning |
|---|---|
| `0` | The first displayed answer option |
| `1` | The second displayed answer option |
| `2` | The third displayed answer option |
| `cannot_judge` | The text is too malformed, truncated, or uninterpretable to determine what it asks |

Choose the best answer supported by each text independently.

`cannot_judge` is not an Unknown answer. If an answer option says Unknown, Not
answerable, Not enough information, Cannot be determined, or equivalent, enter
that option's number. Use `cannot_judge` only when the text itself cannot be
reliably interpreted or evaluated.

Minor grammatical awkwardness is not sufficient for `cannot_judge` when the
situation and question remain understandable.

## 5. Semantic Validity

### `semantic_validity`

Allowed values: `valid`, `invalid`, `uncertain`.

**Valid** means that Text A and Text B preserve all answer-relevant meaning:

- the same people, groups, identities, and attributes;
- the same participant roles and relationships;
- the same events, actions, and evidence;
- the same question target and predicate;
- the same negation, comparison, and quantifier structure;
- the same degree of answerability or uncertainty; and
- the same correspondence between the text and the shared answer options.

Changes in wording, tone, sentence structure, punctuation, contractions, or
discourse markers are allowed when they do not change these properties.

**Invalid** means that at least one answer-relevant semantic property differs.
A pair can be invalid even when `answer_a` and `answer_b` happen to be equal.
For example, added evidence, a changed participant role, or stronger wording
may leave the final answer unchanged while still altering the item.

**Uncertain** should be rare. Use it only when careful reading does not support
a defensible valid/invalid decision. Explain the uncertainty in
`annotator_notes`.

## 6. Evidence-State Preservation

### `evidence_state_preserved`

Allowed values: `yes`, `no`, `uncertain`.

Select **yes** when both texts have the same answerability state:

- both texts lack sufficient evidence to select a specific participant; or
- both texts provide sufficient evidence to resolve their respective questions.

The supported answer itself does not need to be the same for
`evidence_state_preserved = yes`. Answer identity is assessed separately
through `answer_a`, `answer_b`, and `semantic_validity`.

Select **no** when one text changes whether the question can be resolved. Common
cases include:

- one text permits only an Unknown-type answer while the other supports a
  specific answer;
- one text drops a fact needed to resolve the question;
- one text adds a fact that resolves a previously unresolved question; or
- evidence becomes weaker, stronger, or less specific enough to change
  answerability.

Select **uncertain** only when the evidence-state comparison cannot be resolved
after careful reading. Add an explanation in `annotator_notes`.

Evidence-state preservation and answer equality are related but distinct. Two
texts can both be answerable while supporting different answers, or can retain
the same answer despite a meaningful evidence change.

## 7. Error Tags

Every error field is prefilled with `0`. Change it to `1` when the error is
present and retain `0` when it is absent. Multiple errors may be marked for the
same pair. A remaining `0` means that the error was reviewed and judged absent;
it must not be used to indicate an unfinished row.

The error tags are pair-level comparative judgments. Do not infer the provenance
or priority of either text. Evidence omission and addition are directional from
Text A to Text B: record whether Text B omits information from or adds
information to Text A. This direction does not indicate which text is a source
or a rewrite. When the direction of another error would help interpretation,
identify the affected side as Text A or Text B in `annotator_notes`.

| Column | Enter `1` when... |
|---|---|
| `error_answer_differs` | The best supported numbered answers clearly differ between Text A and Text B. If either answer is `cannot_judge`, use `0` here and mark `error_malformed`. |
| `error_evidence_omission` | Text B lacks an answer-relevant fact, event, relation, or qualifier that is present in Text A. |
| `error_evidence_addition` | Text B contains an answer-relevant fact, event, relation, or qualifier that is absent from Text A. |
| `error_evidence_alteration` | Corresponding evidence appears in both texts, but its strength, specificity, scope, temporal or causal relation, modality, or answer-relevant interpretation changes. Mere sentence reordering, restructuring, or stylistic emphasis is not an error unless it changes meaning or answer interpretation. |
| `error_participant_or_role_change` | A person, group, identity term, attribute, relationship, role, action, or answer-option correspondence changes. This includes swapping participants. |
| `error_coreference_change` | A pronoun or referring expression changes its referent, becomes ambiguous, or resolves an ambiguity differently. |
| `error_question_target_change` | The question asks about a different person, group, property, event, action, or predicate. |
| `error_negation_comparison_or_quantifier` | Negation, polarity, comparison, degree, frequency, modality, or a quantifier such as all, some, more, less, always, or never changes. |
| `error_ambiguity_state_change` | One text supports a specific answer while the other remains unresolved, or a previously answerable item becomes unresolved. |
| `error_answer_cue_or_leading` | One text reveals, strongly implies, favors, rules out, or leads toward an answer beyond the evidence in the other text. |
| `error_new_framing_or_judgment` | One text introduces a new purpose, perspective, personal experience, emotion, urgency, moral judgment, stereotype, opinion, speculation, or probability claim. |
| `error_malformed` | At least one text is truncated, corrupted, internally incoherent, or uninterpretable enough to prevent reliable evaluation. Ordinary awkwardness does not qualify. |
| `error_other` | A semantic problem is present but is not captured by the listed tags. Describe it in `annotator_notes`. |

### Distinguishing related evidence errors

- Use `error_evidence_omission` when information present in Text A is missing
  from Text B.
- Use `error_evidence_addition` when information appears in Text B without
  support in Text A.
- Use `error_evidence_alteration` when corresponding information exists in both
  texts but its force, specificity, scope, temporal or causal relation,
  modality, or interpretation changes.
- Do not mark ordinary sentence restructuring or proposition reordering when
  it leaves the meaning and answer interpretation unchanged.
- Mark more than one when necessary. For example, replacing one fact with a new
  fact can involve both omission and addition.
- Add `error_ambiguity_state_change` whenever an evidence difference changes
  whether a specific answer can be selected.

Generic conversational markers such as "Could you tell me...," "I'm
wondering...," or "Quick question..." are not errors by themselves, provided
that they do not introduce a factual personal experience, new motivation,
emotional stance, urgency, judgment, or task purpose.

## 8. Naturalness Ratings

### `naturalness_a_1_to_5` and `naturalness_b_1_to_5`

Rate each text independently as a plausible user prompt. Judge fluency,
coherence, phrasing, conversational plausibility, and formatting. Do not reward
or penalize a text because of its answer, demographic content, or perceived
condition.

Rate only the displayed Text A or Text B prompt body. Do not include the shared
answer options, spreadsheet layout, identifiers, or annotation interface in the
naturalness judgment.

| Score | Definition |
|---|---|
| `1` | Highly unnatural or severely awkward; difficult to follow, though not necessarily uninterpretable |
| `2` | Noticeably unnatural; multiple awkward constructions or strongly mechanical phrasing |
| `3` | Understandable but somewhat awkward, templated, or unlike an ordinary user request |
| `4` | Natural and fluent; plausible as a real user prompt, with only minor issues |
| `5` | Very natural and fluent; reads readily as an authentic, well-formed user prompt |

Content correctness and naturalness are separate. A semantically invalid text
can still sound natural, and a semantically valid text can sound awkward.

## 9. Pairwise Preference

### `preferred_version`

Allowed values: `A`, `B`, `same`, `cannot_judge`.

Choose the version that is more natural as a user prompt.

- Enter `A` when Text A is clearly more natural.
- Enter `B` when Text B is clearly more natural.
- Enter `same` when neither has a meaningful naturalness advantage.
- Enter `cannot_judge` only when malformed or uninterpretable text prevents a
  reliable comparison.

Base this decision on linguistic naturalness rather than semantic validity. The
preferred text need not be the semantically better-preserved text.

## 10. Consistency Rules

Before completing a row, check the following.

| Situation | Required coding |
|---|---|
| Both answers are numbered and differ | `error_answer_differs = 1` |
| Either answer is `cannot_judge` because a text is uninterpretable | `error_malformed = 1` |
| `semantic_validity = valid` | Both answers should normally match; `evidence_state_preserved = yes`; all semantic error tags should be `0` |
| `semantic_validity = invalid` or `uncertain` | At least one error tag must be `1` |
| `evidence_state_preserved = no` | `error_ambiguity_state_change`, an evidence error, or both should normally be `1` |
| `semantic_validity = uncertain` or `evidence_state_preserved = uncertain` | A note is required |
| `error_other = 1` | A note describing the error is required |
| `preferred_version = A` | The A naturalness score should normally be at least as high as the B score |
| `preferred_version = B` | The B naturalness score should normally be at least as high as the A score |
| `preferred_version = same` | Equal scores are typical, but adjacent scores are permitted when the practical difference is negligible |

These rules identify likely input mistakes rather than replacing annotator
judgment. If a justified exception occurs, explain it briefly in
`annotator_notes`.

## 11. Annotator Notes

### `annotator_notes`

Notes are optional for clear valid cases. A concise note is required when:

- either answer is `cannot_judge`;
- `semantic_validity` or `evidence_state_preserved` is `uncertain`;
- `error_other = 1`;
- the direction of an evidence alteration or another non-directional error
  would otherwise be unclear; or
- the coding violates one of the usual consistency rules in Section 10.

Use `A` and `B` when referring to the displayed texts. Do not speculate about
their provenance.

## 12. Synthetic Examples

The examples below are invented for training and do not occur in the audit.

### Example 1: Valid surface reformulation

Text A states that Casey and Morgan were both at a station but gives no evidence
about who lost a ticket. Text B asks the same question in a single
conversational sentence and adds no evidence.

- Both answers: the numbered Unknown-type option
- `semantic_validity = valid`
- `evidence_state_preserved = yes`
- All error tags: `0`
- Naturalness and preference: judge the wording independently

### Example 2: Evidence omission changes answerability

Text A states that Riley submitted the report and asks who submitted it. Text B
mentions Riley and another person but omits who submitted the report.

- `error_answer_differs = 1`
- `error_evidence_omission = 1`
- `error_ambiguity_state_change = 1`
- `evidence_state_preserved = no`
- `semantic_validity = invalid`

### Example 3: New evidence resolves an unresolved question

Text A mentions Jordan and Taylor near a payment desk without saying who paid.
Text B adds that Jordan handed the cashier a card.

- `error_answer_differs = 1` when the added fact changes the selected option
- `error_evidence_addition = 1`
- `error_ambiguity_state_change = 1`
- `evidence_state_preserved = no`
- `semantic_validity = invalid`

### Example 4: Negation changes the question

Text A asks who arrived late. Text B asks who did not arrive late.

- `error_negation_comparison_or_quantifier = 1`
- `error_answer_differs = 1` when the supported options differ
- `semantic_validity = invalid`
- `evidence_state_preserved` may still be `yes` if both questions remain equally
  answerable; semantic equivalence and answerability are separate.

### Example 5: Naturalness differs without semantic error

Both texts preserve every fact and support the same answer, but Text A uses
mechanical field labels while Text B is fluent and conversational.

- `semantic_validity = valid`
- `evidence_state_preserved = yes`
- All error tags: `0`
- Naturalness scores may differ
- Preference may favor the more natural text

### Example 6: Same answer but altered evidence

Text A says that Lee arrived first and asks who arrived first. Text B says that
Lee arrived much earlier than everyone else and asks the same question.

Both texts support Lee as the answer, but Text B strengthens the evidence and
adds a comparison not present in Text A.

- Both numbered answers may be the same
- `semantic_validity = invalid`
- `evidence_state_preserved = yes`
- `error_evidence_alteration = 1`
- `error_negation_comparison_or_quantifier = 1`

## 13. File Completion and Submission

Before submission:

1. Confirm that all 2,640 rows retain their original `annotation_order`,
   `annotator_id`, and `audit_id`.
2. Confirm that `answer_a`, `answer_b`, `semantic_validity`,
   `evidence_state_preserved`, both naturalness fields, and
   `preferred_version` are complete for every row.
3. Confirm that all entries use the exact allowed values and capitalization.
4. Confirm that all error columns contain only `0` or `1`.
5. Confirm that every `invalid` or `uncertain` row has at least one error field
   set to `1`, and that every row with `error_other = 1` has a note.
6. Save the completed XLSX file without modifying text or answer options.
7. Preserve the independent completed file before any adjudication or error
   correction involving another annotator.

After all independent files are frozen, disagreements may be reviewed in a
separate adjudication stage. Do not overwrite the original independent labels.

## 14. Quick Reference

| Field group | Allowed values |
|---|---|
| `answer_a`, `answer_b` | `0`, `1`, `2`, `cannot_judge` |
| `semantic_validity` | `valid`, `invalid`, `uncertain` |
| `evidence_state_preserved` | `yes`, `no`, `uncertain` |
| Every `error_*` field | `0`, `1` |
| Naturalness fields | `1`, `2`, `3`, `4`, `5` |
| `preferred_version` | `A`, `B`, `same`, `cannot_judge` |
