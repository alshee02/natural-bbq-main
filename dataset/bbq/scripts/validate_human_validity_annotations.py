#!/usr/bin/env python3
"""Validate completed Natural-BBQ human-validity annotation files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "data/processed/human_validity_audit_2640"

IDENTITY_COLUMNS = ["annotation_order", "annotator_id", "audit_id"]
STIMULUS_COLUMNS = ["text_a", "text_b", "answer_0", "answer_1", "answer_2"]
REQUIRED_RESPONSE_COLUMNS = [
    "answer_a",
    "answer_b",
    "semantic_validity",
    "evidence_state_preserved",
    "naturalness_a_1_to_5",
    "naturalness_b_1_to_5",
    "preferred_version",
]
ERROR_COLUMNS = [
    "error_answer_differs",
    "error_evidence_omission",
    "error_evidence_addition",
    "error_evidence_alteration",
    "error_participant_or_role_change",
    "error_coreference_change",
    "error_question_target_change",
    "error_negation_comparison_or_quantifier",
    "error_ambiguity_state_change",
    "error_answer_cue_or_leading",
    "error_new_framing_or_judgment",
    "error_malformed",
    "error_other",
]
ALL_COLUMNS = (
    IDENTITY_COLUMNS
    + STIMULUS_COLUMNS
    + REQUIRED_RESPONSE_COLUMNS[:4]
    + ERROR_COLUMNS
    + REQUIRED_RESPONSE_COLUMNS[4:]
    + ["annotator_notes"]
)

ALLOWED_VALUES = {
    "answer_a": {"0", "1", "2", "cannot_judge"},
    "answer_b": {"0", "1", "2", "cannot_judge"},
    "semantic_validity": {"valid", "invalid", "uncertain"},
    "evidence_state_preserved": {"yes", "no", "uncertain"},
    "naturalness_a_1_to_5": {"1", "2", "3", "4", "5"},
    "naturalness_b_1_to_5": {"1", "2", "3", "4", "5"},
    "preferred_version": {"A", "B", "same", "cannot_judge"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate completed human-validity CSV or XLSX files."
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    return parser.parse_args()


def read_annotations(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name="Annotation", dtype=object)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=object)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalized(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def validate_file(path: Path, template_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    frame = read_annotations(path)

    if list(frame.columns) != ALL_COLUMNS:
        missing = [column for column in ALL_COLUMNS if column not in frame.columns]
        extra = [column for column in frame.columns if column not in ALL_COLUMNS]
        errors.append(f"column mismatch; missing={missing}, extra={extra}")
        return errors, warnings
    if len(frame) != 2_640:
        errors.append(f"expected 2,640 rows, found {len(frame):,}")
    if frame["audit_id"].nunique(dropna=False) != len(frame):
        errors.append("audit_id values are not unique")

    annotator_ids = frame["annotator_id"].dropna().astype(str).unique()
    if len(annotator_ids) != 1:
        errors.append(f"expected one annotator_id, found {annotator_ids.tolist()}")
        annotator_id = None
    else:
        annotator_id = annotator_ids[0]

    if annotator_id:
        template_path = template_dir / f"human_validity_{annotator_id}.csv"
        if template_path.exists() and path.resolve() != template_path.resolve():
            template = pd.read_csv(template_path, dtype=object)
            left = frame[IDENTITY_COLUMNS + STIMULUS_COLUMNS].map(normalized)
            right = template[IDENTITY_COLUMNS + STIMULUS_COLUMNS].map(normalized)
            if not left.equals(right):
                errors.append("identifier, text, answer-option, or row order differs from template")
        elif not template_path.exists():
            warnings.append(f"template not found: {template_path}")

    for column in REQUIRED_RESPONSE_COLUMNS:
        values = frame[column].map(normalized)
        blank_rows = frame.loc[values.eq(""), "annotation_order"].tolist()
        if blank_rows:
            errors.append(
                f"{column}: {len(blank_rows):,} blank values; first rows={blank_rows[:5]}"
            )
        invalid = ~values.isin(ALLOWED_VALUES[column]) & ~values.eq("")
        if invalid.any():
            examples = sorted(values[invalid].unique().tolist())[:5]
            errors.append(f"{column}: invalid values {examples}")

    normalized_errors = frame[ERROR_COLUMNS].map(normalized)
    for column in ERROR_COLUMNS:
        invalid = ~normalized_errors[column].isin({"0", "1"})
        if invalid.any():
            examples = sorted(normalized_errors.loc[invalid, column].unique().tolist())[:5]
            errors.append(f"{column}: values must be 0 or 1; found {examples}")
    error_sum = normalized_errors.eq("1").sum(axis=1)

    semantic = frame["semantic_validity"].map(normalized)
    strict_error_rows = semantic.isin({"invalid", "uncertain"}) & error_sum.eq(0)
    if strict_error_rows.any():
        rows = frame.loc[strict_error_rows, "annotation_order"].tolist()
        errors.append(
            "invalid/uncertain rows require at least one error tag; "
            f"count={len(rows):,}, first rows={rows[:5]}"
        )

    notes = frame["annotator_notes"].map(normalized)
    missing_other_notes = normalized_errors["error_other"].eq("1") & notes.eq("")
    if missing_other_notes.any():
        rows = frame.loc[missing_other_notes, "annotation_order"].tolist()
        errors.append(
            f"error_other=1 requires a note; count={len(rows):,}, first rows={rows[:5]}"
        )

    answer_a = frame["answer_a"].map(normalized)
    answer_b = frame["answer_b"].map(normalized)
    malformed = normalized_errors["error_malformed"].eq("1")
    cannot_judge = answer_a.eq("cannot_judge") | answer_b.eq("cannot_judge")
    if (cannot_judge & ~malformed).any():
        rows = frame.loc[cannot_judge & ~malformed, "annotation_order"].tolist()
        errors.append(
            f"cannot_judge requires error_malformed=1; count={len(rows):,}, first rows={rows[:5]}"
        )

    both_numbered = answer_a.isin({"0", "1", "2"}) & answer_b.isin({"0", "1", "2"})
    answers_differ = both_numbered & answer_a.ne(answer_b)
    differs_tag = normalized_errors["error_answer_differs"].eq("1")
    if (answers_differ & ~differs_tag).any():
        rows = frame.loc[answers_differ & ~differs_tag, "annotation_order"].tolist()
        errors.append(
            f"different numbered answers require error_answer_differs=1; count={len(rows):,}, first rows={rows[:5]}"
        )
    if (both_numbered & answer_a.eq(answer_b) & differs_tag).any():
        rows = frame.loc[
            both_numbered & answer_a.eq(answer_b) & differs_tag, "annotation_order"
        ].tolist()
        errors.append(
            f"equal numbered answers require error_answer_differs=0; count={len(rows):,}, first rows={rows[:5]}"
        )

    valid = semantic.eq("valid")
    evidence = frame["evidence_state_preserved"].map(normalized)
    invalid_valid_rows = valid & (
        answer_a.ne(answer_b) | evidence.ne("yes") | error_sum.ne(0)
    )
    if invalid_valid_rows.any():
        rows = frame.loc[invalid_valid_rows, "annotation_order"].tolist()
        errors.append(
            "valid rows require matching answers, preserved evidence, and zero errors; "
            f"count={len(rows):,}, first rows={rows[:5]}"
        )

    naturalness_a = pd.to_numeric(
        frame["naturalness_a_1_to_5"].map(normalized), errors="coerce"
    )
    naturalness_b = pd.to_numeric(
        frame["naturalness_b_1_to_5"].map(normalized), errors="coerce"
    )
    preference = frame["preferred_version"].map(normalized)
    preference_mismatch = (
        (preference.eq("A") & naturalness_a.lt(naturalness_b))
        | (preference.eq("B") & naturalness_b.lt(naturalness_a))
        | (preference.eq("same") & (naturalness_a - naturalness_b).abs().gt(1))
    )
    if preference_mismatch.any():
        rows = frame.loc[preference_mismatch, "annotation_order"].tolist()
        warnings.append(
            "preference and naturalness scores may be inconsistent; "
            f"count={len(rows):,}, first rows={rows[:5]}"
        )

    return errors, warnings


def main() -> None:
    args = parse_args()
    any_errors = False
    for path in args.files:
        errors, warnings = validate_file(path, args.template_dir)
        print(f"\n{path}")
        for warning in warnings:
            print(f"  WARNING: {warning}")
        for error in errors:
            print(f"  ERROR: {error}")
        if not errors:
            print("  PASS")
        any_errors = any_errors or bool(errors)
    if any_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
