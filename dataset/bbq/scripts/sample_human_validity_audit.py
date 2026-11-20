#!/usr/bin/env python3
"""Draw the stratified Natural-BBQ human-validity audit sample.

The design contains two non-overlapping components for every
category x context condition x style-cluster stratum:

* representative: four items from each within-stratum SBERT rank quartile;
* challenge: four items from the lowest within-stratum SBERT rank decile,
  prioritizing fallback-generated items when available.

The script writes one analysis master file and three blinded, independently
ordered annotation files containing the same item set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/processed/rewrite_output_bon/bbq_bon_vfull_selected_matched_body.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/human_validity_audit_2640"
GUIDELINE_PATH = PROJECT_ROOT / "dataset/bbq/ANNOTATION_GUIDELINE.md"

STRATUM_COLUMNS = ["category", "context_condition", "target_cluster_id"]
REQUIRED_COLUMNS = [
    "bbq_item_id",
    "source_file",
    "source_row_index",
    "question_polarity",
    "context_condition",
    "category",
    "label",
    "ans0",
    "ans1",
    "ans2",
    "original_context",
    "original_question",
    "natural_question",
    "target_cluster_id",
    "sbert_similarity",
    "candidate_stage",
    "fallback_used",
]

ANNOTATION_COLUMNS = [
    "answer_a",
    "answer_b",
    "semantic_validity",
    "evidence_state_preserved",
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
    "naturalness_a_1_to_5",
    "naturalness_b_1_to_5",
    "preferred_version",
    "annotator_notes",
]

ANNOTATION_SCHEMA = {
    "answer_a": ["0", "1", "2", "cannot_judge"],
    "answer_b": ["0", "1", "2", "cannot_judge"],
    "semantic_validity": ["valid", "invalid", "uncertain"],
    "evidence_state_preserved": ["yes", "no", "uncertain"],
    "error_columns": ["0", "1"],
    "naturalness_a_1_to_5": ["1", "2", "3", "4", "5"],
    "naturalness_b_1_to_5": ["1", "2", "3", "4", "5"],
    "preferred_version": ["A", "B", "same", "cannot_judge"],
    "notes": {
        "answer_fields": (
            "Select the numbered Unknown/Cannot answer option when that is the answer. "
            "Use cannot_judge only when the text itself cannot be interpreted or rated."
        ),
        "blinding": (
            "Condition labels and sampling metadata are withheld. The field-based "
            "format may still make the benchmark condition recognizable."
        ),
    },
}

ERROR_COLUMNS = [column for column in ANNOTATION_COLUMNS if column.startswith("error_")]
REQUIRED_RESPONSE_COLUMNS = [
    "answer_a",
    "answer_b",
    "semantic_validity",
    "evidence_state_preserved",
    "naturalness_a_1_to_5",
    "naturalness_b_1_to_5",
    "preferred_version",
]
WORKSHEET_PASSWORD = os.environ.get("WORKSHEET_PASSWORD", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the 2,640-pair Natural-BBQ human-validity audit."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotators", type=int, default=3)
    return parser.parse_args()


def _opaque_audit_id(bbq_item_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"natural-bbq-audit:{seed}:{bbq_item_id}".encode()).hexdigest()
    return f"HV-{digest[:12].upper()}"


def _sample_indices(indices: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    if len(indices) < n:
        raise ValueError(f"Cannot sample {n} rows from a pool of {len(indices)}")
    return rng.choice(indices, size=n, replace=False).tolist()


def assign_rank_cells(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Assign equal-frequency quartiles and deciles with randomized tie breaks."""
    ranked = group.copy()
    ranked["_tie_break"] = rng.random(len(ranked))
    ranked = ranked.sort_values(
        ["sbert_similarity", "_tie_break", "bbq_item_id"], kind="mergesort"
    )
    positions = np.arange(len(ranked))
    ranked["sbert_rank_percentile"] = (positions + 0.5) / len(ranked)
    ranked["sbert_quartile"] = (positions * 4 // len(ranked) + 1).astype(int)
    ranked["sbert_decile"] = (positions * 10 // len(ranked) + 1).astype(int)
    ranked["quartile_population_n"] = ranked.groupby("sbert_quartile")[
        "bbq_item_id"
    ].transform("size")
    return ranked.drop(columns="_tie_break")


def sample_stratum(
    group: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = assign_rank_cells(group, rng)

    low_decile = ranked[ranked["sbert_decile"] == 1]
    fallback = low_decile[low_decile["fallback_used"]].index.to_numpy()
    non_fallback = low_decile[~low_decile["fallback_used"]].index.to_numpy()

    challenge_indices = _sample_indices(fallback, min(4, len(fallback)), rng)
    remaining = 4 - len(challenge_indices)
    if remaining:
        challenge_indices.extend(_sample_indices(non_fallback, remaining, rng))

    challenge = ranked.loc[challenge_indices].copy()
    challenge["sample_component"] = "challenge"
    challenge["challenge_selection"] = np.where(
        challenge["fallback_used"], "fallback_priority", "low_decile_random"
    )

    representative_parts = []
    remaining_ranked = ranked.drop(index=challenge_indices)
    for quartile in range(1, 5):
        pool = remaining_ranked[
            remaining_ranked["sbert_quartile"] == quartile
        ].index.to_numpy()
        selected = _sample_indices(pool, 4, rng)
        representative_parts.append(remaining_ranked.loc[selected])

    representative = pd.concat(representative_parts).copy()
    representative["sample_component"] = "representative"
    representative["challenge_selection"] = ""
    return representative, challenge


def build_sample(source: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    representative_parts = []
    challenge_parts = []

    grouped = source.groupby(STRATUM_COLUMNS, sort=True, observed=True)
    for _, group in grouped:
        representative, challenge = sample_stratum(group, rng)
        representative_parts.append(representative)
        challenge_parts.append(challenge)

    representative = pd.concat(representative_parts, ignore_index=True)
    challenge = pd.concat(challenge_parts, ignore_index=True)
    sample = pd.concat([representative, challenge], ignore_index=True)

    population_n = len(source)
    sample["stratum_id"] = (
        sample["category"].astype(str)
        + "|"
        + sample["context_condition"].astype(str)
        + "|cluster="
        + sample["target_cluster_id"].astype(str)
    )
    sample["stratum_population_n"] = sample.groupby("stratum_id")[
        "bbq_item_id"
    ].transform(lambda _: 0)
    stratum_sizes = source.groupby(STRATUM_COLUMNS, observed=True).size()
    sample["stratum_population_n"] = [
        int(stratum_sizes.loc[(row.category, row.context_condition, row.target_cluster_id)])
        for row in sample.itertuples()
    ]
    sample["stratum_weight"] = sample["stratum_population_n"] / population_n

    is_representative = sample["sample_component"] == "representative"
    sample["representative_design_weight"] = np.nan
    sample["representative_normalized_weight"] = np.nan
    sample.loc[is_representative, "representative_design_weight"] = (
        sample.loc[is_representative, "quartile_population_n"] / 4
    )
    sample.loc[is_representative, "representative_normalized_weight"] = (
        sample.loc[is_representative, "representative_design_weight"] / population_n
    )

    sample["gold_answer_text"] = [
        row[int(label)] for row, label in zip(
            sample[["ans0", "ans1", "ans2"]].to_numpy(), sample["label"]
        )
    ]
    sample["audit_id"] = sample["bbq_item_id"].map(
        lambda item_id: _opaque_audit_id(str(item_id), seed)
    )
    sample["sampling_seed"] = seed

    order = {
        "representative": 0,
        "challenge": 1,
    }
    sample["_component_order"] = sample["sample_component"].map(order)
    sample = sample.sort_values(
        STRATUM_COLUMNS
        + ["_component_order", "sbert_quartile", "sbert_similarity", "bbq_item_id"]
    ).drop(columns="_component_order")
    return sample.reset_index(drop=True)


def validate_sample(source: pd.DataFrame, sample: pd.DataFrame) -> None:
    n_strata = source.groupby(STRATUM_COLUMNS, observed=True).ngroups
    if n_strata != 132:
        raise ValueError(f"Expected 132 populated strata, found {n_strata}")
    if len(sample) != 2_640 or sample["bbq_item_id"].nunique() != 2_640:
        raise ValueError("The audit must contain 2,640 unique pairs")

    component_counts = sample["sample_component"].value_counts().to_dict()
    if component_counts != {"representative": 2_112, "challenge": 528}:
        raise ValueError(f"Unexpected component counts: {component_counts}")

    per_stratum = sample.groupby(STRATUM_COLUMNS + ["sample_component"]).size()
    if not (per_stratum.xs("representative", level="sample_component") == 16).all():
        raise ValueError("Every stratum must contain 16 representative items")
    if not (per_stratum.xs("challenge", level="sample_component") == 4).all():
        raise ValueError("Every stratum must contain four challenge items")

    representative = sample[sample["sample_component"] == "representative"]
    quartile_counts = representative.groupby(STRATUM_COLUMNS + ["sbert_quartile"]).size()
    if len(quartile_counts) != 132 * 4 or not (quartile_counts == 4).all():
        raise ValueError("Every stratum must contain four representative items per quartile")

    challenge = sample[sample["sample_component"] == "challenge"]
    if not (challenge["sbert_decile"] == 1).all():
        raise ValueError("Every challenge item must come from the lowest stratum decile")

    if not np.isclose(
        representative["representative_normalized_weight"].sum(), 1.0
    ):
        raise ValueError("Representative analysis weights must sum to one")

    expected_category = sample["category"].value_counts()
    if not (expected_category == 240).all():
        raise ValueError("Each category must contribute 240 pairs")
    expected_condition = sample["context_condition"].value_counts()
    if expected_condition.to_dict() != {"ambig": 1_320, "disambig": 1_320}:
        raise ValueError(f"Unexpected condition counts: {expected_condition.to_dict()}")
    expected_cluster = sample["target_cluster_id"].value_counts()
    if not (expected_cluster == 440).all():
        raise ValueError("Each style cluster must contribute 440 pairs")


def annotation_frame(
    sample: pd.DataFrame,
    annotator: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one blinded annotation sheet and its private A/B mapping."""
    annotator_id = f"annotator_{annotator}"
    rng = np.random.default_rng(seed + annotator * 10_003)
    assigned = sample[
        [
            "audit_id",
            "stratum_id",
            "sample_component",
            "original_context",
            "original_question",
            "natural_question",
            "ans0",
            "ans1",
            "ans2",
        ]
    ].copy()
    assigned["original_text"] = (
        "Context: "
        + assigned["original_context"].astype(str)
        + "\nQuestion: "
        + assigned["original_question"].astype(str)
    )
    assigned["original_is_a"] = False

    # Block randomization keeps A/B balanced within every stratum and audit
    # component: 8/8 for representative and 2/2 for challenge rows.
    blocks = assigned.groupby(["stratum_id", "sample_component"], sort=True)
    for _, block in blocks:
        if len(block) % 2:
            raise ValueError("A/B randomization blocks must have even size")
        original_a_indices = rng.choice(
            block.index.to_numpy(), size=len(block) // 2, replace=False
        )
        assigned.loc[original_a_indices, "original_is_a"] = True

    assigned["natural_is_a"] = ~assigned["original_is_a"]
    assigned["text_a"] = np.where(
        assigned["original_is_a"], assigned["original_text"], assigned["natural_question"]
    )
    assigned["text_b"] = np.where(
        assigned["original_is_a"], assigned["natural_question"], assigned["original_text"]
    )

    mapping = assigned[["audit_id", "original_is_a", "natural_is_a"]].copy()
    mapping.insert(0, "annotator_id", annotator_id)
    mapping["a_version"] = np.where(mapping["original_is_a"], "original", "natural")
    mapping["b_version"] = np.where(mapping["original_is_a"], "natural", "original")
    mapping["original_is_a"] = mapping["original_is_a"].astype(int)
    mapping["natural_is_a"] = mapping["natural_is_a"].astype(int)
    mapping = mapping[
        [
            "annotator_id",
            "audit_id",
            "a_version",
            "b_version",
            "original_is_a",
            "natural_is_a",
        ]
    ].sort_values("audit_id")

    visible_columns = ["audit_id", "text_a", "text_b", "ans0", "ans1", "ans2"]
    shuffled = assigned.iloc[rng.permutation(len(assigned))][visible_columns].reset_index(
        drop=True
    )
    shuffled.insert(0, "annotation_order", np.arange(1, len(shuffled) + 1))
    shuffled.insert(1, "annotator_id", annotator_id)
    shuffled = shuffled.rename(
        columns={"ans0": "answer_0", "ans1": "answer_1", "ans2": "answer_2"}
    )
    for column in ANNOTATION_COLUMNS:
        shuffled[column] = 0 if column in ERROR_COLUMNS else ""
    return shuffled, mapping.reset_index(drop=True)


def _add_list_validation(
    worksheet,
    column_index: int,
    values: list[str],
    max_row: int,
) -> None:
    column_letter = get_column_letter(column_index)
    validation = DataValidation(
        type="list",
        formula1=f'"{",".join(values)}"',
        allow_blank=True,
        showErrorMessage=True,
        errorTitle="Invalid value",
        error=f"Choose one of: {', '.join(values)}",
        showInputMessage=True,
        promptTitle="Allowed values",
        prompt=", ".join(values),
    )
    worksheet.add_data_validation(validation)
    validation.add(f"{column_letter}2:{column_letter}{max_row}")


def write_annotation_workbook(annotations: pd.DataFrame, path: Path) -> None:
    """Write a protected, validated XLSX annotation interface."""
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Annotation"
    worksheet.sheet_view.showGridLines = False
    worksheet.freeze_panes = "D2"
    worksheet.sheet_properties.tabColor = "456F65"

    worksheet.append(list(annotations.columns))
    for row in annotations.itertuples(index=False, name=None):
        worksheet.append(list(row))

    header_fills = {
        "identifier": PatternFill("solid", fgColor="59636E"),
        "stimulus": PatternFill("solid", fgColor="477A9E"),
        "judgment": PatternFill("solid", fgColor="4D7C68"),
        "error": PatternFill("solid", fgColor="A56235"),
    }
    input_fill = PatternFill("solid", fgColor="EAF4EF")
    error_fill = PatternFill("solid", fgColor="FFF0E4")
    locked_fill = PatternFill("solid", fgColor="F3F5F6")
    warning_fill = PatternFill("solid", fgColor="F8D7DA")

    column_index = {name: i + 1 for i, name in enumerate(annotations.columns)}
    stimulus_columns = {"text_a", "text_b", "answer_0", "answer_1", "answer_2"}
    identifier_columns = {"annotation_order", "annotator_id", "audit_id"}

    for name, index in column_index.items():
        header = worksheet.cell(row=1, column=index)
        if name in identifier_columns:
            header.fill = header_fills["identifier"]
        elif name in stimulus_columns:
            header.fill = header_fills["stimulus"]
        elif name in ERROR_COLUMNS:
            header.fill = header_fills["error"]
        else:
            header.fill = header_fills["judgment"]
        header.font = Font(color="FFFFFF", bold=True)
        header.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    worksheet.row_dimensions[1].height = 42

    width_by_name = {
        "annotation_order": 12,
        "annotator_id": 14,
        "audit_id": 19,
        "text_a": 62,
        "text_b": 62,
        "answer_0": 28,
        "answer_1": 28,
        "answer_2": 28,
        "answer_a": 17,
        "answer_b": 17,
        "semantic_validity": 19,
        "evidence_state_preserved": 24,
        "naturalness_a_1_to_5": 22,
        "naturalness_b_1_to_5": 22,
        "preferred_version": 19,
        "annotator_notes": 42,
    }
    for name, index in column_index.items():
        width = width_by_name.get(name, 23 if name in ERROR_COLUMNS else 18)
        worksheet.column_dimensions[get_column_letter(index)].width = width

    max_row = worksheet.max_row
    input_columns = set(ANNOTATION_COLUMNS)
    for row_index in range(2, max_row + 1):
        worksheet.row_dimensions[row_index].height = 72
        for name, index in column_index.items():
            cell = worksheet.cell(row=row_index, column=index)
            cell.alignment = Alignment(
                vertical="top",
                horizontal="left" if name in stimulus_columns | {"annotator_notes"} else "center",
                wrap_text=name in stimulus_columns | {"annotator_notes"},
            )
            if name in input_columns:
                cell.protection = Protection(locked=False)
                cell.fill = error_fill if name in ERROR_COLUMNS else input_fill
            else:
                cell.protection = Protection(locked=True)
                cell.fill = locked_fill

    validation_values = {
        "answer_a": ANNOTATION_SCHEMA["answer_a"],
        "answer_b": ANNOTATION_SCHEMA["answer_b"],
        "semantic_validity": ANNOTATION_SCHEMA["semantic_validity"],
        "evidence_state_preserved": ANNOTATION_SCHEMA["evidence_state_preserved"],
        "naturalness_a_1_to_5": ANNOTATION_SCHEMA["naturalness_a_1_to_5"],
        "naturalness_b_1_to_5": ANNOTATION_SCHEMA["naturalness_b_1_to_5"],
        "preferred_version": ANNOTATION_SCHEMA["preferred_version"],
    }
    for name, values in validation_values.items():
        _add_list_validation(worksheet, column_index[name], values, max_row)
    for name in ERROR_COLUMNS:
        _add_list_validation(
            worksheet, column_index[name], ANNOTATION_SCHEMA["error_columns"], max_row
        )

    semantic_col = get_column_letter(column_index["semantic_validity"])
    first_error_col = get_column_letter(column_index[ERROR_COLUMNS[0]])
    last_error_col = get_column_letter(column_index[ERROR_COLUMNS[-1]])
    notes_col = get_column_letter(column_index["annotator_notes"])
    other_col = get_column_letter(column_index["error_other"])
    worksheet.conditional_formatting.add(
        f"{semantic_col}2:{semantic_col}{max_row}",
        FormulaRule(
            formula=[
                f'AND(OR(${semantic_col}2="invalid",${semantic_col}2="uncertain"),'
                f'SUM(${first_error_col}2:${last_error_col}2)=0)'
            ],
            fill=warning_fill,
        ),
    )
    worksheet.conditional_formatting.add(
        f"{notes_col}2:{notes_col}{max_row}",
        FormulaRule(
            formula=[f'AND(${other_col}2=1,${notes_col}2="")'], fill=warning_fill
        ),
    )

    worksheet.protection.sheet = True
    worksheet.protection.sort = True
    worksheet.protection.autoFilter = True
    worksheet.protection.formatCells = True
    worksheet.protection.formatColumns = True
    worksheet.protection.formatRows = True
    worksheet.protection.insertRows = True
    worksheet.protection.deleteRows = True
    worksheet.protection.set_password(WORKSHEET_PASSWORD)

    instructions = workbook.create_sheet("Instructions")
    instructions.sheet_view.showGridLines = False
    instructions.sheet_properties.tabColor = "DCEAE5"
    instruction_rows = [
        ("Paired Prompt Human Evaluation", "Use the accompanying ANNOTATION_GUIDELINE.md."),
        ("Workflow", "Answer and rate A before reading B; then compare the pair."),
        ("Unknown answers", "Choose the numbered Unknown/Cannot answer option. cannot_judge is only for uninterpretable text."),
        ("Error fields", "All error fields start at 0. Change only applicable errors to 1."),
        ("Evidence direction", "Code omission/addition from A to B: omission means B drops information from A; addition means B introduces information absent from A."),
        ("Evidence state", "Judge whether A and B share the same answerability state. Their supported answers need not be identical."),
        ("Naturalness", "Rate only each prompt body, excluding answer options and the spreadsheet interface."),
        ("Completion", "Fill answer A/B, semantic validity, evidence preservation, both naturalness scores, and preference for every row."),
        ("Invalid/uncertain", "At least one error field must be 1."),
        ("error_other", "A note is required whenever error_other is 1."),
        ("Ordering", "Do not sort, filter, reorder, delete, or insert rows."),
        ("Independence", "Do not discuss audit items with other annotators before submission."),
    ]
    for row in instruction_rows:
        instructions.append(row)
    instructions.column_dimensions["A"].width = 24
    instructions.column_dimensions["B"].width = 105
    for row_index in range(1, len(instruction_rows) + 1):
        instructions.row_dimensions[row_index].height = 34
        instructions.cell(row_index, 1).font = Font(bold=True, color="456F65")
        for column in (1, 2):
            cell = instructions.cell(row_index, column)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            cell.protection = Protection(locked=True)
    instructions.protection.sheet = True
    instructions.protection.set_password(WORKSHEET_PASSWORD)

    workbook.save(path)


def build_summary(source: pd.DataFrame, sample: pd.DataFrame, seed: int) -> dict:
    representative = sample[sample["sample_component"] == "representative"]
    challenge = sample[sample["sample_component"] == "challenge"]
    return {
        "input": str(DEFAULT_INPUT),
        "seed": seed,
        "population_pairs": int(len(source)),
        "strata": int(source.groupby(STRATUM_COLUMNS, observed=True).ngroups),
        "sample_pairs": int(len(sample)),
        "representative_pairs": int(len(representative)),
        "challenge_pairs": int(len(challenge)),
        "independent_judgments_with_three_annotators": int(len(sample) * 3),
        "category_counts": {
            str(k): int(v) for k, v in sample["category"].value_counts().sort_index().items()
        },
        "condition_counts": {
            str(k): int(v)
            for k, v in sample["context_condition"].value_counts().sort_index().items()
        },
        "cluster_counts": {
            str(k): int(v)
            for k, v in sample["target_cluster_id"].value_counts().sort_index().items()
        },
        "challenge_fallback_pairs": int(challenge["fallback_used"].sum()),
        "challenge_strata_with_fallback": int(
            challenge.loc[challenge["fallback_used"], "stratum_id"].nunique()
        ),
        "representative_fallback_pairs": int(representative["fallback_used"].sum()),
        "representative_weight_sum": float(
            representative["representative_normalized_weight"].sum()
        ),
        "sbert_similarity": {
            "representative_min": float(representative["sbert_similarity"].min()),
            "representative_median": float(representative["sbert_similarity"].median()),
            "representative_max": float(representative["sbert_similarity"].max()),
            "challenge_min": float(challenge["sbert_similarity"].min()),
            "challenge_median": float(challenge["sbert_similarity"].median()),
            "challenge_max": float(challenge["sbert_similarity"].max()),
        },
    }


def main() -> None:
    args = parse_args()
    source = pd.read_parquet(args.input, columns=REQUIRED_COLUMNS)
    if source["bbq_item_id"].duplicated().any():
        raise ValueError("Input contains duplicate bbq_item_id values")
    if source[REQUIRED_COLUMNS].isna().any().any():
        missing = source[REQUIRED_COLUMNS].isna().sum()
        raise ValueError(f"Required input values are missing:\n{missing[missing > 0]}")

    sample = build_sample(source, args.seed)
    validate_sample(source, sample)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    master_path = args.output_dir / "human_validity_sample_2640_master.csv"
    sample.to_csv(master_path, index=False, encoding="utf-8-sig")

    item_sets = []
    presentation_orders = []
    mappings = []
    for annotator in range(1, args.annotators + 1):
        annotations, mapping = annotation_frame(sample, annotator, args.seed)
        path = args.output_dir / f"human_validity_annotator_{annotator}.csv"
        annotations.to_csv(path, index=False, encoding="utf-8-sig")
        workbook_path = args.output_dir / f"human_validity_annotator_{annotator}.xlsx"
        write_annotation_workbook(annotations, workbook_path)
        item_sets.append(set(annotations["audit_id"]))
        presentation_orders.append(tuple(annotations["audit_id"]))
        mappings.append(mapping)
    if any(item_set != item_sets[0] for item_set in item_sets[1:]):
        raise ValueError("Annotator files do not contain identical item sets")
    if len(set(presentation_orders)) != args.annotators:
        raise ValueError("Annotator presentation orders must differ")

    private_dir = args.output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_mapping = pd.concat(mappings, ignore_index=True)
    mapping_path = private_dir / "human_validity_ab_mapping_private.csv"
    private_mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")

    for annotator_id, group in private_mapping.groupby("annotator_id"):
        counts = group["original_is_a"].value_counts().to_dict()
        if counts != {0: 1_320, 1: 1_320}:
            raise ValueError(f"Unbalanced A/B mapping for {annotator_id}: {counts}")
    assignment_sets = [
        frozenset(group.loc[group["original_is_a"] == 1, "audit_id"])
        for _, group in private_mapping.groupby("annotator_id", sort=True)
    ]
    if len(set(assignment_sets)) != args.annotators:
        raise ValueError("Annotators must have independently randomized A/B assignments")

    schema_path = args.output_dir / "annotation_schema.json"
    schema = dict(ANNOTATION_SCHEMA)
    schema["error_fields"] = ERROR_COLUMNS
    schema["required_completion_fields"] = REQUIRED_RESPONSE_COLUMNS
    schema["error_default"] = 0
    schema["completion_rule"] = (
        "All required completion fields must be nonblank. For invalid or uncertain "
        "semantic validity, at least one error field must equal 1. error_other=1 "
        "requires a nonblank annotator note."
    )
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    guideline_output_path = args.output_dir / "ANNOTATION_GUIDELINE.md"
    shutil.copyfile(GUIDELINE_PATH, guideline_output_path)

    summary = build_summary(source, sample, args.seed)
    summary["input"] = str(args.input.resolve())
    summary["output_dir"] = str(args.output_dir.resolve())
    summary_path = args.output_dir / "sampling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(sample):,} unique pairs to {master_path}")
    print(f"  representative: {summary['representative_pairs']:,}")
    print(f"  challenge:      {summary['challenge_pairs']:,}")
    print(f"  challenge fallback items: {summary['challenge_fallback_pairs']:,}")
    print(f"  annotator files: {args.annotators} CSV + {args.annotators} XLSX")
    print(f"  private A/B mapping: {mapping_path}")
    print(f"  annotation schema: {schema_path}")
    print(f"  annotation guideline: {guideline_output_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
