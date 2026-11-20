"""Build matched-body A/B prompt parquets for Natural-BBQ evaluation.

Conditions
----------
A  Original-MCQ            : original context+question + choices + digit-only constraint
B  Natural-MCQ             : natural_question + choices + digit-only constraint

Output parquets (one per condition):
  data/processed/bias_eval_v2_matched_body/prompts/prompts_{condition}.parquet

Usage
-----
  python dataset/bbq/scripts/build_bias_eval_prompts.py \\
      --input   data/processed/rewrite_output_bon/bbq_bon_vfull_selected_matched_body.parquet \\
      --dev-ids data/processed/natural_bbq/prompt_dev_item_ids.json \\
      --out-dir data/processed/bias_eval_v2_matched_body/prompts \\
      --overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RENDERER_VERSION = "renderer_v1"

EvalCondition = Literal[
    "A_original_mcq",
    "B_natural_mcq",
]

# Metadata columns copied from source into every prompt parquet
META_COLS = [
    "bbq_item_id",
    "category",
    "context_condition",
    "question_polarity",
    "label",
    "ans0",
    "ans1",
    "ans2",
    "answer_info_json",
    "additional_metadata_json",
    "rewrite_status",
    "candidate_stage",
    "fallback_used",
    "target_cluster_id",
    "heldout_delta_style",
    "heldout_style_check_status",
    "unused_delta_style",
    "unused_style_check_status",
    "sbert_similarity",
]

_DIGIT_INSTRUCTION = "Answer with only a single digit: 0, 1, or 2."


def _options_block(ans0: str, ans1: str, ans2: str) -> str:
    return f"0. {ans0}\n1. {ans1}\n2. {ans2}"


def render_A(row: pd.Series) -> str:
    """Render the canonical Original-MCQ condition."""
    opts = _options_block(row["ans0"], row["ans1"], row["ans2"])
    return (
        f"Context: {row['original_context']}\n"
        f"Question: {row['original_question']}\n"
        f"{opts}\n\n"
        f"{_DIGIT_INSTRUCTION}"
    )


def render_B(row: pd.Series) -> str:
    """Natural-MCQ: natural question + choices + digit-only."""
    opts = _options_block(row["ans0"], row["ans1"], row["ans2"])
    return f"{row['natural_question']}\n\n{opts}\n\n{_DIGIT_INSTRUCTION}"


_RENDERERS = {
    "A_original_mcq":         render_A,
    "B_natural_mcq":          render_B,
}

_CONDITION_META = {
    "A_original_mcq":         {"prompt_body_type": "original", "choices_visible": True,  "response_constraint": "digit_only"},
    "B_natural_mcq":          {"prompt_body_type": "natural",  "choices_visible": True,  "response_constraint": "digit_only"},
}


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_condition(df: pd.DataFrame, condition: EvalCondition) -> pd.DataFrame:
    renderer = _RENDERERS[condition]
    meta = _CONDITION_META[condition]

    prompt_texts = df.apply(renderer, axis=1)
    prompt_hashes = prompt_texts.map(_sha256_short)

    out_cols = {c: df[c] for c in META_COLS if c in df.columns}
    out = pd.DataFrame(out_cols)
    out["eval_condition"]      = condition
    out["prompt_text"]         = prompt_texts.values
    out["prompt_hash"]         = prompt_hashes.values
    out["prompt_body_type"]    = meta["prompt_body_type"]
    out["choices_visible"]     = meta["choices_visible"]
    out["response_constraint"] = meta["response_constraint"]
    out["renderer_version"]    = RENDERER_VERSION

    # Reorder: identity columns first
    lead = ["bbq_item_id", "eval_condition", "prompt_text", "prompt_hash",
            "prompt_body_type", "choices_visible", "response_constraint", "renderer_version"]
    rest = [c for c in out.columns if c not in lead]
    return out[lead + rest]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build A/B bias-eval prompt parquets."
    )
    ap.add_argument("--input", type=Path,
                    default=PROJECT_ROOT / "data/processed/rewrite_output_bon/bbq_bon_vfull_selected_matched_body.parquet")
    ap.add_argument("--dev-ids", type=Path,
                    default=PROJECT_ROOT / "data/processed/natural_bbq/prompt_dev_item_ids.json")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "data/processed/bias_eval_v2_matched_body/prompts")
    ap.add_argument("--conditions", nargs="+",
                    default=list(_RENDERERS.keys()),
                    choices=list(_RENDERERS.keys()),
                    help="Which conditions to build (default: all).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Load source
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Exclude prompt-dev items (they should already be absent, but be explicit)
    if args.dev_ids.exists():
        dev_payload = json.loads(args.dev_ids.read_text(encoding="utf-8"))
        dev_id_set = set(dev_payload["item_ids"])
        before = len(df)
        df = df[~df["bbq_item_id"].isin(dev_id_set)].copy()
        if len(df) < before:
            print(f"  Excluded {before - len(df)} prompt-dev rows.")
    else:
        dev_id_set = set()
        print(f"  [warn] dev-ids file not found: {args.dev_ids}. No exclusion applied.")

    print(f"  Items for evaluation: {len(df)}")

    # Deduplicate bbq_item_id (should be 1 per item in selected parquet)
    if df["bbq_item_id"].duplicated().any():
        raise ValueError("bbq_item_id duplicates detected in source. Cannot build unambiguous prompts.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        out_path = args.out_dir / f"prompts_{condition}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"  [skip] {out_path} exists. Use --overwrite.")
            continue

        print(f"\nBuilding condition {condition} ...")
        cdf = build_condition(df, condition)

        # Sanity: unique item-condition pairs
        assert cdf[["bbq_item_id", "eval_condition"]].duplicated().sum() == 0, \
            f"Duplicate item-condition pairs in {condition}"
        assert cdf["prompt_text"].notna().all(), f"Null prompt_text in {condition}"
        n_dup_hashes = len(cdf) - cdf["prompt_hash"].nunique()
        if n_dup_hashes > 0:
            print(f"  [info] {n_dup_hashes} rows share a prompt_hash with another row "
                  f"(different bbq_item_id, identical rendered text — expected for some BBQ scenarios).")

        cdf.to_parquet(out_path, index=False)
        print(f"  Wrote {len(cdf):,} rows → {out_path}")

    print("\nDone. Summary:")
    for condition in args.conditions:
        p = args.out_dir / f"prompts_{condition}.parquet"
        if p.exists():
            n = len(pd.read_parquet(p, columns=["bbq_item_id"]))
            print(f"  {condition}: {n:,} rows")


if __name__ == "__main__":
    main()
