"""Fallback retry for BON items with zero valid candidates after full pipeline.

Identifies missing items, builds retry prompts, and submits them as an
OpenAI Batch API job.  Results are collected by collect_batch_fallback_bon.py.

Retry prompt strategy
---------------------
  - hard_check_status == "retry_style_failure" (exact_copy):
      original user_prompt + EXACT_COPY_RETRY_INSTRUCTION
  - all other failures:
      original user_prompt unchanged

Output
------
  --batch-jsonl   : JSONL file for Batch API upload
  --batch-meta    : JSON with batch_id (read by collect_batch_fallback_bon.py)
  --candidates    : parquet of candidates sent for retry (metadata for join)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXACT_COPY_RETRY_INSTRUCTION = """\
<RETRY_FEEDBACK>
The previous rewrite preserved the source wording and sentence structure too closely.

Rewrite the item again while following all original semantic-preservation constraints.
This time:
- change the sentence structure and phrasing;
- integrate the context and question more naturally;
- apply the style inferred from the reference more clearly;
- do not merely concatenate or lightly edit the source context and question;
- preserve every fact, identity, relationship, role, event, polarity,
  ambiguity condition, and answer mapping;
- do not add new information or imply an answer.

Return only:
{"natural_question": "..."}
</RETRY_FEEDBACK>"""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "natural_question": {
            "type": "string",
            "description": "The BBQ item rewritten as a single natural open-ended question.",
        },
    },
    "required": ["natural_question"],
    "additionalProperties": False,
}


def load_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")
    return OpenAI()


def find_missing_item_ids(
    bon_df: pd.DataFrame,
    style_scored: pd.DataFrame,
    item_id_col: str = "bbq_item_id",
) -> list[str]:
    all_ids = set(bon_df[item_id_col].astype(str))
    valid_ids = set(style_scored[item_id_col].astype(str))
    return sorted(all_ids - valid_ids)


def build_retry_user_prompt(row: pd.Series) -> str:
    status = str(row.get("hard_check_status", ""))
    base_prompt = str(row["user_prompt"])
    if status == "retry_style_failure":
        return base_prompt + "\n\n" + EXACT_COPY_RETRY_INSTRUCTION
    return base_prompt


def make_batch_request(row: pd.Series, model: str, max_tokens: int, temperature: float | None) -> dict:
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": str(row["system_prompt"])},
            {"role": "user", "content": build_retry_user_prompt(row)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "bbq_rewrite",
                "strict": True,
                "schema": REWRITE_SCHEMA,
            },
        },
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    return {
        "custom_id": str(row["rewrite_id"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Identify BON items with zero valid candidates, build retry prompts, "
            "and submit as an OpenAI Batch API job."
        )
    )
    parser.add_argument("--bon-inputs", type=Path, required=True)
    parser.add_argument("--hard-checked", type=Path, required=True)
    parser.add_argument("--style-scored", type=Path, required=True)
    parser.add_argument(
        "--batch-jsonl", type=Path, required=True,
        help="Output JSONL for Batch API upload.",
    )
    parser.add_argument(
        "--batch-meta", type=Path, required=True,
        help="Output JSON to store batch_id (read by collect_batch_fallback_bon.py).",
    )
    parser.add_argument(
        "--candidates", type=Path, required=True,
        help="Output parquet of candidates sent for retry (metadata for join in collect step).",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in [args.batch_jsonl, args.batch_meta, args.candidates]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite to replace.")

    bon_df = pd.read_parquet(args.bon_inputs)
    hard_checked = pd.read_parquet(args.hard_checked)
    style_scored = pd.read_parquet(args.style_scored)

    missing_item_ids = find_missing_item_ids(bon_df, style_scored)
    n_total = bon_df["bbq_item_id"].nunique()
    n_missing = len(missing_item_ids)

    print(f"Total items: {n_total}")
    print(f"Items with ≥1 valid candidate: {n_total - n_missing}")
    print(f"Items with 0 valid candidates (to retry): {n_missing}")

    args.batch_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Early exit: nothing to retry
    # ------------------------------------------------------------------
    if n_missing == 0:
        print("[fallback] No missing items. Writing empty outputs.")
        empty = hard_checked.iloc[:0].copy()
        empty.to_parquet(args.candidates, index=False)
        args.batch_jsonl.write_text("", encoding="utf-8")
        meta = {
            "n_missing_items": 0,
            "missing_item_ids": [],
            "n_candidates": 0,
            "batch_id": None,
            "skipped": True,
        }
        args.batch_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
        return

    # ------------------------------------------------------------------
    # Identify candidates to retry
    # ------------------------------------------------------------------
    missing_set = set(missing_item_ids)
    valid_rewrite_ids = set(style_scored["rewrite_id"].astype(str))

    candidates = hard_checked[
        hard_checked["bbq_item_id"].astype(str).isin(missing_set)
        & ~hard_checked["rewrite_id"].astype(str).isin(valid_rewrite_ids)
    ].copy().reset_index(drop=True)

    if candidates.empty:
        print("[fallback] Missing items detected but no retryable candidates found.")
        hard_checked.iloc[:0].to_parquet(args.candidates, index=False)
        args.batch_jsonl.write_text("", encoding="utf-8")
        meta = {
            "n_missing_items": n_missing,
            "missing_item_ids": missing_item_ids,
            "n_candidates": 0,
            "batch_id": None,
            "skipped": True,
        }
        args.batch_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return

    status_counts = candidates["hard_check_status"].value_counts().to_dict()
    print(f"\nCandidates to retry: {len(candidates)}")
    print(f"  hard_check_status: " + ", ".join(f"{k}={v}" for k, v in status_counts.items()))

    # ------------------------------------------------------------------
    # Build JSONL
    # ------------------------------------------------------------------
    with args.batch_jsonl.open("w", encoding="utf-8") as f:
        for _, row in candidates.iterrows():
            req = make_batch_request(row, args.model, args.max_output_tokens, args.temperature)
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"Wrote {len(candidates):,} requests → {args.batch_jsonl}")

    # ------------------------------------------------------------------
    # Save candidate metadata (for join in collect step)
    # ------------------------------------------------------------------
    candidates.to_parquet(args.candidates, index=False)

    # ------------------------------------------------------------------
    # Upload + submit batch
    # ------------------------------------------------------------------
    client = load_openai_client()
    print(f"Uploading {args.batch_jsonl} ...")
    with args.batch_jsonl.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  file_id: {uploaded.id}")

    print("Submitting batch ...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  batch_id: {batch.id}")
    print(f"  status:   {batch.status}")

    meta = {
        "n_missing_items": n_missing,
        "missing_item_ids": missing_item_ids,
        "n_candidates": len(candidates),
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "skipped": False,
    }
    args.batch_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote batch meta: {args.batch_meta}")
    print("Run collect_batch_fallback_bon.py when the batch completes.")


if __name__ == "__main__":
    main()
