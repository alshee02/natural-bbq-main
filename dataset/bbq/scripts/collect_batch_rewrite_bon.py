"""Poll OpenAI Batch API until completion, download results, save as parquet.

The output schema is the shared contract consumed by hard_checks.py,
sbert_content_filter.py, and style_distance_check.py.

Usage:
  python collect_batch_rewrite_bon.py \\
      --batch-meta <meta.json> \\
      --inputs     <bon_inputs.parquet> \\
      --output     <rewrite_outputs.parquet>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Shared rewrite-result schema used by the downstream filtering stages.
RESULT_COLUMNS = [
    "rewrite_id",
    "bbq_item_id",
    "source_file",
    "source_row_index",
    "example_id",
    "question_index",
    "question_polarity",
    "context_condition",
    "category",
    "label",
    "ans0",
    "ans1",
    "ans2",
    "original_context",
    "original_question",
    "answer_info_json",
    "additional_metadata_json",
    "stratum_key",
    "target_cluster_id",
    "reference_ids",
    "reference_sample_types",
    "reference_texts",
    "rewrite_model",
    "system_prompt",
    "user_prompt",
    "prompt_char_count",
    "api",
    "requested_model",
    "response_id",
    "response_status",
    "rewrite_status",
    "natural_question",
    "raw_response_text",
    "error_type",
    "error_message",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "api_elapsed_seconds",
    "created_at_utc",
]

# Batch API pricing (50% discount vs standard)
BATCH_INPUT_COST_PER_1M = 0.375   # gpt-5.4-mini
BATCH_OUTPUT_COST_PER_1M = 2.25   # gpt-5.4-mini


def load_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")
    return OpenAI()


def poll_until_done(client: Any, batch_id: str, poll_interval: int) -> Any:
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [{datetime.now(UTC).strftime('%H:%M:%S')}] "
            f"status={batch.status}  "
            f"total={counts.total}  "
            f"completed={counts.completed}  "
            f"failed={counts.failed}"
        )
        if batch.status in terminal:
            return batch
        time.sleep(poll_interval)


def parse_output_line(line: str) -> dict[str, Any]:
    """Parse one line of Batch API output JSONL into a flat result dict."""
    obj = json.loads(line)
    custom_id = str(obj["custom_id"])
    error = obj.get("error")
    response = obj.get("response") or {}

    result: dict[str, Any] = {
        "rewrite_id": custom_id,
        "api": "batch/chat",
        "requested_model": None,
        "response_id": None,
        "response_status": None,
        "rewrite_status": None,
        "natural_question": None,
        "raw_response_text": None,
        "error_type": None,
        "error_message": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "api_elapsed_seconds": None,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    if error:
        result["rewrite_status"] = "error"
        result["error_type"] = str(error.get("code", "batch_error"))
        result["error_message"] = str(error.get("message", ""))
        return result

    status_code = response.get("status_code")
    body = response.get("body") or {}

    result["response_id"] = body.get("id")
    result["response_status"] = str(status_code)
    result["requested_model"] = body.get("model")

    if status_code != 200:
        result["rewrite_status"] = "error"
        result["error_type"] = "http_error"
        result["error_message"] = f"status_code={status_code}"
        return result

    usage = body.get("usage") or {}
    result["input_tokens"] = usage.get("prompt_tokens")
    result["output_tokens"] = usage.get("completion_tokens")
    result["total_tokens"] = usage.get("total_tokens")

    try:
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("No choices in response")
        raw_text = choices[0]["message"]["content"]
        result["raw_response_text"] = raw_text
        payload = json.loads(raw_text)
        nq = payload["natural_question"]
        if not isinstance(nq, str):
            raise ValueError("natural_question must be a string")
        result["rewrite_status"] = "success"
        result["natural_question"] = nq
    except Exception as exc:
        result["rewrite_status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Poll OpenAI Batch API until completion, download results, "
            "and save the shared rewrite-result parquet."
        )
    )
    parser.add_argument(
        "--batch-meta", type=Path, required=True,
        help="JSON written by submit_batch_rewrite_bon.py (contains batch_id).",
    )
    parser.add_argument(
        "--inputs", type=Path, required=True,
        help="Original BON inputs parquet (joined to recover metadata columns).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between status polls. Default: 60.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists. Use --overwrite to replace.")

    meta = json.loads(args.batch_meta.read_text(encoding="utf-8"))

    # Support both single-batch (legacy) and multi-batch meta formats
    if "batches" in meta:
        batch_entries = meta["batches"]
    else:
        batch_entries = [meta]

    client = load_openai_client()

    result_records: list[dict[str, Any]] = []
    completed_batches = []

    for i, entry in enumerate(batch_entries):
        batch_id = entry["batch_id"]
        label = f"batch {i+1}/{len(batch_entries)} ({batch_id})"
        print(f"Polling {label} (every {args.poll_interval}s) ...")
        batch = poll_until_done(client, batch_id, args.poll_interval)

        if batch.status != "completed":
            raise RuntimeError(
                f"Batch {batch_id} ended with status={batch.status}. "
                "Check the OpenAI dashboard for details."
            )

        print(f"\nBatch completed. Downloading output file {batch.output_file_id} ...")
        raw_content = client.files.content(batch.output_file_id).text
        for line in raw_content.splitlines():
            line = line.strip()
            if not line:
                continue
            result_records.append(parse_output_line(line))
        completed_batches.append(batch)
        print(f"  {len(result_records):,} records collected so far.\n")

    results_df = pd.DataFrame(result_records)

    # Join with inputs to recover all metadata columns
    inputs_df = pd.read_parquet(args.inputs)
    inputs_df["rewrite_id"] = inputs_df["rewrite_id"].astype(str)
    results_df["rewrite_id"] = results_df["rewrite_id"].astype(str)

    merged = inputs_df.merge(
        results_df.drop(columns=["requested_model"], errors="ignore"),
        on="rewrite_id",
        how="left",
    )

    # Fill result columns that may be missing after left join
    for col in ["api", "response_id", "response_status", "rewrite_status",
                "natural_question", "raw_response_text", "error_type",
                "error_message", "input_tokens", "output_tokens",
                "total_tokens", "api_elapsed_seconds", "created_at_utc"]:
        if col not in merged.columns:
            merged[col] = None

    merged["requested_model"] = (
        results_df["requested_model"].dropna().iloc[0]
        if "requested_model" in results_df.columns and results_df["requested_model"].notna().any()
        else None
    )

    # Enforce column order
    output_cols = [c for c in RESULT_COLUMNS if c in merged.columns]
    extra_cols = [c for c in merged.columns if c not in RESULT_COLUMNS]
    merged = merged[output_cols + extra_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)

    # Summary
    status_counts = merged["rewrite_status"].value_counts(dropna=False).to_dict()
    n_success = int((merged["rewrite_status"] == "success").sum())
    in_tok = int(pd.to_numeric(merged["input_tokens"], errors="coerce").sum())
    out_tok = int(pd.to_numeric(merged["output_tokens"], errors="coerce").sum())
    cost = (
        in_tok / 1_000_000 * BATCH_INPUT_COST_PER_1M
        + out_tok / 1_000_000 * BATCH_OUTPUT_COST_PER_1M
    )

    batch_ids = [e["batch_id"] for e in batch_entries]
    summary = {
        "batch_ids": batch_ids,
        "n_batches": len(batch_ids),
        "batch_status": "completed",
        "n_rows": int(len(merged)),
        "n_success": n_success,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "estimated_cost_usd_batch_rate": round(cost, 6),
    }
    summary_path = args.output.parent / (args.output.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote: {args.output}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
