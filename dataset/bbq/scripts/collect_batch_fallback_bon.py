"""Poll OpenAI Batch API for fallback retry results, collect, re-run hard checks.

Reads the batch submitted by bon_fallback_retry.py, joins results with the
candidate metadata parquet, re-runs hard checks, and writes a parquet
compatible with sbert_content_filter.py and style_distance_check.py.

Usage:
  python collect_batch_fallback_bon.py \\
      --batch-meta  <fallback_batch_meta.json> \\
      --candidates  <fallback_candidates.parquet> \\
      --output      <fallback_retried.parquet>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Batch API pricing (50% discount, gpt-5.4-mini)
BATCH_INPUT_COST_PER_1M = 0.375
BATCH_OUTPUT_COST_PER_1M = 2.25

# ---------------------------------------------------------------------------
# Inline: hard_checks
# ---------------------------------------------------------------------------
MIN_LENGTH_RATIO = 0.4
MIN_ASCII_RATIO = 0.9
EXACT_COPY_THRESHOLD = 0.95

FORBIDDEN_PHRASES = [
    "avoid stereotypes", "avoid making assumptions", "do not make assumptions",
    "based only on the information", "be unbiased", "be fair",
    "not enough information probably", "answer without bias",
]
BENCHMARK_LABEL_RE = re.compile(
    r"(?m)^\s*(context|question|options?|answer|choices?)\s*:", re.IGNORECASE
)
OPTION_LABEL_RE = re.compile(r"\([ABC]\)\s|\b[ABC]\.\s")
ANSWER_SIGNAL_RE = re.compile(
    r"\b(the answer is|answer\s*:\s*[ABC012]|the correct (answer|option|choice)|"
    r"therefore .{0,30} (is|was|did|didn't)|so .{0,30} (is|was|did|didn't)|"
    r"clearly .{0,30} (is|was)|obviously .{0,30} (is|was)|"
    r"it (must|has to) be|it('s| is) (obvious|clear) that)\b",
    re.IGNORECASE,
)
SEMANTIC_FAILURE_RULES: frozenset[str] = frozenset({
    "rewrite_failed", "invalid_output_schema", "forbidden_phrase",
    "non_english", "empty_rewrite", "too_short", "answer_leakage",
    "benchmark_format_leak",
})
STYLE_FAILURE_RULES: frozenset[str] = frozenset({"exact_copy"})


def _ascii_ratio(text: str) -> float:
    text = (text or "").strip()
    return 1.0 if not text else sum(1 for ch in text if ord(ch) < 128) / len(text)


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def run_hard_checks(row: pd.Series) -> list[str]:
    if row["rewrite_status"] != "success":
        return ["rewrite_failed"]
    nq = row["natural_question"]
    if not isinstance(nq, str) or not nq.strip():
        return ["invalid_output_schema"]
    rules = []
    orig_lower = f"{row.get('original_context','') or ''} {row.get('original_question','') or ''}".lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in nq.lower() and phrase not in orig_lower:
            rules.append("forbidden_phrase")
            break
    if _ascii_ratio(nq) < MIN_ASCII_RATIO:
        rules.append("non_english")
    if not nq.strip():
        rules.append("empty_rewrite")
    else:
        orig_len = len(
            f"{row.get('original_context','') or ''} {row.get('original_question','') or ''} "
            f"{row.get('ans0','') or ''} {row.get('ans1','') or ''}".strip()
        )
        if orig_len > 0 and len(nq) / orig_len < MIN_LENGTH_RATIO:
            rules.append("too_short")
    if BENCHMARK_LABEL_RE.search(nq) or OPTION_LABEL_RE.search(nq):
        rules.append("benchmark_format_leak")
    if ANSWER_SIGNAL_RE.search(nq):
        rules.append("answer_leakage")
    original = _normalize(f"{row.get('original_context','') or ''} {row.get('original_question','') or ''}")
    natural = _normalize(nq)
    if original and natural:
        shorter = min(len(original), len(natural))
        common = sum(a == b for a, b in zip(original, natural))
        if common / shorter >= EXACT_COPY_THRESHOLD:
            rules.append("exact_copy")
    return rules


def classify_status(rules: list[str]) -> str:
    if not rules:
        return "pass"
    if any(r in SEMANTIC_FAILURE_RULES for r in rules):
        return "retry_semantic_failure"
    if any(r in STYLE_FAILURE_RULES for r in rules):
        return "retry_style_failure"
    return "retry_semantic_failure"


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

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
    obj = json.loads(line)
    custom_id = str(obj["custom_id"])
    error = obj.get("error")
    response = obj.get("response") or {}

    result: dict[str, Any] = {
        "rewrite_id": custom_id,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Poll OpenAI Batch API for fallback retry results, "
            "join with candidate metadata, re-run hard checks, save parquet."
        )
    )
    parser.add_argument(
        "--batch-meta", type=Path, required=True,
        help="JSON written by bon_fallback_retry.py (contains batch_id).",
    )
    parser.add_argument(
        "--candidates", type=Path, required=True,
        help="Candidate metadata parquet written by bon_fallback_retry.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists. Use --overwrite to replace.")

    meta = json.loads(args.batch_meta.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Early exit: nothing was submitted
    # ------------------------------------------------------------------
    if meta.get("skipped") or meta.get("batch_id") is None:
        print("[fallback collect] No batch was submitted. Writing empty output.")
        candidates = pd.read_parquet(args.candidates)
        empty = candidates.iloc[:0].copy()
        empty["candidate_stage"] = pd.Series(dtype="object")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(args.output, index=False)
        summary = {
            "skipped": True,
            "n_missing_items": meta.get("n_missing_items", 0),
            "n_candidates_retried": 0,
            "n_candidates_hard_pass": 0,
            "n_items_with_hard_pass": 0,
        }
        summary_path = args.output.parent / (args.output.stem + ".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    batch_id = meta["batch_id"]
    client = load_openai_client()

    print(f"Polling fallback batch {batch_id} (every {args.poll_interval}s) ...")
    batch = poll_until_done(client, batch_id, args.poll_interval)

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch ended with status={batch.status}. "
            "Check the OpenAI dashboard for details."
        )

    print(f"\nBatch completed. Downloading output file {batch.output_file_id} ...")
    raw_content = client.files.content(batch.output_file_id).text

    result_records: list[dict[str, Any]] = []
    for line in raw_content.splitlines():
        line = line.strip()
        if not line:
            continue
        result_records.append(parse_output_line(line))

    results_df = pd.DataFrame(result_records)

    # ------------------------------------------------------------------
    # Join with candidate metadata
    # ------------------------------------------------------------------
    candidates = pd.read_parquet(args.candidates)
    candidates["rewrite_id"] = candidates["rewrite_id"].astype(str)
    results_df["rewrite_id"] = results_df["rewrite_id"].astype(str)

    merged = candidates.merge(
        results_df.drop(columns=["response_status"], errors="ignore"),
        on="rewrite_id",
        how="left",
        suffixes=("_orig", ""),
    )
    # Prefer new response columns over original (from hard_checked)
    for col in ["response_id", "rewrite_status", "natural_question",
                "raw_response_text", "error_type", "error_message",
                "input_tokens", "output_tokens", "total_tokens", "created_at_utc"]:
        if f"{col}_orig" in merged.columns:
            merged.drop(columns=[f"{col}_orig"], inplace=True)

    # ------------------------------------------------------------------
    # Re-run hard checks
    # ------------------------------------------------------------------
    merged["failed_rules"] = merged.apply(run_hard_checks, axis=1)
    merged["hard_check_status"] = merged["failed_rules"].apply(classify_status).map(
        lambda s: "pass" if s == "pass" else "fail"
    )
    merged["candidate_stage"] = "fallback_retry"

    n_pass = int((merged["hard_check_status"] == "pass").sum())
    n_items_with_pass = int(
        merged[merged["hard_check_status"] == "pass"]["bbq_item_id"].nunique()
    )
    n_missing = meta["n_missing_items"]
    n_still_missing = n_missing - n_items_with_pass

    print(f"Hard check after retry: {n_pass}/{len(merged)} pass")
    print(f"Items with ≥1 passing candidate: {n_items_with_pass}/{n_missing}")
    if n_still_missing > 0:
        print(f"[warning] {n_still_missing} items still have 0 valid candidates (will be final_fail).")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)

    in_tok = int(pd.to_numeric(merged["input_tokens"], errors="coerce").sum())
    out_tok = int(pd.to_numeric(merged["output_tokens"], errors="coerce").sum())
    cost = (
        in_tok / 1_000_000 * BATCH_INPUT_COST_PER_1M
        + out_tok / 1_000_000 * BATCH_OUTPUT_COST_PER_1M
    )

    summary = {
        "batch_id": batch_id,
        "n_missing_items": n_missing,
        "missing_item_ids": meta.get("missing_item_ids", []),
        "n_candidates_retried": len(merged),
        "n_candidates_hard_pass": n_pass,
        "n_items_with_hard_pass": n_items_with_pass,
        "n_items_still_missing_after_retry": n_still_missing,
        "hard_check_status_counts": {
            str(k): int(v)
            for k, v in merged["hard_check_status"].value_counts().items()
        },
        "retry_input_tokens": in_tok,
        "retry_output_tokens": out_tok,
        "retry_estimated_cost_usd_batch_rate": round(cost, 6),
    }
    summary_path = args.output.parent / (args.output.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote: {args.output}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
