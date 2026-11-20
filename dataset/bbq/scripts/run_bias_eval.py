"""Natural-BBQ A/B evaluation for open-weight models.

Reads a pre-built prompt parquet (one condition), runs inference on each item,
and stores raw output.  Parsing is done separately.

Supported models:
  Qwen/Qwen3-4B-Instruct-2507
  google/gemma-3-4b-it
  meta-llama/Llama-3.1-8B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3

Usage
-----
  # Single condition
  python dataset/bbq/scripts/run_bias_eval.py \\
      --input  data/processed/bias_eval_v2_matched_body/prompts/prompts_B_natural_mcq.parquet \\
      --model  Qwen/Qwen3-4B-Instruct-2507 \\
      --batch-size 16

  # Dry-run (no inference, just show what would happen)
  python dataset/bbq/scripts/run_bias_eval.py \\
      --input  data/processed/bias_eval_v2_matched_body/prompts/prompts_B_natural_mcq.parquet \\
      --model  Qwen/Qwen3-4B-Instruct-2507 \\
      --dry-run

  # Resume a partial run
  python dataset/bbq/scripts/run_bias_eval.py \\
      --input  data/processed/bias_eval_v2_matched_body/prompts/prompts_B_natural_mcq.parquet \\
      --model  Qwen/Qwen3-4B-Instruct-2507 \\
      --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Condition-letter lookup
_CONDITION_LETTER = {
    "A_original_mcq": "A",
    "B_natural_mcq": "B",
}

# Default max_new_tokens per condition
_DEFAULT_MAX_TOKENS = {
    "A_original_mcq": 16,
    "B_natural_mcq": 16,
}

RUN_VERSION = "bias_eval_v2_matched_body"


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def _model_family(model_id: str) -> str:
    lower = model_id.lower()
    if "qwen" in lower:
        return "qwen"
    if "gemma" in lower:
        return "gemma"
    if "llama" in lower:
        return "llama"
    if "mistral" in lower:
        return "mistral"
    if "falcon" in lower:
        return "falcon"
    if "mpt" in lower:
        return "mpt"
    if "gpt" in lower:
        return "gpt"
    if "claude" in lower:
        return "claude"
    return "other"


def _model_short(model_id: str) -> str:
    """Short directory name for output paths."""
    parts = model_id.split("/")
    return parts[-1].lower().replace("-", "_")


# ---------------------------------------------------------------------------
# Chat template + inference
# ---------------------------------------------------------------------------

def apply_template(tokenizer, msgs: list[dict], model_name: str) -> str:
    # Base models without a chat template: use raw prompt text
    if not getattr(tokenizer, "chat_template", None):
        return msgs[-1]["content"]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "Qwen" in model_name:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(msgs, **kwargs)


def run_inference_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    model_name: str,
) -> list[dict]:
    """Run batched inference and return per-item dicts with raw_output and metadata."""
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    texts = [apply_template(tokenizer, msgs, model_name) for msgs in messages_batch]

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    results = []
    for i, out in enumerate(outputs):
        gen_tokens = out[input_len:]
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        results.append({
            "raw_output": raw_text,
            "input_tokens": input_len,
            "output_tokens": len(gen_tokens),
            "latency_seconds": round(elapsed / len(prompts), 4),
            "inference_ok": True,
            "inference_error_type": None,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run BBQ bias evaluation inference on a prompt parquet.",
    )
    ap.add_argument("--input", type=Path, required=True,
                    help="Prompt parquet (e.g. prompts_B_natural_mcq.parquet)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output parquet path. Default: auto-generated under "
                         "bias_eval_v2_matched_body/raw_predictions/{model}/"
                         "predictions_{letter}.parquet")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Override max_new_tokens (default: 16)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit the number of items for a smoke test")
    ap.add_argument("--item-ids", type=Path, default=None,
                    help="JSON file with list of bbq_item_ids to process")
    ap.add_argument("--resume", action="store_true",
                    help="Skip items already present in existing output")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output file entirely")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show plan without running inference")
    args = ap.parse_args()

    # ---- Load prompt parquet ----
    prompts_df = pd.read_parquet(args.input)
    eval_condition = prompts_df["eval_condition"].iloc[0]
    if eval_condition not in _CONDITION_LETTER:
        raise ValueError(
            f"Unsupported condition {eval_condition!r}; expected A_original_mcq "
            "or B_natural_mcq."
        )
    cond_letter = _CONDITION_LETTER[eval_condition]
    print(f"Condition: {eval_condition} ({cond_letter})")
    print(f"Items loaded: {len(prompts_df):,}")

    # ---- Resolve max_new_tokens ----
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else _DEFAULT_MAX_TOKENS[eval_condition]
    )

    # ---- Resolve output path ----
    if args.output is None:
        model_dir = _model_short(args.model)
        args.output = (
            PROJECT_ROOT / "data/processed/bias_eval_v2_matched_body/raw_predictions"
            / model_dir / f"predictions_{cond_letter}.parquet"
        )
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"max_new_tokens: {max_new_tokens}")

    # ---- Filter items ----
    if args.item_ids:
        ids = json.loads(args.item_ids.read_text(encoding="utf-8"))
        if isinstance(ids, dict):
            ids = ids.get("item_ids", ids.get("ids", []))
        prompts_df = prompts_df[prompts_df["bbq_item_id"].isin(set(ids))].copy()
        print(f"Filtered to {len(prompts_df):,} items by --item-ids")

    if args.limit:
        prompts_df = prompts_df.head(args.limit).copy()
        print(f"Limited to {len(prompts_df):,} items")

    # ---- Resume: skip only identical item-prompt pairs ----
    # A selected candidate can change while retaining the same bbq_item_id.
    # Comparing prompt hashes prevents stale predictions from being reused.
    existing_df: pd.DataFrame | None = None
    if args.resume and args.output.exists() and not args.overwrite:
        existing_df = pd.read_parquet(args.output)
        completed_df = existing_df
        if "inference_ok" in completed_df.columns:
            completed_df = completed_df[completed_df["inference_ok"]]
        before = len(prompts_df)
        if "prompt_hash" in completed_df.columns and "prompt_hash" in prompts_df.columns:
            existing_keys = set(zip(
                completed_df["bbq_item_id"].astype(str),
                completed_df["prompt_hash"].astype(str),
            ))
            pending = [
                (str(item_id), str(prompt_hash)) not in existing_keys
                for item_id, prompt_hash in zip(
                    prompts_df["bbq_item_id"], prompts_df["prompt_hash"]
                )
            ]
            prompts_df = prompts_df.loc[pending].copy()
            resume_key = "bbq_item_id + prompt_hash"
        else:
            existing_ids = set(completed_df["bbq_item_id"].astype(str))
            prompts_df = prompts_df[
                ~prompts_df["bbq_item_id"].astype(str).isin(existing_ids)
            ].copy()
            resume_key = "bbq_item_id"
        print(
            f"Resume ({resume_key}): skipping {before - len(prompts_df):,} "
            f"identical completed prompts, {len(prompts_df):,} remaining"
        )

    if args.overwrite and args.output.exists():
        existing_df = None

    if len(prompts_df) == 0:
        print("Nothing to do — all items already processed.")
        return

    # ---- Dry run ----
    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(prompts_df):,} items")
        print(f"  Model: {args.model}")
        print(f"  Condition: {eval_condition}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  max_new_tokens: {max_new_tokens}")
        n_batches = (len(prompts_df) + args.batch_size - 1) // args.batch_size
        print(f"  Batches: {n_batches}")
        return

    # ---- Load model ----
    print(f"\nLoading model: {args.model}")
    # MPT/Falcon: 커스텀 코드가 최신 transformers와 비호환 → 네이티브 구현 사용
    skip_remote = _model_family(args.model) in ("mpt", "falcon")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", trust_remote_code=not skip_remote,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=not skip_remote,
    )
    model.eval()

    # ---- Run inference in batches ----
    model_family = _model_family(args.model)
    run_ts = datetime.now(timezone.utc).isoformat()
    total = len(prompts_df)
    result_rows: list[dict] = []

    prompts_list = prompts_df.to_dict("records")
    for batch_start in range(0, total, args.batch_size):
        batch = prompts_list[batch_start : batch_start + args.batch_size]
        batch_prompts = [r["prompt_text"] for r in batch]

        try:
            infer_results = run_inference_batch(
                model, tokenizer, batch_prompts, max_new_tokens, args.model,
            )
        except Exception as exc:
            infer_results = [
                {
                    "raw_output": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_seconds": 0.0,
                    "inference_ok": False,
                    "inference_error_type": type(exc).__name__,
                }
            ] * len(batch)

        for row, inf in zip(batch, infer_results):
            result_rows.append({
                "bbq_item_id": row["bbq_item_id"],
                "eval_condition": row["eval_condition"],
                "model_id": args.model,
                "model_family": model_family,
                "prompt_text": row["prompt_text"],
                "prompt_hash": row["prompt_hash"],
                "raw_output": inf["raw_output"],
                "request_id": str(uuid.uuid4()),
                "inference_ok": inf["inference_ok"],
                "inference_error_type": inf["inference_error_type"],
                "input_tokens": inf["input_tokens"],
                "output_tokens": inf["output_tokens"],
                "latency_seconds": inf["latency_seconds"],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": max_new_tokens,
                "run_timestamp": run_ts,
                "run_version": RUN_VERSION,
                # Carry forward metadata for downstream convenience
                "category": row["category"],
                "context_condition": row["context_condition"],
                "question_polarity": row["question_polarity"],
                "label": row["label"],
                "ans0": row["ans0"],
                "ans1": row["ans1"],
                "ans2": row["ans2"],
                "answer_info_json": row["answer_info_json"],
                "additional_metadata_json": row["additional_metadata_json"],
            })

        done = min(batch_start + len(batch), total)
        last_raw = infer_results[-1]["raw_output"]
        print(f"[{done:,}/{total:,}] last: {last_raw!r:.80}")

    # ---- Build output dataframe ----
    out_df = pd.DataFrame(result_rows)

    # Merge with existing results if resuming
    if existing_df is not None and len(existing_df) > 0:
        out_df = pd.concat([existing_df, out_df], ignore_index=True)
        out_df.drop_duplicates(subset=["bbq_item_id"], keep="last", inplace=True)

    # ---- Write ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)

    n_ok = out_df["inference_ok"].sum()
    n_fail = (~out_df["inference_ok"]).sum()
    print(f"\nWrote {len(out_df):,} rows → {args.output}")
    print(f"  inference_ok: {n_ok:,}  |  failed: {n_fail:,}")

    # ---- Write manifest ----
    manifest = {
        "run_version": RUN_VERSION,
        "eval_condition": eval_condition,
        "model_id": args.model,
        "model_family": model_family,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": max_new_tokens,
        "n_items": len(out_df),
        "n_inference_ok": int(n_ok),
        "n_inference_fail": int(n_fail),
        "created_at": run_ts,
        "prompt_source": str(args.input),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
