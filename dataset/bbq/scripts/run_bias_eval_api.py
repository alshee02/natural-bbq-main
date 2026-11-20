"""BBQ bias evaluation — API-based inference for closed-source models.

Supports: OpenAI (GPT-*), Anthropic (Claude), Google (Gemini)

Modes:
  realtime (default) — concurrent async API calls
  batch              — submit batch job, poll for completion (50% cheaper)

Usage
-----
  # Realtime
  python dataset/bbq/scripts/run_bias_eval_api.py \
      --input  data/processed/bias_eval_v2_matched_body/prompts/prompts_A_original_mcq.parquet \
      --model  gemini-3.5-flash --concurrency 30

  # Batch (OpenAI / Anthropic only, 50% cheaper)
  python dataset/bbq/scripts/run_bias_eval_api.py \
      --input  data/processed/bias_eval_v2_matched_body/prompts/prompts_A_original_mcq.parquet \
      --model  gpt-4.1 --batch

  # Check / collect batch results
  python dataset/bbq/scripts/run_bias_eval_api.py \
      --batch-collect data/processed/bias_eval_v2_matched_body/raw_predictions/gpt_4_1/batch_A.json

Environment variables:
  OPENAI_API_KEY     — for GPT models
  ANTHROPIC_API_KEY  — for Claude models
  GOOGLE_API_KEY     — for Gemini models
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_CONDITION_LETTER = {
    "A_original_mcq": "A",
    "B_natural_mcq": "B",
}

_DEFAULT_MAX_TOKENS = {
    "A_original_mcq": 16,
    "B_natural_mcq": 16,
}

RUN_VERSION = "bias_eval_v2_matched_body"
MAX_RETRIES = 6
RETRY_BASE_DELAY = 2.0
SAVE_EVERY = 2000  # save checkpoint every N items
BATCH_POLL_INTERVAL = 60  # seconds between batch status checks


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(model: str) -> str:
    lower = model.lower()
    if any(x in lower for x in ("gpt", "o1-", "o3-", "o4-")):
        return "openai"
    if "claude" in lower:
        return "anthropic"
    if "gemini" in lower:
        return "google"
    raise ValueError(f"Cannot detect provider for model: {model!r}. "
                     "Use --provider to specify explicitly.")


def _model_family(model_id: str) -> str:
    lower = model_id.lower()
    if "gpt" in lower:
        return "gpt"
    if "claude" in lower:
        return "claude"
    if "gemini" in lower:
        return "gemini"
    return "other"


def _model_short(model_id: str) -> str:
    name = model_id.split("/")[-1]
    return name.lower().replace("-", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# API call wrappers with retry
# ---------------------------------------------------------------------------

def _error_result(exc_type: str) -> dict:
    return {
        "raw_output": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": 0.0,
        "inference_ok": False,
        "inference_error_type": exc_type,
    }


def _is_retryable(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(x in msg for x in (
        "rate", "429", "500", "502", "503", "529",
        "timeout", "overload", "capacity", "busy",
    ))


async def call_openai(client, model, prompt, max_tokens, temperature, sem):
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
                elapsed = time.perf_counter() - t0
                return {
                    "raw_output": (resp.choices[0].message.content or "").strip(),
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                    "latency_seconds": round(elapsed, 4),
                    "inference_ok": True,
                    "inference_error_type": None,
                }
            except Exception as exc:
                if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                    return _error_result(type(exc).__name__)
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    return _error_result("max_retries_exceeded")


async def call_anthropic(client, model, prompt, max_tokens, temperature, sem):
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                kwargs = dict(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    thinking={"type": "disabled"},
                )
                # Claude Sonnet 5+ deprecated temperature
                if temperature and "sonnet-5" not in model and "fable" not in model and "opus-5" not in model:
                    kwargs["temperature"] = temperature
                resp = await client.messages.create(**kwargs)
                elapsed = time.perf_counter() - t0
                raw = resp.content[0].text.strip() if resp.content else ""
                return {
                    "raw_output": raw,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "latency_seconds": round(elapsed, 4),
                    "inference_ok": True,
                    "inference_error_type": None,
                }
            except Exception as exc:
                if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                    return _error_result(type(exc).__name__)
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    return _error_result("max_retries_exceeded")


async def call_google(client, model, prompt, max_tokens, temperature, sem):
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                from google.genai import types
                t0 = time.perf_counter()
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=0,
                        ),
                        safety_settings=[
                            types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH",
                                threshold="OFF",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="OFF",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT",
                                threshold="OFF",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="OFF",
                            ),
                        ],
                    ),
                )
                elapsed = time.perf_counter() - t0
                raw = ""
                block_reason = None
                try:
                    raw = resp.text.strip() if resp.text else ""
                except (ValueError, AttributeError):
                    raw = ""
                # Check block reason
                if not raw:
                    if hasattr(resp, "prompt_feedback") and resp.prompt_feedback:
                        br = getattr(resp.prompt_feedback, "block_reason", None)
                        block_reason = str(br) if br else None
                    if not block_reason and hasattr(resp, "candidates") and resp.candidates:
                        fr = getattr(resp.candidates[0], "finish_reason", None)
                        if fr and str(fr) != "STOP":
                            block_reason = str(fr)
                usage = getattr(resp, "usage_metadata", None)
                error_type = None
                if not raw:
                    error_type = f"blocked:{block_reason}" if block_reason else "empty_response"
                return {
                    "raw_output": raw,
                    "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                    "latency_seconds": round(elapsed, 4),
                    "inference_ok": bool(raw),
                    "inference_error_type": error_type,
                }
            except Exception as exc:
                if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                    return _error_result(type(exc).__name__)
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    return _error_result("max_retries_exceeded")


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def create_client(provider: str, sync: bool = False):
    if provider == "openai":
        if sync:
            from openai import OpenAI
            return OpenAI()
        from openai import AsyncOpenAI
        return AsyncOpenAI()
    elif provider == "anthropic":
        if sync:
            import anthropic
            return anthropic.Anthropic()
        from anthropic import AsyncAnthropic
        return AsyncAnthropic()
    elif provider == "google":
        from google import genai
        return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Batch mode — OpenAI
# ---------------------------------------------------------------------------

def _build_openai_batch_jsonl(
    items: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> list[str]:
    """Build JSONL lines for OpenAI Batch API."""
    lines = []
    for item in items:
        req = {
            "custom_id": item["bbq_item_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": item["prompt_text"]}],
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
            },
        }
        lines.append(json.dumps(req, ensure_ascii=False))
    return lines


OPENAI_BATCH_LIMIT = 49_999  # OpenAI max 50,000 per batch


def submit_openai_batch(
    items: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    metadata_path: Path,
) -> dict:
    """Submit OpenAI batch job(s). Splits into chunks if >50k items."""
    from openai import OpenAI
    import io
    client = OpenAI()

    # Split into chunks if needed
    chunks = [items[i:i + OPENAI_BATCH_LIMIT]
              for i in range(0, len(items), OPENAI_BATCH_LIMIT)]

    batch_ids = []
    file_ids = []
    submitted_at = datetime.now(timezone.utc).isoformat()

    for ci, chunk in enumerate(chunks):
        suffix = f" (part {ci+1}/{len(chunks)})" if len(chunks) > 1 else ""
        lines = _build_openai_batch_jsonl(chunk, model, max_tokens, temperature)
        jsonl_content = "\n".join(lines).encode("utf-8")

        file_obj = client.files.create(
            file=io.BytesIO(jsonl_content),
            purpose="batch",
        )
        print(f"  Uploaded input file{suffix}: {file_obj.id} ({len(lines):,} requests)")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  Batch created{suffix}: {batch.id}  status={batch.status}")

        batch_ids.append(batch.id)
        file_ids.append(file_obj.id)

    meta = {
        "provider": "openai",
        "batch_ids": batch_ids,
        "batch_id": batch_ids[0],  # backward compat
        "input_file_ids": file_ids,
        "n_parts": len(chunks),
        "model": model,
        "n_requests": len(items),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "submitted_at": submitted_at,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Metadata saved: {metadata_path}")
    return meta


def collect_openai_batch(metadata_path: Path, items_lookup: dict[str, dict]) -> list[dict] | None:
    """Poll and collect OpenAI batch results. Returns None if not all parts ready."""
    from openai import OpenAI
    client = OpenAI()

    meta = json.loads(metadata_path.read_text())
    batch_ids = meta.get("batch_ids", [meta["batch_id"]])

    all_results = []
    all_done = True

    for bi, batch_id in enumerate(batch_ids):
        suffix = f" (part {bi+1}/{len(batch_ids)})" if len(batch_ids) > 1 else ""
        batch = client.batches.retrieve(batch_id)

        print(f"  Batch {batch.id}{suffix}: {batch.status}")
        if batch.request_counts:
            print(f"    completed={batch.request_counts.completed} "
                  f"failed={batch.request_counts.failed} "
                  f"total={batch.request_counts.total}")

        if batch.status not in ("completed", "failed", "expired", "cancelled"):
            all_done = False
            continue

        if batch.status != "completed":
            print(f"  Batch ended with status: {batch.status}")
            if batch.errors and batch.errors.data:
                for err in batch.errors.data[:5]:
                    print(f"    Error: {err.code} — {err.message}")
            continue

        # Download results for this part
        output_file = client.files.content(batch.output_file_id)
        for line in output_file.text.strip().split("\n"):
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            item = items_lookup.get(custom_id, {})
            resp = obj.get("response", {})
            body = resp.get("body", {})

            if resp.get("status_code") == 200 and body.get("choices"):
                raw = (body["choices"][0]["message"]["content"] or "").strip()
                usage = body.get("usage", {})
                all_results.append({
                    **item,
                    "raw_output": raw,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "inference_ok": True,
                    "inference_error_type": None,
                })
            else:
                err = body.get("error", {})
                all_results.append({
                    **item,
                    "raw_output": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "inference_ok": False,
                    "inference_error_type": err.get("code", "batch_error"),
                })

    if not all_done:
        return None

    return all_results


# ---------------------------------------------------------------------------
# Batch mode — Anthropic
# ---------------------------------------------------------------------------

def _build_anthropic_batch_requests(
    items: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    """Build request dicts for Anthropic Message Batches API."""
    requests = []
    for item in items:
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": item["prompt_text"]}],
            "thinking": {"type": "disabled"},
        }
        # Claude Sonnet 5+ deprecated temperature
        if temperature and "sonnet-5" not in model and "fable" not in model and "opus-5" not in model:
            params["temperature"] = temperature
        requests.append({
            "custom_id": item["bbq_item_id"],
            "params": params,
        })
    return requests


def submit_anthropic_batch(
    items: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    metadata_path: Path,
) -> dict:
    """Submit an Anthropic message batch. Returns batch metadata."""
    import anthropic
    client = anthropic.Anthropic()

    requests = _build_anthropic_batch_requests(items, model, max_tokens, temperature)

    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch created: {batch.id}")
    print(f"  Status: {batch.processing_status}")

    meta = {
        "provider": "anthropic",
        "batch_id": batch.id,
        "model": model,
        "n_requests": len(requests),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": batch.processing_status,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Metadata saved: {metadata_path}")
    return meta


def collect_anthropic_batch(metadata_path: Path, items_lookup: dict[str, dict]) -> list[dict] | None:
    """Poll and collect Anthropic batch results. Returns None if not ready."""
    import anthropic
    client = anthropic.Anthropic()

    meta = json.loads(metadata_path.read_text())
    batch = client.messages.batches.retrieve(meta["batch_id"])

    print(f"  Batch {batch.id}: {batch.processing_status}")
    counts = batch.request_counts
    print(f"    processing={counts.processing} succeeded={counts.succeeded} "
          f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}")

    if batch.processing_status != "ended":
        return None

    # Stream results
    results = []
    for entry in client.messages.batches.results(meta["batch_id"]):
        custom_id = entry.custom_id
        item = items_lookup.get(custom_id, {})

        if entry.result.type == "succeeded":
            msg = entry.result.message
            raw = msg.content[0].text.strip() if msg.content else ""
            results.append({
                **item,
                "raw_output": raw,
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "inference_ok": True,
                "inference_error_type": None,
            })
        else:
            results.append({
                **item,
                "raw_output": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "inference_ok": False,
                "inference_error_type": entry.result.type,
            })

    return results


# ---------------------------------------------------------------------------
# Batch mode — Google
# ---------------------------------------------------------------------------

GOOGLE_BATCH_INLINE_LIMIT = 10_000  # keep inline requests under 20MB


def submit_google_batch(
    items: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    metadata_path: Path,
) -> dict:
    """Submit Google batch job(s). Splits into chunks if needed."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))

    chunks = [items[i:i + GOOGLE_BATCH_INLINE_LIMIT]
              for i in range(0, len(items), GOOGLE_BATCH_INLINE_LIMIT)]

    batch_names = []
    submitted_at = datetime.now(timezone.utc).isoformat()

    for ci, chunk in enumerate(chunks):
        suffix = f" (part {ci+1}/{len(chunks)})" if len(chunks) > 1 else ""

        inline_requests = []
        for item in chunk:
            inline_requests.append({
                "contents": [{
                    "parts": [{"text": item["prompt_text"]}],
                    "role": "user",
                }],
            })

        batch_job = client.batches.create(
            model=model,
            src=inline_requests,
            config={
                "display_name": f"bbq_eval_{ci}",
            },
        )
        print(f"  Batch created{suffix}: {batch_job.name}  state={batch_job.state.name}")
        batch_names.append(batch_job.name)

    # Save custom_id mapping (Google batch doesn't have custom_id)
    item_ids_by_part = []
    for ci, chunk in enumerate(chunks):
        item_ids_by_part.append([item["bbq_item_id"] for item in chunk])

    meta = {
        "provider": "google",
        "batch_names": batch_names,
        "item_ids_by_part": item_ids_by_part,
        "n_parts": len(chunks),
        "model": model,
        "n_requests": len(items),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "submitted_at": submitted_at,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Metadata saved: {metadata_path}")
    return meta


def collect_google_batch(metadata_path: Path, items_lookup: dict[str, dict]) -> list[dict] | None:
    """Poll and collect Google batch results. Returns None if not all parts ready."""
    from google import genai

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    meta = json.loads(metadata_path.read_text())
    batch_names = meta["batch_names"]
    item_ids_by_part = meta["item_ids_by_part"]

    completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    all_results = []
    all_done = True

    for pi, batch_name in enumerate(batch_names):
        suffix = f" (part {pi+1}/{len(batch_names)})" if len(batch_names) > 1 else ""
        batch_job = client.batches.get(name=batch_name)

        print(f"  Batch {batch_name}{suffix}: {batch_job.state.name}")

        if batch_job.state.name not in completed_states:
            all_done = False
            continue

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"  Batch ended with state: {batch_job.state.name}")
            # Mark all items in this part as failed
            for item_id in item_ids_by_part[pi]:
                item = items_lookup.get(item_id, {})
                all_results.append({
                    **item,
                    "raw_output": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "inference_ok": False,
                    "inference_error_type": f"batch_{batch_job.state.name}",
                })
            continue

        # Collect results
        part_item_ids = item_ids_by_part[pi]
        if batch_job.dest and batch_job.dest.inlined_responses:
            for i, inline_resp in enumerate(batch_job.dest.inlined_responses):
                item_id = part_item_ids[i] if i < len(part_item_ids) else f"unknown_{i}"
                item = items_lookup.get(item_id, {})

                if inline_resp.response:
                    raw = ""
                    try:
                        raw = inline_resp.response.text.strip() if inline_resp.response.text else ""
                    except (ValueError, AttributeError):
                        raw = ""
                    usage = getattr(inline_resp.response, "usage_metadata", None)
                    all_results.append({
                        **item,
                        "raw_output": raw,
                        "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                        "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                        "inference_ok": bool(raw),
                        "inference_error_type": None if raw else "empty_response",
                    })
                else:
                    all_results.append({
                        **item,
                        "raw_output": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "inference_ok": False,
                        "inference_error_type": str(getattr(inline_resp, "error", "unknown")),
                    })

    if not all_done:
        return None

    return all_results


# ---------------------------------------------------------------------------
# Async batch processing
# ---------------------------------------------------------------------------

async def process_all(
    items: list[dict],
    client,
    model: str,
    provider: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    output_path: Path,
    existing_df: pd.DataFrame | None,
    model_family: str,
    run_ts: str,
    chunk_size: int = 500,
) -> list[dict]:
    """Process all items in chunks, saving checkpoints periodically."""
    sem = asyncio.Semaphore(concurrency)
    call_fn = {
        "openai": call_openai,
        "anthropic": call_anthropic,
        "google": call_google,
    }[provider]

    all_results: list[dict] = []
    total = len(items)

    for start in range(0, total, chunk_size):
        chunk = items[start : start + chunk_size]
        tasks = [
            call_fn(client, model, item["prompt_text"], max_tokens, temperature, sem)
            for item in chunk
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for item, r in zip(chunk, results):
            if isinstance(r, Exception):
                inf = _error_result(type(r).__name__)
            else:
                inf = r
            all_results.append(_build_result_row(
                item, inf, model, model_family, max_tokens, temperature, run_ts,
            ))

        done = min(start + len(chunk), total)
        n_ok = sum(1 for r in all_results if r.get("inference_ok"))
        last_raw = all_results[-1]["raw_output"]
        print(f"  [{done:,}/{total:,}] ok={n_ok:,}  last: {last_raw!r:.60}")

        # Periodic checkpoint
        if done % SAVE_EVERY < chunk_size or done == total:
            _save_checkpoint(all_results, existing_df, output_path)

    return all_results


def _save_checkpoint(results: list[dict], existing_df: pd.DataFrame | None, path: Path):
    df = pd.DataFrame(results)
    if existing_df is not None and len(existing_df) > 0:
        df = pd.concat([existing_df, df], ignore_index=True)
        df.drop_duplicates(subset=["bbq_item_id"], keep="last", inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_result_row(item: dict, inf: dict, model: str, model_family: str,
                      max_tokens: int, temperature: float, run_ts: str) -> dict:
    """Build a unified result row from item metadata + inference result."""
    return {
        "bbq_item_id": item["bbq_item_id"],
        "eval_condition": item["eval_condition"],
        "model_id": model,
        "model_family": model_family,
        "prompt_text": item["prompt_text"],
        "prompt_hash": item["prompt_hash"],
        "raw_output": inf.get("raw_output", ""),
        "request_id": str(uuid.uuid4()),
        "inference_ok": inf.get("inference_ok", False),
        "inference_error_type": inf.get("inference_error_type"),
        "input_tokens": inf.get("input_tokens", 0),
        "output_tokens": inf.get("output_tokens", 0),
        "latency_seconds": inf.get("latency_seconds", 0.0),
        "temperature": temperature,
        "top_p": 1.0,
        "max_new_tokens": max_tokens,
        "run_timestamp": run_ts,
        "run_version": RUN_VERSION,
        "category": item["category"],
        "context_condition": item["context_condition"],
        "question_polarity": item["question_polarity"],
        "label": item["label"],
        "ans0": item["ans0"],
        "ans1": item["ans1"],
        "ans2": item["ans2"],
        "answer_info_json": item["answer_info_json"],
        "additional_metadata_json": item["additional_metadata_json"],
    }


def _save_final(results: list[dict], existing_df: pd.DataFrame | None,
                output_path: Path, model: str, model_family: str,
                provider: str, eval_condition: str, max_tokens: int,
                temperature: float, concurrency: int, run_ts: str,
                prompt_source: str):
    """Save final parquet + manifest."""
    out_df = pd.DataFrame(results)
    if existing_df is not None and len(existing_df) > 0:
        out_df = pd.concat([existing_df, out_df], ignore_index=True)
        out_df.drop_duplicates(subset=["bbq_item_id"], keep="last", inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    n_ok = int(out_df["inference_ok"].sum())
    n_fail = int((~out_df["inference_ok"]).sum())
    total_in = int(out_df["input_tokens"].sum())
    total_out = int(out_df["output_tokens"].sum())

    print(f"\nWrote {len(out_df):,} rows → {output_path}")
    print(f"  inference_ok: {n_ok:,}  |  failed: {n_fail:,}")
    print(f"  Tokens: {total_in:,} input, {total_out:,} output")

    manifest = {
        "run_version": RUN_VERSION,
        "eval_condition": eval_condition,
        "model_id": model,
        "model_family": model_family,
        "provider": provider,
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "concurrency": concurrency,
        "n_items": len(out_df),
        "n_inference_ok": n_ok,
        "n_inference_fail": n_fail,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "created_at": run_ts,
        "prompt_source": prompt_source,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Manifest: {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run BBQ bias evaluation via API for closed-source models.",
    )
    ap.add_argument("--input", type=Path, default=None,
                    help="Prompt parquet (e.g. prompts_A_original_mcq.parquet)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output parquet path (default: auto-generated)")
    ap.add_argument("--model", default=None,
                    help="Model ID (e.g. gpt-4.1, claude-sonnet-5-20250801, gemini-3.5-flash)")
    ap.add_argument("--provider", default=None, choices=["openai", "anthropic", "google"],
                    help="Override auto-detected provider")
    ap.add_argument("--concurrency", type=int, default=20,
                    help="Max concurrent API requests (default: 20)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Skip items already in existing output")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    # Batch mode
    ap.add_argument("--batch", action="store_true",
                    help="Submit as batch job (50%% cheaper, OpenAI/Anthropic only)")
    ap.add_argument("--batch-collect", type=Path, default=None,
                    help="Collect results from a previously submitted batch "
                         "(pass the batch metadata JSON path)")
    ap.add_argument("--batch-poll", action="store_true",
                    help="With --batch: poll until batch completes instead of returning")
    args = ap.parse_args()

    # =================================================================
    # Batch collect mode (standalone)
    # =================================================================
    if args.batch_collect:
        meta = json.loads(args.batch_collect.read_text())
        provider = meta["provider"]
        model = meta["model"]
        print(f"Collecting batch results: {args.batch_collect}")
        print(f"  Provider: {provider}, Model: {model}")

        # Need input parquet to reconstruct full rows
        if args.input is None:
            print("ERROR: --input is required with --batch-collect to rebuild full result rows.")
            sys.exit(1)

        prompts_df = pd.read_parquet(args.input)
        items = prompts_df.to_dict("records")
        items_lookup = {item["bbq_item_id"]: item for item in items}

        if provider == "openai":
            batch_results = collect_openai_batch(args.batch_collect, items_lookup)
        elif provider == "anthropic":
            batch_results = collect_anthropic_batch(args.batch_collect, items_lookup)
        elif provider == "google":
            batch_results = collect_google_batch(args.batch_collect, items_lookup)
        else:
            print(f"ERROR: Batch collect not supported for provider: {provider}")
            sys.exit(1)

        if batch_results is None:
            print("\n  Batch not yet complete. Run again later.")
            return

        if not batch_results:
            print("\n  Batch returned no results (failed/expired).")
            return

        # Build final result rows
        eval_condition = prompts_df["eval_condition"].iloc[0]
        if eval_condition not in _CONDITION_LETTER:
            raise ValueError(
                f"Unsupported condition {eval_condition!r}; expected "
                "A_original_mcq or B_natural_mcq."
            )
        cond_letter = _CONDITION_LETTER[eval_condition]
        model_family = _model_family(model)
        run_ts = meta.get("submitted_at", datetime.now(timezone.utc).isoformat())

        if args.output is None:
            model_dir = _model_short(model)
            args.output = (
                PROJECT_ROOT / "data/processed/bias_eval_v2_matched_body/raw_predictions"
                / model_dir / f"predictions_{cond_letter}.parquet"
            )

        final_rows = []
        for r in batch_results:
            item = items_lookup.get(r.get("bbq_item_id", ""), r)
            final_rows.append(_build_result_row(
                item, r, model, model_family,
                meta.get("max_tokens", 16), meta.get("temperature", 0.0), run_ts,
            ))

        # Batch collection may target a copy of the prior full prediction
        # file. Preserve unchanged rows and replace only collected item IDs.
        existing_df = (
            pd.read_parquet(args.output) if args.output.exists() else None
        )
        _save_final(
            final_rows, existing_df, args.output, model, model_family, provider,
            eval_condition, meta.get("max_tokens", 16), meta.get("temperature", 0.0),
            0, run_ts, str(args.input),
        )
        print(f"\n  Collected {len(final_rows):,} results.")
        return

    # =================================================================
    # Normal mode: requires --input and --model
    # =================================================================
    if not args.input or not args.model:
        ap.error("--input and --model are required (unless using --batch-collect)")

    # ---- Provider ----
    provider = args.provider or _detect_provider(args.model)
    print(f"Provider: {provider}")

    # ---- Load prompts ----
    prompts_df = pd.read_parquet(args.input)
    eval_condition = prompts_df["eval_condition"].iloc[0]
    if eval_condition not in _CONDITION_LETTER:
        raise ValueError(
            f"Unsupported condition {eval_condition!r}; expected "
            "A_original_mcq or B_natural_mcq."
        )
    cond_letter = _CONDITION_LETTER[eval_condition]
    print(f"Condition: {eval_condition} ({cond_letter})")
    print(f"Items loaded: {len(prompts_df):,}")

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else _DEFAULT_MAX_TOKENS[eval_condition]
    )

    # ---- Output path ----
    if args.output is None:
        model_dir = _model_short(args.model)
        args.output = (
            PROJECT_ROOT / "data/processed/bias_eval_v2_matched_body/raw_predictions"
            / model_dir / f"predictions_{cond_letter}.parquet"
        )
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"max_new_tokens: {max_new_tokens}")

    # ---- Limit ----
    if args.limit:
        prompts_df = prompts_df.head(args.limit).copy()
        print(f"Limited to {len(prompts_df):,} items")

    # ---- Resume: skip only identical item-prompt pairs ----
    # Candidate reselection can change a prompt without changing bbq_item_id.
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

    items = prompts_df.to_dict("records")

    # ---- Dry run ----
    if args.dry_run:
        mode = "batch (50% off)" if args.batch else "realtime"
        print(f"\n[DRY RUN] Would process {len(prompts_df):,} items ({mode})")
        print(f"  Provider: {provider}")
        print(f"  Model: {args.model}")
        avg_input = 135
        avg_output = 10
        n = len(prompts_df)
        total_in = n * avg_input / 1_000_000
        total_out = n * avg_output / 1_000_000
        prices_rt = {
            "openai": (2.0, 8.0),
            "anthropic": (3.0, 15.0),
            "google": (0.30, 2.50),
        }
        pi, po = prices_rt.get(provider, (0, 0))
        discount = 0.5 if args.batch else 1.0
        est = (total_in * pi + total_out * po) * discount
        print(f"  Est. tokens: ~{total_in:.1f}M in, ~{total_out:.2f}M out")
        print(f"  Est. cost: ~${est:.2f}")
        return

    # =================================================================
    # Batch submit mode
    # =================================================================
    if args.batch:
        if provider not in ("openai", "anthropic", "google"):
            print(f"ERROR: Batch mode only supports openai/anthropic/google, not {provider}.")
            sys.exit(1)

        batch_meta_path = args.output.parent / f"batch_{cond_letter}.json"
        print(f"\nSubmitting batch ({len(items):,} items)...")

        if provider == "openai":
            meta = submit_openai_batch(
                items, args.model, max_new_tokens, args.temperature, batch_meta_path,
            )
        elif provider == "anthropic":
            meta = submit_anthropic_batch(
                items, args.model, max_new_tokens, args.temperature, batch_meta_path,
            )
        else:
            meta = submit_google_batch(
                items, args.model, max_new_tokens, args.temperature, batch_meta_path,
            )

        if args.batch_poll:
            print(f"\nPolling for completion (checking every {BATCH_POLL_INTERVAL}s)...")
            items_lookup = {item["bbq_item_id"]: item for item in items}
            while True:
                time.sleep(BATCH_POLL_INTERVAL)
                if provider == "openai":
                    batch_results = collect_openai_batch(batch_meta_path, items_lookup)
                elif provider == "anthropic":
                    batch_results = collect_anthropic_batch(batch_meta_path, items_lookup)
                else:
                    batch_results = collect_google_batch(batch_meta_path, items_lookup)

                if batch_results is not None:
                    break

            if batch_results:
                model_family = _model_family(args.model)
                run_ts = meta.get("submitted_at", datetime.now(timezone.utc).isoformat())
                final_rows = [
                    _build_result_row(
                        items_lookup.get(r.get("bbq_item_id", ""), r), r,
                        args.model, model_family, max_new_tokens,
                        args.temperature, run_ts,
                    )
                    for r in batch_results
                ]
                _save_final(
                    final_rows, existing_df, args.output, args.model,
                    model_family, provider, eval_condition, max_new_tokens,
                    args.temperature, 0, run_ts, str(args.input),
                )
            else:
                print("Batch failed — no results.")
        else:
            print(f"\nBatch submitted. To collect results later:")
            print(f"  python {__file__} \\")
            print(f"      --batch-collect {batch_meta_path} \\")
            print(f"      --input {args.input}")
        return

    # =================================================================
    # Realtime mode
    # =================================================================
    print(f"Concurrency: {args.concurrency}")
    print(f"\nInitializing {provider} client...")
    client = create_client(provider)

    model_family = _model_family(args.model)
    run_ts = datetime.now(timezone.utc).isoformat()

    print(f"\nStarting inference ({len(items):,} items)...\n")
    all_results = asyncio.run(process_all(
        items=items,
        client=client,
        model=args.model,
        provider=provider,
        max_tokens=max_new_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        output_path=args.output,
        existing_df=existing_df,
        model_family=model_family,
        run_ts=run_ts,
    ))

    _save_final(
        all_results, existing_df, args.output, args.model, model_family,
        provider, eval_condition, max_new_tokens, args.temperature,
        args.concurrency, run_ts, str(args.input),
    )


if __name__ == "__main__":
    main()
