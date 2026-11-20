"""Upload JSONL and submit to OpenAI Batch API.

Saves batch metadata to a JSON file so that collect_batch_rewrite_bon.py
can retrieve the results later.

If the input JSONL exceeds --max-file-mb, it is split into chunks and
submitted as multiple batches. The meta file stores a "batches" list in
that case; collect_batch_rewrite_bon.py handles both formats.

Usage:
  python submit_batch_rewrite_bon.py --input <batch.jsonl> --batch-meta <meta.json>
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path


MAX_FILE_MB = 190  # stay safely under the 200 MB API limit


def load_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")
    return OpenAI()


def split_jsonl(path: Path, max_bytes: int) -> list[bytes]:
    """Split a JSONL file into chunks, each ≤ max_bytes."""
    chunks: list[bytes] = []
    current = io.BytesIO()
    current_size = 0
    with path.open("rb") as f:
        for line in f:
            if current_size + len(line) > max_bytes and current_size > 0:
                chunks.append(current.getvalue())
                current = io.BytesIO()
                current_size = 0
            current.write(line)
            current_size += len(line)
    if current_size > 0:
        chunks.append(current.getvalue())
    return chunks


def upload_and_submit(client, chunk: bytes, chunk_label: str) -> dict:
    print(f"  Uploading {chunk_label} ({len(chunk) / 1024 / 1024:.1f} MB) ...")
    uploaded = client.files.create(
        file=("batch_chunk.jsonl", io.BytesIO(chunk), "application/jsonl"),
        purpose="batch",
    )
    print(f"    file_id: {uploaded.id}")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"    batch_id: {batch.id}  status: {batch.status}")
    return {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "endpoint": batch.endpoint,
        "completion_window": batch.completion_window,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload JSONL and submit to OpenAI Batch API."
    )
    parser.add_argument("--input", type=Path, required=True, help="Batch input JSONL file.")
    parser.add_argument(
        "--batch-meta", type=Path, required=True,
        help="Output JSON to store batch metadata.",
    )
    parser.add_argument(
        "--max-file-mb", type=int, default=MAX_FILE_MB,
        help=f"Max upload size per batch in MB. Default: {MAX_FILE_MB}.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_meta.exists() and not args.overwrite:
        raise FileExistsError(f"{args.batch_meta} exists. Use --overwrite to replace.")

    client = load_openai_client()

    total_mb = args.input.stat().st_size / 1024 / 1024
    max_bytes = args.max_file_mb * 1024 * 1024

    if total_mb <= args.max_file_mb:
        # Single batch (original behaviour)
        print(f"Uploading {args.input} ({total_mb:.1f} MB) ...")
        with args.input.open("rb") as f:
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
            "batch_id": batch.id,
            "input_file_id": uploaded.id,
            "status": batch.status,
            "endpoint": batch.endpoint,
            "completion_window": batch.completion_window,
        }
    else:
        # Multi-batch: split and submit each chunk
        print(f"Input is {total_mb:.1f} MB — splitting into ≤{args.max_file_mb} MB chunks ...")
        chunks = split_jsonl(args.input, max_bytes)
        print(f"Split into {len(chunks)} chunks.")
        batches = []
        for i, chunk in enumerate(chunks):
            label = f"chunk {i+1}/{len(chunks)}"
            entry = upload_and_submit(client, chunk, label)
            batches.append(entry)
        meta = {"batches": batches}

    args.batch_meta.parent.mkdir(parents=True, exist_ok=True)
    args.batch_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote batch meta: {args.batch_meta}")
    print("Run collect_batch_rewrite_bon.py when the batch completes.")


if __name__ == "__main__":
    main()
