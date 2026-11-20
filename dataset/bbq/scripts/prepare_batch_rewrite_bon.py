"""Convert BON rewrite inputs parquet to OpenAI Batch API JSONL.

Each line is one Chat Completions request:
  custom_id == rewrite_id   (used to join results back to inputs in collect step)

Usage:
  python prepare_batch_rewrite_bon.py --input <inputs.parquet> --output <batch.jsonl>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


def make_batch_request(
    row: pd.Series,
    model: str,
    max_tokens: int,
    temperature: float | None,
) -> dict:
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": str(row["system_prompt"])},
            {"role": "user", "content": str(row["user_prompt"])},
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
        description="Prepare OpenAI Batch API JSONL from BON rewrite inputs parquet."
    )
    parser.add_argument("--input", type=Path, required=True, help="BON inputs parquet.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL for Batch API.")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. Default: None (model default).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists. Use --overwrite to replace.")

    df = pd.read_parquet(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            req = make_batch_request(row, args.model, args.max_output_tokens, args.temperature)
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df):,} requests → {args.output}")
    print(f"  model:            {args.model}")
    print(f"  max_output_tokens: {args.max_output_tokens}")
    print(f"  temperature:       {args.temperature}")


if __name__ == "__main__":
    main()
