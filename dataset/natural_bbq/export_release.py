#!/usr/bin/env python3
"""Export the public Natural-BBQ release (JSONL) from the pipeline output.

Selects the public fields from the (privacy-cleaned) paired parquet, parses the
embedded JSON annotation columns into nested objects, and writes one JSON object
per line. Internal pipeline diagnostics (candidate bookkeeping, per-bank style
variants, hard/content check statuses, etc.) are dropped.

Usage
-----
  python dataset/natural_bbq/export_release.py \
      --input  data/processed/rewrite_output_bon/natural_bbq_paired.parquet \
      --output dataset/natural_bbq/natural_bbq.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Ordered public fields. Columns absent in the source are skipped with a warning.
PUBLIC_FIELDS = [
    "bbq_item_id",         # unique item id
    "example_id",          # original BBQ example_id
    "category",            # BBQ category
    "context_condition",   # ambig / disambig
    "question_polarity",   # neg / nonneg
    "question_index",
    "original_context",
    "original_question",
    "natural_question",    # the Natural-BBQ rewrite
    "ans0", "ans1", "ans2",
    "label",               # gold answer index (0/1/2)
    "target_cluster_id",   # assigned WildChat style cluster
    "rewrite_model",
    "fallback_used",
    "sbert_similarity",    # content-preservation diagnostic
    "delta_style",         # matched-body held-out style shift diagnostic
]

# JSON-string columns -> parsed nested objects under a cleaned key.
JSON_FIELDS = {
    "answer_info_json": "answer_info",
    "additional_metadata_json": "additional_metadata",
}


def _clean(v):
    if hasattr(v, "item"):  # numpy scalar -> python scalar
        v = v.item()
    if isinstance(v, float):
        return round(v, 6)
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        default="data/processed/rewrite_output_bon/natural_bbq_paired.parquet",
    )
    ap.add_argument("--output", default="dataset/natural_bbq/natural_bbq.jsonl")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    cols = [c for c in PUBLIC_FIELDS if c in df.columns]
    missing = [c for c in PUBLIC_FIELDS if c not in df.columns]
    if missing:
        print(f"[warn] missing public fields (skipped): {missing}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {c: _clean(row[c]) for c in cols}
            for src, dst in JSON_FIELDS.items():
                if src in df.columns:
                    try:
                        rec[dst] = json.loads(row[src]) if row[src] else None
                    except (TypeError, ValueError):
                        rec[dst] = row[src]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"[ok] wrote {n} paired items -> {out}")


if __name__ == "__main__":
    main()
