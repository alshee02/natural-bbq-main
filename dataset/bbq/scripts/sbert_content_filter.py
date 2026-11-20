from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data/processed/rewrite_batch/bbq_rewrite_hard_checks.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/rewrite_batch/bbq_rewrite_sbert_filtered.parquet"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.75


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = df[df["hard_check_status"] == "pass"].reset_index(drop=True)

    original = df["original_context"] + " " + df["original_question"]
    rewritten = df["natural_question"]

    model = SentenceTransformer(MODEL_NAME)
    original_emb = model.encode(original.tolist(), normalize_embeddings=True)
    rewritten_emb = model.encode(rewritten.tolist(), normalize_embeddings=True)

    df["sbert_similarity"] = np.sum(original_emb * rewritten_emb, axis=1)
    df["content_check_status"] = np.where(
        df["sbert_similarity"] >= SIMILARITY_THRESHOLD, "pass", "fail"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(df["content_check_status"].value_counts())
    print(df["sbert_similarity"].describe())


if __name__ == "__main__":
    main()
