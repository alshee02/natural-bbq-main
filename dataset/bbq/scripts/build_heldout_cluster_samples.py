"""Encode heldout WildChat and assign to clusters via nearest centroid (cosine).

Outputs one parquet per cluster to --output-dir.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WILDCHAT_PATH = PROJECT_ROOT / "data/interim/wildchat_filtered.parquet"
CENTROIDS_PATH = PROJECT_ROOT / "data/processed/wildchat_clusters/k_006/centroids_normalized.npy"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/wildchat_heldout_by_cluster"
MODEL_NAME = "StyleDistance/styledistance"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Assign heldout WildChat texts to clusters.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Max texts to keep per cluster (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wildchat = pd.read_parquet(WILDCHAT_PATH)
    heldout = wildchat[wildchat["data_split"] == "heldout"].reset_index(drop=True)
    print(f"Heldout texts: {len(heldout):,}")

    centroids = np.load(CENTROIDS_PATH)  # (k, d), L2-normalized
    k = len(centroids)
    print(f"Clusters: {k}")

    model = SentenceTransformer(MODEL_NAME)
    print("Encoding heldout texts...")
    emb = model.encode(
        heldout["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256,
    )

    # argmax cosine similarity == argmax dot product for normalized vectors
    cluster_ids = np.argmax(emb @ centroids.T, axis=1)
    heldout = heldout.copy()
    heldout["cluster_id"] = cluster_ids

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cid in range(k):
        subset = heldout[heldout["cluster_id"] == cid].reset_index(drop=True)
        if args.sample is not None:
            subset = subset.sample(n=min(args.sample, len(subset)), random_state=args.seed)
        out = args.output_dir / f"cluster_{cid:03d}.parquet"
        subset[["prompt_id", "text", "cluster_id"]].to_parquet(out, index=False)
        print(f"  cluster {cid:03d}: {len(subset):,} texts → {out.name}")


if __name__ == "__main__":
    main()
