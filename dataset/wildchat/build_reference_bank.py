from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Regex to detect conversation-format texts (e.g. "User: ... Assistant: ...")
_CONV_PATTERN = re.compile(r"\b(User|Assistant|Human|AI)\s*:", re.IGNORECASE)
# Detect redacted content in text body
_REDACTED_PATTERN = re.compile(r"\[redacted\]|<redacted>|REDACTED", re.IGNORECASE)


def is_clean_reference(text: str) -> bool:
    """Return True if text is a clean single user prompt (not a conversation log, not redacted)."""
    return not _CONV_PATTERN.search(text) and not _REDACTED_PATTERN.search(text)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CLUSTERS = (
    PROJECT_ROOT / "data/processed/wildchat_clusters/final/wildchat_clusters.parquet"
)
DEFAULT_EMBEDDINGS = (
    PROJECT_ROOT / "embeddings/wildchat_style/reference/embeddings.npy"
)
DEFAULT_BBQ_ASSIGNMENTS = (
    PROJECT_ROOT / "data/processed/bbq_cluster_assignments.parquet"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/reference_bank.parquet"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "data/processed/reference_bank.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data/processed/reference_bank.summary.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports/reference_bank/reference_bank_examples.md"
DEFAULT_CLUSTER_OUTPUT_DIR = PROJECT_ROOT / "data/processed/reference_bank_by_cluster"
DEFAULT_UNUSED_OUTPUT_DIR = PROJECT_ROOT / "data/processed/reference_bank_unused_by_cluster"

OUTPUT_COLUMNS = [
    "reference_id",
    "cluster_id",
    "cluster_size",
    "cluster_pi",
    "bank_size",
    "sample_type",
    "sample_rank",
    "selection_seed",
    "random_score",
    "embedding_index",
    "prompt_id",
    "text_hash",
    "conversation_id",
    "data_split",
    "country",
    "model",
    "timestamp",
    "cosine_distance_to_centroid",
    "text",
]


@dataclass(frozen=True)
class Config:
    clusters_path: Path = DEFAULT_CLUSTERS
    embeddings_path: Path = DEFAULT_EMBEDDINGS
    bbq_assignments_path: Path = DEFAULT_BBQ_ASSIGNMENTS
    output_path: Path = DEFAULT_OUTPUT
    csv_output_path: Path | None = DEFAULT_CSV_OUTPUT
    summary_path: Path = DEFAULT_SUMMARY
    report_path: Path = DEFAULT_REPORT
    cluster_output_dir: Path | None = DEFAULT_CLUSTER_OUTPUT_DIR
    unused_output_dir: Path | None = DEFAULT_UNUSED_OUTPUT_DIR
    # Adaptive bank size: M_c = min(|C_c|, max(min_bank_size, ceil(N_c_bbq / bbq_reuse_ratio)))
    min_bank_size: int = 1000
    bbq_reuse_ratio: int = 10
    # Sampling composition ratios (must sum to 1.0)
    centroid_near_ratio: float = 0.2
    diversity_ratio: float = 0.6
    random_ratio: float = 0.2
    # FPS: subsample pool size before running farthest-point sampling
    fps_candidate_size: int = 5000
    seed: int = 42
    examples_per_cluster: int = 8
    overwrite: bool = False


def validate_config(config: Config) -> None:
    if not config.clusters_path.exists():
        raise FileNotFoundError(f"Cluster parquet not found: {config.clusters_path}")
    if not config.embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {config.embeddings_path}")
    if not config.bbq_assignments_path.exists():
        raise FileNotFoundError(
            f"BBQ assignments not found: {config.bbq_assignments_path}"
        )
    if config.min_bank_size <= 0:
        raise ValueError("min_bank_size must be > 0")
    if config.bbq_reuse_ratio <= 0:
        raise ValueError("bbq_reuse_ratio must be > 0")
    ratio_sum = config.centroid_near_ratio + config.diversity_ratio + config.random_ratio
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-6):
        raise ValueError(f"Sampling ratios must sum to 1.0, got {ratio_sum}")
    if config.fps_candidate_size <= 0:
        raise ValueError("fps_candidate_size must be > 0")

    paths = [config.output_path, config.summary_path, config.report_path]
    if config.csv_output_path is not None:
        paths.append(config.csv_output_path)
    for path in paths:
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite.")

    if config.cluster_output_dir is not None and config.cluster_output_dir.exists():
        existing = list(config.cluster_output_dir.glob("cluster_*.parquet"))
        if existing and not config.overwrite:
            raise FileExistsError(
                f"Cluster output files already exist in {config.cluster_output_dir}. "
                "Use --overwrite."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stable_hash_fraction(value: str, seed: int, namespace: str) -> float:
    raw = f"{namespace}:{seed}:{value}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def make_reference_id(
    cluster_id: int,
    sample_type: str,
    sample_rank: int,
    prompt_id: str,
    text_hash: str,
) -> str:
    raw = f"{cluster_id}:{sample_type}:{sample_rank}:{prompt_id}:{text_hash}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"ref_{cluster_id}_{sample_type}_{sample_rank:03d}_{digest}"


def compute_bank_size(
    cluster_size: int,
    n_bbq: int,
    min_bank_size: int,
    bbq_reuse_ratio: int,
) -> int:
    return min(cluster_size, max(min_bank_size, math.ceil(n_bbq / bbq_reuse_ratio)))


def split_counts(total: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    """Distribute `total` into len(ratios) buckets using largest-remainder."""
    raw = [total * r for r in ratios]
    floors = [int(math.floor(v)) for v in raw]
    remainder = total - sum(floors)
    remainders = sorted(
        range(len(ratios)), key=lambda i: -(raw[i] - floors[i])
    )
    for i in remainders[:remainder]:
        floors[i] += 1
    return tuple(floors)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clusters(path: Path) -> pd.DataFrame:
    columns = [
        "embedding_index", "prompt_id", "text_hash", "conversation_id",
        "data_split", "country", "model", "timestamp", "text",
        "cluster_id", "cosine_distance_to_centroid",
    ]
    df = pd.read_parquet(path, columns=columns)
    df["cluster_id"] = df["cluster_id"].astype("int32")
    df["embedding_index"] = df["embedding_index"].astype("int64")
    df["cosine_distance_to_centroid"] = df["cosine_distance_to_centroid"].astype("float32")

    before = len(df)
    df = df[df["text"].apply(is_clean_reference)].reset_index(drop=True)
    print(f"Text filter: {before:,} → {len(df):,} (removed {before - len(df):,} conversation/redacted texts)")
    return df


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def load_bbq_counts(path: Path) -> dict[int, int]:
    df = pd.read_parquet(path, columns=["target_cluster_id"])
    return df["target_cluster_id"].value_counts().to_dict()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def select_centroid_near(cluster_df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0:
        return cluster_df.iloc[0:0].copy()
    return (
        cluster_df
        .sort_values(
            ["cosine_distance_to_centroid", "prompt_id"],
            ascending=[True, True],
            kind="mergesort",
        )
        .head(n)
        .copy()
    )


def farthest_point_sampling(
    emb: np.ndarray,
    embedding_indices: np.ndarray,
    n: int,
    seed: int,
    candidate_size: int,
) -> np.ndarray:
    """
    Run FPS on a random subsample of the cluster embeddings.
    Returns selected positions into `embedding_indices`.
    """
    rng = np.random.default_rng(seed)
    m = len(embedding_indices)
    n = min(n, m)

    # Subsample candidates to keep FPS tractable
    if m > candidate_size:
        candidate_pos = rng.choice(m, candidate_size, replace=False)
    else:
        candidate_pos = np.arange(m)

    candidate_emb = emb[embedding_indices[candidate_pos]].astype(np.float32)
    n = min(n, len(candidate_pos))

    # FPS
    start = int(rng.integers(len(candidate_pos)))
    selected_pos = [start]
    min_dists = 1.0 - candidate_emb @ candidate_emb[start]
    min_dists[start] = -np.inf

    for _ in range(n - 1):
        next_pos = int(np.argmax(min_dists))
        selected_pos.append(next_pos)
        new_dists = 1.0 - candidate_emb @ candidate_emb[next_pos]
        np.minimum(min_dists, new_dists, out=min_dists)
        min_dists[next_pos] = -np.inf

    return candidate_pos[selected_pos]


def select_diversity(
    cluster_df: pd.DataFrame,
    emb: np.ndarray,
    n: int,
    seed: int,
    candidate_size: int,
    exclude_prompt_ids: set[str],
) -> pd.DataFrame:
    if n <= 0:
        return cluster_df.iloc[0:0].copy()

    pool = cluster_df[~cluster_df["prompt_id"].isin(exclude_prompt_ids)].reset_index(drop=True)
    if pool.empty:
        return pool

    embedding_indices = pool["embedding_index"].to_numpy()
    selected_pos = farthest_point_sampling(emb, embedding_indices, n, seed, candidate_size)
    return pool.iloc[selected_pos].copy()


def select_stable_random(
    cluster_df: pd.DataFrame,
    n: int,
    seed: int,
    cluster_id: int,
    exclude_prompt_ids: set[str],
) -> pd.DataFrame:
    if n <= 0:
        return cluster_df.iloc[0:0].copy()

    pool = cluster_df[~cluster_df["prompt_id"].isin(exclude_prompt_ids)].copy()
    if pool.empty:
        return pool

    namespace = f"reference_bank_random_cluster_{cluster_id}"
    pool["random_score"] = [
        stable_hash_fraction(str(pid), seed, namespace)
        for pid in pool["prompt_id"]
    ]
    return (
        pool.sort_values(["random_score", "prompt_id"], ascending=[True, True], kind="mergesort")
        .head(n)
        .copy()
    )


def attach_selection_metadata(
    selected_df: pd.DataFrame,
    sample_type: str,
    cluster_id: int,
    cluster_size: int,
    cluster_pi: float,
    bank_size: int,
    seed: int,
) -> pd.DataFrame:
    selected_df = selected_df.copy()
    selected_df["sample_type"] = sample_type
    selected_df["sample_rank"] = range(1, len(selected_df) + 1)
    selected_df["selection_seed"] = seed if sample_type == "random" else pd.NA
    selected_df["cluster_size"] = cluster_size
    selected_df["cluster_pi"] = cluster_pi
    selected_df["bank_size"] = bank_size

    if "random_score" not in selected_df.columns:
        selected_df["random_score"] = float("nan")
    else:
        selected_df["random_score"] = selected_df["random_score"].astype("float64")

    selected_df["reference_id"] = [
        make_reference_id(
            cluster_id=cluster_id,
            sample_type=sample_type,
            sample_rank=int(rank),
            prompt_id=str(pid),
            text_hash=str(th),
        )
        for rank, pid, th in zip(
            selected_df["sample_rank"],
            selected_df["prompt_id"],
            selected_df["text_hash"],
        )
    ]
    return selected_df


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_reference_bank(config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_config(config)
    df = load_clusters(config.clusters_path)
    emb = load_embeddings(config.embeddings_path)
    bbq_counts = load_bbq_counts(config.bbq_assignments_path)

    total_rows = len(df)
    cluster_ids = sorted(int(v) for v in df["cluster_id"].unique())
    selected_parts: list[pd.DataFrame] = []
    unused_parts: dict[int, pd.DataFrame] = {}
    cluster_summaries: dict[str, Any] = {}

    for cluster_id in cluster_ids:
        cluster_df = df[df["cluster_id"] == cluster_id].copy()
        cluster_size = len(cluster_df)
        cluster_pi = cluster_size / total_rows
        n_bbq = bbq_counts.get(cluster_id, 0)

        bank_size = compute_bank_size(
            cluster_size, n_bbq, config.min_bank_size, config.bbq_reuse_ratio
        )
        n_centroid, n_diversity, n_random = split_counts(
            bank_size,
            (config.centroid_near_ratio, config.diversity_ratio, config.random_ratio),
        )

        print(
            f"Cluster {cluster_id}: size={cluster_size:,}, bbq={n_bbq:,}, "
            f"bank={bank_size} (centroid={n_centroid}, diversity={n_diversity}, random={n_random})"
        )

        centroid_df = select_centroid_near(cluster_df, n_centroid)
        excluded = set(str(v) for v in centroid_df["prompt_id"])

        diversity_df = select_diversity(
            cluster_df=cluster_df,
            emb=emb,
            n=n_diversity,
            seed=config.seed,
            candidate_size=config.fps_candidate_size,
            exclude_prompt_ids=excluded,
        )
        excluded |= set(str(v) for v in diversity_df["prompt_id"])

        random_df = select_stable_random(
            cluster_df=cluster_df,
            n=n_random,
            seed=config.seed,
            cluster_id=cluster_id,
            exclude_prompt_ids=excluded,
        )

        meta = dict(
            cluster_id=cluster_id,
            cluster_size=cluster_size,
            cluster_pi=cluster_pi,
            bank_size=bank_size,
            seed=config.seed,
        )
        centroid_df = attach_selection_metadata(centroid_df, "centroid_near", **meta)
        diversity_df = attach_selection_metadata(diversity_df, "diversity", **meta)
        random_df = attach_selection_metadata(random_df, "random", **meta)

        selected_prompt_ids = (
            set(str(v) for v in centroid_df["prompt_id"])
            | set(str(v) for v in diversity_df["prompt_id"])
            | set(str(v) for v in random_df["prompt_id"])
        )
        unused_parts[cluster_id] = cluster_df[
            ~cluster_df["prompt_id"].isin(selected_prompt_ids)
        ].copy()

        selected_parts.extend([centroid_df, diversity_df, random_df])
        cluster_summaries[str(cluster_id)] = {
            "cluster_size": cluster_size,
            "cluster_pi": cluster_pi,
            "n_bbq": n_bbq,
            "bank_size": bank_size,
            "centroid_near_selected": len(centroid_df),
            "diversity_selected": len(diversity_df),
            "random_selected": len(random_df),
            "total_selected": len(centroid_df) + len(diversity_df) + len(random_df),
            "unused": len(unused_parts[cluster_id]),
        }

    bank = pd.concat(selected_parts, ignore_index=True)
    bank = bank[OUTPUT_COLUMNS].sort_values(
        ["cluster_id", "sample_type", "sample_rank"], kind="mergesort"
    )

    summary = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
        },
        "stats": {
            "input_rows": total_rows,
            "cluster_count": len(cluster_ids),
            "reference_bank_rows": len(bank),
            "centroid_near_rows": int((bank["sample_type"] == "centroid_near").sum()),
            "diversity_rows": int((bank["sample_type"] == "diversity").sum()),
            "random_rows": int((bank["sample_type"] == "random").sum()),
            "cluster_summaries": cluster_summaries,
        },
    }
    return bank, unused_parts, summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def truncate_text(text: str, limit: int = 260) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def write_report(
    bank: pd.DataFrame,
    summary: dict[str, Any],
    report_path: Path,
    examples_per_cluster: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# WildChat Reference Bank Examples",
        "",
        f"- Rows: {summary['stats']['reference_bank_rows']:,}",
        f"- Clusters: {summary['stats']['cluster_count']}",
        f"- Centroid-near rows: {summary['stats']['centroid_near_rows']:,}",
        f"- Diversity rows: {summary['stats']['diversity_rows']:,}",
        f"- Random rows: {summary['stats']['random_rows']:,}",
        "",
    ]

    for cluster_id, cs in summary["stats"]["cluster_summaries"].items():
        cluster_bank = bank[bank["cluster_id"] == int(cluster_id)]
        lines.extend([
            f"## Cluster {cluster_id}",
            "",
            f"- WildChat size: {cs['cluster_size']:,}",
            f"- pi_k: {cs['cluster_pi']:.6f}",
            f"- BBQ assigned: {cs['n_bbq']:,}",
            f"- Bank size: {cs['bank_size']}",
            f"- Selected: centroid={cs['centroid_near_selected']}, "
            f"diversity={cs['diversity_selected']}, random={cs['random_selected']}",
            "",
        ])
        for sample_type in ["centroid_near", "diversity", "random"]:
            sample_df = cluster_bank[
                cluster_bank["sample_type"] == sample_type
            ].head(examples_per_cluster)
            lines.extend([f"### {sample_type}", ""])
            for row in sample_df.itertuples(index=False):
                lines.append(
                    f"{int(row.sample_rank)}. "
                    f"`{row.prompt_id}` "
                    f"(dist={float(row.cosine_distance_to_centroid):.4f}) "
                    f"{truncate_text(row.text)}"
                )
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    bank: pd.DataFrame,
    unused_parts: dict[int, pd.DataFrame],
    summary: dict[str, Any],
    config: Config,
) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)

    bank.to_parquet(config.output_path, index=False)
    if config.csv_output_path is not None:
        config.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        bank.to_csv(config.csv_output_path, index=False)

    if config.cluster_output_dir is not None:
        config.cluster_output_dir.mkdir(parents=True, exist_ok=True)
        for stale in config.cluster_output_dir.glob("cluster_*.parquet"):
            stale.unlink()
        cluster_paths: dict[str, str] = {}
        for cluster_id, cluster_bank in bank.groupby("cluster_id", sort=True):
            cluster_path = config.cluster_output_dir / f"cluster_{int(cluster_id):03d}.parquet"
            cluster_bank.to_parquet(cluster_path, index=False)
            cluster_paths[str(int(cluster_id))] = str(cluster_path)
        summary["outputs"] = {"cluster_paths": cluster_paths}

    if config.unused_output_dir is not None:
        config.unused_output_dir.mkdir(parents=True, exist_ok=True)
        for stale in config.unused_output_dir.glob("cluster_*.parquet"):
            stale.unlink()
        unused_paths: dict[str, str] = {}
        for cluster_id, unused_df in unused_parts.items():
            unused_path = config.unused_output_dir / f"cluster_{int(cluster_id):03d}.parquet"
            unused_df.to_parquet(unused_path, index=False)
            unused_paths[str(int(cluster_id))] = str(unused_path)
        summary["outputs"] = {**summary.get("outputs", {}), "unused_paths": unused_paths}

    config.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(bank, summary, config.report_path, config.examples_per_cluster)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Build an adaptive WildChat reference bank with centroid-near, "
            "diversity (FPS), and random sampling."
        )
    )
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--bbq-assignments", type=Path, default=DEFAULT_BBQ_ASSIGNMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cluster-output-dir", type=Path, default=DEFAULT_CLUSTER_OUTPUT_DIR)
    parser.add_argument("--no-cluster-files", action="store_true")
    parser.add_argument("--unused-output-dir", type=Path, default=DEFAULT_UNUSED_OUTPUT_DIR)
    parser.add_argument("--no-unused-files", action="store_true")
    parser.add_argument("--min-bank-size", type=int, default=1000)
    parser.add_argument("--bbq-reuse-ratio", type=int, default=10)
    parser.add_argument("--centroid-near-ratio", type=float, default=0.2)
    parser.add_argument("--diversity-ratio", type=float, default=0.6)
    parser.add_argument("--random-ratio", type=float, default=0.2)
    parser.add_argument("--fps-candidate-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-per-cluster", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    return Config(
        clusters_path=args.clusters,
        embeddings_path=args.embeddings,
        bbq_assignments_path=args.bbq_assignments,
        output_path=args.output,
        csv_output_path=None if args.no_csv else args.csv_output,
        summary_path=args.summary,
        report_path=args.report,
        cluster_output_dir=None if args.no_cluster_files else args.cluster_output_dir,
        unused_output_dir=None if args.no_unused_files else args.unused_output_dir,
        min_bank_size=args.min_bank_size,
        bbq_reuse_ratio=args.bbq_reuse_ratio,
        centroid_near_ratio=args.centroid_near_ratio,
        diversity_ratio=args.diversity_ratio,
        random_ratio=args.random_ratio,
        fps_candidate_size=args.fps_candidate_size,
        seed=args.seed,
        examples_per_cluster=args.examples_per_cluster,
        overwrite=args.overwrite,
    )


def main() -> None:
    config = parse_args()
    bank, unused_parts, summary = build_reference_bank(config)
    write_outputs(bank, unused_parts, summary, config)

    print(f"Wrote reference bank: {config.output_path}")
    if config.csv_output_path is not None:
        print(f"Wrote reference bank CSV: {config.csv_output_path}")
    if config.cluster_output_dir is not None:
        print(f"Wrote per-cluster files: {config.cluster_output_dir}")
    if config.unused_output_dir is not None:
        print(f"Wrote unused per-cluster files: {config.unused_output_dir}")
    print(f"Wrote summary: {config.summary_path}")
    print(f"Wrote examples: {config.report_path}")
    print(json.dumps(summary["stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
