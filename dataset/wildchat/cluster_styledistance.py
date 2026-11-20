from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import dump
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Assumes this file is placed under e.g.:
#   nat_prom/src/style/cluster_wildchat_style.py
# Override with CLI paths if your repository layout differs.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EMBEDDINGS = (
    PROJECT_ROOT / "embeddings/wildchat_style/reference/embeddings.npy"
)
DEFAULT_METADATA = (
    PROJECT_ROOT / "embeddings/wildchat_style/reference/metadata.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data/processed/wildchat_clusters"
)

DEFAULT_K_VALUES = (4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36)
DEFAULT_STABILITY_SEEDS = (42, 43, 44)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    embeddings_path: Path = DEFAULT_EMBEDDINGS
    metadata_path: Path = DEFAULT_METADATA
    output_dir: Path = DEFAULT_OUTPUT_DIR

    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    seed: int = 42

    # MiniBatchKMeans initialization / streaming updates.
    minibatch_size: int = 4096
    init_sample_size: int = 50_000
    n_init: int = 3
    max_iter: int = 100
    partial_fit_epochs: int = 1

    # Final cosine-consistent spherical refinement.
    cosine_refine_iters: int = 5
    cosine_refine_tol: float = 1e-4
    assignment_batch_size: int = 50_000

    # K evaluation.
    silhouette_sample_size: int = 5_000
    stratified_silhouette_per_cluster: int = 300

    stability_sample_size: int = 20_000
    stability_seeds: tuple[int, ...] = DEFAULT_STABILITY_SEEDS

    # Input validation.
    norm_check_sample_size: int = 10_000
    norm_tolerance: float = 1e-2

    # Optional finalization: only this K gets a full metadata parquet copy.
    final_k: int | None = None

    # Run management.
    resume: bool = False
    overwrite: bool = False


def validate_config(config: Config) -> None:
    if not config.embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {config.embeddings_path}"
        )
    if not config.metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {config.metadata_path}"
        )

    if not config.k_values:
        raise ValueError("At least one K value is required")
    if any(k <= 1 for k in config.k_values):
        raise ValueError("All K values must be > 1")
    if len(set(config.k_values)) != len(config.k_values):
        raise ValueError("k_values must not contain duplicates")

    if config.minibatch_size <= 0:
        raise ValueError("minibatch_size must be > 0")
    if config.init_sample_size <= 0:
        raise ValueError("init_sample_size must be > 0")
    if config.n_init <= 0:
        raise ValueError("n_init must be > 0")
    if config.max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if config.partial_fit_epochs < 0:
        raise ValueError("partial_fit_epochs must be >= 0")

    if config.cosine_refine_iters < 0:
        raise ValueError("cosine_refine_iters must be >= 0")
    if config.cosine_refine_tol < 0:
        raise ValueError("cosine_refine_tol must be >= 0")
    if config.assignment_batch_size <= 0:
        raise ValueError("assignment_batch_size must be > 0")

    if config.silhouette_sample_size < 0:
        raise ValueError("silhouette_sample_size must be >= 0")
    if config.stratified_silhouette_per_cluster < 0:
        raise ValueError(
            "stratified_silhouette_per_cluster must be >= 0"
        )

    if config.stability_sample_size < 0:
        raise ValueError("stability_sample_size must be >= 0")
    if len(config.stability_seeds) < 2:
        raise ValueError(
            "At least two stability seeds are required"
        )

    if config.norm_check_sample_size < 0:
        raise ValueError("norm_check_sample_size must be >= 0")
    if config.norm_tolerance <= 0:
        raise ValueError("norm_tolerance must be > 0")

    if config.final_k is not None and config.final_k not in config.k_values:
        raise ValueError("final_k must be included in k_values")

    if config.resume and config.overwrite:
        raise ValueError(
            "--resume and --overwrite cannot be used together"
        )


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def normalize_rows(
    array: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def normalized_float32(
    array: np.ndarray,
) -> np.ndarray:
    return normalize_rows(
        np.asarray(array, dtype=np.float32)
    ).astype(np.float32, copy=False)


def sample_indices(
    row_count: int,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    if sample_size <= 0:
        return np.empty(0, dtype=np.int64)

    actual_size = min(sample_size, row_count)
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(
            row_count,
            size=actual_size,
            replace=False,
        ).astype(np.int64, copy=False)
    )


def iter_array_batches(
    array: np.ndarray,
    batch_size: int,
) -> Iterable[tuple[int, int, np.ndarray]]:
    row_count = array.shape[0]
    for start in range(0, row_count, batch_size):
        end = min(start + batch_size, row_count)
        yield start, end, array[start:end]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def load_embeddings(path: Path) -> np.memmap:
    embeddings = np.load(path, mmap_mode="r")

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, got shape {embeddings.shape}"
        )
    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Expected float32 embeddings, got {embeddings.dtype}"
        )
    if embeddings.shape[0] == 0:
        raise ValueError("Embedding matrix is empty")

    return embeddings


def metadata_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def validate_embedding_indices(
    metadata_path: Path,
    expected_rows: int,
) -> None:
    """
    Verify metadata row i has embedding_index == i without loading all
    indices into memory.
    """
    parquet_file = pq.ParquetFile(metadata_path)
    expected_start = 0

    for batch in parquet_file.iter_batches(
        columns=["embedding_index"],
        batch_size=65_536,
    ):
        indices = batch.column(0).to_numpy(
            zero_copy_only=False,
        ).astype(np.int64, copy=False)

        expected = np.arange(
            expected_start,
            expected_start + len(indices),
            dtype=np.int64,
        )

        if not np.array_equal(indices, expected):
            mismatch = np.flatnonzero(indices != expected)
            first_local = int(mismatch[0])
            raise RuntimeError(
                "metadata embedding_index is not sequential: "
                f"row={expected_start + first_local:,}, "
                f"found={int(indices[first_local])}, "
                f"expected={int(expected[first_local])}"
            )

        expected_start += len(indices)

    if expected_start != expected_rows:
        raise RuntimeError(
            "Metadata embedding_index count mismatch: "
            f"{expected_start:,} != {expected_rows:,}"
        )


def validate_normalized_embeddings(
    embeddings: np.memmap,
    sample_size: int,
    tolerance: float,
    seed: int,
) -> dict[str, float]:
    if sample_size <= 0:
        return {
            "sample_size": 0,
            "mean_norm": float("nan"),
            "min_norm": float("nan"),
            "max_norm": float("nan"),
            "max_abs_deviation_from_1": float("nan"),
        }

    indices = sample_indices(
        embeddings.shape[0],
        sample_size,
        seed,
    )
    sample = np.asarray(
        embeddings[indices],
        dtype=np.float32,
    )
    norms = np.linalg.norm(sample, axis=1)
    max_deviation = float(
        np.max(np.abs(norms - 1.0))
    )

    stats = {
        "sample_size": int(len(indices)),
        "mean_norm": float(norms.mean()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "max_abs_deviation_from_1": max_deviation,
    }

    if max_deviation > tolerance:
        raise RuntimeError(
            "Embeddings do not appear L2-normalized. "
            f"max |norm-1|={max_deviation:.6g} > "
            f"tolerance={tolerance:.6g}. "
            "This pipeline assumes normalized StyleDistance embeddings."
        )

    return stats


# ---------------------------------------------------------------------------
# Output state / resume
# ---------------------------------------------------------------------------

def prepare_output_dir(
    output_dir: Path,
    overwrite: bool,
    resume: bool,
) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return
        if resume:
            return
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use --resume to keep completed K runs or "
            "--overwrite to restart."
        )

    output_dir.mkdir(parents=True, exist_ok=True)


def k_output_dir(
    root: Path,
    k: int,
) -> Path:
    return root / f"k_{k:03d}"


def required_k_outputs(
    output_dir: Path,
) -> list[Path]:
    return [
        output_dir / "labels.npy",
        output_dir / "cosine_distance_to_centroid.npy",
        output_dir / "centroids_normalized.npy",
        output_dir / "initial_centroids.npy",
        output_dir / "initial_minibatch_kmeans.joblib",
        output_dir / "summary.json",
    ]


def is_k_complete(
    output_dir: Path,
    row_count: int,
    k: int,
) -> bool:
    paths = required_k_outputs(output_dir)
    if not all(path.exists() for path in paths):
        return False

    try:
        labels = np.load(
            output_dir / "labels.npy",
            mmap_mode="r",
        )
        distances = np.load(
            output_dir / "cosine_distance_to_centroid.npy",
            mmap_mode="r",
        )
        centroids = np.load(
            output_dir / "centroids_normalized.npy",
            mmap_mode="r",
        )

        if labels.shape != (row_count,):
            return False
        if distances.shape != (row_count,):
            return False
        if centroids.shape[0] != k:
            return False
    except Exception:
        return False

    return True


# ---------------------------------------------------------------------------
# Cosine-consistent clustering
# ---------------------------------------------------------------------------

def cosine_assign(
    embeddings: np.ndarray,
    centroids_normalized: np.ndarray,
    batch_size: int,
    *,
    return_distances: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Assign each normalized embedding to the centroid with maximum cosine
    similarity. Distance is 1 - cosine_similarity.
    """
    row_count = embeddings.shape[0]
    labels = np.empty(row_count, dtype=np.int32)
    distances = (
        np.empty(row_count, dtype=np.float32)
        if return_distances
        else None
    )

    centroid_t = centroids_normalized.T

    for start, end, batch in iter_array_batches(
        embeddings,
        batch_size,
    ):
        batch_array = normalized_float32(batch)
        similarities = batch_array @ centroid_t

        batch_labels = np.argmax(
            similarities,
            axis=1,
        ).astype(np.int32, copy=False)

        labels[start:end] = batch_labels

        if distances is not None:
            best_similarities = similarities[
                np.arange(end - start),
                batch_labels,
            ]
            distances[start:end] = (
                1.0 - best_similarities
            ).astype(np.float32, copy=False)

    return labels, distances


def fit_initial_minibatch_kmeans(
    embeddings: np.ndarray,
    k: int,
    seed: int,
    config: Config,
) -> MiniBatchKMeans:
    """
    Initialize on a fixed random sample, then optionally partial_fit over the
    full matrix in batches. The final centers are only initialization for the
    subsequent spherical/cosine refinement.
    """
    row_count = embeddings.shape[0]
    init_indices = sample_indices(
        row_count,
        min(config.init_sample_size, row_count),
        seed,
    )
    init_sample = normalized_float32(embeddings[init_indices])

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=config.minibatch_size,
        n_init=config.n_init,
        max_iter=config.max_iter,
        reassignment_ratio=0.01,
        verbose=0,
    )
    model.fit(init_sample)

    for _ in range(config.partial_fit_epochs):
        for _, _, batch in iter_array_batches(
            embeddings,
            config.minibatch_size,
        ):
            model.partial_fit(normalized_float32(batch))

    return model


def spherical_refine(
    embeddings: np.ndarray,
    initial_centroids: np.ndarray,
    k: int,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Refine normalized centroids using repeated cosine assignment and centroid
    recomputation. Final assignments, distances, and representative selection
    all use the same cosine geometry.
    """
    centroids = normalize_rows(
        np.asarray(initial_centroids, dtype=np.float32)
    ).astype(np.float32, copy=False)

    previous_labels: np.ndarray | None = None
    history: list[dict[str, Any]] = []

    for iteration in range(config.cosine_refine_iters):
        labels, _ = cosine_assign(
            embeddings,
            centroids,
            config.assignment_batch_size,
            return_distances=False,
        )

        sums = np.zeros(
            (k, embeddings.shape[1]),
            dtype=np.float64,
        )
        counts = np.bincount(
            labels,
            minlength=k,
        ).astype(np.int64, copy=False)

        for start, end, batch in iter_array_batches(
            embeddings,
            config.assignment_batch_size,
        ):
            batch_labels = labels[start:end]
            batch_array = normalized_float32(batch)

            for cluster_id in range(k):
                mask = batch_labels == cluster_id
                if np.any(mask):
                    sums[cluster_id] += batch_array[mask].sum(
                        axis=0,
                        dtype=np.float64,
                    )

        new_centroids = centroids.copy()
        nonempty = counts > 0

        if np.any(nonempty):
            means = sums[nonempty] / counts[nonempty, None]
            new_centroids[nonempty] = normalize_rows(
                means.astype(np.float32)
            )

        changed_ratio = None
        if previous_labels is not None:
            changed_ratio = float(
                np.mean(labels != previous_labels)
            )

        history.append(
            {
                "iteration": iteration + 1,
                "changed_ratio": changed_ratio,
                "empty_clusters": int(np.sum(~nonempty)),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
            }
        )

        centroids = new_centroids.astype(
            np.float32,
            copy=False,
        )

        if (
            changed_ratio is not None
            and changed_ratio <= config.cosine_refine_tol
        ):
            break

        previous_labels = labels

    final_labels, final_distances = cosine_assign(
        embeddings,
        centroids,
        config.assignment_batch_size,
        return_distances=True,
    )

    assert final_distances is not None
    return (
        final_labels,
        final_distances,
        centroids,
        history,
    )


def fit_cosine_clusterer(
    embeddings: np.ndarray,
    k: int,
    seed: int,
    config: Config,
) -> tuple[
    MiniBatchKMeans,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    initial_model = fit_initial_minibatch_kmeans(
        embeddings=embeddings,
        k=k,
        seed=seed,
        config=config,
    )

    (
        labels,
        distances,
        centroids_normalized,
        refine_history,
    ) = spherical_refine(
        embeddings=embeddings,
        initial_centroids=initial_model.cluster_centers_,
        k=k,
        config=config,
    )

    return (
        initial_model,
        labels,
        distances,
        centroids_normalized,
        refine_history,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def global_sampled_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> float | None:
    if len(indices) == 0:
        return None

    sample_labels = labels[indices]
    if np.unique(sample_labels).size < 2:
        return None

    sample_embeddings = normalized_float32(embeddings[indices])

    return float(
        silhouette_score(
            sample_embeddings,
            sample_labels,
            metric="cosine",
        )
    )


def stratified_silhouette_indices(
    labels: np.ndarray,
    k: int,
    per_cluster: int,
    seed: int,
) -> np.ndarray:
    if per_cluster <= 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []

    for cluster_id in range(k):
        members = np.flatnonzero(labels == cluster_id)
        if len(members) == 0:
            continue

        take = min(per_cluster, len(members))
        chosen = rng.choice(
            members,
            size=take,
            replace=False,
        ).astype(np.int64, copy=False)
        selected.append(chosen)

    if not selected:
        return np.empty(0, dtype=np.int64)

    return np.sort(np.concatenate(selected))


def stratified_sampled_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    per_cluster: int,
    seed: int,
) -> tuple[float | None, int]:
    indices = stratified_silhouette_indices(
        labels=labels,
        k=k,
        per_cluster=per_cluster,
        seed=seed,
    )

    if len(indices) == 0:
        return None, 0

    sample_labels = labels[indices]
    if np.unique(sample_labels).size < 2:
        return None, len(indices)

    sample_embeddings = normalized_float32(embeddings[indices])

    score = float(
        silhouette_score(
            sample_embeddings,
            sample_labels,
            metric="cosine",
        )
    )
    return score, len(indices)


def stability_analysis(
    embeddings: np.ndarray,
    k: int,
    config: Config,
    stability_indices: np.ndarray,
) -> dict[str, Any]:
    if len(stability_indices) == 0:
        return {
            "sample_size": 0,
            "seeds": list(config.stability_seeds),
            "pairwise_ari": {},
            "mean_ari": None,
            "min_ari": None,
            "max_ari": None,
        }

    sample_embeddings = normalized_float32(embeddings[stability_indices])

    labelings: dict[int, np.ndarray] = {}

    stability_config = Config(
        **{
            **asdict(config),
            "init_sample_size": min(
                config.init_sample_size,
                len(sample_embeddings),
            ),
            "partial_fit_epochs": 0,
            "final_k": None,
            "resume": False,
            "overwrite": False,
        }
    )

    for stability_seed in config.stability_seeds:
        (
            _initial_model,
            labels,
            _distances,
            _centroids,
            _history,
        ) = fit_cosine_clusterer(
            embeddings=sample_embeddings,
            k=k,
            seed=stability_seed,
            config=stability_config,
        )
        labelings[stability_seed] = labels

    pairwise: dict[str, float] = {}
    values: list[float] = []

    for seed_a, seed_b in combinations(
        config.stability_seeds,
        2,
    ):
        ari = float(
            adjusted_rand_score(
                labelings[seed_a],
                labelings[seed_b],
            )
        )
        pairwise[f"{seed_a}_vs_{seed_b}"] = ari
        values.append(ari)

    return {
        "sample_size": int(len(stability_indices)),
        "seeds": list(config.stability_seeds),
        "pairwise_ari": pairwise,
        "mean_ari": float(np.mean(values)) if values else None,
        "min_ari": float(np.min(values)) if values else None,
        "max_ari": float(np.max(values)) if values else None,
    }


# ---------------------------------------------------------------------------
# Cluster summaries
# ---------------------------------------------------------------------------

def summarize_clusters(
    labels: np.ndarray,
    cosine_distances: np.ndarray,
    k: int,
) -> dict[str, Any]:
    counts = np.bincount(
        labels,
        minlength=k,
    ).astype(np.int64, copy=False)

    proportions = counts / labels.shape[0]

    mean_distances = np.full(
        k,
        np.nan,
        dtype=np.float64,
    )
    median_distances = np.full(
        k,
        np.nan,
        dtype=np.float64,
    )

    for cluster_id in range(k):
        cluster_values = cosine_distances[
            labels == cluster_id
        ]
        if len(cluster_values) > 0:
            mean_distances[cluster_id] = float(
                cluster_values.mean()
            )
            median_distances[cluster_id] = float(
                np.median(cluster_values)
            )

    return {
        "cluster_counts": {
            str(i): int(counts[i])
            for i in range(k)
        },
        "cluster_proportions": {
            str(i): float(proportions[i])
            for i in range(k)
        },
        "mean_cosine_distance_to_centroid": {
            str(i): float(mean_distances[i])
            for i in range(k)
        },
        "median_cosine_distance_to_centroid": {
            str(i): float(median_distances[i])
            for i in range(k)
        },
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "min_cluster_ratio": float(proportions.min()),
        "max_cluster_ratio": float(proportions.max()),
    }


# ---------------------------------------------------------------------------
# Per-K run
# ---------------------------------------------------------------------------

def run_one_k(
    embeddings: np.memmap,
    k: int,
    config: Config,
    global_silhouette_indices: np.ndarray,
    stability_indices: np.ndarray,
) -> dict[str, Any]:
    output_dir = k_output_dir(
        config.output_dir,
        k,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_path = output_dir / "labels.npy"
    distances_path = (
        output_dir / "cosine_distance_to_centroid.npy"
    )
    centroids_path = (
        output_dir / "centroids_normalized.npy"
    )
    initial_centroids_path = (
        output_dir / "initial_centroids.npy"
    )
    model_path = (
        output_dir / "initial_minibatch_kmeans.joblib"
    )
    summary_path = output_dir / "summary.json"

    started = time.perf_counter()

    print(f"\n=== K={k} ===")
    print("Fitting MiniBatchKMeans initialization + full-data partial_fit")

    (
        initial_model,
        labels,
        cosine_distances,
        centroids_normalized,
        refine_history,
    ) = fit_cosine_clusterer(
        embeddings=embeddings,
        k=k,
        seed=config.seed,
        config=config,
    )

    print("Computing global sampled cosine silhouette")
    global_silhouette = global_sampled_silhouette(
        embeddings=embeddings,
        labels=labels,
        indices=global_silhouette_indices,
    )

    print("Computing cluster-stratified cosine silhouette")
    (
        stratified_silhouette,
        stratified_sample_size,
    ) = stratified_sampled_silhouette(
        embeddings=embeddings,
        labels=labels,
        k=k,
        per_cluster=config.stratified_silhouette_per_cluster,
        seed=config.seed,
    )

    print("Computing multi-seed stability (pairwise ARI)")
    stability = stability_analysis(
        embeddings=embeddings,
        k=k,
        config=config,
        stability_indices=stability_indices,
    )

    print("Saving compact per-K outputs")
    np.save(labels_path, labels)
    np.save(distances_path, cosine_distances)
    np.save(
        centroids_path,
        centroids_normalized.astype(
            np.float32,
            copy=False,
        ),
    )
    np.save(
        initial_centroids_path,
        initial_model.cluster_centers_.astype(
            np.float32,
            copy=False,
        ),
    )
    dump(initial_model, model_path)

    cluster_summary = summarize_clusters(
        labels=labels,
        cosine_distances=cosine_distances,
        k=k,
    )

    elapsed_seconds = int(
        time.perf_counter() - started
    )

    summary = {
        "k": k,
        "method": {
            "initialization": "MiniBatchKMeans",
            "final_assignment": (
                "nearest L2-normalized centroid by cosine similarity"
            ),
            "distance": "1 - cosine_similarity",
            "spherical_refinement": True,
        },
        "config": {
            **asdict(config),
            "embeddings_path": str(config.embeddings_path),
            "metadata_path": str(config.metadata_path),
            "output_dir": str(config.output_dir),
            "k_values": list(config.k_values),
            "stability_seeds": list(config.stability_seeds),
        },
        "outputs": {
            "labels_path": str(labels_path),
            "cosine_distances_path": str(distances_path),
            "centroids_normalized_path": str(centroids_path),
            "initial_centroids_path": str(initial_centroids_path),
            "initial_model_path": str(model_path),
            "summary_path": str(summary_path),
        },
        "stats": {
            "rows_clustered": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "initial_minibatch_inertia": float(
                initial_model.inertia_
            ),
            "initial_minibatch_n_iter": int(
                initial_model.n_iter_
            ),
            "global_silhouette_sample_size": int(
                len(global_silhouette_indices)
            ),
            "global_cosine_silhouette": global_silhouette,
            "stratified_silhouette_sample_size": int(
                stratified_sample_size
            ),
            "stratified_cosine_silhouette": (
                stratified_silhouette
            ),
            "stability": stability,
            "spherical_refine_history": refine_history,
            "elapsed_seconds": elapsed_seconds,
            **cluster_summary,
        },
    }

    write_json_atomic(
        summary_path,
        summary,
    )

    return summary


# ---------------------------------------------------------------------------
# Optional final metadata materialization
# ---------------------------------------------------------------------------

def append_final_cluster_columns_to_metadata(
    metadata_path: Path,
    output_path: Path,
    labels: np.ndarray,
    cosine_distances: np.ndarray,
) -> int:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    if temp_path.exists():
        temp_path.unlink()
    if output_path.exists():
        output_path.unlink()

    parquet_file = pq.ParquetFile(metadata_path)
    writer: pq.ParquetWriter | None = None
    rows_written = 0

    try:
        for batch in parquet_file.iter_batches(
            batch_size=65_536
        ):
            table = pa.Table.from_batches([batch])

            index_column = (
                table.column("embedding_index")
                .to_numpy(zero_copy_only=False)
                .astype(np.int64, copy=False)
            )

            cluster_ids = pa.array(
                labels[index_column],
                type=pa.int32(),
            )
            distance_values = pa.array(
                cosine_distances[index_column],
                type=pa.float32(),
            )

            table = table.append_column(
                "cluster_id",
                cluster_ids,
            )
            table = table.append_column(
                "cosine_distance_to_centroid",
                distance_values,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    temp_path,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)
            rows_written += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    os.replace(temp_path, output_path)
    return rows_written


def materialize_final_k(
    config: Config,
    row_count: int,
) -> dict[str, Any] | None:
    if config.final_k is None:
        return None

    selected_dir = k_output_dir(
        config.output_dir,
        config.final_k,
    )

    labels = np.load(
        selected_dir / "labels.npy",
        mmap_mode="r",
    )
    distances = np.load(
        selected_dir / "cosine_distance_to_centroid.npy",
        mmap_mode="r",
    )

    final_dir = config.output_dir / "final"
    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clusters_path = (
        final_dir / "wildchat_clusters.parquet"
    )
    selection_path = final_dir / "selection.json"

    rows_written = append_final_cluster_columns_to_metadata(
        metadata_path=config.metadata_path,
        output_path=clusters_path,
        labels=labels,
        cosine_distances=distances,
    )

    if rows_written != row_count:
        raise RuntimeError(
            "Final metadata row count mismatch: "
            f"{rows_written:,} != {row_count:,}"
        )

    payload = {
        "selected_k": config.final_k,
        "selection_policy": (
            "User-specified final K after reviewing cosine silhouette, "
            "multi-seed ARI stability, minimum cluster ratio, and "
            "qualitative cluster interpretability."
        ),
        "clusters_path": str(clusters_path),
        "source_k_dir": str(selected_dir),
        "rows_written": int(rows_written),
    }

    write_json_atomic(
        selection_path,
        payload,
    )
    return payload


# ---------------------------------------------------------------------------
# Root pipeline
# ---------------------------------------------------------------------------

def cluster_embeddings(
    config: Config,
) -> dict[str, Any]:
    validate_config(config)

    prepare_output_dir(
        output_dir=config.output_dir,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    embeddings = load_embeddings(
        config.embeddings_path
    )
    row_count, embedding_dim = embeddings.shape

    metadata_rows = metadata_row_count(
        config.metadata_path
    )
    if metadata_rows != row_count:
        raise RuntimeError(
            f"Metadata rows ({metadata_rows:,}) do not match "
            f"embeddings ({row_count:,})"
        )

    print("Validating metadata-to-embedding alignment")
    validate_embedding_indices(
        metadata_path=config.metadata_path,
        expected_rows=row_count,
    )

    print("Validating L2-normalized embeddings")
    norm_stats = validate_normalized_embeddings(
        embeddings=embeddings,
        sample_size=config.norm_check_sample_size,
        tolerance=config.norm_tolerance,
        seed=config.seed,
    )

    print(f"Embeddings: {config.embeddings_path}")
    print(f"Metadata: {config.metadata_path}")
    print(f"Rows: {row_count:,}")
    print(f"Embedding dim: {embedding_dim}")
    print(
        "K values: "
        + ", ".join(str(k) for k in config.k_values)
    )

    # Same global sample across all K for fair silhouette comparison.
    global_silhouette_indices = sample_indices(
        row_count=row_count,
        sample_size=config.silhouette_sample_size,
        seed=config.seed + 10_001,
    )

    # Same fixed sample across all K and all stability seeds.
    stability_indices = sample_indices(
        row_count=row_count,
        sample_size=config.stability_sample_size,
        seed=config.seed + 20_001,
    )

    samples_path = (
        config.output_dir / "evaluation_samples.npz"
    )
    np.savez(
        samples_path,
        global_silhouette_indices=global_silhouette_indices,
        stability_indices=stability_indices,
    )

    started = time.perf_counter()
    summaries: list[dict[str, Any]] = []

    for k in config.k_values:
        output_dir = k_output_dir(
            config.output_dir,
            k,
        )

        if (
            config.resume
            and is_k_complete(
                output_dir=output_dir,
                row_count=row_count,
                k=k,
            )
        ):
            print(f"\nSkipping completed K={k}")
            summary_path = output_dir / "summary.json"
            summaries.append(
                json.loads(
                    summary_path.read_text(
                        encoding="utf-8"
                    )
                )
            )
            continue

        # Remove incomplete K directory before recomputing.
        if output_dir.exists():
            shutil.rmtree(output_dir)

        summaries.append(
            run_one_k(
                embeddings=embeddings,
                k=k,
                config=config,
                global_silhouette_indices=(
                    global_silhouette_indices
                ),
                stability_indices=stability_indices,
            )
        )

    finalization = materialize_final_k(
        config=config,
        row_count=row_count,
    )

    root_summary = {
        "config": {
            **asdict(config),
            "embeddings_path": str(config.embeddings_path),
            "metadata_path": str(config.metadata_path),
            "output_dir": str(config.output_dir),
            "k_values": list(config.k_values),
            "stability_seeds": list(
                config.stability_seeds
            ),
        },
        "input": {
            "rows": int(row_count),
            "embedding_dim": int(embedding_dim),
            "norm_check": norm_stats,
        },
        "evaluation_samples": {
            "path": str(samples_path),
            "global_silhouette_sample_size": int(
                len(global_silhouette_indices)
            ),
            "stability_sample_size": int(
                len(stability_indices)
            ),
        },
        "k_summaries": {
            str(summary["k"]): {
                "summary_path": summary["outputs"]["summary_path"],
                "labels_path": summary["outputs"]["labels_path"],
                "centroids_path": (
                    summary["outputs"]["centroids_normalized_path"]
                ),
                "global_cosine_silhouette": (
                    summary["stats"]["global_cosine_silhouette"]
                ),
                "stratified_cosine_silhouette": (
                    summary["stats"][
                        "stratified_cosine_silhouette"
                    ]
                ),
                "stability_mean_ari": (
                    summary["stats"]["stability"]["mean_ari"]
                ),
                "stability_min_ari": (
                    summary["stats"]["stability"]["min_ari"]
                ),
                "min_cluster_ratio": (
                    summary["stats"]["min_cluster_ratio"]
                ),
                "max_cluster_ratio": (
                    summary["stats"]["max_cluster_ratio"]
                ),
            }
            for summary in summaries
        },
        "selection": {
            "automatic_selection": False,
            "note": (
                "No automatic best-K selection is performed because the "
                "planned criterion also includes qualitative cluster "
                "interpretability. Review cosine silhouette, ARI stability, "
                "minimum cluster ratio, and qualitative cluster examples."
            ),
            "finalization": finalization,
        },
        "stats": {
            "elapsed_seconds": int(
                time.perf_counter() - started
            ),
        },
    }

    write_json_atomic(
        config.output_dir / "summary.json",
        root_summary,
    )
    return root_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_int_values(
    values: list[str],
) -> tuple[int, ...]:
    parsed: list[int] = []

    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))

    return tuple(parsed)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster L2-normalized StyleDistance embeddings with a "
            "cosine-consistent pipeline: MiniBatchKMeans initialization, "
            "spherical centroid refinement, cosine silhouette evaluation, "
            "and multi-seed ARI stability analysis."
        )
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--k-values",
        nargs="+",
        default=[str(k) for k in DEFAULT_K_VALUES],
        help=(
            "K candidates, e.g. --k-values 4 6 8 "
            "or --k-values 4,6,8."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--init-sample-size", type=int, default=50_000)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--partial-fit-epochs", type=int, default=1)

    parser.add_argument("--cosine-refine-iters", type=int, default=5)
    parser.add_argument("--cosine-refine-tol", type=float, default=1e-4)
    parser.add_argument(
        "--assignment-batch-size",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--stratified-silhouette-per-cluster",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--stability-sample-size",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--stability-seeds",
        nargs="+",
        default=[str(s) for s in DEFAULT_STABILITY_SEEDS],
        help="Seeds for ARI stability, e.g. --stability-seeds 42 43 44.",
    )

    parser.add_argument(
        "--norm-check-sample-size",
        type=int,
        default=10_000,
    )
    parser.add_argument("--norm-tolerance", type=float, default=1e-2)

    parser.add_argument(
        "--final-k",
        type=int,
        default=None,
        help=(
            "Optional user-selected final K. Only this K gets a full "
            "metadata parquet with cluster columns."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip K runs whose required outputs are complete.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory and restart.",
    )

    args = parser.parse_args()

    return Config(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        k_values=parse_int_values(args.k_values),
        seed=args.seed,
        minibatch_size=args.minibatch_size,
        init_sample_size=args.init_sample_size,
        n_init=args.n_init,
        max_iter=args.max_iter,
        partial_fit_epochs=args.partial_fit_epochs,
        cosine_refine_iters=args.cosine_refine_iters,
        cosine_refine_tol=args.cosine_refine_tol,
        assignment_batch_size=args.assignment_batch_size,
        silhouette_sample_size=args.silhouette_sample_size,
        stratified_silhouette_per_cluster=(
            args.stratified_silhouette_per_cluster
        ),
        stability_sample_size=args.stability_sample_size,
        stability_seeds=parse_int_values(args.stability_seeds),
        norm_check_sample_size=args.norm_check_sample_size,
        norm_tolerance=args.norm_tolerance,
        final_k=args.final_k,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def main() -> None:
    config = parse_args()
    summary = cluster_embeddings(config)

    print(
        "Wrote clustering summary: "
        f"{config.output_dir / 'summary.json'}"
    )
    print(
        json.dumps(
            summary["k_summaries"],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
