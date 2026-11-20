from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data/processed/rewrite_batch/bbq_rewrite_sbert_filtered.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/rewrite_batch/bbq_rewrite_style_checked.parquet"
UNUSED_REFS_DIR = PROJECT_ROOT / "data/processed/reference_bank_unused_by_cluster"
HELDOUT_CLUSTERS_DIR = PROJECT_ROOT / "data/processed/wildchat_heldout_by_cluster"
MODEL_NAME = "StyleDistance/styledistance"


def batch_mean_similarity(item_emb: np.ndarray, ref_emb: np.ndarray) -> np.ndarray:
    """Mean cosine similarity for a batch of items against a reference set.
    item_emb: (N, D), ref_emb: (M, D), both L2-normalized.
    Returns: (N,) array of mean similarities.
    """
    # mean_j <x, r_j> = <x, mean_j r_j>; avoid materializing N x M.
    return item_emb @ ref_emb.mean(axis=0)


def batch_independent_max_similarity(
    original_emb: np.ndarray,
    rewritten_emb: np.ndarray,
    ref_emb: np.ndarray,
    query_batch_size: int = 2048,
    reference_batch_size: int = 8192,
    progress_label: Optional[str] = None,
    compute_device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent max-reference style similarity (symmetric).

    For each item, independently find the nearest WildChat reference for the
    rewrite AND for the original.

    s_nat[i]  = max_j  cosine_sim(rewritten_emb[i], ref_emb[j])
    s_orig[i] = max_j  cosine_sim(original_emb[i],  ref_emb[j])

    Unlike ``batch_maxref_similarity``, the reference chosen for the original is
    NOT constrained to be the same as the one chosen for the rewrite, so neither
    side has a selection advantage.

    Returns:
        s_nat:    (N,) max cosine similarity of rewrite to refs.
        s_orig:   (N,) max cosine similarity of original to refs.
        best_j_nat:  (N,) argmax reference index for each rewrite.
        best_j_orig: (N,) argmax reference index for each original.
    """
    if query_batch_size <= 0 or reference_batch_size <= 0:
        raise ValueError("Batch sizes must be positive integers.")
    if len(ref_emb) == 0:
        raise ValueError("Reference embedding bank must not be empty.")

    device = torch.device(compute_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA indepmax scoring requested, but CUDA is unavailable.")
        return batch_independent_max_similarity_cuda(
            original_emb=original_emb,
            rewritten_emb=rewritten_emb,
            ref_emb=ref_emb,
            query_batch_size=query_batch_size,
            reference_batch_size=reference_batch_size,
            progress_label=progress_label,
            compute_device=device,
        )

    n_items = len(rewritten_emb)
    s_nat = np.full(n_items, -np.inf, dtype=np.float32)
    s_orig = np.full(n_items, -np.inf, dtype=np.float32)
    best_j_nat = np.full(n_items, -1, dtype=np.int64)
    best_j_orig = np.full(n_items, -1, dtype=np.int64)
    n_query_batches = (n_items + query_batch_size - 1) // query_batch_size

    for query_batch_idx, query_start in enumerate(
        range(0, n_items, query_batch_size), start=1
    ):
        query_end = min(query_start + query_batch_size, n_items)
        rewritten_batch = rewritten_emb[query_start:query_end]
        original_batch = original_emb[query_start:query_end]
        b = len(rewritten_batch)
        batch_best_score_nat = np.full(b, -np.inf, dtype=np.float32)
        batch_best_j_nat = np.full(b, -1, dtype=np.int64)
        batch_best_score_orig = np.full(b, -np.inf, dtype=np.float32)
        batch_best_j_orig = np.full(b, -1, dtype=np.int64)

        for ref_start in range(0, len(ref_emb), reference_batch_size):
            ref_end = min(ref_start + reference_batch_size, len(ref_emb))
            ref_block = ref_emb[ref_start:ref_end]

            sim_nat = rewritten_batch @ ref_block.T
            local_j_nat = sim_nat.argmax(axis=1)
            local_score_nat = sim_nat[np.arange(b), local_j_nat]
            improved_nat = local_score_nat > batch_best_score_nat
            batch_best_score_nat[improved_nat] = local_score_nat[improved_nat]
            batch_best_j_nat[improved_nat] = ref_start + local_j_nat[improved_nat]

            sim_orig = original_batch @ ref_block.T
            local_j_orig = sim_orig.argmax(axis=1)
            local_score_orig = sim_orig[np.arange(b), local_j_orig]
            improved_orig = local_score_orig > batch_best_score_orig
            batch_best_score_orig[improved_orig] = local_score_orig[improved_orig]
            batch_best_j_orig[improved_orig] = ref_start + local_j_orig[improved_orig]

        s_nat[query_start:query_end] = batch_best_score_nat
        s_orig[query_start:query_end] = batch_best_score_orig
        best_j_nat[query_start:query_end] = batch_best_j_nat
        best_j_orig[query_start:query_end] = batch_best_j_orig

        if (
            query_batch_idx == 1
            or query_batch_idx == n_query_batches
            or query_batch_idx % 10 == 0
        ):
            label = f" {progress_label}" if progress_label else ""
            print(
                f"    indepmax{label}: query batch "
                f"{query_batch_idx}/{n_query_batches}",
                flush=True,
            )

    return s_nat, s_orig, best_j_nat, best_j_orig


def batch_independent_max_similarity_cuda(
    original_emb: np.ndarray,
    rewritten_emb: np.ndarray,
    ref_emb: np.ndarray,
    query_batch_size: int,
    reference_batch_size: int,
    progress_label: Optional[str],
    compute_device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact blockwise independent max-reference search using CUDA."""
    n_items = len(rewritten_emb)
    s_nat = np.empty(n_items, dtype=np.float32)
    s_orig = np.empty(n_items, dtype=np.float32)
    best_j_nat = np.empty(n_items, dtype=np.int64)
    best_j_orig = np.empty(n_items, dtype=np.int64)
    n_query_batches = (n_items + query_batch_size - 1) // query_batch_size

    torch.set_float32_matmul_precision("highest")
    reference_tensor = torch.as_tensor(ref_emb, device=compute_device)

    with torch.inference_mode():
        for query_batch_idx, query_start in enumerate(
            range(0, n_items, query_batch_size), start=1
        ):
            query_end = min(query_start + query_batch_size, n_items)
            rewritten_batch = torch.as_tensor(
                rewritten_emb[query_start:query_end], device=compute_device
            )
            original_batch = torch.as_tensor(
                original_emb[query_start:query_end], device=compute_device
            )
            b = len(rewritten_batch)
            batch_best_score_nat = torch.full(
                (b,), -torch.inf, dtype=rewritten_batch.dtype, device=compute_device,
            )
            batch_best_j_nat = torch.full(
                (b,), -1, dtype=torch.long, device=compute_device,
            )
            batch_best_score_orig = torch.full(
                (b,), -torch.inf, dtype=original_batch.dtype, device=compute_device,
            )
            batch_best_j_orig = torch.full(
                (b,), -1, dtype=torch.long, device=compute_device,
            )

            for ref_start in range(0, len(reference_tensor), reference_batch_size):
                ref_end = min(ref_start + reference_batch_size, len(reference_tensor))
                ref_block = reference_tensor[ref_start:ref_end]

                sim_nat = rewritten_batch @ ref_block.T
                local_score_nat, local_j_nat = sim_nat.max(dim=1)
                improved_nat = local_score_nat > batch_best_score_nat
                batch_best_score_nat[improved_nat] = local_score_nat[improved_nat]
                batch_best_j_nat[improved_nat] = ref_start + local_j_nat[improved_nat]

                sim_orig = original_batch @ ref_block.T
                local_score_orig, local_j_orig = sim_orig.max(dim=1)
                improved_orig = local_score_orig > batch_best_score_orig
                batch_best_score_orig[improved_orig] = local_score_orig[improved_orig]
                batch_best_j_orig[improved_orig] = ref_start + local_j_orig[improved_orig]

            s_nat[query_start:query_end] = batch_best_score_nat.cpu().numpy()
            s_orig[query_start:query_end] = batch_best_score_orig.cpu().numpy()
            best_j_nat[query_start:query_end] = batch_best_j_nat.cpu().numpy()
            best_j_orig[query_start:query_end] = batch_best_j_orig.cpu().numpy()

            if (
                query_batch_idx == 1
                or query_batch_idx == n_query_batches
                or query_batch_idx % 10 == 0
            ):
                label = f" {progress_label}" if progress_label else ""
                print(
                    f"    indepmax{label} [cuda]: query batch "
                    f"{query_batch_idx}/{n_query_batches}",
                    flush=True,
                )

    return s_nat, s_orig, best_j_nat, best_j_orig


def batch_maxref_similarity(
    original_emb: np.ndarray,
    rewritten_emb: np.ndarray,
    ref_emb: np.ndarray,
    query_batch_size: int = 2048,
    reference_batch_size: int = 8192,
    progress_label: Optional[str] = None,
    compute_device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Max-ref style similarity.

    For each item, find the WildChat reference sentence j* that is most similar
    to the rewrite. Then compare the rewrite and the original against that same j*.

    s_nat[i] = max_j  cosine_sim(rewritten_emb[i], ref_emb[j])
    s_orig[i] = cosine_sim(original_emb[i], ref_emb[j*])   where j* = argmax above

    Args:
        original_emb:  (N, D) L2-normalized embeddings of original texts.
        rewritten_emb: (N, D) L2-normalized embeddings of rewritten texts.
        ref_emb:       (M, D) L2-normalized embeddings of WildChat references.

    Returns:
        s_nat:  (N,) max cosine similarity of rewrite to ref.
        s_orig: (N,) cosine similarity of original to the same argmax ref.
        best_j: (N,) index of the reference selected for each rewrite.
    """
    if query_batch_size <= 0 or reference_batch_size <= 0:
        raise ValueError("Batch sizes must be positive integers.")
    if len(ref_emb) == 0:
        raise ValueError("Reference embedding bank must not be empty.")

    device = torch.device(compute_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA maxref scoring requested, but CUDA is unavailable.")
        return batch_maxref_similarity_cuda(
            original_emb=original_emb,
            rewritten_emb=rewritten_emb,
            ref_emb=ref_emb,
            query_batch_size=query_batch_size,
            reference_batch_size=reference_batch_size,
            progress_label=progress_label,
            compute_device=device,
        )

    n_items = len(rewritten_emb)
    s_nat = np.full(n_items, -np.inf, dtype=np.float32)
    s_orig = np.empty(n_items, dtype=np.float32)
    best_j = np.full(n_items, -1, dtype=np.int64)
    n_query_batches = (n_items + query_batch_size - 1) // query_batch_size

    # This is exact max-reference search. Blocking changes only memory usage:
    # peak similarity storage is query_batch_size x reference_batch_size.
    for query_batch_idx, query_start in enumerate(
        range(0, n_items, query_batch_size), start=1
    ):
        query_end = min(query_start + query_batch_size, n_items)
        rewritten_batch = rewritten_emb[query_start:query_end]
        batch_best_score = np.full(len(rewritten_batch), -np.inf, dtype=np.float32)
        batch_best_j = np.full(len(rewritten_batch), -1, dtype=np.int64)

        for ref_start in range(0, len(ref_emb), reference_batch_size):
            ref_end = min(ref_start + reference_batch_size, len(ref_emb))
            similarities = rewritten_batch @ ref_emb[ref_start:ref_end].T
            local_j = similarities.argmax(axis=1)
            local_score = similarities[np.arange(len(rewritten_batch)), local_j]
            improved = local_score > batch_best_score
            batch_best_score[improved] = local_score[improved]
            batch_best_j[improved] = ref_start + local_j[improved]

        s_nat[query_start:query_end] = batch_best_score
        best_j[query_start:query_end] = batch_best_j
        selected_refs = ref_emb[batch_best_j]
        s_orig[query_start:query_end] = np.einsum(
            "ij,ij->i",
            original_emb[query_start:query_end],
            selected_refs,
            optimize=True,
        )

        if (
            query_batch_idx == 1
            or query_batch_idx == n_query_batches
            or query_batch_idx % 10 == 0
        ):
            label = f" {progress_label}" if progress_label else ""
            print(
                f"    maxref{label}: query batch "
                f"{query_batch_idx}/{n_query_batches}",
                flush=True,
            )

    return s_nat, s_orig, best_j


def batch_maxref_similarity_cuda(
    original_emb: np.ndarray,
    rewritten_emb: np.ndarray,
    ref_emb: np.ndarray,
    query_batch_size: int,
    reference_batch_size: int,
    progress_label: Optional[str],
    compute_device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact blockwise max-reference search using CUDA matrix products."""
    n_items = len(rewritten_emb)
    s_nat = np.empty(n_items, dtype=np.float32)
    s_orig = np.empty(n_items, dtype=np.float32)
    best_j = np.empty(n_items, dtype=np.int64)
    n_query_batches = (n_items + query_batch_size - 1) // query_batch_size

    # Highest precision disables TF32 shortcuts while retaining exact search.
    torch.set_float32_matmul_precision("highest")
    reference_tensor = torch.as_tensor(ref_emb, device=compute_device)

    with torch.inference_mode():
        for query_batch_idx, query_start in enumerate(
            range(0, n_items, query_batch_size), start=1
        ):
            query_end = min(query_start + query_batch_size, n_items)
            rewritten_batch = torch.as_tensor(
                rewritten_emb[query_start:query_end], device=compute_device
            )
            batch_best_score = torch.full(
                (len(rewritten_batch),),
                -torch.inf,
                dtype=rewritten_batch.dtype,
                device=compute_device,
            )
            batch_best_j = torch.full(
                (len(rewritten_batch),),
                -1,
                dtype=torch.long,
                device=compute_device,
            )

            for ref_start in range(0, len(reference_tensor), reference_batch_size):
                ref_end = min(
                    ref_start + reference_batch_size, len(reference_tensor)
                )
                similarities = (
                    rewritten_batch @ reference_tensor[ref_start:ref_end].T
                )
                local_score, local_j = similarities.max(dim=1)
                improved = local_score > batch_best_score
                batch_best_score[improved] = local_score[improved]
                batch_best_j[improved] = ref_start + local_j[improved]

            original_batch = torch.as_tensor(
                original_emb[query_start:query_end], device=compute_device
            )
            selected_refs = reference_tensor[batch_best_j]
            batch_orig_score = (original_batch * selected_refs).sum(dim=1)

            s_nat[query_start:query_end] = batch_best_score.cpu().numpy()
            s_orig[query_start:query_end] = batch_orig_score.cpu().numpy()
            best_j[query_start:query_end] = batch_best_j.cpu().numpy()

            if (
                query_batch_idx == 1
                or query_batch_idx == n_query_batches
                or query_batch_idx % 10 == 0
            ):
                label = f" {progress_label}" if progress_label else ""
                print(
                    f"    maxref{label} [cuda]: query batch "
                    f"{query_batch_idx}/{n_query_batches}",
                    flush=True,
                )

    return s_nat, s_orig, best_j


def encode_refs(
    model: SentenceTransformer,
    refs_dir: Path,
    cluster_ids: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    cluster_ref_emb: dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        ref_path = refs_dir / f"cluster_{int(cid):03d}.parquet"
        ref_texts = pd.read_parquet(ref_path)["text"].tolist()
        cluster_ref_emb[int(cid)] = model.encode(
            ref_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        print(f"  cluster {int(cid):03d}: {len(ref_texts):,} refs encoded")
    return cluster_ref_emb


def compute_style_scores(
    df: pd.DataFrame,
    original_emb: np.ndarray,
    rewritten_emb: np.ndarray,
    cluster_ref_emb: dict[int, np.ndarray],
    prefix: str,
    mode: str = "mean",
    query_batch_size: int = 2048,
    reference_batch_size: int = 8192,
    compute_device: str = "cpu",
) -> pd.DataFrame:
    """Compute style similarity scores.

    Args:
        mode: "mean" (default) — mean cosine sim to all refs;
              "maxref" — max cosine sim of rewrite to refs, original anchored to same ref;
              "indepmax" — independent max cosine sim for both rewrite and original.
    """
    cluster_ids = sorted(df["target_cluster_id"].unique())
    cluster_id_arr = df["target_cluster_id"].to_numpy()
    s_orig = np.empty(len(df), dtype=np.float32)
    s_nat = np.empty(len(df), dtype=np.float32)
    best_ref_index = np.full(len(df), -1, dtype=np.int64)
    for cid in cluster_ids:
        mask = cluster_id_arr == cid
        ref_emb = cluster_ref_emb[int(cid)]
        print(
            f"  scoring cluster {int(cid):03d}: "
            f"{int(mask.sum()):,} candidates x {len(ref_emb):,} refs",
            flush=True,
        )
        if mode == "indepmax":
            cluster_s_nat, cluster_s_orig, cluster_j_nat, cluster_j_orig = (
                batch_independent_max_similarity(
                    original_emb[mask],
                    rewritten_emb[mask],
                    ref_emb,
                    query_batch_size=query_batch_size,
                    reference_batch_size=reference_batch_size,
                    progress_label=f"{prefix}/c{int(cid)}",
                    compute_device=compute_device,
                )
            )
            s_nat[mask] = cluster_s_nat
            s_orig[mask] = cluster_s_orig
            best_ref_index[mask] = cluster_j_nat
        elif mode == "maxref":
            cluster_s_nat, cluster_s_orig, cluster_best_j = batch_maxref_similarity(
                original_emb[mask],
                rewritten_emb[mask],
                ref_emb,
                query_batch_size=query_batch_size,
                reference_batch_size=reference_batch_size,
                progress_label=f"{prefix}/c{int(cid)}",
                compute_device=compute_device,
            )
            s_nat[mask] = cluster_s_nat
            s_orig[mask] = cluster_s_orig
            best_ref_index[mask] = cluster_best_j
        else:
            s_orig[mask] = batch_mean_similarity(original_emb[mask], ref_emb)
            s_nat[mask] = batch_mean_similarity(rewritten_emb[mask], ref_emb)

    df[f"{prefix}_style_sim_original"] = s_orig
    df[f"{prefix}_style_sim_natural"] = s_nat
    df[f"{prefix}_delta_style"] = s_nat - s_orig
    df[f"{prefix}_style_check_status"] = np.where(df[f"{prefix}_delta_style"] > 0, "pass", "fail")
    if mode in ("maxref", "indepmax"):
        df[f"{prefix}_best_ref_index"] = best_ref_index

    print(f"\n[{prefix}]")
    print(df[f"{prefix}_style_check_status"].value_counts())
    print(df[f"{prefix}_delta_style"].describe())
    t_stat, t_pval = stats.ttest_rel(s_orig, s_nat)
    w_stat, w_pval = stats.wilcoxon(s_orig, s_nat)
    print(f"paired t-test: t={t_stat:.4f}, p={t_pval:.4e}")
    print(f"wilcoxon:      W={w_stat:.1f}, p={w_pval:.4e}")

    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--retry-input",
        type=Path,
        default=None,
        help="Optional fallback-candidate input to score in the same run.",
    )
    parser.add_argument(
        "--retry-output",
        type=Path,
        default=None,
        help="Output for --retry-input; required when --retry-input is set.",
    )
    parser.add_argument("--unused-refs-dir", type=Path, default=UNUSED_REFS_DIR)
    parser.add_argument("--heldout-clusters-dir", type=Path, default=HELDOUT_CLUSTERS_DIR)
    parser.add_argument("--mode", choices=["mean", "maxref", "indepmax"], default="mean",
                        help="Similarity aggregation mode: 'mean' (default), 'maxref', or 'indepmax'.")
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
        help="SentenceTransformer encoding batch size.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=2048,
        help="Number of candidates per exact maxref query block.",
    )
    parser.add_argument(
        "--reference-batch-size",
        type=int,
        default=8192,
        help="Number of references per exact maxref reference block.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "SentenceTransformer and maxref device, e.g. cpu or cuda. "
            "Defaults to auto-detection."
        ),
    )
    args = parser.parse_args()

    if (args.retry_input is None) != (args.retry_output is None):
        parser.error("--retry-input and --retry-output must be provided together.")

    def load_candidates(path: Path, source: str) -> pd.DataFrame:
        candidates = pd.read_parquet(path)
        if "content_check_status" in candidates.columns:
            candidates = candidates[candidates["content_check_status"] == "pass"]
        elif "rewrite_status" in candidates.columns:
            candidates = candidates[candidates["rewrite_status"] == "success"]
        candidates = candidates.copy()
        candidates["_style_score_source"] = source
        if "candidate_stage" not in candidates.columns:
            candidates["candidate_stage"] = source
        else:
            candidates["candidate_stage"] = candidates["candidate_stage"].fillna(source)
        print(f"Loaded {len(candidates):,} valid candidates from {path}", flush=True)
        return candidates

    frames = [load_candidates(args.input, "initial")]
    if args.retry_input is not None:
        frames.append(load_candidates(args.retry_input, "fallback_retry"))
    df = pd.concat(frames, ignore_index=True, sort=False)

    model = SentenceTransformer(MODEL_NAME, device=args.device)
    compute_device = str(model.device)
    print(f"StyleDistance compute device: {compute_device}", flush=True)
    cluster_ids = sorted(df["target_cluster_id"].unique())

    # Matched-body comparison: both sides contain prompt body only. In
    # particular, answer options must not be appended only to the original.
    original = (df["original_context"] + "\n" + df["original_question"])
    rewrite_col = "rewritten_question" if "rewritten_question" in df.columns else "natural_question"
    rewritten = df[rewrite_col].tolist()

    # Original prompt bodies repeat across best-of-N candidates. Encode each
    # distinct body once, then restore row order exactly.
    original_codes, unique_original = pd.factorize(original, sort=False)
    print(
        f"Encoding {len(unique_original):,} unique original prompt bodies "
        f"for {len(df):,} candidate rows",
        flush=True,
    )
    unique_original_emb = model.encode(
        unique_original.tolist(),
        normalize_embeddings=True,
        batch_size=args.encode_batch_size,
        show_progress_bar=True,
    )
    original_emb = unique_original_emb[original_codes]
    print(f"Encoding {len(rewritten):,} natural prompt bodies", flush=True)
    rewritten_emb = model.encode(
        rewritten,
        normalize_embeddings=True,
        batch_size=args.encode_batch_size,
        show_progress_bar=True,
    )

    # Heldout
    if args.heldout_clusters_dir is not None and args.heldout_clusters_dir.exists():
        print(f"\nEncoding heldout refs: {args.heldout_clusters_dir}")
        heldout_ref_emb = encode_refs(
            model,
            args.heldout_clusters_dir,
            cluster_ids,
            batch_size=args.encode_batch_size,
        )
        df = compute_style_scores(
            df,
            original_emb,
            rewritten_emb,
            heldout_ref_emb,
            prefix="heldout",
            mode=args.mode,
            query_batch_size=args.query_batch_size,
            reference_batch_size=args.reference_batch_size,
            compute_device=compute_device,
        )

    # Unused
    if args.unused_refs_dir is not None and args.unused_refs_dir.exists():
        print(f"\nEncoding unused refs: {args.unused_refs_dir}")
        unused_ref_emb = encode_refs(
            model,
            args.unused_refs_dir,
            cluster_ids,
            batch_size=args.encode_batch_size,
        )
        df = compute_style_scores(
            df,
            original_emb,
            rewritten_emb,
            unused_ref_emb,
            prefix="unused",
            mode=args.mode,
            query_batch_size=args.query_batch_size,
            reference_batch_size=args.reference_batch_size,
            compute_device=compute_device,
        )

    # Primary style_check_status: heldout if available, else unused
    if "heldout_delta_style" in df.columns:
        df["style_sim_original"] = df["heldout_style_sim_original"]
        df["style_sim_natural"] = df["heldout_style_sim_natural"]
        df["delta_style"] = df["heldout_delta_style"]
        df["style_check_status"] = df["heldout_style_check_status"]
    else:
        df["style_sim_original"] = df["unused_style_sim_original"]
        df["style_sim_natural"] = df["unused_style_sim_natural"]
        df["delta_style"] = df["unused_delta_style"]
        df["style_check_status"] = df["unused_style_check_status"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.retry_output is None:
        df.drop(columns="_style_score_source").to_parquet(args.output, index=False)
        print(f"\nWrote: {args.output}")
    else:
        args.retry_output.parent.mkdir(parents=True, exist_ok=True)
        initial_scored = df[df["_style_score_source"] == "initial"].drop(
            columns="_style_score_source"
        )
        retry_scored = df[df["_style_score_source"] == "fallback_retry"].drop(
            columns="_style_score_source"
        )
        initial_scored.to_parquet(args.output, index=False)
        retry_scored.to_parquet(args.retry_output, index=False)
        print(f"\nWrote: {args.output}")
        print(f"Wrote: {args.retry_output}")


if __name__ == "__main__":
    main()
