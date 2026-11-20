from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

# Keep Hugging Face downloads on the standard HTTP path in this environment.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Assumes this script is placed under something like:
#   nat_prom/src/style/<this_file>.py
# Override with --input / --output-root if your layout differs.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = PROJECT_ROOT / "data/interim/wildchat_filtered.parquet"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "embeddings/wildchat_style"
DEFAULT_MODEL_NAME = "StyleDistance/styledistance"


# ---------------------------------------------------------------------------
# Source / metadata schema
# ---------------------------------------------------------------------------

# Read only columns needed for embedding alignment and downstream clustering /
# reference selection. Other fields remain in the original parquet and can be
# joined later by prompt_id.
SOURCE_COLUMNS = [
    "prompt_id",
    "text_hash",
    "conversation_id",
    "data_split",
    "country",
    "model",
    "timestamp",
    "text",
]

METADATA_COLUMNS = [
    "embedding_index",
    *SOURCE_COLUMNS,
]

METADATA_SCHEMA = pa.schema(
    [
        pa.field("embedding_index", pa.int64(), nullable=False),
        pa.field("prompt_id", pa.string(), nullable=False),
        pa.field("text_hash", pa.string(), nullable=False),
        pa.field("conversation_id", pa.string(), nullable=False),
        pa.field("data_split", pa.string(), nullable=False),
        pa.field("country", pa.string(), nullable=True),
        pa.field("model", pa.string(), nullable=True),
        pa.field("timestamp", pa.string(), nullable=True),
        pa.field("text", pa.string(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    input_path: Path = DEFAULT_INPUT
    output_root: Path = DEFAULT_OUTPUT_ROOT

    model_name: str = DEFAULT_MODEL_NAME
    model_revision: str | None = None

    data_split: str = "reference"
    limit: int | None = None

    # Arrow scan chunk size and internal model inference batch size are
    # intentionally separate.
    scan_batch_size: int = 8192
    encode_batch_size: int = 256

    device: str | None = None
    normalize_embeddings: bool = True

    progress_every: int = 10_000

    # Full-run safety.
    resume: bool = False
    overwrite: bool = False


def validate_config(config: Config) -> None:
    if not config.input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {config.input_path}")
    if config.data_split not in {"reference", "heldout", "all"}:
        raise ValueError("data_split must be one of: reference, heldout, all")
    if config.limit is not None and config.limit <= 0:
        raise ValueError("limit must be > 0")
    if config.scan_batch_size <= 0:
        raise ValueError("scan_batch_size must be > 0")
    if config.encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be > 0")
    if config.progress_every <= 0:
        raise ValueError("progress_every must be > 0")
    if config.resume and config.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")


def output_dir_for(config: Config) -> Path:
    return config.output_root / config.data_split


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_sentence_transformer(
    model_name: str,
    model_revision: str | None,
    device: str | None,
):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "StyleDistance encoding requires sentence-transformers and torch. "
            "Install them with: pip install sentence-transformers torch"
        ) from exc

    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if model_revision is not None:
        kwargs["revision"] = model_revision

    return SentenceTransformer(model_name, **kwargs)


# ---------------------------------------------------------------------------
# Arrow dataset scanning
# ---------------------------------------------------------------------------

def split_filter(data_split: str):
    if data_split == "all":
        return None
    return ds.field("data_split") == data_split


def open_source_dataset(input_path: Path) -> ds.Dataset:
    return ds.dataset(input_path, format="parquet")


def count_selected_rows(
    dataset: ds.Dataset,
    data_split: str,
    limit: int | None,
) -> tuple[int, int]:
    """
    Returns:
      available_count: rows matching the split
      target_count: rows actually encoded after applying limit
    """
    filter_expr = split_filter(data_split)
    available_count = int(dataset.count_rows(filter=filter_expr))
    target_count = min(available_count, limit) if limit is not None else available_count
    return available_count, target_count


def iter_selected_batches(
    dataset: ds.Dataset,
    config: Config,
    *,
    skip_rows: int = 0,
    target_rows: int,
) -> Iterator[pa.RecordBatch]:
    """
    Stream split-filtered RecordBatches without loading the full reference
    subset into RAM.

    use_threads=False keeps scan order deterministic across resume runs.
    """
    scanner = dataset.scanner(
        columns=SOURCE_COLUMNS,
        filter=split_filter(config.data_split),
        batch_size=config.scan_batch_size,
        use_threads=False,
    )

    skipped = 0
    yielded = 0

    for batch in scanner.to_batches():
        if yielded >= target_rows:
            break

        batch_rows = batch.num_rows
        if batch_rows == 0:
            continue

        # Resume support: skip rows already committed in checkpoint.
        if skipped + batch_rows <= skip_rows:
            skipped += batch_rows
            continue

        if skipped < skip_rows:
            local_skip = skip_rows - skipped
            batch = batch.slice(local_skip)
            skipped = skip_rows

        remaining = target_rows - yielded
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)

        if batch.num_rows > 0:
            yielded += batch.num_rows
            yield batch


# ---------------------------------------------------------------------------
# Metadata part writing / consolidation
# ---------------------------------------------------------------------------

def record_batch_column(batch: pa.RecordBatch, name: str) -> pa.Array:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Column {name!r} not found in RecordBatch")
    return batch.column(index)


def cast_string_array(array: pa.Array) -> pa.Array:
    if pa.types.is_string(array.type):
        return array
    return array.cast(pa.string(), safe=False)


def build_metadata_table(
    batch: pa.RecordBatch,
    start_index: int,
) -> pa.Table:
    n = batch.num_rows
    arrays = [
        pa.array(
            np.arange(start_index, start_index + n, dtype=np.int64),
            type=pa.int64(),
        ),
        cast_string_array(record_batch_column(batch, "prompt_id")),
        cast_string_array(record_batch_column(batch, "text_hash")),
        cast_string_array(record_batch_column(batch, "conversation_id")),
        cast_string_array(record_batch_column(batch, "data_split")),
        cast_string_array(record_batch_column(batch, "country")),
        cast_string_array(record_batch_column(batch, "model")),
        cast_string_array(record_batch_column(batch, "timestamp")),
        cast_string_array(record_batch_column(batch, "text")),
    ]
    return pa.Table.from_arrays(arrays, schema=METADATA_SCHEMA)


def write_metadata_part_atomic(
    table: pa.Table,
    parts_dir: Path,
    part_index: int,
) -> Path:
    parts_dir.mkdir(parents=True, exist_ok=True)

    final_path = parts_dir / f"part-{part_index:06d}.parquet"
    temp_path = parts_dir / f".part-{part_index:06d}.parquet.tmp"

    if temp_path.exists():
        temp_path.unlink()

    pq.write_table(table, temp_path, compression="snappy")
    os.replace(temp_path, final_path)
    return final_path


def consolidate_metadata_parts(
    parts_dir: Path,
    temp_output_path: Path,
) -> int:
    """Consolidate metadata parts without loading all metadata into RAM."""
    part_paths = sorted(parts_dir.glob("part-*.parquet"))
    if not part_paths:
        raise RuntimeError(f"No metadata parts found in {parts_dir}")

    if temp_output_path.exists():
        temp_output_path.unlink()

    writer: pq.ParquetWriter | None = None
    rows_written = 0

    try:
        for part_path in part_paths:
            parquet_file = pq.ParquetFile(part_path)
            for batch in parquet_file.iter_batches(batch_size=65_536):
                table = pa.Table.from_batches([batch], schema=METADATA_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_output_path,
                        METADATA_SCHEMA,
                        compression="snappy",
                    )
                writer.write_table(table)
                rows_written += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return rows_written


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def run_signature(
    config: Config,
    *,
    available_count: int,
    target_count: int,
) -> str:
    payload = {
        "input": file_identity(config.input_path),
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "data_split": config.data_split,
        "limit": config.limit,
        "scan_batch_size": config.scan_batch_size,
        "encode_batch_size": config.encode_batch_size,
        "normalize_embeddings": config.normalize_embeddings,
        "available_count": available_count,
        "target_count": target_count,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Output preparation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    metadata_path: Path
    embeddings_path: Path
    summary_path: Path

    metadata_parts_dir: Path
    checkpoint_path: Path
    partial_embeddings_path: Path
    partial_metadata_path: Path


def make_output_paths(config: Config) -> OutputPaths:
    output_dir = output_dir_for(config)
    return OutputPaths(
        output_dir=output_dir,
        metadata_path=output_dir / "metadata.parquet",
        embeddings_path=output_dir / "embeddings.npy",
        summary_path=output_dir / "summary.json",
        metadata_parts_dir=output_dir / "_metadata_parts",
        checkpoint_path=output_dir / "_checkpoint.json",
        partial_embeddings_path=output_dir / "_embeddings.partial.npy",
        partial_metadata_path=output_dir / "_metadata.partial.parquet",
    )


def clean_outputs(paths: OutputPaths) -> None:
    for path in [
        paths.metadata_path,
        paths.embeddings_path,
        paths.summary_path,
        paths.checkpoint_path,
        paths.partial_embeddings_path,
        paths.partial_metadata_path,
    ]:
        if path.exists():
            path.unlink()

    if paths.metadata_parts_dir.exists():
        shutil.rmtree(paths.metadata_parts_dir)


def prepare_output_state(config: Config, paths: OutputPaths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    final_exists = (
        paths.metadata_path.exists()
        or paths.embeddings_path.exists()
        or paths.summary_path.exists()
    )
    partial_exists = (
        paths.checkpoint_path.exists()
        or paths.partial_embeddings_path.exists()
        or paths.metadata_parts_dir.exists()
    )

    if config.overwrite:
        clean_outputs(paths)
        return

    if config.resume:
        if not paths.checkpoint_path.exists():
            raise FileNotFoundError(
                f"--resume requested but checkpoint not found: {paths.checkpoint_path}"
            )
        return

    if final_exists or partial_exists:
        raise FileExistsError(
            f"Output state already exists under {paths.output_dir}. "
            "Use --resume to continue a partial run or --overwrite to restart."
        )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def batch_texts(batch: pa.RecordBatch) -> list[str]:
    values = record_batch_column(batch, "text").to_pylist()
    return ["" if value is None else str(value) for value in values]


def encode_texts(config: Config) -> dict[str, Any]:
    validate_config(config)

    dataset = open_source_dataset(config.input_path)
    available_count, row_count = count_selected_rows(
        dataset,
        config.data_split,
        config.limit,
    )

    if row_count == 0:
        raise ValueError(f"No rows found for data_split={config.data_split!r}")

    paths = make_output_paths(config)
    prepare_output_state(config, paths)

    signature = run_signature(
        config,
        available_count=available_count,
        target_count=row_count,
    )

    print(f"Input: {config.input_path}")
    print(f"Split: {config.data_split}")
    print(f"Rows available: {available_count:,}")
    print(f"Rows selected for encoding: {row_count:,}")
    print(f"Output directory: {paths.output_dir}")

    rows_encoded = 0
    next_part_index = 0
    embedding_dim: int | None = None
    embeddings_memmap: np.memmap | None = None

    if config.resume:
        checkpoint = load_checkpoint(paths.checkpoint_path)

        if checkpoint.get("run_signature") != signature:
            raise RuntimeError(
                "Checkpoint configuration does not match the current run. "
                "Use --overwrite to restart."
            )

        rows_encoded = int(checkpoint["rows_encoded"])
        next_part_index = int(checkpoint["next_part_index"])
        embedding_dim = int(checkpoint["embedding_dim"])

        if not paths.partial_embeddings_path.exists():
            raise FileNotFoundError(
                f"Partial embeddings file missing: {paths.partial_embeddings_path}"
            )

        embeddings_memmap = np.load(
            paths.partial_embeddings_path,
            mmap_mode="r+",
        )

        if embeddings_memmap.shape != (row_count, embedding_dim):
            raise RuntimeError(
                "Partial embedding shape does not match checkpoint: "
                f"{embeddings_memmap.shape} vs {(row_count, embedding_dim)}"
            )

        print(
            f"Resuming from row {rows_encoded:,}/{row_count:,}, "
            f"metadata part {next_part_index}"
        )

    # Load model only after input/output validation succeeds.
    model = load_sentence_transformer(
        config.model_name,
        config.model_revision,
        config.device,
    )

    session_start_rows = rows_encoded
    started = time.perf_counter()
    next_progress = (
        ((rows_encoded // config.progress_every) + 1)
        * config.progress_every
    )

    for batch in iter_selected_batches(
        dataset,
        config,
        skip_rows=rows_encoded,
        target_rows=row_count - rows_encoded,
    ):
        current_start = rows_encoded
        texts = batch_texts(batch)

        batch_embeddings = model.encode(
            texts,
            batch_size=config.encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=config.normalize_embeddings,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

        if batch_embeddings.ndim != 2:
            raise RuntimeError(
                f"Expected 2D embeddings, got shape {batch_embeddings.shape}"
            )

        if embeddings_memmap is None:
            embedding_dim = int(batch_embeddings.shape[1])
            embeddings_memmap = np.lib.format.open_memmap(
                paths.partial_embeddings_path,
                dtype=np.float32,
                mode="w+",
                shape=(row_count, embedding_dim),
            )

        if embedding_dim != int(batch_embeddings.shape[1]):
            raise RuntimeError(
                "Embedding dimension changed during the run: "
                f"{embedding_dim} -> {batch_embeddings.shape[1]}"
            )

        current_end = current_start + batch.num_rows

        # Write embeddings first. If a crash occurs before checkpoint update,
        # the same slice is recomputed and overwritten on resume.
        embeddings_memmap[current_start:current_end] = batch_embeddings
        embeddings_memmap.flush()

        metadata_table = build_metadata_table(
            batch=batch,
            start_index=current_start,
        )
        write_metadata_part_atomic(
            table=metadata_table,
            parts_dir=paths.metadata_parts_dir,
            part_index=next_part_index,
        )

        rows_encoded = current_end
        next_part_index += 1

        checkpoint_payload = {
            "run_signature": signature,
            "rows_encoded": rows_encoded,
            "next_part_index": next_part_index,
            "embedding_dim": embedding_dim,
            "row_count": row_count,
            "updated_at_unix": time.time(),
        }
        write_json_atomic(paths.checkpoint_path, checkpoint_payload)

        if rows_encoded >= next_progress or rows_encoded == row_count:
            elapsed = time.perf_counter() - started
            session_rows = rows_encoded - session_start_rows
            speed = session_rows / elapsed if elapsed > 0 else 0.0
            remaining_rows = row_count - rows_encoded
            eta_seconds = remaining_rows / speed if speed > 0 else float("inf")
            completed_fraction = rows_encoded / row_count

            eta_text = (
                f"{eta_seconds/60:.1f} min"
                if np.isfinite(eta_seconds)
                else "unknown"
            )

            print(
                f"[{rows_encoded:,}/{row_count:,}] "
                f"{completed_fraction:.1%} "
                f"speed={speed:,.1f} prompts/s "
                f"elapsed={elapsed/60:.1f} min "
                f"ETA={eta_text}"
            )

            while next_progress <= rows_encoded:
                next_progress += config.progress_every

    if embeddings_memmap is None or embedding_dim is None:
        raise RuntimeError("No embeddings were written")

    if rows_encoded != row_count:
        raise RuntimeError(
            f"Encoded row count mismatch: {rows_encoded:,} != {row_count:,}"
        )

    embeddings_memmap.flush()
    del embeddings_memmap

    metadata_rows = consolidate_metadata_parts(
        parts_dir=paths.metadata_parts_dir,
        temp_output_path=paths.partial_metadata_path,
    )

    if metadata_rows != row_count:
        raise RuntimeError(
            f"Metadata row count mismatch: {metadata_rows:,} != {row_count:,}"
        )

    os.replace(paths.partial_embeddings_path, paths.embeddings_path)
    os.replace(paths.partial_metadata_path, paths.metadata_path)

    elapsed_seconds = int(time.perf_counter() - started)

    summary = {
        "config": {
            **asdict(config),
            "input_path": str(config.input_path),
            "output_root": str(config.output_root),
        },
        "model": {
            "name": config.model_name,
            "revision": config.model_revision,
        },
        "outputs": {
            "output_dir": str(paths.output_dir),
            "metadata_path": str(paths.metadata_path),
            "embeddings_path": str(paths.embeddings_path),
            "summary_path": str(paths.summary_path),
        },
        "stats": {
            "rows_available_in_split": available_count,
            "rows_encoded": row_count,
            "embedding_dim": embedding_dim,
            "elapsed_seconds": elapsed_seconds,
            "dtype": "float32",
            "normalized": config.normalize_embeddings,
            "scan_batch_size": config.scan_batch_size,
            "encode_batch_size": config.encode_batch_size,
        },
    }

    write_json_atomic(paths.summary_path, summary)

    # Final outputs are complete; remove recovery artifacts.
    if paths.checkpoint_path.exists():
        paths.checkpoint_path.unlink()
    if paths.metadata_parts_dir.exists():
        shutil.rmtree(paths.metadata_parts_dir)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Stream filtered WildChat prompts from Parquet, encode them with "
            "StyleDistance, write embeddings to an NPY memmap, and preserve "
            "explicit metadata-to-embedding alignment. "
            "By default, data_split == 'reference' is encoded."
        )
    )

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional Hugging Face model revision/commit for reproducibility.",
    )

    parser.add_argument(
        "--data-split",
        default="reference",
        choices=["reference", "heldout", "all"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test row limit applied after split filtering.",
    )

    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=8192,
        help="PyArrow scanner batch size.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Internal SentenceTransformer inference batch size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device: cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Store raw embeddings instead of L2-normalized embeddings.",
    )

    parser.add_argument("--progress-every", type=int, default=10_000)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint and partial outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs/partials and restart from scratch.",
    )

    args = parser.parse_args()

    return Config(
        input_path=args.input,
        output_root=args.output_root,
        model_name=args.model_name,
        model_revision=args.model_revision,
        data_split=args.data_split,
        limit=args.limit,
        scan_batch_size=args.scan_batch_size,
        encode_batch_size=args.encode_batch_size,
        device=args.device,
        normalize_embeddings=not args.no_normalize_embeddings,
        progress_every=args.progress_every,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def main() -> None:
    config = parse_args()
    summary = encode_texts(config)

    print(f"Wrote metadata: {summary['outputs']['metadata_path']}")
    print(f"Wrote embeddings: {summary['outputs']['embeddings_path']}")
    print(f"Wrote summary: {summary['outputs']['summary_path']}")
    print(json.dumps(summary["stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
