from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BBQ_DIR = PROJECT_ROOT / "dataset/bbq/data"
DEFAULT_REFERENCE_SUMMARY = PROJECT_ROOT / "data/processed/reference_bank.summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/bbq_cluster_assignments.parquet"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "data/processed/bbq_cluster_assignments.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data/processed/bbq_cluster_assignments.summary.json"
DEFAULT_STRATIFY_COLUMNS = ("category", "context_condition")

OUTPUT_COLUMNS = [
    "bbq_item_id",
    "source_file",
    "source_row_index",
    "example_id",
    "question_index",
    "question_polarity",
    "context_condition",
    "category",
    "label",
    "ans0",
    "ans1",
    "ans2",
    "context",
    "question",
    "answer_info_json",
    "additional_metadata_json",
    "stratum_key",
    "stratum_size",
    "assignment_rank_in_stratum",
    "assignment_seed",
    "assignment_method",
    "target_cluster_id",
    "target_cluster_pi",
    "target_cluster_quota_in_stratum",
]


@dataclass(frozen=True)
class Config:
    bbq_dir: Path = DEFAULT_BBQ_DIR
    reference_summary_path: Path = DEFAULT_REFERENCE_SUMMARY
    output_path: Path = DEFAULT_OUTPUT
    csv_output_path: Path | None = DEFAULT_CSV_OUTPUT
    summary_path: Path = DEFAULT_SUMMARY
    stratify_columns: tuple[str, ...] = DEFAULT_STRATIFY_COLUMNS
    seed: int = 42
    overwrite: bool = False


def validate_config(config: Config) -> None:
    if not config.bbq_dir.exists():
        raise FileNotFoundError(f"BBQ directory not found: {config.bbq_dir}")
    if not config.reference_summary_path.exists():
        raise FileNotFoundError(
            f"Reference bank summary not found: {config.reference_summary_path}"
        )
    if not config.stratify_columns:
        raise ValueError("At least one stratify column is required")

    paths = [config.output_path, config.summary_path]
    if config.csv_output_path is not None:
        paths.append(config.csv_output_path)
    for path in paths:
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite.")


def stable_hash_fraction(value: str, seed: int, namespace: str) -> float:
    raw = f"{namespace}:{seed}:{value}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def stable_digest(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_bbq_item_id(
    source_stem: str,
    row: dict[str, Any],
) -> str:
    raw = json_dumps(
        {
            "source": source_stem,
            "example_id": row.get("example_id"),
            "question_index": row.get("question_index"),
            "question_polarity": row.get("question_polarity"),
            "context_condition": row.get("context_condition"),
            "category": row.get("category"),
            "context": row.get("context"),
            "question": row.get("question"),
            "ans0": row.get("ans0"),
            "ans1": row.get("ans1"),
            "ans2": row.get("ans2"),
        }
    )
    example_id = str(row.get("example_id", "na")).replace(" ", "_")
    return f"bbq_{source_stem}_{example_id}_{stable_digest(raw)}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def load_bbq_items(bbq_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    jsonl_paths = sorted(bbq_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No JSONL files found in {bbq_dir}")

    for path in jsonl_paths:
        source_stem = path.stem
        for row_index, row in enumerate(read_jsonl(path)):
            record = {
                "bbq_item_id": make_bbq_item_id(source_stem, row),
                "source_file": path.name,
                "source_row_index": row_index,
                "example_id": row.get("example_id"),
                "question_index": row.get("question_index"),
                "question_polarity": row.get("question_polarity"),
                "context_condition": row.get("context_condition"),
                "category": row.get("category") or source_stem,
                "label": row.get("label"),
                "ans0": row.get("ans0"),
                "ans1": row.get("ans1"),
                "ans2": row.get("ans2"),
                "context": row.get("context"),
                "question": row.get("question"),
                "answer_info_json": json_dumps(row.get("answer_info")),
                "additional_metadata_json": json_dumps(row.get("additional_metadata")),
            }
            records.append(record)

    df = pd.DataFrame.from_records(records)
    if df["bbq_item_id"].duplicated().any():
        duplicated = df.loc[df["bbq_item_id"].duplicated(), "bbq_item_id"].head()
        raise ValueError(f"Duplicate bbq_item_id values found: {duplicated.tolist()}")
    return df


def load_cluster_proportions(summary_path: Path) -> dict[int, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    cluster_summaries = payload["stats"]["cluster_summaries"]
    proportions = {
        int(cluster_id): float(stats["cluster_pi"])
        for cluster_id, stats in cluster_summaries.items()
    }
    total = sum(proportions.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Cluster proportions must sum to 1; got {total}")
    return dict(sorted(proportions.items()))


def largest_remainder_quota(
    total: int,
    proportions: dict[int, float],
) -> dict[int, int]:
    raw = {
        cluster_id: total * proportion
        for cluster_id, proportion in proportions.items()
    }
    quota = {
        cluster_id: int(math.floor(value))
        for cluster_id, value in raw.items()
    }
    remaining = total - sum(quota.values())
    remainders = sorted(
        (
            (raw[cluster_id] - quota[cluster_id], cluster_id)
            for cluster_id in proportions
        ),
        key=lambda item: (-item[0], item[1]),
    )

    for _, cluster_id in remainders[:remaining]:
        quota[cluster_id] += 1
    return quota


def make_stratum_key(row: pd.Series, columns: tuple[str, ...]) -> str:
    parts = [f"{column}={row[column]}" for column in columns]
    return "|".join(parts)


def assign_one_stratum(
    stratum_df: pd.DataFrame,
    stratum_key: str,
    proportions: dict[int, float],
    seed: int,
) -> pd.DataFrame:
    stratum_df = stratum_df.copy()
    stratum_size = len(stratum_df)
    quota = largest_remainder_quota(stratum_size, proportions)

    namespace = f"bbq_cluster_assignment:{stratum_key}"
    stratum_df["_assignment_score"] = [
        stable_hash_fraction(str(item_id), seed, namespace)
        for item_id in stratum_df["bbq_item_id"]
    ]
    stratum_df = stratum_df.sort_values(
        ["_assignment_score", "bbq_item_id"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    assigned_clusters: list[int] = []
    for cluster_id, count in quota.items():
        assigned_clusters.extend([cluster_id] * count)
    if len(assigned_clusters) != stratum_size:
        raise RuntimeError(
            f"Quota mismatch for {stratum_key}: "
            f"{len(assigned_clusters)} != {stratum_size}"
        )

    stratum_df["stratum_key"] = stratum_key
    stratum_df["stratum_size"] = stratum_size
    stratum_df["assignment_rank_in_stratum"] = range(1, stratum_size + 1)
    stratum_df["assignment_seed"] = seed
    stratum_df["assignment_method"] = "stratified_quota_largest_remainder"
    stratum_df["target_cluster_id"] = assigned_clusters
    stratum_df["target_cluster_pi"] = [
        proportions[int(cluster_id)]
        for cluster_id in stratum_df["target_cluster_id"]
    ]
    stratum_df["target_cluster_quota_in_stratum"] = [
        quota[int(cluster_id)]
        for cluster_id in stratum_df["target_cluster_id"]
    ]

    return stratum_df.drop(columns=["_assignment_score"])


def assign_clusters(config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_config(config)
    bbq_df = load_bbq_items(config.bbq_dir)
    proportions = load_cluster_proportions(config.reference_summary_path)

    missing_columns = [
        column for column in config.stratify_columns if column not in bbq_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing stratify columns in BBQ data: {missing_columns}")

    assigned_parts: list[pd.DataFrame] = []
    stratum_summaries: dict[str, Any] = {}

    grouped = bbq_df.groupby(list(config.stratify_columns), dropna=False, sort=True)
    for stratum_values, stratum_df in grouped:
        if not isinstance(stratum_values, tuple):
            stratum_values = (stratum_values,)
        stratum_key = "|".join(
            f"{column}={value}"
            for column, value in zip(config.stratify_columns, stratum_values)
        )
        assigned_df = assign_one_stratum(
            stratum_df=stratum_df,
            stratum_key=stratum_key,
            proportions=proportions,
            seed=config.seed,
        )
        counts = assigned_df["target_cluster_id"].value_counts().sort_index()
        stratum_summaries[stratum_key] = {
            "size": int(len(assigned_df)),
            "target_cluster_counts": {
                str(int(cluster_id)): int(count)
                for cluster_id, count in counts.items()
            },
        }
        assigned_parts.append(assigned_df)

    assignments = pd.concat(assigned_parts, ignore_index=True)
    assignments = assignments[OUTPUT_COLUMNS].sort_values(
        ["category", "context_condition", "assignment_rank_in_stratum"],
        kind="mergesort",
    )

    overall_counts = assignments["target_cluster_id"].value_counts().sort_index()
    summary = {
        "config": {
            **asdict(config),
            "bbq_dir": str(config.bbq_dir),
            "reference_summary_path": str(config.reference_summary_path),
            "output_path": str(config.output_path),
            "csv_output_path": (
                str(config.csv_output_path)
                if config.csv_output_path is not None
                else None
            ),
            "summary_path": str(config.summary_path),
            "stratify_columns": list(config.stratify_columns),
        },
        "stats": {
            "bbq_rows": int(len(assignments)),
            "source_files": sorted(assignments["source_file"].unique().tolist()),
            "cluster_proportions": {
                str(cluster_id): proportion
                for cluster_id, proportion in proportions.items()
            },
            "assigned_cluster_counts": {
                str(int(cluster_id)): int(count)
                for cluster_id, count in overall_counts.items()
            },
            "assigned_cluster_proportions": {
                str(int(cluster_id)): float(count / len(assignments))
                for cluster_id, count in overall_counts.items()
            },
            "stratum_count": len(stratum_summaries),
            "strata": stratum_summaries,
        },
    }
    return assignments, summary


def write_outputs(
    assignments: pd.DataFrame,
    summary: dict[str, Any],
    config: Config,
) -> None:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(config.output_path, index=False)

    if config.csv_output_path is not None:
        config.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(config.csv_output_path, index=False)

    config.summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Assign BBQ items to target WildChat style clusters using "
            "stratified quota assignment."
        )
    )
    parser.add_argument("--bbq-dir", type=Path, default=DEFAULT_BBQ_DIR)
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=DEFAULT_REFERENCE_SUMMARY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--stratify-columns",
        nargs="+",
        default=list(DEFAULT_STRATIFY_COLUMNS),
        help=(
            "Columns used to form strata. Default: category context_condition. "
            "You may add question_polarity for stricter balancing."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    return Config(
        bbq_dir=args.bbq_dir,
        reference_summary_path=args.reference_summary,
        output_path=args.output,
        csv_output_path=None if args.no_csv else args.csv_output,
        summary_path=args.summary,
        stratify_columns=tuple(args.stratify_columns),
        seed=args.seed,
        overwrite=args.overwrite,
    )


def main() -> None:
    config = parse_args()
    assignments, summary = assign_clusters(config)
    write_outputs(assignments, summary, config)

    print(f"Wrote BBQ cluster assignments: {config.output_path}")
    if config.csv_output_path is not None:
        print(f"Wrote BBQ cluster assignments CSV: {config.csv_output_path}")
    print(f"Wrote summary: {config.summary_path}")
    print(json.dumps(summary["stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
