from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Use the standard Hugging Face HTTP path in this environment.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise RuntimeError(
        "This script requires pyarrow. Install it with: pip install pyarrow"
    ) from exc


# ---------------------------------------------------------------------------
# Fixed dataset / output
# ---------------------------------------------------------------------------

DATASET_NAME = "allenai/WildChat-4.8M"
DATASET_SPLIT = "train"
EXPECTED_ROWS = 3_199_860

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/interim/wildchat_filtered.parquet"
DEFAULT_SUMMARY = PROJECT_ROOT / "data/interim/wildchat_filtered.summary.json"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "prompt_id",
    "text_hash",
    "conversation_id",
    "data_split",
    "turn_identifier",
    "hashed_ip",
    "country",
    "turn_index",
    "user_turn_index",
    "model",
    "timestamp",
    "language",
    "text",
    "token_count",
    "char_count",
    "line_count",
    "url_count",
    "source_turn_count",
]

ARROW_SCHEMA = pa.schema(
    [
        pa.field("prompt_id", pa.string(), nullable=False),
        pa.field("text_hash", pa.string(), nullable=False),
        pa.field("conversation_id", pa.string(), nullable=False),
        pa.field("data_split", pa.string(), nullable=False),
        pa.field("turn_identifier", pa.string(), nullable=True),
        pa.field("hashed_ip", pa.string(), nullable=True),
        pa.field("country", pa.string(), nullable=True),
        pa.field("turn_index", pa.int64(), nullable=False),
        pa.field("user_turn_index", pa.int64(), nullable=False),
        pa.field("model", pa.string(), nullable=True),
        pa.field("timestamp", pa.string(), nullable=True),
        pa.field("language", pa.string(), nullable=True),
        pa.field("text", pa.string(), nullable=False),
        pa.field("token_count", pa.int64(), nullable=False),
        pa.field("char_count", pa.int64(), nullable=False),
        pa.field("line_count", pa.int64(), nullable=False),
        pa.field("url_count", pa.int64(), nullable=False),
        pa.field("source_turn_count", pa.int64(), nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# Filtering patterns
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```")
HTML_TAG_RE = re.compile(
    r"</?(html|body|div|span|p|a|script|style|table|tr|td|ul|ol|li|"
    r"h[1-6]|br|input|form|button)(\s|>|/)",
    re.IGNORECASE,
)
CODE_LINE_RE = re.compile(
    r"^\s*(import\s+\w|from\s+[\w.]+\s+import\s+|def\s+\w+\s*\(|"
    r"class\s+\w+\s*[:(]|function\s+\w*\s*\(|const\s+\w+\s*=|"
    r"let\s+\w+\s*=|var\s+\w+\s*=|if\s*\(|for\s*\(|while\s*\(|"
    r"try\s*\{|catch\s*\(|#include)",
    re.IGNORECASE,
)
# Intentionally case-sensitive to avoid natural-language false positives such as
# "Select the best name from this list".
SQL_LINE_RE = re.compile(r"^\s*SELECT\s+.+\s+FROM\s+\w")
CODE_SYMBOL_RE = re.compile(r"[{};=<>]|->|::")

CONTEXT_MARKER_RE = re.compile(r"(?im)^\s*context\s*:")
QUESTION_MARKER_RE = re.compile(r"(?im)^\s*question\s*:")
CHOICES_MARKER_RE = re.compile(r"(?im)^\s*choices?\s*:")
ANSWER_MARKER_RE = re.compile(r"(?im)^\s*answer\s*:")
LETTERED_OPTION_RE = re.compile(r"(?m)^\s*[A-D][\).]\s+\S")

JAILBREAK_RE = re.compile(
    r"\b(ignore (all )?(previous|prior) instructions|"
    r"developer mode|do anything now|dan mode|jailbreak)\b",
    re.IGNORECASE,
)

# Episode/season/resolution extraction requests
EXTRACTION_RE = re.compile(
    r"^(send|show)\s+only\s+(episode|season|resolution|series|chapter)\s+(number|num)",
    re.IGNORECASE,
)

# Roleplay/persona jailbreak openers
ROLEPLAY_RE = re.compile(
    r"^(act as|you are a|pretend to be|interpret as|respond as|"
    r"interact as|behave as|play (the role of|as))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    output_path: Path = DEFAULT_OUTPUT
    summary_path: Path = DEFAULT_SUMMARY

    seed: int = 42
    heldout_ratio: float = 0.20
    write_batch_size: int = 50_000
    progress_every: int = 50_000

    language: str = "English"
    min_ascii_ratio: float = 0.97
    min_tokens: int = 5
    max_tokens: int = 300
    max_chars: int = 1000
    max_lines: int = 10
    max_urls: int = 1
    max_code_line_ratio: float = 0.10
    max_symbol_ratio: float = 0.15

    first_user_turn_only: bool = True
    deduplicate: bool = True


def validate_config(config: Config) -> None:
    if not 0.0 < config.heldout_ratio < 1.0:
        raise ValueError("heldout_ratio must be in (0, 1)")
    if config.write_batch_size <= 0:
        raise ValueError("write_batch_size must be > 0")
    if config.progress_every <= 0:
        raise ValueError("progress_every must be > 0")
    if config.min_tokens < 0 or config.max_tokens < config.min_tokens:
        raise ValueError("invalid token bounds")


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Remove superficial whitespace noise while preserving wording.

    - normalize CRLF/CR to LF
    - strip outer whitespace
    - collapse repeated whitespace within each non-empty line
    - remove empty lines
    - keep line breaks between non-empty lines
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def approx_token_count(text: str) -> int:
    """Approximate word-like token count, not a model tokenizer count."""
    return len(WORD_RE.findall(text))


def text_digest(text: str) -> bytes:
    """
    128-bit digest used for exact-ish large-scale deduplication.

    The in-memory set stores compact bytes objects instead of hex strings.
    """
    dedup_text = " ".join(text.casefold().split())
    return hashlib.blake2b(
        dedup_text.encode("utf-8"),
        digest_size=16,
    ).digest()


def digest_hex(digest: bytes) -> str:
    return digest.hex()


def stable_hash_fraction(value: str, seed: int, namespace: str) -> float:
    raw = f"{namespace}:{seed}:{value}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def get_conversation_id(row: dict[str, Any]) -> str:
    conversation_id = row.get("conversation_hash") or row.get("conversation_id")
    if conversation_id:
        return str(conversation_id)

    conversation = row.get("conversation") or []
    raw = json.dumps(
        conversation,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return "derived_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def assign_data_split(
    conversation_id: str,
    seed: int,
    heldout_ratio: float,
) -> str:
    fraction = stable_hash_fraction(
        conversation_id,
        seed=seed,
        namespace="reference_heldout_split",
    )
    return "heldout" if fraction < heldout_ratio else "reference"


def make_prompt_id(conversation_id: str, turn_index: int, text: str) -> str:
    raw = f"{conversation_id}:{turn_index}:{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def safe_str(value: Any) -> str | None:
    return None if value is None else str(value)


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def ascii_ratio(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 1.0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / len(text)


def has_non_latin_script(text: str) -> bool:
    """Return True if the text contains any CJK, Cyrillic, Arabic, or Hebrew characters."""
    import unicodedata
    for ch in text:
        if ord(ch) < 128:
            continue
        name = unicodedata.name(ch, "")
        if any(script in name for script in ("CJK", "CYRILLIC", "ARABIC", "HEBREW", "HANGUL", "HIRAGANA", "KATAKANA")):
            return True
    return False


def has_non_latin_script_prefix(text: str, prefix_len: int = 50) -> bool:
    """Return True if the first prefix_len characters contain non-Latin script."""
    return has_non_latin_script(text[:prefix_len])


def is_target_language(value: Any, target: str) -> bool:
    if value is None:
        return False

    value_norm = str(value).strip().lower()
    target_norm = target.strip().lower()

    if target_norm in {"en", "eng", "english"}:
        return value_norm in {"en", "eng", "english"}

    return value_norm == target_norm


def symbol_ratio(text: str) -> float:
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 0.0

    symbol_count = sum(not ch.isalnum() for ch in non_space)
    return symbol_count / len(non_space)


def code_line_ratio(text: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    code_like = sum(
        bool(CODE_LINE_RE.search(line) or SQL_LINE_RE.search(line))
        for line in lines
    )
    return code_like / len(lines)


def is_code_heavy(text: str, config: Config) -> bool:
    if CODE_FENCE_RE.search(text):
        return True
    if HTML_TAG_RE.search(text):
        return True
    if code_line_ratio(text) > config.max_code_line_ratio:
        return True

    return (
        CODE_SYMBOL_RE.search(text) is not None
        and symbol_ratio(text) > config.max_symbol_ratio
    )


def is_benchmark_like(text: str) -> bool:
    """
    Detect strong benchmark scaffolding.

    Plain natural-user A/B/C questions are not rejected only because they have
    three options.
    """
    has_context = CONTEXT_MARKER_RE.search(text) is not None
    has_question = QUESTION_MARKER_RE.search(text) is not None
    has_choices = CHOICES_MARKER_RE.search(text) is not None
    has_answer = ANSWER_MARKER_RE.search(text) is not None
    option_count = len(LETTERED_OPTION_RE.findall(text))

    if has_context and has_question:
        return True
    if has_question and has_choices and option_count >= 2:
        return True
    if has_question and has_answer and option_count >= 2:
        return True
    if has_choices and has_answer and option_count >= 2:
        return True
    return False


def rejection_reason(
    text: str,
    row: dict[str, Any],
    message: dict[str, Any],
    config: Config,
) -> str | None:
    language = message.get("language") or row.get("language")
    if not is_target_language(language, config.language):
        return "language"

    # WildChat's own per-message language label is sometimes wrong (e.g. short
    # English brand/code tokens mixed into otherwise non-English text getting
    # misclassified as "English"). Verify independently instead of trusting it.
    if config.language.strip().lower() in {"en", "eng", "english"}:
        if ascii_ratio(text) < config.min_ascii_ratio:
            return "language_script_mismatch"
        if has_non_latin_script(text):
            return "language_script_mismatch"
        if has_non_latin_script_prefix(text, prefix_len=50):
            return "language_script_mismatch"

    if message.get("redacted") or row.get("redacted"):
        return "redacted"

    if not text:
        return "empty"

    token_count = approx_token_count(text)
    if token_count < config.min_tokens:
        return "too_short"
    if token_count > config.max_tokens:
        return "too_long"

    if len(text) > config.max_chars:
        return "too_many_chars"
    if len(text.splitlines()) > config.max_lines:
        return "too_many_lines"
    if len(URL_RE.findall(text)) > config.max_urls:
        return "url_heavy"

    if is_code_heavy(text, config):
        return "code_heavy"
    if is_benchmark_like(text):
        return "benchmark_like"
    if JAILBREAK_RE.search(text):
        return "jailbreak_like"
    if EXTRACTION_RE.match(text):
        return "extraction_like"
    if ROLEPLAY_RE.match(text):
        return "roleplay_like"

    return None


# ---------------------------------------------------------------------------
# Output records
# ---------------------------------------------------------------------------

def make_record(
    row: dict[str, Any],
    message: dict[str, Any],
    text: str,
    text_hash: str,
    conversation_id: str,
    data_split: str,
    turn_index: int,
    user_turn_index: int,
) -> dict[str, Any]:
    return {
        "prompt_id": make_prompt_id(conversation_id, turn_index, text),
        "text_hash": text_hash,
        "conversation_id": conversation_id,
        "data_split": data_split,
        "turn_identifier": safe_str(message.get("turn_identifier")),
        "hashed_ip": safe_str(row.get("hashed_ip")),
        "country": safe_str(row.get("country")),
        "turn_index": int(turn_index),
        "user_turn_index": int(user_turn_index),
        "model": safe_str(row.get("model")),
        "timestamp": safe_str(row.get("timestamp")),
        "language": safe_str(message.get("language") or row.get("language")),
        "text": text,
        "token_count": approx_token_count(text),
        "char_count": len(text),
        "line_count": len(text.splitlines()),
        "url_count": len(URL_RE.findall(text)),
        "source_turn_count": safe_int(row.get("turn")),
    }


# ---------------------------------------------------------------------------
# Incremental Parquet writer
# ---------------------------------------------------------------------------

class ParquetBatchWriter:
    def __init__(
        self,
        output_path: Path,
        schema: pa.Schema,
    ) -> None:
        self.output_path = output_path
        self.schema = schema
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.output_path.exists():
            self.output_path.unlink()

        self._writer: pq.ParquetWriter | None = None
        self.rows_written = 0

    def write(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        table = pa.Table.from_pylist(records, schema=self.schema)

        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.output_path,
                self.schema,
                compression="snappy",
            )

        self._writer.write_table(table)
        self.rows_written += len(records)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            return

        # Create an empty parquet file with the correct schema if no row survived.
        empty_arrays = [
            pa.array([], type=field.type)
            for field in self.schema
        ]
        empty_table = pa.Table.from_arrays(
            empty_arrays,
            schema=self.schema,
        )
        pq.write_table(
            empty_table,
            self.output_path,
            compression="snappy",
        )

    def __enter__(self) -> "ParquetBatchWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Full 3.2M-row preprocessing
# ---------------------------------------------------------------------------

def preprocess_full_dataset(config: Config) -> dict[str, int]:
    validate_config(config)

    print(f"Loading full dataset: {DATASET_NAME} [{DATASET_SPLIT}]")
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        keep_in_memory=False,
    )

    total_rows = len(dataset)
    print(f"Loaded rows: {total_rows:,}")

    if total_rows != EXPECTED_ROWS:
        print(
            "WARNING: dataset row count differs from the expected "
            f"{EXPECTED_ROWS:,}; continuing with {total_rows:,} rows."
        )

    stats: dict[str, int] = {
        "rows_total": total_rows,
        "rows_seen": 0,
        "messages_seen": 0,
        "user_messages_seen": 0,
        "kept": 0,
        "kept_reference": 0,
        "kept_heldout": 0,
    }

    # Store compact 128-bit digests rather than full hex strings.
    seen_text_hashes: set[bytes] = set()
    batch: list[dict[str, Any]] = []

    started = time.perf_counter()

    with ParquetBatchWriter(
        output_path=config.output_path,
        schema=ARROW_SCHEMA,
    ) as writer:
        for row_index, row in enumerate(dataset, start=1):
            stats["rows_seen"] += 1

            conversation = row.get("conversation") or []
            if not isinstance(conversation, list):
                stats["rejected_bad_conversation"] = (
                    stats.get("rejected_bad_conversation", 0) + 1
                )
                continue

            conversation_id = get_conversation_id(row)
            data_split = assign_data_split(
                conversation_id=conversation_id,
                seed=config.seed,
                heldout_ratio=config.heldout_ratio,
            )

            user_turn_index = 0

            for turn_index, message in enumerate(conversation):
                stats["messages_seen"] += 1

                if not isinstance(message, dict):
                    stats["rejected_bad_message"] = (
                        stats.get("rejected_bad_message", 0) + 1
                    )
                    continue

                if str(message.get("role", "")).lower() != "user":
                    continue

                stats["user_messages_seen"] += 1
                user_turn_index += 1

                if config.first_user_turn_only and user_turn_index != 1:
                    stats["rejected_followup_turn"] = (
                        stats.get("rejected_followup_turn", 0) + 1
                    )
                    continue

                text = normalize_text(str(message.get("content") or ""))

                reason = rejection_reason(
                    text=text,
                    row=row,
                    message=message,
                    config=config,
                )
                if reason is not None:
                    key = f"rejected_{reason}"
                    stats[key] = stats.get(key, 0) + 1
                    continue

                digest = text_digest(text)

                if config.deduplicate:
                    if digest in seen_text_hashes:
                        stats["rejected_duplicate"] = (
                            stats.get("rejected_duplicate", 0) + 1
                        )
                        continue
                    seen_text_hashes.add(digest)

                batch.append(
                    make_record(
                        row=row,
                        message=message,
                        text=text,
                        text_hash=digest_hex(digest),
                        conversation_id=conversation_id,
                        data_split=data_split,
                        turn_index=turn_index,
                        user_turn_index=user_turn_index,
                    )
                )

                stats["kept"] += 1
                stats[f"kept_{data_split}"] += 1

                if len(batch) >= config.write_batch_size:
                    writer.write(batch)
                    batch.clear()

            if row_index % config.progress_every == 0:
                elapsed = time.perf_counter() - started
                rows_per_sec = row_index / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{row_index:,}/{total_rows:,}] "
                    f"kept={stats['kept']:,} "
                    f"reference={stats['kept_reference']:,} "
                    f"heldout={stats['kept_heldout']:,} "
                    f"speed={rows_per_sec:,.1f} rows/s"
                )

        if batch:
            writer.write(batch)
            batch.clear()

        stats["rows_written"] = writer.rows_written

    stats["unique_text_hashes_seen"] = len(seen_text_hashes)
    stats["elapsed_seconds"] = int(time.perf_counter() - started)

    return stats


def write_summary(
    config: Config,
    stats: dict[str, int],
) -> None:
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": {
            "name": DATASET_NAME,
            "split": DATASET_SPLIT,
            "expected_rows": EXPECTED_ROWS,
        },
        "config": {
            **asdict(config),
            "output_path": str(config.output_path),
            "summary_path": str(config.summary_path),
        },
        "stats": dict(sorted(stats.items())),
    }

    config.summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the full allenai/WildChat-4.8M public dataset "
            "(currently 3,199,860 rows) into a filtered English user-prompt pool."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-ratio", type=float, default=0.20)
    parser.add_argument("--write-batch-size", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=50_000)

    parser.add_argument("--language", default="English")
    parser.add_argument("--min-ascii-ratio", type=float, default=0.9)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--max-lines", type=int, default=20)
    parser.add_argument("--max-urls", type=int, default=1)
    parser.add_argument("--max-code-line-ratio", type=float, default=0.30)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.22)

    parser.add_argument(
        "--include-followup-turns",
        action="store_true",
        help="Keep all user turns instead of only the first user turn.",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable global text deduplication.",
    )

    args = parser.parse_args()

    return Config(
        output_path=args.output,
        summary_path=args.summary,
        seed=args.seed,
        heldout_ratio=args.heldout_ratio,
        write_batch_size=args.write_batch_size,
        progress_every=args.progress_every,
        language=args.language,
        min_ascii_ratio=args.min_ascii_ratio,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_chars=args.max_chars,
        max_lines=args.max_lines,
        max_urls=args.max_urls,
        max_code_line_ratio=args.max_code_line_ratio,
        max_symbol_ratio=args.max_symbol_ratio,
        first_user_turn_only=not args.include_followup_turns,
        deduplicate=not args.no_deduplicate,
    )


def main() -> None:
    config = parse_args()
    stats = preprocess_full_dataset(config)
    write_summary(config, stats)

    print(f"Wrote filtered prompts: {config.output_path}")
    print(f"Wrote summary: {config.summary_path}")
    print(json.dumps(dict(sorted(stats.items())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
