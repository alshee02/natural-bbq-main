import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data/processed/rewrite_batch/bbq_rewrite_batch_outputs.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/rewrite_batch/bbq_rewrite_hard_checks.parquet"
MIN_LENGTH_RATIO = 0.4
MIN_ASCII_RATIO = 0.9
EXACT_COPY_THRESHOLD = 0.95  # normalized char overlap ratio

FORBIDDEN_PHRASES = [
    "avoid stereotypes",
    "avoid making assumptions",
    "do not make assumptions",
    "based only on the information",
    "be unbiased",
    "be fair",
    "not enough information probably",
    "answer without bias",
]

# benchmark_format_leak
BENCHMARK_LABEL_RE = re.compile(
    r"(?m)^\s*(context|question|options?|answer|choices?)\s*:",
    re.IGNORECASE,
)
OPTION_LABEL_RE = re.compile(r"\([ABC]\)\s|\b[ABC]\.\s")

# answer_leakage — signals that the rewrite reveals or argues for an answer
ANSWER_SIGNAL_RE = re.compile(
    r"\b(the answer is|answer\s*:\s*[ABC012]|the correct (answer|option|choice)|"
    r"therefore .{0,30} (is|was|did|didn't)|so .{0,30} (is|was|did|didn't)|"
    r"clearly .{0,30} (is|was)|obviously .{0,30} (is|was)|"
    r"it (must|has to) be|it('s| is) (obvious|clear) that)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Existing checks
# ---------------------------------------------------------------------------

def check_forbidden_phrases(
    original_context: str,
    original_question: str,
    natural_question: str,
) -> str | None:
    original_lower = f"{original_context or ''} {original_question or ''}".lower()
    natural_lower = (natural_question or "").lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in natural_lower and phrase not in original_lower:
            return "forbidden_phrase"
    return None


def ascii_ratio(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 1.0
    return sum(1 for ch in text if ord(ch) < 128) / len(text)


def check_language(natural_question: str) -> str | None:
    if ascii_ratio(natural_question) < MIN_ASCII_RATIO:
        return "non_english"
    return None


def check_length_sanity(
    original_context: str,
    original_question: str,
    ans0: str,
    ans1: str,
    natural_question: str,
) -> str | None:
    natural = (natural_question or "").strip()
    if not natural:
        return "empty_rewrite"
    original_len = len(
        f"{original_context or ''} {original_question or ''} {ans0 or ''} {ans1 or ''}".strip()
    )
    if original_len == 0:
        return None
    if len(natural) / original_len < MIN_LENGTH_RATIO:
        return "too_short"
    return None


# ---------------------------------------------------------------------------
# New checks
# ---------------------------------------------------------------------------

def check_invalid_output_schema(natural_question: str) -> str | None:
    """Catches None, non-string, or whitespace-only natural_question."""
    if not isinstance(natural_question, str) or not natural_question.strip():
        return "invalid_output_schema"
    return None



def check_benchmark_format_leak(natural_question: str) -> str | None:
    """
    Detects residual benchmark structure:
    - Section labels like 'Context:', 'Question:', 'Options:', 'Answer:'
    - Option labels like '(A)', '(B)', 'A. ', 'B. '
    """
    text = natural_question or ""
    if BENCHMARK_LABEL_RE.search(text):
        return "benchmark_format_leak"
    if OPTION_LABEL_RE.search(text):
        return "benchmark_format_leak"
    return None


def check_answer_leakage(natural_question: str) -> str | None:
    """
    Flags rewrites that explicitly reveal or argue toward an answer
    (e.g. 'the answer is', 'clearly X is', 'therefore X did').
    """
    if ANSWER_SIGNAL_RE.search(natural_question or ""):
        return "answer_leakage"
    return None


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def check_exact_copy(
    original_context: str,
    original_question: str,
    natural_question: str,
) -> str | None:
    """
    Flags rewrites that are near-identical to the concatenation of the
    original context and question (no meaningful style transformation occurred).

    Uses character-level overlap: len(common prefix) / len(shorter string).
    """
    original = _normalize(f"{original_context} {original_question}")
    natural = _normalize(natural_question)

    if not original or not natural:
        return None

    shorter = min(len(original), len(natural))
    common = sum(a == b for a, b in zip(original, natural))
    overlap = common / shorter

    if overlap >= EXACT_COPY_THRESHOLD:
        return "exact_copy"
    return None


# ---------------------------------------------------------------------------
# Retry routing
# ---------------------------------------------------------------------------

# Rules indicating potential meaning corruption → retry with original prompt
SEMANTIC_FAILURE_RULES: frozenset[str] = frozenset({
    "rewrite_failed",
    "invalid_output_schema",
    "forbidden_phrase",
    "non_english",
    "empty_rewrite",
    "too_short",
    "answer_leakage",
    "benchmark_format_leak",
})

# Rules indicating style-only failure → retry with augmented prompt
STYLE_FAILURE_RULES: frozenset[str] = frozenset({"exact_copy"})


def classify_status(rules: list[str]) -> str:
    """Three-way status for retry routing.

    - "pass"                   : no rules triggered
    - "retry_semantic_failure" : at least one rule that may indicate meaning
                                 corruption; retry with the original prompt
    - "retry_style_failure"    : only style rules (exact_copy); retry with an
                                 augmented prompt requesting greater variation
    """
    if not rules:
        return "pass"
    if any(r in SEMANTIC_FAILURE_RULES for r in rules):
        return "retry_semantic_failure"
    if any(r in STYLE_FAILURE_RULES for r in rules):
        return "retry_style_failure"
    return "retry_semantic_failure"  # unknown rule → conservative


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_hard_checks(row: pd.Series) -> list[str]:
    if row["rewrite_status"] != "success":
        return ["rewrite_failed"]

    nq = row["natural_question"]
    checks = [
        # schema first — remaining checks assume nq is a non-empty string
        check_invalid_output_schema(nq),
        check_forbidden_phrases(row["original_context"], row["original_question"], nq),
        check_language(nq),
        check_length_sanity(row["original_context"], row["original_question"], row["ans0"], row["ans1"], nq),
        check_benchmark_format_leak(nq),
        check_answer_leakage(nq),
        check_exact_copy(row["original_context"], row["original_question"], nq),
    ]
    return [rule for rule in checks if rule is not None]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df["failed_rules"] = df.apply(run_hard_checks, axis=1)
    df["hard_check_status"] = df["failed_rules"].apply(classify_status)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(df["hard_check_status"].value_counts())
    rule_counts: dict[str, int] = {}
    for rules in df["failed_rules"]:
        for rule in rules:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    print(rule_counts)


if __name__ == "__main__":
    main()
