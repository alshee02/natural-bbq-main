"""Response state enum and parse result dataclass for BBQ bias evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ResponseState(str, Enum):
    """Final unified response state for all conditions (Section 15)."""
    CHOICE_0 = "CHOICE_0"
    CHOICE_1 = "CHOICE_1"
    CHOICE_2 = "CHOICE_2"
    REFUSAL = "REFUSAL"
    NO_ANSWER = "NO_ANSWER"
    CONFLICT = "CONFLICT"
    TECHNICAL_UNRESOLVED = "TECHNICAL_UNRESOLVED"

    @classmethod
    def from_label(cls, label: Optional[int]) -> ResponseState:
        if label == 0:
            return cls.CHOICE_0
        if label == 1:
            return cls.CHOICE_1
        if label == 2:
            return cls.CHOICE_2
        return cls.TECHNICAL_UNRESOLVED


@dataclass
class ParseResult:
    """Result from any parser stage."""
    pred_label: Optional[int]
    parse_status: str
    parse_evidence: str
    parse_ok: bool

    @classmethod
    def valid(cls, label: int, status: str, evidence: str = "") -> ParseResult:
        return cls(pred_label=label, parse_status=status,
                   parse_evidence=evidence, parse_ok=True)

    @classmethod
    def invalid(cls, status: str, evidence: str = "") -> ParseResult:
        return cls(pred_label=None, parse_status=status,
                   parse_evidence=evidence, parse_ok=False)
