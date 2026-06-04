"""Evaluator protocol shared across all three axes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    score: float  # 0..100
    passed: bool
    details: dict[str, object] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
