"""Engine-only provisioning value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

DONE = "done"
SKIPPED = "skipped"
FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StepResult:
    step: str
    state: str
    detail: str | None = None
    hard: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["DONE", "FAILED", "SKIPPED", "StepResult"]
