"""Root-owned installation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class StepResult:
    step: str
    state: str
    detail: str | None = None
    hard: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProductInstallUnavailable(RuntimeError):
    def __init__(
        self,
        capability: str,
        *,
        detail: str | None = None,
        recommended_next_action: str | None = None,
    ) -> None:
        self.capability = capability
        self.detail = detail
        self.recommended_next_action = recommended_next_action
        super().__init__(detail or capability)


__all__ = ["ProductInstallUnavailable", "StepResult"]
