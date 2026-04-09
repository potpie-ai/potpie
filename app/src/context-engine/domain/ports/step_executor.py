"""Execution boundary: apply one planned step deterministically."""

from __future__ import annotations

from typing import Protocol

from domain.ingestion_event_models import EpisodeStep, ExecutionResult


class StepExecutor(Protocol):
    """Single-step deterministic application (invokes :class:`ContextGraphWriter` internally)."""

    def execute(self, step: EpisodeStep) -> ExecutionResult:
        ...
