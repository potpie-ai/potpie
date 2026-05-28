"""Async port for synthesizing ``goal=answer`` summaries from a resolved bundle.

Implementations translate an :class:`IntelligenceBundle` into a short natural
language summary that replaces the canned count-string ``answer.summary``.
Returning ``None`` causes the envelope to fall back to the deterministic count
string so callers always receive a non-empty summary even when the LLM is
offline, unconfigured, or failing.
"""

from __future__ import annotations

from typing import Protocol

from domain.intelligence_models import IntelligenceBundle


class AnswerSynthesizerPort(Protocol):
    async def synthesize(self, bundle: IntelligenceBundle) -> str | None:
        """Return a synthesized summary, or ``None`` to trigger the count fallback."""
        ...
