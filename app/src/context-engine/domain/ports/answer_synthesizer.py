"""Async port for synthesizing ``goal=answer`` summaries from a resolved envelope.

Implementations translate an :class:`AgentEnvelope` into a short natural-language
summary that replaces the canned fallback ``answer.summary``. Returning ``None``
causes the caller to fall back to a deterministic summary, so callers always
receive a non-empty answer even when the LLM is offline, unconfigured, or failing.
"""

from __future__ import annotations

from typing import Protocol

from domain.agent_envelope import AgentEnvelope


class AnswerSynthesizerPort(Protocol):
    async def synthesize(self, envelope: AgentEnvelope) -> str | None:
        """Return a synthesized summary, or ``None`` to trigger the fallback."""
        ...
