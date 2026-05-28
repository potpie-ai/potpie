"""No-op synthesizer — returns ``None`` so the caller falls back to a summary."""

from __future__ import annotations

from domain.agent_envelope import AgentEnvelope


class NullAnswerSynthesizer:
    async def synthesize(self, envelope: AgentEnvelope) -> str | None:
        _ = envelope
        return None
