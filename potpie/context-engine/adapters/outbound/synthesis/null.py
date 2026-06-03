"""No-op synthesizer — returns ``None`` so the envelope falls back to counts."""

from __future__ import annotations

from domain.intelligence_models import IntelligenceBundle


class NullAnswerSynthesizer:
    async def synthesize(self, bundle: IntelligenceBundle) -> str | None:
        _ = bundle
        return None
