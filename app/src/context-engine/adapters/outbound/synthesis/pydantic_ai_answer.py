"""LLM-backed answer synthesizer using ``pydantic_ai.Agent`` directly.

We intentionally skip the pydantic-deep heavy agent here — synthesis is a
single-shot structured-output call with no tool use, so the extra framing
would only add latency.
"""

from __future__ import annotations

import asyncio
import logging
import os

from adapters.outbound.synthesis.prompt import (
    SYNTHESIS_INSTRUCTIONS,
    build_synthesis_prompt,
)
from domain.intelligence_models import IntelligenceBundle

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 8.0


class PydanticAIAnswerSynthesizer:
    """`AnswerSynthesizerPort` backed by a pydantic-ai string-output Agent.

    Graceful degradation: any exception (import, network, validation, timeout)
    logs and returns ``None`` so the envelope falls back to the count string.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._model = model or os.getenv(
            "CONTEXT_ENGINE_ANSWER_SYNTHESIS_MODEL", "openai:gpt-5.4-mini"
        )
        self._timeout_s = timeout_s if timeout_s is not None else _DEFAULT_TIMEOUT_S

    async def synthesize(self, bundle: IntelligenceBundle) -> str | None:
        try:
            from pydantic_ai import Agent  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("pydantic_ai not installed; skipping answer synthesis")
            return None

        prompt = build_synthesis_prompt(bundle)
        try:
            agent = Agent(
                self._model,
                output_type=str,
                system_prompt=SYNTHESIS_INSTRUCTIONS,
            )
            result = await asyncio.wait_for(
                agent.run(prompt), timeout=self._timeout_s
            )
        except Exception:
            logger.exception("answer synthesis failed; falling back to count summary")
            return None

        output = getattr(result, "output", None) or getattr(result, "data", None)
        if not isinstance(output, str):
            return None
        text = output.strip()
        return text or None
