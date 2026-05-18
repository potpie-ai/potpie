"""LLM-backed answer synthesizer using ``pydantic_ai.Agent`` directly.

We intentionally skip the pydantic-deep heavy agent here — synthesis is a
single-shot structured-output call with no tool use, so the extra framing
would only add latency.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from adapters.outbound.synthesis.prompt import (
    SYNTHESIS_INSTRUCTIONS,
    build_synthesis_prompt,
)
from domain.intelligence_models import IntelligenceBundle
from domain.ports.telemetry import CostEvent, NoOpTelemetry, TelemetryPort

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 8.0


class PydanticAIAnswerSynthesizer:
    """`AnswerSynthesizerPort` backed by a pydantic-ai string-output Agent.

    Graceful degradation: any exception (import, network, validation, timeout)
    logs and returns ``None`` so the envelope falls back to the count string.

    When a :class:`TelemetryPort` is provided, every call emits a
    ``kind="synthesis"`` :class:`CostEvent` with the model id and (when the
    SDK exposes them) input/output token counts plus wall-time latency.
    The most recent run's usage is also exposed via :attr:`last_usage` so the
    resolve envelope can surface ``meta.cost.synthesis``.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_s: float | None = None,
        telemetry: TelemetryPort | None = None,
    ) -> None:
        # Use the Responses endpoint for gpt-5 family models. The
        # synthesis path doesn't use function tools today, so it works on
        # /v1/chat/completions too — but staying on /v1/responses keeps the
        # reconciliation and synthesis paths consistent and tolerates future
        # tool-augmented synthesis.
        self._model = model or os.getenv(
            "CONTEXT_ENGINE_ANSWER_SYNTHESIS_MODEL", "openai-responses:gpt-5.4-mini"
        )
        self._timeout_s = timeout_s if timeout_s is not None else _DEFAULT_TIMEOUT_S
        self._telemetry: TelemetryPort = telemetry or NoOpTelemetry()
        self.last_usage: dict[str, int | str | None] | None = None

    async def synthesize(self, bundle: IntelligenceBundle) -> str | None:
        self.last_usage = None
        try:
            from pydantic_ai import Agent  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("pydantic_ai not installed; skipping answer synthesis")
            return None

        prompt = build_synthesis_prompt(bundle)
        t0 = time.perf_counter()
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
        latency_ms = int((time.perf_counter() - t0) * 1000)

        usage = _extract_usage(result)
        self.last_usage = {
            "model": self._model,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_ms": latency_ms,
        }
        try:
            self._telemetry.record_cost(
                CostEvent(
                    pot_id=bundle.request.pot_id,
                    kind="synthesis",
                    model=self._model,
                    input_tokens=_int_or_none(usage.get("input_tokens")),
                    output_tokens=_int_or_none(usage.get("output_tokens")),
                    total_tokens=_int_or_none(usage.get("total_tokens")),
                    latency_ms=latency_ms,
                )
            )
        except Exception:
            logger.debug("telemetry: synthesis cost emission failed", exc_info=True)

        output = getattr(result, "output", None) or getattr(result, "data", None)
        if not isinstance(output, str):
            return None
        text = output.strip()
        return text or None


def _extract_usage(result: object) -> dict[str, int | None]:
    """Best-effort pydantic-ai usage extraction across SDK versions."""
    usage_callable = getattr(result, "usage", None)
    try:
        u = usage_callable() if callable(usage_callable) else usage_callable
    except Exception:
        u = None
    if u is None:
        return {}
    inp = getattr(u, "input_tokens", None)
    if inp is None:
        inp = getattr(u, "request_tokens", None)
    out = getattr(u, "output_tokens", None)
    if out is None:
        out = getattr(u, "response_tokens", None)
    total = getattr(u, "total_tokens", None)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
