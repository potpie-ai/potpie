from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.models.openai import OpenAIModel, OpenAIStreamedResponse

from app.modules.utils.logger import setup_logger
from app.modules.intelligence.provider.openrouter_usage_context import (
    push_usage as push_usage_context,
    estimate_cost_for_log,
)


logger = setup_logger(__name__)


class OpenRouterGlmStreamedResponse(OpenAIStreamedResponse):
    """
    Streamed response that pushes OpenRouter usage (including cost) to the task context
    when a chunk contains usage information. This mirrors the behavior used for Gemini,
    but without any Gemini-specific tool-signature logic.
    """

    def _map_usage(self, response: Any) -> Any:
        # Capture usage from raw chunk.usage
        u = getattr(response, "usage", None)
        if u is not None:
            # Try to get cost from common OpenRouter fields
            cost = getattr(u, "total_cost", None)
            if cost is None:
                cost = getattr(u, "cost", None)

            # SDK may store extra fields in model_dump / model_extra; try those too
            if cost is None and hasattr(u, "model_dump"):
                try:
                    data = u.model_dump()
                except Exception:
                    data = {}
                cost = data.get("total_cost") or data.get("cost")
                if cost is None:
                    pc = data.get("prompt_cost") or 0
                    cc = data.get("completion_cost") or 0
                    if pc or cc:
                        cost = float(pc) + float(cc)
            if cost is None and getattr(u, "model_extra", None):
                extra = getattr(u, "model_extra", {}) or {}
                cost = extra.get("total_cost") or extra.get("cost")

            prompt_tokens = (
                getattr(u, "prompt_tokens", 0)
                or getattr(u, "input_tokens", 0)
                or 0
            )
            completion_tokens = (
                getattr(u, "completion_tokens", 0)
                or getattr(u, "output_tokens", 0)
                or 0
            )
            total_tokens = getattr(u, "total_tokens", None)
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens

            # Always push when we have usage: use API cost if present, else estimate
            if prompt_tokens or completion_tokens:
                from_api = cost is not None
                if cost is None:
                    cost = estimate_cost_for_log(prompt_tokens, completion_tokens)

                push_usage_context(
                    str(self.model_name),
                    int(prompt_tokens),
                    int(completion_tokens),
                    int(total_tokens),
                    float(cost),
                )

                suffix = "" if from_api else " (estimated)"
                msg = (
                    f"[OpenRouter cost] model={self.model_name} "
                    f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
                    f"cost={cost} credits{suffix}"
                )
                logger.info(msg)
                # Also print so it reliably appears in Celery worker logs
                print(msg, flush=True)

        return super()._map_usage(response)


class OpenRouterGlmModel(OpenAIModel):
    """
    Custom OpenAIModel variant for OpenRouter-backed GLM models.

    It uses a custom streamed response class that extracts OpenRouter usage,
    including cost, from stream chunks and pushes it into the Celery task context.
    """

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        """Use OpenRouterGlmStreamedResponse so we capture cost from stream chunks."""
        return OpenRouterGlmStreamedResponse

