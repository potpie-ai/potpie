from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic_ai.messages import ToolCallPart, ModelMessage, ModelResponse
from pydantic_ai.models.openai import OpenAIModel, OpenAIStreamedResponse
from pydantic_ai.models import ModelRequestParameters

from app.modules.utils.logger import setup_logger
from app.modules.intelligence.provider.openrouter_usage_context import (
    push_usage as push_usage_context,
    estimate_cost_for_log,
)


# ------------------------------------------------------------------
# DEBUG CONFIG
# ------------------------------------------------------------------

DEBUG_FLOW = False  # set True to trace flow (init, _process_response, etc.)


def dbg(msg: str):
    if DEBUG_FLOW:
        logging.warning(f"[DEBUG-FLOW] {msg}")
        print(f"🧭 [DEBUG-FLOW] {msg}")


# ------------------------------------------------------------------
# LOGGING SETUP (use app logger so logs show in Celery worker too)
# ------------------------------------------------------------------

logger = setup_logger(__name__)


# ------------------------------------------------------------------
# STREAMED RESPONSE: capture OpenRouter cost from stream chunks
# ------------------------------------------------------------------


class OpenRouterStreamedResponse(OpenAIStreamedResponse):
    """
    Streamed response that pushes OpenRouter usage (including cost) to the task context
    when a chunk contains usage with cost (OpenRouter sends this in the last chunk).
    """

    def _map_usage(self, response: Any) -> Any:
        # Capture cost from raw chunk.usage (OpenRouter returns prompt_cost, completion_cost, total_cost)
        u = getattr(response, "usage", None)
        if u is not None:
            cost = getattr(u, "total_cost", None)
            if cost is None:
                cost = getattr(u, "cost", None)
            # SDK may strip extra fields; try model_dump() / model_extra for OpenRouter cost
            if cost is None and hasattr(u, "model_dump"):
                try:
                    data = u.model_dump()
                    cost = data.get("total_cost") or data.get("cost")
                    if cost is None:
                        pc = data.get("prompt_cost") or 0
                        cc = data.get("completion_cost") or 0
                        if pc or cc:
                            cost = float(pc) + float(cc)
                except Exception:
                    pass
            if cost is None and getattr(u, "model_extra", None):
                cost = u.model_extra.get("total_cost") or u.model_extra.get("cost")

            prompt_tokens = getattr(u, "prompt_tokens", 0) or getattr(u, "input_tokens", 0) or 0
            completion_tokens = getattr(u, "completion_tokens", 0) or getattr(u, "output_tokens", 0) or 0
            total_tokens = getattr(u, "total_tokens", None)
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens

            # Always push when we have usage: use API cost if present, else estimate so logs show a cost
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
                print(msg, flush=True)
        return super()._map_usage(response)


class OpenRouterGeminiModel(OpenAIModel):
    """
    Custom OpenAIModel variant that:
    1. Adds OpenRouter's required Gemini-specific metadata (thought_signature)
    2. Tracks token usage and costs from OpenRouter's usage object
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        dbg("ENTER __init__ (OpenRouterGeminiModel)")
        super().__init__(*args, **kwargs)
        dbg("RETURNED from super().__init__")

        self._tool_call_signatures: Dict[str, str] = {}
        self._tool_call_signature_counter: int = 0

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        self.total_prompt_cost = 0.0
        self.total_completion_cost = 0.0
        self.total_cost = 0.0

        self.request_count = 0

        dbg("EXIT __init__")

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        """Use OpenRouterStreamedResponse so we capture cost from stream chunks (OpenRouter sends usage in last chunk)."""
        return OpenRouterStreamedResponse

    # ------------------------------------------------------------------
    # SIGNATURE EXTRACTION
    # ------------------------------------------------------------------

    def _capture_signatures_from_response(self, response) -> None:
        dbg("ENTER _capture_signatures_from_response")

        try:
            choice = response.choices[0]
        except (AttributeError, IndexError):
            dbg("No choices found on response")
            return

        tool_calls = getattr(choice.message, "tool_calls", None) or []
        dbg(f"Tool calls found: {len(tool_calls)}")

        for call in tool_calls:
            signature = self._extract_signature_from_tool_call(call)
            if signature:
                dbg(f"Captured signature for tool_call_id={call.id}")
                self._tool_call_signatures[call.id] = signature

        dbg("EXIT _capture_signatures_from_response")

    def _extract_signature_from_tool_call(self, tool_call: Any) -> str | None:
        dbg("ENTER _extract_signature_from_tool_call")

        function_obj = getattr(tool_call, "function", None)
        if function_obj is None:
            dbg("No function object on tool call")
            return None

        for accessor in (
            getattr(function_obj, "thought_signature", None),
            getattr(function_obj, "model_dump", None) and function_obj.model_dump(),
        ):
            if isinstance(accessor, str) and accessor:
                dbg("Found thought_signature as string")
                return accessor
            if isinstance(accessor, dict):
                signature = accessor.get("thought_signature") or accessor.get(
                    "thoughtSignature"
                )
                if signature:
                    dbg("Found thought_signature inside dict")
                    return signature

        dbg("No thought_signature found")
        return None

    def _get_or_create_tool_call_signature(self, tool_call_id: str) -> str:
        dbg(f"ENTER _get_or_create_tool_call_signature ({tool_call_id})")

        if tool_call_id in self._tool_call_signatures:
            dbg("Signature already exists")
            return self._tool_call_signatures[tool_call_id]

        signature = f"{tool_call_id}-{self._tool_call_signature_counter:08d}"
        self._tool_call_signatures[tool_call_id] = signature
        self._tool_call_signature_counter += 1

        dbg(f"Generated new signature: {signature}")
        return signature

    # ------------------------------------------------------------------
    # USAGE EXTRACTION
    # ------------------------------------------------------------------

    def _extract_and_log_usage(self, response) -> None:
        """Extract usage from OpenRouter response and log. OpenRouter returns usage.cost (credits)."""
        self.request_count += 1

        usage = getattr(response, "usage", None)
        if not usage:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        # OpenRouter returns a single "cost" in credits (not prompt_cost/completion_cost)
        cost = getattr(usage, "cost", None)
        if cost is None:
            cost = getattr(usage, "total_cost", 0.0) or 0.0

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.total_cost += float(cost) if cost is not None else 0.0

        cost_str = f", cost={cost} credits" if cost is not None else ""
        msg = (
            f"[OpenRouter usage] model={self.model_name} "
            f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
            f"total_tokens={total_tokens}{cost_str}"
        )
        logger.info(msg)
        # Guarantee visibility in Celery worker (conversation flow runs there)
        print(msg, flush=True)
        # Collect for stream end event so API (uvicorn) can log it too
        push_usage_context(
            self.model_name,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            float(cost) if cost is not None else None,
        )

    # ------------------------------------------------------------------
    # RESPONSE PROCESSING (CRITICAL CONTROL POINT)
    # ------------------------------------------------------------------

    def _process_response(self, response):
        dbg("ENTER _process_response (custom)")

        dbg("Calling _extract_and_log_usage")
        self._extract_and_log_usage(response)

        dbg("Calling _capture_signatures_from_response")
        self._capture_signatures_from_response(response)

        dbg("CALLING super()._process_response")
        model_response = super()._process_response(response)
        dbg("RETURNED from super()._process_response")

        dbg(f"ModelResponse parts count = {len(model_response.parts)}")

        for part in model_response.parts:
            dbg(f"Inspecting part type={type(part).__name__}")
            if isinstance(part, ToolCallPart):
                dbg(f"ToolCallPart detected: {part.tool_call_id}")
                signature = self._tool_call_signatures.get(part.tool_call_id)
                if signature:
                    dbg(f"Attaching thought_signature={signature}")
                    setattr(part, "thought_signature", signature)

        dbg("EXIT _process_response")
        return model_response

    # ------------------------------------------------------------------
    # TOOL CALL MAPPING (OUTBOUND)
    # ------------------------------------------------------------------

    def _map_tool_call(self, t: ToolCallPart):
        dbg(f"ENTER _map_tool_call (tool_call_id={t.tool_call_id})")

        tool_call_param = super()._map_tool_call(t)
        dbg("RETURNED from super()._map_tool_call")

        function_payload = tool_call_param.get("function")
        dbg(f"Function payload pre-injection: {function_payload}")

        if isinstance(function_payload, dict):
            signature = getattr(t, "thought_signature", None) or self._tool_call_signatures.get(
                t.tool_call_id
            )

            if not signature:
                dbg("No signature found — generating one")
                signature = self._get_or_create_tool_call_signature(t.tool_call_id)

            function_payload["thought_signature"] = signature
            self._tool_call_signatures[t.tool_call_id] = signature

            dbg(f"Injected thought_signature={signature}")

        dbg("EXIT _map_tool_call")
        return tool_call_param

    # ------------------------------------------------------------------
    # MESSAGE MAPPING (ASYNC BOUNDARY)
    # ------------------------------------------------------------------

    async def _map_messages(
        self,
        messages: List[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> List[Any]:
        dbg("ENTER _map_messages")

        dbg(f"Messages count = {len(messages)}")

        for i, message in enumerate(messages):
            dbg(f"Message[{i}] type={type(message).__name__}")

            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        dbg(f"ToolCallPart in history: {part.tool_call_id}")

                        if not getattr(part, "thought_signature", None):
                            dbg("Missing signature in history — attaching")
                            signature = self._tool_call_signatures.get(
                                part.tool_call_id
                            ) or self._get_or_create_tool_call_signature(
                                part.tool_call_id
                            )
                            setattr(part, "thought_signature", signature)

        dbg("CALLING super()._map_messages")
        result = await super()._map_messages(messages, model_request_parameters)
        dbg("RETURNED from super()._map_messages")

        dbg("EXIT _map_messages")
        return result

    # ------------------------------------------------------------------
    # USAGE SUMMARY
    # ------------------------------------------------------------------

    def get_usage_summary(self) -> Dict[str, Any]:
        dbg("ENTER get_usage_summary")
        return {
            "total_requests": self.request_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_prompt_cost": self.total_prompt_cost,
            "total_completion_cost": self.total_completion_cost,
            "total_cost": self.total_cost,
            "average_tokens_per_request": (
                self.total_tokens / self.request_count
                if self.request_count > 0
                else 0
            ),
            "average_cost_per_request": (
                self.total_cost / self.request_count
                if self.request_count > 0
                else 0
            ),
        }

    def reset_counters(self) -> None:
        dbg("RESETTING counters")

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_prompt_cost = 0.0
        self.total_completion_cost = 0.0
        self.total_cost = 0.0
        self.request_count = 0

        logger.info("[OpenRouter Usage] Counters reset")



