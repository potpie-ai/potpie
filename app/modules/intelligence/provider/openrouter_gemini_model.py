from __future__ import annotations

from typing import Any, Dict, List

from pydantic_ai.messages import ToolCallPart, ModelMessage, ModelResponse
from pydantic_ai.models.openai import OpenAIModel, chat
from pydantic_ai.models import ModelRequestParameters


class OpenRouterGeminiModel(OpenAIModel):
    """
    Custom OpenAIModel variant that adds OpenRouter's required Gemini-specific metadata.

    Gemini models accessed through OpenRouter expect every function/tool call to include a
    `thought_signature`. OpenRouter forwards the signature produced by Gemini in the response
    and requires clients to send it back on subsequent requests. If the signature is missing,
    Google rejects the request with INVALID_ARGUMENT, which is what we observed in production.

    This wrapper captures any `thought_signature` present in tool call responses and ensures that
    every assistant tool call message sent back to OpenRouter includes the signature (falling back
    to a deterministic one if the provider did not supply it).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_call_signatures: Dict[str, str] = {}
        self._tool_call_signature_counter: int = 0

    def _capture_signatures_from_response(self, response: chat.ChatCompletion) -> None:
        """Extract thought signatures from the raw OpenRouter response if present."""
        try:
            choice = response.choices[0]
        except (AttributeError, IndexError):
            return

        tool_calls = getattr(choice.message, "tool_calls", None) or []
        for call in tool_calls:
            signature = self._extract_signature_from_tool_call(call)
            if signature:
                self._tool_call_signatures[call.id] = signature

    def _extract_signature_from_tool_call(self, tool_call: Any) -> str | None:
        """Return the `thought_signature` field from a tool call if it exists."""
        function_obj = getattr(tool_call, "function", None)
        if function_obj is None:
            return None

        # OpenRouter surfaces provider-specific fields via model_dump extras.
        for accessor in (
            getattr(function_obj, "thought_signature", None),
            getattr(function_obj, "model_dump", None) and function_obj.model_dump(),
        ):
            if isinstance(accessor, str) and accessor:
                return accessor
            if isinstance(accessor, dict):
                signature = accessor.get("thought_signature") or accessor.get(
                    "thoughtSignature"
                )
                if signature:
                    return signature
        return None

    def _get_or_create_tool_call_signature(self, tool_call_id: str) -> str:
        """Get or create a deterministic signature for a tool call.

        Args:
            tool_call_id: The tool call identifier

        Returns:
            A deterministic signature string in the format "{tool_call_id}-{counter:08d}"
        """
        if tool_call_id in self._tool_call_signatures:
            return self._tool_call_signatures[tool_call_id]

        # Generate a deterministic signature using a monotonic counter
        signature = f"{tool_call_id}-{self._tool_call_signature_counter:08d}"
        self._tool_call_signatures[tool_call_id] = signature
        self._tool_call_signature_counter += 1
        return signature

    def _process_response(self, response: chat.ChatCompletion):
        """Capture tool call signatures before letting the base class build the response."""
        self._capture_signatures_from_response(response)
        model_response = super()._process_response(response)

        # Attach the captured signatures to the ToolCallPart so we can re-use them later.
        for part in model_response.parts:
            if isinstance(part, ToolCallPart):
                signature = self._tool_call_signatures.get(part.tool_call_id)
                if signature:
                    setattr(part, "thought_signature", signature)
        return model_response

    def _map_tool_call(self, t: ToolCallPart):
        """Inject `thought_signature` metadata into every tool call we send back."""
        tool_call_param = super()._map_tool_call(t)
        function_payload = tool_call_param.get("function")
        if isinstance(function_payload, dict):
            signature: str | None = getattr(
                t, "thought_signature", None
            ) or self._tool_call_signatures.get(t.tool_call_id)
            if not signature:
                # Generate a deterministic placeholder so the provider always receives a signature.
                signature = self._get_or_create_tool_call_signature(t.tool_call_id)
            self._tool_call_signatures[t.tool_call_id] = signature
            # Always set the signature, even if it was already there (ensures it's present)
            function_payload["thought_signature"] = signature
        return tool_call_param

    async def _map_messages(
        self,
        messages: List[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> List[chat.ChatCompletionMessageParam]:
        """Override to ensure all tool calls have thought_signature before mapping."""
        # Ensure model_request_parameters is a ModelRequestParameters object, not a dict
        # This can happen if pydantic-ai passes it as a dict in some cases
        if isinstance(model_request_parameters, dict):
            model_request_parameters = ModelRequestParameters(
                **model_request_parameters
            )

        # First, ensure all ToolCallParts in message history have signatures
        for message in messages:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        # Ensure the ToolCallPart has a signature attribute
                        if not hasattr(part, "thought_signature") or not getattr(
                            part, "thought_signature"
                        ):
                            # Try to get from cache, or generate a new one
                            signature = self._tool_call_signatures.get(
                                part.tool_call_id
                            )
                            if not signature:
                                signature = self._get_or_create_tool_call_signature(
                                    part.tool_call_id
                                )
                            setattr(part, "thought_signature", signature)

        # Now call the parent implementation which will use our _map_tool_call override
        return await super()._map_messages(messages, model_request_parameters)
