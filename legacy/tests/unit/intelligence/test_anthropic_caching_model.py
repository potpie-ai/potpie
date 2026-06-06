import pytest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.providers.anthropic import AnthropicProvider

from app.modules.intelligence.provider.anthropic_caching_model import (
    CachingAnthropicModel,
)


def _model() -> CachingAnthropicModel:
    return CachingAnthropicModel(
        model_name="claude-sonnet-4-20250514",
        provider=AnthropicProvider(api_key="test-key"),
    )


@pytest.mark.asyncio
async def test_request_accepts_current_pydantic_ai_request_parameters(monkeypatch):
    model = _model()
    parameters = ModelRequestParameters()
    expected_response = ModelResponse(parts=[TextPart(content="ok")])
    captured = {}

    async def fake_messages_create(
        messages,
        stream,
        model_settings,
        model_request_parameters,
    ):
        captured["messages"] = messages
        captured["stream"] = stream
        captured["model_settings"] = model_settings
        captured["model_request_parameters"] = model_request_parameters
        return object()

    monkeypatch.setattr(model, "_messages_create", fake_messages_create)
    monkeypatch.setattr(model, "_process_response", lambda _: expected_response)

    result = await model.request(
        [ModelRequest(parts=[UserPromptPart(content="hi")])],
        {},
        parameters,
    )

    assert result is expected_response
    assert captured["stream"] is False
    assert isinstance(captured["model_request_parameters"], ModelRequestParameters)
    assert captured["model_request_parameters"].allow_text_output is True


@pytest.mark.asyncio
async def test_messages_create_preserves_structured_system_prompt(monkeypatch):
    model = _model()
    captured = {}

    async def fake_map_message(*_args, **_kwargs):
        return (
            [{"type": "text", "text": "structured system prompt"}],
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(model, "_map_message", fake_map_message)
    model.client.beta.messages.create = fake_create

    await model._messages_create(
        [ModelRequest(parts=[UserPromptPart(content="hi")])],
        False,
        {},
        ModelRequestParameters(),
    )

    assert captured["system"] == [
        {
            "type": "text",
            "text": "structured system prompt",
            "cache_control": {"type": "ephemeral"},
        }
    ]
