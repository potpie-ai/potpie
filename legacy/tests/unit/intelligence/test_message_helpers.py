from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart

from app.modules.intelligence.agents.chat_agents import message_helpers


pytestmark = pytest.mark.unit


def test_is_user_message_only_matches_model_requests_with_user_prompt():
    user_msg = ModelRequest(parts=[UserPromptPart(content="hello")])
    system_msg = ModelRequest(parts=[SystemPromptPart(content="system")])
    response_msg = ModelResponse(parts=[TextPart(content="answer")])

    assert message_helpers.is_user_message(user_msg) is True
    assert message_helpers.is_user_message(system_msg) is False
    assert message_helpers.is_user_message(response_msg) is False


def test_tool_message_classification_for_request_and_response_shapes():
    request_tool_call = ModelRequest(
        parts=[SimpleNamespace(tool_name="search", tool_call_id="call-1", args={"q": "x"})]
    )
    request_tool_result = ModelRequest(
        parts=[SimpleNamespace(tool_name="search", tool_call_id="call-1", content="done")]
    )
    nested_tool_call = ModelRequest(
        parts=[
            SimpleNamespace(
                part=SimpleNamespace(tool_name="search", tool_call_id="call-2", args="{}")
            )
        ]
    )
    response_tool_call = ModelResponse(
        parts=[SimpleNamespace(part_kind="tool-call", tool_name="search", tool_call_id="call-3")]
    )
    response_tool_result = ModelResponse(
        parts=[SimpleNamespace(tool_name="search", tool_call_id="call-3", content="result")]
    )

    assert message_helpers.is_tool_call_message(request_tool_call) is True
    assert message_helpers.is_tool_call_message(request_tool_result) is False
    assert message_helpers.is_tool_call_message(nested_tool_call) is True
    assert message_helpers.is_tool_call_message(response_tool_call) is True

    assert message_helpers.is_tool_result_message(request_tool_result) is True
    assert message_helpers.is_tool_result_message(request_tool_call) is False
    assert message_helpers.is_tool_result_message(response_tool_result) is True
    assert message_helpers.is_tool_result_message(response_tool_call) is False


def test_is_llm_response_message_requires_non_empty_text():
    assert (
        message_helpers.is_llm_response_message(
            ModelResponse(parts=[TextPart(content="  useful answer  ")])
        )
        is True
    )
    assert (
        message_helpers.is_llm_response_message(
            ModelResponse(parts=[TextPart(content="   ")])
        )
        is False
    )
    assert (
        message_helpers.is_llm_response_message(
            ModelRequest(parts=[UserPromptPart(content="hello")])
        )
        is False
    )


def test_extract_tool_call_ids_collects_top_level_and_nested_ids():
    request_msg = ModelRequest(
        parts=[
            SimpleNamespace(tool_call_id="call-1"),
            SimpleNamespace(part=SimpleNamespace(tool_call_id="call-2")),
        ]
    )
    response_msg = ModelResponse(
        parts=[
            SimpleNamespace(tool_call_id="call-3"),
            SimpleNamespace(result=SimpleNamespace(tool_call_id="call-4")),
        ]
    )

    assert message_helpers.extract_tool_call_ids(request_msg) == {"call-1", "call-2"}
    assert message_helpers.extract_tool_call_ids(response_msg) == {"call-3", "call-4"}


def test_extract_tool_call_info_supports_request_nested_and_response_parts():
    request_msg = ModelRequest(
        parts=[SimpleNamespace(tool_name="search", tool_call_id="call-1", args={"q": "abc"})]
    )
    nested_request_msg = ModelRequest(
        parts=[
            SimpleNamespace(
                part=SimpleNamespace(tool_name="bash", tool_call_id="call-2", args=["ls", "-la"])
            )
        ]
    )
    response_msg = ModelResponse(
        parts=[SimpleNamespace(part_kind="tool-call", tool_name="open", tool_call_id="call-3", args="{}")]
    )

    assert message_helpers.extract_tool_call_info(request_msg) == (
        "search",
        '{"q": "abc"}',
        "call-1",
    )
    assert message_helpers.extract_tool_call_info(nested_request_msg) == (
        "bash",
        "['ls', '-la']",
        "call-2",
    )
    assert message_helpers.extract_tool_call_info(response_msg) == (
        "open",
        "{}",
        "call-3",
    )


def test_extract_tool_result_info_supports_result_objects_and_content_strings():
    request_msg = ModelRequest(
        parts=[
            SimpleNamespace(
                tool_name="search",
                tool_call_id="call-1",
                result=SimpleNamespace(content="request result"),
            )
        ]
    )
    response_msg = ModelResponse(
        parts=[SimpleNamespace(tool_name="search", tool_call_id="call-2", content="response result")]
    )
    empty_msg = ModelRequest(
        parts=[SimpleNamespace(tool_name="search", tool_call_id="call-3", content="   ")]
    )

    assert message_helpers.extract_tool_result_info(request_msg) == (
        "search",
        "request result",
        "call-1",
    )
    assert message_helpers.extract_tool_result_info(response_msg) == (
        "search",
        "response result",
        "call-2",
    )
    assert message_helpers.extract_tool_result_info(empty_msg) is None


def test_serialize_messages_to_text_includes_json_fallback_and_text_parts():
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="system"),
                UserPromptPart(content={"prompt": "user"}),
                SimpleNamespace(tool_name="search", tool_call_id="call-1", args={"q": "abc"}),
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(content="answer"),
                SimpleNamespace(tool_name="search", tool_call_id="call-1", content="done"),
            ]
        ),
    ]

    serialized = message_helpers.serialize_messages_to_text(messages)

    assert "system" in serialized
    assert "{'prompt': 'user'}" in serialized
    assert '"tool_name": "search"' in serialized
    assert "answer" in serialized


def test_extract_system_prompt_and_ensure_history_ends_with_model_request():
    messages = [
        ModelRequest(parts=[SystemPromptPart(content="system one")]),
        ModelRequest(parts=[SystemPromptPart(content={"note": "second"})]),
        ModelResponse(parts=[TextPart(content="answer")]),
    ]

    system_prompt = message_helpers.extract_system_prompt_from_messages(messages)
    ensured = message_helpers.ensure_history_ends_with_model_request(messages)

    assert system_prompt == "system one\n{'note': 'second'}"
    assert isinstance(ensured[-1], ModelRequest)
    assert ensured[-1].parts[0].content == ""
    assert message_helpers.ensure_history_ends_with_model_request([]) == []
    assert (
        message_helpers.ensure_history_ends_with_model_request(
            [ModelRequest(parts=[UserPromptPart(content="keep me")])]
        )[-1].parts[0].content
        == "keep me"
    )
