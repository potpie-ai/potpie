"""Tests for the pydantic-ai message → work-event normalizer."""

from __future__ import annotations

import pytest

from potpie_context_engine.application.services.agent_work_events import build_work_events

pytestmark = pytest.mark.unit


def _request(*parts: dict) -> dict:
    return {"kind": "request", "parts": list(parts)}


def _response(*parts: dict) -> dict:
    return {"kind": "response", "parts": list(parts)}


def test_emits_prompt_from_explicit_field_even_when_messages_missing() -> None:
    rows = build_work_events(
        prompt="hello batch",
        messages_json=None,
        final_response=None,
        error=None,
    )
    assert [r.event_kind for r in rows] == ["prompt"]
    assert rows[0].body == "hello batch"


def test_falls_back_to_first_user_prompt_in_history() -> None:
    history = [
        _request(
            {"part_kind": "system-prompt", "content": "sys"},
            {"part_kind": "user-prompt", "content": "fallback"},
        ),
        _response({"part_kind": "text", "content": "done"}),
    ]
    rows = build_work_events(
        prompt=None,
        messages_json=history,
        final_response="done",
        error=None,
    )
    prompt = rows[0]
    assert prompt.event_kind == "prompt"
    assert prompt.body == "fallback"
    assert prompt.payload["system_prompts"] == ["sys"]


def test_tool_call_and_return_are_paired_in_order() -> None:
    history = [
        _request({"part_kind": "user-prompt", "content": "go"}),
        _response(
            {"part_kind": "text", "content": "let me look"},
            {
                "part_kind": "tool-call",
                "tool_name": "context_search",
                "args": {"query": "foo"},
                "tool_call_id": "call_1",
            },
        ),
        _request(
            {
                "part_kind": "tool-return",
                "tool_name": "context_search",
                "tool_call_id": "call_1",
                "content": [{"id": "x", "summary": "y"}],
            }
        ),
        _response({"part_kind": "text", "content": "all set"}),
    ]
    rows = build_work_events(
        prompt="go",
        messages_json=history,
        final_response="all set",
        error=None,
    )
    kinds = [r.event_kind for r in rows]
    assert kinds == [
        "prompt",
        "model_messages",
        "tool_call",
        "tool_result",
        "plan_output",
    ]
    tool_call = rows[2]
    assert tool_call.title == "context_search"
    assert tool_call.payload["args"] == {"query": "foo"}
    tool_result = rows[3]
    assert tool_result.title == "context_search"
    assert "summary" in (tool_result.body or "")
    assert rows[4].title == "Final response"


def test_thinking_part_emitted_as_model_message() -> None:
    history = [
        _request({"part_kind": "user-prompt", "content": "?"}),
        _response(
            {"part_kind": "thinking", "content": "reasoning..."},
            {"part_kind": "text", "content": "done"},
        ),
    ]
    rows = build_work_events(
        prompt="?",
        messages_json=history,
        final_response="done",
        error=None,
    )
    thinking_rows = [r for r in rows if r.title == "Thinking"]
    assert len(thinking_rows) == 1
    assert thinking_rows[0].event_kind == "model_messages"
    assert thinking_rows[0].body == "reasoning..."


def test_error_is_appended_as_trailing_event() -> None:
    rows = build_work_events(
        prompt="go",
        messages_json=None,
        final_response=None,
        error="boom",
    )
    assert rows[-1].event_kind == "error"
    assert rows[-1].body == "boom"


def test_only_last_text_is_promoted_to_plan_output() -> None:
    history = [
        _request({"part_kind": "user-prompt", "content": "go"}),
        _response({"part_kind": "text", "content": "step 1"}),
        _request({"part_kind": "tool-return", "tool_name": "t", "content": "ok"}),
        _response({"part_kind": "text", "content": "final"}),
    ]
    rows = build_work_events(
        prompt="go",
        messages_json=history,
        final_response="final",
        error=None,
    )
    text_rows = [r for r in rows if r.event_kind in {"model_messages", "plan_output"}]
    assert [r.event_kind for r in text_rows] == ["model_messages", "plan_output"]
    assert text_rows[1].body == "final"


def test_truncates_huge_bodies() -> None:
    huge = "x" * 100_000
    rows = build_work_events(
        prompt=huge,
        messages_json=None,
        final_response=None,
        error=None,
    )
    body = rows[0].body or ""
    assert len(body) < len(huge)
    assert "truncated" in body
