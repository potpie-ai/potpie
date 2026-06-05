"""Translate a pydantic-ai message history into reconciliation work-event rows.

The reconciliation agent's trace lives in two places:

- ``BatchAgentOutcome.prompt`` — the user prompt that opened the run.
- ``BatchAgentOutcome.agent_messages_json`` — every ``ModelMessage`` produced
  by ``pydantic-ai``, in order. Each message has ``parts``: system / user
  prompts, model text, tool calls, tool returns.

The UI (``PotEventsPanel``) renders these as an ordered list of *work events*
on a reconciliation run. This module is the bridge: it walks the message
history once and yields a sequence of ``WorkEventRecord`` rows ready for
``ReconciliationLedgerPort.record_run_work_event``.

Mapping
-------
- The opening ``user-prompt`` becomes a single ``prompt`` event. Any system
  prompt is folded into its payload, not emitted separately (the UI doesn't
  show it usefully).
- Each ``thinking`` or interim ``text`` part on a model response becomes a
  ``model_messages`` event so reasoning is visible step-by-step.
- The final ``text`` response (the last ``response`` message in the history)
  is promoted to ``plan_output`` so the wrap-up shows distinctly from
  interim reasoning.
- Each ``tool-call`` part becomes a ``tool_call`` event whose title is the
  tool name and whose payload carries the call arguments.
- Each ``tool-return`` part becomes a ``tool_result`` event whose title is
  the tool name and whose body is the returned content (truncated for the
  ledger; the full payload remains in the JSON column).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

_TOOL_RESULT_BODY_MAX = 4000
_TEXT_BODY_MAX = 16000
_PROMPT_BODY_MAX = 32000


@dataclass(slots=True, frozen=True)
class WorkEventRecord:
    """One row to append to a reconciliation run's work-event log."""

    event_kind: str
    title: str | None
    body: str | None
    payload: dict[str, Any]


def build_work_events(
    *,
    prompt: str | None,
    messages_json: list[dict[str, Any]] | None,
    final_response: str | None,
    error: str | None,
) -> list[WorkEventRecord]:
    """Compose the ordered work-event log for one batch agent run.

    ``prompt`` is included even when ``messages_json`` is empty (e.g. the
    agent crashed before producing any response) so the UI still surfaces
    what was sent. ``error`` is appended as the trailing event when present.
    """
    out: list[WorkEventRecord] = []
    out.extend(_emit_prompt(prompt, messages_json))
    out.extend(_emit_from_messages(messages_json, final_response))
    if error:
        out.append(
            WorkEventRecord(
                event_kind="error",
                title="agent error",
                body=_truncate(error, _TEXT_BODY_MAX),
                payload={"error": error},
            )
        )
    return out


def _emit_prompt(
    prompt: str | None,
    messages_json: list[dict[str, Any]] | None,
) -> Iterator[WorkEventRecord]:
    """Yield the initial prompt event.

    We prefer the prompt string captured by the agent (verbatim batch payload).
    If that's missing, fall back to mining the first ``request`` message in
    the history for a ``user-prompt`` part so we always emit *something*.
    """
    body: str | None = prompt
    system_prompts: list[str] = []
    if messages_json:
        for msg in messages_json:
            if not isinstance(msg, dict) or msg.get("kind") != "request":
                continue
            for part in msg.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                kind = part.get("part_kind")
                content = part.get("content")
                if kind == "system-prompt" and isinstance(content, str):
                    system_prompts.append(content)
                elif (
                    kind == "user-prompt" and body is None and isinstance(content, str)
                ):
                    body = content
            break  # only the very first request carries the opener
    if body is None and not system_prompts:
        return
    payload: dict[str, Any] = {}
    if system_prompts:
        payload["system_prompts"] = [
            _truncate(p, _PROMPT_BODY_MAX) for p in system_prompts
        ]
    yield WorkEventRecord(
        event_kind="prompt",
        title="Batch prompt",
        body=_truncate(body, _PROMPT_BODY_MAX) if body is not None else None,
        payload=payload,
    )


def _emit_from_messages(
    messages_json: list[dict[str, Any]] | None,
    final_response: str | None,
) -> Iterator[WorkEventRecord]:
    if not messages_json:
        if final_response:
            yield WorkEventRecord(
                event_kind="plan_output",
                title="Final response",
                body=_truncate(final_response, _TEXT_BODY_MAX),
                payload={},
            )
        return

    last_response_index = _last_response_index(messages_json)
    for idx, msg in enumerate(messages_json):
        if not isinstance(msg, dict):
            continue
        kind = msg.get("kind")
        parts = msg.get("parts") or []
        if kind == "request":
            # Skip the opening user-prompt (already emitted) — surface only
            # tool-returns coming back from the prior tool call.
            for part in parts:
                row = _row_from_request_part(part)
                if row is not None:
                    yield row
        elif kind == "response":
            is_last_response = idx == last_response_index
            for part in parts:
                row = _row_from_response_part(part, is_last_response=is_last_response)
                if row is not None:
                    yield row


def _last_response_index(messages_json: list[dict[str, Any]]) -> int:
    for idx in range(len(messages_json) - 1, -1, -1):
        msg = messages_json[idx]
        if isinstance(msg, dict) and msg.get("kind") == "response":
            return idx
    return -1


def _row_from_request_part(part: Any) -> WorkEventRecord | None:
    if not isinstance(part, dict):
        return None
    kind = part.get("part_kind")
    if kind != "tool-return":
        return None
    tool_name = _str(part.get("tool_name"))
    content = part.get("content")
    body = _content_to_text(content)
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_call_id": _str(part.get("tool_call_id")),
        "content": content,
    }
    timestamp = _str(part.get("timestamp"))
    if timestamp:
        payload["timestamp"] = timestamp
    return WorkEventRecord(
        event_kind="tool_result",
        title=tool_name or "tool result",
        body=_truncate(body, _TOOL_RESULT_BODY_MAX),
        payload=payload,
    )


def _row_from_response_part(
    part: Any, *, is_last_response: bool
) -> WorkEventRecord | None:
    if not isinstance(part, dict):
        return None
    kind = part.get("part_kind")
    if kind == "tool-call":
        tool_name = _str(part.get("tool_name"))
        args = part.get("args")
        body = _content_to_text(args)
        return WorkEventRecord(
            event_kind="tool_call",
            title=tool_name or "tool call",
            body=_truncate(body, _TEXT_BODY_MAX) if body else None,
            payload={
                "tool_name": tool_name,
                "tool_call_id": _str(part.get("tool_call_id")),
                "args": args,
            },
        )
    if kind == "text":
        content = part.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        # The final text from the last response message is the agent's
        # wrap-up — surface it as plan_output so it visually anchors the run.
        if is_last_response:
            return WorkEventRecord(
                event_kind="plan_output",
                title="Final response",
                body=_truncate(content, _TEXT_BODY_MAX),
                payload={"content": content},
            )
        return WorkEventRecord(
            event_kind="model_messages",
            title="Model message",
            body=_truncate(content, _TEXT_BODY_MAX),
            payload={"content": content},
        )
    if kind == "thinking":
        content = part.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        return WorkEventRecord(
            event_kind="model_messages",
            title="Thinking",
            body=_truncate(content, _TEXT_BODY_MAX),
            payload={"thinking": content},
        )
    return None


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str, indent=2)
    except Exception:
        return str(content)


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated {len(text) - limit} chars]"


def _str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


__all__ = ["WorkEventRecord", "build_work_events"]
