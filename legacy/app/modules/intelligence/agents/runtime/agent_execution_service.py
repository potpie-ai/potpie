"""Backend-agnostic agent run-loop shared by the Celery and Hatchet backends.

Extracted from ``app/celery/tasks/agent_tasks.py`` so the same start/chunk/end and
cancellation semantics drive every backend. Events go through an ``AgentRunSink``;
the caller owns backend specifics (enqueue, status orchestration, usage/cost
accounting, and the terminal completed-end event).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Optional

from app.modules.conversations.exceptions import GenerationCancelled
from app.modules.intelligence.agents.runtime.ports import AgentRunSink
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_CANCEL_MESSAGE = "Generation cancelled by user"


def serialize_tool_calls(tool_calls: Any) -> list:
    """Serialize agent tool calls to JSON-able values (model_dump / dict / str)."""
    result: list = []
    if not tool_calls:
        return result
    for tc in tool_calls:
        if hasattr(tc, "model_dump"):
            result.append(tc.model_dump())
        elif hasattr(tc, "dict"):
            result.append(tc.dict())
        else:
            result.append(str(tc))
    return result


def _safe_flush(flush_partial: Optional[Callable[[], Any]]) -> None:
    """Flush buffered partial output on cancellation; never raise (we must still end)."""
    if flush_partial is None:
        return
    try:
        flush_partial()
    except Exception as e:
        logger.warning("Failed to flush partial message on cancellation: %s", e)


async def run_agent_turn(
    *,
    start_payload: dict,
    chunk_stream: AsyncIterator[Any],
    sink: AgentRunSink,
    flush_partial: Optional[Callable[[], Any]] = None,
    cancel_message: str = DEFAULT_CANCEL_MESSAGE,
) -> bool:
    """Drive one agent turn: emit ``start``, stream ``chunk`` events, handle cancellation.

    Returns ``True`` when the stream completes, ``False`` when cancelled. On normal
    completion it does NOT emit the terminal ``end`` event — the caller emits the
    completed-end (with usage/cost) so each backend controls terminal semantics.
    """
    sink.emit("start", start_payload)
    try:
        async for chunk in chunk_stream:
            if sink.is_cancelled():
                logger.info("Agent run cancelled (cooperative check)")
                _safe_flush(flush_partial)
                sink.emit("end", {"status": "cancelled", "message": cancel_message})
                return False
            sink.emit(
                "chunk",
                {
                    "content": getattr(chunk, "message", "") or "",
                    "citations_json": getattr(chunk, "citations", None) or [],
                    "tool_calls_json": serialize_tool_calls(
                        getattr(chunk, "tool_calls", None)
                    ),
                },
            )
        # A cooperative cancel can make chunk_stream end cleanly (the agent stops
        # yielding) instead of raising — without this final check the run would be
        # reported as completed. Re-check after the loop so it's treated as cancelled.
        if sink.is_cancelled():
            logger.info("Agent run cancelled (cooperative check)")
            _safe_flush(flush_partial)
            sink.emit("end", {"status": "cancelled", "message": cancel_message})
            return False
        return True
    except GenerationCancelled:
        logger.info("Agent run cancelled (GenerationCancelled)")
        _safe_flush(flush_partial)
        sink.emit("end", {"status": "cancelled", "message": cancel_message})
        return False
