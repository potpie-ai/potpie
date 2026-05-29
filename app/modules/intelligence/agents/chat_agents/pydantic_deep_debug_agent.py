from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, AsyncGenerator, List

import anyio
from langchain_core.tools import StructuredTool

from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError, ModelRetry, UserError
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from app.modules.intelligence.agents.chat_agent import (
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
    ToolCallEventType,
    ToolCallResponse,
)
from app.modules.intelligence.agents.chat_agents.agent_config import AgentConfig
from app.modules.intelligence.agents.chat_agents.multi_agent.utils.tool_utils import (
    wrap_structured_tools,
)
from app.modules.intelligence.agents.chat_agents.tool_helpers import (
    get_tool_call_info_content,
    get_tool_response_message,
    get_tool_result_info_content,
    get_tool_run_message,
)
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tracing.logfire_tracer import (
    is_logfire_enabled,
    logfire_trace_metadata,
    should_instrument_pydantic_ai,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _build_logfire_metadata(ctx: ChatContext) -> dict:
    """Best-effort metadata dictionary for logfire baggage.

    Reads optional attrs via getattr so the helper works with any
    ChatContext-like object (including test stubs)."""
    return {
        "user_id": getattr(ctx, "user_id", None),
        "conversation_id": getattr(ctx, "conversation_id", None),
        "agent_id": getattr(ctx, "curr_agent_id", None) or "debug_agent",
        "project_id": getattr(ctx, "project_id", None),
        "project_name": getattr(ctx, "project_name", None),
        "repository": getattr(ctx, "repository", None),
        "branch": getattr(ctx, "branch", None),
        "local_mode": getattr(ctx, "local_mode", None),
        "runtime": "pydantic_deep_debug_agent",
    }


@contextmanager
def _logfire_span(name: str, **attrs: Any):
    """Open a logfire span only when logfire is initialized.

    Wraps the import and span creation in try/except so any logfire-related
    failure is non-fatal. Exception info from the wrapped block is forwarded
    to the span's __exit__ so the span records the failure correctly."""
    if not is_logfire_enabled():
        yield None
        return

    span_cm = None
    try:
        import logfire

        span_cm = logfire.span(name, **attrs)
        span_cm.__enter__()
    except Exception as exc:
        logger.debug("Logfire span %s enter failed (non-fatal): %s", name, exc)
        span_cm = None

    try:
        yield span_cm
    except BaseException:
        if span_cm is not None:
            exc_type, exc_val, exc_tb = sys.exc_info()
            try:
                span_cm.__exit__(exc_type, exc_val, exc_tb)
            except Exception as cleanup_exc:
                logger.debug(
                    "Logfire span %s exit failed (non-fatal): %s",
                    name,
                    cleanup_exc,
                )
            span_cm = None
        raise
    finally:
        if span_cm is not None:
            try:
                span_cm.__exit__(None, None, None)
            except Exception as exc:
                logger.debug(
                    "Logfire span %s exit failed (non-fatal): %s", name, exc
                )


@contextmanager
def _safe_trace_metadata(**kwargs: Any):
    """Wrap logfire_trace_metadata so failure to **enter** the helper is non-fatal.

    ``logfire_trace_metadata`` itself is already defensive (it catches
    ``set_baggage`` failures internally and yields a no-op context). This
    wrapper is belt-and-braces in case future tracer changes introduce a new
    fault path on enter/exit. Exceptions raised from the wrapped block always
    propagate — we never swallow agent errors.
    """
    cm = None
    try:
        cm = logfire_trace_metadata(**kwargs)
        cm.__enter__()
    except Exception as exc:
        logger.debug("logfire_trace_metadata enter failed (non-fatal): %s", exc)
        cm = None

    try:
        yield
    finally:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception as exc:
                logger.debug(
                    "logfire_trace_metadata exit failed (non-fatal): %s", exc
                )


class PydanticDeepDebugAgent(ChatAgent):
    """Thin pydantic-deep adapter for the DebugAgent tool allow-list."""

    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
    ) -> None:
        self.llm_provider = llm_provider
        self.config = config
        self.tools = tools

    def _build_instructions(self, ctx: ChatContext) -> str:
        task = self.config.tasks[0] if self.config.tasks else None
        task_description = task.description if task else ""
        expected_output = task.expected_output if task else "Markdown response"
        node_ids = ctx.node_ids or []

        # The task description IS the canonical debug protocol (debug_task_prompt
        # with its strict Phase 1-7 workflow and tool-call requirements). Surrounding
        # it with extra workflow/tool-discipline guidance dilutes those instructions
        # and causes the model to skip persistence calls (e.g. record_hypothesis).
        # Keep this wrapper to project/role context only and let the task own the rest.
        return f"""
Role: {self.config.role}
Goal: {self.config.goal}
Backstory:
{self.config.backstory}

Current project:
- Project ID: {ctx.project_id}
- Project name: {ctx.project_name}
- Node IDs: {", ".join(str(node_id) for node_id in node_ids) or "none"}
- Local mode: {ctx.local_mode}
- Repository: {getattr(ctx, "repository", None) or "none"}
- Branch: {getattr(ctx, "branch", None) or "none"}

Additional context:
{ctx.additional_context or "none"}

Task instructions (follow these exactly):
{task_description}

Expected output:
{expected_output}
        """.strip()

    def _create_agent(self, ctx: ChatContext):
        try:
            from pydantic_deep import create_deep_agent
        except ImportError as exc:
            raise ImportError(
                "pydantic-deep is required for PydanticDeepDebugAgent. "
                "Install the context-engine[all] dependency set or add pydantic-deep."
            ) from exc

        wrapped_tools = wrap_structured_tools(self.tools)

        # Build the deep-agent shell WITHOUT passing tools= — pydantic-deep's
        # tool loop extracts `tool.function` and re-registers via `agent.tool(func)`,
        # which (a) collapses names to the inner wrapper's __name__ and (b) infers
        # `takes_ctx=True` for **kwargs wrappers, breaking schema generation.
        # We attach the wrapped Tool objects directly via Agent.add_tool, which
        # preserves the explicit name, description, schema, and takes_ctx=False.
        agent = create_deep_agent(
            model=self.llm_provider.get_pydantic_model(),
            instructions=self._build_instructions(ctx),
            output_type=str,
            include_todo=False,
            include_filesystem=False,
            include_subagents=False,
            include_skills=False,
            include_builtin_subagents=False,
            include_plan=False,
            include_memory=False,
            include_teams=False,
            include_checkpoints=False,
            include_history_archive=False,
            web_search=False,
            web_fetch=False,
            context_manager=False,
            cost_tracking=False,
            model_settings={"max_tokens": 32000},
            instrument=should_instrument_pydantic_ai(),
        )
        for tool in wrapped_tools:
            agent._function_toolset.add_tool(tool)
        return agent

    def _set_sandbox_context(self, ctx: ChatContext) -> None:
        from app.modules.intelligence.tools.sandbox.context import (
            set_run_context as _set_sandbox_run_context,
        )
        from app.modules.intelligence.tools.code_changes_manager import (
            _init_code_changes_manager,
        )
        from app.modules.intelligence.tools.hypothesis_state_tool import (
            _reset_hypothesis_store,
        )

        local_mode = ctx.local_mode if hasattr(ctx, "local_mode") else False

        # Populate ContextVars so tunnel-dependent tools (DAP, search_bash,
        # terminal, etc.) can route to the VS Code extension via Socket.IO.
        _init_code_changes_manager(
            conversation_id=ctx.conversation_id,
            agent_id=getattr(ctx, "curr_agent_id", None),
            user_id=ctx.user_id,
            tunnel_url=getattr(ctx, "tunnel_url", None),
            local_mode=local_mode,
            repository=getattr(ctx, "repository", None),
            branch=getattr(ctx, "branch", None),
        )
        _reset_hypothesis_store(conversation_id=ctx.conversation_id or "")
        _set_sandbox_run_context(
            user_id=ctx.user_id,
            conversation_id=ctx.conversation_id,
            branch=getattr(ctx, "branch", None),
            local_mode=local_mode,
        )

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        self._set_sandbox_context(ctx)

        metadata = _build_logfire_metadata(ctx)
        with _safe_trace_metadata(**metadata):
            with _logfire_span("debug_agent_run", **metadata):
                try:
                    from pydantic_deep import create_default_deps

                    agent = self._create_agent(ctx)
                    result = await agent.run(
                        ctx.query,
                        deps=create_default_deps(),
                        message_history=[
                            ModelResponse([TextPart(content=msg)])
                            for msg in (ctx.history or [])
                        ],
                    )
                    return ChatAgentResponse(
                        response=str(result.output),
                        tool_calls=[],
                        citations=[],
                    )
                except Exception as exc:
                    logger.exception("Error in pydantic-deep debug agent run")
                    return ChatAgentResponse(
                        response=f"An error occurred while processing your request: {str(exc)}",
                        tool_calls=[],
                        citations=[],
                    )

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        self._set_sandbox_context(ctx)

        metadata = _build_logfire_metadata(ctx)
        with _safe_trace_metadata(**metadata):
            with _logfire_span("debug_agent_stream", **metadata):
                try:
                    from pydantic_deep import create_default_deps
                except ImportError:
                    logger.exception(
                        "pydantic-deep is required for PydanticDeepDebugAgent streaming"
                    )
                    yield ChatAgentResponse(
                        response="An error occurred: pydantic-deep is not installed.",
                        tool_calls=[],
                        citations=[],
                    )
                    return

                try:
                    agent = self._create_agent(ctx)
                except Exception as exc:
                    logger.exception(
                        "Failed to construct pydantic-deep agent for streaming"
                    )
                    yield ChatAgentResponse(
                        response=f"\n\n*The agent failed to start: {str(exc)}*\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                    return

                try:
                    async with agent.iter(
                        user_prompt=ctx.query,
                        deps=create_default_deps(),
                        message_history=[
                            ModelResponse([TextPart(content=msg)])
                            for msg in (ctx.history or [])
                        ],
                    ) as run:
                        async for node in run:
                            if Agent.is_model_request_node(node):
                                try:
                                    async with node.stream(run.ctx) as request_stream:
                                        # Reasoning models emit ThinkingPart events
                                        # separately from text. Wrap them in <think>
                                        # tags so the webview renders reasoning in the
                                        # same collapsible block as the code agent
                                        # (which gets <think> tags inline from its model).
                                        thinking_open = False
                                        async for event in request_stream:
                                            if isinstance(
                                                event, PartStartEvent
                                            ) and isinstance(event.part, ThinkingPart):
                                                text = event.part.content or ""
                                                if not thinking_open:
                                                    thinking_open = True
                                                    text = "<think>" + text
                                                yield ChatAgentResponse(
                                                    response=text,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                                continue
                                            if isinstance(
                                                event, PartDeltaEvent
                                            ) and isinstance(
                                                event.delta, ThinkingPartDelta
                                            ):
                                                text = event.delta.content_delta or ""
                                                if not thinking_open:
                                                    thinking_open = True
                                                    text = "<think>" + text
                                                yield ChatAgentResponse(
                                                    response=text,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                                continue
                                            if isinstance(
                                                event, PartStartEvent
                                            ) and isinstance(event.part, TextPart):
                                                prefix = (
                                                    "</think>\n\n"
                                                    if thinking_open
                                                    else ""
                                                )
                                                thinking_open = False
                                                yield ChatAgentResponse(
                                                    response=prefix + event.part.content,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, PartDeltaEvent
                                            ) and isinstance(
                                                event.delta, TextPartDelta
                                            ):
                                                prefix = (
                                                    "</think>\n\n"
                                                    if thinking_open
                                                    else ""
                                                )
                                                thinking_open = False
                                                yield ChatAgentResponse(
                                                    response=prefix
                                                    + event.delta.content_delta,
                                                    tool_calls=[],
                                                    citations=[],
                                                )
                                        # Close an unterminated think block (model went
                                        # straight from reasoning to a tool call with no
                                        # text in between).
                                        if thinking_open:
                                            yield ChatAgentResponse(
                                                response="</think>\n\n",
                                                tool_calls=[],
                                                citations=[],
                                            )
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in model request stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while processing your request. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Model request stream would block - continuing..."
                                    )
                                    continue
                                except Exception:
                                    logger.exception(
                                        "Unexpected error in model request stream"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_call_tools_node(node):
                                try:
                                    async with node.stream(run.ctx) as handle_stream:
                                        async for event in handle_stream:
                                            if isinstance(event, FunctionToolCallEvent):
                                                tool_args = event.part.args_as_dict()
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        ToolCallResponse(
                                                            call_id=event.part.tool_call_id
                                                            or "",
                                                            event_type=ToolCallEventType.CALL,
                                                            tool_name=event.part.tool_name,
                                                            tool_response=get_tool_run_message(
                                                                event.part.tool_name,
                                                                tool_args,
                                                            ),
                                                            tool_call_details={
                                                                "summary": get_tool_call_info_content(
                                                                    event.part.tool_name,
                                                                    tool_args,
                                                                )
                                                            },
                                                        )
                                                    ],
                                                    citations=[],
                                                )
                                            if isinstance(
                                                event, FunctionToolResultEvent
                                            ):
                                                yield ChatAgentResponse(
                                                    response="",
                                                    tool_calls=[
                                                        ToolCallResponse(
                                                            call_id=event.result.tool_call_id
                                                            or "",
                                                            event_type=ToolCallEventType.RESULT,
                                                            tool_name=event.result.tool_name
                                                            or "unknown tool",
                                                            tool_response=get_tool_response_message(
                                                                event.result.tool_name
                                                                or "unknown tool",
                                                                result=event.result.content,
                                                            ),
                                                            tool_call_details={
                                                                "summary": get_tool_result_info_content(
                                                                    event.result.tool_name
                                                                    or "unknown tool",
                                                                    event.result.content,
                                                                )
                                                            },
                                                        )
                                                    ],
                                                    citations=[],
                                                )
                                except (
                                    ModelRetry,
                                    AgentRunError,
                                    UserError,
                                ) as pydantic_error:
                                    logger.warning(
                                        f"Pydantic-ai error in tool call stream: {pydantic_error}"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*Encountered an issue while calling tools. Trying to recover...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue
                                except anyio.WouldBlock:
                                    logger.warning(
                                        "Tool call stream would block - continuing..."
                                    )
                                    continue
                                except Exception:
                                    logger.exception(
                                        "Unexpected error in tool call stream"
                                    )
                                    yield ChatAgentResponse(
                                        response="\n\n*An unexpected error occurred during tool execution. Continuing...*\n\n",
                                        tool_calls=[],
                                        citations=[],
                                    )
                                    continue

                            elif Agent.is_end_node(node):
                                logger.info(
                                    "pydantic-deep debug stream completed successfully"
                                )

                except (ModelRetry, AgentRunError, UserError) as pydantic_error:
                    logger.exception("Pydantic-ai error in run_stream method")
                    yield ChatAgentResponse(
                        response=f"\n\n*The agent encountered an error: {str(pydantic_error)}*\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                except Exception:
                    logger.exception("Error in debug agent run_stream method")
                    yield ChatAgentResponse(
                        response="\n\n*An error occurred during streaming*\n\n",
                        tool_calls=[],
                        citations=[],
                    )
