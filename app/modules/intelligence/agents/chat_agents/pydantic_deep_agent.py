"""Pydantic-Deep adapter for chat agents.

Drop-in replacement for ``PydanticRagAgent`` that runs the same chat-agent
contract on top of `pydantic-deep
<https://github.com/vstorm-co/pydantic-deepagents>`_ instead of a bare
``pydantic_ai.Agent``.

What we use from pydantic-deep
------------------------------

* The deep agent's built-in **todo** toolset (planning).
* The deep agent's **context manager** (auto-compression / token tracking).
* The eviction processor and ``patch_tool_calls`` history processor.
* Its BASE_PROMPT preamble (deep-agent norms about action vs. preamble).

What we deliberately keep ours
------------------------------

* **Sandbox / file / shell tools.** ``include_filesystem=False`` so the
  deep agent does not register ``read_file``/``write_file``/``execute``
  on top of our ``sandbox_*`` tools.
* **Web search / fetch.** Disabled — Potpie has its own ``web_search_tool``
  / ``webpage_extractor`` wired through the registry.
* **Subagents / skills / teams / plan.** Off — multi-agent flows live in
  ``PydanticMultiAgent``.
* **Memory & checkpoint stores.** Off — we don't need the on-disk
  ``MEMORY.md`` / checkpoint snapshots in chat-agent runs.

The class subclasses :class:`PydanticRagAgent` so all of the existing
streaming / multimodal / MCP / fallback machinery is reused unchanged.
The only override is :meth:`_create_agent`, which returns an
``Agent[DeepAgentDeps, str]`` and pre-binds ``deps`` to ``iter`` / ``run``
so the parent's streaming code can keep calling them without knowing the
agent now needs a deps argument.
"""

from __future__ import annotations

import functools
from contextlib import asynccontextmanager
from typing import List

from langchain_core.tools import StructuredTool
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.toolsets.function import FunctionToolset

from app.modules.intelligence.agents.chat_agent import ChatContext
from app.modules.intelligence.agents.chat_agents.agent_config import AgentConfig
from app.modules.intelligence.agents.chat_agents.multi_agent.utils.tool_utils import (
    wrap_structured_tools,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tracing.logfire_tracer import (
    should_instrument_pydantic_ai,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


# Tools the deep agent registers itself via ``include_todo=True``. We must
# strip these from the user-supplied tool list so the FunctionToolset and
# the deep-todo toolset don't both register tools under the same name.
_DEEP_BUILTIN_TODO_TOOL_NAMES = frozenset(
    {
        "read_todos",
        "write_todos",
        "add_todo",
        "update_todo_status",
        "remove_todo",
        "add_subtask",
        "set_dependency",
        "get_available_tasks",
    }
)


class PydanticDeepRagAgent(PydanticRagAgent):
    """``PydanticRagAgent`` drop-in backed by ``pydantic-deep``.

    Constructor signature is intentionally identical to ``PydanticRagAgent``
    so call sites in ``system_agents`` / ``runtime_agent`` only need to
    swap the class.
    """

    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
        mcp_servers: List[dict] | None = None,
    ) -> None:
        super().__init__(llm_provider, config, tools, mcp_servers=mcp_servers)
        # Filter our tool list once so we don't fight the deep-todo toolset.
        # We mutate ``self.tools`` (the parent already deduped/sanitized
        # names in __init__) in place.
        self.tools = [
            tool
            for tool in self.tools
            if tool.name not in _DEEP_BUILTIN_TODO_TOOL_NAMES
        ]

    # ----- agent construction -------------------------------------------------

    def _create_agent(self, ctx: ChatContext) -> Agent:
        from pydantic_deep import DeepAgentDeps, create_deep_agent

        config = self.config

        # Tools — wrapped once, served via a FunctionToolset so their
        # explicit JSON schemas survive (passing them through
        # ``create_deep_agent(tools=...)`` would re-register them via
        # ``agent.tool(fn)``, which forces ``takes_ctx=True`` and corrupts
        # the schema for our adapted callables).
        wrapped_tools = wrap_structured_tools(self.tools)
        function_toolset = FunctionToolset(
            tools=wrapped_tools,
            id="potpie-rag-tools",
        )

        # MCP servers — same pattern as the parent class, with one
        # tool_prefix per server to avoid name clashes with built-ins
        # (e.g. ``read_todos``).
        mcp_toolsets: List[MCPServerStreamableHTTP] = []
        for i, mcp_server in enumerate(self.mcp_servers):
            try:
                name = mcp_server.get("name") or f"mcp{i}"
                prefix = "".join(
                    c if c.isalnum() or c in "_-" else "_" for c in str(name).lower()
                )
                if not prefix:
                    prefix = f"mcp{i}"
                mcp_toolsets.append(
                    MCPServerStreamableHTTP(
                        url=mcp_server["link"],
                        timeout=10.0,
                        tool_prefix=f"{prefix}_",
                    )
                )
            except Exception:
                logger.warning(
                    "Failed to create MCP server %s",
                    mcp_server.get("name", "unknown"),
                    exc_info=True,
                )
                continue

        # Multimodal + role/task instructions. We reuse the parent's helpers
        # so the wording across the two adapters stays consistent.
        multimodal_instructions = self._prepare_multimodal_instructions(ctx)
        task_block = self._create_task_description(
            task_config=config.tasks[0], ctx=ctx
        )
        instructions = f"""
# Agent Execution Guidelines

You are an AI assistant that helps users with code analysis and tasks. Follow these principles:

1. **Be thorough**: Analyze code carefully before making recommendations
2. **Use tools effectively**: Leverage available tools to gather information
3. **Provide clear explanations**: Explain your reasoning and findings
4. **Handle errors gracefully**: If a tool fails, try alternative approaches

## Tool Usage Best Practices

- Use available tools to gather information before generating responses
- Use `fetch_file` with `with_line_numbers=true` for precise code references
- Use `ask_knowledge_graph_queries` for semantic code search
- Use `get_code_file_structure` to understand project layout
- Verify your findings before presenting conclusions

## Output Guidelines

- Structure responses with clear headings
- Include relevant code snippets with file paths
- Summarize key findings at the end

<!-- CACHE_BREAKPOINT -->

Your Identity:
Role: {config.role}
Goal: {config.goal}
Backstory:
{config.backstory}

{multimodal_instructions}

CURRENT CONTEXT AND AGENT TASK OVERVIEW:
{task_block}
"""

        allow_parallel_tools = self.llm_provider.chat_config.capabilities.get(
            "supports_tool_parallelism", True
        )

        agent_extra_kwargs: dict = {
            "output_retries": 3,
            "defer_model_check": True,
            "end_strategy": "exhaustive",
            "instrument": should_instrument_pydantic_ai(),
        }
        if mcp_toolsets:
            agent_extra_kwargs["mcp_servers"] = mcp_toolsets
        if not allow_parallel_tools:
            # Same probing as the parent — pydantic-ai's API for disabling
            # parallel tool calls has shifted across releases. We forward
            # whichever kwarg the current Agent constructor accepts; if
            # none are present we fall through silently.
            import inspect

            try:
                signature = inspect.signature(Agent.__init__)
                if "allow_parallel_tool_calls" in signature.parameters:
                    agent_extra_kwargs["allow_parallel_tool_calls"] = False
                elif "max_parallel_tool_calls" in signature.parameters:
                    agent_extra_kwargs["max_parallel_tool_calls"] = 1
                elif "tool_parallelism" in signature.parameters:
                    agent_extra_kwargs["tool_parallelism"] = False
            except Exception:
                logger.warning(
                    "Failed to detect parallel tool kwarg on Agent.__init__",
                    exc_info=True,
                )

        agent: Agent = create_deep_agent(
            model=self.llm_provider.get_pydantic_model(),
            instructions=instructions,
            toolsets=[function_toolset],
            # Built-ins we keep on:
            include_todo=True,
            context_manager=True,  # auto-compression of long histories
            patch_tool_calls=True,
            # Built-ins we deliberately disable so we don't shadow our own
            # tools or grow the prompt with unrelated guidance:
            include_filesystem=False,  # use ``sandbox_*`` tools from the registry
            include_subagents=False,  # multi-agent flows live in PydanticMultiAgent
            include_skills=False,
            include_builtin_subagents=False,
            include_plan=False,
            include_teams=False,
            include_memory=False,
            include_history_archive=False,
            include_checkpoints=False,
            web_search=False,
            web_fetch=False,
            cost_tracking=False,
            thinking="high",
            history_processors=[self._history_processor],
            model_settings={"max_tokens": 14000},
            **agent_extra_kwargs,
        )

        # Pre-bind a deps instance so the parent's run/run_stream code can
        # keep calling ``agent.iter(user_prompt=..., message_history=...)``
        # without knowing about ``DeepAgentDeps``. Each ``_create_agent``
        # call returns a fresh agent, so we can stash the deps on the
        # instance and shadow ``run`` / ``iter`` at the instance level.
        deps = DeepAgentDeps()
        _bind_default_deps(agent, deps)
        return agent


def _bind_default_deps(agent: Agent, deps: object) -> None:
    """Patch ``agent.iter`` and ``agent.run`` to default ``deps`` to ``deps``.

    pydantic-deep agents are typed ``Agent[DeepAgentDeps, str]`` and tools
    such as the todo toolset reach for ``ctx.deps`` at runtime. The chat
    agent harness was written for a deps-less ``pydantic_ai.Agent``, so we
    transparently provide a default deps here. Callers who supply their
    own ``deps=`` keep that value (we only set the default).
    """
    original_iter = agent.iter
    original_run = agent.run

    @asynccontextmanager
    async def iter_with_deps(*args, **kwargs):
        kwargs.setdefault("deps", deps)
        async with original_iter(*args, **kwargs) as agent_run:
            yield agent_run

    @functools.wraps(original_run)
    async def run_with_deps(*args, **kwargs):
        kwargs.setdefault("deps", deps)
        return await original_run(*args, **kwargs)

    # Bind on the instance — leave the class methods untouched.
    agent.iter = iter_with_deps  # type: ignore[method-assign]
    agent.run = run_with_deps  # type: ignore[method-assign]


__all__ = ["PydanticDeepRagAgent"]
