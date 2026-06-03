"""
Tool allow-list tests for DebugAgent._build_agent (Task 3).

These tests assert that the DebugAgent requests EXACTLY the spec-defined tool
list from `tools_provider.get_tools(...)` — no more, no fewer — and explicitly
verify that excluded categories (KG/code-graph, web, broad sandbox shell/edit/
git, broad todo, requirements) never appear in the request.

The setup mirrors `test_debug_agent_routing.py` so the import chain works
without spawning real LLMs / DB connections.
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Capture original sys.modules / env state *before* this file stubs them, so the
# autouse fixture below can restore it after this module's tests. Otherwise the
# fake torch/sentence_transformers and test DB env leak into the rest of the
# suite and can hide real import/config regressions.
# ---------------------------------------------------------------------------
_STUBBED_MODULE_NAMES = (
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.modules",
    "torch._jit_internal",
    "torch._sources",
    "torch._VF",
    "sentence_transformers",
)
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES}
_ORIGINAL_ENV = {key: os.environ.get(key) for key in ("POSTGRES_SERVER", "REDIS_URL")}

# ---------------------------------------------------------------------------
# Step 1 — set env vars that module-level code reads at import time
# ---------------------------------------------------------------------------
os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


# ---------------------------------------------------------------------------
# Step 2 — stub out the heavy ML / transformers packages that are dragged in
#          through the import chain before they are ever actually needed.
# ---------------------------------------------------------------------------
def _register_stub(name: str, **attrs: object) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


_torch_stub = _register_stub("torch")
_register_stub("torch.nn", functional=MagicMock())
_register_stub("torch.nn.functional")
_register_stub("torch.nn.modules")
_register_stub("torch._jit_internal")
_register_stub("torch._sources")
_register_stub("torch._VF")
_torch_stub._VF = MagicMock()  # type: ignore[attr-defined]
_torch_stub.functional = MagicMock()  # type: ignore[attr-defined]
_torch_stub.nn = sys.modules["torch.nn"]  # type: ignore[attr-defined]

_st_stub = _register_stub("sentence_transformers")
_st_stub.SentenceTransformer = MagicMock(name="SentenceTransformer")  # type: ignore[attr-defined]


@pytest.fixture(autouse=True, scope="module")
def _restore_global_import_state():
    """Restore sys.modules / env after this module's tests so the fake torch and
    sentence_transformers stubs (and test DB env) don't poison later tests."""
    yield
    for name, original in _ORIGINAL_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    for key, original in _ORIGINAL_ENV.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


# ---------------------------------------------------------------------------
# Spec expectations (Task 3)
# ---------------------------------------------------------------------------

EXPECTED_BASE_TOOLS: tuple[str, ...] = (
    "parse_failure_signal",
    "get_workspace_debug_context",
    "get_code_file_structure",
    "search_text",
    "search_bash",
    "fetch_file",
    "fetch_files_batch",
    "run_validation",
    "record_hypothesis",
    "update_hypothesis_status",
    "append_hypothesis_evidence",
    "list_hypotheses",
    "read_todos",
    "add_todo",
    "update_todo_status",
    "get_available_tasks",
)

EXPECTED_DAP_TOOLS: tuple[str, ...] = (
    "start_debug_session",
    "set_breakpoints",
    "take_debug_snapshot",
    "step_over",
    "step_into",
    "step_out",
    "continue_execution",
    "evaluate_expression",
    "list_debug_sessions",
    "stop_debug_session",
)

EXPECTED_TERMINAL_TOOLS: tuple[str, ...] = (
    "execute_terminal_command",
    "terminal_session_output",
    "terminal_session_signal",
)

# Tools the spec explicitly forbids the DebugAgent from requesting.
EXCLUDED_WEB_TOOLS: tuple[str, ...] = (
    "webpage_extractor",
    "web_search_tool",
)
EXCLUDED_KG_TOOLS: tuple[str, ...] = (
    "ask_knowledge_graph_queries",
    "query_context_graph",
    "analyze_code_structure",
    "get_code_from_node_id",
    "get_code_from_probable_node_name",
    "get_code_graph_from_node_id",
    "get_code_graph_from_node_name",
)
EXCLUDED_SANDBOX_TOOLS: tuple[str, ...] = (
    "sandbox_text_editor",
    "sandbox_shell",
    "sandbox_search",
    "sandbox_git",
)
EXCLUDED_BROAD_TODO_TOOLS: tuple[str, ...] = (
    "write_todos",
    "remove_todo",
    "add_subtask",
    "set_dependency",
)
EXCLUDED_REQUIREMENTS_TOOLS: tuple[str, ...] = (
    "add_requirements",
    "get_requirements",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_debug_agent():
    """Return a DebugAgent with all constructor deps mocked out."""
    from app.modules.intelligence.agents.chat_agents.system_agents.debug_agent import (
        DebugAgent,
    )

    llm_provider = MagicMock()
    tools_provider = MagicMock()
    prompt_provider = MagicMock()

    # _build_agent fetches the tools and then routes on supports_pydantic. We
    # don't care about the tools' identity — only the names passed in.
    tools_provider.get_tools.return_value = []
    llm_provider.supports_pydantic.return_value = True

    return DebugAgent(
        llm_provider=llm_provider,
        tools_provider=tools_provider,
        prompt_provider=prompt_provider,
    )


def _make_minimal_ctx():
    """Return a minimal ChatContext that satisfies _build_agent requirements."""
    from app.modules.intelligence.agents.chat_agent import ChatContext

    return ChatContext(
        project_id="proj-test",
        project_name="test-project",
        curr_agent_id="agent-1",
        history=[],
        query="why is my code broken?",
    )


def _requested_tool_names(agent, ctx) -> list[str]:
    """Invoke _build_agent and return the names passed to get_tools(...) as a list."""
    from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig

    deep_sentinel = MagicMock(name="PydanticDeepDebugAgent_instance")
    DeepCls = MagicMock(return_value=deep_sentinel)

    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.PydanticDeepDebugAgent",
        DeepCls,
    ), patch.object(MultiAgentConfig, "should_use_multi_agent", return_value=False):
        agent._build_agent(ctx)

    # get_tools(...) is called exactly once with a list of names.
    assert agent.tools_provider.get_tools.call_count == 1, (
        "Expected tools_provider.get_tools to be called exactly once during _build_agent, "
        f"but it was called {agent.tools_provider.get_tools.call_count} times."
    )
    call = agent.tools_provider.get_tools.call_args
    # Support both positional and keyword invocations.
    if call.args:
        names = call.args[0]
    else:
        names = call.kwargs.get("tool_names") or call.kwargs.get("names")
    assert names is not None, "Could not locate the tool names argument in the call."
    return list(names)


# ---------------------------------------------------------------------------
# Test 1 — exact allow-list shape
# ---------------------------------------------------------------------------


def test_debug_agent_requests_only_spec_tools_from_tools_provider():
    """
    _build_agent must call tools_provider.get_tools(...) exactly once with a list
    that contains every spec-included tool, no excluded tool, and the full union of
    base + DAP + terminal tool tuples.
    """
    agent = _make_debug_agent()
    ctx = _make_minimal_ctx()
    # local_mode controls post-fetch DAP filtering — not what's requested from
    # the provider. The request must always be the union; the filter happens
    # AFTER fetching. To make this explicit, set local_mode=True so no filter
    # runs and the returned names = what was passed in.
    ctx.local_mode = True

    names = _requested_tool_names(agent, ctx)

    # 1. Exact length.
    expected_count = (
        len(EXPECTED_BASE_TOOLS)
        + len(EXPECTED_DAP_TOOLS)
        + len(EXPECTED_TERMINAL_TOOLS)
    )
    assert len(names) == expected_count, (
        f"Expected {expected_count} tool names "
        f"({len(EXPECTED_BASE_TOOLS)} base + {len(EXPECTED_DAP_TOOLS)} DAP + "
        f"{len(EXPECTED_TERMINAL_TOOLS)} terminal), got {len(names)}: {names}"
    )

    # 2. Every spec tool present.
    for tool in EXPECTED_BASE_TOOLS:
        assert tool in names, f"Missing required base tool: {tool!r}"
    for tool in EXPECTED_DAP_TOOLS:
        assert tool in names, f"Missing required DAP tool: {tool!r}"
    for tool in EXPECTED_TERMINAL_TOOLS:
        assert tool in names, f"Missing required terminal tool: {tool!r}"

    # 3. No excluded tool present.
    forbidden = (
        EXCLUDED_WEB_TOOLS
        + EXCLUDED_KG_TOOLS
        + EXCLUDED_SANDBOX_TOOLS
        + EXCLUDED_BROAD_TODO_TOOLS
        + EXCLUDED_REQUIREMENTS_TOOLS
    )
    leaked = [t for t in forbidden if t in names]
    assert not leaked, f"DebugAgent must not request excluded tools, but found: {leaked}"


# ---------------------------------------------------------------------------
# Test 2 — KG / code-graph tools
# ---------------------------------------------------------------------------


def test_debug_agent_excludes_kg_tools():
    """No knowledge-graph / code-graph tools may be requested."""
    agent = _make_debug_agent()
    ctx = _make_minimal_ctx()
    ctx.local_mode = True

    names = _requested_tool_names(agent, ctx)

    for tool in EXCLUDED_KG_TOOLS:
        assert tool not in names, (
            f"KG/code-graph tool {tool!r} must NOT be requested by DebugAgent, "
            f"but it appears in: {names}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Web tools
# ---------------------------------------------------------------------------


def test_debug_agent_excludes_web_tools():
    """Neither webpage_extractor nor web_search_tool may be requested."""
    agent = _make_debug_agent()
    ctx = _make_minimal_ctx()
    ctx.local_mode = True

    names = _requested_tool_names(agent, ctx)

    for tool in EXCLUDED_WEB_TOOLS:
        assert tool not in names, (
            f"Web tool {tool!r} must NOT be requested by DebugAgent, "
            f"but it appears in: {names}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Sandbox shell/edit/git tools
# ---------------------------------------------------------------------------


def test_debug_agent_excludes_sandbox_shell_edit_git_tools():
    """Broad sandbox tools (shell/edit/search/git) must NOT be requested."""
    agent = _make_debug_agent()
    ctx = _make_minimal_ctx()
    ctx.local_mode = True

    names = _requested_tool_names(agent, ctx)

    for tool in EXCLUDED_SANDBOX_TOOLS:
        assert tool not in names, (
            f"Sandbox tool {tool!r} must NOT be requested by DebugAgent, "
            f"but it appears in: {names}"
        )


# ---------------------------------------------------------------------------
# Test 5 — Broad todo tools
# ---------------------------------------------------------------------------


def test_debug_agent_excludes_broad_todo_tools():
    """Broad todo tools (write_todos, remove_todo, add_subtask, set_dependency) excluded."""
    agent = _make_debug_agent()
    ctx = _make_minimal_ctx()
    ctx.local_mode = True

    names = _requested_tool_names(agent, ctx)

    for tool in EXCLUDED_BROAD_TODO_TOOLS:
        assert tool not in names, (
            f"Broad todo tool {tool!r} must NOT be requested by DebugAgent, "
            f"but it appears in: {names}"
        )


# ---------------------------------------------------------------------------
# Test 6 — Requirements tools
# ---------------------------------------------------------------------------


def test_debug_agent_excludes_requirements_tools():
    """Requirements tools (add_requirements, get_requirements) excluded."""
    agent = _make_debug_agent()
    ctx = _make_minimal_ctx()
    ctx.local_mode = True

    names = _requested_tool_names(agent, ctx)

    for tool in EXCLUDED_REQUIREMENTS_TOOLS:
        assert tool not in names, (
            f"Requirements tool {tool!r} must NOT be requested by DebugAgent, "
            f"but it appears in: {names}"
        )


# ---------------------------------------------------------------------------
# Test 7 — Focused todo tools must be included
# ---------------------------------------------------------------------------


def test_debug_agent_includes_focused_todo_tools():
    """Focused todo tools (read/add/update_status/get_available) must be requested."""
    agent = _make_debug_agent()
    ctx = _make_minimal_ctx()
    ctx.local_mode = True

    names = _requested_tool_names(agent, ctx)

    for tool in ("read_todos", "add_todo", "update_todo_status", "get_available_tasks"):
        assert tool in names, (
            f"Focused todo tool {tool!r} must be requested by DebugAgent, "
            f"but it is missing from: {names}"
        )


# ---------------------------------------------------------------------------
# Test 8 — Hypothesis tools must be included (extra sanity check)
# ---------------------------------------------------------------------------


def test_debug_agent_includes_hypothesis_tools():
    """All four hypothesis-tracking tools must be requested."""
    agent = _make_debug_agent()
    ctx = _make_minimal_ctx()
    ctx.local_mode = True

    names = _requested_tool_names(agent, ctx)

    for tool in (
        "record_hypothesis",
        "update_hypothesis_status",
        "append_hypothesis_evidence",
        "list_hypotheses",
    ):
        assert tool in names, (
            f"Hypothesis tool {tool!r} must be requested by DebugAgent, "
            f"but it is missing from: {names}"
        )


# ---------------------------------------------------------------------------
# Test 9 — Module-level constants are wired into the requested list
# ---------------------------------------------------------------------------


def test_debug_agent_module_constants_match_spec():
    """
    The module-level DEBUG_AGENT_BASE_TOOLS and DEBUG_AGENT_DAP_TOOLS tuples must
    exactly match the spec — this guards against drift between the spec and the
    code without requiring a full _build_agent invocation.
    """
    from app.modules.intelligence.agents.chat_agents.system_agents import debug_agent

    assert set(debug_agent.DEBUG_AGENT_BASE_TOOLS) == set(EXPECTED_BASE_TOOLS)
    assert set(debug_agent.DEBUG_AGENT_DAP_TOOLS) == set(EXPECTED_DAP_TOOLS)
    assert set(debug_agent.DEBUG_AGENT_TERMINAL_TOOLS) == set(EXPECTED_TERMINAL_TOOLS)
    # LOCAL_MODE_ONLY_TOOL_NAMES must cover DAP + terminal for the non-local filter.
    assert debug_agent.DAP_TOOL_NAMES == set(debug_agent.DEBUG_AGENT_DAP_TOOLS)
    assert debug_agent.TERMINAL_TOOL_NAMES == set(debug_agent.DEBUG_AGENT_TERMINAL_TOOLS)
    assert debug_agent.LOCAL_MODE_ONLY_TOOL_NAMES == (
        debug_agent.DAP_TOOL_NAMES | debug_agent.TERMINAL_TOOL_NAMES
    )


def test_debug_agent_includes_terminal_tools_in_local_mode_request():
    """Terminal tools must be requested (filtered out only when local_mode=False)."""
    agent = _make_debug_agent()
    ctx = _make_minimal_ctx()
    ctx.local_mode = True

    names = _requested_tool_names(agent, ctx)

    for tool in EXPECTED_TERMINAL_TOOLS:
        assert tool in names, (
            f"Terminal tool {tool!r} must be requested by DebugAgent in local_mode, "
            f"but it is missing from: {names}"
        )
