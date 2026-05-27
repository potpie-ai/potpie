"""
Smoke tests for DebugAgent._build_agent routing (Task 5).

These tests verify that the dispatch logic always returns a PydanticDeepDebugAgent
regardless of `supports_pydantic` or `MultiAgentConfig.should_use_multi_agent`,
and never instantiates PydanticRagAgent or PydanticMultiAgent.

The test file pre-populates sys.modules with lightweight stubs for the heavy ML /
DB packages that are dragged in transitively by the DebugAgent import chain:

  debug_agent → pydantic_deep_debug_agent → tool_service → change_detection_tool
              → inference_service → sentence_transformers → torch

This pattern is consistent with how other tests in this project handle import-time
side-effects (env vars for the DB URL, Neo4j stubs in conftest, etc.).
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Step 1 — set env vars that module-level code reads at import time
# ---------------------------------------------------------------------------
# database.py reads POSTGRES_SERVER at module level; give it a parseable URL
# so create_engine() doesn't explode.  Value never connects to anything.
os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Step 2 — stub out the heavy ML / transformers packages that are dragged in
#          through the import chain before they are ever actually needed.
# ---------------------------------------------------------------------------
# We only do this if the real packages aren't already importable in this env;
# this keeps the stubs out of the way in prod where the real packages exist.
# Stub the ML packages that drag in torch / transformers.  torch has a Python 3.13
# incompatibility (IndentationError in rnn.py) so we must prevent it from loading.
# We only stub if the real package hasn't already been successfully imported.
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
# Make "from torch import _VF as _VF, functional as functional" work.
_torch_stub._VF = MagicMock()  # type: ignore[attr-defined]
_torch_stub.functional = MagicMock()  # type: ignore[attr-defined]
_torch_stub.nn = sys.modules["torch.nn"]  # type: ignore[attr-defined]

_st_stub = _register_stub("sentence_transformers")
_st_stub.SentenceTransformer = MagicMock(name="SentenceTransformer")  # type: ignore[attr-defined]


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

    # tools_provider.get_tools() is called inside _build_agent; return an empty list.
    tools_provider.get_tools.return_value = []

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


# ---------------------------------------------------------------------------
# Test 1 — DebugAgent._build_agent always returns PydanticDeepDebugAgent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("supports_pydantic", [True, False])
@pytest.mark.parametrize("should_use_multi", [True, False])
def test_debug_agent_always_returns_pydantic_deep(supports_pydantic, should_use_multi):
    """
    _build_agent must return PydanticDeepDebugAgent regardless of whether the LLM
    advertises pydantic support or whether MultiAgentConfig says multi-agent is on.

    All other code paths (PydanticRagAgent, PydanticMultiAgent) have been removed.
    """
    from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig

    agent = _make_debug_agent()
    agent.llm_provider.supports_pydantic.return_value = supports_pydantic

    deep_sentinel = MagicMock(name="PydanticDeepDebugAgent_instance")
    DeepCls = MagicMock(return_value=deep_sentinel)

    ctx = _make_minimal_ctx()

    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.PydanticDeepDebugAgent",
        DeepCls,
    ), patch.object(
        MultiAgentConfig, "should_use_multi_agent", return_value=should_use_multi
    ):
        result = agent._build_agent(ctx)

    assert result is deep_sentinel, (
        "Expected PydanticDeepDebugAgent to be used for "
        f"supports_pydantic={supports_pydantic}, should_use_multi={should_use_multi}, "
        f"but got {result!r}"
    )
    DeepCls.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2 — neither PydanticRagAgent nor PydanticMultiAgent is constructed
# ---------------------------------------------------------------------------


def test_debug_agent_does_not_construct_pydantic_rag_or_multi():
    """
    Even when supports_pydantic=True and multi-agent is enabled — historically the
    routing paths that selected PydanticRagAgent / PydanticMultiAgent — _build_agent
    must NOT construct either class. They are no longer imported by debug_agent.py;
    we patch them at their source modules and assert they were never called.
    """
    from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig

    agent = _make_debug_agent()
    agent.llm_provider.supports_pydantic.return_value = True

    deep_sentinel = MagicMock(name="PydanticDeepDebugAgent_instance")
    DeepCls = MagicMock(return_value=deep_sentinel)
    RagCls = MagicMock(name="PydanticRagAgent_class")
    MultiCls = MagicMock(name="PydanticMultiAgent_class")

    ctx = _make_minimal_ctx()

    # Patch the canonical source modules so any rogue import path would land here.
    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.PydanticDeepDebugAgent",
        DeepCls,
    ), patch(
        "app.modules.intelligence.agents.chat_agents.pydantic_agent.PydanticRagAgent",
        RagCls,
    ), patch(
        "app.modules.intelligence.agents.chat_agents.pydantic_multi_agent.PydanticMultiAgent",
        MultiCls,
    ), patch.object(
        MultiAgentConfig, "should_use_multi_agent", return_value=True
    ):
        result = agent._build_agent(ctx)

    assert result is deep_sentinel
    RagCls.assert_not_called()
    MultiCls.assert_not_called()

    # Sanity-check: debug_agent should not even reference these names anymore.
    debug_agent_mod = sys.modules[
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent"
    ]
    assert not hasattr(debug_agent_mod, "PydanticRagAgent"), (
        "debug_agent.py should no longer import PydanticRagAgent"
    )
    assert not hasattr(debug_agent_mod, "PydanticMultiAgent"), (
        "debug_agent.py should no longer import PydanticMultiAgent"
    )


# ---------------------------------------------------------------------------
# Test 3 — DAP tool filtering still happens, observable via PydanticDeepDebugAgent
# ---------------------------------------------------------------------------


def test_debug_agent_excludes_dap_tools_when_not_local_mode():
    """DAP and terminal tools should not be exposed outside VS Code/local-mode runs."""
    from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig

    agent = _make_debug_agent()
    agent.llm_provider.supports_pydantic.return_value = True
    agent.tools_provider.get_tools.return_value = [
        types.SimpleNamespace(name="fetch_file"),
        types.SimpleNamespace(name="start_debug_session"),
        types.SimpleNamespace(name="take_debug_snapshot"),
        types.SimpleNamespace(name="execute_terminal_command"),
        types.SimpleNamespace(name="terminal_session_output"),
    ]

    deep_sentinel = MagicMock(name="PydanticDeepDebugAgent_instance")
    DeepCls = MagicMock(return_value=deep_sentinel)

    ctx = _make_minimal_ctx()
    ctx.local_mode = False

    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.PydanticDeepDebugAgent",
        DeepCls,
    ), patch.object(
        MultiAgentConfig, "should_use_multi_agent", return_value=False
    ):
        agent._build_agent(ctx)

    passed_tools = DeepCls.call_args.args[2]
    assert [tool.name for tool in passed_tools] == ["fetch_file"]


def test_debug_agent_keeps_dap_and_terminal_tools_when_local_mode():
    """VS Code/local-mode runs keep DAP and Potpie terminal tools."""
    from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig

    agent = _make_debug_agent()
    agent.llm_provider.supports_pydantic.return_value = True
    agent.tools_provider.get_tools.return_value = [
        types.SimpleNamespace(name="fetch_file"),
        types.SimpleNamespace(name="start_debug_session"),
        types.SimpleNamespace(name="take_debug_snapshot"),
        types.SimpleNamespace(name="execute_terminal_command"),
        types.SimpleNamespace(name="terminal_session_output"),
    ]

    deep_sentinel = MagicMock(name="PydanticDeepDebugAgent_instance")
    DeepCls = MagicMock(return_value=deep_sentinel)

    ctx = _make_minimal_ctx()
    ctx.local_mode = True

    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.PydanticDeepDebugAgent",
        DeepCls,
    ), patch.object(
        MultiAgentConfig, "should_use_multi_agent", return_value=False
    ):
        agent._build_agent(ctx)

    passed_tools = DeepCls.call_args.args[2]
    assert [tool.name for tool in passed_tools] == [
        "fetch_file",
        "start_debug_session",
        "take_debug_snapshot",
        "execute_terminal_command",
        "terminal_session_output",
    ]


# ---------------------------------------------------------------------------
# Test 4 — MultiAgentConfig default for debugging_agent is False (A7.1 gate)
# ---------------------------------------------------------------------------


def test_multi_agent_config_default_for_debugging_agent_is_false(monkeypatch):
    """
    With no env-var override, should_use_multi_agent("debugging_agent") must
    return False (A7.1 change).
    """
    # Remove any env-var override that might be set in the test environment.
    monkeypatch.delenv("DEBUG_MULTI_AGENT", raising=False)
    # Ensure the global kill-switch is not accidentally masking via ENABLE_MULTI_AGENT=false.
    monkeypatch.delenv("ENABLE_MULTI_AGENT", raising=False)

    # Reload so class attributes pick up the (now-absent) env-var state.
    import app.modules.intelligence.agents.multi_agent_config as cfg_mod
    importlib.reload(cfg_mod)
    MultiAgentConfig = cfg_mod.MultiAgentConfig

    result = MultiAgentConfig.should_use_multi_agent("debugging_agent")
    assert result is False, (
        "Expected MultiAgentConfig.should_use_multi_agent('debugging_agent') "
        f"to return False by default (A7.1), but got {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — env-var override DEBUG_MULTI_AGENT=true flips the flag
# ---------------------------------------------------------------------------


def test_multi_agent_config_env_var_override_works_for_debugging_agent(monkeypatch):
    """
    Setting DEBUG_MULTI_AGENT=true must override the A7.1 default and return True.

    How to flip back to multi-agent later: set DEBUG_MULTI_AGENT=true in the
    deployment environment (no code change required).
    """
    monkeypatch.setenv("DEBUG_MULTI_AGENT", "true")
    monkeypatch.delenv("ENABLE_MULTI_AGENT", raising=False)

    import app.modules.intelligence.agents.multi_agent_config as cfg_mod
    importlib.reload(cfg_mod)
    MultiAgentConfig = cfg_mod.MultiAgentConfig

    result = MultiAgentConfig.should_use_multi_agent("debugging_agent")
    assert result is True, (
        "Expected DEBUG_MULTI_AGENT=true to enable multi-agent for debugging_agent, "
        f"but got {result!r}"
    )
