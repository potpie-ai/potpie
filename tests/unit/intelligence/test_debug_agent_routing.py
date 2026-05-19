"""
Smoke tests for DebugAgent._build_agent routing (A7.1 / A7.2).

These tests verify that the dispatch logic chooses PydanticRagAgent vs
PydanticMultiAgent correctly based on MultiAgentConfig and supports_pydantic,
WITHOUT spawning any real LLMs.

The test file pre-populates sys.modules with lightweight stubs for the heavy ML /
DB packages that are dragged in transitively by the DebugAgent import chain:

  debug_agent → pydantic_multi_agent → tool_service → change_detection_tool
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
# Test 1 — multi-agent disabled + pydantic supported → PydanticRagAgent
# ---------------------------------------------------------------------------

def test_debug_agent_uses_pydantic_rag_when_multi_disabled():
    """
    When MultiAgentConfig returns False for debugging_agent AND the LLM supports
    pydantic, _build_agent must return a PydanticRagAgent (not PydanticMultiAgent).

    Sentinel mocks replace the real classes so we don't need to instantiate them
    (which would require real LLM credentials / DB connections).
    """
    from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig

    agent = _make_debug_agent()
    agent.llm_provider.supports_pydantic.return_value = True

    rag_sentinel = MagicMock(name="PydanticRagAgent_instance")
    RagCls = MagicMock(return_value=rag_sentinel)
    MultiCls = MagicMock(name="PydanticMultiAgent_class")

    ctx = _make_minimal_ctx()

    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.PydanticRagAgent",
        RagCls,
    ), patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.PydanticMultiAgent",
        MultiCls,
    ), patch.object(
        MultiAgentConfig, "should_use_multi_agent", return_value=False
    ):
        result = agent._build_agent(ctx)

    assert result is rag_sentinel, (
        "Expected PydanticRagAgent to be used when multi-agent is disabled"
    )
    MultiCls.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2 — pydantic unsupported → fallback PydanticRagAgent
# ---------------------------------------------------------------------------

def test_debug_agent_uses_pydantic_rag_when_pydantic_unsupported():
    """
    When the LLM does NOT support pydantic, _build_agent must fall back to
    PydanticRagAgent regardless of the multi-agent config.
    """
    from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig

    agent = _make_debug_agent()
    agent.llm_provider.supports_pydantic.return_value = False

    rag_sentinel = MagicMock(name="PydanticRagAgent_instance")
    RagCls = MagicMock(return_value=rag_sentinel)
    MultiCls = MagicMock(name="PydanticMultiAgent_class")

    ctx = _make_minimal_ctx()

    # Even if multi-agent config says True, pydantic=False forces the fallback.
    with patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.PydanticRagAgent",
        RagCls,
    ), patch(
        "app.modules.intelligence.agents.chat_agents.system_agents.debug_agent.PydanticMultiAgent",
        MultiCls,
    ), patch.object(
        MultiAgentConfig, "should_use_multi_agent", return_value=True
    ):
        result = agent._build_agent(ctx)

    assert result is rag_sentinel, (
        "Expected PydanticRagAgent fallback when pydantic is unsupported"
    )
    MultiCls.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — MultiAgentConfig default for debugging_agent is False (A7.1 gate)
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
# Test 4 — env-var override DEBUG_MULTI_AGENT=true flips the flag
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
