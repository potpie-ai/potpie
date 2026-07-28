"""Reconciliation-agent construction uses its injected runtime config."""

from __future__ import annotations

from types import ModuleType

import pytest

from potpie_context_core.reconciliation_config import ReconciliationConfig
from potpie_context_engine.adapters.outbound.reconciliation.factory import (
    try_pydantic_deep_reconciliation_agent,
)

pytestmark = pytest.mark.unit


def test_agent_factories_coexist_independently_of_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = (
        "potpie_context_engine.adapters.outbound.reconciliation.pydantic_deep_agent"
    )
    fake_module = ModuleType(module_name)

    class FakeAgent:
        def __init__(self, *, tools=None) -> None:
            self.tools = tools

    fake_module.PydanticDeepReconciliationAgent = FakeAgent
    monkeypatch.setitem(__import__("sys").modules, module_name, fake_module)
    tools = object()

    monkeypatch.setenv("CONTEXT_ENGINE_AGENT_PLANNER_ENABLED", "0")
    enabled = try_pydantic_deep_reconciliation_agent(
        tools=tools,
        reconciliation_config=ReconciliationConfig(agent_planner_enabled=True),
    )
    monkeypatch.setenv("CONTEXT_ENGINE_AGENT_PLANNER_ENABLED", "1")
    disabled = try_pydantic_deep_reconciliation_agent(
        tools=tools,
        reconciliation_config=ReconciliationConfig(agent_planner_enabled=False),
    )

    assert isinstance(enabled, FakeAgent)
    assert enabled.tools is tools
    assert disabled is None
