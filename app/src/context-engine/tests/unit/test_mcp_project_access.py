"""Tests for MCP project allowlist."""

import pytest

from adapters.inbound.mcp import project_access as pa


@pytest.fixture(autouse=True)
def reset_mcp_log_state():
    pa._log_state.allowlist_logged = False
    pa._log_state.trust_all_logged = False
    yield
    pa._log_state.allowlist_logged = False
    pa._log_state.trust_all_logged = False


def test_mcp_deny_when_unconfigured(monkeypatch):
    monkeypatch.delenv("CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS", raising=False)
    monkeypatch.delenv("CONTEXT_ENGINE_MCP_TRUST_ALL_PROJECTS", raising=False)
    with pytest.raises(ValueError, match="MCP access denied"):
        pa.assert_mcp_project_allowed("any-id")


def test_mcp_allowlist(monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS", '["a","b"]')
    pa.assert_mcp_project_allowed("a")
    with pytest.raises(ValueError, match="not permitted"):
        pa.assert_mcp_project_allowed("c")


def test_mcp_trust_all(monkeypatch):
    monkeypatch.delenv("CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS", raising=False)
    monkeypatch.setenv("CONTEXT_ENGINE_MCP_TRUST_ALL_PROJECTS", "true")
    pa.assert_mcp_project_allowed("anything")
