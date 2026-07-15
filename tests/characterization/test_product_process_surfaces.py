"""Behavior locks for relocating Potpie's product processes.

These tests resolve the implementations through the root distribution's
console-script metadata. They intentionally avoid asserting module paths so a
pure relocation can change ownership without changing the public surfaces.
"""

# ruff: noqa: S101 - pytest characterization tests use assertions intentionally.

from __future__ import annotations

import asyncio
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import tomllib

from fastapi import APIRouter
from typer.main import get_command


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SCRIPTS = {"potpie", "potpie-daemon", "potpie-mcp"}
EXPECTED_CLI_COMMANDS = {
    "auth",
    "backend",
    "cloud",
    "config",
    "confluence",
    "daemon",
    "doctor",
    "git",
    "github",
    "graph",
    "jira",
    "ledger",
    "linear",
    "login",
    "logout",
    "pot",
    "record",
    "resolve",
    "search",
    "service",
    "setup",
    "skills",
    "source",
    "status",
    "telemetry",
    "timeline",
    "ui",
    "use",
    "whoami",
}
EXPECTED_MCP_TOOLS = {
    "context_record",
    "context_resolve",
    "context_search",
    "context_status",
}
EXPECTED_DAEMON_ROUTES = {
    ("/attr", frozenset({"POST"})),
    ("/health", frozenset({"GET"})),
    ("/rpc", frozenset({"POST"})),
}


def _scripts() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["scripts"]


def _load_script(name: str) -> tuple[ModuleType, Any]:
    module_name, separator, attribute_name = _scripts()[name].partition(":")
    assert separator == ":", f"{name} must use a module:callable target"
    module = import_module(module_name)
    target = getattr(module, attribute_name)
    assert callable(target), f"{name} target must be callable"
    return module, target


def test_root_distribution_exposes_exactly_three_product_processes() -> None:
    scripts = _scripts()
    assert set(scripts) == EXPECTED_SCRIPTS
    for name in sorted(EXPECTED_SCRIPTS):
        _load_script(name)


def test_cli_top_level_command_surface_is_unchanged() -> None:
    module, _main = _load_script("potpie")
    command = get_command(module.app)
    assert set(command.commands) == EXPECTED_CLI_COMMANDS


def test_daemon_process_routes_are_unchanged(monkeypatch: Any) -> None:
    module, _main = _load_script("potpie-daemon")
    host = SimpleNamespace(backend=SimpleNamespace(profile="characterization"))
    monkeypatch.setattr(module, "build_host_shell", lambda: host)
    monkeypatch.setattr(module, "build_ui_api_router", lambda _host: APIRouter())
    monkeypatch.setattr(module, "mount_ui_static", lambda _app: None)

    app = module.create_app(
        token="characterization-token",  # noqa: S106 - non-secret test fixture
        base_url="http://127.0.0.1:1",
        pid=1,
        log_file="characterization.log",
    )
    routes = {
        (route.path, frozenset(route.methods or ()))
        for route in app.routes
        if route.path in {"/health", "/rpc", "/attr"}
    }
    assert routes == EXPECTED_DAEMON_ROUTES


def test_mcp_tool_surface_is_unchanged() -> None:
    module, _main = _load_script("potpie-mcp")
    tools = asyncio.run(module.mcp.list_tools())
    assert {tool.name for tool in tools} == EXPECTED_MCP_TOOLS
