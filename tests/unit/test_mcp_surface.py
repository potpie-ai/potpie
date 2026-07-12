"""Root ownership and runtime routing for the public MCP surface."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli import host_cli
from potpie.cli.commands import bootstrap
from potpie.mcp import server
from potpie.setup.contracts import ProductStatusResult

pytestmark = pytest.mark.unit

EXPECTED_TOOLS = {
    "context_resolve",
    "context_search",
    "context_record",
    "context_status",
}


class _Envelope:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


class _ContextClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def resolve(self, request: Any) -> _Envelope:
        self.calls.append(("resolve", request))
        return _Envelope({"pot_id": request.pot_id, "task": request.task})

    async def search(self, request: Any) -> _Envelope:
        self.calls.append(("search", request))
        return _Envelope({"pot_id": request.pot_id, "query": request.query})

    async def record(self, request: Any) -> Any:
        self.calls.append(("record", request))
        return SimpleNamespace(
            accepted=True,
            status="recorded",
            record_id="record-1",
            mutations_applied=1,
            detail=None,
        )


class _StatusService:
    def __init__(self, result: ProductStatusResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def get(self, **kwargs: Any) -> ProductStatusResult:
        self.calls.append(kwargs)
        return self.result


def _status_result(*, runtime_mode: str = "daemon") -> ProductStatusResult:
    return ProductStatusResult(
        schema_version="1",
        ready=True,
        runtime_mode=runtime_mode,
        daemon_state="up",
        pot_id="pot-1",
        pot_name="default",
        backend="embedded",
        backend_ready=True,
        storage_ready=True,
        ingestion_ready=True,
        source_count=2,
        skills_state="ready",
        setup_state="configured",
    )


def _runtime(*, runtime_mode: str = "daemon") -> Any:
    context = _ContextClient()
    status = _StatusService(_status_result(runtime_mode=runtime_mode))
    return SimpleNamespace(
        engine=SimpleNamespace(context=context),
        settings=SimpleNamespace(runtime_mode=runtime_mode),
        status=status,
    )


@pytest.fixture(autouse=True)
def trust_test_pots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_MCP_ALLOWED_POTS", raising=False)
    monkeypatch.setenv("CONTEXT_ENGINE_MCP_TRUST_ALL_POTS", "true")


def test_mcp_discovery_exposes_exactly_four_tools() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    assert {tool.name for tool in tools} == EXPECTED_TOOLS


@pytest.mark.parametrize("runtime_mode", ["daemon", "in-process"])
def test_context_tools_route_through_runtime_engine(
    monkeypatch: pytest.MonkeyPatch,
    runtime_mode: str,
) -> None:
    runtime = _runtime(runtime_mode=runtime_mode)
    monkeypatch.setattr(server, "get_runtime", lambda: runtime)

    resolved = server.context_resolve(
        "pot-1",
        "trace auth",
        repo_name="potpie",
        services="api, worker",
        include="decisions,docs",
    )
    searched = server.context_search(
        "pot-1", "oauth callback", include="docs", repo_name="potpie"
    )
    recorded = server.context_record(
        "pot-1",
        "decision",
        "Keep MCP in root",
        source_refs="spec:boundary,adr:2",
        details="Product-owned surface",
    )

    assert resolved == {"ok": True, "pot_id": "pot-1", "task": "trace auth"}
    assert searched == {
        "ok": True,
        "pot_id": "pot-1",
        "query": "oauth callback",
    }
    assert recorded == {
        "ok": True,
        "status": "recorded",
        "record_id": "record-1",
        "mutations_applied": 1,
        "detail": None,
    }

    context = runtime.engine.context
    resolve_request = context.calls[0][1]
    assert resolve_request.scope == {
        "repo_name": "potpie",
        "services": ["api", "worker"],
    }
    assert resolve_request.include == ("decisions", "docs")
    record_request = context.calls[2][1]
    assert record_request.source_refs == ("spec:boundary", "adr:2")
    assert record_request.details == {
        "confidence": 0.7,
        "visibility": "project",
        "text": "Product-owned surface",
    }


def test_context_status_matches_cli_flat_status_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(server, "get_runtime", lambda: runtime)
    monkeypatch.setattr(bootstrap, "get_cli_runtime", lambda: runtime)

    mcp_data = server.context_status("pot-1", intent="review", harness="codex")
    cli_result = CliRunner().invoke(
        host_cli.app, ["--json", "status", "--pot", "pot-1"]
    )

    assert cli_result.exit_code == 0, cli_result.output
    assert json.loads(cli_result.stdout)["data"] == mcp_data
    assert set(mcp_data) == {
        "schema_version",
        "ready",
        "runtime_mode",
        "daemon_state",
        "pot_id",
        "pot_name",
        "backend",
        "backend_ready",
        "storage_ready",
        "ingestion_ready",
        "source_count",
        "last_ingestion_at",
        "skills_state",
        "setup_state",
        "issues",
        "recommended_next_action",
    }
    assert runtime.status.calls[0] == {"pot_id": "pot-1", "harness": "codex"}


def test_mcp_runtime_failure_uses_protocol_native_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from potpie.runtime.errors import RuntimeDaemonUnavailable

    class _UnavailableContext:
        async def search(self, request: Any) -> Any:
            del request
            raise RuntimeDaemonUnavailable()

    runtime = SimpleNamespace(
        engine=SimpleNamespace(context=_UnavailableContext()),
    )
    monkeypatch.setattr(server, "get_runtime", lambda: runtime)

    assert server.context_search("pot-1", "query") == {
        "ok": False,
        "error": "RuntimeDaemonUnavailable",
        "detail": "The Potpie daemon is not reachable.",
    }
