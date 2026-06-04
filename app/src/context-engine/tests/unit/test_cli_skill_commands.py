"""CLI parity with the agent skill surface.

Covers `potpie status`, `potpie resolve`, `potpie overview`, `potpie record`
added to close the skill-vs-CLI gap (docs/context-graph/implementation-next-steps.md
#5, reviewed 2026-04-22).
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main

pytestmark = pytest.mark.unit

runner = CliRunner()


class _FakeClient:
    def __init__(self) -> None:
        self.status_calls: list[dict[str, Any]] = []
        self.graph_query_calls: list[dict[str, Any]] = []
        self.record_calls: list[tuple[dict[str, Any], bool]] = []
        self.status_response: dict[str, Any] = {"ok": True, "capabilities": {}}
        self.graph_response: dict[str, Any] = {
            "kind": "resolve_context",
            "goal": "answer",
            "result": {
                "answer": {"summary": "Chose Neo4j for temporal edges."},
                "ok": True,
            },
        }
        self.record_response: dict[str, Any] = {"ok": True, "event_id": "evt-1"}

    def status(self, body: dict[str, Any]) -> dict[str, Any]:
        self.status_calls.append(body)
        return self.status_response

    def context_graph_query(self, body: dict[str, Any]) -> dict[str, Any]:
        self.graph_query_calls.append(body)
        return self.graph_response

    def record(self, body: dict[str, Any], *, sync: bool = False) -> dict[str, Any]:
        self.record_calls.append((body, sync))
        return self.record_response


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    client = _FakeClient()
    monkeypatch.setattr(cli_main, "_cli_client_or_exit", lambda _verbose: client)
    monkeypatch.setattr(cli_main, "_pot_id_or_git", lambda _ref, cwd=None: "pot-1")
    return client


# --- status ----------------------------------------------------------------------


def test_status_builds_body_and_prints_json(fake: _FakeClient) -> None:
    result = runner.invoke(
        cli_main.app,
        ["--json", "status", "--intent", "fix bug", "--repo", "org/repo", "--pr", "42"],
    )
    assert result.exit_code == 0, result.stdout
    assert len(fake.status_calls) == 1
    body = fake.status_calls[0]
    assert body["pot_id"] == "pot-1"
    assert body["intent"] == "fix bug"
    assert body["scope"] == {"repo_name": "org/repo", "pr_number": 42}
    assert '"ok": true' in result.stdout


def test_status_omits_empty_intent(fake: _FakeClient) -> None:
    result = runner.invoke(cli_main.app, ["--json", "status"])
    assert result.exit_code == 0, result.stdout
    body = fake.status_calls[0]
    assert "intent" not in body


# --- resolve ---------------------------------------------------------------------


def test_resolve_plain_prints_summary(fake: _FakeClient) -> None:
    result = runner.invoke(
        cli_main.app, ["resolve", "why did we choose Neo4j?"]
    )
    assert result.exit_code == 0, result.stdout
    body = fake.graph_query_calls[0]
    assert body["goal"] == "answer"
    assert body["query"] == "why did we choose Neo4j?"
    assert body["pot_id"] == "pot-1"
    assert "Chose Neo4j for temporal edges." in result.stdout


def test_resolve_with_explicit_pot_and_scope(fake: _FakeClient) -> None:
    result = runner.invoke(
        cli_main.app,
        [
            "--json",
            "resolve",
            "pot-1",
            "what changed?",
            "--file",
            "app/x.py",
            "--services",
            "api,worker",
            "--include",
            "decisions,recent_changes",
        ],
    )
    assert result.exit_code == 0, result.stdout
    body = fake.graph_query_calls[0]
    assert body["query"] == "what changed?"
    assert body["scope"]["file_path"] == "app/x.py"
    assert body["scope"]["services"] == ["api", "worker"]
    assert body["include"] == ["decisions", "recent_changes"]


# --- overview --------------------------------------------------------------------


def test_overview_sends_aggregate_and_graph_overview(fake: _FakeClient) -> None:
    result = runner.invoke(cli_main.app, ["--json", "overview"])
    assert result.exit_code == 0, result.stdout
    body = fake.graph_query_calls[0]
    assert body["goal"] == "aggregate"
    assert body["include"] == ["graph_overview"]
    assert body["pot_id"] == "pot-1"


# --- record ----------------------------------------------------------------------


def test_record_builds_payload_and_supports_sync(fake: _FakeClient) -> None:
    result = runner.invoke(
        cli_main.app,
        [
            "--json",
            "record",
            "--type",
            "decision",
            "--summary",
            "Adopted Neo4j",
            "--details",
            '{"rationale": "temporal edges"}',
            "--source-refs",
            "pr:7,issue:12",
            "--confidence",
            "0.9",
            "--sync",
            "--idempotency-key",
            "idem-123",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert len(fake.record_calls) == 1
    body, sync_flag = fake.record_calls[0]
    assert sync_flag is True
    assert body["pot_id"] == "pot-1"
    assert body["idempotency_key"] == "idem-123"
    payload = body["record"]
    assert payload["type"] == "decision"
    assert payload["summary"] == "Adopted Neo4j"
    assert payload["details"] == {"rationale": "temporal edges"}
    assert payload["source_refs"] == ["pr:7", "issue:12"]
    assert payload["confidence"] == 0.9


def test_record_rejects_non_object_details(fake: _FakeClient) -> None:
    result = runner.invoke(
        cli_main.app,
        [
            "record",
            "--type",
            "decision",
            "--summary",
            "x",
            "--details",
            '"a string"',
        ],
    )
    assert result.exit_code == 1
    assert fake.record_calls == []


def test_record_rejects_invalid_json_details(fake: _FakeClient) -> None:
    result = runner.invoke(
        cli_main.app,
        [
            "record",
            "--type",
            "decision",
            "--summary",
            "x",
            "--details",
            "{not json",
        ],
    )
    assert result.exit_code == 1
    assert fake.record_calls == []
