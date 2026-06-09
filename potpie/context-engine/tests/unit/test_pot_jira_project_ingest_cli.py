"""CLI coverage for ``potpie pot jira-project ingest``.

The command lives in :mod:`adapters.inbound.cli.commands.pots`. Stubs the
module-level ``_potpie_api_client`` factory and ``resolve_pot_id`` helper
so the test exercises the command body end-to-end without touching real
host wiring or HTTP.
"""

from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from adapters.inbound.cli.commands import _common, pots
from adapters.outbound.http.potpie_context_api_client import (
    PotpieContextApiError,
)


class _FakeClient:
    """Minimal ``PotpieContextApiClient`` stub recording submit_event kwargs."""

    def __init__(self, *, response: tuple[int, dict[str, Any]]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def submit_event(self, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        self.calls.append(kwargs)
        return self.response


def _install(monkeypatch, fake: _FakeClient, *, json_mode: bool) -> None:
    monkeypatch.setattr(pots, "_potpie_api_client", lambda: fake)
    monkeypatch.setattr(pots, "resolve_pot_id", lambda host, pot: "pot-1")
    monkeypatch.setattr(_common, "get_host", lambda: object())
    _common.set_json(json_mode)


def test_cli_jira_project_ingest_submits_repo_less_event(monkeypatch) -> None:
    fake = _FakeClient(response=(202, {"event_id": "evt-1", "job_id": "job-1"}))
    _install(monkeypatch, fake, json_mode=True)

    result = CliRunner().invoke(
        pots.jira_project_app,
        ["ingest", "PROJ", "--pot", "pot-1", "--count", "7"],
    )

    assert result.exit_code == 0, result.stdout
    assert len(fake.calls) == 1
    sent = fake.calls[0]
    assert sent["pot_id"] == "pot-1"
    assert sent["source_system"] == "jira"
    assert sent["event_type"] == "jira_project"
    assert sent["action"] == "one_shot_ingest"
    assert sent["payload"] == {"project_key": "PROJ", "count": 7}
    assert sent["provider"] is None
    assert sent["provider_host"] is None
    assert sent["repo_name"] is None
    assert sent["source_id"].startswith("one_shot_ingest:jira:proj:")
    assert '"event_id": "evt-1"' in result.stdout
    assert '"job_id": "job-1"' in result.stdout
    assert '"batch_id"' not in result.stdout


def test_cli_jira_project_ingest_plain_message_for_duplicate(monkeypatch) -> None:
    fake = _FakeClient(
        response=(409, {"error": "duplicate_event", "event_id": "evt-dup"})
    )
    _install(monkeypatch, fake, json_mode=False)

    result = CliRunner().invoke(
        pots.jira_project_app,
        ["ingest", "PROJ", "--pot", "pot-1"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Queued Jira project ingest for PROJ" in result.stdout
    assert "evt-dup" in result.stdout


def test_cli_jira_project_ingest_surfaces_api_error(monkeypatch) -> None:
    class _BoomClient(_FakeClient):
        def submit_event(self, **kwargs: Any) -> tuple[int, dict[str, Any]]:
            raise PotpieContextApiError(500, "boom")

    fake = _BoomClient(response=(500, {}))
    _install(monkeypatch, fake, json_mode=False)

    result = CliRunner().invoke(
        pots.jira_project_app, ["ingest", "PROJ", "--pot", "pot-1"]
    )

    assert result.exit_code != 0
    assert "Jira ingest failed" in result.output
