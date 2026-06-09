"""CLI coverage for the ``linear-team`` / ``jira-project`` diff-sync commands."""

from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from adapters.inbound.cli.commands import _common, pots
from adapters.outbound.http.potpie_context_api_client import PotpieContextApiError


class _FakeClient:
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


def test_linear_diff_sync_emits_diff_sync_event(monkeypatch) -> None:
    fake = _FakeClient(response=(202, {"event_id": "evt-1", "batch_id": "b-1"}))
    _install(monkeypatch, fake, json_mode=True)

    result = CliRunner().invoke(
        pots.linear_team_app,
        ["diff-sync", "ENG", "--pot", "pot-1", "--since", "2026-06-01T00:00:00Z"],
    )

    assert result.exit_code == 0, result.stdout
    sent = fake.calls[0]
    assert sent["source_system"] == "linear"
    assert sent["event_type"] == "linear_team"
    assert sent["action"] == "diff_sync"
    assert sent["payload"]["team"] == "ENG"
    assert sent["payload"]["since"] == "2026-06-01T00:00:00Z"
    assert sent["source_id"].startswith("diff_sync:linear:eng:")


def test_linear_diff_sync_omits_since_when_absent(monkeypatch) -> None:
    fake = _FakeClient(response=(202, {"event_id": "evt-1"}))
    _install(monkeypatch, fake, json_mode=True)

    result = CliRunner().invoke(
        pots.linear_team_app, ["diff-sync", "ENG", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.stdout
    assert "since" not in fake.calls[0]["payload"]


def test_jira_diff_sync_emits_diff_sync_event(monkeypatch) -> None:
    fake = _FakeClient(response=(202, {"event_id": "evt-2", "job_id": "j-2"}))
    _install(monkeypatch, fake, json_mode=True)

    result = CliRunner().invoke(
        pots.jira_project_app,
        ["diff-sync", "PROJ", "--pot", "pot-1", "--count", "9"],
    )

    assert result.exit_code == 0, result.stdout
    sent = fake.calls[0]
    assert sent["source_system"] == "jira"
    assert sent["event_type"] == "jira_project"
    assert sent["action"] == "diff_sync"
    assert sent["payload"] == {"project_key": "PROJ", "count": 9}
    assert sent["source_id"].startswith("diff_sync:jira:proj:")


def test_jira_diff_sync_duplicate_message(monkeypatch) -> None:
    fake = _FakeClient(response=(409, {"event_id": "evt-dup"}))
    _install(monkeypatch, fake, json_mode=False)

    result = CliRunner().invoke(
        pots.jira_project_app, ["diff-sync", "PROJ", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.stdout
    assert "Queued Jira project diff-sync for PROJ" in result.stdout


def test_diff_sync_surfaces_api_error(monkeypatch) -> None:
    class _BoomClient(_FakeClient):
        def submit_event(self, **kwargs: Any) -> tuple[int, dict[str, Any]]:
            raise PotpieContextApiError(500, "boom")

    fake = _BoomClient(response=(500, {}))
    _install(monkeypatch, fake, json_mode=False)

    result = CliRunner().invoke(
        pots.linear_team_app, ["diff-sync", "ENG", "--pot", "pot-1"]
    )

    assert result.exit_code != 0
    assert "Linear diff-sync failed" in result.output
