from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main


def test_cli_jira_project_ingest_submits_repo_less_event(
    monkeypatch,
) -> None:
    sent: dict[str, Any] = {}

    class FakeClient:
        def submit_event(self, **kwargs: Any) -> tuple[int, dict[str, Any]]:
            sent.update(kwargs)
            return 202, {"event_id": "evt-1", "batch_id": "batch-1"}

    monkeypatch.setattr(cli_main, "load_cli_env", lambda: None)
    monkeypatch.setattr(cli_main, "_pot_id_or_git", lambda pot_opt, cwd=None: "pot-1")
    monkeypatch.setattr(cli_main, "_cli_client_or_exit", lambda verbose: FakeClient())

    result = CliRunner().invoke(
        cli_main.app,
        [
            "--json",
            "pot",
            "jira-project",
            "ingest",
            "PROJ",
            "--pot",
            "pot-1",
            "--count",
            "7",
        ],
    )

    assert result.exit_code == 0, result.stdout
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
