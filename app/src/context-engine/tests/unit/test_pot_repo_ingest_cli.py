"""Contract tests for the `potpie pot repo ingest` CLI + its API client method.

This wires the one-shot ingestion skill to a host trigger. Two surfaces under
test:

1. ``PotpieContextApiClient.submit_event`` — POSTs a normalized event to
   ``/api/v2/context/events/reconcile``. Pins URL, body shape, and status
   code branching (202 queued / 409 duplicate).
2. ``potpie pot repo ingest`` (Typer command) — argument parsing, owner/repo
   normalization, pot resolution, and that the resulting `submit_event` call
   carries the right (source_system, event_type, action) tuple so the
   reconciliation agent routes to the ``repo_one_shot_ingestion`` playbook.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main
from adapters.outbound.http.potpie_context_api_client import (
    PotpieContextApiClient,
    PotpieContextApiError,
)

pytestmark = pytest.mark.unit

runner = CliRunner()


# ---------------------------------------------------------------------------
# API client — POST /events/reconcile
# ---------------------------------------------------------------------------


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: httpx.Response,
    captured: dict[str, Any],
) -> None:
    """Replace httpx.Client with a fake that records the outgoing request."""

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any, **k: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            return response

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )


def test_submit_event_queued_202(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_httpx(
        monkeypatch,
        response=httpx.Response(
            202, json={"status": "queued", "event_id": "evt-1", "batch_id": "b-1"}
        ),
        captured=captured,
    )

    c = PotpieContextApiClient("http://example.com", "k")
    status, body = c.submit_event(
        pot_id="pot-1",
        source_system="github",
        event_type="repository",
        action="one_shot_ingest",
        repo_name="acme/api",
        source_id="one_shot_ingest:acme/api:42",
        payload={"owner": "acme", "repo": "api", "count": 50},
    )

    assert status == 202
    assert body == {"status": "queued", "event_id": "evt-1", "batch_id": "b-1"}

    # URL targets the canonical reconcile endpoint.
    assert captured["url"].endswith("/api/v2/context/events/reconcile")

    # Request body has every field /events/reconcile requires.
    sent = captured["json"]
    assert sent["pot_id"] == "pot-1"
    assert sent["source_system"] == "github"
    assert sent["event_type"] == "repository"
    assert sent["action"] == "one_shot_ingest"
    assert sent["repo_name"] == "acme/api"
    assert sent["source_id"] == "one_shot_ingest:acme/api:42"
    assert sent["provider"] == "github"
    assert sent["provider_host"] == "github.com"
    assert sent["payload"] == {"owner": "acme", "repo": "api", "count": 50}


def test_submit_event_duplicate_409_returns_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_httpx(
        monkeypatch,
        response=httpx.Response(
            409,
            json={
                "detail": {
                    "error": "duplicate_event",
                    "event_id": "evt-existing",
                    "message": "Event already recorded for this scope and source_id.",
                }
            },
        ),
        captured=captured,
    )

    c = PotpieContextApiClient("http://example.com", "k")
    status, body = c.submit_event(
        pot_id="pot-1",
        source_system="github",
        event_type="repository",
        action="one_shot_ingest",
        repo_name="acme/api",
        source_id="dup-source-id",
    )

    assert status == 409
    assert body["error"] == "duplicate_event"
    assert body["event_id"] == "evt-existing"


def test_submit_event_400_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_httpx(
        monkeypatch,
        response=httpx.Response(400, json={"detail": "missing repo_name"}),
        captured=captured,
    )

    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(PotpieContextApiError) as ei:
        c.submit_event(
            pot_id="pot-1",
            source_system="github",
            event_type="repository",
            action="one_shot_ingest",
            repo_name="",
            source_id="x",
        )
    assert ei.value.status_code == 400


# ---------------------------------------------------------------------------
# CLI — potpie pot repo ingest
# ---------------------------------------------------------------------------


class _FakeIngestClient:
    """Captures submit_event calls. Mirrors the API client surface the CLI uses."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: tuple[int, dict[str, Any]] = (
            202,
            {"status": "queued", "event_id": "evt-1", "batch_id": "b-1"},
        )

    def submit_event(self, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        self.calls.append(kwargs)
        return self.response


@pytest.fixture
def fake_ingest(monkeypatch: pytest.MonkeyPatch) -> _FakeIngestClient:
    client = _FakeIngestClient()
    monkeypatch.setattr(cli_main, "_cli_client_or_exit", lambda _verbose: client)
    monkeypatch.setattr(
        cli_main, "_pot_id_or_git", lambda _ref, cwd=None: "pot-1"
    )
    monkeypatch.setattr(cli_main, "load_cli_env", lambda: None)
    return client


def test_cli_ingest_happy_path_json(fake_ingest: _FakeIngestClient) -> None:
    result = runner.invoke(
        cli_main.app,
        ["--json", "pot", "repo", "ingest", "Acme/API", "--pot", "pot-1"],
    )
    assert result.exit_code == 0, result.stdout

    assert len(fake_ingest.calls) == 1
    call = fake_ingest.calls[0]
    # Repo is lowercased before submission.
    assert call["repo_name"] == "acme/api"
    assert call["payload"]["owner"] == "acme"
    assert call["payload"]["repo"] == "api"
    # Routes to the one-shot ingestion playbook.
    assert call["source_system"] == "github"
    assert call["event_type"] == "repository"
    assert call["action"] == "one_shot_ingest"
    # Pot resolution is respected.
    assert call["pot_id"] == "pot-1"
    # source_id has the stable prefix and a per-invocation suffix.
    assert call["source_id"].startswith("one_shot_ingest:acme/api:")
    # Default count.
    assert call["payload"]["count"] == 50
    # JSON output contains the event id.
    assert "evt-1" in result.stdout


def test_cli_ingest_count_flag_is_forwarded(fake_ingest: _FakeIngestClient) -> None:
    result = runner.invoke(
        cli_main.app,
        ["pot", "repo", "ingest", "acme/api", "--count", "25"],
    )
    assert result.exit_code == 0, result.stdout
    call = fake_ingest.calls[0]
    assert call["payload"]["count"] == 25


def test_cli_ingest_rejects_invalid_owner_repo(
    fake_ingest: _FakeIngestClient,
) -> None:
    result = runner.invoke(
        cli_main.app, ["pot", "repo", "ingest", "not-a-slash-form"]
    )
    assert result.exit_code != 0
    assert not fake_ingest.calls, (
        "CLI should reject before any API call when owner/repo is malformed"
    )


def test_cli_ingest_handles_409_duplicate(fake_ingest: _FakeIngestClient) -> None:
    fake_ingest.response = (
        409,
        {
            "error": "duplicate_event",
            "event_id": "evt-existing",
            "message": "Event already recorded for this scope and source_id.",
        },
    )
    result = runner.invoke(
        cli_main.app, ["pot", "repo", "ingest", "acme/api"]
    )
    # Duplicate is not a hard failure — the event already exists.
    assert result.exit_code == 0, result.stdout
    assert "Duplicate" in result.stdout or "duplicate" in result.stdout
    assert "evt-existing" in result.stdout


def test_cli_ingest_api_error_surfaces_nonzero_exit(
    fake_ingest: _FakeIngestClient,
) -> None:
    def boom(**_: Any) -> tuple[int, dict[str, Any]]:
        raise PotpieContextApiError(404, {"detail": "pot not found"})

    fake_ingest.submit_event = boom  # type: ignore[method-assign]
    result = runner.invoke(
        cli_main.app, ["pot", "repo", "ingest", "acme/api"]
    )
    assert result.exit_code != 0


def test_cli_ingest_count_bounds_enforced(fake_ingest: _FakeIngestClient) -> None:
    # min=1
    r1 = runner.invoke(
        cli_main.app, ["pot", "repo", "ingest", "acme/api", "--count", "0"]
    )
    assert r1.exit_code != 0
    # max=500
    r2 = runner.invoke(
        cli_main.app,
        ["pot", "repo", "ingest", "acme/api", "--count", "9999"],
    )
    assert r2.exit_code != 0
    assert not fake_ingest.calls, (
        "out-of-range count should be rejected before any API call"
    )
