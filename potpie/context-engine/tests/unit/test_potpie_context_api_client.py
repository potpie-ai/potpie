"""Potpie /api/v2/context HTTP client."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from potpie.context_engine.domain.errors import CapabilityNotImplemented
from potpie.context_engine.adapters.outbound.http.potpie_context_api_client import (
    IngestRejectedError,
    PotpieContextApiClient,
    PotpieContextApiError,
)


def test_client_context_graph_query_is_not_supported() -> None:
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(CapabilityNotImplemented) as exc:
        c.context_graph_query({"pot_id": "p1", "query": "q", "limit": 8})
    assert exc.value.capability == "http.context_graph_query"


def test_client_context_graph_query_bearer_surface_is_not_supported() -> None:
    c = PotpieContextApiClient(
        "http://example.com",
        auth_headers={"Authorization": "Bearer id-token"},
    )
    with pytest.raises(CapabilityNotImplemented) as exc:
        c.context_graph_query({"pot_id": "p1", "query": "q"})
    assert exc.value.capability == "http.context_graph_query"


def test_client_uses_auth_header_provider_for_get_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any, **k: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            assert "/api/v2/context/pots" in url
            headers = kwargs["headers"]
            assert headers.get("Authorization") == "Bearer dynamic-token"
            assert "Content-Type" not in headers
            return httpx.Response(200, json=[{"id": "c1"}])

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient(
        "http://example.com",
        auth_headers_provider=lambda: (
            provider_calls.append("called") or {"Authorization": "Bearer dynamic-token"}
        ),
    )

    rows = c.list_context_pots()

    assert rows == [{"id": "c1"}]
    assert provider_calls == ["called"]


def _always_401_get_client(get_calls: list[str]) -> type:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            get_calls.append(kwargs["headers"].get("Authorization", ""))
            return httpx.Response(401)

    return FakeClient


def test_client_refreshes_auth_on_401_via_reauth_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 401 forces a re-auth and retries once with the refreshed headers."""
    get_calls: list[str] = []

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            get_calls.append(kwargs["headers"].get("Authorization", ""))
            if len(get_calls) == 1:
                return httpx.Response(401)
            return httpx.Response(200, json=[{"id": "c1"}])

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient(
        "http://example.com",
        auth_headers={"Authorization": "Bearer stale-token"},
        reauth_provider=lambda: {"Authorization": "Bearer fresh-token"},
    )

    rows = c.list_context_pots()

    assert rows == [{"id": "c1"}]
    # First request uses the stale token; the retry uses the refreshed one.
    assert get_calls == ["Bearer stale-token", "Bearer fresh-token"]


def test_client_does_not_retry_401_without_reauth_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No reauth hook (e.g. plain API key) → a 401 surfaces, no wasted retry."""
    get_calls: list[str] = []
    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        _always_401_get_client(get_calls),
    )
    c = PotpieContextApiClient("http://example.com", "k")

    with pytest.raises(PotpieContextApiError) as ei:
        c.list_context_pots()

    assert ei.value.status_code == 401
    assert len(get_calls) == 1


def test_client_does_not_retry_401_when_reauth_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reauth that returns identical headers → no retry (retry couldn't succeed)."""
    get_calls: list[str] = []
    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        _always_401_get_client(get_calls),
    )
    c = PotpieContextApiClient(
        "http://example.com",
        auth_headers={"Authorization": "Bearer t"},
        reauth_provider=lambda: {"Authorization": "Bearer t"},
    )

    with pytest.raises(PotpieContextApiError) as ei:
        c.list_context_pots()

    assert ei.value.status_code == 401
    assert len(get_calls) == 1


def test_client_ingest_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert "/ingest" in url
            return httpx.Response(
                202,
                json={"status": "queued", "event_id": "e1", "job_id": "j1"},
            )

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, data = c.ingest(
        {
            "pot_id": "p",
            "name": "n",
            "episode_body": "b",
            "source_description": "cli",
            "reference_time": "2025-01-01T00:00:00+00:00",
        },
        sync=False,
    )
    assert code == 202
    assert data["event_id"] == "e1"


def test_submit_event_omits_non_repo_scope_fields_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            sent["url"] = url
            sent["payload"] = kwargs.get("json")
            return httpx.Response(202, json={"status": "queued", "event_id": "e1"})

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, data = c.submit_event(
        pot_id="pot-1",
        source_system="linear",
        event_type="linear_team",
        action="one_shot_ingest",
        source_id="one_shot_ingest:linear:eng:42",
        payload={"team": "ENG", "count": 120},
        provider=None,
        provider_host=None,
        repo_name=None,
    )

    assert code == 202
    assert data["event_id"] == "e1"
    assert sent["url"].endswith("/api/v2/context/events/reconcile")
    assert "provider" not in sent["payload"]
    assert "provider_host" not in sent["payload"]
    assert "repo_name" not in sent["payload"]
    assert sent["payload"]["payload"] == {"team": "ENG", "count": 120}


def test_client_context_graph_query_does_not_attempt_remote_error_path() -> None:
    c = PotpieContextApiClient("http://example.com", "bad")
    with pytest.raises(CapabilityNotImplemented) as ei:
        c.context_graph_query({"pot_id": "p", "query": "q", "limit": 1})
    assert ei.value.capability == "http.context_graph_query"


def test_list_context_pots_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any, **k: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            assert "/api/v2/context/pots" in url
            assert kwargs["headers"].get("X-API-Key") == "k"
            return httpx.Response(200, json=[{"id": "c1", "display_name": "x"}])

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    rows = c.list_context_pots()
    assert rows == [{"id": "c1", "display_name": "x"}]


def test_create_context_pot_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any, **k: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith("/api/v2/context/pots")
            assert kwargs["json"] == {"slug": "n"}
            return httpx.Response(200, json={"id": "new-id", "slug": "n"})

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    out = c.create_context_pot(slug="n")
    assert out["id"] == "new-id"


def test_get_context_pot_slug_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any, **k: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith("/api/v2/context/pots/slug-availability/my-pot")
            assert kwargs["headers"].get("X-API-Key") == "k"
            return httpx.Response(200, json={"slug": "my-pot", "available": True})

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    out = c.get_context_pot_slug_availability("my-pot")
    assert out == {"slug": "my-pot", "available": True}


def test_list_pot_repositories_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any, **k: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith("/api/v2/context/pots/p1/repositories")
            assert kwargs["headers"].get("X-API-Key") == "k"
            return httpx.Response(
                200,
                json=[{"id": "r1", "repo_name": "o/r", "provider": "github"}],
            )

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    rows = c.list_pot_repositories("p1")
    assert rows == [{"id": "r1", "repo_name": "o/r", "provider": "github"}]


def test_add_pot_repository_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any, **k: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith("/api/v2/context/pots/p1/repositories")
            assert kwargs["json"] == {
                "owner": "o",
                "repo": "r",
                "provider": "github",
                "provider_host": "github.com",
            }
            return httpx.Response(200, json={"id": "rid", "repo_name": "o/r"})

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    out = c.add_pot_repository("p1", owner="o", repo="r")
    assert out["id"] == "rid"


def test_json_sanitize_datetime() -> None:
    from potpie.context_engine.adapters.outbound.http.potpie_context_api_client import _json_body_for_httpx
    from datetime import datetime, timezone

    dt = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    out = _json_body_for_httpx({"as_of": dt, "n": 1})
    assert out["as_of"] == dt.isoformat()
    assert out["n"] == 1


def test_json_sanitize_datetime_nested_list() -> None:
    from potpie.context_engine.adapters.outbound.http.potpie_context_api_client import _json_body_for_httpx
    from datetime import datetime, timezone

    dt = datetime(2025, 3, 1, tzinfo=timezone.utc)
    out = _json_body_for_httpx({"dates": [dt, dt]})
    assert out["dates"] == [dt.isoformat(), dt.isoformat()]


def test_json_sanitize_datetime_passthrough_non_datetime() -> None:
    from potpie.context_engine.adapters.outbound.http.potpie_context_api_client import _json_body_for_httpx

    out = _json_body_for_httpx({"x": 42, "y": "hello", "z": None})
    assert out == {"x": 42, "y": "hello", "z": None}


def test_client_ingest_sync_passes_param(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            captured["params"] = kwargs.get("params")
            return httpx.Response(200, json={"event_id": "sync-1", "status": "ok"})

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, data = c.ingest(
        {"pot_id": "p", "name": "n", "episode_body": "b", "source_description": "s"},
        sync=True,
    )
    assert code == 200
    assert captured["params"] == {"sync": "true"}


def test_client_ingest_async_no_param(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            captured["params"] = kwargs.get("params")
            return httpx.Response(202, json={"event_id": "e1", "status": "queued"})

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, _ = c.ingest(
        {"pot_id": "p", "name": "n", "episode_body": "b", "source_description": "s"},
        sync=False,
    )
    assert code == 202
    assert captured["params"] is None


def test_client_ingest_duplicate_409_returns_event_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            return httpx.Response(
                409,
                json={"detail": {"error": "duplicate_ingest", "event_id": "dup-123"}},
            )

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, data = c.ingest(
        {"pot_id": "p", "name": "n", "episode_body": "b", "source_description": "s"},
        sync=True,
    )
    assert code == 409
    assert data["error"] == "duplicate_ingest"
    assert data["event_id"] == "dup-123"


def test_client_ingest_non_duplicate_409_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            return httpx.Response(409, json={"detail": "Conflict"})

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(PotpieContextApiError) as exc_info:
        c.ingest(
            {
                "pot_id": "p",
                "name": "n",
                "episode_body": "b",
                "source_description": "s",
            },
            sync=True,
        )
    assert exc_info.value.status_code == 409


def test_client_ingest_422_raises_ingest_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any, **k: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            return httpx.Response(
                422,
                json={
                    "status": "reconciliation_rejected",
                    "event_id": "e-rej",
                    "mutation_id": None,
                    "errors": [
                        {"entity": "adr:1", "issue": "unknown canonical labels: X"}
                    ],
                    "downgrades": [],
                },
            )

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(IngestRejectedError) as exc_info:
        c.ingest(
            {
                "pot_id": "p",
                "name": "n",
                "episode_body": "b",
                "source_description": "s",
            },
            sync=True,
        )
    assert exc_info.value.body["status"] == "reconciliation_rejected"
    assert exc_info.value.body["errors"][0]["entity"] == "adr:1"


def test_client_get_event_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith("/api/v2/context/events/e1")
            assert kwargs["headers"].get("X-API-Key") == "k"
            return httpx.Response(200, json={"event_id": "e1", "status": "done"})

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    out = c.get_event("e1")
    assert out == {"event_id": "e1", "status": "done"}


def test_client_list_events_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith("/api/v2/context/pots/p1/events")
            captured["params"] = kwargs.get("params")
            return httpx.Response(
                200,
                json={"items": [{"event_id": "e1", "status": "queued"}]},
            )

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    out = c.list_events("p1", limit=5, status="queued", ingestion_kind="raw_episode")
    assert captured["params"] == {
        "limit": 5,
        "status": "queued",
        "ingestion_kind": "raw_episode",
    }
    assert out["items"][0]["event_id"] == "e1"


def test_client_reset_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert "/reset" in url
            return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    result = c.reset({"pot_id": "p"})
    assert result["ok"] is True


def test_client_record_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert "/record" in url
            return httpx.Response(
                200, json={"ok": True, "status": "accepted", "event_id": "r1"}
            )

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    result = c.record(
        {"pot_id": "p", "record_type": "decision", "summary": "use postgres"}
    )
    assert result["ok"] is True


def test_client_status_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert "/status" in url
            return httpx.Response(
                200,
                json={"ok": True, "coverage": {"status": "complete"}, "fallbacks": 0},
            )

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    result = c.status({"pot_id": "p"})
    assert result["ok"] is True
    assert result["coverage"]["status"] == "complete"


def test_client_health_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith("/health")
            return httpx.Response(200, json={"status": "ok"})

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, body = c.get_health()
    assert code == 200
    assert body == {"status": "ok"}


def test_client_health_non_200_returns_none_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "potpie.context_engine.adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, body = c.get_health()
    assert code == 503
    assert body is None


def test_client_context_graph_query_result_surface_is_not_supported() -> None:
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(CapabilityNotImplemented) as exc:
        c.context_graph_query({"pot_id": "p", "query": "q"})
    assert exc.value.capability == "http.context_graph_query"


def test_client_context_graph_query_500_path_is_not_supported() -> None:
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(CapabilityNotImplemented) as exc_info:
        c.context_graph_query({"pot_id": "p", "query": "q", "limit": 1})
    assert exc_info.value.capability == "http.context_graph_query"


def test_client_context_graph_query_json_error_path_is_not_supported() -> None:
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(CapabilityNotImplemented) as exc_info:
        c.context_graph_query({"pot_id": "p", "query": "q", "limit": 1})
    assert exc_info.value.capability == "http.context_graph_query"
