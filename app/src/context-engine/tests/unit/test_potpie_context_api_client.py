"""Potpie /api/v2/context HTTP client."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from adapters.outbound.http.potpie_context_api_client import (
    CONTEXT_API_PREFIX,
    IngestRejectedError,
    PotpieContextApiClient,
    PotpieContextApiError,
)


def test_client_context_graph_query_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert CONTEXT_API_PREFIX in url
            assert url.endswith("/query/context-graph")
            assert kwargs["headers"].get("X-API-Key") == "k"
            body = kwargs.get("json") or {}
            assert body["pot_id"] == "p1"
            return httpx.Response(
                200, json={"kind": "semantic_search", "result": [{"uuid": "u"}]}
            )

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    out = c.context_graph_query({"pot_id": "p1", "query": "q", "limit": 8})
    assert out["result"] == [{"uuid": "u"}]


def test_client_classify_modified_edges_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith("/maintenance/classify-modified-edges")
            body = kwargs.get("json") or {}
            assert body["pot_id"] == "p1"
            assert body["dry_run"] is True
            return httpx.Response(
                200,
                json={"ok": True, "examined": 0, "would_update": 0, "dry_run": True},
            )

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    out = c.classify_modified_edges({"pot_id": "p1", "dry_run": True})
    assert out.get("ok") is True


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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
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


def test_client_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            return httpx.Response(401, json={"detail": "Invalid API key"})

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "bad")
    with pytest.raises(PotpieContextApiError) as ei:
        c.context_graph_query({"pot_id": "p", "query": "q", "limit": 1})
    assert ei.value.status_code == 401


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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    out = c.add_pot_repository("p1", owner="o", repo="r")
    assert out["id"] == "rid"


def test_json_sanitize_datetime() -> None:
    from adapters.outbound.http.potpie_context_api_client import _json_body_for_httpx
    from datetime import datetime, timezone

    dt = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    out = _json_body_for_httpx({"as_of": dt, "n": 1})
    assert out["as_of"] == dt.isoformat()
    assert out["n"] == 1


def test_json_sanitize_datetime_nested_list() -> None:
    from adapters.outbound.http.potpie_context_api_client import _json_body_for_httpx
    from datetime import datetime, timezone

    dt = datetime(2025, 3, 1, tzinfo=timezone.utc)
    out = _json_body_for_httpx({"dates": [dt, dt]})
    assert out["dates"] == [dt.isoformat(), dt.isoformat()]


def test_json_sanitize_datetime_passthrough_non_datetime() -> None:
    from adapters.outbound.http.potpie_context_api_client import _json_body_for_httpx

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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, data = c.ingest({"pot_id": "p", "name": "n", "episode_body": "b", "source_description": "s"}, sync=True)
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, _ = c.ingest({"pot_id": "p", "name": "n", "episode_body": "b", "source_description": "s"}, sync=False)
    assert code == 202
    assert captured["params"] is None


def test_client_ingest_duplicate_409_returns_event_id(monkeypatch: pytest.MonkeyPatch) -> None:
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, data = c.ingest(
        {"pot_id": "p", "name": "n", "episode_body": "b", "source_description": "s"},
        sync=True,
    )
    assert code == 409
    assert data["error"] == "duplicate_ingest"
    assert data["event_id"] == "dup-123"


def test_client_ingest_non_duplicate_409_raises(monkeypatch: pytest.MonkeyPatch) -> None:
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(PotpieContextApiError) as exc_info:
        c.ingest({"pot_id": "p", "name": "n", "episode_body": "b", "source_description": "s"}, sync=True)
    assert exc_info.value.status_code == 409


def test_client_ingest_422_raises_ingest_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
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
                    "episode_uuid": None,
                    "errors": [{"entity": "adr:1", "issue": "unknown canonical labels: X"}],
                    "downgrades": [],
                },
            )

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(IngestRejectedError) as exc_info:
        c.ingest(
            {"pot_id": "p", "name": "n", "episode_body": "b", "source_description": "s"},
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
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
            return httpx.Response(200, json={"ok": True, "status": "accepted", "event_id": "r1"})

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    result = c.record({"pot_id": "p", "record_type": "decision", "summary": "use postgres"})
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, body = c.get_health()
    assert code == 200
    assert body == {"status": "ok"}


def test_client_health_non_200_returns_none_body(monkeypatch: pytest.MonkeyPatch) -> None:
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
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    code, body = c.get_health()
    assert code == 503
    assert body is None


def test_client_context_graph_query_result_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert "/query/context-graph" in url
            return httpx.Response(200, json={"ok": True, "answer": {}})

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    result = c.context_graph_query({"pot_id": "p", "query": "q"})
    assert result["ok"] is True


def test_client_500_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(PotpieContextApiError) as exc_info:
        c.context_graph_query({"pot_id": "p", "query": "q", "limit": 1})
    assert exc_info.value.status_code == 500


def test_client_error_detail_contains_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            return httpx.Response(422, json={"detail": [{"msg": "field required"}]})

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client", FakeClient
    )
    c = PotpieContextApiClient("http://example.com", "k")
    with pytest.raises(PotpieContextApiError) as exc_info:
        c.context_graph_query({"pot_id": "p", "query": "q", "limit": 1})
    assert exc_info.value.status_code == 422
    assert isinstance(exc_info.value.detail, dict)
