"""Potpie /api/v2/context HTTP client."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from adapters.outbound.http.potpie_context_api_client import (
    CONTEXT_API_PREFIX,
    PotpieContextApiClient,
    PotpieContextApiError,
)


def test_client_search_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert CONTEXT_API_PREFIX in url
            assert url.endswith("/query/search")
            assert kwargs["headers"].get("X-API-Key") == "k"
            body = kwargs.get("json") or {}
            assert body["pot_id"] == "p1"
            return httpx.Response(200, json=[{"uuid": "u", "name": "n"}])

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    rows = c.search({"pot_id": "p1", "query": "q", "limit": 8})
    assert rows == [{"uuid": "u", "name": "n"}]


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
        c.search({"pot_id": "p", "query": "q", "limit": 1})
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
            assert kwargs["json"] == {"display_name": "n"}
            return httpx.Response(200, json={"id": "new-id", "display_name": "n"})

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_context_api_client.httpx.Client",
        FakeClient,
    )
    c = PotpieContextApiClient("http://example.com", "k")
    out = c.create_context_pot(display_name="n")
    assert out["id"] == "new-id"


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
