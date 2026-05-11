"""Local Potpie API client for dev parse/chat workflows."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from adapters.outbound.http.potpie_local_api_client import (
    LOCAL_API_PREFIX,
    PotpieLocalApiClient,
    PotpieLocalApiError,
)


def test_parse_directory_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith(f"{LOCAL_API_PREFIX}/parse")
            assert kwargs["json"]["repo_path"] == "/tmp/repo"
            assert kwargs["json"]["branch_name"] == "main"
            return httpx.Response(200, json={"project_id": "p1", "status": "submitted"})

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_local_api_client.httpx.Client",
        FakeClient,
    )
    client = PotpieLocalApiClient("http://localhost:8001")
    out = client.parse_directory(repo_path="/tmp/repo", branch_name="main")
    assert out["project_id"] == "p1"


def test_create_conversation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith(f"{LOCAL_API_PREFIX}/conversations")
            assert kwargs["params"] == {"hidden": "true"}
            assert kwargs["json"]["project_ids"] == ["proj-1"]
            assert kwargs["json"]["agent_ids"] == ["codebase_qna_agent"]
            return httpx.Response(200, json={"conversation_id": "conv-1"})

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_local_api_client.httpx.Client",
        FakeClient,
    )
    client = PotpieLocalApiClient("http://localhost:8001")
    out = client.create_conversation(
        project_id="proj-1", agent_id="codebase_qna_agent"
    )
    assert out["conversation_id"] == "conv-1"


def test_send_message_uses_form_data(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            assert url.endswith(f"{LOCAL_API_PREFIX}/conversations/conv-1/message")
            assert kwargs["params"] == {"stream": "false"}
            assert kwargs["data"] == {"content": "hello"}
            return httpx.Response(200, json={"message": "hi back"})

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_local_api_client.httpx.Client",
        FakeClient,
    )
    client = PotpieLocalApiClient("http://localhost:8001")
    out = client.send_message("conv-1", "hello")
    assert out["message"] == "hi back"


def test_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, *a: Any, **k: Any) -> httpx.Response:
            return httpx.Response(500, json={"detail": "boom"})

        def get(self, *a: Any, **k: Any) -> httpx.Response:
            raise AssertionError("unused")

    monkeypatch.setattr(
        "adapters.outbound.http.potpie_local_api_client.httpx.Client",
        FakeClient,
    )
    client = PotpieLocalApiClient("http://localhost:8001")
    with pytest.raises(PotpieLocalApiError) as excinfo:
        client.parse_directory(repo_path="/tmp/repo")
    assert excinfo.value.status_code == 500
