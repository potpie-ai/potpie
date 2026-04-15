import pytest
from fastapi.testclient import TestClient

from app.modules.intelligence.tools.code_query_tools.colgrep_search_tool import (
    ColgrepHealthTool,
    ColgrepSearchTool,
    ColgrepSearchToolInput,
)
from scripts.colgrep_api_server import app


pytestmark = pytest.mark.unit


class DummyBashTool:
    def __init__(self):
        self.repo_manager = object()
        self.calls = []

    def _run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "success": True,
            "output": "match one\nmatch two",
            "error": "",
            "exit_code": 0,
        }


def test_colgrep_input_requires_query_or_pattern():
    with pytest.raises(ValueError, match="query is required"):
        ColgrepSearchToolInput(project_id="p1")


def test_colgrep_search_tool_api_success(monkeypatch):
    tool = ColgrepSearchTool.__new__(ColgrepSearchTool)
    tool.bash_tool = DummyBashTool()
    tool.api_base_url = "http://colgrep-api:8080"
    tool.default_timeout_ms = 120_000
    tool.allow_local_fallback = False

    captured = {}

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [{"path": "/app/a.py", "score": 1.0}],
                "raw_results": [],
                "latency_ms": 100,
                "queue_wait_ms": 0,
            }

    class DummyClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, endpoint, json):
            captured["endpoint"] = endpoint
            captured["payload"] = json
            return DummyResponse()

    monkeypatch.setattr(
        "app.modules.intelligence.tools.code_query_tools.colgrep_search_tool.httpx.Client",
        DummyClient,
    )

    result = tool.run(project_id="project-123", query="auth token validation", top_k=5)

    assert "Ranked paths" in result
    assert "/app/a.py" in result
    assert captured["endpoint"] == "http://colgrep-api:8080/search"
    assert "project_id" not in captured["payload"]
    assert captured["payload"]["query"] == "auth token validation"
    assert captured["payload"]["top_k"] == 5
    assert captured["payload"]["timeout_ms"] == 120_000
    assert tool.bash_tool.calls == []


def test_colgrep_search_tool_api_includes_target_paths(monkeypatch):
    tool = ColgrepSearchTool.__new__(ColgrepSearchTool)
    tool.bash_tool = DummyBashTool()
    tool.api_base_url = "http://colgrep-api:8080"
    tool.default_timeout_ms = 120_000
    tool.allow_local_fallback = False

    captured = {}

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [],
                "raw_results": [],
                "latency_ms": 1,
                "queue_wait_ms": 0,
            }

    class DummyClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, endpoint, json):
            captured["payload"] = json
            return DummyResponse()

    monkeypatch.setattr(
        "app.modules.intelligence.tools.code_query_tools.colgrep_search_tool.httpx.Client",
        DummyClient,
    )

    tool.run(
        project_id="p1",
        query="drawer logic",
        target_paths=["SampleMgmt", "Modules/Analytical"],
    )

    assert captured["payload"]["target_paths"] == [
        "SampleMgmt",
        "Modules/Analytical",
    ]


def test_colgrep_search_tool_local_fallback_when_enabled(monkeypatch):
    tool = ColgrepSearchTool.__new__(ColgrepSearchTool)
    tool.bash_tool = DummyBashTool()
    tool.api_base_url = "http://colgrep-api:8080"
    tool.default_timeout_ms = 120_000
    tool.allow_local_fallback = True

    def fail_api(_input):
        raise RuntimeError("api down")

    monkeypatch.setattr(tool, "_call_colgrep_api", fail_api)

    result = tool.run(project_id="project-123", query="auth", top_k=5)

    assert result == "match one\nmatch two"
    assert len(tool.bash_tool.calls) == 1
    call = tool.bash_tool.calls[0]
    assert "colgrep search -y --results 5" in call["command"]
    assert "auth" in call["command"]


def test_colgrep_search_tool_api_failure_without_fallback(monkeypatch):
    tool = ColgrepSearchTool.__new__(ColgrepSearchTool)
    tool.bash_tool = DummyBashTool()
    tool.api_base_url = "http://colgrep-api:8080"
    tool.default_timeout_ms = 120_000
    tool.allow_local_fallback = False

    def fail_api(_input):
        raise RuntimeError("api unavailable")

    monkeypatch.setattr(tool, "_call_colgrep_api", fail_api)

    result = tool.run(project_id="project-123", query="auth")

    assert "ColGREP API search failed" in result
    assert "api unavailable" in result


def test_colgrep_health_ok(monkeypatch):
    h = ColgrepHealthTool.__new__(ColgrepHealthTool)
    h.api_base_url = "http://colgrep:8080"

    class DummyResponse:
        def __init__(self, url: str):
            self.url = url
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class DummyClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def get(self, url):
            captured_url["url"] = url
            return DummyResponse(url)

    captured_url = {}

    monkeypatch.setattr(
        "app.modules.intelligence.tools.code_query_tools.colgrep_search_tool.httpx.Client",
        DummyClient,
    )

    out = h.run()
    assert "OK" in out
    assert captured_url["url"] == "http://colgrep:8080/healthz"


def test_colgrep_health_falls_back_to_legacy_health(monkeypatch):
    h = ColgrepHealthTool.__new__(ColgrepHealthTool)
    h.api_base_url = "http://colgrep:8080"

    class DummyResponse:
        def __init__(self, url: str, status_code: int):
            self.url = url
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")
            return None

        def json(self):
            return {"ok": True}

    class DummyClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def get(self, url):
            urls.append(url)
            if url.endswith("/healthz"):
                return DummyResponse(url, 404)
            return DummyResponse(url, 200)

    urls = []

    monkeypatch.setattr(
        "app.modules.intelligence.tools.code_query_tools.colgrep_search_tool.httpx.Client",
        DummyClient,
    )

    out = h.run()
    assert "OK" in out
    assert urls == [
        "http://colgrep:8080/healthz",
        "http://colgrep:8080/health",
    ]


def test_colgrep_api_server_missing_query_returns_contract_error():
    client = TestClient(app)

    response = client.post("/search", json={"top_k": 5, "timeout_ms": 120000})

    assert response.status_code == 400
    assert response.json() == {"error": "query is required"}


def test_colgrep_api_server_invalid_target_paths_returns_contract_error(monkeypatch):
    monkeypatch.setenv("COLGREP_SERVER_DEFAULT_PATH", "/tmp")
    client = TestClient(app)

    response = client.post(
        "/search",
        json={
            "query": "drawer logic",
            "top_k": 5,
            "timeout_ms": 120000,
            "target_paths": ["../../etc", "does/not/exist"],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid target_paths",
        "invalid_target_paths": ["../../etc", "does/not/exist"],
    }


def test_colgrep_api_server_empty_query_returns_contract_error():
    client = TestClient(app)

    response = client.post("/search", json={"query": ""})
    assert response.status_code == 400
    assert response.json() == {"error": "query is required"}


def test_colgrep_api_server_whitespace_query_returns_contract_error():
    client = TestClient(app)

    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 400
    assert response.json() == {"error": "query is required"}


def test_colgrep_api_server_empty_body_returns_contract_error():
    client = TestClient(app)

    response = client.post("/search", json={})
    assert response.status_code == 400
    assert response.json() == {"error": "query is required"}


def test_colgrep_api_server_invalid_timeout_ms():
    client = TestClient(app)

    response = client.post("/search", json={"query": "test", "timeout_ms": 500})
    assert response.status_code == 400
    assert response.json() == {"error": "invalid request"}


def test_colgrep_api_server_invalid_top_k():
    client = TestClient(app)

    response = client.post("/search", json={"query": "test", "top_k": 0})
    assert response.status_code == 400
    assert response.json() == {"error": "invalid request"}


def test_colgrep_api_server_healthz():
    client = TestClient(app)

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_colgrep_input_rejects_whitespace_only_query():
    with pytest.raises(ValueError, match="query is required"):
        ColgrepSearchToolInput(project_id="p1", query="   ")


def test_colgrep_input_rejects_empty_string_query():
    with pytest.raises(ValueError, match="query is required"):
        ColgrepSearchToolInput(project_id="p1", query="")


def test_colgrep_resolved_target_paths_filters_blanks():
    inp = ColgrepSearchToolInput(
        project_id="p1", query="test", target_paths=["", " ", "real/path"]
    )
    resolved = ColgrepSearchTool._resolved_target_paths(inp)
    assert resolved == ["real/path"]


def test_colgrep_resolved_target_paths_empty_list_returns_none():
    inp = ColgrepSearchToolInput(project_id="p1", query="test", target_paths=[])
    resolved = ColgrepSearchTool._resolved_target_paths(inp)
    assert resolved is None


def test_colgrep_working_directory_ignored_when_target_paths_set():
    inp = ColgrepSearchToolInput(
        project_id="p1",
        query="test",
        target_paths=["a/b"],
        working_directory="c/d",
    )
    resolved = ColgrepSearchTool._resolved_target_paths(inp)
    assert resolved == ["a/b"]


def test_colgrep_working_directory_used_as_target_path_when_no_target_paths():
    inp = ColgrepSearchToolInput(
        project_id="p1", query="test", working_directory="src/lib"
    )
    resolved = ColgrepSearchTool._resolved_target_paths(inp)
    assert resolved == ["src/lib"]


def test_colgrep_format_api_success_empty_results():
    body = {
        "results": [],
        "raw_results": [],
        "latency_ms": 50,
        "queue_wait_ms": 0,
    }
    out = ColgrepSearchTool._format_api_success(body)
    assert "no results" in out


def test_colgrep_format_api_success_timeout_error_in_body():
    body = {
        "error": "colgrep search timed out",
        "latency_ms": 120000,
        "results": [],
        "raw_results": [],
    }
    out = ColgrepSearchTool._format_api_success(body)
    assert "timed out" in out
    assert "latency_ms=120000" in out


def test_colgrep_format_api_success_none_score():
    body = {
        "results": [{"path": "/app/bar.py", "score": None}],
        "raw_results": [],
        "latency_ms": 10,
        "queue_wait_ms": 0,
    }
    out = ColgrepSearchTool._format_api_success(body)
    assert "/app/bar.py" in out
    assert "score=None" in out


def test_colgrep_health_no_base_url():
    h = ColgrepHealthTool.__new__(ColgrepHealthTool)
    h.api_base_url = ""
    out = h.run()
    assert "not set" in out


def test_colgrep_health_handles_extra_kwargs():
    h = ColgrepHealthTool.__new__(ColgrepHealthTool)
    h.api_base_url = ""
    out = h.run(project_id="p1", conversation_id="c1")
    assert "not set" in out
