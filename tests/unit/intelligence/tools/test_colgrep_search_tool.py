import asyncio

import pytest
from fastapi.testclient import TestClient

from app.modules.intelligence.tools.code_query_tools.colgrep_search_tool import (
    ColgrepHealthTool,
    ColgrepSearchTool,
    ColgrepSearchToolInput,
    MergedSearchOutput,
    SearchTask,
    SearchTaskResult,
    _decompose_query,
    _extract_keywords,
    _is_broad_query,
)
from scripts.colgrep_api_server import app


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_tool(monkeypatch, *, responses=None, capture=None):
    """Build a ColgrepSearchTool with a monkeypatched httpx.Client."""
    tool = ColgrepSearchTool.__new__(ColgrepSearchTool)
    tool.bash_tool = DummyBashTool()
    tool.api_base_url = "http://colgrep-api:8080"
    tool.default_timeout_ms = 120_000
    tool.allow_local_fallback = False

    call_index = {"n": 0}
    if capture is None:
        capture = {"payloads": []}
    if responses is None:
        responses = [
            {
                "results": [{"path": "/app/a.py", "score": 1.0}],
                "raw_results": [],
                "latency_ms": 100,
                "queue_wait_ms": 0,
            }
        ]

    class DummyResponse:
        status_code = 200

        def __init__(self, body):
            self._body = body

        def json(self):
            return self._body

        @property
        def text(self):
            return str(self._body)

    class DummyClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, endpoint, json):
            idx = min(call_index["n"], len(responses) - 1)
            call_index["n"] += 1
            capture["payloads"].append(json)
            return DummyResponse(responses[idx])

    monkeypatch.setattr(
        "app.modules.intelligence.tools.code_query_tools.colgrep_search_tool.httpx.Client",
        DummyClient,
    )
    return tool, capture


# ---------------------------------------------------------------------------
# Broad-query detection
# ---------------------------------------------------------------------------

class TestBroadQueryDetection:
    def test_short_keyword_query_is_not_broad(self):
        assert not _is_broad_query("auth token validation")

    def test_long_query_is_broad(self):
        assert _is_broad_query(
            "How reagent compartment thermal handling is done in CH systems"
        )

    def test_intent_phrase_makes_broad(self):
        assert _is_broad_query("how temperature is handled")

    def test_look_for_makes_broad(self):
        assert _is_broad_query("look for drawer open close logic")

    def test_multi_concept_makes_broad(self):
        assert _is_broad_query("temperature control, reagent cooling, tray monitoring")

    def test_single_short_query_not_broad(self):
        assert not _is_broad_query("PID loop")

    def test_workflow_keyword_makes_broad(self):
        assert _is_broad_query("workflow for sample processing")

    def test_configuration_keyword_makes_broad(self):
        assert _is_broad_query("configuration of the thermal subsystem")


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------

class TestQueryDecomposition:
    def test_narrow_query_returns_itself(self):
        result = _decompose_query("PID loop controller")
        assert len(result) == 1
        assert "PID" in result[0]

    def test_comma_separated_concepts_split(self):
        result = _decompose_query(
            "temperature control, reagent cooling, tray monitoring"
        )
        assert len(result) >= 2
        assert len(result) <= 4

    def test_long_query_chunked(self):
        result = _decompose_query(
            "How reagent compartment thermal handling is done in CH systems"
        )
        assert len(result) >= 2
        assert len(result) <= 4
        for sq in result:
            assert len(sq.split()) <= 8

    def test_max_four_subqueries(self):
        result = _decompose_query(
            "a concept, b concept, c concept, d concept, e concept, f concept"
        )
        assert len(result) <= 4


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

class TestKeywordExtraction:
    def test_removes_stopwords(self):
        result = _extract_keywords("how is the temperature handled in the system")
        assert "how" not in result.lower()
        assert "the" not in result.lower()
        assert "temperature" in result.lower()

    def test_preserves_technical_terms(self):
        result = _extract_keywords("PID thermistor setpoint controller")
        assert "PID" in result
        assert "thermistor" in result


# ---------------------------------------------------------------------------
# Search plan construction
# ---------------------------------------------------------------------------

class TestSearchPlan:
    def test_narrow_query_single_path_one_task(self):
        tasks = ColgrepSearchTool._build_search_plan(
            "PID loop", None, 5, 120_000
        )
        assert len(tasks) == 1
        assert tasks[0].query == "PID loop"
        assert tasks[0].target_path is None

    def test_narrow_query_multi_path_fans_out(self):
        tasks = ColgrepSearchTool._build_search_plan(
            "PID loop", ["CH/MMCC", "Modules/Analytical"], 5, 120_000
        )
        assert len(tasks) == 2
        paths = {t.target_path for t in tasks}
        assert paths == {"CH/MMCC", "Modules/Analytical"}

    def test_broad_query_decomposes(self):
        tasks = ColgrepSearchTool._build_search_plan(
            "How reagent compartment thermal handling is done in CH systems",
            None,
            5,
            120_000,
        )
        assert len(tasks) >= 2
        queries = {t.query for t in tasks}
        assert len(queries) >= 2

    def test_broad_query_multi_path_cartesian(self):
        tasks = ColgrepSearchTool._build_search_plan(
            "temperature control, reagent cooling",
            ["CH/MMCC", "Modules/Analytical"],
            5,
            120_000,
        )
        queries = {t.query for t in tasks}
        paths = {t.target_path for t in tasks}
        assert len(queries) >= 2
        assert paths == {"CH/MMCC", "Modules/Analytical"}
        assert len(tasks) == len(queries) * len(paths)


# ---------------------------------------------------------------------------
# Result merging
# ---------------------------------------------------------------------------

class TestResultMerging:
    def test_dedup_by_path_keeps_best_score(self):
        results = ColgrepSearchTool._merge_search_results(
            [
                SearchTaskResult(
                    task=SearchTask("q1", None, 5, 120000),
                    results=[
                        {"path": "/a.py", "score": 3.0},
                        {"path": "/b.py", "score": 2.0},
                    ],
                    latency_ms=100,
                ),
                SearchTaskResult(
                    task=SearchTask("q2", None, 5, 120000),
                    results=[
                        {"path": "/a.py", "score": 5.0},
                        {"path": "/c.py", "score": 1.0},
                    ],
                    latency_ms=200,
                ),
            ],
            10,
        )
        paths = {r["path"]: r for r in results.results}
        assert paths["/a.py"]["score"] == 5.0
        assert paths["/a.py"]["hit_count"] == 2
        assert "/b.py" in paths
        assert "/c.py" in paths

    def test_multi_query_hits_ranked_higher(self):
        results = ColgrepSearchTool._merge_search_results(
            [
                SearchTaskResult(
                    task=SearchTask("q1", None, 5, 120000),
                    results=[
                        {"path": "/common.py", "score": 2.0},
                        {"path": "/rare.py", "score": 10.0},
                    ],
                    latency_ms=100,
                ),
                SearchTaskResult(
                    task=SearchTask("q2", None, 5, 120000),
                    results=[
                        {"path": "/common.py", "score": 3.0},
                    ],
                    latency_ms=100,
                ),
            ],
            10,
        )
        assert results.results[0]["path"] == "/common.py"

    def test_partial_success_metadata(self):
        results = ColgrepSearchTool._merge_search_results(
            [
                SearchTaskResult(
                    task=SearchTask("q1", "path1", 5, 120000),
                    results=[{"path": "/ok.py", "score": 1.0}],
                    latency_ms=100,
                ),
                SearchTaskResult(
                    task=SearchTask("q2", "path2", 5, 120000),
                    error="colgrep search timed out",
                    timed_out=True,
                    latency_ms=120000,
                ),
            ],
            10,
        )
        assert results.succeeded == 1
        assert results.timed_out == 1
        assert results.is_partial is True
        assert len(results.results) == 1

    def test_all_failed_zero_succeeded(self):
        results = ColgrepSearchTool._merge_search_results(
            [
                SearchTaskResult(
                    task=SearchTask("q1", None, 5, 120000),
                    error="timeout",
                    timed_out=True,
                    latency_ms=120000,
                ),
            ],
            10,
        )
        assert results.succeeded == 0
        assert len(results.results) == 0

    def test_top_k_limits_output(self):
        task_results = [
            SearchTaskResult(
                task=SearchTask("q1", None, 5, 120000),
                results=[{"path": f"/f{i}.py", "score": float(i)} for i in range(20)],
                latency_ms=100,
            ),
        ]
        merged = ColgrepSearchTool._merge_search_results(task_results, 3)
        assert len(merged.results) == 3


# ---------------------------------------------------------------------------
# Narrow fast-path (existing behavior preserved)
# ---------------------------------------------------------------------------

def test_colgrep_input_requires_query_or_pattern():
    with pytest.raises(ValueError, match="query is required"):
        ColgrepSearchToolInput(project_id="p1")


def test_colgrep_search_tool_api_success(monkeypatch):
    tool, captured = _make_tool(monkeypatch)
    result = tool.run(project_id="project-123", query="auth token validation", top_k=5)

    assert "Ranked paths" in result
    assert "/app/a.py" in result
    assert len(captured["payloads"]) == 1
    p = captured["payloads"][0]
    assert p["query"] == "auth token validation"
    assert p["top_k"] == 5
    assert p["timeout_ms"] == 120_000
    assert "project_id" not in p


def test_colgrep_search_tool_api_includes_target_paths_fanout(monkeypatch):
    """Multiple target_paths now fan out into separate requests."""
    tool, captured = _make_tool(
        monkeypatch,
        responses=[
            {
                "results": [{"path": "/app/a.py", "score": 2.0}],
                "raw_results": [],
                "latency_ms": 50,
                "queue_wait_ms": 0,
            },
            {
                "results": [{"path": "/app/b.py", "score": 1.0}],
                "raw_results": [],
                "latency_ms": 60,
                "queue_wait_ms": 0,
            },
        ],
    )

    # Monkeypatch the async path too since multi-path triggers decomposition
    async def fake_execute(tasks):
        results = []
        for t in tasks:
            results.append(SearchTaskResult(
                task=t,
                results=[
                    {"path": f"/app/{t.target_path}.py", "score": 1.0}
                ],
                latency_ms=50,
            ))
        return results

    monkeypatch.setattr(tool, "_execute_search_plan", fake_execute)

    result = tool.run(
        project_id="p1",
        query="drawer logic",
        target_paths=["SampleMgmt", "Modules/Analytical"],
    )

    assert "SampleMgmt" in result or "Modules/Analytical" in result
    assert "search plan" in result


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


# ---------------------------------------------------------------------------
# Broad query end-to-end
# ---------------------------------------------------------------------------

def test_broad_query_triggers_decomposition(monkeypatch):
    tool = ColgrepSearchTool.__new__(ColgrepSearchTool)
    tool.bash_tool = DummyBashTool()
    tool.api_base_url = "http://colgrep-api:8080"
    tool.default_timeout_ms = 120_000
    tool.allow_local_fallback = False

    executed_tasks = []

    async def fake_execute(tasks):
        executed_tasks.extend(tasks)
        results = []
        for t in tasks:
            results.append(SearchTaskResult(
                task=t,
                results=[{"path": f"/result_{t.query[:8]}.py", "score": 1.0}],
                latency_ms=50,
            ))
        return results

    monkeypatch.setattr(tool, "_execute_search_plan", fake_execute)

    result = tool.run(
        project_id="p1",
        query="How reagent compartment thermal handling is done in CH systems",
    )

    assert len(executed_tasks) >= 2
    assert "search plan" in result
    assert "subqueries=" in result


def test_partial_success_returns_results(monkeypatch):
    tool = ColgrepSearchTool.__new__(ColgrepSearchTool)
    tool.bash_tool = DummyBashTool()
    tool.api_base_url = "http://colgrep-api:8080"
    tool.default_timeout_ms = 120_000
    tool.allow_local_fallback = False

    async def fake_execute(tasks):
        results = []
        for i, t in enumerate(tasks):
            if i == 0:
                results.append(SearchTaskResult(
                    task=t,
                    results=[{"path": "/good.py", "score": 5.0}],
                    latency_ms=50,
                ))
            else:
                results.append(SearchTaskResult(
                    task=t,
                    error="colgrep search timed out",
                    timed_out=True,
                    latency_ms=120000,
                ))
        return results

    monkeypatch.setattr(tool, "_execute_search_plan", fake_execute)

    result = tool.run(
        project_id="p1",
        query="How thermal handling and cooling works",
        target_paths=["CH/MMCC"],
    )

    assert "Partial results" in result or "/good.py" in result
    assert "timed_out=" in result


def test_all_requests_fail_returns_failure(monkeypatch):
    tool = ColgrepSearchTool.__new__(ColgrepSearchTool)
    tool.bash_tool = DummyBashTool()
    tool.api_base_url = "http://colgrep-api:8080"
    tool.default_timeout_ms = 120_000
    tool.allow_local_fallback = False

    async def fake_execute(tasks):
        return [
            SearchTaskResult(
                task=t,
                error="colgrep search timed out",
                timed_out=True,
                latency_ms=120000,
            )
            for t in tasks
        ]

    monkeypatch.setattr(tool, "_execute_search_plan", fake_execute)

    result = tool.run(
        project_id="p1",
        query="How reagent compartment thermal handling is done in CH systems",
    )

    assert "All" in result and "failed" in result


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

def test_format_merged_output_shows_metadata():
    merged = MergedSearchOutput(
        results=[{"path": "/a.py", "score": 5.0, "hit_count": 2}],
        raw_results=[],
        subqueries=["reagent temperature", "cooling chiller"],
        target_paths_searched=["CH/MMCC"],
        total_tasks=4,
        succeeded=3,
        failed=0,
        timed_out=1,
        is_partial=True,
        total_latency_ms=5000,
    )
    out = ColgrepSearchTool._format_merged_output(merged)
    assert "subqueries=2" in out
    assert "ok=3" in out
    assert "timed_out=1" in out
    assert "Partial results" in out
    assert "/a.py" in out
    assert "appeared in 2 subqueries" in out


# ---------------------------------------------------------------------------
# Health check (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Server contract tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Input edge cases
# ---------------------------------------------------------------------------

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
