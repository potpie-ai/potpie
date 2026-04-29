"""
Tests for Langfuse trace access tools used by the Prompt Tuner agent.
"""

import pytest

from app.modules.intelligence.tools.prompt_tuner.langfuse_tools import (
    FetchLangfuseTraceTool,
    ListLangfuseTracesTool,
    fetch_langfuse_trace_tool,
    list_langfuse_traces_tool,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                "error",
                request=None,
                response=self,
            )


class FakeLangfuseClient:
    requests = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params=None, auth=None):
        self.requests.append({"url": url, "params": params or {}, "auth": auth})

        if url.endswith("/api/public/traces/trace-123"):
            return FakeResponse(
                {
                    "id": "trace-123",
                    "name": "prompt_tuner_agent",
                    "timestamp": "2026-04-29T10:00:00Z",
                    "tags": ["prompt_tuner_agent"],
                    "userId": "user-1",
                    "sessionId": "conversation-1",
                    "metadata": {"run_id": "run-1"},
                    "input": "Why did the agent over-search?",
                    "output": "It used broad search terms.",
                }
            )

        if url.startswith("http://langfuse.test/api/public/observations"):
            return FakeResponse(
                {
                    "data": [
                        {
                            "name": "model-call",
                            "type": "GENERATION",
                            "model": "openrouter/test",
                            "usage": {"input": 12, "output": 8, "total": 20},
                            "latency": 123,
                            "input": "messages",
                            "output": "answer",
                            "startTime": "2026-04-29T10:00:01Z",
                        },
                        {
                            "name": "fetch_langfuse_trace",
                            "type": "SPAN",
                            "latency": 5,
                            "input": {"trace_id": "trace-123"},
                            "output": "trace body",
                            "startTime": "2026-04-29T10:00:02Z",
                        },
                    ]
                }
            )

        if url.endswith("/api/public/traces"):
            return FakeResponse(
                {
                    "data": [
                        {
                            "id": "trace-123",
                            "name": "prompt_tuner_agent",
                            "timestamp": "2026-04-29T10:00:00Z",
                            "tags": ["prompt_tuner_agent"],
                            "userId": "user-1",
                            "usage": {"input": 12, "output": 8},
                        }
                    ]
                }
            )

        return FakeResponse({}, status_code=404, text="not found")


@pytest.fixture(autouse=True)
def langfuse_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.delenv("LANGFUSE_API_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)


@pytest.fixture
def fake_httpx_client(monkeypatch):
    FakeLangfuseClient.requests = []
    monkeypatch.setattr(
        "app.modules.intelligence.tools.prompt_tuner.langfuse_tools.httpx.Client",
        FakeLangfuseClient,
    )
    return FakeLangfuseClient


def test_fetch_langfuse_trace_uses_langfuse_host_and_formats_observations(
    fake_httpx_client,
):
    tool = FetchLangfuseTraceTool(None, "user-1")

    result = tool.run(trace_id="trace-123")

    assert "## Trace: trace-123" in result
    assert "### Trace Input" in result
    assert "Why did the agent over-search?" in result
    assert "### LLM Generations (1)" in result
    assert "model-call" in result
    assert "Tokens: input=12, output=8, total=20" in result
    assert "### Tool Calls / Spans (1)" in result
    assert "fetch_langfuse_trace" in result

    assert fake_httpx_client.requests[0]["url"] == (
        "http://langfuse.test/api/public/traces/trace-123"
    )
    assert fake_httpx_client.requests[1]["url"] == (
        "http://langfuse.test/api/public/observations?traceId=trace-123&limit=100"
    )


def test_list_langfuse_traces_passes_filters_to_langfuse(fake_httpx_client):
    tool = ListLangfuseTracesTool(None, "user-1")

    result = tool.run(
        limit=5,
        user_id="user-1",
        tags=["prompt_tuner_agent"],
        name="prompt_tuner_agent",
    )

    assert "## Recent Traces (1)" in result
    assert "Trace ID: `trace-123`" in result
    assert "prompt_tuner_agent" in result

    request = fake_httpx_client.requests[0]
    assert request["url"] == "http://langfuse.test/api/public/traces"
    assert request["params"] == {
        "limit": 5,
        "userId": "user-1",
        "name": "prompt_tuner_agent",
        "tags": ["prompt_tuner_agent"],
    }


def test_langfuse_tools_report_missing_host(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    fetch_result = FetchLangfuseTraceTool(None, "user-1").run(trace_id="trace-123")
    list_result = ListLangfuseTracesTool(None, "user-1").run()

    assert fetch_result == "LANGFUSE_HOST is not configured."
    assert list_result == "LANGFUSE_HOST is not configured."


def test_prompt_tuner_langfuse_structured_tools_are_registered():
    fetch_tool = fetch_langfuse_trace_tool(None, "user-1")
    list_tool = list_langfuse_traces_tool(None, "user-1")

    assert fetch_tool.name == "fetch_langfuse_trace"
    assert list_tool.name == "list_langfuse_traces"
    assert fetch_tool.args_schema is not None
    assert list_tool.args_schema is not None
