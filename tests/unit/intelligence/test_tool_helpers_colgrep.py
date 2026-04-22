import pytest

from app.modules.intelligence.agents.chat_agents.tool_helpers import (
    get_tool_call_info_content,
    get_tool_response_message,
    get_tool_result_info_content,
    get_tool_run_message,
)


pytestmark = pytest.mark.unit


def test_search_colgrep_run_message_uses_query():
    message = get_tool_run_message(
        "search_colgrep",
        {"query": "auth token validation in backend"},
    )

    assert message == "Searching ColGREP index: auth token validation in backend"


def test_search_colgrep_call_info_includes_scope_and_limits():
    summary = get_tool_call_info_content(
        "search_colgrep",
        {
            "query": "auth token validation",
            "target_paths": ["app/modules/auth", "app/modules/intelligence"],
            "top_k": 7,
            "timeout_ms": 45000,
        },
    )

    assert "searching ColGREP index" in summary
    assert "auth token validation" in summary
    assert "target paths" in summary
    assert "top 7" in summary
    assert "45000ms" in summary


def test_search_colgrep_response_message_reports_ranked_paths():
    result = (
        "(server timing: latency_ms=120, queue_wait_ms=4)\n"
        "Ranked paths (3):\n"
        "  1. app/auth.py  score=0.99\n"
    )

    message = get_tool_response_message(
        "search_colgrep",
        args={"query": "auth token validation"},
        result=result,
    )

    assert message == "ColGREP search completed: auth token validation — 3 ranked path(s)"


def test_search_colgrep_result_summary_returns_renderable_content():
    content = (
        "(search plan: subqueries=2, paths=repo-wide, requests=2, ok=2, latency_ms=240)\n"
        "Ranked paths (2):\n"
        "  1. app/auth.py  score=0.99\n"
    )

    summary = get_tool_result_info_content("search_colgrep", content)

    assert "Ranked paths (2)" in summary
    assert "app/auth.py" in summary


def test_check_colgrep_health_maps_like_other_tools():
    run_message = get_tool_run_message("check_colgrep_health", {})
    response_message = get_tool_response_message(
        "check_colgrep_health",
        result="✅ ColGREP service OK (http://colgrep-api:8080/healthz): {'status': 'ok'}",
    )
    summary = get_tool_result_info_content(
        "check_colgrep_health",
        "✅ ColGREP service OK (http://colgrep-api:8080/healthz): {'status': 'ok'}",
    )

    assert run_message == "Checking ColGREP service health"
    assert response_message == "ColGREP health check passed"
    assert "ColGREP service OK" in summary
