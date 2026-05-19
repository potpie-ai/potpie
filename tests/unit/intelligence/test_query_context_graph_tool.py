"""Unit tests for query_context_graph tool (stub).

Covers:
  1. Stub returns available=False, results==[], message mentions fallback tools.
  2. Stub accepts and ignores arbitrary query/project_id/limit values.
  3. ContextGraphResult and QueryContextGraphOutput round-trip through Pydantic.
  4. Tool is registered in ToolService._initialize_tools and retrievable by name.
  5. "query_context_graph" is in DebugAgent's tool list at the FIRST position of
     the discovery section (immediately before ask_knowledge_graph_queries).
"""
from __future__ import annotations

import os

# Set a dummy POSTGRES_SERVER before any app module is imported.
# app/core/database.py calls create_engine(os.getenv("POSTGRES_SERVER")) at
# module level; without a value the import explodes.  The value only needs to
# be a syntactically valid URL — the engine is never actually connected to in
# these unit tests.
os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
# Also set REDIS_URL to avoid similar issues with other eagerly-initialised modules.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from unittest.mock import MagicMock, patch

import pytest

from pydantic import ValidationError

from app.modules.intelligence.tools.query_context_graph_tool import (
    ContextGraphResult,
    QueryContextGraphInput,
    QueryContextGraphOutput,
    query_context_graph,
    query_context_graph_tool,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Stub always returns available=False with empty results and a message
# ---------------------------------------------------------------------------


def test_stub_available_is_false():
    result = query_context_graph(
        query="where is payment timeout handled",
        project_id="proj-123",
    )
    assert result["available"] is False


def test_stub_results_is_empty_list():
    result = query_context_graph(
        query="find all error handlers",
        project_id="proj-abc",
    )
    assert result["results"] == []


def test_stub_message_mentions_ask_knowledge_graph_queries():
    result = query_context_graph(
        query="any query",
        project_id="any-project",
    )
    assert result["message"] is not None
    assert "ask_knowledge_graph_queries" in result["message"]


def test_stub_message_mentions_fallback_theme():
    """Message must reference the fallback concept so the agent can act on it."""
    result = query_context_graph(
        query="any query",
        project_id="any-project",
    )
    assert result["message"] is not None
    # The message should tell the agent to fall back to alternative tools
    msg = result["message"].lower()
    assert "fall back" in msg or "fallback" in msg


# ---------------------------------------------------------------------------
# 2. Stub accepts and ignores arbitrary inputs — must not raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,project_id,limit",
    [
        ("where is payment timeout handled", "proj_123", 10),
        ("", "proj_123", 10),
        ("x", "proj_123", 100_000),
    ],
)
def test_stub_accepts_varied_inputs(query, project_id, limit):
    result = query_context_graph(query=query, project_id=project_id, limit=limit)
    # Stub ignores inputs and always returns the unavailable shape
    assert result["available"] is False
    assert result["results"] == []


def test_input_schema_default_limit_is_ten():
    assert QueryContextGraphInput.model_fields["limit"].default == 10


def test_context_graph_result_score_out_of_bounds_rejected():
    with pytest.raises(ValidationError):
        ContextGraphResult(file="a.py", score=1.5)
    with pytest.raises(ValidationError):
        ContextGraphResult(file="a.py", score=-0.1)


def test_context_graph_result_score_bounds_accepted():
    ContextGraphResult(file="a.py", score=0.0)
    ContextGraphResult(file="a.py", score=1.0)


# ---------------------------------------------------------------------------
# 3. Pydantic round-trip for ContextGraphResult and QueryContextGraphOutput
# ---------------------------------------------------------------------------


def test_context_graph_result_roundtrip_all_fields():
    original = ContextGraphResult(
        file="src/payments/adapter.py",
        symbol="PaymentAdapter.charge",
        snippet="    raise PaymentTimeoutError('timed out')",
        score=0.92,
    )
    dumped = original.model_dump()
    restored = ContextGraphResult.model_validate(dumped)

    assert restored.file == original.file
    assert restored.symbol == original.symbol
    assert restored.snippet == original.snippet
    assert restored.score == original.score


def test_context_graph_result_roundtrip_optional_fields_none():
    original = ContextGraphResult(file="src/app.py", score=0.5)
    dumped = original.model_dump()
    restored = ContextGraphResult.model_validate(dumped)

    assert restored.file == "src/app.py"
    assert restored.symbol is None
    assert restored.snippet is None
    assert restored.score == 0.5


def test_query_context_graph_output_roundtrip_stub_shape():
    original = QueryContextGraphOutput(
        available=False,
        results=[],
        message="context graph service not yet wired; agent should fall back to "
        "ask_knowledge_graph_queries / search_text / file structure tools",
    )
    dumped = original.model_dump()
    restored = QueryContextGraphOutput.model_validate(dumped)

    assert restored.available is False
    assert restored.results == []
    assert "ask_knowledge_graph_queries" in (restored.message or "")


def test_query_context_graph_output_roundtrip_real_service_shape():
    """Simulate what the real service would return when it has hits."""
    hit = ContextGraphResult(
        file="src/checkout/createOrder.ts",
        symbol="CheckoutService.createOrder",
        snippet="  throw new PaymentTimeoutError(err.message);",
        score=0.95,
    )
    original = QueryContextGraphOutput(
        available=True,
        results=[hit],
        message=None,
    )
    dumped = original.model_dump()
    restored = QueryContextGraphOutput.model_validate(dumped)

    assert restored.available is True
    assert len(restored.results) == 1
    assert restored.results[0].file == "src/checkout/createOrder.ts"
    assert restored.message is None


# ---------------------------------------------------------------------------
# 4. Tool is registered in ToolService and is NOT embedding-dependent
#    (tested via AST inspection of tool_service.py source to avoid the
#    heavyweight module-level import chain that pulls in torch/transformers)
# ---------------------------------------------------------------------------


def test_tool_registered_in_tool_service_source():
    """_initialize_tools() in tool_service.py must contain the key 'query_context_graph'.

    We inspect the source via AST rather than importing ToolService, which would
    pull in sentence_transformers → torch, an ML dependency incompatible with the
    Python 3.13 venv used in unit-test runs.
    """
    import ast
    import pathlib

    src_path = (
        pathlib.Path(__file__).parents[3]
        / "app" / "modules" / "intelligence" / "tools" / "tool_service.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    registered_keys: list[str] = []
    for node in ast.walk(tree):
        # Find dict literals that look like {"tool_name": <call>, ...}
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    registered_keys.append(key.value)

    assert "query_context_graph" in registered_keys, (
        f"'query_context_graph' not found as a key in any dict in tool_service.py. "
        f"Keys found: {registered_keys}"
    )


def test_tool_not_in_embedding_dependent_tools_source():
    """EMBEDDING_DEPENDENT_TOOLS in tool_service.py must NOT contain 'query_context_graph'."""
    import ast
    import pathlib

    src_path = (
        pathlib.Path(__file__).parents[3]
        / "app" / "modules" / "intelligence" / "tools" / "tool_service.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find the set literal assigned to EMBEDDING_DEPENDENT_TOOLS
    embedding_tool_names: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "EMBEDDING_DEPENDENT_TOOLS"
                for t in node.targets
            )
        ):
            if isinstance(node.value, ast.Set):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant):
                        embedding_tool_names.append(elt.value)

    assert "query_context_graph" not in embedding_tool_names, (
        "'query_context_graph' must NOT be in EMBEDDING_DEPENDENT_TOOLS"
    )


def test_query_context_graph_tool_import_exists_in_tool_service_source():
    """tool_service.py must import from query_context_graph_tool."""
    import pathlib

    src_path = (
        pathlib.Path(__file__).parents[3]
        / "app" / "modules" / "intelligence" / "tools" / "tool_service.py"
    )
    source = src_path.read_text(encoding="utf-8")
    assert "query_context_graph_tool" in source, (
        "tool_service.py must import from query_context_graph_tool"
    )


# ---------------------------------------------------------------------------
# 5. "query_context_graph" is FIRST in DebugAgent's discovery section
# ---------------------------------------------------------------------------


def test_query_context_graph_is_first_discovery_tool_in_debug_agent():
    """query_context_graph must appear in the DebugAgent tool list AND must be
    immediately before ask_knowledge_graph_queries (first in discovery section).

    Tested via AST inspection to avoid importing the heavy debug_agent module chain.
    """
    import ast
    import pathlib

    src_path = (
        pathlib.Path(__file__).parents[3]
        / "app" / "modules" / "intelligence" / "agents"
        / "chat_agents" / "system_agents" / "debug_agent.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    tool_list: list[str] = []
    for node in ast.walk(tree):
        # Look for a call: *.get_tools([...], ...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_tools"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            for elt in node.args[0].elts:
                if isinstance(elt, ast.Constant):
                    tool_list.append(elt.value)

    assert tool_list, "Could not find get_tools([...]) call in debug_agent.py"
    assert "query_context_graph" in tool_list, (
        "'query_context_graph' not found in DebugAgent tool list"
    )

    qcg_idx = tool_list.index("query_context_graph")
    assert "ask_knowledge_graph_queries" in tool_list, (
        "'ask_knowledge_graph_queries' not found in DebugAgent tool list"
    )
    akg_idx = tool_list.index("ask_knowledge_graph_queries")

    assert qcg_idx < akg_idx, (
        f"'query_context_graph' (index {qcg_idx}) must come before "
        f"'ask_knowledge_graph_queries' (index {akg_idx}) — it is first in the discovery section"
    )
    # Verify it immediately precedes ask_knowledge_graph_queries (no other discovery
    # tool inserted between them by mistake)
    assert qcg_idx + 1 == akg_idx, (
        f"'query_context_graph' (index {qcg_idx}) must be IMMEDIATELY before "
        f"'ask_knowledge_graph_queries' (index {akg_idx})"
    )


def test_discovery_priority_order_first_entry_matches_tool_name():
    """DISCOVERY_PRIORITY_ORDER[0] must be exactly 'query_context_graph'."""
    from app.modules.intelligence.agents.chat_agents.system_agents.debug_hypothesis_contract import (
        DISCOVERY_PRIORITY_ORDER,
    )

    assert DISCOVERY_PRIORITY_ORDER[0] == "query_context_graph", (
        f"Expected DISCOVERY_PRIORITY_ORDER[0] == 'query_context_graph', "
        f"got {DISCOVERY_PRIORITY_ORDER[0]!r}"
    )
