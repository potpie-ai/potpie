"""Unit tests for hypothesis state tools (A5).

Covers:
  1. record_hypothesis returns a record with a non-empty id, status=PROPOSED by default,
     and both created_at / updated_at set.
  2. Two record_hypothesis calls produce DIFFERENT ids.
  3. update_hypothesis_status updates the status and bumps updated_at (strictly greater).
  4. update_hypothesis_status for an unknown id returns an error ({"error": ...}) not a
     silent no-op — consistent with how todos returns a plain string for unknown ids.
  5. append_hypothesis_evidence appends without losing prior entries.
  6. list_hypotheses returns ALL created records in id order.
  7. Cross-execution-context isolation: two separate ContextVar contexts do NOT see each
     other's hypotheses (mirrors the parallel-agent-run isolation of TodoStorage).
  8. Two sequential update_hypothesis_status calls both apply (last-write-wins, no corruption).
  9. Status round-trips through Pydantic serialisation (enum value persists and reads
     back as the same enum member).
 10. Tool registration: all four tool names appear as keys in tool_service.py's
     _initialize_tools dict (AST inspection — avoids importing the heavy ML chain).
 11. All four tool names appear in DebugAgent's get_tools([...]) call (AST inspection).
"""

from __future__ import annotations

import os

# Set mandatory env vars before any app module is imported (same pattern as A8 tests).
os.environ.setdefault("POSTGRES_SERVER", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import contextvars
from datetime import datetime, timezone

import pytest

from app.modules.intelligence.agents.chat_agents.system_agents.debug_hypothesis_contract import (
    HypothesisStatus,
)
from app.modules.intelligence.tools.hypothesis_state_tool import (
    HypothesisRecord,
    HypothesisStore,
    RecordHypothesisInput,
    UpdateHypothesisStatusInput,
    AppendHypothesisEvidenceInput,
    _hypothesis_store_ctx,
    get_hypothesis_store,
    record_hypothesis,
    update_hypothesis_status,
    append_hypothesis_evidence,
    list_hypotheses,
    create_hypothesis_state_tools,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_store() -> HypothesisStore:
    """Create a brand-new HypothesisStore, install it in the current ContextVar,
    and return it.  Used to guarantee a clean state before each test."""
    store = HypothesisStore()
    _hypothesis_store_ctx.set(store)
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_store():
    """Each test gets its own fresh HypothesisStore in the current context."""
    _fresh_store()
    yield
    # Clean up — reset to None so the next test's autouse fixture can re-create
    _hypothesis_store_ctx.set(None)


# ---------------------------------------------------------------------------
# 1. record_hypothesis — basic properties
# ---------------------------------------------------------------------------


def test_record_hypothesis_returns_nonempty_id():
    result = record_hypothesis(title="Payment timeout not mapped to 402")
    assert result["id"] != ""
    assert result["id"].startswith("hyp_")


def test_record_hypothesis_default_status_is_proposed():
    result = record_hypothesis(title="Some hypothesis")
    assert result["status"] == HypothesisStatus.PROPOSED.value


def test_record_hypothesis_timestamps_are_set():
    result = record_hypothesis(title="Another hypothesis")
    assert result["created_at"] is not None
    assert result["updated_at"] is not None


def test_record_hypothesis_created_and_updated_at_equal_on_creation():
    result = record_hypothesis(title="Timestamps equal at creation")
    assert result["created_at"] == result["updated_at"]


def test_record_hypothesis_with_explicit_status():
    result = record_hypothesis(
        title="Debugging in progress",
        status=HypothesisStatus.DEBUGGING,
    )
    assert result["status"] == HypothesisStatus.DEBUGGING.value


def test_record_hypothesis_with_evidence_and_plan():
    result = record_hypothesis(
        title="With evidence",
        evidence=["Stack trace shows timeout at chargeCard"],
        validation_plan=["Set breakpoint at chargeCard"],
    )
    assert result["evidence"] == ["Stack trace shows timeout at chargeCard"]
    assert result["validation_plan"] == ["Set breakpoint at chargeCard"]


# ---------------------------------------------------------------------------
# 2. Two consecutive calls produce different ids
# ---------------------------------------------------------------------------


def test_two_record_calls_produce_different_ids():
    r1 = record_hypothesis(title="Hypothesis A")
    r2 = record_hypothesis(title="Hypothesis B")
    assert r1["id"] != r2["id"]


def test_ids_are_sequential():
    r1 = record_hypothesis(title="First")
    r2 = record_hypothesis(title="Second")
    r3 = record_hypothesis(title="Third")
    assert r1["id"] == "hyp_1"
    assert r2["id"] == "hyp_2"
    assert r3["id"] == "hyp_3"


# ---------------------------------------------------------------------------
# 3. update_hypothesis_status — updates status and bumps updated_at
# ---------------------------------------------------------------------------


def test_update_status_changes_status():
    r = record_hypothesis(title="Will be updated")
    result = update_hypothesis_status(
        hypothesis_id=r["id"],
        status=HypothesisStatus.DEBUGGING,
    )
    assert result["status"] == HypothesisStatus.DEBUGGING.value


def test_update_status_bumps_updated_at():
    import time

    r = record_hypothesis(title="Timestamp bump test")
    created_updated = r["updated_at"]

    # Small sleep to ensure monotonically later timestamp
    time.sleep(0.01)

    result = update_hypothesis_status(
        hypothesis_id=r["id"],
        status=HypothesisStatus.SUPPORTED,
    )
    # updated_at after the update must be strictly later
    assert result["updated_at"] > created_updated


def test_update_status_does_not_change_created_at():
    r = record_hypothesis(title="Created at must stay")
    result = update_hypothesis_status(
        hypothesis_id=r["id"],
        status=HypothesisStatus.REJECTED,
    )
    assert result["created_at"] == r["created_at"]


# ---------------------------------------------------------------------------
# 4. update_hypothesis_status — unknown id returns error
# ---------------------------------------------------------------------------


def test_update_status_unknown_id_returns_error():
    result = update_hypothesis_status(
        hypothesis_id="hyp_999",
        status=HypothesisStatus.REJECTED,
    )
    assert "error" in result
    assert "hyp_999" in result["error"]


def test_update_status_unknown_id_is_not_silent():
    """Calling update_hypothesis_status with an unknown id MUST NOT silently succeed."""
    result = update_hypothesis_status(
        hypothesis_id="nonexistent",
        status=HypothesisStatus.VALIDATED,
    )
    # Error is surfaced, not swallowed
    assert "error" in result
    assert result.get("status") is None  # no status field on an error response


# ---------------------------------------------------------------------------
# 5. append_hypothesis_evidence — appends without losing prior entries
# ---------------------------------------------------------------------------


def test_append_evidence_grows_list():
    r = record_hypothesis(
        title="Evidence accumulator",
        evidence=["Initial observation"],
    )
    update_result = append_hypothesis_evidence(
        hypothesis_id=r["id"],
        evidence="Second observation",
    )
    assert len(update_result["evidence"]) == 2
    assert update_result["evidence"][0] == "Initial observation"
    assert update_result["evidence"][1] == "Second observation"


def test_append_evidence_to_empty_list():
    r = record_hypothesis(title="No initial evidence")
    result = append_hypothesis_evidence(
        hypothesis_id=r["id"],
        evidence="First observation",
    )
    assert result["evidence"] == ["First observation"]


def test_append_evidence_multiple_times():
    r = record_hypothesis(title="Multiple appends")
    for i in range(5):
        result = append_hypothesis_evidence(
            hypothesis_id=r["id"],
            evidence=f"Observation {i}",
        )
    assert len(result["evidence"]) == 5


def test_append_evidence_unknown_id_returns_error():
    result = append_hypothesis_evidence(
        hypothesis_id="hyp_404",
        evidence="This should fail",
    )
    assert "error" in result
    assert "hyp_404" in result["error"]


def test_append_evidence_bumps_updated_at():
    import time

    r = record_hypothesis(title="Updated at check")
    original_updated = r["updated_at"]
    time.sleep(0.01)
    result = append_hypothesis_evidence(
        hypothesis_id=r["id"],
        evidence="New observation",
    )
    assert result["updated_at"] > original_updated


# ---------------------------------------------------------------------------
# 6. list_hypotheses — returns all created records in id order
# ---------------------------------------------------------------------------


def test_list_hypotheses_returns_all_records():
    record_hypothesis(title="Alpha")
    record_hypothesis(title="Beta")
    record_hypothesis(title="Gamma")
    result = list_hypotheses()
    assert len(result["hypotheses"]) == 3


def test_list_hypotheses_empty_when_none_recorded():
    result = list_hypotheses()
    assert result["hypotheses"] == []


def test_list_hypotheses_in_creation_order():
    r1 = record_hypothesis(title="First")
    r2 = record_hypothesis(title="Second")
    result = list_hypotheses()
    ids = [h["id"] for h in result["hypotheses"]]
    assert ids == [r1["id"], r2["id"]]


# ---------------------------------------------------------------------------
# 7. Cross-execution-context isolation
# ---------------------------------------------------------------------------


def test_cross_context_isolation():
    """Two separate ContextVar execution contexts must NOT see each other's hypotheses.

    This mirrors how separate async agent runs (different ContextVar contexts)
    maintain isolated TodoStorage instances in the todos family.
    """
    # Current context already has a fresh store (via autouse fixture).
    record_hypothesis(title="Hypothesis in context A")
    record_hypothesis(title="Another one in context A")
    assert len(list_hypotheses()["hypotheses"]) == 2

    # Run a callable in a brand-new ContextVar context (simulates a parallel agent run)
    captured: list = []

    def run_in_context_b() -> None:
        # Install a fresh store for this context
        _hypothesis_store_ctx.set(HypothesisStore())
        # Record a different hypothesis
        record_hypothesis(title="Hypothesis in context B")
        # This context should see only 1 hypothesis
        captured.append(list_hypotheses()["hypotheses"])

    # contextvars.copy_context().run() creates an isolated copy of the context,
    # which starts with the current ContextVar values but writes back to the copy —
    # the original context is unaffected.
    ctx_b = contextvars.copy_context()
    ctx_b.run(run_in_context_b)

    # Context B saw exactly 1 hypothesis (its own)
    assert len(captured[0]) == 1
    assert captured[0][0]["title"] == "Hypothesis in context B"

    # Context A still sees 2 hypotheses (unaffected by context B)
    assert len(list_hypotheses()["hypotheses"]) == 2


# ---------------------------------------------------------------------------
# 8. Sequential updates — last-write-wins, no corruption
# ---------------------------------------------------------------------------


def test_two_sequential_status_updates_both_apply():
    r = record_hypothesis(title="Sequential updates")
    update_hypothesis_status(
        hypothesis_id=r["id"],
        status=HypothesisStatus.DEBUGGING,
    )
    second = update_hypothesis_status(
        hypothesis_id=r["id"],
        status=HypothesisStatus.SUPPORTED,
    )
    assert second["status"] == HypothesisStatus.SUPPORTED.value


def test_sequential_updates_do_not_corrupt_other_hypotheses():
    r1 = record_hypothesis(title="Target of updates")
    r2 = record_hypothesis(title="Untouched sibling")

    update_hypothesis_status(hypothesis_id=r1["id"], status=HypothesisStatus.DEBUGGING)
    update_hypothesis_status(hypothesis_id=r1["id"], status=HypothesisStatus.REJECTED)

    all_hyps = {h["id"]: h for h in list_hypotheses()["hypotheses"]}
    assert all_hyps[r1["id"]]["status"] == HypothesisStatus.REJECTED.value
    # r2 should still be PROPOSED — untouched
    assert all_hyps[r2["id"]]["status"] == HypothesisStatus.PROPOSED.value


# ---------------------------------------------------------------------------
# 9. Status round-trips through Pydantic serialisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", list(HypothesisStatus))
def test_status_roundtrip_through_serialization(status: HypothesisStatus):
    """Every HypothesisStatus value must survive a model_dump / model_validate cycle."""
    r = record_hypothesis(title=f"Status round-trip {status.value}", status=status)
    # Deserialise back into a HypothesisRecord
    restored = HypothesisRecord.model_validate(r)
    assert restored.status == status


@pytest.mark.parametrize("status", list(HypothesisStatus))
def test_update_status_roundtrip(status: HypothesisStatus):
    """update_hypothesis_status must persist and return the exact same enum member."""
    r = record_hypothesis(title="Roundtrip after update")
    result = update_hypothesis_status(hypothesis_id=r["id"], status=status)
    restored = HypothesisRecord.model_validate(result)
    assert restored.status == status


# ---------------------------------------------------------------------------
# 10. Tool registration in tool_service.py (AST inspection — no heavy import)
# ---------------------------------------------------------------------------


def test_all_four_tools_registered_in_tool_service_source():
    """Verify tool_service.py wires up hypothesis state tools via create_hypothesis_state_tools().

    The four tool names are registered via a loop (same as the todos pattern), so they
    don't appear as literal strings in tool_service.py.  Instead we verify:
      (a) create_hypothesis_state_tools is imported and called in tool_service.py, AND
      (b) each tool name is declared as a literal in hypothesis_state_tool.py itself
          (inside create_hypothesis_state_tools), proving the factory produces them.

    This mirrors the A8 test pattern: AST inspection avoids the heavy ML import chain.
    """
    import ast
    import pathlib

    tools_dir = (
        pathlib.Path(__file__).parents[3]
        / "app"
        / "modules"
        / "intelligence"
        / "tools"
    )

    # (a) tool_service.py must import and call create_hypothesis_state_tools
    ts_source = (tools_dir / "tool_service.py").read_text(encoding="utf-8")
    assert "create_hypothesis_state_tools" in ts_source, (
        "tool_service.py must import create_hypothesis_state_tools from hypothesis_state_tool"
    )

    # (b) hypothesis_state_tool.py must declare all four tool names as string literals
    hs_source = (tools_dir / "hypothesis_state_tool.py").read_text(encoding="utf-8")
    hs_tree = ast.parse(hs_source)

    declared_names: list[str] = []
    for node in ast.walk(hs_tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            declared_names.append(node.value)

    expected = [
        "record_hypothesis",
        "update_hypothesis_status",
        "append_hypothesis_evidence",
        "list_hypotheses",
    ]
    for name in expected:
        assert name in declared_names, (
            f"'{name}' not found as a string literal in hypothesis_state_tool.py. "
            f"create_hypothesis_state_tools() must define a SimpleTool with this name."
        )


def test_hypothesis_state_tool_import_in_tool_service_source():
    """tool_service.py must import from hypothesis_state_tool."""
    import pathlib

    src_path = (
        pathlib.Path(__file__).parents[3]
        / "app"
        / "modules"
        / "intelligence"
        / "tools"
        / "tool_service.py"
    )
    source = src_path.read_text(encoding="utf-8")
    assert "hypothesis_state_tool" in source, (
        "tool_service.py must import from hypothesis_state_tool"
    )


# ---------------------------------------------------------------------------
# 11. DebugAgent tool list (AST inspection)
# ---------------------------------------------------------------------------


def test_all_four_tools_in_debug_agent_tool_list():
    """All four hypothesis tool names must appear in DebugAgent's get_tools([...]) call."""
    import ast
    import pathlib

    src_path = (
        pathlib.Path(__file__).parents[3]
        / "app"
        / "modules"
        / "intelligence"
        / "agents"
        / "chat_agents"
        / "system_agents"
        / "debug_agent.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    tool_list: list[str] = []
    for node in ast.walk(tree):
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

    expected = [
        "record_hypothesis",
        "update_hypothesis_status",
        "append_hypothesis_evidence",
        "list_hypotheses",
    ]
    for name in expected:
        assert name in tool_list, (
            f"'{name}' not found in DebugAgent's get_tools([...]) list. "
            f"Found: {tool_list}"
        )


# ---------------------------------------------------------------------------
# 12. Pydantic input schema sanity checks
# ---------------------------------------------------------------------------


def test_record_hypothesis_input_defaults():
    inp = RecordHypothesisInput(title="Test")
    assert inp.status == HypothesisStatus.PROPOSED
    assert inp.evidence == []
    assert inp.validation_plan == []


def test_update_hypothesis_status_input_requires_both_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UpdateHypothesisStatusInput(hypothesis_id="hyp_1")  # missing status

    with pytest.raises(ValidationError):
        UpdateHypothesisStatusInput(status=HypothesisStatus.DEBUGGING)  # missing id


def test_append_hypothesis_evidence_input_requires_both_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AppendHypothesisEvidenceInput(hypothesis_id="hyp_1")  # missing evidence

    with pytest.raises(ValidationError):
        AppendHypothesisEvidenceInput(evidence="Observation")  # missing id


# ---------------------------------------------------------------------------
# 13. create_hypothesis_state_tools returns the expected tool names
# ---------------------------------------------------------------------------


def test_create_hypothesis_state_tools_returns_four_tools():
    tools = create_hypothesis_state_tools()
    names = {t.name for t in tools}
    assert names == {
        "record_hypothesis",
        "update_hypothesis_status",
        "append_hypothesis_evidence",
        "list_hypotheses",
    }


# ---------------------------------------------------------------------------
# 14. conversation_id on HypothesisRecord (A5.1)
# ---------------------------------------------------------------------------


def test_record_carries_store_conversation_id():
    """HypothesisRecord should carry the conversation_id from the bound store."""
    store = HypothesisStore(conversation_id="conv_xyz_123")
    _hypothesis_store_ctx.set(store)
    result = record_hypothesis(title="Conv-id test hypothesis")
    assert result["conversation_id"] == "conv_xyz_123"


def test_record_default_conversation_id_is_empty_string():
    """When HypothesisStore is created with no conversation_id, it defaults to empty string."""
    # The autouse fixture installs a fresh default store (no conversation_id)
    result = record_hypothesis(title="Default conversation_id hypothesis")
    assert result["conversation_id"] == ""
