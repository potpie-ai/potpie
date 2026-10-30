"""Agent reads and status through the typed local engine boundary.

These run the composed local runtime over the in-memory graph backend and go
through ``LocalEngineClient`` (the same typed operations the daemon serves), so
the agent door, the graph service, the readers and the workbench are all real.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli import main as cli_main
from potpie.cli.commands import _common
from potpie.runtime.composition import build_local_runtime
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.agent_context_port import DEFAULT_INTENT_INCLUDES
from potpie_context_engine.core.ports.agent_context import StatusRequest
from potpie_context_engine.requests import (
    RecordRequest,
    ResolveRequest,
    SearchRequest,
)

pytestmark = pytest.mark.unit

_POT = "default"


@pytest.fixture()
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    composed = build_local_runtime(backend=InMemoryGraphBackend())
    composed.engine.pots.create_pot(name=_POT, use=True)
    _common.set_runtime(composed)
    yield composed
    _common.set_json(False)


def _client():
    return _common.get_engine_client(_POT)


def _facts(envelope, include: str) -> list[str]:
    return [
        str(item.payload.get("fact", ""))
        for item in envelope.items
        if item.include == include
    ]


def test_search_honours_an_explicit_intent(runtime) -> None:
    env = _common.run_engine_operation(
        _client().search(SearchRequest(query="pool exhausted", intent="debugging"))
    )

    assert env.intent == "debugging"
    assert env.metadata["intent_source"] == "explicit"
    assert {report.include for report in env.coverage} == set(
        DEFAULT_INTENT_INCLUDES["debugging"]
    )
    assert "docs" not in {report.include for report in env.coverage}


def test_search_falls_back_to_unknown_on_an_unrecognised_intent(runtime) -> None:
    env = _common.run_engine_operation(
        _client().search(SearchRequest(query="anything", intent="not-an-intent"))
    )

    assert env.intent == "unknown"


def test_bare_search_includes_documents(runtime) -> None:
    env = _common.run_engine_operation(
        _client().search(SearchRequest(query="limitation of liability cap"))
    )

    assert env.intent == "unknown"
    assert "docs" in {report.include for report in env.coverage}


def test_resolve_infers_the_intent_from_the_task(runtime) -> None:
    env = _common.run_engine_operation(
        _client().resolve(ResolveRequest(task="why is checkout failing"))
    )

    assert env.intent == "debugging"
    assert env.metadata["intent_source"] == "inferred"


def test_a_repo_folder_preference_only_surfaces_in_its_scope(runtime) -> None:
    receipt = _common.run_engine_operation(
        _client().record(
            RecordRequest(
                record_type="preference",
                summary="Use repository adapters for payment clients",
                details={
                    "policy_kind": "architecture",
                    "prescription": "Use repository adapters for payment clients",
                },
                scope={
                    "repo": "https://github.com/acme/shop.git",
                    "folder": "src/payments",
                },
            )
        )
    )
    assert receipt.accepted

    def resolve(scope):
        return _common.run_engine_operation(
            _client().resolve(
                ResolveRequest(include=("coding_preferences",), scope=scope)
            )
        )

    matching = resolve(
        {"repo": "git@github.com:acme/shop.git", "path": "src/payments/client.py"}
    )
    unrelated = resolve(
        {"repo": "github.com/acme/other", "path": "src/payments/client.py"}
    )

    assert any(
        "repository adapters" in fact for fact in _facts(matching, "coding_preferences")
    )
    assert not any(
        "repository adapters" in fact
        for fact in _facts(unrelated, "coding_preferences")
    )


def test_a_project_preference_only_surfaces_in_its_project(runtime) -> None:
    receipt = _common.run_engine_operation(
        _client().record(
            RecordRequest(
                record_type="preference",
                summary="Keep checkout handlers small",
                details={"policy_kind": "structure"},
                scope={"project": "project:checkout"},
            )
        )
    )
    assert receipt.accepted

    def resolve(project):
        return _common.run_engine_operation(
            _client().resolve(
                ResolveRequest(
                    include=("coding_preferences",), scope={"project": project}
                )
            )
        )

    for project in ("checkout", "project:checkout"):
        assert any(
            "checkout handlers" in fact
            for fact in _facts(resolve(project), "coding_preferences")
        )
    assert not any(
        "checkout handlers" in fact
        for fact in _facts(resolve("billing-app"), "coding_preferences")
    )


def _owner_payload(team: str) -> dict:
    return {
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "owners",
                "subject": {"key": "service:payments-api", "type": "Service"},
                "predicate": "OWNED_BY",
                "object": {"key": team, "type": "Team"},
                "truth": "source_observation",
                "evidence": [{"source_ref": "repo:owners"}],
                "description": f"payments-api is owned by {team}",
            }
        ]
    }


def _commit_owner_handover(runtime) -> str:
    """Two ownership writes; the second supersedes the first and leaves the
    first team orphaned, which the quality summary reports as a finding."""
    pot = runtime.engine.pots.active_pot()
    workbench = runtime.engine.graph_workbench
    for team in ("team:platform", "team:product"):
        proposal = workbench.propose(_owner_payload(team), pot_id=pot.pot_id)
        assert proposal.ok, proposal.issues
        assert workbench.commit(proposal.plan_id, pot_id=pot.pot_id).ok
    return pot.pot_id


def test_status_reports_open_graph_quality_findings(runtime) -> None:
    # The backend's analytics projection only counts claims, so status used to
    # report a healthy graph no matter what `graph quality` had open.
    pot_id = _commit_owner_handover(runtime)
    summary = runtime.engine.graph_workbench.quality(
        pot_id=pot_id,
        report="summary",
        subgraph=None,
        limit=20,
        confidence_threshold=0.5,
    )
    open_findings = int(summary.metrics["total_findings"])
    assert open_findings >= 1

    report = runtime.engine.agent_context.status(
        StatusRequest(pot_id=pot_id, harness="claude", intent="feature")
    )

    block = report.data_plane["quality"]
    assert block["source"] == "quality_summary"
    assert block["open_findings"] == open_findings
    assert sum(block["quality_counts"].values()) == open_findings
    assert "graph quality summary" in (report.recommended_next_action or "")


def test_cli_status_reports_open_graph_quality_findings(runtime) -> None:
    _commit_owner_handover(runtime)

    result = CliRunner().invoke(cli_main.app, ["--json", "status", "--pot", _POT])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    block = payload["data_plane"]["quality"]
    assert block["source"] == "quality_summary"
    assert block["open_findings"] >= 1
    assert "graph quality summary" in payload["recommended_next_action"]


def _dependency_payload(op: str) -> dict:
    operation = {
        "op": op,
        "subgraph": "infra_topology",
        "subject": {"key": "service:payments-api", "type": "Service"},
        "predicate": "DEPENDS_ON",
        "object": {"key": "service:ledger-api", "type": "Service"},
    }
    if op == "link_entities":
        operation.update(
            truth="source_observation",
            evidence=[{"source_ref": "repo:manifest"}],
            description="payments-api depends on ledger-api",
        )
    else:
        operation["reason"] = "dependency removed"
    return {"operations": [operation]}


def test_propose_with_approved_by_pre_approves_a_review_required_plan(
    runtime,
) -> None:
    from potpie_context_engine.requests import CommitRequest, ProposeRequest

    client = _client()
    seeded = _common.run_engine_operation(
        client.propose(ProposeRequest(mutation=_dependency_payload("link_entities")))
    )
    assert _common.run_engine_operation(
        client.commit(CommitRequest(plan_id=seeded.plan_id))
    ).ok
    ending = _dependency_payload("end_relation_validity")

    unapproved = _common.run_engine_operation(
        client.propose(ProposeRequest(mutation=ending))
    )
    approved = _common.run_engine_operation(
        client.propose(ProposeRequest(mutation=ending, approved_by="user:alice"))
    )

    assert unapproved.status == "review_required"
    assert approved.status == "validated"
    assert approved.approval.approved_by == "user:alice"
    committed = _common.run_engine_operation(
        client.commit(CommitRequest(plan_id=approved.plan_id))
    )
    assert committed.ok and committed.status == "committed"
