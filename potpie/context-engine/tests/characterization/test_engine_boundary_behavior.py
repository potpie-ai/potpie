from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from adapters.outbound.ledger.self_hosted_client import FixtureEventLedgerClient
from adapters.outbound.skills.template_resources import PackageTemplateResources
from bootstrap.host_wiring import build_host_shell
from domain.ports.agent_context import RecordRequest, ResolveRequest, SearchRequest
from domain.ports.ledger.client import LedgerEvent
from domain.ports.services.graph_service import GraphReadRequest

pytestmark = pytest.mark.unit


@pytest.fixture()
def boundary_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ledger = FixtureEventLedgerClient()
    ledger.seed(
        "github",
        [
            LedgerEvent(
                event_id="event-pr-42",
                source_id="github",
                provider="github",
                kind="pr_merge",
                payload={"repo": "potpie-ai/potpie", "number": 42},
                occurred_at=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
            ),
            LedgerEvent(
                event_id="event-issue-7",
                source_id="github",
                provider="github",
                kind="issue_closed",
                payload={"repo": "potpie-ai/potpie", "number": 7},
                occurred_at=datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc),
            ),
        ],
    )
    return build_host_shell(
        backend=InMemoryGraphBackend(),
        ledger_client=ledger,
        template_resources=PackageTemplateResources("potpie.cli"),
    )


def test_context_pot_and_source_behavior(boundary_host) -> None:
    pot = boundary_host.pots.create_pot(name="boundary", use=True)
    source = boundary_host.pots.add_source(
        pot_id=pot.pot_id,
        kind="repo",
        location="https://github.com/potpie-ai/potpie.git",
        name="potpie",
    )

    receipt = boundary_host.agent_context.record(
        RecordRequest(
            pot_id=pot.pot_id,
            record_type="preference",
            summary="use ruff for Python linting",
            details={
                "policy_kind": "style",
                "prescription": "use ruff for Python linting",
            },
            scope={"language": "python"},
            source_refs=("repo:AGENTS.md",),
        )
    )
    resolved = boundary_host.agent_context.resolve(
        ResolveRequest(
            pot_id=pot.pot_id,
            task="Choose the Python linter",
            intent="feature",
            include=("coding_preferences",),
            scope={"language": "python"},
        )
    )
    searched = boundary_host.agent_context.search(
        SearchRequest(
            pot_id=pot.pot_id,
            query="ruff",
            include=("coding_preferences",),
            scope={"language": "python"},
        )
    )

    assert receipt.accepted is True
    assert receipt.status == "recorded"
    assert receipt.mutations_applied == 1
    assert receipt.metadata["graph_contract_version"] == "v1.5"
    assert "ruff" in json.dumps(resolved.to_dict()).lower()
    assert "ruff" in json.dumps(searched.to_dict()).lower()
    assert boundary_host.pots.active_pot() == pot
    assert boundary_host.pots.list_pots() == [pot]
    assert (
        boundary_host.pots.source_status(pot_id=pot.pot_id, source_id=source.source_id)
        == source
    )
    assert boundary_host.pots.list_sources(pot_id=pot.pot_id) == [source]

    boundary_host.pots.remove_source(pot_id=pot.pot_id, source_id=source.source_id)
    assert boundary_host.pots.list_sources(pot_id=pot.pot_id) == []


def test_graph_propose_commit_read_and_history_behavior(boundary_host) -> None:
    pot = boundary_host.pots.create_pot(name="boundary", use=True)
    payload = {
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "infra_topology",
                "subject": {"key": "service:payments-api", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:ledger-api", "type": "Service"},
                "truth": "source_observation",
                "description": "payments depends on ledger to post entries",
                "evidence": [
                    {
                        "source_ref": "repo:deploy/manifest.yaml",
                        "authority": "repository_metadata",
                    }
                ],
            }
        ]
    }

    proposal = boundary_host.graph_workbench.propose(payload, pot_id=pot.pot_id)
    before_commit = boundary_host.graph.read(
        GraphReadRequest(
            pot_id=pot.pot_id,
            subgraph="infra_topology",
            view="service_neighborhood",
            scope={"service": "payments-api"},
            depth=2,
        )
    )
    committed = boundary_host.graph_workbench.commit(
        proposal.plan_id,
        pot_id=pot.pot_id,
    )
    after_commit = boundary_host.graph.read(
        GraphReadRequest(
            pot_id=pot.pot_id,
            subgraph="infra_topology",
            view="service_neighborhood",
            scope={"service": "payments-api"},
            depth=2,
        )
    )
    history = boundary_host.graph_workbench.history(
        pot_id=pot.pot_id,
        plan_id=proposal.plan_id,
    )

    assert proposal.ok is True
    assert proposal.status == "validated"
    assert proposal.risk == "low"
    assert proposal.diff.entity_upserts == 2
    assert proposal.diff.edge_upserts == 1
    assert before_commit.items == ()
    assert committed.ok is True
    assert committed.status == "committed"
    assert committed.diff == proposal.diff
    assert after_commit.view == "infra_topology.service_neighborhood"
    assert after_commit.items
    after_payload = after_commit.to_dict()
    assert "service:ledger-api" in json.dumps(after_payload)
    assert after_payload["source_refs"] == ["repo:deploy/manifest.yaml"]
    assert any(entry.plan_id == proposal.plan_id for entry in history.entries)


def test_ledger_and_timeline_behavior(boundary_host, tmp_path: Path) -> None:
    pot = boundary_host.pots.create_pot(name="boundary", use=True)

    assert boundary_host.ledger.status().available is True
    assert [
        source.source_id for source in boundary_host.ledger.sources(pot_id=pot.pot_id)
    ] == ["github"]
    queried = boundary_host.ledger.query(
        pot_id=pot.pot_id,
        source_id="github",
        kind="pr_merge",
    )
    pulled = boundary_host.ledger.pull(
        pot_id=pot.pot_id,
        source_id="github",
        limit=1,
    )
    next_page = boundary_host.ledger.pull(
        pot_id=pot.pot_id,
        source_id="github",
        limit=10,
    )

    assert [event.event_id for event in queried.events] == ["event-pr-42"]
    assert queried.next_cursor is None
    assert [event.event_id for event in pulled.events] == ["event-pr-42"]
    assert pulled.has_more is True
    assert [event.event_id for event in next_page.events] == ["event-issue-7"]
    assert json.loads((tmp_path / "ledger_cursors.json").read_text()) == {
        f"{pot.pot_id}:github": "2"
    }

    proposal = boundary_host.graph_workbench.propose(
        {
            "operations": [
                {
                    "op": "append_event",
                    "verb": "deployed",
                    "occurred_at": "2026-07-03T09:30:00Z",
                    "description": "deployed payments api version 2",
                    "actor": {"key": "person:alice", "type": "Person"},
                    "targets": [{"key": "service:payments-api", "type": "Service"}],
                    "evidence": [
                        {
                            "source_ref": "deploy:42",
                            "authority": "external_system",
                        }
                    ],
                }
            ]
        },
        pot_id=pot.pot_id,
    )
    committed = boundary_host.graph_workbench.commit(
        proposal.plan_id,
        pot_id=pot.pot_id,
    )
    timeline = boundary_host.graph.read(
        GraphReadRequest(
            pot_id=pot.pot_id,
            subgraph="recent_changes",
            view="timeline",
            limit=10,
        )
    )

    assert proposal.status == "validated"
    assert committed.status == "committed"
    assert timeline.view == "recent_changes.timeline"
    assert timeline.items
    assert "deployed payments api version 2" in json.dumps(timeline.to_dict())
