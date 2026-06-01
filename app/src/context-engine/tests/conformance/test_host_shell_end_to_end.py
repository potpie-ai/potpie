"""End-to-end: the whole architecture through the HostShell.

Proves the wiring holds together: CLI-equivalent calls route
``HostShell -> services -> ports -> backend/ledger`` and the documented journey
(setup → record → resolve → status, plus ledger pull → reconcile) works on the
in-memory backend. This is the "feel of things / find anything missing" check.
"""

from __future__ import annotations

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from adapters.outbound.ledger.self_hosted_client import FixtureEventLedgerClient
from bootstrap.host_wiring import build_host_shell
from domain.lifecycle import DONE, NOT_IMPLEMENTED, PLANNED, SKIPPED, SetupPlan
from domain.ports.agent_context import (
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    StatusRequest,
)
from domain.ports.ledger.client import LedgerEvent


@pytest.fixture()
def host(tmp_path, monkeypatch):
    # Isolate pot/skill/cursor state under tmp; use a shared in-memory backend
    # so record→resolve round-trips within the test process.
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    return build_host_shell(backend=InMemoryGraphBackend())


def test_setup_record_resolve_status_journey(host):
    pot = host.pots.create_pot(name="default", repo="potpie", use=True)
    host.skills.install(agent="claude")

    receipt = host.agent_context.record(
        RecordRequest(
            pot_id=pot.pot_id,
            record_type="decision",
            summary="adopt hexagonal ports",
            scope={"service": "context-engine"},
        )
    )
    assert receipt.accepted and receipt.mutations_applied >= 1

    env = host.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, intent="feature", include=("raw_graph",))
    )
    assert len(env.items) == 1
    assert "hexagonal" in dict(env.items[0].payload)["fact"]

    report = host.agent_context.status(
        StatusRequest(pot_id=pot.pot_id, harness="claude", intent="feature")
    )
    assert report.backend_ready
    assert report.data_plane["counts"]["claims"] == 1
    assert report.skills is not None and not report.skills.missing


def test_setup_orchestrator_provisions_and_creates_default_pot(host):
    report = host.setup.run(SetupPlan(repo="potpie", agent="claude"))
    assert report.ok  # every hard step succeeded
    states = {s.step: s.state for s in report.steps}
    assert states["config"] == DONE
    assert states["backend.provision"] == DONE
    assert states["pot.default"] == DONE
    assert states["daemon"] == SKIPPED  # in-process host: nothing to start
    assert states["auth"] == NOT_IMPLEMENTED  # soft gap; does not fail setup
    active = host.pots.active_pot()
    assert active is not None and active.name == "default"


def test_setup_is_idempotent(host):
    host.setup.run(SetupPlan(repo="potpie"))
    report = host.setup.run(SetupPlan(repo="potpie"))
    assert report.ok
    source = next(s for s in report.steps if s.step == "source")
    assert source.state == SKIPPED  # repo already registered on re-run


def test_setup_dry_run_executes_nothing(host):
    steps = host.setup.plan(SetupPlan())
    assert [s.step for s in steps][:3] == ["config", "installer", "backend.provision"]
    assert all(s.state == PLANNED for s in steps)
    assert host.pots.active_pot() is None  # plan() creates nothing


def test_search_returns_envelope(host):
    pot = host.pots.create_pot(name="default", use=True)
    host.agent_context.record(
        RecordRequest(pot_id=pot.pot_id, record_type="preference", summary="use ruff")
    )
    env = host.agent_context.search(SearchRequest(pot_id=pot.pot_id, query="ruff", include=("raw_graph",)))
    assert env.pot_id == pot.pot_id


def test_unsupported_include_is_flagged_not_crashed(host):
    pot = host.pots.create_pot(name="default", use=True)
    env = host.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, intent="feature", include=("owners",))
    )
    # 'owners' is advertised but reader-less → honest unsupported, no crash.
    assert any(u.name == "owners" and u.reason == "not_implemented" for u in env.unsupported_includes)


def test_ledger_pull_reconciles_into_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    fixture = FixtureEventLedgerClient()
    fixture.seed(
        "github",
        [
            LedgerEvent(
                event_id="pr1",
                source_id="github",
                provider="github",
                kind="pr_merge",
                payload={"subject": "svc:api", "object": "svc:db", "fact": "api depends on db"},
            )
        ],
    )
    host = build_host_shell(backend=InMemoryGraphBackend(), ledger_client=fixture)
    pot = host.pots.create_pot(name="default", use=True)

    assert host.ledger.status().available
    page, result = host.ledger.pull(pot_id=pot.pot_id, source_id="github", apply=True)
    assert len(page.events) == 1
    assert result.claims_written == 1

    env = host.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, include=("raw_graph",))
    )
    assert any("api depends on db" in dict(i.payload).get("fact", "") for i in env.items)


def test_ledger_pull_dry_run_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    fixture = FixtureEventLedgerClient()
    fixture.seed("github", [LedgerEvent(event_id="pr1", source_id="github", provider="github", kind="x", payload={})])
    host = build_host_shell(backend=InMemoryGraphBackend(), ledger_client=fixture)
    pot = host.pots.create_pot(name="default", use=True)

    page, result = host.ledger.pull(pot_id=pot.pot_id, source_id="github", apply=False)
    assert len(page.events) == 1 and result is None
    env = host.agent_context.resolve(ResolveRequest(pot_id=pot.pot_id, include=("raw_graph",)))
    assert len(env.items) == 0
