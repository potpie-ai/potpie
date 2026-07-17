"""End-to-end: the whole architecture through the HostShell.

Proves the wiring holds together: CLI-equivalent calls route
``HostShell -> services -> ports -> backend/ledger`` and the documented journey
(setup → record → resolve → status, plus ledger pull → reconcile) works on the
in-memory backend. This is the "feel of things / find anything missing" check.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from potpie_context_engine.adapters.outbound.ledger.self_hosted_client import FixtureEventLedgerClient
from potpie.services.host_wiring import build_host_shell
from potpie_context_engine.domain.context_records import ContextRecordValidationError
from potpie_context_engine.domain.lifecycle import (
    DONE,
    NOT_IMPLEMENTED,
    PLANNED,
    SKIPPED,
    SetupPlan,
    SetupPreview,
)
from potpie_context_engine.domain.ports.agent_context import (
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    StatusRequest,
)
from potpie_context_engine.domain.ports.ledger.client import LedgerEvent


@pytest.fixture()
def host(tmp_path, monkeypatch):
    # Isolate pot/skill/cursor state under tmp; use a shared in-memory backend
    # so record→resolve round-trips within the test process.
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return build_host_shell(backend=InMemoryGraphBackend())


class _PreparedEmbedder:
    name = "prepared-test-embedder"
    dimensions = 3

    def __init__(self) -> None:
        self.prepared = False

    def embed(self, text: str) -> tuple[float, ...]:
        return (0.1, 0.2, 0.3)

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]

    def prepare(self) -> dict[str, object]:
        self.prepared = True
        return {
            "provider": "sentence-transformers",
            "model": "all-MiniLM-L6-v2",
            "dimensions": self.dimensions,
            "cache_folder": "/tmp/potpie-model-cache",
        }


def test_setup_record_resolve_status_journey(host):
    pot = host.pots.create_pot(name="default", repo="potpie", use=True)
    host.skills.install(agent="claude")

    receipt = host.agent_context.record(
        RecordRequest(
            pot_id=pot.pot_id,
            record_type="decision",
            summary="adopt hexagonal ports",
            details={
                "title": "adopt hexagonal ports",
                "rationale": "Keep application services isolated behind ports.",
            },
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


def test_context_record_rejects_malformed_known_record_details(host):
    pot = host.pots.create_pot(name="default", repo="potpie", use=True)

    with pytest.raises(ContextRecordValidationError, match="bug_pattern: 'kind'"):
        host.agent_context.record(
            RecordRequest(
                pot_id=pot.pot_id,
                record_type="bug_pattern",
                summary="QueuePool limit exceeded under load",
                details={"kind": 123},
            )
        )

    status = host.graph.data_plane_status(pot.pot_id)
    assert status.counts.get("claims", 0) == 0


def test_context_record_rejects_summary_only_known_record(host):
    pot = host.pots.create_pot(name="default", repo="potpie", use=True)

    with pytest.raises(ContextRecordValidationError, match="bug_pattern: 'kind'"):
        host.agent_context.record(
            RecordRequest(
                pot_id=pot.pot_id,
                record_type="bug_pattern",
                summary="QueuePool limit exceeded under load",
            )
        )

    status = host.graph.data_plane_status(pot.pot_id)
    assert status.counts.get("claims", 0) == 0


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


def test_setup_prepares_backend_embedder(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    embedder = _PreparedEmbedder()
    local_host = build_host_shell(backend=InMemoryGraphBackend(embedder=embedder))

    report = local_host.setup.run(
        SetupPlan(repo=None, embeddings="sentence-transformers")
    )

    step = next(s for s in report.steps if s.step == "embeddings.model")
    assert step.state == DONE
    assert step.metadata["model"] == "all-MiniLM-L6-v2"
    assert embedder.prepared is True


def test_setup_is_idempotent(host):
    host.setup.run(SetupPlan(repo="potpie"))
    report = host.setup.run(SetupPlan(repo="potpie"))
    assert report.ok
    source = next(s for s in report.steps if s.step == "source")
    assert source.state == SKIPPED  # repo already registered on re-run


def test_setup_dry_run_executes_nothing(host):
    steps = host.setup.plan(SetupPlan())
    assert [s.step for s in steps][:4] == [
        "config",
        "installer",
        "embeddings.model",
        "backend.provision",
    ]
    assert all(s.state == PLANNED for s in steps)
    assert host.pots.active_pot() is None  # plan() creates nothing


def test_setup_preview_classifies_owners_and_host_gated_hardness(host):
    # preview() is the dry-run primary: per-step owner + hard/soft + skip reason.
    daemon = host.setup.preview(SetupPlan(host_mode="daemon"))
    in_proc = host.setup.preview(SetupPlan(host_mode="in_process"))
    assert isinstance(daemon, SetupPreview) and daemon.ok_to_run

    by_step = {s.step: s for s in daemon.steps}
    assert by_step["config"].owner and by_step["config"].hard is True
    assert by_step["embeddings.model"].hard is False
    assert by_step["auth"].hard is False  # soft step
    # HU16 seams are present in the plan and hard for the flat-file profile.
    assert "state_store.provision" in by_step and by_step["state_store.provision"].hard
    assert "migrator.migrate" in by_step

    # host_mode flips daemon/installer hardness + carries a skip reason.
    assert by_step["daemon"].hard is True
    ip = {s.step: s for s in in_proc.steps}
    assert ip["daemon"].hard is False and ip["daemon"].skip_reason
    assert ip["installer"].hard is False and ip["installer"].skip_reason

    assert host.pots.active_pot() is None  # preview executes nothing


def test_setup_run_state_store_and_migrator_skip_cleanly(host):
    report = host.setup.run(SetupPlan(repo="potpie"))
    states = {s.step: s.state for s in report.steps}
    assert states["state_store.provision"] == SKIPPED  # flat-file: nothing to provision
    assert states["migrator.migrate"] == SKIPPED  # flat-file: no migrations
    assert report.ok  # the new hard steps do not block a clean run


def test_stub_backend_profiles_registered_and_fail_closed():
    from potpie_context_engine.adapters.outbound.graph.backends import KNOWN_PROFILES, build_backend
    from potpie_context_engine.domain.errors import CapabilityNotImplemented

    assert "falkordb" in KNOWN_PROFILES
    assert "falkordb_lite" in KNOWN_PROFILES

    for profile in ("postgres", "chroma", "hosted"):
        assert profile in KNOWN_PROFILES
        backend = build_backend(profile)
        assert backend.profile == profile
        assert backend.capabilities().implemented() == ()  # nothing built yet
        with pytest.raises(CapabilityNotImplemented):
            backend.inspection.neighborhood(pot_id="p", entity_key="e")
        with pytest.raises(CapabilityNotImplemented):
            backend.provision(SetupPlan(backend=profile))


def test_search_returns_envelope(host):
    pot = host.pots.create_pot(name="default", use=True)
    host.agent_context.record(
        RecordRequest(
            pot_id=pot.pot_id,
            record_type="preference",
            summary="use ruff",
            details={"policy_kind": "style", "prescription": "use ruff"},
        )
    )
    env = host.agent_context.search(
        SearchRequest(pot_id=pot.pot_id, query="ruff", include=("raw_graph",))
    )
    assert env.pot_id == pot.pot_id


def test_unsupported_include_is_flagged_not_crashed(host):
    pot = host.pots.create_pot(name="default", use=True)
    env = host.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, intent="feature", include=("nonsense",))
    )
    # An include outside the advertised vocab → honest unknown_include, no crash.
    # (Every advertised family now has a reader, so there is no reader-less
    # advertised include left to exercise the not_implemented path.)
    assert any(
        u.name == "nonsense" and u.reason == "unknown_include"
        for u in env.unsupported_includes
    )


def test_ledger_pull_is_read_only(tmp_path, monkeypatch):
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
                payload={
                    "subject": "svc:api",
                    "object": "svc:db",
                    "fact": "api depends on db",
                },
            )
        ],
    )
    host = build_host_shell(backend=InMemoryGraphBackend(), ledger_client=fixture)
    pot = host.pots.create_pot(name="default", use=True)

    assert host.ledger.status().available
    page = host.ledger.pull(pot_id=pot.pot_id, source_id="github")
    assert len(page.events) == 1

    env = host.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, include=("raw_graph",))
    )
    assert len(env.items) == 0


def test_ledger_query_inspects_history_without_advancing_cursor(tmp_path, monkeypatch):
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
                payload={},
            ),
            LedgerEvent(
                event_id="iss1",
                source_id="github",
                provider="github",
                kind="issue_create",
                payload={},
            ),
        ],
    )
    host = build_host_shell(backend=InMemoryGraphBackend(), ledger_client=fixture)
    pot = host.pots.create_pot(name="default", use=True)

    # query filters by kind and is read-only (no cursor advance).
    page = host.ledger.query(pot_id=pot.pot_id, source_id="github", kind="pr_merge")
    assert [e.event_id for e in page.events] == ["pr1"]

    # because query never touched the cursor, a later pull still sees both events.
    pulled = host.ledger.pull(pot_id=pot.pot_id, source_id="github")
    assert len(pulled.events) == 2


def test_ledger_pull_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    fixture = FixtureEventLedgerClient()
    fixture.seed(
        "github",
        [
            LedgerEvent(
                event_id="pr1",
                source_id="github",
                provider="github",
                kind="x",
                payload={},
            )
        ],
    )
    host = build_host_shell(backend=InMemoryGraphBackend(), ledger_client=fixture)
    pot = host.pots.create_pot(name="default", use=True)

    page = host.ledger.pull(pot_id=pot.pot_id, source_id="github")
    assert len(page.events) == 1
    env = host.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, include=("raw_graph",))
    )
    assert len(env.items) == 0
