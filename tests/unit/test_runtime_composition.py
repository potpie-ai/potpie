"""Root-owned local runtime composition behavior."""

# ruff: noqa: S101, S108 - pytest assertions and an inert fixture cache path.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from potpie.runtime.composition import build_local_runtime, default_backend_profile
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.ledger.self_hosted_client import (
    FixtureEventLedgerClient,
)
from potpie_context_engine.bootstrap.observability_runtime import (
    get_observability,
    set_observability,
)
from potpie_context_engine.core.lifecycle import (
    DONE,
    NOT_IMPLEMENTED,
    PLANNED,
    SKIPPED,
    SetupPlan,
    SetupPreview,
)
from potpie_context_engine.core.ports.agent_context import (
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    SkillNudge,
    StatusRequest,
)
from potpie_context_engine.core.ports.graph_service import DataPlaneStatus
from potpie_context_engine.domain.ports.ledger.client import LedgerEvent
from potpie_context_engine.domain.ports.observability import NoOpObservability


class _PreparedEmbedder:
    name = "prepared-test-embedder"
    dimensions = 3

    def __init__(self) -> None:
        self.prepared = False

    def embed(self, text: str) -> tuple[float, ...]:
        del text
        return (0.1, 0.2, 0.3)

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]

    def prepare(self) -> dict[str, object]:
        self.prepared = True
        return {
            "provider": "sentence-transformers",
            "model": "all-MiniLM-L6-v2",
            "dimensions": self.dimensions,
            "cache_folder": "/tmp/potpie-model-cache",
        }


@pytest.fixture()
def runtime(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
    return build_local_runtime(backend=InMemoryGraphBackend())


def test_setup_orchestrator_provisions_and_creates_default_pot(runtime) -> None:
    report = runtime.root.setup.run(SetupPlan(repo="potpie", agent="claude"))

    assert report.ok
    states = {step.step: step.state for step in report.steps}
    assert states["config"] == DONE
    assert states["backend.provision"] == DONE
    assert states["pot.default"] == DONE
    assert states["daemon"] == SKIPPED
    assert states["auth"] == NOT_IMPLEMENTED
    active = runtime.root.pots.active_pot()
    assert active is not None and active.name == "default"


def test_setup_prepares_backend_embedder(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
    embedder = _PreparedEmbedder()
    runtime = build_local_runtime(backend=InMemoryGraphBackend(embedder=embedder))

    report = runtime.root.setup.run(
        SetupPlan(repo=None, embeddings="sentence-transformers")
    )

    step = next(item for item in report.steps if item.step == "embeddings.model")
    assert step.state == DONE
    assert step.metadata["model"] == "all-MiniLM-L6-v2"
    assert embedder.prepared is True


def test_setup_is_idempotent(runtime) -> None:
    runtime.root.setup.run(SetupPlan(repo="potpie"))
    report = runtime.root.setup.run(SetupPlan(repo="potpie"))

    assert report.ok
    source = next(step for step in report.steps if step.step == "source")
    assert source.state == SKIPPED


def test_setup_plan_executes_nothing(runtime) -> None:
    steps = runtime.root.setup.plan(SetupPlan())

    assert [step.step for step in steps] == [
        "config",
        "installer",
        "embeddings.model",
        "backend.provision",
        "pot.init",
        "state_store.provision",
        "migrator.migrate",
        "pot.default",
        "daemon",
        "auth",
        "source",
        "skills",
    ]
    assert all(step.state == PLANNED for step in steps)
    assert runtime.root.pots.active_pot() is None


def test_setup_preview_classifies_owners_and_host_gated_hardness(runtime) -> None:
    daemon = runtime.root.setup.preview(SetupPlan(host_mode="daemon"))
    in_process = runtime.root.setup.preview(SetupPlan(host_mode="in_process"))

    assert isinstance(daemon, SetupPreview) and daemon.ok_to_run
    by_step = {step.step: step for step in daemon.steps}
    assert all(step.owner for step in daemon.steps)
    assert {step: planned.hard for step, planned in by_step.items()} == {
        "config": True,
        "installer": True,
        "embeddings.model": False,
        "backend.provision": True,
        "pot.init": True,
        "state_store.provision": True,
        "migrator.migrate": True,
        "pot.default": True,
        "daemon": True,
        "auth": False,
        "source": False,
        "skills": False,
    }

    local_steps = {step.step: step for step in in_process.steps}
    assert local_steps["daemon"].hard is False
    assert local_steps["daemon"].skip_reason
    assert local_steps["installer"].hard is False
    assert local_steps["installer"].skip_reason
    assert runtime.root.pots.active_pot() is None


def test_setup_state_store_and_migrator_skip_cleanly(runtime) -> None:
    report = runtime.root.setup.run(SetupPlan(repo="potpie"))
    states = {step.step: step.state for step in report.steps}

    assert states["state_store.provision"] == SKIPPED
    assert states["migrator.migrate"] == SKIPPED
    assert report.ok


def test_config_location_persistence_and_redaction_survive_reconstruction(
    runtime, tmp_path
) -> None:
    runtime.root.config.set("profile", "managed")
    runtime.root.config.set("service.apiKey", "sk-local-secret")

    config_path = tmp_path / "config.json"
    assert config_path.is_file()
    assert not config_path.with_suffix(".tmp").exists()
    assert runtime.root.config.list_public() == {
        "profile": "managed",
        "service.apiKey": "<redacted>",
    }

    reconstructed = build_local_runtime(backend=InMemoryGraphBackend())

    assert reconstructed.root.config.get("profile") == "managed"
    assert reconstructed.root.config.get("service.apiKey") == "sk-local-secret"
    assert reconstructed.root.config.list_public()["service.apiKey"] == "<redacted>"


def test_pot_source_and_repo_defaults_survive_runtime_reconstruction(
    runtime, tmp_path
) -> None:
    pot = runtime.root.pots.create_pot(name="shop", use=True)
    source = runtime.root.pots.add_source(
        pot_id=pot.pot_id,
        kind="repo",
        location="github.com/acme/shop",
        name="shop-source",
    )
    runtime.root.pots.set_repo_default(
        repo="https://github.com/Acme/Shop.git", pot_id=pot.pot_id
    )

    assert (tmp_path / "pots.json").is_file()
    reconstructed = build_local_runtime(backend=InMemoryGraphBackend())

    active = reconstructed.root.pots.active_pot()
    assert active is not None and active.pot_id == pot.pot_id
    assert reconstructed.root.pots.list_pots() == [active]
    assert reconstructed.root.pots.list_sources(pot_id=pot.pot_id) == [source]
    assert (
        reconstructed.root.pots.repo_default(repo="github.com/acme/shop") == pot.pot_id
    )


def test_local_identity_remains_no_auth(runtime) -> None:
    identity = runtime.root.auth.whoami()

    assert identity.subject == "local"
    assert identity.mode == "none"
    assert identity.detail == "local OSS; no auth required"
    assert runtime.root.auth.logout() is None


def test_agent_context_delegates_and_composes_status(runtime) -> None:
    graph = MagicMock()
    pots = MagicMock()
    skills = MagicMock()
    service = type(runtime.engine.agent_context)(
        graph=graph,
        pots=pots,
        skills=skills,
        profile="local",
    )

    resolve_request = ResolveRequest(pot_id="pot-1", task="resolve")
    search_request = SearchRequest(pot_id="pot-1", query="needle")
    record_request = RecordRequest(
        pot_id="pot-1", record_type="learning", summary="record"
    )
    resolve_result = object()
    search_result = object()
    record_result = object()
    graph.resolve.return_value = resolve_result
    graph.search.return_value = search_result
    graph.record.return_value = record_result

    assert service.resolve(resolve_request) is resolve_result
    assert service.search(search_request) is search_result
    assert service.record(record_request) is record_result
    graph.resolve.assert_called_once_with(resolve_request)
    graph.search.assert_called_once_with(search_request)
    graph.record.assert_called_once_with(record_request)

    pots.aggregate_status.return_value = SimpleNamespace(
        active_pot=SimpleNamespace(pot_id="pot-1", name="shop"),
        pot_count=1,
        sources=(SimpleNamespace(name="shop-source"),),
    )
    graph.data_plane_status.return_value = DataPlaneStatus(
        pot_id="pot-1",
        backend_profile="embedded",
        backend_ready=True,
        reader_backed_includes=("raw_graph",),
        counts={"claims": 2},
        freshness={"state": "fresh"},
        quality={"score": 1},
    )
    nudge = SkillNudge(
        agent="codex",
        missing=("potpie-cli",),
        install_command="potpie skills install --agent codex",
    )
    skills.nudge.return_value = nudge

    status = service.status(StatusRequest(pot_id="", intent="feature", harness="codex"))

    pots.aggregate_status.assert_called_once_with(pot_id="")
    graph.data_plane_status.assert_called_once_with("pot-1")
    skills.nudge.assert_called_once_with(agent="codex")
    assert status.pot_id == "pot-1"
    assert status.active_pot == "shop"
    assert status.backend_ready is True
    assert status.data_plane == {
        "backend_profile": "embedded",
        "backend_ready": True,
        "reader_backed_includes": ["raw_graph"],
        "counts": {"claims": 2},
        "freshness": {"state": "fresh"},
        "quality": {"score": 1},
    }
    assert status.pot_summary == {"pot_count": 1, "sources": ["shop-source"]}
    assert status.skills is nudge
    assert status.recommended_next_action == (
        "Run 'potpie resolve \"<task>\"' to pull context for your work."
    )
    assert status.metadata == {"intent": "feature"}


def test_local_runtime_rejects_runtime_only_backend() -> None:
    backend = InMemoryGraphBackend()
    runtime_only = SimpleNamespace(
        profile=backend.profile,
        mutation=backend.mutation,
        claim_query=backend.claim_query,
        semantic=backend.semantic,
        inspection=backend.inspection,
        analytics=backend.analytics,
        snapshot=backend.snapshot,
        capabilities=backend.capabilities,
        bind_definition=backend.bind_definition,
    )

    with pytest.raises(TypeError, match="must implement deployment provisioning"):
        build_local_runtime(backend=runtime_only)


@pytest.mark.parametrize("profile", ["falkordb", "falkordb_lite"])
def test_local_runtime_accepts_backend_env(
    profile: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", profile)
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
    if profile == "falkordb_lite":
        monkeypatch.setenv("FALKORDB_MODE", "server")

    runtime = build_local_runtime()

    assert runtime.root.backend.profile == profile


def test_local_runtime_defaults_to_falkordb_lite(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
    monkeypatch.delenv("CONTEXT_ENGINE_BACKEND", raising=False)
    monkeypatch.delenv("GRAPH_DB_BACKEND", raising=False)

    runtime = build_local_runtime()

    assert default_backend_profile() == "falkordb_lite"
    assert runtime.root.backend.profile == "falkordb_lite"


def test_default_backend_ignores_blank_primary_env(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "   ")
    monkeypatch.setenv("GRAPH_DB_BACKEND", " embedded ")

    assert default_backend_profile() == "embedded"


def test_local_runtime_wires_process_observability() -> None:
    observability = NoOpObservability()
    original = get_observability()

    try:
        build_local_runtime(backend=InMemoryGraphBackend(), observability=observability)
        assert get_observability() is observability
    finally:
        set_observability(original)


def test_ledger_query_is_read_only_and_pull_does_not_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
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
    runtime = build_local_runtime(backend=InMemoryGraphBackend(), ledger_client=fixture)
    pot = runtime.root.pots.create_pot(name="default", use=True)

    queried = runtime.root.ledger.query(
        pot_id=pot.pot_id, source_id="github", kind="pr_merge"
    )
    pulled = runtime.root.ledger.pull(pot_id=pot.pot_id, source_id="github")
    envelope = runtime.engine.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, include=("raw_graph",))
    )

    assert [event.event_id for event in queried.events] == ["pr1"]
    assert len(pulled.events) == 2
    assert len(envelope.items) == 0
