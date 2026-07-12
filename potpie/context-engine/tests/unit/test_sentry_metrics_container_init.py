from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

from potpie_context_engine.adapters.outbound.graph import Neo4jGraphWriter
from potpie_context_engine.application.services.source_connector_registry import (
    SourceConnectorRegistry,
)
from potpie_context_engine.bootstrap import ingestion_server, standalone_container
from potpie_context_engine.bootstrap.http_projects import ExplicitPotResolution
from potpie_context_engine.domain.ports.observability import NoOpObservability
from potpie_context_engine.domain.ports.telemetry import NoOpTelemetry


@dataclass(frozen=True)
class _Settings:
    def is_enabled(self) -> bool:
        return False

    def neo4j_uri(self) -> str | None:
        return None

    def neo4j_user(self) -> str | None:
        return None

    def neo4j_password(self) -> str | None:
        return None

    def graph_db_backend(self) -> str:
        return "neo4j"

    def falkordb_url(self) -> str | None:
        return None

    def falkordb_graph_name(self) -> str:
        return "context_graph"

    def falkordb_mode(self) -> str:
        return "lite"

    def falkordb_lite_path(self) -> str:
        return ".potpie/context_graph/falkordb.db"

    def backfill_max_prs_per_run(self) -> int:
        return 25


@dataclass(frozen=True)
class _GraphBackend:
    settings: _Settings
    writer: Neo4jGraphWriter


@dataclass(frozen=True)
class _ContextGraph:
    backend: _GraphBackend
    backed_includes: frozenset[str] = frozenset()


def test_build_ingestion_server_does_not_configure_product_sentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(
        sys.modules, "potpie.runtime.telemetry.sentry_metrics", raising=False
    )
    _patch_container_graph(monkeypatch)

    for _ in range(2):
        ingestion_server.build_ingestion_server(
            settings=_Settings(),
            pots=ExplicitPotResolution({"pot": "owner/repo"}),
            connectors=SourceConnectorRegistry(),
            telemetry=NoOpTelemetry(),
            observability=NoOpObservability(),
        )

    assert "potpie.runtime.telemetry.sentry_metrics" not in sys.modules


def test_standalone_container_delegates_to_ingestion_server_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = ingestion_server.IngestionServerContainer(
        settings=_Settings(),
        graph_writer=Neo4jGraphWriter(_Settings()),
        pots=ExplicitPotResolution({"pot": "owner/repo"}),
    )
    calls: list[dict[str, bool]] = []

    def build(**_kwargs: object) -> ingestion_server.IngestionServerContainer:
        calls.append({"called": True})
        return container

    monkeypatch.setattr(
        standalone_container,
        "merged_pot_repo_map",
        lambda: {"pot": "owner/repo"},
    )
    monkeypatch.setattr(
        standalone_container, "get_context_graph_job_queue", lambda: None
    )
    monkeypatch.setattr(
        standalone_container,
        "try_pydantic_deep_reconciliation_agent",
        lambda: None,
    )
    monkeypatch.delenv("CONTEXT_ENGINE_GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(standalone_container, "build_ingestion_server", build)

    built = standalone_container.build_standalone_context_engine_container()

    assert built is container
    assert calls == [{"called": True}]


def _patch_container_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    import potpie_context_engine.domain.coherence as coherence

    monkeypatch.setattr(
        ingestion_server,
        "Neo4jGraphWriter",
        lambda settings: Neo4jGraphWriter(settings),
    )
    monkeypatch.setattr(
        ingestion_server,
        "Neo4jGraphBackend",
        lambda settings, *, writer: _GraphBackend(settings=settings, writer=writer),
    )
    monkeypatch.setattr(
        ingestion_server,
        "ContextGraphService",
        lambda backend: _ContextGraph(backend=backend),
    )
    monkeypatch.setattr(
        coherence,
        "assert_runtime_coherence",
        lambda *, reader_backed_includes: None,
    )
