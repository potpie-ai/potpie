"""Shared POT-2700 fixtures: isolated stores and deterministic failure windows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
import uuid

from potpie_context_engine.core.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_engine.core.reconciliation import MutationBatch


def reliability_profiles() -> tuple[str, ...]:
    return tuple(
        os.environ.get("POTPIE_RELIABILITY_PROFILES", "in_memory,embedded").split(",")
    )


@dataclass
class WritePause:
    """Pause a patched write at a precise boundary; never rely on sleeps."""

    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def wait(self) -> None:
        self.entered.set()
        if not self.release.wait(10):
            raise TimeoutError("reliability test did not release the paused write")


def assertion_batch(
    *, valid_at="2026-01-01T00:00:00+00:00", description="API depends on DB"
):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Service",)),
            EntityUpsert("service:db", ("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:api",
                "service:db",
                {
                    "claim_key": "claim:reliability",
                    "subgraph": "infra_topology",
                    "truth": "source_observation",
                    "source_ref": "fixture:reliability",
                    "environment": "prod",
                    "valid_at": valid_at,
                    "valid_from": valid_at,
                    "description": description,
                },
            )
        ],
    )


@contextmanager
def isolated_backend(profile: str, home: Path, monkeypatch):
    from potpie_context_engine.adapters.outbound.graph.backends import build_backend
    from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
        EmbeddedGraphBackend,
    )

    pot = "reliability-" + uuid.uuid4().hex
    monkeypatch.setenv("CONTEXT_ENGINE_EMBEDDER", "none")
    monkeypatch.setenv(
        "CONTEXT_ENGINE_FALKORDB_GRAPH_NAME", "reliability_" + uuid.uuid4().hex
    )
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_LITE_PATH", str(home / "graph.db"))
    if profile == "neo4j":
        endpoint = os.environ.get("PROTOCOL_TEST_NEO4J_URI")
        if not endpoint:
            raise RuntimeError(
                "selected Neo4j reliability tests require PROTOCOL_TEST_NEO4J_URI"
            )
        for suffix, value in {
            "URI": endpoint,
            "USERNAME": os.environ.get("PROTOCOL_TEST_NEO4J_USERNAME", "neo4j"),
            "PASSWORD": os.environ.get("PROTOCOL_TEST_NEO4J_PASSWORD", "test"),
        }.items():
            monkeypatch.setenv("CONTEXT_ENGINE_NEO4J_" + suffix, value)
    if profile == "falkordb":
        endpoint = os.environ.get("PROTOCOL_TEST_FALKORDB_URL")
        if not endpoint:
            raise RuntimeError(
                "selected FalkorDB reliability tests require PROTOCOL_TEST_FALKORDB_URL"
            )
        monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_URL", endpoint)
    backend = (
        EmbeddedGraphBackend(home=home)
        if profile == "embedded"
        else build_backend(profile)
    )
    try:
        yield pot, backend
    finally:
        if profile == "neo4j":
            backend.mutation.reset_pot(pot)
        if profile in {"falkordb", "falkordb_lite"}:
            graph = backend.graph_provider()
            if graph is not None:
                graph.delete()
        if profile == "falkordb_lite":
            from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
                shutdown_embedded_servers,
            )

            shutdown_embedded_servers()
