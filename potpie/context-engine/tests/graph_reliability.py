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


def backend_factory(profile: str, home: Path, monkeypatch, *, require_endpoint=True):
    """Configure an isolated backend and return a reopenable constructor.

    Endpoint credentials and optional native dependencies are checked here so
    protocol, temporal and reliability suites use the same profile policy.
    """
    import importlib.util
    import pytest
    from potpie_context_engine.adapters.outbound.graph.backends import build_backend

    monkeypatch.setenv("CONTEXT_ENGINE_EMBEDDER", "none")
    monkeypatch.setenv("CONTEXT_GRAPH_ENABLED", "1")
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_GRAPH_NAME", "test_" + uuid.uuid4().hex)
    monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_LITE_PATH", str(home / "graph.db"))
    monkeypatch.setenv("CONTEXT_ENGINE_LADYBUG_PATH", str(home / "ladybug.db"))
    if profile == "ladybug" and importlib.util.find_spec("ladybug") is None:
        if os.environ.get("POTPIE_REQUIRE_LADYBUG") == "1":
            pytest.fail("required Ladybug native dependency is unavailable")
        pytest.skip("optional Ladybug native dependency is unavailable")
    if profile == "falkordb_lite":
        pytest.importorskip("redislite.falkordb_client")
    if profile in {"neo4j", "falkordb"}:
        prefix = "PROTOCOL_TEST_" + (
            "NEO4J_URI" if profile == "neo4j" else "FALKORDB_URL"
        )
        endpoint = os.environ.get(prefix)
        if not endpoint and require_endpoint:
            pytest.skip(prefix + " must name an isolated test server")
        if endpoint:
            if profile == "neo4j":
                for suffix, value in {
                    "URI": endpoint,
                    "USERNAME": os.environ.get("PROTOCOL_TEST_NEO4J_USERNAME", "neo4j"),
                    "PASSWORD": os.environ.get("PROTOCOL_TEST_NEO4J_PASSWORD", "test"),
                }.items():
                    monkeypatch.setenv("CONTEXT_ENGINE_NEO4J_" + suffix, value)
            else:
                monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_URL", endpoint)
                monkeypatch.setenv("CONTEXT_ENGINE_FALKORDB_MODE", "server")

    def factory():
        if profile in {"embedded", "ladybug"}:
            if profile == "embedded":
                from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
                    EmbeddedGraphBackend,
                )

                return EmbeddedGraphBackend(home=home)
            from potpie_context_engine.adapters.outbound.graph.backends.ladybug_backend import (
                LadybugGraphBackend,
            )

            return LadybugGraphBackend(home=home)
        return build_backend(profile)

    return factory


@contextmanager
def isolated_backend(profile: str, home: Path, monkeypatch, *, pots=()):
    factory = backend_factory(profile, home, monkeypatch)
    pot = "reliability-" + uuid.uuid4().hex
    backend = factory()
    cleanup_pots = (pot, *pots)
    try:
        yield pot, backend
    finally:
        if profile == "neo4j":
            for pot_id in cleanup_pots:
                backend.mutation.reset_pot(pot_id)
        if profile in {"falkordb", "falkordb_lite"}:
            graph = backend.graph_provider()
            if graph is not None:
                graph.delete()
        if profile == "falkordb_lite":
            from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
                shutdown_embedded_servers,
            )

            shutdown_embedded_servers()
