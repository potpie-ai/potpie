"""The Ladybug profile resolves through the shared backend registry."""

# ruff: noqa: S101 - pytest assertions.

import sys
from pathlib import Path
from types import SimpleNamespace

from potpie_context_engine.adapters.outbound.graph.backends import (
    KNOWN_PROFILES,
    build_backend,
)


def test_ladybug_registry_uses_configured_database_path(monkeypatch, tmp_path):
    database_path = tmp_path / "configured" / "graph.db"
    monkeypatch.setenv("CONTEXT_ENGINE_LADYBUG_PATH", str(database_path))
    embedder = object()
    constructed = []

    def backend_factory(**kwargs):
        constructed.append(kwargs)
        return SimpleNamespace(profile="ladybug")

    monkeypatch.setitem(
        sys.modules,
        "potpie_context_engine.adapters.outbound.graph.backends.ladybug_backend",
        SimpleNamespace(LadybugGraphBackend=backend_factory),
    )

    backend = build_backend(" Ladybug ", embedder=embedder)

    assert "ladybug" in KNOWN_PROFILES
    assert backend.profile == "ladybug"
    assert constructed == [{"db_path": Path(database_path), "embedder": embedder}]
