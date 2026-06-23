"""``GraphBackend`` profiles + the ``build_backend`` registry.

One registry maps a profile name to a concrete ``GraphBackend``. Adding a
storage option means adding a profile here and implementing the six capability
ports behind it — never changing the services that depend on ``GraphBackend``.

    in_memory   real, tests + conformance + POC          (InMemoryGraphBackend)
    embedded    OSS local default (real, JSON-persisted)  (EmbeddedGraphBackend)
    neo4j       shape-first production target            (Neo4jGraphBackend)
    falkordb    lightweight graph profile                (FalkorDBGraphBackend)
    falkordb_lite embedded FalkorDBLite profile          (FalkorDBLiteGraphBackend)
    postgres    pgvector profile (registered stub)       (StubGraphBackend)
    chroma      vector profile (registered stub)         (StubGraphBackend)
    hosted      managed profile (registered stub)        (StubGraphBackend)

``postgres``/``chroma``/``hosted`` are documented profiles whose owners have not
landed a body yet: they resolve to a ``StubGraphBackend`` whose capability ports
and ``provision`` raise ``CapabilityNotImplemented`` — a real (if unbuilt) seam,
so ``backend list`` shows them and selecting one is the structured
not-implemented contract, not a crash.
"""

from __future__ import annotations

from typing import Any

from context_engine.adapters.outbound.graph.backends.embedded_backend import EmbeddedGraphBackend
from context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from context_engine.adapters.outbound.graph.backends.stub_backend import StubGraphBackend
from context_engine.domain.ports.graph.backend import GraphBackend

KNOWN_PROFILES: tuple[str, ...] = (
    "in_memory",
    "embedded",
    "neo4j",
    "falkordb",
    "falkordb_lite",
    "postgres",
    "chroma",
    "hosted",
)

# Documented profiles routed to a fail-closed StubGraphBackend until built.
_STUB_PROFILES: frozenset[str] = frozenset({"postgres", "chroma", "hosted"})


def build_backend(
    profile: str, *, settings: Any = None, embedder: Any = None
) -> GraphBackend:
    """Construct the ``GraphBackend`` for a profile name.

    ``in_memory`` and ``embedded`` need no settings; ``neo4j`` / ``falkordb`` /
    ``falkordb_lite`` need engine settings (lazy-imported so the graph drivers are optional).
    ``postgres`` / ``chroma`` / ``hosted`` resolve to a fail-closed
    ``StubGraphBackend``.
    Unknown profiles raise ``ValueError`` — the CLI maps that to a validation error.

    ``embedder`` (an :class:`EmbedderPort`) powers semantic retrieval on graph
    backends; when omitted the bundled local embedder is built by default so
    retrieval needs no API key (``CONTEXT_ENGINE_EMBEDDER=none`` disables it).
    """
    name = (profile or "").strip().lower().replace("-", "_")
    if name == "falkordblite":
        name = "falkordb_lite"
    if embedder is None and name in (
        "in_memory",
        "embedded",
        "neo4j",
        "falkordb",
        "falkordb_lite",
    ):
        from context_engine.adapters.outbound.intelligence.local_embedder import build_embedder

        embedder = build_embedder()
    if name == "in_memory":
        return InMemoryGraphBackend(embedder=embedder)
    if name == "embedded":
        return EmbeddedGraphBackend(embedder=embedder)
    if name == "neo4j":
        # Lazy import keeps the neo4j driver optional for other profiles.
        from context_engine.adapters.outbound.graph.backends.neo4j_backend import Neo4jGraphBackend

        if settings is None:
            from context_engine.adapters.outbound.settings_env import EnvContextEngineSettings

            settings = EnvContextEngineSettings()
        return Neo4jGraphBackend(settings, embedder=embedder)
    if name == "falkordb":
        # Lazy import keeps FalkorDB/FalkorDBLite optional for other profiles.
        from context_engine.adapters.outbound.graph.backends.falkordb_backend import (
            FalkorDBGraphBackend,
        )

        if settings is None:
            from context_engine.adapters.outbound.settings_env import EnvContextEngineSettings

            settings = EnvContextEngineSettings()
        return FalkorDBGraphBackend(settings, embedder=embedder)
    if name == "falkordb_lite":
        # Explicit Lite profile: same adapter bundle, mode pinned to embedded Lite.
        from context_engine.adapters.outbound.graph.backends.falkordb_backend import (
            FalkorDBLiteGraphBackend,
        )

        if settings is None:
            from context_engine.adapters.outbound.settings_env import EnvContextEngineSettings

            settings = EnvContextEngineSettings()
        return FalkorDBLiteGraphBackend(settings, embedder=embedder)
    if name in _STUB_PROFILES:
        return StubGraphBackend(name)
    raise ValueError(
        f"Unknown graph backend profile '{profile}'. "
        f"Known profiles: {', '.join(KNOWN_PROFILES)}."
    )


__all__ = [
    "KNOWN_PROFILES",
    "EmbeddedGraphBackend",
    "InMemoryGraphBackend",
    "StubGraphBackend",
    "build_backend",
]
