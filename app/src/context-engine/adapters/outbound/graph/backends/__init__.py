"""``GraphBackend`` profiles + the ``build_backend`` registry.

One registry maps a profile name to a concrete ``GraphBackend``. Adding a
storage option means adding a profile here and implementing the six capability
ports behind it — never changing the services that depend on ``GraphBackend``.

    in_memory   real, tests + conformance + POC          (InMemoryGraphBackend)
    embedded    OSS local default (real, JSON-persisted)  (EmbeddedGraphBackend)
    neo4j       shape-first production target            (Neo4jGraphBackend)
    falkor_lite embedded FalkorDBLite, no server/Docker   (FalkorGraphBackend)
    falkor      server-mode FalkorDB (needs FALKORDB_URL) (FalkorGraphBackend)
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

from adapters.outbound.graph.backends.embedded_backend import EmbeddedGraphBackend
from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from adapters.outbound.graph.backends.stub_backend import StubGraphBackend
from domain.ports.graph.backend import GraphBackend

KNOWN_PROFILES: tuple[str, ...] = (
    "in_memory",
    "embedded",
    "neo4j",
    "falkor_lite",
    "falkor",
    "postgres",
    "chroma",
    "hosted",
)

# Documented profiles routed to a fail-closed StubGraphBackend until built.
_STUB_PROFILES: frozenset[str] = frozenset({"postgres", "chroma", "hosted"})


def _default_settings() -> Any:
    from adapters.outbound.settings_env import EnvContextEngineSettings

    return EnvContextEngineSettings()


def build_backend(profile: str, *, settings: Any = None) -> GraphBackend:
    """Construct the ``GraphBackend`` for a profile name.

    ``in_memory`` and ``embedded`` need no settings; ``neo4j`` and the FalkorDB
    profiles need the engine settings (lazy-imported so the neo4j / falkordb
    drivers stay optional). ``falkor_lite`` is the embedded, no-server profile;
    ``falkor`` is server-mode (needs ``FALKORDB_URL``). ``postgres`` / ``chroma``
    / ``hosted`` resolve to a fail-closed ``StubGraphBackend``. Unknown profiles
    raise ``ValueError`` — the CLI maps that to a validation error.
    """
    name = (profile or "").strip().lower()
    if name == "in_memory":
        return InMemoryGraphBackend()
    if name == "embedded":
        return EmbeddedGraphBackend()
    if name == "neo4j":
        # Lazy import keeps the neo4j driver optional for other profiles.
        from adapters.outbound.graph.backends.neo4j import Neo4jGraphBackend

        return Neo4jGraphBackend(settings or _default_settings())
    if name in ("falkor_lite", "falkor"):
        # Lazy import keeps the falkordb / redislite clients optional. The mode
        # is fixed by the profile (lite = embedded FalkorDBLite, falkor = server).
        from adapters.outbound.graph.backends.falkor import FalkorGraphBackend

        mode = "lite" if name == "falkor_lite" else "server"
        return FalkorGraphBackend(settings or _default_settings(), mode=mode)
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
