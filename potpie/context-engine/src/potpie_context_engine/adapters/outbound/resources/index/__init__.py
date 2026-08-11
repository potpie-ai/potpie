"""``build_resource_index`` — the profile registry for the retrieval index.

One registry maps a profile name to a concrete ``ResourceIndexPort``, the way
``adapters/outbound/graph/backends`` does for storage. Adding a retrieval
strategy means adding a profile here; it never means changing the reader, the
facade, or the CLI.

    sqlite_hybrid  BM25 + vectors, fused by reciprocal rank   — the default
    sqlite_fts     BM25 only, stdlib sqlite3 alone
    none           nothing; search returns match_mode="disabled"

Selection copies ``default_backend_profile()`` exactly — environment variable,
then the local config file, then a literal default, each ``.strip()``ed so a
blank value falls through rather than selecting an empty profile.

An unknown profile is a ``ValueError``, which the CLI already maps to a
validation error. It is deliberately not a silent fall back to the default: a
typo in ``CONTEXT_ENGINE_RESOURCE_INDEX`` that quietly produced working search
would be discovered only by wondering why a setting had no effect.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from potpie_context_core.ports.resource_index import (
    RESOURCE_INDEX_PROFILE_UNKNOWN,
    ResourceIndexError,
    ResourceIndexPort,
)
from potpie_context_engine.adapters.outbound.resources.index._unimplemented import (
    NullResourceIndex,
)
from potpie_context_engine.adapters.outbound.resources.index.drain import (
    ResourceIndexDrain,
)
from potpie_context_engine.adapters.outbound.resources.index.sqlite_fts import (
    SqliteFtsResourceIndex,
)
from potpie_context_engine.adapters.outbound.resources.index.sqlite_hybrid import (
    SqliteHybridResourceIndex,
)

KNOWN_PROFILES: tuple[str, ...] = ("sqlite_hybrid", "sqlite_fts", "none")

DEFAULT_PROFILE = "sqlite_hybrid"

#: Truthy values that turn on the shared-store honesty note in ``status``. The
#: deployer is the only party that knows whether the resource *files* sit on a
#: volume more than one replica mounts, so it is declared rather than sniffed.
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def default_resource_index_profile() -> str:
    """Env, then local config, then the literal default — each ``strip()``ed."""
    raw = (os.getenv("CONTEXT_ENGINE_RESOURCE_INDEX") or "").strip().lower()
    if raw:
        return raw
    configured = _local_config().get("resource_index")
    if isinstance(configured, str) and configured.strip():
        return configured.strip().lower()
    return DEFAULT_PROFILE


def resource_store_is_shared() -> bool:
    return (
        os.getenv("CONTEXT_ENGINE_RESOURCE_STORE_SHARED") or ""
    ).strip().lower() in _TRUTHY


def build_resource_index(
    profile: str | None = None,
    *,
    home: Path | None = None,
    embedder: Any = None,
    shared_store: bool | None = None,
) -> ResourceIndexPort:
    """Construct the index for a profile name.

    ``embedder`` is an :class:`EmbedderPort`; when omitted the bundled local
    embedder is built, so semantic retrieval needs no API key and no setup step
    (``CONTEXT_ENGINE_EMBEDDER=none`` disables it, and the hybrid profile then
    reports itself lexical rather than pretending).
    """
    name = (profile or default_resource_index_profile()).strip().lower()
    name = name.replace("-", "_")
    if name in {"off", "disabled"}:
        name = "none"
    if name not in KNOWN_PROFILES:
        raise ResourceIndexError(
            RESOURCE_INDEX_PROFILE_UNKNOWN,
            f"unknown resource index profile: {name!r}",
            detail=f"known profiles: {', '.join(KNOWN_PROFILES)}",
            recommended_next_action=(
                "Set CONTEXT_ENGINE_RESOURCE_INDEX to one of "
                f"{', '.join(KNOWN_PROFILES)}."
            ),
        )
    if name == "none":
        return NullResourceIndex(
            detail="resource index profile is 'none'; search returns no chunks"
        )
    shared = resource_store_is_shared() if shared_store is None else shared_store
    kwargs: dict[str, Any] = {"shared_store": shared}
    if home is not None:
        kwargs["home"] = home
    if name == "sqlite_fts":
        # No embedder: the lexical profile must not pull a model into memory,
        # and passing one would make ``capabilities()`` read as a lie.
        return SqliteFtsResourceIndex(**kwargs)
    return SqliteHybridResourceIndex(
        embedder=embedder if embedder is not None else _default_embedder(), **kwargs
    )


def _default_embedder() -> Any:
    from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
        build_embedder,
    )

    return build_embedder()


def _local_config() -> dict[str, Any]:
    import json

    home = (os.getenv("CONTEXT_ENGINE_HOME") or "").strip()
    root = Path(home).expanduser() if home else Path.home() / ".potpie"
    try:
        with open(root / "config.json", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


__all__ = [
    "DEFAULT_PROFILE",
    "KNOWN_PROFILES",
    "NullResourceIndex",
    "ResourceIndexDrain",
    "SqliteFtsResourceIndex",
    "SqliteHybridResourceIndex",
    "build_resource_index",
    "default_resource_index_profile",
    "resource_store_is_shared",
]
