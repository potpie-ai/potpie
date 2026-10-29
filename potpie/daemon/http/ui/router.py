"""Read-only JSON API for the graph-explorer UI, for the local and a managed host.

Every route resolves a *host* (``?host=local|managed``, default the CLI's
active origin) and then a pot within it (explicit ``?pot=`` or that host's
active pot), and reads through that host's :class:`ExplorerSource`. Nothing
here mutates the graph -- the UI is a browse/select surface, in keeping with
the "harness is the intelligence" model. Commit history and rollback previews
(``commits.py``) are typed operations; applying a preview is left to the CLI.

The daemon serves this API for both hosts rather than each host serving its
own copy. The browser only ever talks to loopback, so a managed host's key
stays on this machine (the daemon reads it from the host registry) instead of
riding in a page or a URL the browser would keep in its history.

Every route is behind ``require_ui_credential`` and ``require_same_origin``:
it reads the same graphs the typed protocol does, and proxying a managed host
means an anonymous caller would otherwise be spending the stored remote key.
See ``auth.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from potpie.daemon.http.ui.auth import (
    require_bearer,
    require_same_origin,
    require_ui_credential,
    ui_auth,
)
from potpie.daemon.http.ui.commits import (
    EngineClientFactory,
    boundary_http_error,
    build_commit_router,
)
from potpie.daemon.http.ui.source import (
    ExplorerSource,
    ExplorerSourceError,
    LocalExplorerSource,
    ManagedExplorerSource,
)
from potpie.pots.resolution import archived_pot_message, is_archived, match_pot_ref

from potpie_context_engine.core.errors import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotArchived,
    PotNotFound,
)
from potpie_context_engine.core.graph_entity_summary import (
    normalize_entity_properties,
)

# Labels that carry no display meaning (every node has the base :Entity label).
_BASE_LABELS = {"Entity"}

# Authoritative entity-key prefix → type label (the V1.5 ontology identity
# policy). Preferred over node labels for display, since labels can accumulate
# on a node (e.g. an entity touched by more than one upsert) whereas the key
# prefix is canonical.
_PREFIX_LABEL = {
    "repo": "Repository",
    "service": "Service",
    "environment": "Environment",
    "datastore": "DataStore",
    "cluster": "Cluster",
    "dependency": "Dependency",
    "api_contract": "APIContract",
    "team": "Team",
    "person": "Person",
    "activity": "Activity",
    "period": "Period",
    "preference": "Preference",
    "policy": "Policy",
    "bug_pattern": "BugPattern",
    "fix": "Fix",
    "decision": "Decision",
    "document": "Document",
}

#: Builds the managed host's source per request; raises when none is usable.
ManagedSourceFactory = Callable[[], ExplorerSource]


def _resolve_pot(pots: Any, pot: str | None) -> str:
    """Explicit ``pot`` ref → id, else the active pot. 4xx if neither resolves."""
    if pot:
        live, archived = match_pot_ref(pots.list_pots(), pot)
        if live is not None:
            return live.pot_id
        if archived is not None:
            raise HTTPException(status_code=409, detail=archived_pot_message(archived))
        raise HTTPException(status_code=404, detail=f"no pot matching {pot!r}")
    active = pots.active_pot()
    if active is None:
        raise HTTPException(status_code=409, detail="no active pot")
    return active.pot_id


def _node_type(key: str, labels: tuple[str, ...] | list[str]) -> str:
    # Canonical key prefix wins (e.g. ``activity:github:pr-848`` → Activity),
    # even if the node also carries other labels.
    prefix = key.split(":", 1)[0] if ":" in key else ""
    expected = _PREFIX_LABEL.get(prefix)
    if expected:
        return expected
    for lbl in labels:
        if lbl not in _BASE_LABELS:
            return lbl
    return expected or "Entity"


def _caption(key: str, props: dict[str, Any]) -> str:
    for field in ("summary", "title", "name", "description"):
        val = props.get(field)
        if isinstance(val, str) and val.strip():
            text = val.strip()
            return text if len(text) <= 80 else f"{text[:77].rstrip()}..."
    # else the most specific part of the canonical key
    tail = key.split(":")[-1] if ":" in key else key
    return tail or key


def _slice_to_graph(sl: Any) -> dict[str, Any]:
    nodes = []
    for n in sl.nodes:
        labels = list(n.labels)
        props = normalize_entity_properties(dict(n.properties), entity_key=n.key)
        nodes.append(
            {
                "id": n.key,
                "key": n.key,
                "labels": labels,
                "type": _node_type(n.key, tuple(labels)),
                "caption": _caption(n.key, props),
                "summary": props.get("summary") or props.get("description") or "",
                "properties": props,
            }
        )
    edges = []
    for e in sl.edges:
        props = dict(getattr(e, "properties", None) or {})
        # Two claims can share endpoints and predicate; the explorer selects and
        # diffs edges by id, so prefer the record identity when there is one.
        edge_id = props.get("record_id") or props.get("uuid")
        if not edge_id:
            edge_id = f"{e.from_key}|{e.predicate}|{e.to_key}"
            if props.get("claim_key"):
                edge_id += f"|{props['claim_key']}"
        edges.append(
            {
                "id": str(edge_id),
                "source": e.from_key,
                "target": e.to_key,
                "predicate": e.predicate,
            }
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "truncated": bool(getattr(sl, "truncated", False)),
    }


def _default_origin() -> str:
    from potpie.config import host_registry

    return host_registry.current_origin()


def _listed_origins() -> tuple[str, ...]:
    """Origins worth listing: local always, managed once one is configured."""
    from potpie.config import host_registry

    return host_registry.configured_origins()


def build_ui_api_router(
    *,
    pots: Any,
    graph: Any,
    backend: Any,
    engine_client: EngineClientFactory | None = None,
    managed_source: ManagedSourceFactory | None = None,
) -> APIRouter:
    """Build the ``/ui/api`` router from explicit runtime services.

    ``pots``/``graph``/``backend``/``engine_client`` are this daemon's own
    services (the local source); ``managed_source`` builds the managed host's
    source per request and defaults to the host registry's managed host.

    The credential and origin checks are router-level dependencies rather than
    per-route ones, so a route added here later is authenticated by
    construction, not by whoever remembers to add the decorator.
    """
    router = APIRouter(
        dependencies=[Depends(require_ui_credential), Depends(require_same_origin)]
    )
    local = LocalExplorerSource(
        pots=pots, graph=graph, backend=backend, engine_client=engine_client
    )
    build_managed = managed_source or ManagedExplorerSource.from_registry

    def source_for(origin: str | None) -> tuple[ExplorerSource, str]:
        """``(source, origin)`` for a ``?host=`` value, defaulting to the active one."""
        name = (origin or "").strip().lower() or _default_origin()
        if name == "local":
            return local, "local"
        if name != "managed":
            raise HTTPException(status_code=400, detail=f"unknown host {name!r}")
        try:
            return build_managed(), "managed"
        except Exception as exc:  # noqa: BLE001 - unconfigured or unusable
            raise HTTPException(status_code=503, detail=_detail(exc)) from exc

    async def select(
        origin: str | None, pot: str | None
    ) -> tuple[ExplorerSource, str, str]:
        """``(source, origin, pot_id)``: the pot resolved on the host that owns it."""
        source, resolved = source_for(origin)
        pot_id = await asyncio.to_thread(_resolve_pot, source.pots, pot)
        return source, resolved, pot_id

    async def guarded(fn: Callable[[], Awaitable[Any]]) -> Any:
        # Map domain errors to HTTP so the SPA gets a clean JSON error body.
        try:
            return await fn()
        except HTTPException:
            raise
        except ExplorerSourceError as exc:
            raise boundary_http_error(exc.error) from exc
        except CapabilityNotImplemented as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except PotArchived as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PotNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ContextEngineDisabled as exc:
            # A managed host that stops answering mid-read: "that host is
            # down", not "the explorer is broken".
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            remote_error = _remote_host_error(exc)
            if remote_error is None:
                raise
            raise boundary_http_error(remote_error) from exc

    @router.get("/api/pots")
    async def list_pots() -> dict[str, Any]:
        async def go():
            active_origin = _default_origin()
            rows: list[dict[str, Any]] = []
            unavailable: dict[str, str] = {}
            counts_complete = True
            active: dict[str, str] | None = None
            for origin in _listed_origins():
                try:
                    scoped, _ = source_for(origin)
                    listed, active_id = await asyncio.to_thread(_listing, scoped.pots)
                except Exception as exc:  # noqa: BLE001
                    # One unreachable host must not blank the selector: the
                    # pots you can still open are the useful part of the answer.
                    unavailable[origin] = _detail(exc)
                    continue
                # Archived pots cannot be selected, so the picker omits them.
                live = [p for p in listed if not is_archived(p)]
                try:
                    counts = await scoped.pot_counts(live)
                except Exception as exc:  # noqa: BLE001
                    # Counts are a browsing aid; a pot without them is listed
                    # without them, never shown as empty.
                    counts = {}
                    unavailable[f"{origin} counts"] = _detail(exc)
                for p in live:
                    is_active = bool(p.active) or p.pot_id == active_id
                    row: dict[str, Any] = {
                        "id": p.pot_id,
                        "name": p.name,
                        "origin": origin,
                        "active": is_active,
                    }
                    if p.pot_id in counts:
                        row.update(counts[p.pot_id])
                    else:
                        counts_complete = False
                    rows.append(row)
                    # Each host keeps its own pointer; the one that counts as
                    # "the" active pot belongs to the active host.
                    if is_active and origin == active_origin:
                        active = {"id": p.pot_id, "name": p.name, "origin": origin}
            return {
                "pots": rows,
                "active": active,
                "active_origin": active_origin,
                "unavailable": unavailable,
                "counts_complete": counts_complete,
            }

        return await guarded(go)

    @router.post("/api/handoff", dependencies=[Depends(require_bearer)])
    def handoff(request: Request) -> dict[str, Any]:
        """Trade the daemon token for a code a browser navigation can carry.

        ``potpie ui`` calls this and puts the code in the URL it opens; the
        shell handler spends it for a cookie. The token itself never reaches
        the browser, where it would sit in history for the life of the profile.
        """
        code, expires_in = ui_auth(request).mint_code()
        return {"code": code, "expires_in": expires_in}

    @router.post("/api/pots/use")
    async def use_pot(
        ref: str = Body(..., embed=True),
        origin: str | None = Body(None, embed=True, alias="host"),
    ) -> dict[str, Any]:
        async def go():
            from potpie.config import host_registry

            # A qualified ``managed:api`` carries its own host.
            qualified, bare = host_registry.split_ref(ref)
            scoped, resolved = source_for(qualified or origin)
            pot = await asyncio.to_thread(
                scoped.pots.use_pot, ref=bare if qualified else ref
            )
            # Persisted only after the host accepted it, so a failed switch
            # never strands the CLI on a host nobody chose. The same pointer
            # `potpie pot use` writes: picking a pot here moves the terminal.
            if host_registry.persisted_origin() != resolved:
                host_registry.set_persisted_origin(resolved)
            return {
                "id": pot.pot_id,
                "name": pot.name,
                "origin": resolved,
                "active": True,
            }

        return await guarded(go)

    @router.get("/api/catalog")
    async def catalog(
        pot: str | None = Query(None),
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        async def go():
            source, _, pot_id = await select(origin, pot)
            return (await source.catalog(pot_id)).to_dict()

        return await guarded(go)

    @router.get("/api/status")
    async def status(
        pot: str | None = Query(None),
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        async def go():
            source, resolved, pot_id = await select(origin, pot)
            dp = await source.data_plane_status(pot_id)
            return {
                "pot_id": pot_id,
                "origin": resolved,
                "backend_profile": dp.backend_profile,
                "backend_ready": bool(dp.backend_ready),
                "counts": dict(dp.counts),
            }

        return await guarded(go)

    @router.get("/api/search")
    async def search(
        q: str = Query(...),
        entity_type: str | None = Query(None, alias="type"),
        predicate: str | None = Query(None),
        subgraph: str | None = Query(None),
        scope: str | None = Query(None),
        truth: str | None = Query(None),
        environment: str | None = Query(None),
        external_id: str | None = Query(None),
        limit: int = Query(15, ge=1, le=100),
        pot: str | None = Query(None),
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        async def go():
            source, _, pot_id = await select(origin, pot)
            result = await source.search_entities(
                pot_id,
                {
                    "query": q,
                    "type": entity_type,
                    "predicate": predicate,
                    "subgraph": subgraph,
                    "scope": _parse_scope(scope),
                    "truth": truth,
                    "environment": environment,
                    "external_id": external_id,
                    "limit": limit,
                    "supporting_claims": 5,
                },
            )
            return result.to_dict()

        return await guarded(go)

    @router.get("/api/graph")
    async def whole_graph(
        pot: str | None = Query(None),
        include_invalid: bool = Query(False),
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        async def go():
            source, _, pot_id = await select(origin, pot)
            sl = await source.graph_slice(pot_id, include_invalidated=include_invalid)
            return {"pot_id": pot_id, **_slice_to_graph(sl)}

        return await guarded(go)

    @router.get("/api/neighborhood")
    async def neighborhood(
        key: str = Query(...),
        depth: int = Query(1, ge=1, le=4),
        pot: str | None = Query(None),
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        async def go():
            source, _, pot_id = await select(origin, pot)
            sl = await source.neighborhood(pot_id, key=key, depth=depth)
            return {"pot_id": pot_id, **_slice_to_graph(sl)}

        return await guarded(go)

    @router.get("/api/read")
    async def read_view(
        subgraph: str = Query(...),
        view: str = Query(...),
        query: str | None = Query(None),
        scope: str | None = Query(None),
        environment: str | None = Query(None),
        depth: int | None = Query(None),
        direction: str | None = Query(None),
        limit: int = Query(12, ge=1, le=100),
        pot: str | None = Query(None),
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        async def go():
            source, _, pot_id = await select(origin, pot)
            env = await source.read(
                pot_id,
                {
                    "subgraph": subgraph,
                    "view": view,
                    "query": query,
                    "scope": _parse_scope(scope),
                    "environment": environment,
                    "depth": depth,
                    "direction": direction,
                    "limit": limit,
                    "detail": "full",
                    "relations": "full",
                },
            )
            return env.to_dict()

        return await guarded(go)

    @router.post("/api/telemetry/session")
    def ui_session(
        payload: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, bool]:
        """Record that the SPA finished its initial pots/graph load.

        The CLI ``potpie ui`` probe hits ``GET /ui/api/pots`` only. This POST is
        the Q11 signal: the explorer actually loaded in a browser.
        """
        try:
            from potpie.cli.telemetry.usage_events import (
                capture_usage_command_succeeded,
            )

            properties: dict[str, bool] = {}
            if isinstance(payload, dict) and isinstance(payload.get("had_graph"), bool):
                properties["had_graph"] = payload["had_graph"]
            capture_usage_command_succeeded(
                command="ui",
                result_kind="ui_session",
                properties=dict(properties) if properties else None,
            )
        except Exception:  # noqa: BLE001, S110 — analytics must never fail the SPA
            pass
        return {"ok": True}

    async def select_for_commits(
        origin: str | None, pot: str | None
    ) -> tuple[ExplorerSource, str]:
        source, _, pot_id = await select(origin, pot)
        return source, pot_id

    router.include_router(
        build_commit_router(select=select_for_commits, guarded=guarded)
    )
    return router


def _listing(pots: Any) -> tuple[list[Any], str | None]:
    listed = list(pots.list_pots())
    active = pots.active_pot()
    return listed, getattr(active, "pot_id", None) if active is not None else None


def _remote_host_error(exc: BaseException) -> object | None:
    import sys

    module = sys.modules.get("potpie.runtime.remote")
    remote_error = getattr(module, "RemoteHostError", None) if module else None
    if isinstance(remote_error, type) and isinstance(exc, remote_error):
        return exc.error  # type: ignore[attr-defined]
    return None


def _detail(exc: BaseException) -> str:
    """Readable reason for a host that could not be listed or built."""
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    remote_error = _remote_host_error(exc)
    if remote_error is not None:
        return str(getattr(remote_error, "message", exc))
    return str(exc) or exc.__class__.__name__


def _parse_scope(scope: str | None) -> Mapping[str, str]:
    if not scope:
        return {}
    out: dict[str, str] = {}
    for pair in scope.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, value = pair.split(":", 1)
        if key.strip() and value.strip():
            out[key.strip()] = value.strip()
    return out


__all__ = ["ManagedSourceFactory", "build_ui_api_router"]
