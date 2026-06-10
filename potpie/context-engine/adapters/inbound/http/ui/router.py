"""Read-only JSON API for the local graph-explorer UI.

Every route resolves a pot (explicit ``?pot=`` or the active pot) and delegates
to a ``HostShell`` surface. Nothing here mutates the graph — the UI is a
browse/select surface, in keeping with the "harness is the intelligence" model.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from domain.errors import CapabilityNotImplemented, PotNotFound
from domain.graph_entity_summary import normalize_entity_properties
from domain.ports.claim_query import ClaimQueryFilter
from domain.ports.services.graph_service import (
    GraphCatalogRequest,
    GraphEntitySearchRequest,
    GraphReadRequest,
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


def _resolve_pot(host: Any, pot: str | None) -> str:
    """Explicit ``pot`` ref → id, else the active pot. 400 if neither resolves."""
    pots = host.pots
    if pot:
        for p in pots.list_pots():
            if pot in (p.pot_id, p.name):
                return p.pot_id
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
    if expected and expected in labels:
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
        edges.append(
            {
                "id": f"{e.from_key}|{e.predicate}|{e.to_key}",
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


def build_ui_api_router(host: Any) -> APIRouter:
    """Build the ``/ui/api`` router bound to a concrete in-process ``host``."""
    router = APIRouter()

    def _guarded(fn):
        # Map domain errors to HTTP so the SPA gets a clean JSON error body.
        try:
            return fn()
        except HTTPException:
            raise
        except CapabilityNotImplemented as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except PotNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/pots")
    def list_pots() -> dict[str, Any]:
        def go():
            pots = host.pots.list_pots()
            active = host.pots.active_pot()
            return {
                "pots": [
                    {"id": p.pot_id, "name": p.name, "active": bool(p.active)}
                    for p in pots
                ],
                "active": (
                    {"id": active.pot_id, "name": active.name} if active else None
                ),
            }

        return _guarded(go)

    @router.post("/api/pots/use")
    def use_pot(ref: str = Body(..., embed=True)) -> dict[str, Any]:
        def go():
            pot = host.pots.use_pot(ref=ref)
            return {"id": pot.pot_id, "name": pot.name, "active": True}

        return _guarded(go)

    @router.get("/api/catalog")
    def catalog(pot: str | None = Query(None)) -> dict[str, Any]:
        def go():
            pot_id = _resolve_pot(host, pot)
            return host.graph.catalog(GraphCatalogRequest(pot_id=pot_id)).to_dict()

        return _guarded(go)

    @router.get("/api/status")
    def status(pot: str | None = Query(None)) -> dict[str, Any]:
        def go():
            pot_id = _resolve_pot(host, pot)
            dp = host.graph.data_plane_status(pot_id)
            return {
                "pot_id": pot_id,
                "backend_profile": dp.backend_profile,
                "backend_ready": bool(dp.backend_ready),
                "counts": dict(dp.counts),
            }

        return _guarded(go)

    @router.get("/api/search")
    def search(
        q: str = Query(...),
        type: str | None = Query(None),
        predicate: str | None = Query(None),
        environment: str | None = Query(None),
        limit: int = Query(15, ge=1, le=100),
        pot: str | None = Query(None),
    ) -> dict[str, Any]:
        def go():
            pot_id = _resolve_pot(host, pot)
            result = host.graph.search_entities(
                GraphEntitySearchRequest(
                    pot_id=pot_id,
                    query=q,
                    type=type,
                    predicate=predicate,
                    environment=environment,
                    limit=limit,
                )
            )
            return result.to_dict()

        return _guarded(go)

    @router.get("/api/graph")
    def whole_graph(
        pot: str | None = Query(None),
        include_invalid: bool = Query(False),
    ) -> dict[str, Any]:
        def go():
            pot_id = _resolve_pot(host, pot)
            sl = host.backend.inspection.slice(
                pot_id=pot_id,
                filter_=ClaimQueryFilter(
                    pot_id=pot_id, include_invalidated=include_invalid
                ),
            )
            return {"pot_id": pot_id, **_slice_to_graph(sl)}

        return _guarded(go)

    @router.get("/api/neighborhood")
    def neighborhood(
        key: str = Query(...),
        depth: int = Query(1, ge=1, le=4),
        pot: str | None = Query(None),
    ) -> dict[str, Any]:
        def go():
            pot_id = _resolve_pot(host, pot)
            sl = host.backend.inspection.neighborhood(
                pot_id=pot_id, entity_key=key, depth=depth
            )
            return {"pot_id": pot_id, **_slice_to_graph(sl)}

        return _guarded(go)

    @router.get("/api/read")
    def read_view(
        view: str = Query(...),
        query: str | None = Query(None),
        scope: str | None = Query(None),
        environment: str | None = Query(None),
        depth: int | None = Query(None),
        direction: str | None = Query(None),
        limit: int = Query(12, ge=1, le=100),
        pot: str | None = Query(None),
    ) -> dict[str, Any]:
        def go():
            pot_id = _resolve_pot(host, pot)
            env = host.graph.read(
                GraphReadRequest(
                    pot_id=pot_id,
                    view=view,
                    query=query,
                    scope=_parse_scope(scope),
                    environment=environment,
                    depth=depth,
                    direction=direction,
                    limit=limit,
                )
            )
            meta = dict(env.metadata)
            return {
                "view": meta.get("view"),
                "backed": meta.get("backed"),
                "match_mode": meta.get("match_mode"),
                "overall_confidence": env.overall_confidence,
                "items": [
                    {"include": i.include, "score": i.score, "payload": dict(i.payload)}
                    for i in env.items
                ],
                "coverage": [
                    {"include": c.include, "status": c.status} for c in env.coverage
                ],
                "unsupported_includes": [
                    {"name": u.name, "reason": u.reason}
                    for u in env.unsupported_includes
                ],
            }

        return _guarded(go)

    return router


def _parse_scope(scope: str | None) -> dict[str, str]:
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


__all__ = ["build_ui_api_router"]
