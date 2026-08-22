"""Read-only JSON API for the local graph-explorer UI.

Every route resolves a pot (explicit ``?pot=`` or the active pot) and delegates
to explicit root- and engine-owned services. Nothing here mutates the graph —
the UI is a browse/select surface, in keeping with the "harness is the
intelligence" model.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from potpie_context_engine.core.errors import CapabilityNotImplemented, PotNotFound
from potpie_context_engine.core.graph_entity_summary import (
    normalize_entity_properties,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.ports.graph_service import (
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


def _resolve_pot(pots: Any, pot: str | None) -> str:
    """Explicit ``pot`` ref → id, else the active pot. 400 if neither resolves."""
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


def _counts(graph: Any, pot_id: str) -> dict[str, int]:
    try:
        dp = graph.data_plane_status(pot_id)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, int] = {}
    for key, value in dict(getattr(dp, "counts", {}) or {}).items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _source_count(pots: Any, pot_id: str) -> int:
    try:
        return len(pots.list_sources(pot_id=pot_id))
    except Exception:  # noqa: BLE001
        return 0


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


def build_ui_api_router(*, pots: Any, graph: Any, backend: Any) -> APIRouter:
    """Build the ``/ui/api`` router from explicit runtime services."""
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
            pot_records = pots.list_pots()
            active = pots.active_pot()
            return {
                "pots": [
                    {
                        "id": p.pot_id,
                        "name": p.name,
                        "active": bool(p.active),
                        "source_count": _source_count(pots, p.pot_id),
                        "counts": _counts(graph, p.pot_id),
                    }
                    for p in pot_records
                ],
                "active": (
                    {"id": active.pot_id, "name": active.name} if active else None
                ),
            }

        return _guarded(go)

    @router.post("/api/pots/use")
    def use_pot(ref: str = Body(..., embed=True)) -> dict[str, Any]:
        def go():
            pot = pots.use_pot(ref=ref)
            return {"id": pot.pot_id, "name": pot.name, "active": True}

        return _guarded(go)

    @router.get("/api/catalog")
    def catalog(pot: str | None = Query(None)) -> dict[str, Any]:
        def go():
            pot_id = _resolve_pot(pots, pot)
            return graph.catalog(GraphCatalogRequest(pot_id=pot_id)).to_dict()

        return _guarded(go)

    @router.get("/api/status")
    def status(pot: str | None = Query(None)) -> dict[str, Any]:
        def go():
            pot_id = _resolve_pot(pots, pot)
            dp = graph.data_plane_status(pot_id)
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
        entity_type: str | None = Query(None, alias="type"),
        predicate: str | None = Query(None),
        subgraph: str | None = Query(None),
        scope: str | None = Query(None),
        truth: str | None = Query(None),
        environment: str | None = Query(None),
        external_id: str | None = Query(None),
        limit: int = Query(15, ge=1, le=100),
        pot: str | None = Query(None),
    ) -> dict[str, Any]:
        def go():
            pot_id = _resolve_pot(pots, pot)
            result = graph.search_entities(
                GraphEntitySearchRequest(
                    pot_id=pot_id,
                    query=q,
                    type=entity_type,
                    predicate=predicate,
                    subgraph=subgraph,
                    scope=_parse_scope(scope),
                    truth=truth,
                    environment=environment,
                    external_id=external_id,
                    limit=limit,
                    supporting_claims=5,
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
            pot_id = _resolve_pot(pots, pot)
            sl = backend.inspection.slice(
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
            pot_id = _resolve_pot(pots, pot)
            sl = backend.inspection.neighborhood(
                pot_id=pot_id, entity_key=key, depth=depth
            )
            return {"pot_id": pot_id, **_slice_to_graph(sl)}

        return _guarded(go)

    @router.get("/api/read")
    def read_view(
        subgraph: str = Query(...),
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
            pot_id = _resolve_pot(pots, pot)
            env = graph.read(
                GraphReadRequest(
                    pot_id=pot_id,
                    subgraph=subgraph,
                    view=view,
                    query=query,
                    scope=_parse_scope(scope),
                    environment=environment,
                    depth=depth,
                    direction=direction,
                    limit=limit,
                    detail="full",
                    relations="full",
                )
            )
            return env.to_dict()

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
