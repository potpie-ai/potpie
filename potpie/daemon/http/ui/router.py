"""Read-only JSON API for the graph-explorer UI.

Every route resolves a *host* (``?host=local|managed``) and then a pot within it
(explicit ``?pot=`` or that host's active pot) and delegates to a ``HostShell``
surface. Nothing here mutates the graph — the UI is a browse/select surface, in
keeping with the "harness is the intelligence" model.

The daemon serves this API for **both** hosts rather than each host serving its
own copy. The browser only ever talks to loopback, so a managed host's token
stays on this machine instead of riding in a URL the browser would keep in its
history — and the explorer works against a managed service that serves only the
RPC surface and no SPA of its own.

Every route here is behind ``require_ui_credential``: it reads the same graph
``/rpc`` does, and proxying a managed host means an unauthenticated caller would
otherwise be spending the daemon's stored remote token. See ``auth.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response

from potpie.daemon.http.ui.auth import (
    require_bearer,
    require_same_origin,
    require_ui_credential,
    ui_auth,
)

from potpie_context_core.errors import (
    CapabilityNotImplemented,
    ContextEngineDisabled,
    PotNotFound,
)
from potpie_context_core.graph_entity_summary import (
    normalize_entity_properties,
)
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.ports.graph_service import (
    GraphCatalogRequest,
    GraphEntitySearchRequest,
    GraphReadRequest,
)
from potpie_context_core.workbench_service import normalize_catalog_result

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
    "docsection": "DocumentSection",
}


#: How many pots on a *remote* host get their claim/source counts filled in for
#: the selector. Each one costs two RPC round trips, so an account with a few
#: hundred pots turned a dropdown into seconds of blocking network. The
#: in-process host is exempt: there the same calls are local reads.
#:
#: The counts are a browsing aid, not the data — whichever pot actually gets
#: opened is enriched by ``/api/status``, which is a single call for one pot.
_REMOTE_COUNT_LIMIT: int = 25

#: HTTP for a workbench body that came back ``ok: false``, keyed by its
#: ``status``. This is the CLI's exit-code table (``exit_code_for``) said over
#: HTTP, so the explorer and the terminal classify the same refusal the same
#: way; anything unclassified is a validation failure, because a code nobody has
#: triaged is not grounds for claiming a dependency is down.
_FAILURE_STATUS: dict[str, int] = {
    "unavailable": 503,
    "degraded": 503,
    "not_implemented": 501,
}


def _origins() -> tuple[str, ...]:
    """Origins worth listing: local always, managed once one is configured."""
    from potpie.cli import hosts

    return hosts.configured_origins()


def _default_origin() -> str:
    from potpie.cli import hosts

    return hosts.current_origin()


def _host_for(local_host: Any, origin: str | None) -> tuple[Any, str]:
    """``(host, origin)`` for a ``?host=`` value, defaulting to the active one.

    ``local`` is the in-process host this router was built with — never a host
    rebuilt from the registry, which for the local origin is an RPC client
    aimed at this very daemon and would have us call ourselves.
    """
    from potpie.cli import hosts

    name = (origin or "").strip().lower() or _default_origin()
    if name == hosts.LOCAL:
        return local_host, hosts.LOCAL
    if name != hosts.MANAGED:
        raise HTTPException(status_code=400, detail=f"unknown host {name!r}")
    try:
        return hosts.build_host(hosts.MANAGED), hosts.MANAGED
    except Exception as exc:  # noqa: BLE001 - unconfigured or unreachable
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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


def _usable_counts(dp: Any, pot_id: str) -> dict[str, int]:
    """The numeric counts a ``DataPlaneStatus`` actually carries.

    Raises rather than returning an empty mapping, however it came to be empty:
    ``{}`` is truthy in JS, so the selector rendered "0 claims" for a pot holding
    three and the caller had no way to tell a broken backend from an empty pot.
    Callers omit the key instead, which is what the SPA already renders for a pot
    past the remote count budget.

    An empty result is never "this pot is empty": a healthy backend reports
    ``claims: 0`` for an empty pot, and ``DataPlaneStatus`` swallows a failing
    analytics call into ``{}``. So nothing usable here means the backend could
    not say — which is exactly what the caller must not claim to know.

    Takes the status object rather than fetching one so a caller that already
    has it — ``/api/status`` — is not charged a second round trip, which against
    a managed host is a second RPC call over the network.
    """
    out: dict[str, int] = {}
    for key, value in dict(getattr(dp, "counts", None) or {}).items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    if not out:
        raise ContextEngineDisabled(
            f"backend reported no usable counts for pot {pot_id}"
        )
    return out


def _counts(host: Any, pot_id: str) -> dict[str, int]:
    """Claim/entity counts for one pot; raises if the host cannot say."""
    return _usable_counts(host.graph.data_plane_status(pot_id), pot_id)


def _source_count(host: Any, pot_id: str) -> int:
    """Registered sources for one pot; raises if the host cannot say."""
    return len(host.pots.list_sources(pot_id=pot_id))


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
    """Build the ``/ui/api`` router bound to a concrete in-process ``host``.

    The credential is a router-level dependency rather than a per-route one so
    a route added here later is authenticated by construction, not by whoever
    remembers to add the decorator.
    """
    router = APIRouter(dependencies=[Depends(require_ui_credential)])

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
        except ContextEngineDisabled as exc:
            # A managed host that stops answering mid-read. Building the client
            # opens no connection, so the failure surfaces here rather than in
            # `_host_for` — without this it escaped as a 500 and read to the SPA
            # as "the explorer is broken" instead of "that host is down".
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OSError as exc:
            # The control-plane state on disk failing mid-write — a pot switch
            # racing another writer, a full or read-only home. Left unmapped it
            # escaped the router and Starlette answered a plain-text "Internal
            # Server Error", which the SPA parses as JSON, fails to, and shows
            # as nothing at all.
            raise HTTPException(status_code=500, detail=_detail(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/pots")
    def list_pots() -> dict[str, Any]:
        def go():
            active_origin = _default_origin()
            rows: list[dict[str, Any]] = []
            unavailable: dict[str, str] = {}
            counts_complete = True
            active: dict[str, str] | None = None

            for origin in _origins():
                try:
                    scoped, _ = _host_for(host, origin)
                    pots = scoped.pots.list_pots()
                except Exception as exc:  # noqa: BLE001
                    # One unreachable host must not blank the selector: the
                    # pots you can still open are the useful part of the answer.
                    unavailable[origin] = _detail(exc)
                    continue
                budget = None if scoped is host else _REMOTE_COUNT_LIMIT
                for index, p in enumerate(pots):
                    row = {
                        "id": p.pot_id,
                        "name": p.name,
                        "origin": origin,
                        "active": bool(p.active),
                    }
                    # Counts are omitted rather than zeroed past the budget: a
                    # zero here would read as "this pot is empty" and send you
                    # to the wrong pot, which is worse than no number at all.
                    if budget is None or index < budget:
                        try:
                            row["source_count"] = _source_count(scoped, p.pot_id)
                            row["counts"] = _counts(scoped, p.pot_id)
                        except Exception as exc:  # noqa: BLE001
                            # Same rule as the budget: a pot whose counts could
                            # not be read is listed without them and the reason
                            # is stated, rather than silently shown as empty.
                            counts_complete = False
                            unavailable.setdefault(f"{origin} counts", _detail(exc))
                    else:
                        counts_complete = False
                    rows.append(row)
                    # Each host keeps its own pointer; the one that counts as
                    # "the" active pot belongs to the active host.
                    if p.active and origin == active_origin:
                        active = {
                            "id": p.pot_id,
                            "name": p.name,
                            "origin": origin,
                        }

            return {
                "pots": rows,
                "active": active,
                "active_origin": active_origin,
                "unavailable": unavailable,
                "counts_complete": counts_complete,
            }

        return _guarded(go)

    @router.post("/api/handoff", dependencies=[Depends(require_bearer)])
    def handoff(request: Request) -> dict[str, Any]:
        """Trade the daemon token for a code a browser navigation can carry.

        ``potpie ui`` calls this and puts the code in the URL it opens; the page
        handler spends it for a cookie. The token itself never reaches the
        browser, where it would sit in history for the life of the profile.
        """
        code, expires_in = ui_auth(request).mint_code()
        return {"code": code, "expires_in": expires_in}

    # The one route here that writes. `require_same_origin` is the belt to
    # SameSite's braces: a page in your browser must not be able to move the
    # active pot — and with it the active *host* — out from under the CLI.
    @router.post("/api/pots/use", dependencies=[Depends(require_same_origin)])
    def use_pot(
        ref: str = Body(..., embed=True),
        origin: str | None = Body(None, embed=True, alias="host"),
    ) -> dict[str, Any]:
        def go():
            from potpie.cli import hosts

            # A qualified ``managed:api`` carries its own host, so the selector
            # can hand back exactly what it was given.
            qualified, bare = hosts.split_ref(ref)
            scoped, resolved = _host_for(host, qualified or origin)
            pot = scoped.pots.use_pot(ref=bare if qualified else ref)
            # Persist only after the host accepted it, so a failed switch never
            # strands the CLI on a host nobody chose. This is the same pointer
            # `potpie pot use` writes: picking a pot in the explorer moves the
            # terminal with it, which is the whole point of one registry.
            hosts.set_persisted_origin(resolved)
            return {
                "id": pot.pot_id,
                "name": pot.name,
                "origin": resolved,
                "active": True,
            }

        return _guarded(go)

    @router.get("/api/catalog")
    def catalog(
        pot: str | None = Query(None),
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        def go():
            scoped, _ = _host_for(host, origin)
            pot_id = _resolve_pot(scoped, pot)
            result = scoped.graph.catalog(GraphCatalogRequest(pot_id=pot_id))
            # The same projection ``potpie graph catalog`` serves. Handing back
            # the raw data-plane body advertised the four V1.5 commands where
            # the CLI advertises the twelve-command V2 workbench, so the
            # explorer and the terminal disagreed about what the graph even
            # supports — and the catalog is the one route whose whole job is to
            # be the contract. ``ok`` is restored on top because every other
            # body this router returns carries it and the normalizer drops it
            # for the CLI's envelope, which has its own.
            return {"ok": True, **normalize_catalog_result(result.to_dict())}

        return _guarded(go)

    @router.get("/api/status")
    def status(
        pot: str | None = Query(None),
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        """The header numbers for one pot — or a refusal, never a zero.

        The listing route can *omit* counts it does not have, because the SPA
        renders a row without them. This one cannot: the header reads
        ``counts.entities ?? 0`` / ``counts.claims ?? 0``, so a missing key, a
        null and an empty mapping all render identically to a real zero — "0
        entities / 0 claims" beside ``backend_ready: true``, for a pot that may
        hold thousands. So when the counts are unusable the whole response is
        refused: ``_guarded`` turns that into a 503 the SPA shows as an error
        naming the pot, which is the only shape that does not put a number the
        backend never gave in front of someone choosing where to look.
        """

        def go():
            scoped, resolved = _host_for(host, origin)
            pot_id = _resolve_pot(scoped, pot)
            dp = scoped.graph.data_plane_status(pot_id)
            # Before the dict is built: a partial status with confident-looking
            # readiness is the half-answer this route exists to stop shipping.
            counts = _usable_counts(dp, pot_id)
            return {
                "pot_id": pot_id,
                "origin": resolved,
                "backend_profile": dp.backend_profile,
                "backend_ready": bool(dp.backend_ready),
                "counts": counts,
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
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        def go():
            scoped, _ = _host_for(host, origin)
            pot_id = _resolve_pot(scoped, pot)
            result = scoped.graph.search_entities(
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
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        def go():
            scoped, _ = _host_for(host, origin)
            pot_id = _resolve_pot(scoped, pot)
            sl = scoped.backend.inspection.slice(
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
        origin: str | None = Query(None, alias="host"),
    ) -> dict[str, Any]:
        def go():
            scoped, _ = _host_for(host, origin)
            pot_id = _resolve_pot(scoped, pot)
            sl = scoped.backend.inspection.neighborhood(
                pot_id=pot_id, entity_key=key, depth=depth
            )
            return {"pot_id": pot_id, **_slice_to_graph(sl)}

        return _guarded(go)

    @router.get("/api/read")
    def read_view(
        response: Response,
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
        def go():
            scoped, _ = _host_for(host, origin)
            pot_id = _resolve_pot(scoped, pot)
            env = scoped.graph.read(
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
            body = env.to_dict()
            # A read the workbench *refused* — a view whose required scope was
            # not given, say — is not a 200. A client that checks the status
            # line and nothing else (``jget`` in the SPA, curl, anything
            # scripting this API) rendered the refusal as an empty graph, which
            # reads as "this pot holds nothing" rather than "you did not say
            # which repo". The whole body still travels: ``unsupported`` names
            # the filter to fix, and ``detail`` is where every other failure on
            # this router puts its message.
            if body.get("ok", True) is False:
                response.status_code = _FAILURE_STATUS.get(
                    str(body.get("status") or ""), 400
                )
                body["detail"] = body.get("message") or "graph read failed"
            return body

        return _guarded(go)

    return router


def _detail(exc: Exception) -> str:
    """Readable reason for a host that could not be listed.

    ``_host_for`` has already wrapped registry failures in ``HTTPException``,
    whose ``str()`` is the status line rather than the cause, so the detail is
    unwrapped here — "503: connection refused" tells the user nothing that
    "connection refused" does not.
    """
    if isinstance(exc, HTTPException):
        return str(exc.detail)
    return str(exc) or exc.__class__.__name__


def _parse_scope(scope: str | None) -> dict[str, str]:
    """``key:value[,key:value]`` → mapping, refusing anything else.

    Same contract, and the same three refusals, as ``--scope`` on the CLI
    (``potpie.cli.commands.graph._parse_scope``); ``_guarded`` turns the
    ``ValueError`` into a 400.

    Dropping a malformed pair — which is what this did — is the worst available
    answer. One typo beside one good pair produced a *confident* result computed
    against a narrower filter, with the constraint the caller asked for missing
    from the query and from the response alike, so nothing on either end could
    tell that half the scope had evaporated. A trailing comma is still fine, as
    on the CLI: an empty entry is not a malformed one.
    """
    if not scope:
        return {}
    out: dict[str, str] = {}
    for entry in scope.split(","):
        pair = entry.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"invalid scope entry {pair!r}; expected key:value pairs")
        key, value = pair.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(
                f"invalid scope entry {pair!r}; scope keys must not be empty"
            )
        if not value:
            raise ValueError(
                f"invalid scope entry {pair!r}; scope values must not be empty"
            )
        out[key] = value
    return out


__all__ = ["build_ui_api_router"]
