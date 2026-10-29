"""Explorer routes for graph commit history, recorded diffs and rollback previews.

These routes are included in the ``/ui/api`` router, so its credential and
same-origin checks apply to every one of them. What a browser session may do
here is read: list commits, read a commit's recorded changes, see journal
status and saved plans, and ask for a rollback preview, which is a dry run the
server holds for a few minutes and which changes nothing. Applying a preview
changes the graph and needs the daemon credential and explicit destructive
confirmation, so it stays on the CLI (``potpie graph apply-preview``); there is
no apply route on this surface.

Each route sends the same typed operation the CLI sends, to the host the pot
lives on (``?host=``): in-process through the daemon's own resource manager
and operation coordinator for the local host, so archived-pot refusal, locking
and commit authorization are identical on both surfaces, and through the
remote client for a managed host.

A managed host older than this client answers ``host_outdated``. That becomes
a structured 501 (``detail.status == "host_outdated"``) the page falls back
on, and saved-plan history retries once without ``include_claims`` when that
optional field is all the host lacks.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from potpie_context_engine.requests import (
    CommitShowRequest,
    CommitsRequest,
    HistoryRequest,
    JournalStatusRequest,
    RevertPreviewRequest,
    RollbackPreviewRequest,
)

#: Builds the typed engine client for one resolved pot id.
EngineClientFactory = Callable[[str], Any]

#: ``(source, pot_id)`` for a ``?host=`` and ``?pot=`` pair.
Select = Callable[[str | None, str | None], Awaitable[tuple[Any, str]]]

HOST_OUTDATED = "host_outdated"

_UNAVAILABLE = {
    "ok": False,
    "status": "unavailable",
    "reasons": [
        {
            "code": "unavailable",
            "message": "Commit history is not available from this explorer.",
        }
    ],
}


class PreviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_commit_id: str
    expected_head: str
    mode: Literal["revert", "rollback"]
    pot: str | None = None
    host: Literal["local", "managed"] | None = None


def build_commit_router(
    *,
    select: Select,
    guarded: Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]] | None = None,
) -> APIRouter:
    router = APIRouter()

    async def call(
        origin: str | None,
        pot: str | None,
        operation: Callable[[Any, Any], Awaitable[Any]],
        request: Any,
        *,
        optional: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Send one typed operation for the resolved pot.

        ``optional`` names request fields that refine the answer: a host too
        old for exactly those is asked again without them rather than losing
        a view it does serve.
        """

        async def run() -> dict[str, Any]:
            source, pot_id = await select(origin, pot)
            client = source.engine_client(pot_id)
            if client is None:
                raise HTTPException(status_code=503, detail=_UNAVAILABLE)
            outcome = await operation(client, request)
            if not getattr(outcome, "ok", False) and optional:
                retry = _without_refused_optionals(request, outcome.error, optional)
                if retry is not None:
                    outcome = await operation(client, retry)
            if not getattr(outcome, "ok", False):
                raise boundary_http_error(outcome.error)
            document = outcome.value.to_dict()
            if document.get("ok", True) is False:
                status = document.get("status")
                raise HTTPException(
                    status_code=503 if status == "unavailable" else 409,
                    detail=document,
                )
            return document

        if guarded is None:
            return await run()
        return await guarded(run)

    @router.get("/api/commits")
    async def commits(
        pot: str | None = Query(None),
        host: str | None = Query(None),
        cursor: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        return await call(
            host,
            pot,
            lambda client, request: client.commits(request),
            CommitsRequest(cursor=cursor, limit=limit),
        )

    @router.get("/api/commit")
    async def commit(
        commit_id: str = Query(...),
        pot: str | None = Query(None),
        host: str | None = Query(None),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=200),
    ) -> dict[str, Any]:
        return await call(
            host,
            pot,
            lambda client, request: client.commit_show(request),
            CommitShowRequest(commit_id=commit_id, offset=offset, limit=limit),
        )

    @router.get("/api/journal")
    async def journal(
        pot: str | None = Query(None), host: str | None = Query(None)
    ) -> dict[str, Any]:
        return await call(
            host,
            pot,
            lambda client, request: client.journal_status(request),
            JournalStatusRequest(),
        )

    @router.get("/api/mutation-history")
    async def mutation_history(
        pot: str | None = Query(None),
        host: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        # Saved plans are audit evidence for pots without journal coverage;
        # they are not reconstructable receipts, so claim rows are left out.
        # A host that predates `include_claims` sends the claims too; the view
        # does not read them.
        return await call(
            host,
            pot,
            lambda client, request: client.history(request),
            HistoryRequest(limit=limit, include_claims=False),
            optional=("include_claims",),
        )

    @router.post("/api/rollback/preview")
    async def preview(body: PreviewBody) -> dict[str, Any]:
        if body.mode == "revert":
            return await call(
                body.host,
                body.pot,
                lambda client, request: client.revert_preview(request),
                RevertPreviewRequest(
                    commit_id=body.target_commit_id,
                    expected_head=body.expected_head,
                ),
            )
        return await call(
            body.host,
            body.pot,
            lambda client, request: client.rollback_preview(request),
            RollbackPreviewRequest(
                target_commit_id=body.target_commit_id,
                expected_head=body.expected_head,
            ),
        )

    return router


def _without_refused_optionals(
    request: Any, error: object, optional: tuple[str, ...]
) -> Any | None:
    """``request`` without the refused fields, if they are all optional ones."""
    if getattr(error, "code", None) != HOST_OUTDATED:
        return None
    details = getattr(error, "details", None)
    refused = tuple(details.get("fields") or ()) if isinstance(details, Mapping) else ()
    if not refused or not set(refused) <= set(optional):
        return None
    from potpie.runtime.remote import without_fields

    try:
        return without_fields(request, refused)
    except (TypeError, ValueError):
        return None


def boundary_http_error(error: object) -> HTTPException:
    """One typed boundary error as the explorer's HTTP error."""
    code = str(getattr(error, "code", "commit_operation_failed"))
    if code == HOST_OUTDATED:
        message = str(getattr(error, "message", "the host is older than this client"))
        detail: dict[str, Any] = {
            "ok": False,
            "status": HOST_OUTDATED,
            "message": message,
            "reasons": [{"code": HOST_OUTDATED, "message": message}],
            "recommended_next_action": getattr(error, "recommended_next_action", None),
        }
        details = getattr(error, "details", None)
        if isinstance(details, Mapping) and details:
            detail["details"] = {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in details.items()
            }
        return HTTPException(status_code=501, detail=detail)
    return HTTPException(status_code=_status_for(error), detail=_error_detail(error))


def _status_for(error: object) -> int:
    category = getattr(error, "category", None)
    code = getattr(error, "code", None)
    if code == "pot_not_found":
        return 404
    if code == "validation_error":
        return 400
    if code == "not_implemented":
        return 501
    if category == "authentication":
        return 401
    if category == "authorization":
        return 409 if code == "pot_archived" else 403
    if category in {"selection", "domain"}:
        return 409
    if category in {"dependency", "resource_lifecycle", "protocol_transport"}:
        return 503
    return 500


def _error_detail(error: object) -> dict[str, Any]:
    code = str(getattr(error, "code", "commit_operation_failed"))
    detail: dict[str, Any] = {
        "ok": False,
        "status": code,
        "reasons": [
            {
                "code": code,
                "message": str(getattr(error, "message", "The request failed.")),
            }
        ],
    }
    next_action = getattr(error, "recommended_next_action", None)
    if next_action:
        detail["recommended_next_action"] = next_action
    details = getattr(error, "details", None)
    if isinstance(details, Mapping) and details:
        detail["details"] = dict(details)
    return detail


__all__ = [
    "HOST_OUTDATED",
    "EngineClientFactory",
    "PreviewBody",
    "boundary_http_error",
    "build_commit_router",
]
