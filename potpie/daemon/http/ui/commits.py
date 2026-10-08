"""Explorer routes for graph commit history, recorded diffs and rollback previews.

These routes are included in the ``/ui/api`` router, so its credential and
same-origin checks apply to every one of them. What a browser session may do
here is read: list commits, read a commit's recorded changes, see journal
status and saved plans, and ask for a rollback preview, which is a dry run the
server holds for a few minutes and which changes nothing. Applying a preview
changes the graph and needs the daemon credential and explicit destructive
confirmation, so it stays on the CLI (``potpie graph apply-preview``); there is
no apply route on this surface.

Each route sends the same typed operation the CLI sends, in-process through the
daemon's own resource manager and operation coordinator, so archived-pot
refusal, locking and commit authorization are identical on both surfaces.
"""

from __future__ import annotations

import asyncio
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


def build_commit_router(
    *,
    pots: Any,
    resolve_pot: Callable[[Any, str | None], str],
    engine_client: EngineClientFactory | None,
) -> APIRouter:
    router = APIRouter()

    async def call(
        pot: str | None, operation: Callable[[Any], Awaitable[Any]]
    ) -> dict[str, Any]:
        if engine_client is None:
            raise HTTPException(status_code=503, detail=_UNAVAILABLE)
        pot_id = await asyncio.to_thread(resolve_pot, pots, pot)
        outcome = await operation(engine_client(pot_id))
        if not getattr(outcome, "ok", False):
            error = outcome.error
            raise HTTPException(
                status_code=_status_for(error), detail=_error_detail(error)
            )
        document = outcome.value.to_dict()
        if document.get("ok", True) is False:
            status = document.get("status")
            raise HTTPException(
                status_code=503 if status == "unavailable" else 409,
                detail=document,
            )
        return document

    @router.get("/api/commits")
    async def commits(
        pot: str | None = Query(None),
        cursor: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        return await call(
            pot,
            lambda client: client.commits(CommitsRequest(cursor=cursor, limit=limit)),
        )

    @router.get("/api/commit")
    async def commit(
        commit_id: str = Query(...),
        pot: str | None = Query(None),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=200),
    ) -> dict[str, Any]:
        return await call(
            pot,
            lambda client: client.commit_show(
                CommitShowRequest(commit_id=commit_id, offset=offset, limit=limit)
            ),
        )

    @router.get("/api/journal")
    async def journal(pot: str | None = Query(None)) -> dict[str, Any]:
        return await call(
            pot, lambda client: client.journal_status(JournalStatusRequest())
        )

    @router.get("/api/mutation-history")
    async def mutation_history(
        pot: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        # Saved plans are audit evidence for pots without journal coverage;
        # they are not reconstructable receipts, so claim rows are left out.
        return await call(
            pot,
            lambda client: client.history(
                HistoryRequest(limit=limit, include_claims=False)
            ),
        )

    @router.post("/api/rollback/preview")
    async def preview(body: PreviewBody) -> dict[str, Any]:
        if body.mode == "revert":
            return await call(
                body.pot,
                lambda client: client.revert_preview(
                    RevertPreviewRequest(
                        commit_id=body.target_commit_id,
                        expected_head=body.expected_head,
                    )
                ),
            )
        return await call(
            body.pot,
            lambda client: client.rollback_preview(
                RollbackPreviewRequest(
                    target_commit_id=body.target_commit_id,
                    expected_head=body.expected_head,
                )
            ),
        )

    return router


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


__all__ = ["EngineClientFactory", "PreviewBody", "build_commit_router"]
