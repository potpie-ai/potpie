"""Credential-protected explorer commit routes; writes require same origin."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict

from potpie.daemon.http.ui.auth import require_same_origin


class PreviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_commit_id: str
    expected_head: str
    mode: Literal["revert", "rollback"]
    pot: str
    host: Literal["local", "managed"]


class ApplyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_id: str
    pot: str
    host: Literal["local", "managed"]


def build_commit_router(local_host, *, guarded, host_for, resolve_pot):
    router = APIRouter()

    def call(origin, pot, method, *args, **kwargs):
        def run():
            host, _ = host_for(local_host, origin)
            result = getattr(host.graph_workbench, method)(
                *args, pot_id=resolve_pot(host, pot), **kwargs
            )
            result = jsonable_encoder(result)
            if not result.get("ok", True):
                status = result.get("status")
                raise HTTPException(
                    status_code=503 if status == "unavailable" else 409, detail=result
                )
            return result

        return guarded(run)

    @router.get("/api/commits")
    def commits(
        host: str | None = None,
        pot: str | None = None,
        cursor: str | None = None,
        limit: int = Query(50, ge=1, le=200),
    ):
        return call(host, pot, "commits", cursor=cursor, limit=limit)

    @router.get("/api/commit")
    def commit(
        commit_id: str,
        host: str | None = None,
        pot: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=200),
    ):
        return call(host, pot, "commit_show", commit_id, offset=offset, limit=limit)

    @router.get("/api/journal")
    def journal(host: str | None = None, pot: str | None = None):
        return call(host, pot, "journal_status")

    @router.get("/api/mutation-history")
    def mutation_history(
        host: str | None = None,
        pot: str | None = None,
        limit: int = Query(50, ge=1, le=200),
    ):
        # Saved plans are audit evidence, not reconstructable journal receipts.
        return call(host, pot, "history", limit=limit, include_claims=False)

    @router.post("/api/rollback/preview", dependencies=[Depends(require_same_origin)])
    def preview(body: PreviewBody):
        return call(
            body.host,
            body.pot,
            body.mode + "_preview",
            body.target_commit_id,
            expected_head=body.expected_head,
        )

    @router.post("/api/rollback/apply", dependencies=[Depends(require_same_origin)])
    def apply(body: ApplyBody):
        return call(body.host, body.pot, "apply_preview", body.preview_id)

    return router
