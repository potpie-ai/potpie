"""Authorized public history and server-held rollback previews.

Only target IDs cross the write boundary. Restore plans are rebuilt from native
receipts and checked against the persisted inverse hash before publication.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from potpie_context_engine.core.graph_journal import (
    JournalError,
    RollbackPreview,
    RollbackRequest,
    journal_hash,
    journal_json,
)
from potpie_context_engine.core.journal_fields import capture_changes
from potpie_context_engine.core.journal_history import (
    CommitFilters,
    commit_header,
    list_history,
    rebuild_history,
    receipt_detail,
)
from potpie_context_engine.core.ports.graph.journal import GraphJournalPort
from potpie_context_engine.core.ports.graph.preview_store import (
    RollbackPreviewStorePort,
)


#: Access levels the commit service asks a host's ``authorize`` callback for.
COMMIT_ACCESS_LEVELS = frozenset({"read", "write", "admin"})

#: Recorded on restore receipts and previews when a host names no actor. It is
#: deliberately anonymous: an account or machine name does not belong in history.
UNNAMED_COMMIT_ACTOR = "unnamed"


class CommitAccessDenied(PermissionError):
    """The host's authorization refused a commit-history or rollback request."""


async def deny_commit_access(pot_id: str, access: str) -> None:
    """Default authorization: refuse. A host must state who may roll back."""

    if access not in COMMIT_ACCESS_LEVELS:
        raise ValueError("unknown commit access")
    raise CommitAccessDenied(
        f"commit {access} access for pot {pot_id!r} needs a host-supplied "
        "authorization; none was composed"
    )


def failure(code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "status": code,
        "reasons": [{"code": code, "message": message}],
    }


class GraphCommitService:
    def __init__(
        self,
        *,
        journal: GraphJournalPort | None,
        mirror: Any,
        previews: RollbackPreviewStorePort | None,
        host: str,
        actor: Callable[[], str],
        authorize: Callable[[str, str], Awaitable[object]],
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.journal = journal
        self.mirror = mirror
        self.previews = previews
        self.host = host
        self.actor = actor
        self.authorize = authorize
        self.clock = clock

    async def journal_status_async(self, *, pot_id: str):
        await self.authorize(pot_id, "read")
        if self.journal is None:
            return failure(
                "unavailable", "This backend does not support the graph journal."
            )
        capability, state = await asyncio.gather(
            asyncio.to_thread(self.journal.journal_capability, pot_id),
            asyncio.to_thread(self.journal.journal_state, pot_id),
        )
        return {
            "ok": True,
            "capability": capability,
            "state": state,
            "journal_protocol_version": 1,
        }

    async def disable_rollback_async(self, *, pot_id: str):
        await self.authorize(pot_id, "admin")
        if self.journal is None:
            return failure(
                "unavailable", "This backend does not support the graph journal."
            )
        try:
            state = await asyncio.to_thread(
                self.journal.set_rollback_enabled, pot_id=pot_id, enabled=False
            )
            return {"ok": True, "state": state, "journal_capture_enabled": True}
        except JournalError as exc:
            return failure(exc.code, str(exc))

    async def rebuild_commits_async(self, *, pot_id: str):
        await self.authorize(pot_id, "admin")
        if self.journal is None or self.mirror is None:
            return failure("unavailable", "History index is unavailable.")
        try:
            return {
                "ok": True,
                "coverage": await rebuild_history(
                    self.journal, self.mirror, pot_id=pot_id
                ),
            }
        except JournalError as exc:
            return failure(exc.code, str(exc))

    async def commits_async(
        self,
        *,
        pot_id: str,
        cursor: str | None = None,
        limit: int = 50,
        actor: str | None = None,
        origin: str | None = None,
        logical_key: str | None = None,
    ):
        await self.authorize(pot_id, "read")
        if self.journal is None or self.mirror is None:
            return failure(
                "unavailable",
                "Commit history requires a native journal and listing store.",
            )
        try:
            page = await list_history(
                self.journal,
                self.mirror,
                pot_id=pot_id,
                cursor=cursor,
                limit=limit,
                filters=CommitFilters(actor, origin, logical_key),
            )
            return {"ok": True, **page}
        except JournalError as exc:
            return failure(
                exc.code if exc.code != "invalid_journal" else "invalid_history",
                str(exc),
            )

    async def commit_show_async(
        self, commit_id: str, *, pot_id: str, offset: int = 0, limit: int = 100
    ):
        await self.authorize(pot_id, "read")
        if self.journal is None:
            return failure(
                "unavailable", "This backend does not support the graph journal."
            )
        try:
            detail = await asyncio.to_thread(
                receipt_detail,
                self.journal,
                pot_id=pot_id,
                commit_id=commit_id,
                offset=offset,
                limit=limit,
            )
            return {"ok": True, **detail}
        except JournalError as exc:
            return failure("invalid_detail", str(exc))

    async def revert_preview_async(
        self, target_commit_id: str, *, pot_id: str, expected_head: str
    ):
        return await self._preview(
            target_commit_id, pot_id=pot_id, expected_head=expected_head, mode="revert"
        )

    async def rollback_preview_async(
        self, target_commit_id: str, *, pot_id: str, expected_head: str
    ):
        return await self._preview(
            target_commit_id,
            pot_id=pot_id,
            expected_head=expected_head,
            mode="rollback",
        )

    async def _preview(
        self, target_commit_id: str, *, pot_id: str, expected_head: str, mode: str
    ):
        await self.authorize(pot_id, "write")
        if self.journal is None or self.previews is None:
            return failure(
                "unavailable", "Rollback previews are unavailable on this host."
            )
        if (
            not isinstance(target_commit_id, str)
            or not target_commit_id
            or not isinstance(expected_head, str)
            or not expected_head
        ):
            return failure(
                "invalid_request", "A target commit and expected HEAD are required."
            )
        try:
            plan = await asyncio.to_thread(
                self.journal.plan_restore,
                pot_id=pot_id,
                target_commit_id=target_commit_id,
                expected_head=expected_head,
                mode=mode,
            )
            await self.authorize(pot_id, plan.required_access)
            preview = RollbackPreview(
                "rollback-preview:" + uuid.uuid4().hex,
                self.actor(),
                self.host,
                pot_id,
                plan.journal_generation,
                plan.resource_generation,
                expected_head,
                plan.inverse_hash,
                plan.target_commit_ids,
                plan.required_access,
                self.clock() + timedelta(minutes=10),
            )
            semantic, audit = capture_changes(
                {p.before.record_id: p.before for p in plan.records},
                {p.after.record_id: p.after for p in plan.records},
            )
            state = await asyncio.to_thread(self.journal.journal_state, pot_id)
            # A preview is bounded even when the permitted atomic restore is larger.
            changes = (*semantic, *audit)
            page = changes[: min(100, state.limits.max_detail_records)]
            result = {
                "ok": True,
                "status": "ready",
                "preview": preview,
                "changes": page,
                "affected_record_count": len(plan.records),
                "changes_truncated": len(page) < len(changes),
                "limits": asdict(state.limits),
            }
            if len(journal_json(result).encode()) > 1_000_000:
                result["changes"] = ()
                result["changes_truncated"] = True
            await self.previews.put_async(
                preview, RollbackRequest(target_commit_id, expected_head, mode)
            )
            return result
        except JournalError as exc:
            return failure(
                exc.code if exc.code != "invalid_journal" else "preview_rejected",
                str(exc),
            )

    async def apply_preview_async(self, preview_id: str, *, pot_id: str):
        await self.authorize(pot_id, "write")
        if self.journal is None or self.previews is None:
            return failure(
                "unavailable", "Rollback previews are unavailable on this host."
            )
        try:
            stored = await self.previews.get_async(preview_id=preview_id, pot_id=pot_id)
        except JournalError as exc:
            return failure("preview_modified", str(exc))
        if stored is None:
            return failure("preview_not_found", "Preview is unavailable for this pot.")
        preview, request = stored
        if (preview.pot_id, preview.host, preview.actor) != (
            pot_id,
            self.host,
            self.actor(),
        ):
            return failure(
                "preview_scope_mismatch",
                "Preview belongs to another actor, host, or pot.",
            )
        await self.authorize(pot_id, preview.required_access)
        commit_id = "restore:" + preview.preview_id
        replay = await self._replay(preview, request)
        if replay is not None:
            return replay
        if self.clock() >= preview.expires_at:
            return failure(
                "preview_expired", "Preview expired; generate a new preview."
            )
        try:
            state = await asyncio.to_thread(self.journal.journal_state, pot_id)
            if state is None or (
                state.generation,
                state.resource_generation,
                state.head,
            ) != (
                preview.journal_generation,
                preview.resource_generation,
                preview.expected_head,
            ):
                return await self._replay(preview, request) or failure(
                    "preview_stale",
                    "Graph or resource generation changed; generate a new preview.",
                )
            plan = await asyncio.to_thread(
                self.journal.plan_restore,
                pot_id=pot_id,
                target_commit_id=request.target_commit_id,
                expected_head=request.expected_head,
                mode=request.mode,
            )
            if (plan.inverse_hash, plan.required_access, plan.target_commit_ids) != (
                preview.inverse_hash,
                preview.required_access,
                preview.target_commit_ids,
            ):
                return failure(
                    "preview_modified",
                    "Preview no longer matches the validated inverse.",
                )
            await self.authorize(pot_id, plan.required_access)
            receipt = await asyncio.to_thread(
                self.journal.apply_restore,
                plan,
                mutation_id=commit_id,
                actor=preview.actor,
            )
            return {
                "ok": True,
                "status": "committed",
                "commit": commit_header(receipt),
                "replayed": False,
            }
        except JournalError as exc:
            return await self._replay(preview, request) or failure(
                exc.code if exc.code != "invalid_journal" else "apply_rejected",
                str(exc),
            )

    async def _replay(self, preview, request):
        prior = await asyncio.to_thread(
            self.journal.get_receipt,
            pot_id=preview.pot_id,
            commit_id="restore:" + preview.preview_id,
        )
        if prior is not None:
            await self.authorize(preview.pot_id, prior.required_access)
            fingerprint = journal_hash(
                {
                    "inverse": preview.inverse_hash,
                    "actor": preview.actor,
                    "generation": preview.journal_generation,
                    "head": preview.expected_head,
                    "mode": request.mode,
                    "target": request.target_commit_id,
                    "access": preview.required_access,
                    "resource_generation": preview.resource_generation,
                }
            )
            if prior.fingerprint != fingerprint:
                return failure(
                    "preview_modified", "Preview does not match its published receipt."
                )
            return {
                "ok": True,
                "status": "committed",
                "commit": commit_header(prior),
                "replayed": True,
            }
        return None


class GraphCommitSurface:
    """Explicit shared doors for runtime and workbench facades."""

    commit_service: GraphCommitService

    def disable_rollback(self, **kwargs):
        return asyncio.run(self.disable_rollback_async(**kwargs))

    async def disable_rollback_async(self, **kwargs):
        return await self.commit_service.disable_rollback_async(**kwargs)

    def rebuild_commits(self, **kwargs):
        return asyncio.run(self.rebuild_commits_async(**kwargs))

    async def rebuild_commits_async(self, **kwargs):
        return await self.commit_service.rebuild_commits_async(**kwargs)

    def journal_status(self, **kwargs):
        return asyncio.run(self.journal_status_async(**kwargs))

    async def journal_status_async(self, **kwargs):
        return await self.commit_service.journal_status_async(**kwargs)

    def commits(self, **kwargs):
        return asyncio.run(self.commits_async(**kwargs))

    async def commits_async(self, **kwargs):
        return await self.commit_service.commits_async(**kwargs)

    def commit_show(self, commit_id, **kwargs):
        return asyncio.run(self.commit_show_async(commit_id, **kwargs))

    async def commit_show_async(self, commit_id, **kwargs):
        return await self.commit_service.commit_show_async(commit_id, **kwargs)

    def revert_preview(self, target_commit_id, **kwargs):
        return asyncio.run(self.revert_preview_async(target_commit_id, **kwargs))

    async def revert_preview_async(self, target_commit_id, **kwargs):
        return await self.commit_service.revert_preview_async(
            target_commit_id, **kwargs
        )

    def rollback_preview(self, target_commit_id, **kwargs):
        return asyncio.run(self.rollback_preview_async(target_commit_id, **kwargs))

    async def rollback_preview_async(self, target_commit_id, **kwargs):
        return await self.commit_service.rollback_preview_async(
            target_commit_id, **kwargs
        )

    def apply_preview(self, preview_id, **kwargs):
        return asyncio.run(self.apply_preview_async(preview_id, **kwargs))

    async def apply_preview_async(self, preview_id, **kwargs):
        return await self.commit_service.apply_preview_async(preview_id, **kwargs)
