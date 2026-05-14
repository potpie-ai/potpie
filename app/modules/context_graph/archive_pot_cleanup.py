"""Background cleanup that runs after a pot is archived.

Archiving (``archived_at`` set) is the host-side equivalent of "deleting" a
pot. The user-visible state changes synchronously; the sandbox-side cleanup
runs on a background thread because tearing down a backend container can
block on third-party APIs (Daytona) that we don't want on a request path.

Bare-repo caches are shared across pots, so the default is to leave them
alone. ``CONTEXT_ENGINE_GC_BARE_ON_POT_DELETE=1`` opts in — and we still
cross-check ``ContextGraphPotRepository`` to refuse the deletion if any
other pot still references the cache.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading

from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_repository_model import (
    ContextGraphPotRepository,
)

logger = logging.getLogger(__name__)


def gc_bare_on_pot_delete_enabled() -> bool:
    """Opt-in flag — bare clones are shared across pots, default off."""
    return (os.getenv("CONTEXT_ENGINE_GC_BARE_ON_POT_DELETE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def dispatch_pot_sandbox_cleanup(
    db: Session,
    *,
    pot_id: str,
) -> None:
    """Fire-and-forget background tear-down of a pot's sandbox.

    Reads the pot's repo rows (still attached at archive time) to learn the
    container's ``(user_id, project_id)`` key and to decide whether to
    propose bare-cache GC. After this returns the caller can commit and
    respond — the actual SDK calls happen on a daemon thread.
    """
    rows = (
        db.query(ContextGraphPotRepository)
        .filter(ContextGraphPotRepository.pot_id == pot_id)
        .order_by(ContextGraphPotRepository.created_at.asc())
        .all()
    )
    if not rows:
        return
    first = rows[0]
    user_id = first.added_by_user_id
    if not user_id:
        return

    if gc_bare_on_pot_delete_enabled():
        # Bare caches are shared via ``(provider_host, owner/repo)``. Only GC
        # the ones that no surviving pot still references.
        attached_elsewhere = {
            f"{r.provider_host}|{r.owner}/{r.repo}"
            for r in db.query(ContextGraphPotRepository)
            .filter(ContextGraphPotRepository.pot_id != pot_id)
            .all()
        }
        gc_caches = all(
            f"{r.provider_host}|{r.owner}/{r.repo}" not in attached_elsewhere
            for r in rows
        )
    else:
        gc_caches = False

    def _runner() -> None:
        try:
            from app.modules.intelligence.tools.sandbox.client import (
                get_sandbox_client,
            )
            client = get_sandbox_client()
        except Exception:
            logger.exception(
                "archive_pot_cleanup: sandbox client unavailable for pot=%s",
                pot_id,
            )
            return
        try:
            asyncio.run(
                client.destroy_pot_sandbox(
                    user_id=user_id,
                    project_id=pot_id,
                    delete_repo_caches=gc_caches,
                )
            )
        except Exception:
            logger.exception(
                "archive_pot_cleanup: destroy_pot_sandbox failed for pot=%s",
                pot_id,
            )

    threading.Thread(
        target=_runner,
        name=f"sandbox-pot-cleanup-{pot_id}",
        daemon=True,
    ).start()
