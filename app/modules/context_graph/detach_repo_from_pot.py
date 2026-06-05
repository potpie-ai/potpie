"""``detach_repo_from_pot`` — opposite of ``attach_repo_to_pot``.

Single idempotent verb that:

  1. Removes the ``context_graph_pot_repositories`` row.
  2. Drops the mirrored ``context_graph_pot_sources`` entry.
  3. Recomputes the pot's ``primary_repo_name`` from what's left.
  4. Fires a best-effort sandbox detach so the worktree under the pot's
     container is released (other repos in the pot keep running).

The sandbox call is fire-and-forget — HTTP semantics shouldn't depend on
whether a backend container is reachable.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
from app.modules.context_graph.context_graph_pot_repository_model import (
    ContextGraphPotRepository,
)
from app.modules.context_graph.pot_sources_service import (
    unmirror_repository_from_sources,
)

logger = logging.getLogger(__name__)


class UnknownPotError(LookupError):
    """``pot_id`` not in ``context_graph_pots``."""


class UnknownRepositoryError(LookupError):
    """``repository_id`` not attached to this pot."""


@dataclass(frozen=True, slots=True)
class DetachRepoResult:
    repository_id: str
    pot_id: str
    owner: str
    repo: str


def detach_repo_from_pot(
    db: Session,
    *,
    pot_id: str,
    repository_id: str,
) -> DetachRepoResult:
    pot = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
    if pot is None:
        raise UnknownPotError(f"Unknown pot_id: {pot_id}")
    row = (
        db.query(ContextGraphPotRepository)
        .filter(
            ContextGraphPotRepository.pot_id == pot_id,
            ContextGraphPotRepository.id == repository_id,
        )
        .first()
    )
    if row is None:
        raise UnknownRepositoryError(f"Unknown repository_id: {repository_id}")

    owner = row.owner
    repo_name = row.repo
    added_by = row.added_by_user_id

    unmirror_repository_from_sources(db, row)
    db.delete(row)
    db.flush()
    _recompute_primary_repo_name(db, pot_id)
    db.commit()

    _dispatch_sandbox_detach(
        user_id=added_by,
        pot_id=pot_id,
        repo=f"{owner}/{repo_name}",
    )

    return DetachRepoResult(
        repository_id=repository_id,
        pot_id=pot_id,
        owner=owner,
        repo=repo_name,
    )


def _recompute_primary_repo_name(db: Session, pot_id: str) -> None:
    pot = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
    if pot is None:
        return
    first = (
        db.query(ContextGraphPotRepository)
        .filter(ContextGraphPotRepository.pot_id == pot_id)
        .order_by(ContextGraphPotRepository.created_at.asc())
        .first()
    )
    pot.primary_repo_name = f"{first.owner}/{first.repo}" if first else None


def _dispatch_sandbox_detach(
    *,
    user_id: str | None,
    pot_id: str,
    repo: str,
) -> None:
    """Fire-and-forget sandbox cleanup for one repo of a pot.

    Sandbox cleanup must not block the HTTP response: if the Daytona
    sandbox is gone, in cold start, or rate-limited, the database state is
    still correct and the next provisioning call will re-clone. Runs on a
    background thread so we don't reach into the request's event loop.
    """
    if not user_id:
        return

    def _runner() -> None:
        try:
            from app.modules.intelligence.tools.sandbox.client import (
                get_sandbox_client,
            )

            client = get_sandbox_client()
        except Exception:
            logger.exception(
                "detach_repo_from_pot: sandbox client unavailable for pot=%s repo=%s",
                pot_id,
                repo,
            )
            return
        try:
            asyncio.run(
                client.detach_repo_from_pot(
                    user_id=user_id,
                    project_id=pot_id,
                    repo=repo,
                )
            )
        except Exception:
            logger.exception(
                "detach_repo_from_pot: sandbox detach failed pot=%s repo=%s",
                pot_id,
                repo,
            )

    threading.Thread(
        target=_runner,
        name=f"sandbox-detach-{pot_id}",
        daemon=True,
    ).start()
