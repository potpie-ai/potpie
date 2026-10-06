"""Who may read graph commit history and roll a pot back on a local install.

A local Potpie home has one principal: whoever runs the CLI in-process or holds
the daemon's per-boot credential. Both reach the engine only through the typed
operation boundary, after the resource manager has authenticated the caller
and authorized the operation for one selected pot. That boundary binds a grant
for exactly that pot (``commit_grant``), and ``authorize_local_commit`` refuses
everything else: a call with no grant, a grant for another pot, or an access
level the grant does not carry. Code elsewhere in the process that holds the
graph runtime cannot read history or apply a rollback by reaching around it.

The actor and the host recorded with previews and restore receipts identify
the local owner and this Potpie home without naming the account or the machine.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from potpie_context_engine.core.commit_service import (
    COMMIT_ACCESS_LEVELS,
    CommitAccessDenied,
)

#: The principal recorded on restore receipts and bound to rollback previews.
#: The same for the CLI, the daemon and the explorer, so a preview made in the
#: explorer can be applied from the CLI, and free of any account name.
LOCAL_COMMIT_ACTOR = "local:owner"


@dataclass(frozen=True, slots=True)
class CommitGrant:
    """Commit access the typed boundary granted for one selected pot."""

    pot_id: str
    access: frozenset[str]


_GRANT: ContextVar[CommitGrant | None] = ContextVar("potpie_commit_grant", default=None)


@contextmanager
def commit_grant(
    pot_id: str, access: Iterable[str] = COMMIT_ACCESS_LEVELS
) -> Iterator[CommitGrant]:
    """Grant commit ``access`` on ``pot_id`` to the code running inside the block."""

    levels = frozenset(access)
    unknown = levels - COMMIT_ACCESS_LEVELS
    if unknown:
        raise ValueError(f"unknown commit access: {sorted(unknown)!r}")
    grant = CommitGrant(pot_id=pot_id, access=levels)
    token = _GRANT.set(grant)
    try:
        yield grant
    finally:
        _GRANT.reset(token)


def local_commit_actor() -> str:
    return LOCAL_COMMIT_ACTOR


async def authorize_local_commit(pot_id: str, access: str) -> None:
    """Allow ``access`` on ``pot_id`` only inside a matching typed-boundary grant."""

    if access not in COMMIT_ACCESS_LEVELS:
        raise ValueError("unknown commit access")
    grant = _GRANT.get()
    if grant is None:
        raise CommitAccessDenied(
            "commit history and rollback are served only through Potpie's "
            "authenticated operations"
        )
    if grant.pot_id != pot_id:
        raise CommitAccessDenied(
            f"this request was authorized for pot {grant.pot_id!r}, not {pot_id!r}"
        )
    if access not in grant.access:
        raise CommitAccessDenied(f"commit {access} access was not granted")


def local_commit_host(home: Path) -> str:
    """Name this Potpie home for preview binding without recording its path.

    A preview is valid only on the home that made it. The resolved path would
    say that, but it usually contains the account name, so only a digest of it
    is kept.
    """

    resolved = str(Path(home).expanduser().resolve())
    return "local:" + hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "LOCAL_COMMIT_ACTOR",
    "CommitGrant",
    "authorize_local_commit",
    "commit_grant",
    "local_commit_actor",
    "local_commit_host",
]
