"""Who may read graph commit history and roll a pot back.

Two deployments share the rule "commit access comes from the typed boundary,
never from ambient process state":

* A managed host passes ``authorize_scoped_commit`` to the graph runtime. The
  typed operation handler binds the authorization scope the Resource Manager
  produced for the request (``authorization_scope``), and the scope's
  ``commit_access`` attribute decides the levels: ``reader`` reads,
  ``writer`` reads and writes, ``admin`` has every level. A scope without it
  gets nothing.
* A local install passes ``authorize_local_commit``, described below.

On a local install:

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

from potpie.runtime.resource_manager import AuthorizationScope
from potpie_context_engine.core.commit_service import (
    COMMIT_ACCESS_LEVELS,
    UNNAMED_COMMIT_ACTOR,
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


#: Commit access levels per role named in ``AuthorizationScope.attributes``
#: under ``commit_access``.
COMMIT_ACCESS_BY_ROLE: dict[str, frozenset[str]] = {
    "reader": frozenset({"read"}),
    "writer": frozenset({"read", "write"}),
    "admin": COMMIT_ACCESS_LEVELS,
}

_SCOPE: ContextVar[AuthorizationScope | None] = ContextVar(
    "potpie_authorization_scope", default=None
)


@contextmanager
def authorization_scope(scope: AuthorizationScope) -> Iterator[AuthorizationScope]:
    """Bind the scope one typed operation was authorized under, for its duration."""

    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def current_authorization_scope() -> AuthorizationScope | None:
    """The scope of the typed operation running in this context, if any."""

    return _SCOPE.get()


def commit_access_for(scope: AuthorizationScope | None) -> frozenset[str]:
    """Commit access levels a scope carries.

    ``scope.attributes["commit_access"]`` is a role (``reader``, ``writer``,
    ``admin``) or an explicit collection of levels. Anything else, a missing
    attribute or no scope at all, grants nothing.
    """

    if scope is None:
        return frozenset()
    granted = scope.attributes.get("commit_access")
    if isinstance(granted, str):
        return COMMIT_ACCESS_BY_ROLE.get(granted, frozenset())
    if isinstance(granted, (set, frozenset, list, tuple)):
        levels = frozenset(str(level) for level in granted)
        return levels & COMMIT_ACCESS_LEVELS
    return frozenset()


async def authorize_scoped_commit(pot_id: str, access: str) -> None:
    """Allow ``access`` on ``pot_id`` only inside a matching authorized scope.

    The ``commit_authorize`` callback for an embedding host's graph runtime:
    the request must have been authorized for exactly this pot, and its scope
    must carry the level.
    """

    if access not in COMMIT_ACCESS_LEVELS:
        raise ValueError("unknown commit access")
    scope = _SCOPE.get()
    if scope is None:
        raise CommitAccessDenied(
            "commit history and rollback are served only through Potpie's "
            "authenticated operations"
        )
    if scope.context.value != pot_id:
        raise CommitAccessDenied(
            f"this request was authorized for pot {scope.context.value!r}, "
            f"not {pot_id!r}"
        )
    if access not in commit_access_for(scope):
        raise CommitAccessDenied(f"commit {access} access was not granted")


def current_commit_actor() -> str:
    """The authorized actor of the running operation, for receipts and previews.

    The ``commit_actor`` callback for an embedding host's graph runtime. Outside
    an authorized operation it names nobody, so nothing can be attributed to an
    ambient principal.
    """

    scope = _SCOPE.get()
    return scope.actor_id if scope is not None else UNNAMED_COMMIT_ACTOR


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
    "COMMIT_ACCESS_BY_ROLE",
    "LOCAL_COMMIT_ACTOR",
    "CommitGrant",
    "authorization_scope",
    "authorize_local_commit",
    "authorize_scoped_commit",
    "commit_access_for",
    "commit_grant",
    "current_authorization_scope",
    "current_commit_actor",
    "local_commit_actor",
    "local_commit_host",
]
