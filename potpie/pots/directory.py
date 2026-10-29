"""``PotDirectory``: the pot control plane of one host, whichever host it is.

The CLI's pot, source, grant, ``whoami`` and ``status`` commands read and
write pots through this port, so the same command body serves both origins:

``LocalPotDirectory``
    The in-process pot service of this machine (``potpie.runtime.root_services``).
``ManagedPotDirectory`` (``potpie.runtime.remote``)
    A managed host's ``host.*`` operations, for the caller's actor.

Every method is synchronous and answers with the ``potpie.pots.contracts``
types. Methods only one origin can serve refuse on the other with
:class:`~potpie_context_engine.core.errors.CapabilityNotImplemented` naming the
command, rather than answering for a host they do not reach.

Repo identities: a key derived from a git remote (``github.com/acme/shop``) is
meaningful on any host. A key derived from a local path is not, and is never
sent to a managed host (see :func:`is_remote_repo_key`).
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Final, Protocol, runtime_checkable

from potpie_context_engine.core.errors import CapabilityNotImplemented

from potpie.pots.contracts import (
    PotAggregateStatus,
    PotInfo,
    PotRepoSource,
    SourceInfo,
)

LOCAL: Final = "local"
MANAGED: Final = "managed"

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def is_remote_repo_key(key: str | None) -> bool:
    """True for a repo identity key derived from a git remote, not a path.

    ``potpie.cli.repo_location.repo_identity_key`` keeps paths absolute and
    normalizes remotes to ``host/owner/repo``, so "absolute" is exactly
    "derived from a local path".
    """
    raw = (key or "").strip()
    if not raw:
        return False
    if raw.startswith(("/", "\\", "~")) or _WINDOWS_DRIVE.match(raw):
        return False
    return not (PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute())


@runtime_checkable
class PotDirectory(Protocol):
    """One host's pots, routing pointers, sources and sharing, for the caller."""

    #: ``"local"`` or ``"managed"``.
    origin: str

    @property
    def supports_repo_defaults(self) -> bool: ...

    # --- pots -----------------------------------------------------------------
    def list_pots(self) -> list[PotInfo]:
        """Every pot the caller can reach, archived ones included."""
        ...

    def active_pot(self) -> PotInfo | None: ...

    def create_pot(
        self, *, name: str, repo: str | None = None, use: bool = False
    ) -> PotInfo: ...

    def use_pot(self, *, ref: str) -> PotInfo:
        """Make the pot the caller's active pot on this host."""
        ...

    # --- repo routing defaults ------------------------------------------------
    def repo_default(self, *, repo: str) -> str | None: ...

    def set_repo_default(self, *, repo: str, pot_id: str) -> None: ...

    def clear_repo_default(self, *, repo: str) -> bool: ...

    # --- sources --------------------------------------------------------------
    def list_sources(self, *, pot_id: str) -> list[SourceInfo]: ...

    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo: ...

    def list_repo_sources(self) -> list[PotRepoSource]:
        """The repo→pot index: every live pot's repo sources, joined to it."""
        ...

    # --- status, identity and sharing -----------------------------------------
    def aggregate_status(self, *, pot_id: str | None = None) -> PotAggregateStatus: ...

    def whoami(self) -> Any:
        """``subject``/``mode``/``detail`` of the caller as this host sees it."""
        ...

    def list_grants(self, *, pot_id: str) -> list[Any]: ...

    def grant(self, *, pot_id: str, actor_id: str, role: str) -> Any: ...

    def revoke(self, *, pot_id: str, actor_id: str) -> bool: ...


def single_user_refusal(verb: str) -> CapabilityNotImplemented:
    """Sharing on the local host, which has exactly one user."""
    return CapabilityNotImplemented(
        f"host.{verb}",
        detail=(
            f"'pot {verb}' needs a managed host: the local daemon is single-user, "
            "so a pot there has no one to share with"
        ),
        recommended_next_action="point this CLI at a service with 'potpie host set <url>'",
    )


class LocalPotDirectory:
    """The local host's pot service, behind the :class:`PotDirectory` port.

    Forwards to the in-process ``PotResourceService`` this machine composes.
    Local-only lifecycle (rename, archive, source removal and status) is
    reached through the same object, so a command that has one directory in
    hand never has to ask which origin it is on to call them.
    """

    origin: str = LOCAL

    def __init__(self, pots: Any, *, whoami: Any = None) -> None:
        self._pots = pots
        self._whoami = whoami

    @property
    def service(self) -> Any:
        """The wrapped in-process pot service."""
        return self._pots

    @property
    def supports_repo_defaults(self) -> bool:
        flag = getattr(self._pots, "supports_repo_defaults", None)
        if flag is not None:
            return bool(flag)
        return callable(getattr(self._pots, "set_repo_default", None))

    def list_pots(self) -> list[PotInfo]:
        return list(self._pots.list_pots())

    def active_pot(self) -> PotInfo | None:
        return self._pots.active_pot()

    def create_pot(
        self, *, name: str, repo: str | None = None, use: bool = False
    ) -> PotInfo:
        if repo is None:
            return self._pots.create_pot(name=name, use=use)
        return self._pots.create_pot(name=name, repo=repo, use=use)

    def use_pot(self, *, ref: str) -> PotInfo:
        return self._pots.use_pot(ref=ref)

    def repo_default(self, *, repo: str) -> str | None:
        getter = getattr(self._pots, "repo_default", None)
        return getter(repo=repo) if callable(getter) else None

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        self._pots.set_repo_default(repo=repo, pot_id=pot_id)

    def clear_repo_default(self, *, repo: str) -> bool:
        return bool(self._pots.clear_repo_default(repo=repo))

    def list_sources(self, *, pot_id: str) -> list[SourceInfo]:
        return list(self._pots.list_sources(pot_id=pot_id))

    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo:
        return self._pots.add_source(
            pot_id=pot_id, kind=kind, location=location, name=name
        )

    def list_repo_sources(self) -> list[PotRepoSource]:
        from potpie.pots.resolution import repo_source_index

        return repo_source_index(self._pots)

    def aggregate_status(self, *, pot_id: str | None = None) -> PotAggregateStatus:
        return self._pots.aggregate_status(pot_id=pot_id)

    def whoami(self) -> Any:
        if self._whoami is None:
            raise CapabilityNotImplemented(
                "host.whoami", detail="this local host composes no auth service"
            )
        return self._whoami()

    def list_grants(self, *, pot_id: str) -> list[Any]:
        raise single_user_refusal("grants")

    def grant(self, *, pot_id: str, actor_id: str, role: str) -> Any:
        raise single_user_refusal("grant")

    def revoke(self, *, pot_id: str, actor_id: str) -> bool:
        raise single_user_refusal("revoke")

    # --- local-only lifecycle ---------------------------------------------------
    def rename_pot(self, *, ref: str, new_name: str) -> PotInfo:
        return self._pots.rename_pot(ref=ref, new_name=new_name)

    def archive_pot(self, *, ref: str) -> PotInfo:
        return self._pots.archive_pot(ref=ref)

    def source_status(self, *, pot_id: str, source_id: str) -> SourceInfo:
        return self._pots.source_status(pot_id=pot_id, source_id=source_id)

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        self._pots.remove_source(pot_id=pot_id, source_id=source_id)

    def list_repo_defaults(self) -> dict[str, str]:
        return dict(self._pots.list_repo_defaults())

    def __getattr__(self, name: str) -> Any:
        # Anything else the local service offers (``init``, test doubles'
        # extras) stays reachable; only names this class does not define land
        # here.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._pots, name)


__all__ = [
    "LOCAL",
    "MANAGED",
    "LocalPotDirectory",
    "PotDirectory",
    "is_remote_repo_key",
    "single_user_refusal",
]
