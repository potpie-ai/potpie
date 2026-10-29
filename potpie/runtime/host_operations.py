"""Host operations: pots, sources, routing defaults, grants and identity.

A managed host serves these beside the engine catalog, on the same typed
envelope. They are product operations of root Potpie, not engine operations:
their types reuse ``potpie.pots.contracts`` and the embedding host implements
them with a :class:`HostOperationHandler`. The local daemon does not serve
them (its CLI reaches the local pot service in-process), so they are not part
of the local daemon's operation catalog fingerprint or capabilities.

``host.*`` operations that require a context resolve and authorize it through
the Resource Manager exactly like engine operations; the handler receives the
authenticated actor and, for those, the resolved context.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from potpie.pots.contracts import PotAggregateStatus, PotInfo, SourceInfo
from potpie.runtime.operations import OperationSpec, SafetyClass
from potpie.runtime.resource_manager import AuthenticatedActor
from potpie_context_engine import ContextIdentity


class HostOperation(StrEnum):
    POT_LIST = "host.pot_list"
    POT_CREATE = "host.pot_create"
    POT_USE = "host.pot_use"
    REPO_DEFAULT_SET = "host.repo_default_set"
    REPO_DEFAULT_CLEAR = "host.repo_default_clear"
    POT_STATUS = "host.pot_status"
    SOURCE_LIST = "host.source_list"
    SOURCE_ADD = "host.source_add"
    WHOAMI = "host.whoami"
    GRANT_LIST = "host.grant_list"
    GRANT_SET = "host.grant_set"
    GRANT_REVOKE = "host.grant_revoke"


# --- pots ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PotListRequest:
    """The caller's pots, with its routing pointers in the same answer.

    ``repo`` is a repo identity key (derived from a git remote, never from a
    local path); when given, the answer names the caller's default pot for it.
    ``include_status`` adds per-pot counts for a pot selector.
    """

    repo: str | None = None
    include_archived: bool = False
    include_status: bool = False


@dataclass(frozen=True, slots=True)
class PotCounts:
    """Browsing counts for one pot; absent values mean the host could not say."""

    source_count: int | None = None
    counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PotListResult:
    pots: tuple[PotInfo, ...] = ()
    active_pot_id: str | None = None
    repo_default_pot_id: str | None = None
    status: Mapping[str, PotCounts] = field(default_factory=dict)
    """Per pot id, only when ``include_status`` was asked for."""


@dataclass(frozen=True, slots=True)
class PotCreateRequest:
    """Create a pot (or reuse the caller's live pot of that name)."""

    name: str = ""
    repo: str | None = None
    use: bool = False


@dataclass(frozen=True, slots=True)
class PotUseRequest:
    """Make the selected pot the caller's active pot."""


@dataclass(frozen=True, slots=True)
class PotStatusRequest:
    """Control-plane status of the selected pot."""


# --- repo routing defaults -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepoDefaultSetRequest:
    """Route ``repo`` to the selected pot for the caller."""

    repo: str = ""


@dataclass(frozen=True, slots=True)
class RepoDefaultResult:
    repo: str = ""
    pot_id: str | None = None


@dataclass(frozen=True, slots=True)
class RepoDefaultClearRequest:
    repo: str = ""


@dataclass(frozen=True, slots=True)
class RepoDefaultClearResult:
    repo: str = ""
    cleared: bool = False


# --- sources -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceListRequest:
    """Sources registered to the selected pot."""


@dataclass(frozen=True, slots=True)
class SourceListResult:
    sources: tuple[SourceInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceAddRequest:
    """Register a source (a repo remote, an integration binding) to the pot."""

    kind: str = ""
    location: str = ""
    name: str | None = None


# --- identity and sharing ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WhoamiRequest:
    """Who the host believes the caller is."""


@dataclass(frozen=True, slots=True)
class WhoamiResult:
    subject: str = ""
    mode: str = ""
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PotGrantInfo:
    """One actor's role on one pot."""

    pot_id: str = ""
    actor_id: str = ""
    role: str = ""


@dataclass(frozen=True, slots=True)
class GrantListRequest:
    """Who can reach the selected pot, and as what."""


@dataclass(frozen=True, slots=True)
class GrantListResult:
    grants: tuple[PotGrantInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class GrantSetRequest:
    """Give ``actor_id`` a role (``reader``, ``writer``, ``admin``) on the pot."""

    actor_id: str = ""
    role: str = "writer"


@dataclass(frozen=True, slots=True)
class GrantRevokeRequest:
    actor_id: str = ""


@dataclass(frozen=True, slots=True)
class GrantRevokeResult:
    pot_id: str = ""
    actor_id: str = ""
    revoked: bool = False


# --- the catalog ---------------------------------------------------------------

_READ = SafetyClass.SHARED_CONTEXT_READ
_RESOURCE = SafetyClass.EXCLUSIVE_RESOURCE_MUTATION

_HOST_SPECS = (
    OperationSpec(
        HostOperation.POT_LIST,
        PotListRequest,
        PotListResult,
        _READ,
        context_required=False,
    ),
    OperationSpec(
        HostOperation.POT_CREATE,
        PotCreateRequest,
        PotInfo,
        _RESOURCE,
        context_required=False,
        resource_type="pot_name",
        resource_identity_fields=("name",),
    ),
    # The resource is the caller's active-pot pointer; it is identified by the
    # envelope's selector, not by a request field.
    OperationSpec(
        HostOperation.POT_USE,
        PotUseRequest,
        PotInfo,
        _RESOURCE,
        resource_type="actor_pointer",
        resource_identity_fields=("selector",),
    ),
    OperationSpec(
        HostOperation.REPO_DEFAULT_SET,
        RepoDefaultSetRequest,
        RepoDefaultResult,
        _RESOURCE,
        resource_type="repo_binding",
        resource_identity_fields=("repo",),
    ),
    OperationSpec(
        HostOperation.REPO_DEFAULT_CLEAR,
        RepoDefaultClearRequest,
        RepoDefaultClearResult,
        _RESOURCE,
        context_required=False,
        resource_type="repo_binding",
        resource_identity_fields=("repo",),
    ),
    OperationSpec(
        HostOperation.POT_STATUS, PotStatusRequest, PotAggregateStatus, _READ
    ),
    OperationSpec(
        HostOperation.SOURCE_LIST, SourceListRequest, SourceListResult, _READ
    ),
    OperationSpec(
        HostOperation.SOURCE_ADD,
        SourceAddRequest,
        SourceInfo,
        SafetyClass.EXCLUSIVE_CONTEXT_MUTATION,
    ),
    OperationSpec(
        HostOperation.WHOAMI,
        WhoamiRequest,
        WhoamiResult,
        _READ,
        context_required=False,
    ),
    OperationSpec(HostOperation.GRANT_LIST, GrantListRequest, GrantListResult, _READ),
    OperationSpec(
        HostOperation.GRANT_SET,
        GrantSetRequest,
        PotGrantInfo,
        _RESOURCE,
        resource_type="pot_grant",
        resource_identity_fields=("actor_id",),
    ),
    OperationSpec(
        HostOperation.GRANT_REVOKE,
        GrantRevokeRequest,
        GrantRevokeResult,
        _RESOURCE,
        resource_type="pot_grant",
        resource_identity_fields=("actor_id",),
    ),
)

if {spec.operation for spec in _HOST_SPECS} != set(HostOperation) or len(
    _HOST_SPECS
) != len(HostOperation):
    raise RuntimeError("the host catalog must cover every host operation once")

HOST_OPERATION_CATALOG: Mapping[HostOperation, OperationSpec] = MappingProxyType(
    {HostOperation(spec.operation): spec for spec in _HOST_SPECS}
)


class HostOperationHandler(Protocol):
    """The embedding host's implementation of the ``host.*`` operations.

    ``context`` is the pot the Resource Manager resolved and authorized for
    an operation that requires one, and ``None`` otherwise. Return the
    catalog's result type, or a ``Success``/``Failure`` outcome; a raised
    exception becomes a typed internal error.
    """

    async def handle(
        self,
        operation: HostOperation,
        request: object,
        *,
        actor: AuthenticatedActor,
        context: ContextIdentity | None,
    ) -> object: ...


__all__ = [
    "HOST_OPERATION_CATALOG",
    "GrantListRequest",
    "GrantListResult",
    "GrantRevokeRequest",
    "GrantRevokeResult",
    "GrantSetRequest",
    "HostOperation",
    "HostOperationHandler",
    "PotCounts",
    "PotCreateRequest",
    "PotGrantInfo",
    "PotListRequest",
    "PotListResult",
    "PotStatusRequest",
    "PotUseRequest",
    "RepoDefaultClearRequest",
    "RepoDefaultClearResult",
    "RepoDefaultResult",
    "RepoDefaultSetRequest",
    "SourceAddRequest",
    "SourceListRequest",
    "SourceListResult",
    "WhoamiRequest",
    "WhoamiResult",
]
