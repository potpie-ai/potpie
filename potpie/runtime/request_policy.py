"""What a managed host refuses or rewrites in a request before dispatch.

A local runtime trusts its one principal with everything it asks for. A
managed host does not: the acting principal comes from the credential only,
and the host never touches its own filesystem on a caller's behalf.

``RemoteRequestPolicy`` applies those rules to a decoded, authenticated
engine request:

* ``approved_by`` (propose, commit) is an approval, not an attribution. An
  empty value stays empty, so the engine's approval gate for medium- and
  high-risk plans still applies; a non-empty value must be the actor.
* ``claimed_by`` (inbox claim), ``closed_by`` (inbox mark-applied,
  mark-rejected, close) and the ``actor`` key of ``created_by`` (inbox add)
  are attributions. An empty one is set to the actor; a non-empty one must be
  the actor.
* A version-1 snapshot export ``destination`` or import ``source`` names a
  path on the host, and is refused; version 2 moves the snapshot as data.

Refusals are typed ``AuthorizationError``\\ s: ``caller_assertion_mismatch``
and ``host_path_refused``, each with ``{operation, field}`` details.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from potpie.runtime.protocol import EngineOperationRequest
from potpie.runtime.resource_manager import AuthenticatedActor, AuthorizationError
from potpie_context_engine import Failure, Success
from potpie_context_engine.requests import (
    CommitRequest,
    ExportSnapshotRequest,
    ImportSnapshotRequest,
    InboxAddRequest,
    InboxClaimRequest,
    InboxCloseRequest,
    InboxMarkAppliedRequest,
    InboxMarkRejectedRequest,
    ProposeRequest,
)

CALLER_ASSERTION_MISMATCH = "caller_assertion_mismatch"
HOST_PATH_REFUSED = "host_path_refused"

PolicyOutcome = Success[EngineOperationRequest] | Failure[AuthorizationError]

#: Request fields that name an approver: checked, never filled in.
_APPROVALS: dict[type, str] = {
    ProposeRequest: "approved_by",
    CommitRequest: "approved_by",
}

#: Request fields that attribute the call to someone: filled in when empty.
_ATTRIBUTIONS: dict[type, str] = {
    InboxClaimRequest: "claimed_by",
    InboxMarkAppliedRequest: "closed_by",
    InboxMarkRejectedRequest: "closed_by",
    InboxCloseRequest: "closed_by",
}


class RequestPolicy(Protocol):
    """Refuse or rewrite one authenticated engine request before dispatch."""

    def apply(
        self, request: EngineOperationRequest, *, actor: AuthenticatedActor
    ) -> PolicyOutcome: ...


class RemoteRequestPolicy:
    """The caller-assertion and host-path rules of a managed host."""

    def apply(
        self, request: EngineOperationRequest, *, actor: AuthenticatedActor
    ) -> PolicyOutcome:
        payload = request.payload
        operation = request.operation.value

        path_field = _host_path_field(payload)
        if path_field is not None:
            return Failure(
                AuthorizationError(
                    code=HOST_PATH_REFUSED,
                    message=(
                        "a managed host never reads or writes its own filesystem "
                        f"for a caller ({path_field})"
                    ),
                    details={"operation": operation, "field": path_field},
                    recommended_next_action=(
                        "use snapshot version 2, which moves the snapshot as data"
                    ),
                )
            )

        approval = _APPROVALS.get(type(payload))
        if approval is not None:
            value = _text(getattr(payload, approval))
            if value and value != actor.actor_id:
                return Failure(_mismatch(operation, approval))
            return Success(request)

        attribution = _ATTRIBUTIONS.get(type(payload))
        if attribution is not None:
            value = _text(getattr(payload, attribution))
            if value and value != actor.actor_id:
                return Failure(_mismatch(operation, attribution))
            if not value:
                payload = replace(payload, **{attribution: actor.actor_id})
                return Success(replace(request, payload=payload))
            return Success(request)

        if isinstance(payload, InboxAddRequest):
            created_by = dict(payload.created_by)
            raw = created_by.get("actor")
            value = _text(raw)
            if (value and value != actor.actor_id) or (
                raw not in (None, "") and not isinstance(raw, str)
            ):
                return Failure(_mismatch(operation, "created_by.actor"))
            if not value:
                created_by["actor"] = actor.actor_id
                return Success(
                    replace(request, payload=replace(payload, created_by=created_by))
                )
        return Success(request)


def _host_path_field(payload: object) -> str | None:
    if isinstance(payload, ExportSnapshotRequest) and (
        payload.version == 1 or payload.destination
    ):
        return "destination"
    if isinstance(payload, ImportSnapshotRequest) and (
        payload.version == 1 or payload.source
    ):
        return "source"
    return None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mismatch(operation: str, field: str) -> AuthorizationError:
    return AuthorizationError(
        code=CALLER_ASSERTION_MISMATCH,
        message=(
            f"{field} must be empty or the authenticated caller on a managed host"
        ),
        details={"operation": operation, "field": field},
        recommended_next_action=(
            f"leave {field} empty; the host records the authenticated caller"
            if field != "approved_by"
            else "leave approved_by empty, or approve as yourself"
        ),
    )


__all__ = [
    "CALLER_ASSERTION_MISMATCH",
    "HOST_PATH_REFUSED",
    "PolicyOutcome",
    "RemoteRequestPolicy",
    "RequestPolicy",
]
