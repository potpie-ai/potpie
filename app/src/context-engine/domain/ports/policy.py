"""Centralized authorization port.

The :class:`PolicyPort` is the single decision call site for "should this
actor be allowed to perform this action on this resource?" — including engine
enable/disable gates that previously chained through routes.

Call sites:

- Every HTTP route in ``adapters/inbound/http/api/v1/context/router.py``
  funnels through ``authorize`` before doing any work. MCP and CLI reach the
  engine through HTTP, so the HTTP layer covers them transitively (MCP adds
  its own pot allowlist as a pre-flight tenant boundary).
- ``application/use_cases/process_batch.py`` calls ``apply.write`` once
  before invoking the reconciliation agent — one policy decision per batch
  covers every mutation the agent issues.

Adding a new authorization rule means changing one adapter, not every route.
Decisions carry a structured ``reason`` so callers can surface *why* an
action was denied (``"context_graph_disabled"``, ``"reconciliation_disabled"``,
``"unknown_pot"``, ``"forbidden"``).

The default in-process adapter (`DefaultPolicyAdapter`) wraps the existing
settings + pot-resolution checks. Hosts (Potpie) can swap in a richer adapter
without changing the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from domain.actor import Actor


# Resource taxonomy. Single source of truth — adapters only branch on these.
RESOURCE_POT = "pot"
RESOURCE_CONNECTOR = "connector"
RESOURCE_APPLY = "apply"

# Pot actions
ACTION_POT_READ = "read"
ACTION_POT_SUBMIT_EVENT = "submit_event"
ACTION_POT_RECORD = "record"
ACTION_POT_INGEST_EPISODE = "ingest_episode"
ACTION_POT_RESET = "reset"
ACTION_POT_RESOLVE_CONFLICT = "resolve_conflict"

# Connector actions
ACTION_CONNECTOR_LIST = "list"
ACTION_CONNECTOR_FETCH = "fetch"

# Apply actions
ACTION_APPLY_WRITE = "write"

# Reason taxonomy returned in PolicyDecision.reason. Keep stable — surfaced
# in HTTP responses and the resolve envelope's ``meta.policy_decision``.
REASON_OK = "ok"
REASON_CONTEXT_GRAPH_DISABLED = "context_graph_disabled"
REASON_RECONCILIATION_DISABLED = "reconciliation_disabled"
REASON_AGENT_PLANNER_DISABLED = "agent_planner_disabled"
REASON_RECONCILIATION_AGENT_UNAVAILABLE = "reconciliation_agent_unavailable"
REASON_CONTEXT_GRAPH_UNAVAILABLE = "context_graph_unavailable"
REASON_EPISODIC_UNAVAILABLE = "episodic_unavailable"
REASON_UNKNOWN_POT = "unknown_pot"
REASON_FORBIDDEN = "forbidden"
REASON_UNSUPPORTED_RESOURCE = "unsupported_resource"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Outcome of one authorization check.

    ``reason`` is a stable taxonomy code (callers may key error responses on
    it). ``detail`` is human-readable text. ``status_code`` is the HTTP code
    a route should map to when ``allowed`` is false; defaults to 503 for
    capability gates and 404 for unknown pots.
    """

    allowed: bool
    reason: str = REASON_OK
    detail: str = ""
    status_code: int = 200
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, **metadata: Any) -> "PolicyDecision":
        return cls(allowed=True, reason=REASON_OK, status_code=200, metadata=metadata)

    @classmethod
    def deny(
        cls,
        reason: str,
        detail: str = "",
        status_code: int = 503,
        **metadata: Any,
    ) -> "PolicyDecision":
        return cls(
            allowed=False,
            reason=reason,
            detail=detail or reason.replace("_", " "),
            status_code=status_code,
            metadata=metadata,
        )


class PolicyPort(Protocol):
    """Single authorization decision point.

    Adapters return :class:`PolicyDecision` for one (resource, action) pair
    in a given context. Callers translate ``status_code`` to their transport
    (HTTP status, CLI exit code, MCP error envelope).
    """

    def authorize(
        self,
        *,
        actor: Actor | None,
        resource: str,
        action: str,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        ...
