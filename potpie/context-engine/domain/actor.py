"""Actor: who (user/system) and how (surface/client) submitted an event.

Threaded through every ingestion path (CLI, MCP, HTTP, webhook) so events,
reconciliation runs, and graph episodes all retain a first-class answer to
"who produced this".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ActorSurface = Literal["cli", "mcp", "http", "webhook", "system"]
"""Inbound surface. ``http`` = direct API caller; ``cli`` / ``mcp`` are clients
that go through HTTP but self-declare via ``X-Potpie-Client``; ``webhook`` is an
external provider; ``system`` is internal jobs (backfills, workers)."""

ActorAuthMethod = Literal["api_key", "session", "webhook_signature", "system"]


_ACTOR_SURFACE_BY_VALUE: dict[str, ActorSurface] = {
    "cli": "cli",
    "mcp": "mcp",
    "http": "http",
    "webhook": "webhook",
    "system": "system",
}

_ACTOR_AUTH_METHOD_BY_VALUE: dict[str, ActorAuthMethod] = {
    "api_key": "api_key",
    "session": "session",
    "webhook_signature": "webhook_signature",
    "system": "system",
}

VALID_SURFACES: frozenset[str] = frozenset(_ACTOR_SURFACE_BY_VALUE)


@dataclass(frozen=True, slots=True)
class Actor:
    """The principal that produced an event.

    ``user_id`` is the stable Potpie user id for authenticated humans, or a
    synthetic id for non-human actors (``webhook:github:<delivery>``,
    ``system:<job>``). ``client_name`` identifies the concrete program, e.g.
    ``claude-code``, ``cursor``, ``potpie-cli`` — free-form metadata.
    """

    user_id: str
    surface: ActorSurface
    client_name: str | None = None
    auth_method: ActorAuthMethod = "api_key"

    def to_properties(self) -> dict[str, Any]:
        """Render as Neo4j ``actor_*`` properties (skip empties)."""
        out: dict[str, Any] = {
            "actor_user_id": self.user_id,
            "actor_surface": self.surface,
            "actor_auth_method": self.auth_method,
        }
        if self.client_name:
            out["actor_client_name"] = self.client_name
        return out

    def to_payload(self) -> dict[str, Any]:
        """JSON-serializable view for API responses."""
        return {
            "user_id": self.user_id,
            "surface": self.surface,
            "client_name": self.client_name,
            "auth_method": self.auth_method,
        }


def normalize_surface(value: str | None) -> ActorSurface | None:
    """Case-insensitive normalization; returns None for invalid/empty input."""
    if not value:
        return None
    v = value.strip().lower()
    return _ACTOR_SURFACE_BY_VALUE.get(v)


def normalize_auth_method(value: str | None) -> ActorAuthMethod | None:
    """Case-insensitive normalization; returns None for invalid/empty input."""
    if not value:
        return None
    v = value.strip().lower()
    return _ACTOR_AUTH_METHOD_BY_VALUE.get(v)


SYSTEM_ACTOR: Actor = Actor(
    user_id="system",
    surface="system",
    client_name="context-engine",
    auth_method="system",
)
"""Fallback used by internal jobs (backfills, workers) that have no human actor."""
