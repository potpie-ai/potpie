"""Server-defined agent list scopes (FW003 / CWE-639).

Client query flags must not escalate visibility. Callers pick a fixed mode;
the server maps each mode to include rules.
"""

from enum import Enum


class AgentListMode(str, Enum):
    """Authorized listing profiles for available agents."""

    # Chat / runtime picker: system agents + caller's own + shared-with-caller.
    RUNTIME = "runtime"
    # Management ("All Agents"): only agents owned by the caller.
    OWNED = "owned"


# Legacy privilege-escalation params — accepted for compat but never authorized from.
LEGACY_PRIVILEGE_QUERY_PARAMS = (
    "list_system_agents",
    "include_public",
    "include_shared",
)
