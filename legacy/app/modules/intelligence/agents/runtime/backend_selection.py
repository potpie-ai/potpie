"""Select the execution backend (celery default, hatchet for allowlisted agents).

Hatchet is opt-in and fail-safe: unless ``AGENT_TASK_BACKEND=hatchet`` is set AND
the agent is on the allowlist, runs go to Celery.

Flags (read at call time):
- ``AGENT_TASK_BACKEND``      = ``celery`` | ``hatchet``   (default ``celery``)
- ``HATCHET_AGENT_ALLOWLIST`` = comma-separated agent ids (default ``debugging_agent``)

Note: the debug agent's id is ``debugging_agent`` (see agents_service), so that is the
allowlist default — the plan's shorthand "debug_agent" would not match any agent.
"""

from __future__ import annotations

import os
from typing import Optional

CELERY = "celery"
HATCHET = "hatchet"

DEFAULT_ALLOWLIST = "debugging_agent"


def hatchet_mode_enabled() -> bool:
    """True when AGENT_TASK_BACKEND=hatchet (kept off for normal `make dev`)."""
    return (os.getenv("AGENT_TASK_BACKEND") or CELERY).strip().lower() == HATCHET


def hatchet_allowlist() -> set[str]:
    """Agent ids permitted to run on Hatchet."""
    raw = os.getenv("HATCHET_AGENT_ALLOWLIST")
    if raw is None:
        raw = DEFAULT_ALLOWLIST
    return {a.strip() for a in raw.split(",") if a.strip()}


def select_backend(agent_id: Optional[str]) -> str:
    """Return ``"hatchet"`` for allowlisted agents in hatchet-mode, else ``"celery"``."""
    if not agent_id:
        return CELERY
    if hatchet_mode_enabled() and agent_id in hatchet_allowlist():
        return HATCHET
    return CELERY
