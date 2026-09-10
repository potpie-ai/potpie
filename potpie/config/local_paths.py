"""Root Potpie filesystem path resolution."""

from __future__ import annotations

import os
from pathlib import Path


def default_home() -> Path:
    """Return the configured Potpie home, or ``~/.potpie`` by default."""
    raw = os.getenv("CONTEXT_ENGINE_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".potpie"


def harness_home() -> Path:
    """Return the home under which *harness* directories live.

    Skills install into the agent's own tree (``~/.claude/skills`` and
    friends), which is a different thing from Potpie's state home — so
    ``CONTEXT_ENGINE_HOME`` deliberately does not move them, and a run under
    an "isolated" ``CONTEXT_ENGINE_HOME`` still rewrote the operator's real
    skill set. ``POTPIE_HARNESS_HOME`` is the knob that does relocate them, so
    tests and trial installs can be contained without overriding ``HOME``.
    """
    raw = os.getenv("POTPIE_HARNESS_HOME")
    return Path(raw).expanduser() if raw else Path.home()


__all__ = ["default_home", "harness_home"]
