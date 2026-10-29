"""Resolve the home directory where agents read their skills and instructions.

``POTPIE_HARNESS_HOME`` overrides the user's home directory for installs in
tests and CI. The test suite sets it in an autouse fixture to keep tests from
overwriting the user's skills. ``CONTEXT_ENGINE_HOME`` relocates Potpie's
graph and pot state only, because agents still read their own home directories.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Override the install home independently of ``CONTEXT_ENGINE_HOME``.
HARNESS_HOME_ENV = "POTPIE_HARNESS_HOME"


def harness_home() -> Path:
    """Return the parent directory of agent configuration directories."""
    raw = os.getenv(HARNESS_HOME_ENV)
    if raw and raw.strip():
        return Path(raw.strip()).expanduser()
    return Path.home()


__all__ = ["HARNESS_HOME_ENV", "harness_home"]
