"""Where an agent harness keeps its files — ``$HOME`` unless told otherwise.

``CONTEXT_ENGINE_HOME`` relocates *Potpie's own* state (pot store, resource
tree, embedded graph db). It deliberately does **not** relocate this one.
Someone who moves their state to another volume still runs Claude Code out of
``~/.claude``, and rooting the install under the state home instead would write
eleven skill files somewhere no harness ever looks, then report success — the
same "exit 0 means no exception escaped" shape the rest of this branch exists to
remove, aimed at the files whose entire purpose is being read by something else.

Which left every sandboxed run — CI, Potpie's own test suite, an agent driving
the CLI inside a scratch ``CONTEXT_ENGINE_HOME`` — installing into the
developer's real ``~/.claude``, ``~/.cursor``, ``~/.agents`` and
``~/.config/opencode``, overwriting whatever versions were there. That is the
case ``POTPIE_HARNESS_HOME`` exists for: an explicit knob, so a redirect only
ever happens because someone asked for one. The test suites pin it in an autouse
fixture, which is what makes running them on a dev machine safe.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Set this to install harness files somewhere other than the real home
#: directory. Tests and CI want it; ``CONTEXT_ENGINE_HOME`` deliberately does
#: not imply it.
HARNESS_HOME_ENV = "POTPIE_HARNESS_HOME"


def harness_home() -> Path:
    """The directory ``~/.claude`` & friends hang off of."""
    raw = os.getenv(HARNESS_HOME_ENV)
    if raw and raw.strip():
        return Path(raw.strip()).expanduser()
    return Path.home()


__all__ = ["HARNESS_HOME_ENV", "harness_home"]
