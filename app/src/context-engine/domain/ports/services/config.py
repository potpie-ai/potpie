"""``ConfigService`` — the workspace/config lifecycle seam.

Owns the local home directory (``~/.potpie`` or ``$CONTEXT_ENGINE_HOME``) and the
config file written during ``potpie setup``. The first setup step; everything
downstream resolves paths through here.

    setup step 1 (hard): config.ensure_home(); config.write_defaults(plan)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from domain.lifecycle import SetupPlan


class ConfigService(Protocol):
    """Local config/workspace provisioning + get/set."""

    def ensure_home(self) -> Path:
        """Create the home directory tree if absent; return its path."""
        ...

    def write_defaults(self, plan: SetupPlan) -> Path:
        """Write the default config file (profile, backend, paths) from the plan,
        preserving any values a user already set. Returns the config path."""
        ...

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...

    def probe(self) -> Mapping[str, Any]:
        """Cheap state for ``doctor``/``status`` (home path, config presence)."""
        ...


__all__ = ["ConfigService"]
