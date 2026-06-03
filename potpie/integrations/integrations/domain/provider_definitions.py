"""Runtime metadata for a registered integration provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PortKind = Literal["source_control", "issue_tracker"]


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    """Registered provider (catalog entry). Handlers and adapters are wired in later phases."""

    id: str
    display_name: str
    capabilities: tuple[str, ...]
    """e.g. ``("code_host",)``, ``("issue_tracker",)``."""
    source_kinds: tuple[str, ...]
    """Attachment kinds this provider supports, e.g. ``("repository",)``."""
    port_kind: PortKind
    """Which domain port the adapter implements: SCM vs issue tracker."""
    oss_available: bool = True
