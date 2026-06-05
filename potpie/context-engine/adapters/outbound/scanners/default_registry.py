"""Default working-tree scanner registry for the local profile.

Bundles the deterministic config scanners (P4) into one
:class:`ConfigSourceScannerRegistry` so the host can run ``ingest scan`` over a
local working tree without the standalone/managed container. Adding a scanner is
one line here.
"""

from __future__ import annotations

from adapters.outbound.scanners import (
    CodeownersScanner,
    DependencyManifestScanner,
    KubernetesManifestScanner,
    OpenApiSpecScanner,
)
from application.services.config_scanner_registry import ConfigSourceScannerRegistry


def build_default_scanner_registry() -> ConfigSourceScannerRegistry:
    """Register the built-in deterministic config scanners."""
    registry = ConfigSourceScannerRegistry()
    registry.register(CodeownersScanner())
    registry.register(DependencyManifestScanner())
    registry.register(KubernetesManifestScanner())
    registry.register(OpenApiSpecScanner())
    return registry


__all__ = ["build_default_scanner_registry"]
