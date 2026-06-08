"""Config-source scanners (rebuild plan P4).

Each scanner is a pure ``ConfigFileRef → ScanResult`` function that
emits deterministic claims into the canonical writer.
"""

from adapters.outbound.scanners.codeowners import CodeownersScanner
from adapters.outbound.scanners.dependency_manifest import DependencyManifestScanner
from adapters.outbound.scanners.kubernetes_manifest import KubernetesManifestScanner
from adapters.outbound.scanners.openapi_spec import OpenApiSpecScanner

__all__ = [
    "CodeownersScanner",
    "DependencyManifestScanner",
    "KubernetesManifestScanner",
    "OpenApiSpecScanner",
]
