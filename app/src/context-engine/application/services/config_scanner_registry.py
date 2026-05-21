"""Registry + dispatch for :class:`ConfigSourceScannerPort` instances.

Rebuild plan P4: parallels :class:`SourceConnectorRegistry`. The host
registers scanners at boot; ingestion jobs ask the registry "which
scanners handle this working tree?" and the registry fans out per file.

This module owns no IO. Scanners are pure functions of (file ref,
content); the host supplies content (git read, blob fetch). The
registry just orders the dispatch and aggregates the :class:`ScanResult`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from domain.ports.config_scanner import (
    ConfigFileRef,
    ConfigSourceScannerCapability,
    ConfigSourceScannerPort,
    ScanResult,
    ScannerClaim,
    ScannerEntity,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _AggregatedScanResult:
    """Merged result from running every scanner against a working tree."""

    entities: list[ScannerEntity] = field(default_factory=list)
    claims: list[ScannerClaim] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scanners_run: list[str] = field(default_factory=list)


class ConfigSourceScannerRegistry:
    """Register scanners; dispatch a working tree through every matching one.

    Scanners are registered explicitly during container bootstrap so
    per-host configuration (which scanners are enabled, per-pot
    overrides) stays in the boundary, not in shared code.
    """

    def __init__(self) -> None:
        self._scanners: dict[str, ConfigSourceScannerPort] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, scanner: ConfigSourceScannerPort) -> None:
        kind = scanner.kind()
        if kind in self._scanners:
            logger.warning("replacing already-registered scanner kind=%s", kind)
        self._scanners[kind] = scanner

    def get(self, kind: str) -> ConfigSourceScannerPort | None:
        return self._scanners.get(kind)

    def all(self) -> Sequence[ConfigSourceScannerPort]:
        return list(self._scanners.values())

    def capabilities(self) -> list[ConfigSourceScannerCapability]:
        return [s.capabilities() for s in self._scanners.values()]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def scanners_for(self, file_ref: ConfigFileRef) -> list[ConfigSourceScannerPort]:
        """Return the subset of registered scanners that ``handles()`` this file.

        Multiple scanners may handle the same file (e.g. a doc that is
        both an ADR and an OpenAPI spec); each runs and contributes
        independently to the aggregated result.
        """
        return [s for s in self._scanners.values() if s.handles(file_ref)]

    def scan_file(self, file_ref: ConfigFileRef) -> ScanResult:
        """Run every matching scanner against a single file; merge results.

        Scanner failures are isolated — a raise from one scanner does
        not stop the others. The failure is recorded in ``warnings``
        and the result still includes whatever the others produced.
        """
        merged_entities: list[ScannerEntity] = []
        merged_claims: list[ScannerClaim] = []
        merged_warnings: list[str] = []
        for scanner in self.scanners_for(file_ref):
            try:
                result = scanner.parse_to_claims(file_ref)
            except Exception as exc:
                logger.exception(
                    "scanner %s failed on %s: %s",
                    scanner.kind(),
                    file_ref.path,
                    exc,
                )
                merged_warnings.append(
                    f"scanner {scanner.kind()!r} raised on {file_ref.path}: {exc}"
                )
                continue
            merged_entities.extend(result.entities)
            merged_claims.extend(result.claims)
            merged_warnings.extend(result.warnings)
        return ScanResult(
            entities=tuple(merged_entities),
            claims=tuple(merged_claims),
            warnings=tuple(merged_warnings),
        )

    def scan_files(
        self, file_refs: Iterable[ConfigFileRef]
    ) -> _AggregatedScanResult:
        """Run the full registry over a batch of files.

        Returns the aggregated result + the set of scanner kinds that
        actually fired so callers can record which scanners contributed.
        """
        agg = _AggregatedScanResult()
        fired: set[str] = set()
        for file_ref in file_refs:
            for scanner in self.scanners_for(file_ref):
                fired.add(scanner.kind())
            result = self.scan_file(file_ref)
            agg.entities.extend(result.entities)
            agg.claims.extend(result.claims)
            agg.warnings.extend(result.warnings)
        agg.scanners_run = sorted(fired)
        return agg


__all__ = [
    "ConfigSourceScannerRegistry",
]
