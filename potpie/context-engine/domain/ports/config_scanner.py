"""Config-source scanner port (rebuild plan P4).

A :class:`ConfigSourceScannerPort` is a *deterministic* claim emitter:
given a working tree (or a file ref + its content), it produces a
sequence of :class:`Claim` records that the canonical writer turns
into ``:RELATES_TO`` edges with ``evidence_strength="deterministic"``.

Scanners are distinct from :class:`SourceConnectorPort` in three ways:

1. No webhooks. Scanners trigger from commit-on-main (or scheduled
   tick), not from event payloads.
2. No fetch. Scanners read from a working tree the host provides;
   they never make API calls.
3. Output is a :class:`Claim`, not a ``ContextEvent`` or
   ``ReconciliationPlan``. The claim is the on-graph fact — no LLM
   round-trip between scan and write.

Why this matters: the proper POC's F2 / F1 failures came from the LLM
extractor seeing only the body text and missing the file-path scope.
Scanners stamp scope deterministically (via :class:`PathScope` from
``domain.path_scope``) *before* any LLM enrichment runs.

P4 plans four V1 scanners:

- KubernetesManifestScanner (F1 fix — emits Deployment OF_SERVICE Service)
- CodeownersScanner (F2 fix — emits Service OWNED_BY Person)
- DependencyManifestScanner (Service USES Dependency)
- OpenApiSpecScanner (Service EXPOSES APIContract)

This port defines the contract; the adapter modules implement each
scanner. The application layer talks to a registry of scanners,
parallel to the SourceConnectorRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class ConfigFileRef:
    """One file the scanner can parse.

    ``content`` is supplied by the host (git tree read, working-tree
    read, blob fetch); scanners do not perform IO themselves so they
    stay pure / testable.
    """

    path: str
    content: str
    repo_name: str | None = None
    commit_sha: str | None = None
    observed_at: datetime | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScannerEntity:
    """An entity the scanner deterministically extracted."""

    entity_key: str
    label: str
    name: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScannerClaim:
    """One :class:`RELATES_TO` claim the scanner produced.

    The scanner is responsible for filling ``source_ref`` so re-scans
    of the same file (same commit SHA + path) update the existing edge
    idempotently. ``valid_at`` is the file's effective time (commit
    timestamp); ``observed_at`` is when the scan ran.
    """

    subject_key: str
    predicate: str
    object_key: str
    source_ref: str
    source_system: str
    fact: str
    valid_at: datetime
    evidence_strength: str = "deterministic"
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Aggregated output from one ``parse_to_claims`` call.

    Entities are upserted first, then claims. A scanner that produces
    no claims for a given file returns an empty :class:`ScanResult` —
    callers should not interpret that as failure.
    """

    entities: Sequence[ScannerEntity] = ()
    claims: Sequence[ScannerClaim] = ()
    warnings: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class ConfigSourceScannerCapability:
    """Aggregated manifest for ``context_status`` (parallel to SourceCapability)."""

    kind: str
    description: str
    handles_file_patterns: tuple[str, ...] = ()
    emits_predicates: tuple[str, ...] = ()


class ConfigSourceScannerPort(Protocol):
    """One deterministic config-file → claim emitter behind a stable contract."""

    def kind(self) -> str:
        """Stable scanner identifier (e.g. ``"kubernetes-manifest"``)."""
        ...

    def capabilities(self) -> ConfigSourceScannerCapability:
        """Advertise file patterns + emitted predicates."""
        ...

    def handles(self, file_ref: ConfigFileRef) -> bool:
        """Return True iff this scanner should parse the given file."""
        ...

    def list_files(
        self, *, repo_name: str, working_tree_paths: Iterable[str]
    ) -> Iterable[str]:
        """Filter a working tree's paths to those the scanner handles.

        The host is responsible for enumerating the working tree (e.g.
        ``git ls-tree``); the scanner just picks out matching paths.
        """
        ...

    def parse_to_claims(self, file_ref: ConfigFileRef) -> ScanResult:
        """Parse one config file into deterministic entities + claims.

        Must be idempotent: re-running with the same ``(path, content)``
        produces the same :class:`ScanResult`. Scanners that cannot
        parse a malformed file SHOULD emit a ``warnings`` entry and
        return an otherwise-empty result.
        """
        ...


__all__ = [
    "ConfigFileRef",
    "ConfigSourceScannerCapability",
    "ConfigSourceScannerPort",
    "ScanResult",
    "ScannerClaim",
    "ScannerEntity",
]
