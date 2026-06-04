"""Kubernetes manifest scanner (topology core).

Answers "which environment runs service X?" with a real edge instead of
0% coverage. Reads a Kubernetes Deployment / StatefulSet / DaemonSet
manifest and emits one deterministic claim per workload:

- ``Service -[DEPLOYED_TO]-> Environment`` — the service runs in the
  environment stamped from the path scope (see :mod:`domain.path_scope`).
  The ``environment`` is also stamped on the edge so the topology reader
  can filter by it directly.

The service is inferred from the manifest's labels (preferred:
``app.kubernetes.io/name``, fallback: ``app``) and corroborated by the
path scope; the environment comes from the path (e.g.
``clusters/prod/auth-svc.yaml`` → ``prod``). The workload's
``metadata.name`` + kind are carried as edge provenance properties — the
deployment object itself is not a node in the topology ontology (the
*fact* that a service runs in an env is the edge; the deploy *event*
belongs to a later timeline tier).

YAML is parsed with PyYAML's ``safe_load_all`` — we never execute. Files
that are not Kubernetes (no ``apiVersion`` + ``kind``) are silently
skipped; files that *are* Kubernetes but not a workload kind we know
how to interpret produce a warning + no claims.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - pyyaml is a hard dep in prod
    yaml = None  # type: ignore[assignment]

from domain.identity import IdentityError, get_identity, mint_entity_key
from domain.path_scope import derive_scope
from domain.ports.config_scanner import (
    ConfigFileRef,
    ConfigSourceScannerCapability,
    ScanResult,
    ScannerClaim,
    ScannerEntity,
)

logger = logging.getLogger(__name__)


_WORKLOAD_KINDS = frozenset(
    {
        "Deployment",
        "StatefulSet",
        "DaemonSet",
        "Job",
        "CronJob",
    }
)
_PATH_PATTERN_RE = re.compile(
    r"(^|/)(clusters|k8s|kubernetes|manifests|deploy(?:ments)?|helm/.+/templates)/.+\.(ya?ml)$",
    re.IGNORECASE,
)


class KubernetesManifestScanner:
    """Emits ``Service DEPLOYED_TO Environment`` from workload manifests."""

    SCANNER_KIND = "kubernetes-manifest"

    def __init__(self, *, source_system: str = "kubernetes") -> None:
        self._source_system = source_system

    # ------------------------------------------------------------------
    # ConfigSourceScannerPort
    # ------------------------------------------------------------------
    def kind(self) -> str:
        return self.SCANNER_KIND

    def capabilities(self) -> ConfigSourceScannerCapability:
        return ConfigSourceScannerCapability(
            kind=self.SCANNER_KIND,
            description=(
                "Reads Kubernetes workload manifests; emits Service DEPLOYED_TO "
                "Environment with the environment stamped on the edge."
            ),
            handles_file_patterns=(
                "clusters/**/*.yaml",
                "k8s/**/*.yaml",
                "kubernetes/**/*.yaml",
                "manifests/**/*.yaml",
                "deployments/**/*.yaml",
            ),
            emits_predicates=("DEPLOYED_TO",),
        )

    def handles(self, file_ref: ConfigFileRef) -> bool:
        # YAML extension is necessary but not sufficient — we still
        # confirm the doc is a Kubernetes workload at parse time. We
        # match by path shape here to keep the dispatch cheap.
        return bool(_PATH_PATTERN_RE.search(file_ref.path)) and yaml is not None

    def list_files(
        self, *, repo_name: str, working_tree_paths: Iterable[str]
    ) -> Iterable[str]:
        del repo_name
        return [p for p in working_tree_paths if _PATH_PATTERN_RE.search(p)]

    def parse_to_claims(self, file_ref: ConfigFileRef) -> ScanResult:
        if yaml is None:
            return ScanResult(
                warnings=("kubernetes-manifest: pyyaml not installed — scanner disabled",)
            )

        warnings: list[str] = []
        try:
            documents = list(yaml.safe_load_all(file_ref.content))
        except yaml.YAMLError as exc:
            return ScanResult(
                warnings=(
                    f"kubernetes-manifest:{file_ref.path}: YAML parse error: {exc}",
                )
            )

        scope = derive_scope(file_ref.path)
        observed_at = file_ref.observed_at or datetime.now(tz=timezone.utc)
        valid_at = observed_at

        entities: list[ScannerEntity] = []
        claims: list[ScannerClaim] = []
        seen_entities: set[str] = set()

        for doc_index, doc in enumerate(documents):
            if not isinstance(doc, Mapping):
                continue
            api_version = doc.get("apiVersion")
            kind = doc.get("kind")
            if not isinstance(api_version, str) or not isinstance(kind, str):
                continue
            if kind not in _WORKLOAD_KINDS:
                continue

            metadata = doc.get("metadata")
            if not isinstance(metadata, Mapping):
                warnings.append(
                    f"kubernetes-manifest:{file_ref.path} doc#{doc_index}: missing metadata"
                )
                continue

            name = metadata.get("name")
            if not isinstance(name, str) or not name.strip():
                warnings.append(
                    f"kubernetes-manifest:{file_ref.path} doc#{doc_index}: missing metadata.name"
                )
                continue

            service_name = _infer_service_name(
                doc=doc, metadata=metadata, scope_service=scope.service, fallback=name
            )
            environment = scope.environment

            service_key = _service_key(service_name)
            if service_key is None:
                warnings.append(
                    f"kubernetes-manifest:{file_ref.path}: could not infer service for "
                    f"workload {name!r}"
                )
                continue

            if service_key not in seen_entities:
                entities.append(
                    ScannerEntity(
                        entity_key=service_key,
                        label="Service",
                        name=service_name,
                        properties={"derived_from": "kubernetes-manifest-scanner"},
                    )
                )
                seen_entities.add(service_key)

            # ----- Service DEPLOYED_TO Environment -----
            env_key = _environment_key(environment) if environment else None
            if env_key is None:
                warnings.append(
                    f"kubernetes-manifest:{file_ref.path}: no environment in path scope "
                    f"for workload {name!r}; skipping DEPLOYED_TO"
                )
                continue
            if env_key not in seen_entities:
                entities.append(
                    ScannerEntity(
                        entity_key=env_key,
                        label="Environment",
                        name=environment,
                        properties={"derived_from": "kubernetes-manifest-scanner"},
                    )
                )
                seen_entities.add(env_key)
            claims.append(
                ScannerClaim(
                    subject_key=service_key,
                    predicate="DEPLOYED_TO",
                    object_key=env_key,
                    source_ref=_source_ref(
                        repo=file_ref.repo_name,
                        path=file_ref.path,
                        doc_index=doc_index,
                        predicate="deployed_to",
                    ),
                    source_system=self._source_system,
                    fact=f"service {service_name} deployed to {environment}",
                    valid_at=valid_at,
                    evidence_strength="deterministic",
                    properties={
                        "environment": environment,
                        "workload": name,
                        "workload_kind": kind,
                        "observed_at": observed_at,
                    },
                )
            )

        return ScanResult(
            entities=tuple(entities),
            claims=tuple(claims),
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _infer_service_name(
    *,
    doc: Mapping[str, Any],
    metadata: Mapping[str, Any],
    scope_service: str | None,
    fallback: str,
) -> str:
    """Resolve the service name a workload runs.

    Preference order, most-specific first:

    1. ``spec.selector.matchLabels.app.kubernetes.io/name``
    2. ``metadata.labels.app.kubernetes.io/name``
    3. ``metadata.labels.app``
    4. ``PathScope.service`` (deterministic from file location)
    5. ``metadata.name`` (the deployment name itself — last resort)
    """
    labels = metadata.get("labels") if isinstance(metadata, Mapping) else None
    if isinstance(labels, Mapping):
        for key in ("app.kubernetes.io/name", "app"):
            value = labels.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    spec = doc.get("spec")
    if isinstance(spec, Mapping):
        selector = spec.get("selector")
        if isinstance(selector, Mapping):
            match_labels = selector.get("matchLabels")
            if isinstance(match_labels, Mapping):
                for key in ("app.kubernetes.io/name", "app"):
                    value = match_labels.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

    if scope_service:
        return scope_service

    return fallback


def _service_key(service_name: str | None) -> str | None:
    if not service_name:
        return None
    spec = get_identity("Service")
    if spec is None:
        return None
    try:
        return mint_entity_key(spec, name=service_name)
    except IdentityError:
        return None


def _environment_key(name: str | None) -> str | None:
    if not name:
        return None
    spec = get_identity("Environment")
    if spec is None:
        return None
    try:
        return mint_entity_key(spec, name=name)
    except IdentityError:
        return None


def _source_ref(
    *, repo: str | None, path: str, doc_index: int, predicate: str
) -> str:
    repo_seg = repo or "unknown-repo"
    return f"kubernetes-manifest:{repo_seg}:{path}:doc{doc_index}:{predicate}"


__all__ = ["KubernetesManifestScanner"]
