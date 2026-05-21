"""Kubernetes manifest scanner (rebuild plan P4 / F1 fix).

Proper-POC F1: INFRA reader returned 0% coverage on "which environment
runs auth-svc?" because no edge in the graph linked a Deployment to a
Service. The PR-deployed Activity carried a deployment id but the
Service ↔ Deployment join was missing in the ontology.

This scanner reads a Kubernetes Deployment / StatefulSet / DaemonSet
manifest and emits two deterministic claims:

- ``Deployment -[OF_SERVICE]-> Service`` (singleton: a deployment runs
  one service in this ontology)
- ``Deployment -[DEPLOYED_TO]-> Environment`` (singleton: env stamped
  from the path scope, see :mod:`domain.path_scope`)

The deployment's external_id is the manifest's ``metadata.name``; the
service is inferred from the manifest's labels (preferred:
``app.kubernetes.io/name``, fallback: ``app``) and corroborated by the
path scope. The environment comes from the path (e.g.
``clusters/prod/auth-svc.yaml`` → ``prod``).

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
    """Emits ``Deployment OF_SERVICE Service`` + ``Deployment DEPLOYED_TO Env``."""

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
                "Reads Kubernetes workload manifests; emits Deployment OF_SERVICE "
                "Service and Deployment DEPLOYED_TO Environment (F1 fix)."
            ),
            handles_file_patterns=(
                "clusters/**/*.yaml",
                "k8s/**/*.yaml",
                "kubernetes/**/*.yaml",
                "manifests/**/*.yaml",
                "deployments/**/*.yaml",
            ),
            emits_predicates=("OF_SERVICE", "DEPLOYED_TO"),
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
            namespace = metadata.get("namespace")
            if not isinstance(name, str) or not name.strip():
                warnings.append(
                    f"kubernetes-manifest:{file_ref.path} doc#{doc_index}: missing metadata.name"
                )
                continue

            service_name = _infer_service_name(
                doc=doc, metadata=metadata, scope_service=scope.service, fallback=name
            )
            environment = scope.environment

            deployment_key = _deployment_key(
                name=name, namespace=namespace if isinstance(namespace, str) else None
            )
            if deployment_key is None:
                warnings.append(
                    f"kubernetes-manifest:{file_ref.path}: could not mint deployment key"
                )
                continue

            if deployment_key not in seen_entities:
                entities.append(
                    ScannerEntity(
                        entity_key=deployment_key,
                        label="Deployment",
                        name=name,
                        properties={
                            "kind": kind,
                            "api_version": api_version,
                            "namespace": namespace
                            if isinstance(namespace, str)
                            else None,
                            "derived_from": "kubernetes-manifest-scanner",
                        },
                    )
                )
                seen_entities.add(deployment_key)

            # ----- OF_SERVICE -----
            service_key = _service_key(service_name)
            if service_key:
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
                claims.append(
                    ScannerClaim(
                        subject_key=deployment_key,
                        predicate="OF_SERVICE",
                        object_key=service_key,
                        source_ref=_source_ref(
                            repo=file_ref.repo_name,
                            path=file_ref.path,
                            doc_index=doc_index,
                            predicate="of_service",
                        ),
                        source_system=self._source_system,
                        fact=f"deployment {name} runs service {service_name}",
                        valid_at=valid_at,
                        evidence_strength="deterministic",
                        properties={
                            "kind": kind,
                            "scope_service": scope.service,
                            "observed_at": observed_at,
                        },
                    )
                )
            else:
                warnings.append(
                    f"kubernetes-manifest:{file_ref.path}: could not infer service for "
                    f"deployment {name!r}"
                )

            # ----- DEPLOYED_TO -----
            env_key = _environment_key(environment) if environment else None
            if env_key:
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
                        subject_key=deployment_key,
                        predicate="DEPLOYED_TO",
                        object_key=env_key,
                        source_ref=_source_ref(
                            repo=file_ref.repo_name,
                            path=file_ref.path,
                            doc_index=doc_index,
                            predicate="deployed_to",
                        ),
                        source_system=self._source_system,
                        fact=f"deployment {name} deployed to {environment}",
                        valid_at=valid_at,
                        evidence_strength="deterministic",
                        properties={
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


def _deployment_key(*, name: str, namespace: str | None) -> str | None:
    spec = get_identity("Deployment")
    if spec is None:
        return None
    try:
        if namespace:
            return mint_entity_key(
                spec, external_id=name, extra_segments=("k8s", namespace)
            )
        return mint_entity_key(spec, external_id=name, extra_segments=("k8s",))
    except IdentityError:
        return None


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
