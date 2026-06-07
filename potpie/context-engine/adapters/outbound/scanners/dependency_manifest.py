"""Dependency manifest scanner (rebuild plan P4).

Reads ``pyproject.toml`` / ``requirements.txt`` / ``package.json`` and
emits deterministic ``Service -[USES]-> Dependency`` claims. The service
key is inferred from the path scope (``apps/<svc>/`` etc.); the
dependency name + version come from the manifest.

The Dependency entity uses an ``EXTERNAL_ID`` identity class because a
dependency is canonically identified by ``ecosystem:name`` (e.g.
``pypi:fastapi``, ``npm:react``). We register it lazily here if the
default registry doesn't already carry it.

This scanner is intentionally simple: we extract the dependency name
and the version-constraint string, but we do not resolve versions or
crawl transitive deps — that's a future enrichment job's problem.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Iterable, Iterator, Mapping

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Python <3.11
    tomllib = None  # type: ignore[assignment]

from domain.identity import (
    IdentityClass,
    IdentityError,
    IdentitySpec,
    get_identity,
    mint_entity_key,
    register_identity,
)
from domain.path_scope import derive_scope
from domain.ports.config_scanner import (
    ConfigFileRef,
    ConfigSourceScannerCapability,
    ScanResult,
    ScannerClaim,
    ScannerEntity,
)

logger = logging.getLogger(__name__)


def _ensure_dependency_identity() -> None:
    if get_identity("Dependency") is None:
        register_identity(
            IdentitySpec(
                label="Dependency",
                klass=IdentityClass.EXTERNAL_ID,
                key_prefix="dependency",
            )
        )


_ensure_dependency_identity()


_PYPROJECT_RE = re.compile(r"(^|/)pyproject\.toml$")
_REQUIREMENTS_RE = re.compile(r"(^|/)requirements[^/]*\.txt$")
_PACKAGE_JSON_RE = re.compile(r"(^|/)package\.json$")

_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._\-]*)\s*(.*)$")
_PYPROJECT_DEP_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._\-]*)\s*(.*)$")


class DependencyManifestScanner:
    """Emit ``Service -[USES]-> Dependency`` claims from package manifests."""

    SCANNER_KIND = "dependency-manifest"
    PREDICATE = "USES"

    def kind(self) -> str:
        return self.SCANNER_KIND

    def capabilities(self) -> ConfigSourceScannerCapability:
        return ConfigSourceScannerCapability(
            kind=self.SCANNER_KIND,
            description=(
                "Reads pyproject.toml / requirements*.txt / package.json; emits "
                "deterministic Service USES Dependency claims with ecosystem-tagged "
                "dependency keys."
            ),
            handles_file_patterns=(
                "pyproject.toml",
                "requirements*.txt",
                "package.json",
                "**/pyproject.toml",
                "**/requirements*.txt",
                "**/package.json",
            ),
            emits_predicates=(self.PREDICATE,),
        )

    def handles(self, file_ref: ConfigFileRef) -> bool:
        path = file_ref.path
        return bool(
            _PYPROJECT_RE.search(path)
            or _REQUIREMENTS_RE.search(path)
            or _PACKAGE_JSON_RE.search(path)
        )

    def list_files(
        self, *, repo_name: str, working_tree_paths: Iterable[str]
    ) -> Iterable[str]:
        del repo_name
        return [
            p
            for p in working_tree_paths
            if _PYPROJECT_RE.search(p)
            or _REQUIREMENTS_RE.search(p)
            or _PACKAGE_JSON_RE.search(p)
        ]

    def parse_to_claims(self, file_ref: ConfigFileRef) -> ScanResult:
        warnings: list[str] = []
        scope = derive_scope(file_ref.path)
        if not scope.service:
            # No service scope = the manifest sits at the repo root with
            # no clue who consumes it. We still emit dependencies tied
            # to the repository, but flag that the scope is generic.
            warnings.append(
                f"dependency-manifest:{file_ref.path}: no service in path scope "
                f"— USES edges will be repo-scoped"
            )

        observed_at = file_ref.observed_at or datetime.now(tz=timezone.utc)

        if _PYPROJECT_RE.search(file_ref.path):
            deps = list(_parse_pyproject(file_ref.content, warnings))
            ecosystem = "pypi"
        elif _REQUIREMENTS_RE.search(file_ref.path):
            deps = list(_parse_requirements(file_ref.content, warnings))
            ecosystem = "pypi"
        elif _PACKAGE_JSON_RE.search(file_ref.path):
            deps = list(_parse_package_json(file_ref.content, warnings))
            ecosystem = "npm"
        else:
            return ScanResult()

        subject_key, subject_label, subject_name = _resolve_subject(
            scope_service=scope.service, repo_name=file_ref.repo_name
        )
        if subject_key is None:
            warnings.append(
                f"dependency-manifest:{file_ref.path}: no service or repo — skipping"
            )
            return ScanResult(warnings=tuple(warnings))

        entities: list[ScannerEntity] = []
        claims: list[ScannerClaim] = []
        seen_entities: set[str] = {subject_key}
        seen_edges: set[tuple[str, str]] = set()

        entities.append(
            ScannerEntity(
                entity_key=subject_key,
                label=subject_label,
                name=subject_name,
                properties={"derived_from": "dependency-manifest-scanner"},
            )
        )

        for dep_name, version_spec, kind_tag in deps:
            dep_key = _dependency_key(ecosystem=ecosystem, name=dep_name)
            if dep_key is None:
                warnings.append(
                    f"dependency-manifest:{file_ref.path}: skipping dep {dep_name!r}"
                )
                continue
            if dep_key not in seen_entities:
                entities.append(
                    ScannerEntity(
                        entity_key=dep_key,
                        label="Dependency",
                        name=dep_name,
                        properties={
                            "ecosystem": ecosystem,
                            "derived_from": "dependency-manifest-scanner",
                        },
                    )
                )
                seen_entities.add(dep_key)

            edge_key = (subject_key, dep_key)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            claims.append(
                ScannerClaim(
                    subject_key=subject_key,
                    predicate=self.PREDICATE,
                    object_key=dep_key,
                    source_ref=_source_ref(
                        repo=file_ref.repo_name,
                        path=file_ref.path,
                        ecosystem=ecosystem,
                        dep_name=dep_name,
                    ),
                    source_system="dependency-manifest",
                    fact=(
                        f"{subject_name or subject_key} uses {dep_name}"
                        + (f" {version_spec}" if version_spec else "")
                    ),
                    valid_at=observed_at,
                    evidence_strength="deterministic",
                    properties={
                        "ecosystem": ecosystem,
                        "version_spec": version_spec,
                        "dependency_kind": kind_tag,
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
# Parsers
# ---------------------------------------------------------------------------


def _parse_pyproject(
    content: str, warnings: list[str]
) -> Iterator[tuple[str, str, str]]:
    if tomllib is None:
        warnings.append("pyproject parse skipped: tomllib unavailable")
        return
    try:
        data = tomllib.loads(content)
    except Exception as exc:
        warnings.append(f"pyproject parse error: {exc}")
        return

    # PEP 621 dependencies under [project]
    project = data.get("project")
    if isinstance(project, Mapping):
        main_deps = project.get("dependencies")
        if isinstance(main_deps, list):
            for raw in main_deps:
                if isinstance(raw, str):
                    name, version = _split_dep_string(raw)
                    if name:
                        yield (name, version, "runtime")
        optional = project.get("optional-dependencies")
        if isinstance(optional, Mapping):
            for group, deps in optional.items():
                if not isinstance(deps, list):
                    continue
                for raw in deps:
                    if isinstance(raw, str):
                        name, version = _split_dep_string(raw)
                        if name:
                            yield (name, version, f"optional:{group}")

    # Poetry-style [tool.poetry.dependencies]
    tool = data.get("tool")
    if isinstance(tool, Mapping):
        poetry = tool.get("poetry")
        if isinstance(poetry, Mapping):
            for kind_tag, key in (
                ("runtime", "dependencies"),
                ("dev", "dev-dependencies"),
            ):
                deps = poetry.get(key)
                if not isinstance(deps, Mapping):
                    continue
                for name, version in deps.items():
                    if not isinstance(name, str):
                        continue
                    if name.lower() == "python":
                        continue
                    yield (name, str(version) if version is not None else "", kind_tag)


def _parse_requirements(
    content: str, warnings: list[str]
) -> Iterator[tuple[str, str, str]]:
    del warnings
    for raw_line in content.splitlines():
        # Drop comments + inline comments
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "//")):
            # Skip --options like -e, -r, --index-url etc.
            continue
        match = _REQ_NAME_RE.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        rest = match.group(2).strip()
        # Strip extras: "foo[bar]" → "foo"
        if "[" in name:
            name = name.split("[", 1)[0]
        if name:
            yield (name, rest, "runtime")


def _parse_package_json(
    content: str, warnings: list[str]
) -> Iterator[tuple[str, str, str]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        warnings.append(f"package.json parse error: {exc}")
        return
    if not isinstance(data, Mapping):
        warnings.append("package.json: top-level is not an object")
        return
    for kind_tag, key in (
        ("runtime", "dependencies"),
        ("dev", "devDependencies"),
        ("peer", "peerDependencies"),
        ("optional", "optionalDependencies"),
    ):
        deps = data.get(key)
        if not isinstance(deps, Mapping):
            continue
        for name, version in deps.items():
            if not isinstance(name, str):
                continue
            yield (name, str(version) if version is not None else "", kind_tag)


def _split_dep_string(raw: str) -> tuple[str, str]:
    """Split ``"fastapi>=0.100,<0.110"`` into ``("fastapi", ">=0.100,<0.110")``."""
    match = _PYPROJECT_DEP_RE.match(raw)
    if not match:
        return ("", "")
    name = match.group(1)
    if "[" in name:
        name = name.split("[", 1)[0]
    return (name, match.group(2).strip())


# ---------------------------------------------------------------------------
# Entity / key helpers
# ---------------------------------------------------------------------------


def _resolve_subject(
    *, scope_service: str | None, repo_name: str | None
) -> tuple[str | None, str, str | None]:
    if scope_service:
        spec = get_identity("Service")
        if spec is not None:
            try:
                return (
                    mint_entity_key(spec, name=scope_service),
                    "Service",
                    scope_service,
                )
            except IdentityError:
                return (None, "Service", scope_service)
    if repo_name:
        spec = get_identity("Repository")
        if spec is not None:
            try:
                return (mint_entity_key(spec, name=repo_name), "Repository", None)
            except IdentityError:
                return (None, "Repository", None)
    return (None, "Service", None)


def _dependency_key(*, ecosystem: str, name: str) -> str | None:
    spec = get_identity("Dependency")
    if spec is None:
        return None
    try:
        return mint_entity_key(spec, external_id=name, extra_segments=(ecosystem,))
    except IdentityError:
        return None


def _source_ref(*, repo: str | None, path: str, ecosystem: str, dep_name: str) -> str:
    repo_seg = repo or "unknown-repo"
    return f"dependency-manifest:{repo_seg}:{path}:{ecosystem}:{dep_name}"


__all__ = ["DependencyManifestScanner"]
