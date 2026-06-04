"""OpenAPI spec scanner (rebuild plan P4).

Reads an OpenAPI 3.x spec (JSON or YAML) and emits ``Service -[EXPOSES]->
APIContract`` claims, one per ``(path, method)`` operation. The service
key comes from the path-scope (``services/<svc>/openapi.yaml`` etc.);
the APIContract entity carries a deterministic key built from the
service + path + method, so re-scans converge.

This scanner is intentionally a thin reader — full spec parsing (request
shapes, response codes, security) would balloon the entity store
without obvious reader demand. We extract identity + summary today;
deeper fields can land in a follow-up via ``properties`` or via a
distinct enrichment job.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

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


def _ensure_apicontract_identity() -> None:
    if get_identity("APIContract") is None:
        register_identity(
            IdentitySpec(
                label="APIContract",
                klass=IdentityClass.EXTERNAL_ID,
                key_prefix="api_contract",
            )
        )


_ensure_apicontract_identity()


_FILENAME_RE = re.compile(
    r"(^|/)(openapi|swagger)\.(ya?ml|json)$", re.IGNORECASE
)

_HTTP_METHODS = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
)


class OpenApiSpecScanner:
    """Emit one ``Service -[EXPOSES]-> APIContract`` claim per operation."""

    SCANNER_KIND = "openapi-spec"
    PREDICATE = "EXPOSES"

    def __init__(self, *, source_system: str = "openapi") -> None:
        self._source_system = source_system

    def kind(self) -> str:
        return self.SCANNER_KIND

    def capabilities(self) -> ConfigSourceScannerCapability:
        return ConfigSourceScannerCapability(
            kind=self.SCANNER_KIND,
            description=(
                "Reads OpenAPI 3.x specs (yaml/json); emits Service EXPOSES "
                "APIContract claims (one per operation)."
            ),
            handles_file_patterns=(
                "openapi.yaml",
                "openapi.yml",
                "openapi.json",
                "swagger.yaml",
                "**/openapi.{yaml,yml,json}",
            ),
            emits_predicates=(self.PREDICATE,),
        )

    def handles(self, file_ref: ConfigFileRef) -> bool:
        return bool(_FILENAME_RE.search(file_ref.path))

    def list_files(
        self, *, repo_name: str, working_tree_paths: Iterable[str]
    ) -> Iterable[str]:
        del repo_name
        return [p for p in working_tree_paths if _FILENAME_RE.search(p)]

    def parse_to_claims(self, file_ref: ConfigFileRef) -> ScanResult:
        warnings: list[str] = []
        observed_at = file_ref.observed_at or datetime.now(tz=timezone.utc)
        spec = _load_spec(file_ref, warnings)
        if spec is None:
            return ScanResult(warnings=tuple(warnings))

        scope = derive_scope(file_ref.path)
        info = spec.get("info") if isinstance(spec, Mapping) else None
        info_title = (
            info.get("title")
            if isinstance(info, Mapping) and isinstance(info.get("title"), str)
            else None
        )
        service_name = scope.service or _slugify(info_title) if info_title else scope.service
        if not service_name and file_ref.repo_name:
            # Fall back to using the repo slug as the service name (last resort).
            service_name = file_ref.repo_name

        if not service_name:
            warnings.append(
                f"openapi-spec:{file_ref.path}: cannot infer service "
                f"(no path scope, no info.title, no repo) — skipping"
            )
            return ScanResult(warnings=tuple(warnings))

        service_spec = get_identity("Service")
        contract_spec = get_identity("APIContract")
        if service_spec is None or contract_spec is None:
            return ScanResult(
                warnings=tuple(
                    warnings
                    + ["openapi-spec: identity registry missing Service/APIContract"]
                )
            )

        try:
            service_key = mint_entity_key(service_spec, name=service_name)
        except IdentityError as exc:
            return ScanResult(
                warnings=tuple(
                    warnings + [f"openapi-spec: service key minting failed: {exc}"]
                )
            )

        entities: list[ScannerEntity] = [
            ScannerEntity(
                entity_key=service_key,
                label="Service",
                name=service_name,
                properties={"derived_from": "openapi-spec-scanner"},
            )
        ]
        claims: list[ScannerClaim] = []
        seen_contract_keys: set[str] = set()

        paths = spec.get("paths") if isinstance(spec, Mapping) else None
        if not isinstance(paths, Mapping):
            warnings.append(
                f"openapi-spec:{file_ref.path}: 'paths' missing or not a mapping"
            )
            return ScanResult(
                entities=tuple(entities),
                claims=(),
                warnings=tuple(warnings),
            )

        for raw_path, item in paths.items():
            if not isinstance(raw_path, str) or not isinstance(item, Mapping):
                continue
            for method_key, operation in item.items():
                if not isinstance(method_key, str):
                    continue
                method = method_key.lower()
                if method not in _HTTP_METHODS:
                    continue
                if not isinstance(operation, Mapping):
                    continue

                contract_key = _contract_key(
                    contract_spec=contract_spec,
                    service_name=service_name,
                    path=raw_path,
                    method=method,
                )
                if contract_key is None:
                    warnings.append(
                        f"openapi-spec:{file_ref.path}: failed to mint contract "
                        f"key for {method.upper()} {raw_path}"
                    )
                    continue
                if contract_key in seen_contract_keys:
                    continue
                seen_contract_keys.add(contract_key)

                summary = _summary_or_op_id(operation, default=f"{method.upper()} {raw_path}")
                op_id = operation.get("operationId")

                entities.append(
                    ScannerEntity(
                        entity_key=contract_key,
                        label="APIContract",
                        name=summary,
                        properties={
                            "method": method.upper(),
                            "path": raw_path,
                            "operation_id": op_id if isinstance(op_id, str) else None,
                            "derived_from": "openapi-spec-scanner",
                        },
                    )
                )

                claims.append(
                    ScannerClaim(
                        subject_key=service_key,
                        predicate=self.PREDICATE,
                        object_key=contract_key,
                        source_ref=_source_ref(
                            repo=file_ref.repo_name,
                            path=file_ref.path,
                            method=method,
                            api_path=raw_path,
                        ),
                        source_system=self._source_system,
                        fact=f"{service_name} exposes {method.upper()} {raw_path}",
                        valid_at=observed_at,
                        evidence_strength="deterministic",
                        properties={
                            "method": method.upper(),
                            "path": raw_path,
                            "operation_id": op_id if isinstance(op_id, str) else None,
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
# Helpers
# ---------------------------------------------------------------------------


def _load_spec(
    file_ref: ConfigFileRef, warnings: list[str]
) -> Mapping[str, Any] | None:
    lower = file_ref.path.lower()
    try:
        if lower.endswith(".json"):
            data = json.loads(file_ref.content)
        else:
            if yaml is None:
                warnings.append("openapi-spec: pyyaml not installed — YAML disabled")
                return None
            data = yaml.safe_load(file_ref.content)
    except (json.JSONDecodeError, Exception) as exc:  # pragma: no cover - yaml.YAMLError leaks
        warnings.append(f"openapi-spec:{file_ref.path}: parse error: {exc}")
        return None

    if not isinstance(data, Mapping):
        warnings.append(
            f"openapi-spec:{file_ref.path}: root is not a mapping — skipping"
        )
        return None
    if "openapi" not in data and "swagger" not in data:
        warnings.append(
            f"openapi-spec:{file_ref.path}: missing 'openapi'/'swagger' version — skipping"
        )
        return None
    return data


def _contract_key(
    *,
    contract_spec: Any,
    service_name: str,
    path: str,
    method: str,
) -> str | None:
    """Mint a deterministic APIContract entity_key.

    Shape: ``api_contract:<service>:<method>:<path>`` with the path
    slugified so colons / slashes / templated params are safe in keys.
    """
    safe_path = _slugify_path(path)
    external_id = f"{method}:{safe_path}"
    try:
        return mint_entity_key(
            contract_spec, external_id=external_id, extra_segments=(service_name,)
        )
    except IdentityError:
        return None


def _summary_or_op_id(operation: Mapping[str, Any], *, default: str) -> str:
    summary = operation.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    op_id = operation.get("operationId")
    if isinstance(op_id, str) and op_id.strip():
        return op_id.strip()
    return default


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")


def _slugify_path(path: str) -> str:
    # /users/{id}/orders -> users-id-orders (templated braces stripped)
    stripped = re.sub(r"\{([^}]+)\}", r"\1", path)
    parts = [_slugify(seg) for seg in stripped.split("/") if seg]
    return "-".join(parts) or "root"


def _source_ref(
    *, repo: str | None, path: str, method: str, api_path: str
) -> str:
    repo_seg = repo or "unknown-repo"
    return f"openapi-spec:{repo_seg}:{path}:{method.upper()} {api_path}"


__all__ = ["OpenApiSpecScanner"]
