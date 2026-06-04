"""OpenApiSpecScanner unit tests (rebuild plan P4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

pytest.importorskip("yaml")

from adapters.outbound.scanners.openapi_spec import OpenApiSpecScanner
from domain.ports.config_scanner import ConfigFileRef


def _ref(
    path: str, content: str, *, repo_name: str | None = "acme/api"
) -> ConfigFileRef:
    return ConfigFileRef(
        path=path,
        content=content,
        repo_name=repo_name,
        commit_sha="abc",
        observed_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
    )


SPEC_AUTH = {
    "openapi": "3.0.3",
    "info": {"title": "Auth API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "get": {"summary": "List users", "operationId": "listUsers"},
            "post": {"summary": "Create user", "operationId": "createUser"},
        },
        "/users/{id}": {
            "get": {"summary": "Get user", "operationId": "getUser"},
        },
    },
}


class TestHandles:
    def test_handles_yaml_and_json(self) -> None:
        scanner = OpenApiSpecScanner()
        assert scanner.handles(_ref("services/auth/openapi.yaml", ""))
        assert scanner.handles(_ref("openapi.json", ""))
        assert scanner.handles(_ref("swagger.yml", ""))

    def test_skips_other_files(self) -> None:
        scanner = OpenApiSpecScanner()
        assert not scanner.handles(_ref("docs/api.md", ""))


class TestEmitsExposesEdges:
    def test_one_edge_per_operation_yaml(self) -> None:
        import yaml as yaml_lib

        content = yaml_lib.safe_dump(SPEC_AUTH)
        scanner = OpenApiSpecScanner()
        result = scanner.parse_to_claims(
            _ref("services/auth/openapi.yaml", content)
        )
        # 3 operations defined
        assert len(result.claims) == 3
        for c in result.claims:
            assert c.predicate == "EXPOSES"
            assert c.subject_key == "service:auth"
            assert c.evidence_strength == "deterministic"
            assert c.object_key.startswith("api_contract:")

    def test_json_spec(self) -> None:
        scanner = OpenApiSpecScanner()
        result = scanner.parse_to_claims(
            _ref("services/auth/openapi.json", json.dumps(SPEC_AUTH))
        )
        assert len(result.claims) == 3

    def test_service_from_info_title_when_no_path_scope(self) -> None:
        import yaml as yaml_lib

        scanner = OpenApiSpecScanner()
        # Root spec — no path-scope service. Falls back to info.title slugified.
        result = scanner.parse_to_claims(
            _ref("openapi.yaml", yaml_lib.safe_dump(SPEC_AUTH))
        )
        assert all(c.subject_key == "service:auth-api" for c in result.claims)

    def test_apicontract_keys_include_method_and_path(self) -> None:
        import yaml as yaml_lib

        scanner = OpenApiSpecScanner()
        result = scanner.parse_to_claims(
            _ref("services/auth/openapi.yaml", yaml_lib.safe_dump(SPEC_AUTH))
        )
        keys = {c.object_key for c in result.claims}
        # api_contract:<service>:<method>:<slugified-path>
        assert any("get" in k and "users" in k for k in keys)
        assert any("post" in k and "users" in k for k in keys)

    def test_non_method_keys_ignored(self) -> None:
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Auth API"},
            "paths": {
                "/x": {
                    "parameters": [],  # non-method
                    "get": {"summary": "ok"},
                }
            },
        }
        import yaml as yaml_lib

        scanner = OpenApiSpecScanner()
        result = scanner.parse_to_claims(
            _ref("services/auth/openapi.yaml", yaml_lib.safe_dump(spec))
        )
        assert len(result.claims) == 1


class TestRobustness:
    def test_malformed_json_warns(self) -> None:
        scanner = OpenApiSpecScanner()
        result = scanner.parse_to_claims(_ref("services/auth/openapi.json", "{not json"))
        assert result.claims == ()
        assert any("parse error" in w for w in result.warnings)

    def test_missing_openapi_version_warns(self) -> None:
        import yaml as yaml_lib

        spec_no_version = {"info": {"title": "x"}, "paths": {}}
        scanner = OpenApiSpecScanner()
        result = scanner.parse_to_claims(
            _ref("services/auth/openapi.yaml", yaml_lib.safe_dump(spec_no_version))
        )
        assert result.claims == ()
        assert any("openapi" in w.lower() for w in result.warnings)

    def test_missing_paths_warns(self) -> None:
        import yaml as yaml_lib

        spec = {"openapi": "3.0.0", "info": {"title": "x"}}
        scanner = OpenApiSpecScanner()
        result = scanner.parse_to_claims(
            _ref("services/auth/openapi.yaml", yaml_lib.safe_dump(spec))
        )
        assert result.claims == ()
        assert any("paths" in w for w in result.warnings)
