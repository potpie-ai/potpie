"""DependencyManifestScanner unit tests (rebuild plan P4)."""

from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent

from adapters.outbound.scanners.dependency_manifest import DependencyManifestScanner
from domain.ports.config_scanner import ConfigFileRef


def _ref(path: str, content: str, *, repo_name: str | None = "acme/api") -> ConfigFileRef:
    return ConfigFileRef(
        path=path,
        content=content,
        repo_name=repo_name,
        commit_sha="abc123",
        observed_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
    )


class TestHandles:
    def test_handles_pyproject(self) -> None:
        scanner = DependencyManifestScanner()
        assert scanner.handles(_ref("apps/auth/pyproject.toml", ""))

    def test_handles_requirements_variants(self) -> None:
        scanner = DependencyManifestScanner()
        assert scanner.handles(_ref("apps/auth/requirements.txt", ""))
        assert scanner.handles(_ref("apps/auth/requirements-dev.txt", ""))

    def test_handles_package_json(self) -> None:
        scanner = DependencyManifestScanner()
        assert scanner.handles(_ref("apps/web/package.json", ""))

    def test_skips_arbitrary_files(self) -> None:
        scanner = DependencyManifestScanner()
        assert not scanner.handles(_ref("apps/auth/main.py", ""))
        assert not scanner.handles(_ref("README.md", ""))


class TestPyproject:
    def test_pep621_dependencies(self) -> None:
        content = dedent(
            """
            [project]
            name = "auth-svc"
            dependencies = [
              "fastapi>=0.100",
              "pydantic[email]~=2.0",
            ]

            [project.optional-dependencies]
            dev = ["pytest>=8.0"]
            """
        ).strip()
        scanner = DependencyManifestScanner()
        result = scanner.parse_to_claims(_ref("apps/auth/pyproject.toml", content))
        dep_objects = {c.object_key for c in result.claims}
        assert "dependency:pypi:fastapi" in dep_objects
        assert "dependency:pypi:pydantic" in dep_objects
        assert "dependency:pypi:pytest" in dep_objects
        # Subject is service:auth from path
        assert all(c.subject_key == "service:auth" for c in result.claims)
        # Dev dep is tagged
        pytest_claim = next(c for c in result.claims if c.object_key.endswith(":pytest"))
        assert pytest_claim.properties["dependency_kind"] == "optional:dev"

    def test_poetry_format(self) -> None:
        content = dedent(
            """
            [tool.poetry]
            name = "auth-svc"

            [tool.poetry.dependencies]
            python = "^3.11"
            fastapi = ">=0.100"

            [tool.poetry.dev-dependencies]
            pytest = "^8.0"
            """
        ).strip()
        scanner = DependencyManifestScanner()
        result = scanner.parse_to_claims(_ref("apps/auth/pyproject.toml", content))
        names = {c.object_key for c in result.claims}
        # python is filtered out
        assert "dependency:pypi:fastapi" in names
        assert "dependency:pypi:pytest" in names
        assert "dependency:pypi:python" not in names

    def test_malformed_toml_warns(self) -> None:
        scanner = DependencyManifestScanner()
        result = scanner.parse_to_claims(
            _ref("apps/auth/pyproject.toml", "this is :: not :: toml")
        )
        assert result.claims == ()
        assert any("pyproject parse error" in w for w in result.warnings)


class TestRequirementsTxt:
    def test_basic_requirements(self) -> None:
        content = dedent(
            """
            # primary deps
            fastapi>=0.100
            pydantic[email]~=2.0

            -r requirements-shared.txt
            # next line: weird but legal
            httpx==0.28.0
            """
        ).strip()
        scanner = DependencyManifestScanner()
        result = scanner.parse_to_claims(
            _ref("apps/auth/requirements.txt", content)
        )
        names = {c.object_key for c in result.claims}
        assert "dependency:pypi:fastapi" in names
        assert "dependency:pypi:pydantic" in names
        assert "dependency:pypi:httpx" in names


class TestPackageJson:
    def test_package_json_with_dev_deps(self) -> None:
        content = dedent(
            """
            {
              "name": "web",
              "dependencies": {
                "react": "^18.0.0",
                "next": "^14.0.0"
              },
              "devDependencies": {
                "typescript": "^5.0.0"
              }
            }
            """
        ).strip()
        scanner = DependencyManifestScanner()
        result = scanner.parse_to_claims(_ref("apps/web/package.json", content))
        names = {c.object_key for c in result.claims}
        assert "dependency:npm:react" in names
        assert "dependency:npm:next" in names
        assert "dependency:npm:typescript" in names
        ts = next(c for c in result.claims if c.object_key.endswith(":typescript"))
        assert ts.properties["dependency_kind"] == "dev"

    def test_package_json_malformed(self) -> None:
        scanner = DependencyManifestScanner()
        result = scanner.parse_to_claims(_ref("apps/web/package.json", "{not json"))
        assert result.claims == ()
        assert any("package.json parse error" in w for w in result.warnings)


class TestSubjectResolution:
    def test_no_service_scope_falls_back_to_repo(self) -> None:
        content = "[project]\ndependencies = ['x']\n"
        scanner = DependencyManifestScanner()
        # Root pyproject.toml — no service scope
        result = scanner.parse_to_claims(_ref("pyproject.toml", content))
        assert all(c.subject_key == "repo:acme-api" for c in result.claims)
        assert any("repo-scoped" in w for w in result.warnings)

    def test_no_service_and_no_repo_skips(self) -> None:
        content = "[project]\ndependencies = ['x']\n"
        scanner = DependencyManifestScanner()
        result = scanner.parse_to_claims(
            ConfigFileRef(path="pyproject.toml", content=content, repo_name=None)
        )
        assert result.claims == ()
