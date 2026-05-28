"""KubernetesManifestScanner (topology core).

Confirms the scanner emits ``Service DEPLOYED_TO Environment``
deterministically from a manifest's labels + file path, with the
environment stamped on the edge. The deployment object is not a node
(the fact that a service runs in an env is the edge).
"""

from __future__ import annotations

from datetime import datetime, timezone
from textwrap import dedent

import pytest

pytest.importorskip("yaml")

from adapters.outbound.scanners.kubernetes_manifest import KubernetesManifestScanner
from domain.ports.config_scanner import ConfigFileRef


def _ref(path: str, content: str) -> ConfigFileRef:
    return ConfigFileRef(
        path=path,
        content=content,
        repo_name="acme/platform",
        commit_sha="deadbeef",
        observed_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
    )


MANIFEST_AUTH_PROD = dedent(
    """
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: auth-svc
      namespace: auth
      labels:
        app.kubernetes.io/name: auth-svc
    spec:
      replicas: 3
      selector:
        matchLabels:
          app.kubernetes.io/name: auth-svc
    """
).strip()


class TestHandlesAndDispatch:
    def test_handles_clusters_path(self) -> None:
        scanner = KubernetesManifestScanner()
        assert scanner.handles(_ref("clusters/prod/auth-svc.yaml", MANIFEST_AUTH_PROD))

    def test_handles_k8s_path(self) -> None:
        scanner = KubernetesManifestScanner()
        assert scanner.handles(_ref("k8s/staging/users.yaml", MANIFEST_AUTH_PROD))

    def test_handles_manifests_path(self) -> None:
        scanner = KubernetesManifestScanner()
        assert scanner.handles(_ref("manifests/foo.yml", MANIFEST_AUTH_PROD))

    def test_skips_arbitrary_yaml(self) -> None:
        scanner = KubernetesManifestScanner()
        # Random YAML outside the well-known dirs should not dispatch.
        assert not scanner.handles(_ref("docs/notes.yaml", "foo: bar"))

    def test_list_files_filter(self) -> None:
        scanner = KubernetesManifestScanner()
        out = list(
            scanner.list_files(
                repo_name="acme/platform",
                working_tree_paths=[
                    "clusters/prod/auth.yaml",
                    "docs/notes.yaml",
                    "k8s/staging/users.yaml",
                ],
            )
        )
        assert out == ["clusters/prod/auth.yaml", "k8s/staging/users.yaml"]


class TestF1FixServiceJoin:
    def test_emits_deployed_to_from_path_and_labels(self) -> None:
        scanner = KubernetesManifestScanner()
        result = scanner.parse_to_claims(
            _ref("clusters/prod/auth-svc.yaml", MANIFEST_AUTH_PROD)
        )
        preds = sorted(c.predicate for c in result.claims)
        assert preds == ["DEPLOYED_TO"]

        deployed_to = next(c for c in result.claims if c.predicate == "DEPLOYED_TO")
        assert deployed_to.subject_key == "service:auth-svc"
        assert deployed_to.object_key == "environment:prod"
        assert deployed_to.properties.get("environment") == "prod"
        assert deployed_to.evidence_strength == "deterministic"
        assert deployed_to.source_system == "kubernetes"

    def test_service_inferred_from_path_when_labels_missing(self) -> None:
        manifest = dedent(
            """
            apiVersion: apps/v1
            kind: Deployment
            metadata:
              name: legacy-pod
            """
        ).strip()
        scanner = KubernetesManifestScanner()
        result = scanner.parse_to_claims(_ref("clusters/staging/users.yaml", manifest))
        deployed_to = next(c for c in result.claims if c.predicate == "DEPLOYED_TO")
        # No labels → service falls back to path scope: clusters/<env>/<service>.yaml
        assert deployed_to.subject_key == "service:users"
        assert deployed_to.object_key == "environment:staging"

    def test_multi_doc_manifest_emits_per_workload(self) -> None:
        manifest = MANIFEST_AUTH_PROD + "\n---\n" + dedent(
            """
            apiVersion: apps/v1
            kind: StatefulSet
            metadata:
              name: auth-cache
              namespace: auth
              labels:
                app.kubernetes.io/name: auth-cache
            """
        ).strip()
        scanner = KubernetesManifestScanner()
        result = scanner.parse_to_claims(
            _ref("clusters/prod/auth-svc.yaml", manifest)
        )
        deployed_to_claims = [c for c in result.claims if c.predicate == "DEPLOYED_TO"]
        assert len(deployed_to_claims) == 2  # one per workload
        service_keys = {c.subject_key for c in deployed_to_claims}
        assert service_keys == {"service:auth-svc", "service:auth-cache"}

    def test_non_workload_kind_skipped(self) -> None:
        manifest = dedent(
            """
            apiVersion: v1
            kind: ConfigMap
            metadata:
              name: app-config
              namespace: auth
            data:
              key: value
            """
        ).strip()
        scanner = KubernetesManifestScanner()
        result = scanner.parse_to_claims(
            _ref("clusters/prod/configmap.yaml", manifest)
        )
        assert result.claims == ()


class TestParseRobustness:
    def test_malformed_yaml_warns_and_returns_empty(self) -> None:
        scanner = KubernetesManifestScanner()
        result = scanner.parse_to_claims(
            _ref("clusters/prod/broken.yaml", "::: not yaml :::")
        )
        assert result.claims == ()
        assert any("YAML parse error" in w for w in result.warnings)

    def test_missing_metadata_name_warns(self) -> None:
        manifest = dedent(
            """
            apiVersion: apps/v1
            kind: Deployment
            metadata:
              namespace: auth
            """
        ).strip()
        scanner = KubernetesManifestScanner()
        result = scanner.parse_to_claims(
            _ref("clusters/prod/noname.yaml", manifest)
        )
        assert result.claims == ()
        assert any("metadata.name" in w for w in result.warnings)

    def test_no_path_scope_environment_no_deployed_to_edge(self) -> None:
        # k8s/<env>/... convention; if env is missing we don't guess it, so
        # no DEPLOYED_TO edge is emitted at all (the env is the object).
        scanner = KubernetesManifestScanner()
        result = scanner.parse_to_claims(
            _ref("k8s/foo-svc.yaml", MANIFEST_AUTH_PROD)
        )
        preds = [c.predicate for c in result.claims]
        assert "DEPLOYED_TO" not in preds
        assert any("environment" in w for w in result.warnings)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
