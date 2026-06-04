"""Path-aware scope stamping + MENTIONS provenance (rebuild plan P5 / F2 + F4)."""

from __future__ import annotations

from domain.episode_mentions import build_mentions_edges
from domain.path_scope import (
    PathScope,
    annotate_entity_properties,
    derive_scope,
    stamp_extra_segments,
)


class TestPathScopeMatchers:
    def test_k8s_manifest_stamps_env_and_service(self) -> None:
        scope = derive_scope("clusters/prod/auth-svc.yaml")
        assert scope.environment == "prod"
        assert scope.service == "auth-svc"

    def test_apps_layout_stamps_service_only(self) -> None:
        scope = derive_scope("apps/auth/server/main.go")
        assert scope.service == "auth"
        assert scope.environment is None

    def test_codeowners_tagged(self) -> None:
        scope = derive_scope("apps/auth/CODEOWNERS")
        # The CODEOWNERS matcher tags + the apps-service matcher stamps service.
        assert "codeowners" in scope.tags
        assert scope.service == "auth"

    def test_adr_doc_tagged(self) -> None:
        scope = derive_scope("docs/adr/007-position-b.md")
        assert "adr" in scope.tags

    def test_terraform_envs_stamps_env(self) -> None:
        scope = derive_scope("terraform/envs/staging/main.tf")
        assert scope.environment == "staging"

    def test_unrecognised_path_returns_empty(self) -> None:
        scope = derive_scope("README.md")
        assert scope.is_empty()

    def test_windows_path_normalized(self) -> None:
        scope = derive_scope("clusters\\prod\\auth-svc.yaml")
        assert scope.service == "auth-svc"
        assert scope.environment == "prod"

    def test_case_normalised(self) -> None:
        scope = derive_scope("Clusters/PROD/Auth-Svc.YAML")
        assert scope.service == "auth-svc"
        assert scope.environment == "prod"


class TestPathScopeMerge:
    def test_merged_with_other_wins_on_conflict(self) -> None:
        base = PathScope(service="auth-svc")
        update = PathScope(service="users-svc", environment="prod")
        merged = base.merged_with(update)
        assert merged.service == "users-svc"
        assert merged.environment == "prod"

    def test_merged_with_preserves_missing_fields(self) -> None:
        base = PathScope(service="auth-svc", environment="prod")
        update = PathScope(repo="github:acme/api")
        merged = base.merged_with(update)
        assert merged.service == "auth-svc"
        assert merged.environment == "prod"
        assert merged.repo == "github:acme/api"


class TestAnnotateAndExtraSegments:
    def test_extra_segments_lead_with_service(self) -> None:
        scope = PathScope(service="auth-svc")
        extras = stamp_extra_segments(scope=scope)
        assert extras == ("auth-svc",)

    def test_annotate_does_not_overwrite_existing(self) -> None:
        scope = PathScope(service="auth-svc", environment="prod")
        props = {"service": "users-svc"}  # caller already set service
        out = annotate_entity_properties(scope=scope, properties=props)
        assert out["service"] == "users-svc"  # respected
        assert out["environment"] == "prod"   # filled in


class TestMentionsEdges:
    def test_emits_one_edge_per_unique_target(self) -> None:
        edges = build_mentions_edges(
            activity_entity_key="activity:github:pr:1042",
            mentioned_entity_keys=[
                "service:auth-svc",
                "service:users-svc",
                "service:auth-svc",  # dup
                "service:auth-svc",  # dup
            ],
            source_ref="github:pr:acme/api:1042:body",
            source_system="github",
        )
        targets = {e.to_entity_key for e in edges}
        assert targets == {"service:auth-svc", "service:users-svc"}
        for e in edges:
            assert e.edge_type == "MENTIONS"
            assert e.from_entity_key == "activity:github:pr:1042"
            assert e.properties["source_system"] == "github"
            assert e.properties["evidence_strength"] == "attested"

    def test_filters_self_reference(self) -> None:
        edges = build_mentions_edges(
            activity_entity_key="activity:github:pr:1042",
            mentioned_entity_keys=[
                "activity:github:pr:1042",
                "service:auth-svc",
            ],
            source_ref="github:pr:body",
            source_system="github",
        )
        assert len(edges) == 1
        assert edges[0].to_entity_key == "service:auth-svc"

    def test_empty_mention_list_returns_empty(self) -> None:
        edges = build_mentions_edges(
            activity_entity_key="activity:github:pr:1042",
            mentioned_entity_keys=[],
            source_ref="github:pr:body",
            source_system="github",
        )
        assert edges == []

    def test_skips_invalid_keys(self) -> None:
        edges = build_mentions_edges(
            activity_entity_key="activity:github:pr:1042",
            mentioned_entity_keys=["", None, 42, "service:auth-svc"],  # type: ignore[list-item]
            source_ref="github:pr:body",
            source_system="github",
        )
        assert len(edges) == 1
        assert edges[0].to_entity_key == "service:auth-svc"
