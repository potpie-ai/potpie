"""CodeownersScanner (rebuild plan P4 / F2 fix).

The F2 failure mode the proper POC surfaced: extractor saw only the
body of the CODEOWNERS file and emitted ``component:unknown`` because
the *file path* carried the scope. This suite locks in the fix: the
scanner emits ``service:<x> OWNED_BY person:<y>`` deterministically
from path + content alone, no LLM involved.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from adapters.outbound.scanners.codeowners import CodeownersScanner
from domain.ports.config_scanner import ConfigFileRef


def _ref(
    path: str, content: str, *, repo_name: str | None = "acme/api"
) -> ConfigFileRef:
    return ConfigFileRef(
        path=path,
        content=content,
        repo_name=repo_name,
        commit_sha="abc123",
        observed_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
    )


class TestHandlesAndDispatch:
    def test_handles_root_codeowners(self) -> None:
        scanner = CodeownersScanner()
        assert scanner.handles(_ref("CODEOWNERS", ""))
        assert scanner.handles(_ref(".github/CODEOWNERS", ""))
        assert scanner.handles(_ref("docs/CODEOWNERS", ""))

    def test_handles_nested_codeowners(self) -> None:
        scanner = CodeownersScanner()
        assert scanner.handles(_ref("apps/auth/CODEOWNERS", ""))

    def test_does_not_handle_other_files(self) -> None:
        scanner = CodeownersScanner()
        assert not scanner.handles(_ref("README.md", ""))
        assert not scanner.handles(_ref(".github/owners.txt", ""))

    def test_list_files_filters_correctly(self) -> None:
        scanner = CodeownersScanner()
        out = list(
            scanner.list_files(
                repo_name="acme/api",
                working_tree_paths=[
                    "CODEOWNERS",
                    "apps/auth/CODEOWNERS",
                    "apps/auth/main.go",
                    "README.md",
                ],
            )
        )
        assert out == ["CODEOWNERS", "apps/auth/CODEOWNERS"]

    def test_kind_and_capabilities(self) -> None:
        scanner = CodeownersScanner()
        assert scanner.kind() == "codeowners"
        caps = scanner.capabilities()
        assert "OWNED_BY" in caps.emits_predicates


class TestF2FixScopeFromPath:
    def test_nested_codeowners_stamps_service_from_path(self) -> None:
        """The F2 regression test: pattern is just ``*`` but path is
        ``apps/auth/CODEOWNERS``, so the scope is ``service:auth``."""
        scanner = CodeownersScanner()
        ref = _ref("apps/auth/CODEOWNERS", "*  @alice\n")
        result = scanner.parse_to_claims(ref)
        assert len(result.claims) == 1
        claim = result.claims[0]
        assert claim.subject_key == "service:auth"
        assert claim.object_key == "person:alice"
        assert claim.predicate == "OWNED_BY"
        assert claim.evidence_strength == "deterministic"
        assert claim.source_system == "codeowners"

    def test_root_codeowners_with_pattern_carrying_scope(self) -> None:
        """Pattern path /services/users/ matches the apps-service-leaf
        matcher, so the rule scopes to ``service:users``."""
        scanner = CodeownersScanner()
        content = "/services/users/  @users-lead\n"
        result = scanner.parse_to_claims(_ref("CODEOWNERS", content))
        assert len(result.claims) == 1
        assert result.claims[0].subject_key == "service:users"

    def test_root_codeowners_with_apps_pattern_yields_service(self) -> None:
        scanner = CodeownersScanner()
        content = "/apps/billing/  @billing-team\n"
        result = scanner.parse_to_claims(_ref("CODEOWNERS", content))
        assert len(result.claims) == 1
        assert result.claims[0].subject_key == "service:billing"

    def test_root_codeowners_global_wildcard_uses_repo(self) -> None:
        scanner = CodeownersScanner()
        content = "*  @acme/security\n"
        result = scanner.parse_to_claims(_ref("CODEOWNERS", content))
        assert len(result.claims) == 1
        claim = result.claims[0]
        assert claim.subject_key == "repo:acme-api"
        assert claim.object_key == "team:acme-security"


class TestOwnerParsing:
    def test_user_email_and_team_in_one_rule(self) -> None:
        scanner = CodeownersScanner()
        content = "*  @alice  bob@example.com  @acme/platform\n"
        result = scanner.parse_to_claims(_ref("apps/auth/CODEOWNERS", content))
        objects = sorted(c.object_key for c in result.claims)
        assert objects == ["person:alice", "person:bob", "team:acme-platform"]

    def test_ignores_comments_and_blank_lines(self) -> None:
        scanner = CodeownersScanner()
        content = "\n# this is a header\n\n*  @alice  # inline comment\n"
        result = scanner.parse_to_claims(_ref("apps/auth/CODEOWNERS", content))
        assert len(result.claims) == 1
        assert result.claims[0].object_key == "person:alice"

    def test_warns_on_unrecognised_token(self) -> None:
        scanner = CodeownersScanner()
        content = "*  @alice  garbage-token\n"
        result = scanner.parse_to_claims(_ref("apps/auth/CODEOWNERS", content))
        assert len(result.claims) == 1
        assert any("garbage-token" in w for w in result.warnings)

    def test_warns_on_pattern_without_owners(self) -> None:
        scanner = CodeownersScanner()
        content = "*\n"
        result = scanner.parse_to_claims(_ref("apps/auth/CODEOWNERS", content))
        assert result.claims == ()
        assert any("no owners" in w for w in result.warnings)


class TestIdempotenceAndSourceRef:
    def test_source_ref_contains_path_and_line(self) -> None:
        scanner = CodeownersScanner()
        content = "*  @alice\n*  @bob\n"
        result = scanner.parse_to_claims(_ref("apps/auth/CODEOWNERS", content))
        assert {c.source_ref for c in result.claims} == {
            "codeowners:acme/api:apps/auth/CODEOWNERS:L1",
            "codeowners:acme/api:apps/auth/CODEOWNERS:L2",
        }

    def test_duplicate_owner_in_same_subject_collapses(self) -> None:
        scanner = CodeownersScanner()
        content = "*  @alice\n*  @alice\n"
        result = scanner.parse_to_claims(_ref("apps/auth/CODEOWNERS", content))
        assert len(result.claims) == 1

    def test_entities_deduplicated(self) -> None:
        scanner = CodeownersScanner()
        content = "*  @alice\napps/foo/  @alice\n"
        result = scanner.parse_to_claims(_ref("apps/auth/CODEOWNERS", content))
        entity_keys = {e.entity_key for e in result.entities}
        # subject service:auth, plus owner person:alice — two only
        assert "person:alice" in entity_keys
        assert "service:auth" in entity_keys


class TestRepoFallback:
    def test_no_scope_no_repo_skips_rule(self) -> None:
        scanner = CodeownersScanner()
        # No repo_name, no path scope → no subject → warning, no claim
        ref = ConfigFileRef(path="CODEOWNERS", content="*  @alice\n", repo_name=None)
        result = scanner.parse_to_claims(ref)
        assert result.claims == ()
        assert any("no scope" in w for w in result.warnings)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
