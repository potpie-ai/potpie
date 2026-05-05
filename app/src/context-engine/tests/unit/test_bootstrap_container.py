"""Tests for bootstrap/container.py — dependency injection wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adapters.outbound.source_resolvers.composite import CompositeSourceResolver
from adapters.outbound.source_resolvers.documentation import DocumentationUriResolver
from adapters.outbound.source_resolvers.github_pull_request import GitHubPullRequestResolver
from adapters.outbound.source_resolvers.null import NullSourceResolver
from bootstrap.container import build_container, build_container_with_github_token
from domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo

pytestmark = pytest.mark.unit


class _FakePots:
    """Minimal PotResolutionPort stand-in."""

    def __init__(self, repo_name: str | None = "acme/app") -> None:
        self._repo_name = repo_name

    def resolve_pot(self, pot_id: str) -> ResolvedPot | None:
        if self._repo_name is None:
            return ResolvedPot(pot_id=pot_id, name="empty", repos=[])
        return ResolvedPot(
            pot_id=pot_id,
            name=self._repo_name,
            repos=[
                ResolvedPotRepo(
                    pot_id=pot_id,
                    repo_id=pot_id,
                    provider="github",
                    provider_host="github.com",
                    repo_name=self._repo_name,
                )
            ],
        )

    def known_pot_ids(self) -> list[str]:
        return []

    def find_pots_for_repo(self, ref):
        return []

    def list_pot_repos(self, pot_id: str) -> list[ResolvedPotRepo]:
        resolved = self.resolve_pot(pot_id)
        return list(resolved.repos) if resolved else []

    def get_repo_in_pot(self, pot_id, ref):
        return None


def test_build_container_defaults_to_null_source_resolver() -> None:
    container = build_container(
        pots=_FakePots(),
        source_for_repo=lambda _repo: MagicMock(),
    )
    assert isinstance(container.source_resolver, NullSourceResolver)


def test_build_container_with_github_token_uses_composite_resolver() -> None:
    container = build_container_with_github_token(
        token="ghp_fake",
        pots=_FakePots(),
    )
    assert isinstance(container.source_resolver, CompositeSourceResolver)


def test_build_container_with_github_token_wires_resolution_service() -> None:
    container = build_container_with_github_token(
        token="ghp_fake",
        pots=_FakePots(),
    )
    assert container.resolution_service is not None
    # The resolution_service should now hold the composite, not NullSourceResolver.
    rs_resolver = container.resolution_service._source_resolver
    assert isinstance(rs_resolver, CompositeSourceResolver)


def test_composite_contains_github_and_doc_resolvers() -> None:
    container = build_container_with_github_token(
        token="ghp_fake",
        pots=_FakePots(),
    )
    composite = container.source_resolver
    assert isinstance(composite, CompositeSourceResolver)
    children = composite._children
    assert any(isinstance(c, GitHubPullRequestResolver) for c in children)
    assert any(isinstance(c, DocumentationUriResolver) for c in children)


def test_github_resolver_advertises_summary_verify_snippets() -> None:
    container = build_container_with_github_token(
        token="ghp_fake",
        pots=_FakePots(),
    )
    composite = container.source_resolver
    caps = composite.capabilities()
    policies = set()
    for cap in caps:
        policies.update(cap.policies)
    assert {"summary", "verify", "snippets"}.issubset(policies)
