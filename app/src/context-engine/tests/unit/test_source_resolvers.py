"""Unit tests for the built-in source resolver adapters."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from adapters.outbound.source_resolvers import (
    CompositeSourceResolver,
    DocumentationUriResolver,
    GitHubPullRequestResolver,
    NullSourceResolver,
)
from domain.source_references import SourceReferenceRecord
from domain.source_resolution import (
    BUDGET_EXCEEDED,
    RESOLVER_UNAVAILABLE,
    UNSUPPORTED_SOURCE_POLICY,
    UNSUPPORTED_SOURCE_TYPE,
    ResolverAuthContext,
    ResolverBudget,
    ResolverCapabilityEntry,
    ResolvedSnippet,
    ResolvedSummary,
    ResolvedVerification,
    SourceResolutionResult,
)

pytestmark = pytest.mark.unit


def _ref(
    ref: str = "r",
    source_type: str = "pr",
    source_system: str | None = "github",
    external_id: str | None = None,
    retrieval_uri: str | None = None,
    title: str | None = None,
    summary: str | None = None,
) -> SourceReferenceRecord:
    return SourceReferenceRecord(
        ref=ref,
        source_type=source_type,
        source_system=source_system,
        external_id=external_id,
        retrieval_uri=retrieval_uri,
        title=title,
        summary=summary,
        fetchable=bool(retrieval_uri),
    )


# -----------------------------------------------------------------------
# NullSourceResolver
# -----------------------------------------------------------------------

async def test_null_resolver_references_only_returns_empty() -> None:
    r = NullSourceResolver()
    out = await r.resolve(
        pot_id="p",
        refs=[_ref()],
        source_policy="references_only",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert out.summaries == [] and out.fallbacks == []


async def test_null_resolver_emits_unavailable_fallback_for_summary() -> None:
    r = NullSourceResolver()
    out = await r.resolve(
        pot_id="p",
        refs=[_ref()],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert len(out.fallbacks) == 1
    assert out.fallbacks[0].code == RESOLVER_UNAVAILABLE


def test_null_resolver_advertises_no_capabilities() -> None:
    assert list(NullSourceResolver().capabilities()) == []


# -----------------------------------------------------------------------
# CompositeSourceResolver — dispatch + budget
# -----------------------------------------------------------------------

class _StubResolver:
    def __init__(
        self,
        *,
        provider: str,
        source_kind: str,
        policies: frozenset[str],
        summary_char_len: int = 10,
    ) -> None:
        self._cap = ResolverCapabilityEntry(
            provider=provider, source_kind=source_kind, policies=policies
        )
        self._summary_char_len = summary_char_len
        self.calls: list[dict[str, Any]] = []

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        return (self._cap,)

    async def resolve(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        self.calls.append(
            {"refs": list(refs), "policy": source_policy, "budget": budget}
        )
        out = SourceResolutionResult()
        for ref in refs:
            out.summaries.append(
                ResolvedSummary(
                    ref=ref.ref,
                    source_type=ref.source_type,
                    summary="x" * self._summary_char_len,
                )
            )
        return out


async def test_composite_dispatches_by_provider() -> None:
    github = _StubResolver(
        provider="github", source_kind="repository", policies=frozenset({"summary"})
    )
    docs = _StubResolver(
        provider="documentation",
        source_kind="docs_space",
        policies=frozenset({"summary"}),
    )
    comp = CompositeSourceResolver([github, docs])
    refs = [
        _ref(ref="a", source_type="pr", source_system="github"),
        _ref(ref="b", source_type="documentation", source_system="documentation"),
    ]
    out = await comp.resolve(
        pot_id="p",
        refs=refs,
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert len(out.summaries) == 2
    assert len(github.calls) == 1
    assert len(docs.calls) == 1


async def test_composite_emits_unsupported_source_type_for_unmatched_ref() -> None:
    github = _StubResolver(
        provider="github", source_kind="repository", policies=frozenset({"summary"})
    )
    comp = CompositeSourceResolver([github])
    out = await comp.resolve(
        pot_id="p",
        refs=[_ref(ref="unknown", source_type="mystery", source_system="mystery")],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert any(f.code == UNSUPPORTED_SOURCE_TYPE for f in out.fallbacks)


async def test_composite_enforces_max_refs_budget() -> None:
    github = _StubResolver(
        provider="github", source_kind="repository", policies=frozenset({"summary"})
    )
    comp = CompositeSourceResolver([github])
    refs = [_ref(ref=f"r{i}", source_system="github") for i in range(5)]
    out = await comp.resolve(
        pot_id="p",
        refs=refs,
        source_policy="summary",
        budget=ResolverBudget(max_refs=2, max_total_chars=1000),
        auth=ResolverAuthContext(),
    )
    assert len(out.summaries) == 2
    assert sum(1 for f in out.fallbacks if f.code == BUDGET_EXCEEDED) == 3


async def test_composite_references_only_shortcircuits() -> None:
    github = _StubResolver(
        provider="github", source_kind="repository", policies=frozenset({"summary"})
    )
    comp = CompositeSourceResolver([github])
    out = await comp.resolve(
        pot_id="p",
        refs=[_ref(source_system="github")],
        source_policy="references_only",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert out.summaries == [] and out.fallbacks == []
    assert github.calls == []


def test_composite_merges_capabilities() -> None:
    a = _StubResolver(
        provider="github",
        source_kind="repository",
        policies=frozenset({"summary", "verify"}),
    )
    b = _StubResolver(
        provider="documentation",
        source_kind="docs_space",
        policies=frozenset({"summary"}),
    )
    comp = CompositeSourceResolver([a, b])
    caps = {(c.provider, c.source_kind): c.policies for c in comp.capabilities()}
    assert caps[("github", "repository")] == frozenset({"summary", "verify"})
    assert caps[("documentation", "docs_space")] == frozenset({"summary"})


# -----------------------------------------------------------------------
# GitHubPullRequestResolver
# -----------------------------------------------------------------------

class _FakeSourceControl:
    def __init__(self, pr: dict[str, Any]) -> None:
        self._pr = pr
        self.calls: list[tuple[str, int, bool]] = []

    def get_pull_request(
        self, repo_name: str, pr_number: int, include_diff: bool = False
    ) -> dict[str, Any]:
        self.calls.append((repo_name, pr_number, include_diff))
        return dict(self._pr)

    def get_pull_request_commits(self, repo_name: str, pr_number: int):  # pragma: no cover
        return []

    def get_pull_request_review_comments(self, repo_name, pr_number, limit=100):  # pragma: no cover
        return []

    def get_pull_request_issue_comments(self, repo_name, pr_number, limit=50):  # pragma: no cover
        return []

    def get_issue(self, repo_name, issue_number):  # pragma: no cover
        return {}

    def iter_closed_pulls(self, repo_name):  # pragma: no cover
        return iter(())


async def test_github_resolver_summary_returns_title_and_body() -> None:
    fake = _FakeSourceControl(
        {
            "title": "Fix flaky test",
            "body": "Fixes timing issue in resolver tests.",
            "state": "closed",
            "merged": True,
            "html_url": "https://github.com/acme/widget/pull/42",
        }
    )
    resolver = GitHubPullRequestResolver(
        source_for_repo=lambda _: fake,
        repo_resolver=lambda _p, _r: "acme/widget",
    )
    out = await resolver.resolve(
        pot_id="p",
        refs=[_ref(ref="github:pr:42", external_id="42", source_system="github")],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert len(out.summaries) == 1
    s = out.summaries[0]
    assert "Fix flaky test" in s.summary
    assert s.retrieval_uri == "https://github.com/acme/widget/pull/42"
    assert fake.calls == [("acme/widget", 42, False)]


async def test_github_resolver_verify_marks_real_pr_verified() -> None:
    fake = _FakeSourceControl({"state": "closed", "merged": True, "title": "x"})
    resolver = GitHubPullRequestResolver(
        source_for_repo=lambda _: fake,
        repo_resolver=lambda _p, _r: "acme/widget",
    )
    out = await resolver.resolve(
        pot_id="p",
        refs=[_ref(ref="github:pr:1", external_id="1", source_system="github")],
        source_policy="verify",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert out.verifications[0].verified is True
    assert out.verifications[0].verification_state == "verified"


async def test_github_resolver_snippets_includes_include_diff_flag() -> None:
    fake = _FakeSourceControl(
        {
            "title": "t",
            "diff": (
                "diff --git a/x.py b/x.py\n"
                "--- a/x.py\n"
                "+++ b/x.py\n"
                "@@ -1 +1 @@\n"
                "-old line\n"
                "+new line\n"
            ),
        }
    )
    resolver = GitHubPullRequestResolver(
        source_for_repo=lambda _: fake,
        repo_resolver=lambda _p, _r: "acme/widget",
    )
    out = await resolver.resolve(
        pot_id="p",
        refs=[_ref(ref="github:pr:7", external_id="7", source_system="github")],
        source_policy="snippets",
        budget=ResolverBudget(max_chars_per_item=500, max_total_chars=500),
        auth=ResolverAuthContext(),
    )
    assert fake.calls[0][2] is True
    assert len(out.snippets) == 1
    assert out.snippets[0].location == "b/x.py"


async def test_github_resolver_unsupported_policy_emits_fallback() -> None:
    fake = _FakeSourceControl({})
    resolver = GitHubPullRequestResolver(
        source_for_repo=lambda _: fake,
        repo_resolver=lambda _p, _r: "acme/widget",
    )
    out = await resolver.resolve(
        pot_id="p",
        refs=[_ref(ref="github:pr:1", external_id="1")],
        source_policy="full_if_needed",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert any(f.code == UNSUPPORTED_SOURCE_POLICY for f in out.fallbacks)


async def test_github_resolver_missing_repo_emits_unsupported() -> None:
    fake = _FakeSourceControl({})
    resolver = GitHubPullRequestResolver(
        source_for_repo=lambda _: fake,
        repo_resolver=lambda _p, _r: None,
    )
    out = await resolver.resolve(
        pot_id="p",
        refs=[_ref(ref="github:pr:1", external_id="1")],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert any(f.code == UNSUPPORTED_SOURCE_TYPE for f in out.fallbacks)


# -----------------------------------------------------------------------
# DocumentationUriResolver
# -----------------------------------------------------------------------

async def test_doc_resolver_summary_reuses_stored_fields_without_fetcher() -> None:
    resolver = DocumentationUriResolver()
    ref = _ref(
        ref="docs:onboarding",
        source_type="documentation",
        source_system="documentation",
        retrieval_uri="https://docs.example.com/onboarding",
        title="Onboarding",
        summary="How new engineers get started.",
    )
    out = await resolver.resolve(
        pot_id="p",
        refs=[ref],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert len(out.summaries) == 1
    assert "Onboarding" in out.summaries[0].summary


async def test_doc_resolver_verify_marks_http_refs_verified() -> None:
    resolver = DocumentationUriResolver()
    good = _ref(
        ref="docs:a",
        source_type="documentation",
        retrieval_uri="https://example.com/x",
    )
    bad = _ref(ref="docs:b", source_type="documentation", retrieval_uri=None)
    out = await resolver.resolve(
        pot_id="p",
        refs=[good, bad],
        source_policy="verify",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    by_ref = {v.ref: v for v in out.verifications}
    assert by_ref["docs:a"].verification_state == "verified"
    assert by_ref["docs:b"].verification_state == "verification_failed"


async def test_doc_resolver_snippets_without_fetcher_emits_fallback() -> None:
    resolver = DocumentationUriResolver()
    out = await resolver.resolve(
        pot_id="p",
        refs=[
            _ref(
                ref="docs:a",
                source_type="documentation",
                retrieval_uri="https://example.com/x",
            )
        ],
        source_policy="snippets",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert any(f.code == UNSUPPORTED_SOURCE_POLICY for f in out.fallbacks)


async def test_doc_resolver_capabilities_include_snippets_when_fetcher_wired() -> None:
    async def fetcher(uri: str, auth: ResolverAuthContext) -> str | None:
        return "body"

    resolver = DocumentationUriResolver(content_fetcher=fetcher)
    caps = list(resolver.capabilities())
    assert any("snippets" in c.policies for c in caps)
