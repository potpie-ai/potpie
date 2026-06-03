"""LinearIssueResolver: parsing, policy handling, budget clamping, fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adapters.outbound.source_resolvers.linear_issue import LinearIssueResolver
from domain.source_references import SourceReferenceRecord
from domain.source_resolution import (
    PERMISSION_DENIED,
    RESOLVER_ERROR,
    SOURCE_UNREACHABLE,
    UNSUPPORTED_SOURCE_POLICY,
    UNSUPPORTED_SOURCE_TYPE,
    ResolverAuthContext,
    ResolverBudget,
)

pytestmark = pytest.mark.unit

DATA = Path(__file__).resolve().parent.parent / "data" / "linear"


def _detail() -> dict[str, Any]:
    return json.loads((DATA / "issue_detail.json").read_text(encoding="utf-8"))


class _FakeFetcher:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self.payload = payload
        self.calls: list[str] = []
        self.raise_exc: Exception | None = None

    def get_issue(self, issue_id: str) -> dict[str, Any] | None:
        self.calls.append(issue_id)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.payload


def _ref(ref: str, **overrides: Any) -> SourceReferenceRecord:
    base = {
        "ref": ref,
        "source_type": "linear_issue",
        "source_system": "linear",
    }
    base.update(overrides)
    return SourceReferenceRecord(**base)


@pytest.mark.asyncio
async def test_resolver_capability_entry_advertises_three_policies() -> None:
    r = LinearIssueResolver(fetcher=_FakeFetcher(_detail()))
    [entry] = list(r.capabilities())
    assert entry.provider == "linear"
    assert entry.source_kind == "linear_issue"
    assert entry.policies == frozenset({"summary", "verify", "snippets"})


@pytest.mark.asyncio
async def test_summary_policy_includes_title_and_metadata() -> None:
    fetcher = _FakeFetcher(_detail())
    r = LinearIssueResolver(fetcher=fetcher)
    out = await r.resolve(
        pot_id="pot-1",
        refs=[_ref("linear:issue:ENG-42")],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert len(out.summaries) == 1
    s = out.summaries[0]
    assert s.source_system == "linear"
    assert "ENG-42" in s.summary
    assert "In Progress" in s.summary
    assert s.retrieval_uri == _detail()["url"]
    assert fetcher.calls == ["ENG-42"]


@pytest.mark.asyncio
async def test_verify_policy_marks_issue_verified_with_state_reason() -> None:
    r = LinearIssueResolver(fetcher=_FakeFetcher(_detail()))
    out = await r.resolve(
        pot_id="pot-1",
        refs=[_ref("linear:issue:ENG-42")],
        source_policy="verify",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    [v] = out.verifications
    assert v.verified is True
    assert v.verification_state == "verified"
    assert "In Progress" in (v.reason or "")


@pytest.mark.asyncio
async def test_snippets_policy_returns_description_and_comments() -> None:
    r = LinearIssueResolver(fetcher=_FakeFetcher(_detail()))
    out = await r.resolve(
        pot_id="pot-1",
        refs=[_ref("linear:issue:ENG-42")],
        source_policy="snippets",
        budget=ResolverBudget(max_snippets_per_ref=3),
        auth=ResolverAuthContext(),
    )
    locations = {s.location for s in out.snippets}
    assert "description" in locations
    assert any(loc and loc.startswith("comment:") for loc in locations)


@pytest.mark.asyncio
async def test_unsupported_policy_emits_fallback() -> None:
    r = LinearIssueResolver(fetcher=_FakeFetcher(_detail()))
    out = await r.resolve(
        pot_id="pot-1",
        refs=[_ref("linear:issue:ENG-42")],
        source_policy="full_if_needed",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert out.fallbacks and out.fallbacks[0].code == UNSUPPORTED_SOURCE_POLICY
    assert out.summaries == out.snippets == out.verifications == []


@pytest.mark.asyncio
async def test_non_linear_ref_emits_unsupported_source_type_fallback() -> None:
    r = LinearIssueResolver(fetcher=_FakeFetcher(_detail()))
    out = await r.resolve(
        pot_id="pot-1",
        refs=[
            SourceReferenceRecord(
                ref="github:pr:42",
                source_type="pull_request",
                source_system="github",
                external_id="42",
            )
        ],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert out.fallbacks and out.fallbacks[0].code == UNSUPPORTED_SOURCE_TYPE


@pytest.mark.asyncio
async def test_missing_issue_emits_unsupported_source_type_fallback() -> None:
    r = LinearIssueResolver(fetcher=_FakeFetcher(None))
    out = await r.resolve(
        pot_id="pot-1",
        refs=[_ref("linear:issue:ENG-42")],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert out.fallbacks and out.fallbacks[0].code == UNSUPPORTED_SOURCE_TYPE


@pytest.mark.asyncio
async def test_fetcher_permission_error_becomes_fallback() -> None:
    fetcher = _FakeFetcher(None)
    fetcher.raise_exc = PermissionError("token lacks scope")
    r = LinearIssueResolver(fetcher=fetcher)
    out = await r.resolve(
        pot_id="pot-1",
        refs=[_ref("linear:issue:ENG-42")],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert out.fallbacks and out.fallbacks[0].code == PERMISSION_DENIED


@pytest.mark.asyncio
async def test_fetcher_network_error_becomes_fallback() -> None:
    fetcher = _FakeFetcher(None)
    fetcher.raise_exc = ConnectionError("dns fail")
    r = LinearIssueResolver(fetcher=fetcher)
    out = await r.resolve(
        pot_id="pot-1",
        refs=[_ref("linear:issue:ENG-42")],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert out.fallbacks and out.fallbacks[0].code == SOURCE_UNREACHABLE


@pytest.mark.asyncio
async def test_fetcher_generic_exception_becomes_resolver_error() -> None:
    fetcher = _FakeFetcher(None)
    fetcher.raise_exc = RuntimeError("graphql 500")
    r = LinearIssueResolver(fetcher=fetcher)
    out = await r.resolve(
        pot_id="pot-1",
        refs=[_ref("linear:issue:ENG-42")],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert out.fallbacks and out.fallbacks[0].code == RESOLVER_ERROR


@pytest.mark.asyncio
async def test_summary_clamps_to_budget_max_total_chars() -> None:
    r = LinearIssueResolver(fetcher=_FakeFetcher(_detail()))
    out = await r.resolve(
        pot_id="pot-1",
        refs=[_ref("linear:issue:ENG-42")],
        source_policy="summary",
        budget=ResolverBudget(max_chars_per_item=60, max_total_chars=60),
        auth=ResolverAuthContext(),
    )
    assert len(out.summaries[0].summary) <= 60


@pytest.mark.asyncio
async def test_identifier_parses_from_linear_url() -> None:
    fetcher = _FakeFetcher(_detail())
    r = LinearIssueResolver(fetcher=fetcher)
    out = await r.resolve(
        pot_id="pot-1",
        refs=[
            SourceReferenceRecord(
                ref="https://linear.app/potpie/issue/ENG-42/slug",
                source_type="issue",
                source_system="linear",
            )
        ],
        source_policy="summary",
        budget=ResolverBudget(),
        auth=ResolverAuthContext(),
    )
    assert fetcher.calls == ["ENG-42"]
    assert out.summaries
