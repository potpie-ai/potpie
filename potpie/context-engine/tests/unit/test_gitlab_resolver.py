"""GitLab merge-request resolver: ref parsing, policies, and budget clamping."""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.connectors.gitlab.resolver import (
    GitLabMergeRequestResolver,
)
from potpie_context_core.source_references import SourceReferenceRecord
from potpie_context_engine.domain.source_resolution import (
    PERMISSION_DENIED,
    RESOLVER_ERROR,
    SOURCE_UNREACHABLE,
    UNSUPPORTED_SOURCE_POLICY,
    UNSUPPORTED_SOURCE_TYPE,
    ResolverAuthContext,
    ResolverBudget,
)

_MR = {
    "number": 7,
    "iid": 7,
    "title": "Fix login timeout",
    "body": "Why: the session cache expired early.",
    "state": "merged",
    "merged": True,
    "author": "dana",
    "url": "https://gitlab.corp.example/acme/api/-/merge_requests/7",
}


class _FakePort:
    def __init__(self, data=None, raises=None):
        self._data = data if data is not None else dict(_MR)
        self._raises = raises
        self.calls: list[tuple] = []

    def get_merge_request(self, project, iid, include_diff=False):
        self.calls.append((project, iid, include_diff))
        if self._raises is not None:
            raise self._raises
        return self._data


def _resolver(port=None, project="acme/api"):
    port = port or _FakePort()
    return (
        GitLabMergeRequestResolver(
            source_for_project=lambda _p: port,
            project_resolver=lambda _pot, _ref: project,
        ),
        port,
    )


def _ref(ref="gitlab:mr:acme/api:7", **kwargs):
    kwargs.setdefault("source_type", "merge_request")
    kwargs.setdefault("source_system", "gitlab")
    return SourceReferenceRecord(ref=ref, **kwargs)


async def _resolve(resolver, refs, policy="summary", budget=None):
    return await resolver.resolve(
        pot_id="pot-1",
        refs=refs,
        source_policy=policy,
        budget=budget or ResolverBudget(),
        auth=ResolverAuthContext(),
    )


# ----------------------------------------------------------------------
# Ref parsing
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        "gitlab:mr:acme/api:7",
        "gitlab:mr:acme/platform/api:7",
        "gitlab:mr:7",
        "!7",
        "MR !7",
        "https://gitlab.corp.example/acme/api/-/merge_requests/7",
        "7",
    ],
)
async def test_iid_parses_from_every_supported_ref_shape(value):
    resolver, port = _resolver()
    await _resolve(resolver, [_ref(value)])
    assert port.calls and port.calls[0][1] == 7


async def test_external_id_wins_over_the_ref_string():
    resolver, port = _resolver()
    await _resolve(resolver, [_ref("gitlab:mr:acme/api:7", external_id="9")])
    assert port.calls[0][1] == 9


async def test_unparseable_ref_reports_unsupported_source_type():
    resolver, port = _resolver()
    out = await _resolve(resolver, [_ref("gitlab:wiki:home")])
    assert [f.code for f in out.fallbacks] == [UNSUPPORTED_SOURCE_TYPE]
    assert port.calls == []


async def test_unresolvable_project_reports_unsupported_source_type():
    port = _FakePort()
    resolver = GitLabMergeRequestResolver(
        source_for_project=lambda _p: port,
        project_resolver=lambda _pot, _ref: None,
    )
    out = await _resolve(resolver, [_ref()])
    assert [f.code for f in out.fallbacks] == [UNSUPPORTED_SOURCE_TYPE]
    assert port.calls == []


async def test_async_project_resolver_is_awaited():
    port = _FakePort()

    async def _aresolve(_pot, _ref):
        return "acme/api"

    resolver = GitLabMergeRequestResolver(
        source_for_project=lambda _p: port, project_resolver=_aresolve
    )
    out = await _resolve(resolver, [_ref()])
    assert len(out.summaries) == 1


# ----------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------
async def test_summary_policy_composes_title_state_author_body():
    resolver, _ = _resolver()
    out = await _resolve(resolver, [_ref()], policy="summary")
    assert len(out.summaries) == 1
    summary = out.summaries[0]
    assert summary.source_system == "gitlab"
    assert summary.title == "Fix login timeout"
    assert "state=merged" in summary.summary
    assert "author=dana" in summary.summary
    assert summary.retrieval_uri.endswith("/merge_requests/7")


async def test_summary_policy_does_not_request_the_diff():
    resolver, port = _resolver()
    await _resolve(resolver, [_ref()], policy="summary")
    assert port.calls[0][2] is False


async def test_verify_policy_accepts_gitlab_state_vocabulary():
    resolver, _ = _resolver(_FakePort({**_MR, "state": "opened", "merged": False}))
    out = await _resolve(resolver, [_ref()], policy="verify")
    assert out.verifications[0].verified is True
    assert out.verifications[0].verification_state == "verified"


async def test_verify_policy_flags_an_unknown_state():
    resolver, _ = _resolver(_FakePort({**_MR, "state": "bogus", "merged": False}))
    out = await _resolve(resolver, [_ref()], policy="verify")
    assert out.verifications[0].verified is False
    assert "bogus" in out.verifications[0].reason


async def test_snippets_policy_requests_the_diff_and_splits_per_file():
    data = {
        **_MR,
        "files": [
            {"filename": "a.py", "previous_filename": "a.py", "patch": "@@\n-x\n+y"},
            {"filename": "b.py", "previous_filename": "b.py", "patch": "@@\n+z"},
        ],
    }
    resolver, port = _resolver(_FakePort(data))
    out = await _resolve(resolver, [_ref()], policy="snippets")
    assert port.calls[0][2] is True
    assert len(out.snippets) == 2
    assert [s.location for s in out.snippets] == ["b/a.py", "b/b.py"]
    assert all(s.source_system == "gitlab" for s in out.snippets)


async def test_snippets_policy_falls_back_to_the_body_without_a_diff():
    resolver, _ = _resolver(_FakePort({**_MR, "files": []}))
    out = await _resolve(resolver, [_ref()], policy="snippets")
    assert len(out.snippets) == 1
    assert "session cache" in out.snippets[0].snippet


async def test_unsupported_policy_is_reported():
    resolver, port = _resolver()
    out = await _resolve(resolver, [_ref()], policy="raw")
    assert [f.code for f in out.fallbacks] == [UNSUPPORTED_SOURCE_POLICY]
    assert port.calls == []


# ----------------------------------------------------------------------
# Budget
# ----------------------------------------------------------------------
async def test_per_item_budget_clamps_a_summary():
    resolver, _ = _resolver(_FakePort({**_MR, "body": "x" * 5000}))
    out = await _resolve(
        resolver, [_ref()], budget=ResolverBudget(max_chars_per_item=50)
    )
    assert len(out.summaries[0].summary) <= 50


async def test_total_budget_stops_after_the_first_ref():
    resolver, port = _resolver(_FakePort({**_MR, "body": "y" * 400}))
    out = await _resolve(
        resolver,
        [_ref("gitlab:mr:acme/api:7"), _ref("gitlab:mr:acme/api:8")],
        budget=ResolverBudget(max_chars_per_item=300, max_total_chars=300),
    )
    assert len(out.summaries) == 1
    assert len(port.calls) == 1  # the second ref is never fetched


async def test_snippet_count_is_capped_per_ref():
    files = [
        {"filename": f"f{i}.py", "previous_filename": f"f{i}.py", "patch": f"@@\n+{i}"}
        for i in range(6)
    ]
    resolver, _ = _resolver(_FakePort({**_MR, "files": files}))
    out = await _resolve(
        resolver,
        [_ref()],
        policy="snippets",
        budget=ResolverBudget(max_snippets_per_ref=2),
    )
    assert len(out.snippets) == 2


# ----------------------------------------------------------------------
# Failure translation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc,code",
    [
        (PermissionError("nope"), PERMISSION_DENIED),
        (ConnectionError("down"), SOURCE_UNREACHABLE),
        (TimeoutError("slow"), SOURCE_UNREACHABLE),
        (RuntimeError("boom"), RESOLVER_ERROR),
    ],
)
async def test_transport_failures_map_to_fallback_codes(exc, code):
    resolver, _ = _resolver(_FakePort(raises=exc))
    out = await _resolve(resolver, [_ref()])
    assert [f.code for f in out.fallbacks] == [code]
    assert out.summaries == []


async def test_one_bad_ref_does_not_stop_the_rest():
    resolver, port = _resolver()
    out = await _resolve(resolver, [_ref("not-a-ref"), _ref("gitlab:mr:acme/api:7")])
    assert len(out.summaries) == 1
    assert [f.code for f in out.fallbacks] == [UNSUPPORTED_SOURCE_TYPE]
    assert len(port.calls) == 1


def test_capabilities_declare_all_three_policies():
    resolver, _ = _resolver()
    (cap,) = resolver.capabilities()
    assert cap.provider == "gitlab"
    assert cap.policies == frozenset({"summary", "verify", "snippets"})
