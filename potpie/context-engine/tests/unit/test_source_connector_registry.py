"""SourceConnectorRegistry: routing, manifest, and fan-out behaviour."""

from __future__ import annotations

import asyncio
from typing import Iterable, Mapping, Sequence

from potpie_context_engine.application.services.source_connector_registry import (
    SourceConnectorRegistry,
)
from potpie_context_engine.domain.context_events import ContextEvent
from potpie_context_engine.domain.source_connector import (
    ConnectorScope,
    SourceCapability,
)
from potpie_context_engine.domain.source_references import SourceReferenceRecord
from potpie_context_engine.domain.source_resolution import (
    ResolvedSummary,
    ResolverAuthContext,
    ResolverBudget,
    SourceResolutionResult,
)


class _FakeConnector:
    """Tiny in-memory connector used to exercise the registry."""

    def __init__(self, kind: str, *, fetch: bool = True, webhook: bool = True) -> None:
        self._kind = kind
        self._caps = [
            SourceCapability(
                provider=kind,
                source_kind=f"{kind}_artifact",
                policies=frozenset({"summary"}),
                fetch_capable=fetch,
                list_capable=False,
                webhook_capable=webhook,
                sync_capable=False,
            )
        ]

    def kind(self) -> str:
        return self._kind

    def capabilities(self) -> Sequence[SourceCapability]:
        return tuple(self._caps)

    def list_artifacts(self, scope: ConnectorScope) -> Iterable[SourceReferenceRecord]:
        del scope
        return ()

    def normalize_webhook(
        self, payload: bytes, headers: Mapping[str, str]
    ) -> ContextEvent | None:
        del payload, headers
        return None

    async def fetch(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        del pot_id, source_policy, budget, auth
        out = SourceResolutionResult()
        for r in refs:
            out.summaries.append(
                ResolvedSummary(
                    ref=r.ref,
                    source_type=r.source_type,
                    summary=f"{self._kind}:{r.ref}",
                    source_system=self._kind,
                )
            )
        return out


def test_register_and_lookup_by_kind() -> None:
    reg = SourceConnectorRegistry()
    gh = _FakeConnector("github")
    ln = _FakeConnector("linear")
    reg.register(gh)
    reg.register(ln)

    assert reg.get("github") is gh
    assert reg.get("LINEAR") is ln  # case-insensitive
    assert reg.get("notion") is None
    assert {c.kind() for c in reg.all()} == {"github", "linear"}


def test_find_for_ref_matches_provider() -> None:
    reg = SourceConnectorRegistry()
    reg.register(_FakeConnector("github"))
    reg.register(_FakeConnector("linear"))

    pr_ref = SourceReferenceRecord(
        ref="pr:42", source_type="github_artifact", source_system="github"
    )
    found = reg.find_for_ref(pr_ref, policy="summary")
    assert found is not None and found.kind() == "github"

    issue_ref = SourceReferenceRecord(
        ref="ENG-1", source_type="linear_artifact", source_system="linear"
    )
    assert reg.find_for_ref(issue_ref, policy="summary") is not None

    # Unmatched policy → None
    assert reg.find_for_ref(pr_ref, policy="verify") is None


def test_manifest_for_pot_lists_registered_connectors() -> None:
    reg = SourceConnectorRegistry()
    reg.register(_FakeConnector("github"))
    reg.register(_FakeConnector("notion", fetch=False, webhook=False))

    manifest = reg.manifest_for_pot("pot-1")
    kinds = {m.kind for m in manifest}
    assert kinds == {"github", "notion"}
    for entry in manifest:
        assert entry.enabled is True  # both have at least one capability


def test_aggregated_capabilities_dedupe_policies() -> None:
    reg = SourceConnectorRegistry()
    reg.register(_FakeConnector("github"))
    caps = reg.aggregated_capabilities()
    assert caps and caps[0].provider == "github"
    assert "summary" in caps[0].policies


def test_resolve_dispatches_to_connector() -> None:
    reg = SourceConnectorRegistry()
    reg.register(_FakeConnector("github"))

    refs = [
        SourceReferenceRecord(
            ref="pr:42",
            source_type="github_artifact",
            source_system="github",
        )
    ]
    result = asyncio.run(
        reg.resolve(
            pot_id="p1",
            refs=refs,
            source_policy="summary",
            budget=ResolverBudget(),
            auth=ResolverAuthContext(),
        )
    )
    assert result.summaries and result.summaries[0].summary == "github:pr:42"


def test_resolve_emits_unsupported_for_unknown_provider() -> None:
    reg = SourceConnectorRegistry()
    reg.register(_FakeConnector("github"))

    refs = [
        SourceReferenceRecord(
            ref="page:abc",
            source_type="notion_page",
            source_system="notion",
        )
    ]
    result = asyncio.run(
        reg.resolve(
            pot_id="p1",
            refs=refs,
            source_policy="summary",
            budget=ResolverBudget(),
            auth=ResolverAuthContext(),
        )
    )
    assert not result.summaries
    assert any(f.code == "unsupported_source_type" for f in result.fallbacks)


def test_resolve_short_circuits_under_references_only_policy() -> None:
    reg = SourceConnectorRegistry()
    reg.register(_FakeConnector("github"))
    refs = [
        SourceReferenceRecord(
            ref="pr:42", source_type="github_artifact", source_system="github"
        )
    ]
    result = asyncio.run(
        reg.resolve(
            pot_id="p1",
            refs=refs,
            source_policy="references_only",
            budget=ResolverBudget(),
            auth=ResolverAuthContext(),
        )
    )
    assert not result.summaries
    assert not result.fallbacks


def test_third_source_smoke_test_notion_connector_loads() -> None:
    """Phase 2 smoke test: Notion connector instantiates from scratch
    and registers without touching application/ or domain/.

    If the contract were wrong we would have to edit code outside the
    connector module to add a new source — this test fails the moment
    that becomes true.
    """
    from potpie_context_engine.adapters.outbound.connectors.notion import (
        NotionConnector,
    )

    reg = SourceConnectorRegistry()
    reg.register(
        NotionConnector()
    )  # no fetcher → list_capable=False, fetch_capable=False

    manifest = {m.kind: m for m in reg.manifest_for_pot("p1")}
    assert "notion" in manifest
    notion_caps = manifest["notion"].capabilities
    assert notion_caps and notion_caps[0].provider == "notion"
