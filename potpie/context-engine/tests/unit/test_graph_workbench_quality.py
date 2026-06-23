from __future__ import annotations

from datetime import datetime, timezone

import pytest

from potpie.context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from potpie.context_engine.application.services.graph_workbench import GraphWorkbenchService
from potpie.context_engine.domain.ports.claim_query import ClaimRow

pytestmark = pytest.mark.unit

POT = "p"


class _UnusedPlanStore:
    def save(self, _record) -> None:
        raise AssertionError("plan store should not be used by quality tests")

    def get(self, *, pot_id: str, plan_id: str):
        raise AssertionError("plan store should not be used by quality tests")

    def list(self, **_kwargs):
        raise AssertionError("plan store should not be used by quality tests")


def _service() -> tuple[GraphWorkbenchService, InMemoryGraphBackend]:
    backend = InMemoryGraphBackend()
    return GraphWorkbenchService(backend=backend, plan_store=_UnusedPlanStore()), backend


def _row(
    predicate: str,
    subject: str,
    object_: str,
    *,
    claim_key: str,
    truth: str = "source_observation",
    confidence: float | None = 0.9,
    source_refs: tuple[str, ...] = ("repo:manifest",),
    evidence: tuple[dict, ...] | None = None,
    valid_until: datetime | None = None,
    invalid_at: datetime | None = None,
) -> ClaimRow:
    return ClaimRow(
        pot_id=POT,
        predicate=predicate,
        subject_key=subject,
        object_key=object_,
        valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        invalid_at=invalid_at,
        claim_key=claim_key,
        subgraph="infra_topology",
        truth=truth,
        confidence=confidence,
        source_refs=source_refs,
        evidence=evidence
        if evidence is not None
        else ({"source_ref": source_refs[0]},)
        if source_refs
        else (),
        valid_until=valid_until,
    )


def test_quality_summary_aggregates_deep_report_counts() -> None:
    workbench, backend = _service()
    backend.store.add(_row("DEPENDS_ON", "service:web", "service:api", claim_key="c1"))
    backend.store.add(_row("OWNED_BY", "service:web", "team:a", claim_key="owner-a"))
    backend.store.add(_row("OWNED_BY", "service:web", "team:b", claim_key="owner-b"))

    result = workbench.quality(pot_id=POT, report="summary")

    assert result.ok is True
    assert result.report == "summary"
    assert result.status == "degraded"
    assert result.findings == ()
    assert result.metrics["counts"]["claims"] == 3
    assert result.metrics["backend_quality"]["claim_count"] == 3
    assert result.metrics["quality_counts"]["conflicting_claims"] >= 1
    assert (
        result.metrics["quality_reports"]["conflicting-claims"]["status"]
        == "degraded"
    )
    assert result.metrics["total_findings"] >= 1


def test_quality_duplicate_candidates_are_read_only() -> None:
    workbench, backend = _service()
    backend.store.add(_row("DEPENDS_ON", "service:api-a", "service:db", claim_key="c1"))
    backend.store.add(_row("DEPENDS_ON", "service:api-b", "service:db", claim_key="c2"))
    backend.store.set_entity_label(pot_id=POT, entity_key="service:api-a", labels=("Service",))
    backend.store.set_entity_label(pot_id=POT, entity_key="service:api-b", labels=("Service",))
    backend.store.set_entity_properties(
        pot_id=POT, entity_key="service:api-a", properties={"name": "Payments API"}
    )
    backend.store.set_entity_properties(
        pot_id=POT, entity_key="service:api-b", properties={"name": "payments_api"}
    )
    before = len(backend.store.rows)

    result = workbench.quality(pot_id=POT, report="duplicate-candidates")

    assert result.status == "watch"
    assert result.findings[0].kind == "duplicate-candidate"
    assert set(result.findings[0].entity_keys) == {"service:api-a", "service:api-b"}
    assert len(backend.store.rows) == before


def test_quality_stale_and_low_confidence_find_source_backed_claims() -> None:
    workbench, backend = _service()
    backend.store.add(
        _row(
            "DEPENDS_ON",
            "service:web",
            "service:api",
            claim_key="stale",
            valid_until=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
    )
    backend.store.add(
        _row(
            "DEFINED_IN",
            "service:web",
            "repo:github.com/acme/web",
            claim_key="weak",
            truth="authoritative_fact",
            confidence=0.2,
            source_refs=(),
            evidence=(),
        )
    )

    stale = workbench.quality(pot_id=POT, report="stale-facts")
    low = workbench.quality(pot_id=POT, report="low-confidence", confidence_threshold=0.5)

    assert stale.findings[0].claim_keys == ("stale",)
    assert stale.findings[0].source_refs == ("repo:manifest",)
    assert low.status == "degraded"
    assert low.findings[0].claim_keys == ("weak",)
    assert "missing required evidence" in (low.findings[0].detail or "")


def test_quality_conflicting_claims_detects_singleton_conflict() -> None:
    workbench, backend = _service()
    backend.store.add(_row("OWNED_BY", "service:web", "team:a", claim_key="owner-a"))
    backend.store.add(_row("OWNED_BY", "service:web", "team:b", claim_key="owner-b"))

    result = workbench.quality(pot_id=POT, report="conflicting-claims")

    assert result.status == "degraded"
    assert result.findings[0].kind == "conflicting-claim"
    assert set(result.findings[0].claim_keys) == {"owner-a", "owner-b"}


def test_quality_conflicting_claims_allows_multi_binding_predicates() -> None:
    workbench, backend = _service()
    backend.store.add(_row("USES", "service:web", "datastore:postgres", claim_key="pg"))
    backend.store.add(_row("USES", "service:web", "datastore:redis", claim_key="redis"))

    result = workbench.quality(pot_id=POT, report="conflicting-claims")

    assert result.status == "ok"
    assert result.findings == ()


def test_quality_orphan_entities_reports_entities_with_only_invalid_claims() -> None:
    workbench, backend = _service()
    backend.store.add(
        _row(
            "DEPENDS_ON",
            "service:old",
            "service:gone",
            claim_key="old",
            invalid_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
    )

    result = workbench.quality(pot_id=POT, report="orphan-entities")

    assert result.status == "watch"
    assert {finding.entity_keys[0] for finding in result.findings} == {
        "service:gone",
        "service:old",
    }


def test_quality_projection_drift_reports_invalid_endpoint_pairs() -> None:
    workbench, backend = _service()
    backend.store.add(_row("DEPENDS_ON", "service:web", "team:platform", claim_key="bad"))
    backend.store.set_entity_label(pot_id=POT, entity_key="service:web", labels=("Service",))
    backend.store.set_entity_label(pot_id=POT, entity_key="team:platform", labels=("Team",))

    result = workbench.quality(pot_id=POT, report="projection-drift")

    assert result.status == "degraded"
    assert result.findings[0].kind == "invalid-endpoint-pair"
    assert result.findings[0].claim_keys == ("bad",)


def test_quality_projection_drift_uses_ontology_endpoint_semantics() -> None:
    workbench, backend = _service()
    backend.store.add(
        _row(
            "POLICY_APPLIES_TO",
            "preference:query-graph",
            "repo:github.com/acme/web",
            claim_key="scope",
        )
    )
    backend.store.add(
        _row(
            "MENTIONS",
            "activity:fix-123",
            "bug_pattern:timeout",
            claim_key="wildcard",
        )
    )
    backend.store.add(
        _row(
            "IMPLEMENTED_IN",
            "feature:login",
            "file:src/login.py",
            claim_key="code-asset",
        )
    )
    backend.store.set_entity_label(
        pot_id=POT,
        entity_key="preference:query-graph",
        labels=("Entity", "Preference"),
    )
    backend.store.set_entity_label(
        pot_id=POT,
        entity_key="repo:github.com/acme/web",
        labels=("Entity", "Repository"),
    )
    backend.store.set_entity_label(
        pot_id=POT,
        entity_key="activity:fix-123",
        labels=("Entity", "Activity", "Fix"),
    )
    backend.store.set_entity_label(
        pot_id=POT,
        entity_key="bug_pattern:timeout",
        labels=("Entity", "BugPattern"),
    )
    backend.store.set_entity_label(
        pot_id=POT,
        entity_key="feature:login",
        labels=("Entity", "Feature"),
    )
    backend.store.set_entity_label(
        pot_id=POT,
        entity_key="file:src/login.py",
        labels=("Entity", "FILE"),
    )

    result = workbench.quality(pot_id=POT, report="projection-drift")

    assert result.status == "ok"
    assert result.findings == ()
