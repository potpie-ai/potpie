"""Edge-case tests for reconciliation plan validation (boundaries, ontology errors)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from application.use_cases.reconciliation_validation import (
    MAX_EPISODES,
    MAX_GENERIC_EDGES,
    MAX_GENERIC_ENTITY_UPSERTS,
    MAX_INVALIDATIONS,
    validate_reconciliation_plan,
)
from domain.context_events import EventRef
from domain.errors import ReconciliationPlanValidationError
from domain.graph_mutations import EdgeDelete, EdgeUpsert, EntityUpsert, InvalidationOp
from domain.reconciliation import EpisodeDraft, ReconciliationPlan

pytestmark = pytest.mark.unit


def _ref(pot_id: str = "p1") -> EventRef:
    return EventRef(event_id="e1", source_system="github", pot_id=pot_id)


def _episode() -> EpisodeDraft:
    return EpisodeDraft(
        name="ep",
        episode_body="body",
        source_description="test",
        reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _valid_entity_upsert(key: str = "source-ref:github:pr:1") -> EntityUpsert:
    return EntityUpsert(
        entity_key=key,
        labels=("Entity", "SourceReference"),
        properties={"source_system": "github", "external_id": "pr:1", "ref_type": "pull_request"},
    )


def _valid_edge_upsert() -> EdgeUpsert:
    return EdgeUpsert(
        edge_type="FROM_SOURCE",
        from_entity_key="source-ref:github:pr:1",
        to_entity_key="source-system:github",
        properties={},
    )


def _valid_plan(**kwargs) -> ReconciliationPlan:  # type: ignore[no-untyped-def]
    defaults = dict(event_ref=_ref(), summary="ok", episodes=[])
    defaults.update(kwargs)
    return ReconciliationPlan(**defaults)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_validate_empty_plan_passes() -> None:
    validate_reconciliation_plan(_valid_plan(), "p1")


def test_validate_plan_with_episodes_passes() -> None:
    plan = _valid_plan(episodes=[_episode(), _episode()])
    validate_reconciliation_plan(plan, "p1")


def test_validate_plan_with_valid_structural_mutations_passes() -> None:
    plan = _valid_plan(
        entity_upserts=[_valid_entity_upsert()],
        edge_upserts=[_valid_edge_upsert()],
    )
    validate_reconciliation_plan(plan, "p1")


# ---------------------------------------------------------------------------
# Pot ID mismatch
# ---------------------------------------------------------------------------


def test_validate_pot_id_mismatch_raises() -> None:
    plan = _valid_plan()
    with pytest.raises(ReconciliationPlanValidationError, match="pot_id"):
        validate_reconciliation_plan(plan, "wrong-pot")


# ---------------------------------------------------------------------------
# Episode cap
# ---------------------------------------------------------------------------


def test_validate_exactly_max_episodes_passes() -> None:
    plan = _valid_plan(episodes=[_episode()] * MAX_EPISODES)
    validate_reconciliation_plan(plan, "p1")


def test_validate_exceeds_max_episodes_raises() -> None:
    plan = _valid_plan(episodes=[_episode()] * (MAX_EPISODES + 1))
    with pytest.raises(ReconciliationPlanValidationError, match="too many episodes"):
        validate_reconciliation_plan(plan, "p1")


# ---------------------------------------------------------------------------
# Entity upsert cap
# ---------------------------------------------------------------------------


def test_validate_exactly_max_entity_upserts_passes() -> None:
    upserts = [
        EntityUpsert(
            entity_key=f"source-ref:github:pr:{i}",
            labels=("Entity", "SourceReference"),
            properties={"source_system": "github", "external_id": f"pr:{i}", "ref_type": "pull_request"},
        )
        for i in range(MAX_GENERIC_ENTITY_UPSERTS)
    ]
    plan = _valid_plan(entity_upserts=upserts)
    validate_reconciliation_plan(plan, "p1")


def test_validate_exceeds_max_entity_upserts_raises() -> None:
    upserts = [
        EntityUpsert(
            entity_key=f"source-ref:github:pr:{i}",
            labels=("Entity", "SourceReference"),
            properties={"source_system": "github", "external_id": f"pr:{i}", "ref_type": "pull_request"},
        )
        for i in range(MAX_GENERIC_ENTITY_UPSERTS + 1)
    ]
    plan = _valid_plan(entity_upserts=upserts)
    with pytest.raises(ReconciliationPlanValidationError, match="entity upsert cap"):
        validate_reconciliation_plan(plan, "p1")


# ---------------------------------------------------------------------------
# Edge cap (upserts + deletes combined)
# ---------------------------------------------------------------------------


def test_validate_edge_upsert_plus_delete_combined_cap() -> None:
    half = MAX_GENERIC_EDGES // 2
    edge_upserts = [
        EdgeUpsert(
            edge_type="FROM_SOURCE",
            from_entity_key=f"source-ref:github:pr:{i}",
            to_entity_key="source-system:github",
            properties={},
        )
        for i in range(half + 1)
    ]
    edge_deletes = [
        EdgeDelete(
            edge_type="FROM_SOURCE",
            from_entity_key=f"source-ref:github:pr:{i}",
            to_entity_key="source-system:github",
        )
        for i in range(half + 1)
    ]
    plan = _valid_plan(edge_upserts=edge_upserts, edge_deletes=edge_deletes)
    with pytest.raises(ReconciliationPlanValidationError, match="edge mutation cap"):
        validate_reconciliation_plan(plan, "p1")


# ---------------------------------------------------------------------------
# Invalidation cap
# ---------------------------------------------------------------------------


def test_validate_exactly_max_invalidations_passes() -> None:
    invalidations = [
        InvalidationOp(target_entity_key=f"source-ref:github:pr:{i}", target_edge=None, reason="r")
        for i in range(MAX_INVALIDATIONS)
    ]
    plan = _valid_plan(invalidations=invalidations)
    validate_reconciliation_plan(plan, "p1")


def test_validate_exceeds_max_invalidations_raises() -> None:
    invalidations = [
        InvalidationOp(target_entity_key=f"source-ref:github:pr:{i}", target_edge=None, reason="r")
        for i in range(MAX_INVALIDATIONS + 1)
    ]
    plan = _valid_plan(invalidations=invalidations)
    with pytest.raises(ReconciliationPlanValidationError, match="invalidation cap"):
        validate_reconciliation_plan(plan, "p1")


# ---------------------------------------------------------------------------
# Ontology error sampling
# ---------------------------------------------------------------------------


def test_validate_ontology_error_message_samples_first_8() -> None:
    # Create 12 invalid entity upserts — ontology will reject unknown labels
    upserts = [
        EntityUpsert(
            entity_key=f"thing:x:{i}",
            labels=("Entity", f"UnknownLabel{i}"),
            properties={},
        )
        for i in range(12)
    ]
    plan = _valid_plan(entity_upserts=upserts)
    with pytest.raises(ReconciliationPlanValidationError) as exc_info:
        validate_reconciliation_plan(plan, "p1")
    message = str(exc_info.value)
    assert "more" in message  # suffix "... X more" should appear


def test_validate_ontology_error_no_suffix_when_8_or_fewer() -> None:
    # Each invalid entity upsert produces 2 ontology errors (unknown label + missing
    # public canonical label). 4 entities → 8 errors → no "... X more" suffix.
    upserts = [
        EntityUpsert(
            entity_key=f"thing:x:{i}",
            labels=("Entity", f"UnknownLabel{i}"),
            properties={},
        )
        for i in range(4)
    ]
    plan = _valid_plan(entity_upserts=upserts)
    with pytest.raises(ReconciliationPlanValidationError) as exc_info:
        validate_reconciliation_plan(plan, "p1")
    message = str(exc_info.value)
    assert "more" not in message
