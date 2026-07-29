"""Per-runtime reconciliation configuration is immutable and isolated."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from potpie_context_core.context_events import EventRef
from potpie_context_core.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_core.reconciliation import ReconciliationPlan
from potpie_context_core.reconciliation_config import ReconciliationConfig
from potpie_context_core.reconciliation_flags import reconciliation_config_from_env
from potpie_context_core.reconciliation_validation import validate_reconciliation_plan

pytestmark = pytest.mark.unit


def test_environment_snapshots_coexist_after_environment_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_env = {
        "CONTEXT_ENGINE_RECONCILIATION_ENABLED": "0",
        "CONTEXT_ENGINE_AGENT_PLANNER_ENABLED": "1",
        "CONTEXT_ENGINE_INFER_LABELS": "0",
        "CONTEXT_ENGINE_CONFLICT_DETECT": "0",
        "CONTEXT_ENGINE_AUTO_SUPERSEDE": "1",
        "CONTEXT_ENGINE_CAUSAL_EXPAND": "0",
        "CONTEXT_ENGINE_STRICT_EXTRACTION": "0",
        "CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL": "1",
        "CONTEXT_ENGINE_ONTOLOGY_STRICT": "0",
    }
    for name, value in first_env.items():
        monkeypatch.setenv(name, value)
    first = reconciliation_config_from_env()

    second_env = {
        "CONTEXT_ENGINE_RECONCILIATION_ENABLED": "1",
        "CONTEXT_ENGINE_AGENT_PLANNER_ENABLED": "0",
        "CONTEXT_ENGINE_INFER_LABELS": "1",
        "CONTEXT_ENGINE_CONFLICT_DETECT": "1",
        "CONTEXT_ENGINE_AUTO_SUPERSEDE": "0",
        "CONTEXT_ENGINE_CAUSAL_EXPAND": "1",
        "CONTEXT_ENGINE_STRICT_EXTRACTION": "1",
        "CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL": "1",
        "CONTEXT_ENGINE_ONTOLOGY_STRICT": "1",
    }
    for name, value in second_env.items():
        monkeypatch.setenv(name, value)
    second = reconciliation_config_from_env()

    assert first == ReconciliationConfig(
        enabled=False,
        agent_planner_enabled=True,
        infer_canonical_labels=False,
        conflict_detection=False,
        auto_supersede=True,
        causal_expand=False,
        strict_extraction=False,
        ontology_soft_fail=True,
        ontology_strict=False,
    )
    assert second == ReconciliationConfig(
        enabled=True,
        agent_planner_enabled=False,
        infer_canonical_labels=True,
        conflict_detection=True,
        auto_supersede=False,
        causal_expand=True,
        strict_extraction=True,
        ontology_soft_fail=False,
        ontology_strict=True,
    )


def test_configuration_is_frozen_and_rejects_conflicting_ontology_modes() -> None:
    config = ReconciliationConfig()

    with pytest.raises(FrozenInstanceError):
        config.enabled = False  # type: ignore[misc]
    with pytest.raises(ValueError, match="cannot both be enabled"):
        ReconciliationConfig(ontology_soft_fail=True, ontology_strict=True)


def _label_inference_plan() -> ReconciliationPlan:
    return ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="test", pot_id="p1"),
        summary="deployment",
        entity_upserts=[
            EntityUpsert(
                entity_key="service:ledger",
                labels=("Service",),
                properties={"name": "ledger"},
            ),
            EntityUpsert(
                entity_key="environment:prod",
                labels=("Entity",),
                properties={},
            ),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DEPLOYED_TO",
                from_entity_key="service:ledger",
                to_entity_key="environment:prod",
                properties={},
            )
        ],
    )


def test_explicit_validation_configs_ignore_later_environment_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infer = ReconciliationConfig(infer_canonical_labels=True)
    do_not_infer = ReconciliationConfig(infer_canonical_labels=False)

    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")
    inferred_plan = _label_inference_plan()
    validate_reconciliation_plan(inferred_plan, "p1", config=infer)

    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "1")
    uninferred_plan = _label_inference_plan()
    validate_reconciliation_plan(uninferred_plan, "p1", config=do_not_infer)

    inferred = {item.entity_key: item for item in inferred_plan.entity_upserts}
    uninferred = {item.entity_key: item for item in uninferred_plan.entity_upserts}
    assert "Environment" in inferred["environment:prod"].labels
    assert "Environment" in uninferred["environment:prod"].labels
    assert "Observation" not in uninferred["environment:prod"].labels


def test_label_coherence_remains_enabled_when_inference_is_disabled() -> None:
    """Keep key-prefix coherence active when classifier inference is disabled."""
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", pot_id="p1", source_system="test"),
        summary="test",
        entity_upserts=[
            EntityUpsert(
                entity_key="environment:prod",
                labels=("Entity", "Service", "Environment"),
                properties={"name": "prod"},
            )
        ],
    )

    validate_reconciliation_plan(
        plan,
        "p1",
        config=ReconciliationConfig(infer_canonical_labels=False),
    )

    assert plan.entity_upserts[0].labels == ("Entity", "Environment")
