"""Build one deterministic entity-repair delta for every backend."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from potpie_context_engine.core.definition import GraphDefinition
from potpie_context_engine.core.graph_mutations import EntityUpsert
from potpie_context_engine.core.reconciliation import MutationBatch

from .entity_label_repair import (
    ENTITY_LABEL_TARGET,
    repaired_entity_labels,
    wants_entity_label_repair,
)
from .entity_summary_repair import (
    ENTITY_SUMMARY_TARGET,
    repaired_entity_properties,
    wants_entity_summary_repair,
)


def prepare_entity_repairs(
    rows: Iterable[Mapping[str, Any]],
    *,
    definition: GraphDefinition,
    targets: Sequence[str],
) -> tuple[MutationBatch, dict[str, int]]:
    summaries = wants_entity_summary_repair(targets)
    labels = wants_entity_label_repair(targets)
    repaired = {}
    if summaries:
        repaired[ENTITY_SUMMARY_TARGET] = 0
    if labels:
        repaired[ENTITY_LABEL_TARGET] = 0
    operations = []
    for row in rows:
        key = row["key"]
        props = row["props"]
        if props.get("retired", False):
            continue
        current_labels = tuple(row["labels"])
        fixed_props = repaired_entity_properties(key, props) if summaries else None
        fixed_labels = (
            repaired_entity_labels(
                key, current_labels, entity_types=definition.entity_types
            )
            if labels and current_labels
            else None
        )
        if fixed_props is None and fixed_labels is None:
            continue
        operations.append(
            EntityUpsert(
                key,
                fixed_labels or current_labels or ("Entity",),
                fixed_props or {},
            )
        )
        if fixed_props is not None:
            repaired[ENTITY_SUMMARY_TARGET] += 1
        if fixed_labels is not None:
            repaired[ENTITY_LABEL_TARGET] += 1
    return MutationBatch(
        summary="Repair entity summaries/labels",
        entity_upserts=operations,
    ), repaired
