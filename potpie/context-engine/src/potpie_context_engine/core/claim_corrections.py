"""Exact correction selection and interval effects, shared by every substrate.

Preparation freezes these values over its revision-checked read set. Raw writers
use the same policy inside their atomic boundary. Execution checks the original
interval and applies the reviewed result without consulting its wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Iterable, Mapping, TYPE_CHECKING
import hashlib

from .claim_codec import _as_utc, iso, parse_dt, normalize_timestamp
from .claim_lifecycle import (
    ClaimLifecycleConflict,
    claim_scope,
    close_claim_at,
    validate_incarnation_history,
)
from .ports.claim_query import ClaimRow

if TYPE_CHECKING:
    from .graph_mutations import EdgeDelete, EdgeUpsert, InvalidationOp
    from .reconciliation import MutationBatch


@dataclass(frozen=True, slots=True)
class SupersessionBinding:
    """Preparation-only association between two authored batch objects."""

    effect: EdgeUpsert
    replacement: EdgeUpsert


@dataclass(frozen=True, slots=True)
class ClaimTarget:
    claim_key: str | None
    record_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ClaimLifecycleConflict("correction requires an incarnation identity")
        if self.claim_key is not None and (
            not isinstance(self.claim_key, str) or not self.claim_key
        ):
            raise ClaimLifecycleConflict("invalid correction claim key")

    def to_dict(self) -> dict[str, Any]:
        return {"claim_key": self.claim_key, "record_id": self.record_id}


@dataclass(frozen=True, slots=True)
class ClaimInterval:
    valid_at: datetime | None
    invalid_at: datetime | None
    valid_until: datetime | None

    def __post_init__(self) -> None:
        for name in ("valid_at", "invalid_at", "valid_until"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, datetime) or value.tzinfo is None:
                    raise ClaimLifecycleConflict(
                        "correction timestamps require UTC offsets"
                    )
                object.__setattr__(self, name, _as_utc(value))

    @classmethod
    def from_row(cls, row: ClaimRow) -> ClaimInterval:
        return cls(row.valid_at, row.invalid_at, row.valid_until)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: iso(getattr(self, name))
            for name in ("valid_at", "invalid_at", "valid_until")
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ClaimInterval:
        values = {}
        for name in ("valid_at", "invalid_at", "valid_until"):
            value = raw[name]
            parsed = parse_dt(value)
            if value is not None and parsed is None:
                raise ClaimLifecycleConflict("invalid persisted correction timestamp")
            values[name] = parsed
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ClaimClosureResolution:
    target: ClaimTarget
    before: ClaimInterval
    after: ClaimInterval

    def __post_init__(self) -> None:
        if self.before.valid_at != self.after.valid_at:
            raise ClaimLifecycleConflict(
                "correction cannot change the incarnation start"
            )
        if self.after.valid_at is not None and any(
            end is not None and end <= self.after.valid_at
            for end in (self.after.invalid_at, self.after.valid_until)
        ):
            raise ClaimLifecycleConflict(
                "corrected interval must have positive duration"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ClaimClosureResolution:
        return cls(
            ClaimTarget(**raw["target"]),
            ClaimInterval.from_dict(raw["before"]),
            ClaimInterval.from_dict(raw["after"]),
        )


def apply_claim_closure(row: ClaimRow, closure: ClaimClosureResolution) -> ClaimRow:
    """Validate one physical record and apply its frozen interval."""
    from .claim_lifecycle import require_writable_claim

    require_writable_claim(row)
    if ClaimTarget(row.claim_key, row.record_id) != closure.target:
        raise ClaimLifecycleConflict("frozen correction identity changed")
    if ClaimInterval.from_row(row) != closure.before:
        raise ClaimLifecycleConflict("frozen correction interval changed")
    if row.retired and closure.before != closure.after:
        raise ClaimLifecycleConflict("correction cannot modify a retired incarnation")
    if closure.before == closure.after:
        return row
    return replace(
        row,
        valid_at=closure.after.valid_at,
        invalid_at=closure.after.invalid_at,
        valid_until=closure.after.valid_until,
    )


def _selected_rows(
    op: EdgeDelete | InvalidationOp, rows: list[ClaimRow]
) -> list[ClaimRow]:
    from .graph_mutations import EdgeDelete

    if isinstance(op, EdgeDelete):
        edge = (op.edge_type, op.from_entity_key, op.to_entity_key)
        return [
            row
            for row in rows
            if not row.retired
            and (row.predicate, row.subject_key, row.object_key) == edge
        ]
    for values, kind in (
        (op.target_claim_keys, "claim key"),
        (op.target_claim_record_ids, "incarnation"),
    ):
        if values is not None and len(set(values)) != len(values):
            raise ClaimLifecycleConflict(f"duplicate correction {kind} target")
    if op.target_claim_keys == () or op.target_claim_record_ids == ():
        return []
    selected = [row for row in rows if not row.retired]
    if op.target_edge is not None:
        selected = [
            row
            for row in selected
            if (row.predicate, row.subject_key, row.object_key) == op.target_edge
        ]
    if op.target_entity_key is not None:
        selected = [
            row
            for row in selected
            if op.target_entity_key in (row.subject_key, row.object_key)
        ]
    if op.target_claim_keys is not None:
        selected = [row for row in selected if row.claim_key in op.target_claim_keys]
        if op.target_claim_record_ids is None and len(selected) != len(
            {row.claim_key for row in selected}
        ):
            raise ClaimLifecycleConflict(
                "correction claim key selects multiple incarnations"
            )
    if op.target_claim_record_ids is not None:
        selected = [
            row for row in selected if row.record_id in op.target_claim_record_ids
        ]
        if {row.record_id for row in selected} != set(op.target_claim_record_ids):
            raise ClaimLifecycleConflict(
                "correction incarnation is missing or outside its selected scope"
            )
    if op.target_claim_keys is not None and {row.claim_key for row in selected} != set(
        op.target_claim_keys
    ):
        raise ClaimLifecycleConflict(
            "correction claim key is missing or outside its selected scope"
        )
    return selected


def prepare_claim_closures(
    batch: MutationBatch,
    rows: Iterable[ClaimRow],
    *,
    anchor: datetime,
    bindings: list[SupersessionBinding],
    effective_at: datetime | None = None,
    singleton_predicates: Iterable[str] = (),
) -> list[ClaimRow]:
    """Freeze unresolved closures and explicit derived effects on a private batch.

    Call after raw/prepared input validation. A frozen operation is validated only;
    its selector never chooses additional records. Returns the virtual post-closure
    history used to prepare assertions. All rows must belong to the same pot.
    """
    from .graph_mutations import InvalidationOp

    current = list(rows)
    seen: set[str] = set()
    touched: set[tuple] = set()
    for op in (*batch.edge_deletes, *batch.invalidations):
        unresolved = op.closures is None
        selected = (
            _selected_rows(op, current)
            if unresolved
            else [
                row
                for row in current
                if row.record_id in {c.target.record_id for c in op.closures}
            ]
        )
        if any(row.temporal_diagnostics for row in selected):
            raise ClaimLifecycleConflict(
                "migration_required: correction target has ambiguous stored timestamps"
            )
        if not unresolved and len(_selected_rows(op, selected)) != len(selected):
            raise ClaimLifecycleConflict(
                "frozen correction target is outside its selected scope"
            )
        if unresolved:
            point = effective_at or anchor
            if isinstance(op, InvalidationOp) and op.valid_to is not None:
                try:
                    point = parse_dt(normalize_timestamp(op.valid_to, field="valid_to"))
                except ValueError as exc:
                    raise ClaimLifecycleConflict(str(exc)) from exc
            try:
                point = parse_dt(
                    normalize_timestamp(point, field="correction effective time")
                )
            except ValueError as exc:
                raise ClaimLifecycleConflict(str(exc)) from exc
            closures = []
            for row in selected:
                before = ClaimInterval.from_row(row)
                if (
                    isinstance(op, InvalidationOp)
                    and op.target_claim_record_ids is not None
                ):
                    after = ClaimInterval(
                        before.valid_at,
                        point,
                        point if before.valid_until is not None else None,
                    )
                else:
                    after = ClaimInterval.from_row(
                        close_claim_at(row, point, anchor=anchor)
                    )
                closures.append(
                    ClaimClosureResolution(
                        ClaimTarget(row.claim_key, row.record_id), before, after
                    )
                )
            op.closures = tuple(closures)
            if isinstance(op, InvalidationOp):
                _derived_effects(
                    batch, op, selected, point, bindings, observed_at=anchor
                )
                op.target_claim_keys = (
                    tuple(dict.fromkeys(c.target.claim_key for c in closures))
                    if all(c.target.claim_key is not None for c in closures)
                    else None
                )
                op.target_claim_record_ids = tuple(c.target.record_id for c in closures)
        by_id: dict[str, list[ClaimRow]] = {}
        for row in selected:
            by_id.setdefault(row.record_id, []).append(row)
        for closure in op.closures:
            identity = closure.target.record_id
            if identity in seen:
                raise ClaimLifecycleConflict("duplicate correction incarnation target")
            seen.add(identity)
            matches = by_id.get(identity, [])
            if len(matches) != 1:
                raise ClaimLifecycleConflict(
                    "frozen correction incarnation missing or ambiguous"
                )
            previous = matches[0]
            updated = apply_claim_closure(previous, closure)
            touched.add(claim_scope(previous))
            current = [updated if row is previous else row for row in current]
        if isinstance(op, InvalidationOp):
            if op.target_claim_record_ids is not None and set(
                op.target_claim_record_ids
            ) != {c.target.record_id for c in op.closures}:
                raise ClaimLifecycleConflict(
                    "correction selector differs from frozen incarnations"
                )
            if op.target_claim_keys is not None and set(op.target_claim_keys) != {
                c.target.claim_key
                for c in op.closures
                if c.target.claim_key is not None
            }:
                raise ClaimLifecycleConflict(
                    "correction selector differs from frozen claim keys"
                )
    validate_incarnation_history(
        (row for row in current if claim_scope(row) in touched and not row.retired),
        singleton_predicates=singleton_predicates,
    )
    return current


def _derived_effects(
    batch: MutationBatch,
    op: InvalidationOp,
    selected: list[ClaimRow],
    point: datetime,
    bindings: list[SupersessionBinding],
    *,
    observed_at: datetime,
) -> None:
    from .graph_mutations import EdgeUpsert, EntityUpsert

    if not selected:
        return
    # Semantic selectors always have exact keys/IDs. Only raw structural entity
    # invalidation intentionally projects the lifecycle onto the entity itself.
    if (
        op.target_entity_key
        and op.target_claim_keys is None
        and op.target_claim_record_ids is None
    ):
        props = {
            "invalidated": True,
            "invalidation_reason": op.reason,
            "valid_to": iso(point),
            "invalid_at": iso(point),
        }
        entity = next(
            (
                item
                for item in batch.entity_upserts
                if item.entity_key == op.target_entity_key
            ),
            None,
        )
        if entity is not None:
            entity.properties.update(props)
        else:
            batch.entity_upserts.append(
                EntityUpsert(op.target_entity_key, (), props, preserve_labels=True)
            )
    if not op.superseded_by_key:
        return
    for row in selected:
        old_key = op.target_entity_key or row.object_key
        if not old_key or old_key == op.superseded_by_key:
            continue
        identity = hashlib.blake2b(
            f"{row.pot_id}:{row.record_id}:{op.superseded_by_key}".encode(),
            digest_size=20,
        ).hexdigest()
        replacements = [
            edge
            for edge in batch.edge_upserts
            if edge.edge_type == row.predicate
            and edge.from_entity_key == row.subject_key
            and edge.to_entity_key == op.superseded_by_key
        ]
        if op.replacement_claim_key is not None:
            replacements = [
                edge
                for edge in replacements
                if edge.properties.get("claim_key") == op.replacement_claim_key
            ]
            if len(replacements) != 1:
                raise ClaimLifecycleConflict(
                    "supersession replacement claim is missing or ambiguous"
                )
        elif len(replacements) > 1:
            scoped = [
                edge
                for edge in replacements
                if claim_scope(
                    replace(
                        row,
                        subgraph=edge.properties.get("subgraph"),
                        environment=edge.properties.get("environment"),
                        source_system=edge.properties.get("source_system"),
                        source_ref=edge.properties.get("source_ref"),
                    )
                )
                == claim_scope(row)
            ]
            if len(scoped) != 1:
                raise ClaimLifecycleConflict(
                    "raw supersession requires an exact replacement_claim_key association"
                )
            replacements = scoped
        replacement = replacements[0] if replacements else None
        effect = EdgeUpsert(
            "SUPERSEDES",
            op.superseded_by_key,
            old_key,
            {
                "claim_key": f"supersedes:{identity}",
                "subgraph": row.subgraph,
                "environment": row.environment,
                "source_system": row.source_system,
                "source_ref": row.source_ref,
                "source_identity_stable": False,
                "valid_at": iso(point),
                "observed_at": iso(observed_at),
                "fact": f"{op.superseded_by_key} supersedes {old_key}: {op.reason}",
                "correction_reason": op.reason,
                "supersedes_claim_key": row.claim_key,
                "supersedes_record_id": row.record_id,
                "replacement_claim_key": (
                    replacement.properties.get("claim_key") if replacement else None
                ),
            },
        )
        batch.edge_upserts.append(effect)
        if replacement is not None:
            bindings.append(SupersessionBinding(effect, replacement))


def bind_supersession_references(bindings: Iterable[SupersessionBinding]) -> None:
    """Freeze physical references after resolving all assertions in the batch."""
    for binding in bindings:
        if binding.replacement.resolution is None:
            raise ClaimLifecycleConflict("supersession replacement was not resolved")
        binding.effect.properties["replacement_record_id"] = (
            binding.replacement.resolution.record_id
        )
