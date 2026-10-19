"""One assertion lifecycle policy for preparation and every graph substrate.

Adapters supply normalized candidate rows from their consistency boundary. A
prepared resolution selects physical records once; execution validates those
identities and applies the resolution without selecting replacements.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any

from .claim_codec import _as_utc, iso, parse_dt
from .ports.claim_query import ClaimRow


class ClaimLifecycleConflict(ValueError):
    """An assertion needs explicit correction or a fresh preparation."""


REVIEW_FIELDS = frozenset(
    {
        "evidence_review_required",
        "evidence_review_reason",
        "evidence_review_refs",
    }
)
_TRUTH_FIELDS = (
    "policy_kind",
    "prescription",
    "strength",
    "audience",
    "symptom_signature",
    "fix_steps",
    "verification_status",
    "resolution_status",
    "change_kind",
    "ticket_key",
    "pr_number",
    "code_scope",
)


@dataclass(frozen=True, slots=True)
class ClaimWriteResolution:
    record_id: str
    existing: bool
    effective_at: datetime
    valid_at: datetime | None
    invalid_at: datetime | None
    valid_until: datetime | None
    superseded_record_ids: tuple[str, ...] = ()
    metadata_review: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ClaimLifecycleConflict("claim incarnation identity is required")
        if type(self.existing) is not bool or type(self.metadata_review) is not bool:
            raise ClaimLifecycleConflict("invalid assertion resolution mode")
        for value in (
            self.effective_at,
            self.valid_at,
            self.invalid_at,
            self.valid_until,
        ):
            if value is not None and (
                not isinstance(value, datetime) or value.tzinfo is None
            ):
                raise ClaimLifecycleConflict(
                    "claim lifecycle timestamps require UTC offsets"
                )
        if not isinstance(self.effective_at, datetime):
            raise ClaimLifecycleConflict("assertion effective time is required")
        if any(
            not isinstance(value, str) or not value
            for value in self.superseded_record_ids
        ):
            raise ClaimLifecycleConflict("invalid predecessor incarnation identity")
        if len(set(self.superseded_record_ids)) != len(self.superseded_record_ids):
            raise ClaimLifecycleConflict("duplicate predecessor incarnation identity")
        if self.record_id in self.superseded_record_ids:
            raise ClaimLifecycleConflict("assertion cannot supersede itself")
        if self.valid_at is not None and any(
            end is not None and end <= self.valid_at
            for end in (self.invalid_at, self.valid_until)
        ):
            raise ClaimLifecycleConflict(
                "resolved interval must have positive duration"
            )
        if not self.existing and self.valid_at != self.effective_at:
            raise ClaimLifecycleConflict(
                "new incarnation must start at its assertion time"
            )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        for name in ("effective_at", "valid_at", "invalid_at", "valid_until"):
            raw[name] = iso(getattr(self, name))
        raw["superseded_record_ids"] = list(self.superseded_record_ids)
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ClaimWriteResolution:
        fields = dict(raw)
        for name in ("effective_at", "valid_at", "invalid_at", "valid_until"):
            value = fields[name]
            parsed = parse_dt(value)
            if value is not None and parsed is None:
                raise ClaimLifecycleConflict("invalid persisted lifecycle timestamp")
            fields[name] = parsed
        fields["superseded_record_ids"] = tuple(fields.get("superseded_record_ids", ()))
        return cls(**fields)


def _scope_value(value: str | None, *, lower: bool = False) -> str | None:
    value = value.strip() if value else None
    return value.lower() if value and lower else value or None


def claim_scope(row: ClaimRow) -> tuple[str | None, ...]:
    return (
        row.pot_id,
        _scope_value(row.subgraph, lower=True),
        row.subject_key,
        row.predicate,
        _scope_value(row.environment, lower=True),
        _scope_value(row.source_system),
        _scope_value(row.source_ref),
    )


def has_stable_source(row: ClaimRow) -> bool:
    return bool(
        row.properties.get("source_identity_stable") is True
        and _scope_value(row.source_system)
        and _scope_value(row.source_ref)
    )


def interval_end(row: ClaimRow) -> datetime | None:
    ends = [
        _as_utc(value)
        for value in (row.invalid_at, row.valid_until)
        if value is not None
    ]
    return min(ends) if ends else None


def close_claim_at(row: ClaimRow, point: datetime, *, anchor: datetime) -> ClaimRow:
    """Soft closure shared by ordinary and journal mutations; never erase history."""
    point, anchor = _as_utc(point), _as_utc(anchor)
    end = interval_end(row)
    if row.retired or end is not None and end <= point:
        return row
    if end is not None and end <= anchor:
        raise ClaimLifecycleConflict(
            "changing a closed interval requires explicit correction"
        )
    if row.valid_at is not None and point <= _as_utc(row.valid_at):
        raise ClaimLifecycleConflict("closure must be after the incarnation start")
    return replace(row, invalid_at=point)


def _overlaps(left: ClaimRow, right: ClaimRow) -> bool:
    left_end, right_end = interval_end(left), interval_end(right)
    return not (
        left_end is not None
        and right.valid_at is not None
        and left_end <= _as_utc(right.valid_at)
        or right_end is not None
        and left.valid_at is not None
        and right_end <= _as_utc(left.valid_at)
    )


def _same_logical_claim(left: ClaimRow, right: ClaimRow) -> bool:
    return (
        claim_scope(left) == claim_scope(right)
        and left.object_key == right.object_key
        and (
            not left.claim_key
            or not right.claim_key
            or left.claim_key == right.claim_key
        )
    )


def validate_incarnation_history(
    rows: Iterable[ClaimRow], *, singleton_predicates: Iterable[str] = ()
) -> None:
    """Imports must preserve the same nonoverlapping logical history as writes."""
    groups: dict[tuple, list[ClaimRow]] = {}
    singleton = set(singleton_predicates)
    for row in rows:
        peers = groups.setdefault(claim_scope(row), [])
        if any(
            _same_logical_claim(row, peer) and _overlaps(row, peer) for peer in peers
        ):
            raise ClaimLifecycleConflict(
                "snapshot incarnation conflicts with target: overlapping logical history"
            )
        if row.predicate in singleton and any(
            row.object_key != peer.object_key
            and has_stable_source(row)
            and has_stable_source(peer)
            and (
                row.evidence_strength == "deterministic"
                or peer.evidence_strength == "deterministic"
            )
            and not row.retired
            and not peer.retired
            and _overlaps(row, peer)
            for peer in peers
        ):
            raise ClaimLifecycleConflict(
                "correction overlaps same-source singleton history"
            )
        peers.append(row)


def _singleton_conflicts(
    incoming: ClaimRow, rows: Iterable[ClaimRow]
) -> list[ClaimRow]:
    if incoming.evidence_strength != "deterministic" or not has_stable_source(incoming):
        return []
    return [
        row
        for row in rows
        if not row.retired
        and row.object_key != incoming.object_key
        and claim_scope(row) == claim_scope(incoming)
        and has_stable_source(row)
        and _overlaps(incoming, row)
    ]


def _assert_content_unchanged(incoming: ClaimRow, existing: ClaimRow) -> None:
    for name in ("fact", "description", "truth", "confidence"):
        value = getattr(incoming, name)
        if value is not None and value != getattr(existing, name):
            raise ClaimLifecycleConflict(
                f"changing claim {name} requires explicit correction"
            )
    for name in _TRUTH_FIELDS:
        if name in incoming.properties and incoming.properties[
            name
        ] != existing.properties.get(name):
            raise ClaimLifecycleConflict(
                f"changing claim {name} requires explicit correction"
            )


def resolve_claim_assertion(
    incoming: ClaimRow,
    candidates: Iterable[ClaimRow],
    *,
    singleton: bool,
    anchor: datetime,
    resolution: ClaimWriteResolution | None = None,
    metadata_record_id: str | None = None,
) -> tuple[ClaimRow, ClaimWriteResolution]:
    """Choose once, or validate and apply a previously frozen physical choice."""
    if incoming.retired:
        raise ClaimLifecycleConflict("assertion cannot retire a claim incarnation")
    rows = [row for row in candidates if row.pot_id == incoming.pot_id]
    if resolution is None:
        resolution = _choose(
            incoming,
            rows,
            singleton=singleton,
            anchor=_as_utc(anchor),
            metadata_record_id=metadata_record_id,
        )
    matches = [row for row in rows if row.record_id == resolution.record_id]
    if len(matches) != int(resolution.existing):
        raise ClaimLifecycleConflict(
            "frozen incarnation is missing, duplicated, or already exists"
        )
    existing = matches[0] if matches else None
    if existing is not None:
        if existing.retired or not _same_logical_claim(incoming, existing):
            raise ClaimLifecycleConflict(
                "frozen incarnation does not match assertion scope"
            )
        if (existing.valid_at, existing.invalid_at, existing.valid_until) != (
            resolution.valid_at,
            resolution.invalid_at,
            resolution.valid_until,
        ):
            raise ClaimLifecycleConflict("frozen incarnation lifecycle changed")
        _assert_content_unchanged(incoming, existing)
        if not resolution.metadata_review and any(
            getattr(incoming, name) is not None
            and getattr(incoming, name) != getattr(existing, name)
            for name in ("invalid_at", "valid_until")
        ):
            raise ClaimLifecycleConflict(
                "changing a claim endpoint requires explicit correction"
            )
    for identity in resolution.superseded_record_ids:
        predecessors = [row for row in rows if row.record_id == identity]
        if len(predecessors) != 1:
            raise ClaimLifecycleConflict("frozen predecessor is missing or ambiguous")
        previous = predecessors[0]
        if (
            previous.retired
            or claim_scope(previous) != claim_scope(incoming)
            or previous.object_key == incoming.object_key
            or not has_stable_source(previous)
            or not has_stable_source(incoming)
            or not singleton
            or previous.valid_at is not None
            and _as_utc(previous.valid_at) >= resolution.effective_at
            or interval_end(previous) is not None
            and interval_end(previous) <= resolution.effective_at
        ):
            raise ClaimLifecycleConflict(
                "frozen predecessor does not match singleton scope"
            )
    if resolution.metadata_review:
        if existing is None:
            raise ClaimLifecycleConflict(
                "metadata review requires an existing incarnation"
            )
        incoming = replace(
            existing,
            mutation_id=incoming.mutation_id,
            properties={
                **existing.properties,
                **{
                    key: value
                    for key, value in incoming.properties.items()
                    if key in REVIEW_FIELDS
                },
                "last_mutation_id": incoming.mutation_id,
            },
        )
    elif existing is not None:
        incoming = replace(
            incoming,
            properties={**existing.properties, **incoming.properties},
            claim_key=incoming.claim_key or existing.claim_key,
            fact=incoming.fact if incoming.fact is not None else existing.fact,
            confidence=(
                incoming.confidence
                if incoming.confidence is not None
                else existing.confidence
            ),
            description=incoming.description or existing.description,
            truth=incoming.truth or existing.truth,
            evidence_strength=(
                incoming.evidence_strength
                if incoming.truth is not None
                else existing.evidence_strength
            ),
        )
    resolved = replace(
        incoming,
        record_id=resolution.record_id,
        valid_at=resolution.valid_at,
        invalid_at=resolution.invalid_at,
        valid_until=resolution.valid_until,
    )
    if not resolution.metadata_review:
        if any(
            not row.retired
            and row.record_id != resolution.record_id
            and _same_logical_claim(resolved, row)
            and _overlaps(resolved, row)
            for row in rows
        ):
            raise ClaimLifecycleConflict(
                "frozen assertion overlaps another logical incarnation"
            )
        expected = (
            {row.record_id for row in _singleton_conflicts(resolved, rows)}
            if singleton
            else set()
        )
        if expected != set(resolution.superseded_record_ids):
            raise ClaimLifecycleConflict(
                "frozen singleton predecessor set is incomplete or changed"
            )
    return resolved, resolution


def _choose(
    incoming: ClaimRow,
    rows: list[ClaimRow],
    *,
    singleton: bool,
    anchor: datetime,
    metadata_record_id: str | None,
) -> ClaimWriteResolution:
    point = _as_utc(incoming.valid_at or anchor)
    incoming = replace(incoming, valid_at=point)
    if interval_end(incoming) is not None and interval_end(incoming) <= point:
        raise ClaimLifecycleConflict("assertion interval must have positive duration")
    matches = [
        row for row in rows if not row.retired and _same_logical_claim(incoming, row)
    ]
    if metadata_record_id:
        selected = [row for row in matches if row.record_id == metadata_record_id]
        if len(selected) != 1:
            raise ClaimLifecycleConflict(
                "metadata review incarnation is missing or ambiguous"
            )
        row = selected[0]
        return ClaimWriteResolution(
            row.record_id,
            True,
            point,
            row.valid_at,
            row.invalid_at,
            row.valid_until,
            metadata_review=True,
        )
    overlaps = [row for row in matches if _overlaps(incoming, row)]
    if len(overlaps) > 1:
        raise ClaimLifecycleConflict("ambiguous claim incarnation")
    existing = overlaps[0] if overlaps else None
    if existing is not None:
        end = interval_end(existing)
        if end is not None and end <= anchor:
            raise ClaimLifecycleConflict(
                "assertion overlaps closed history; explicit correction required"
            )
        if existing.valid_at is not None and point < _as_utc(existing.valid_at):
            raise ClaimLifecycleConflict(
                "assertion precedes existing incarnation; explicit correction required"
            )
        incoming = replace(
            incoming,
            valid_at=existing.valid_at,
            invalid_at=existing.invalid_at,
            valid_until=existing.valid_until,
        )
    predecessors = []
    if singleton:
        for row in _singleton_conflicts(incoming, rows):
            end = interval_end(row)
            if end is not None and end <= anchor:
                raise ClaimLifecycleConflict(
                    "singleton assertion overlaps closed history"
                )
            if row.valid_at is not None and _as_utc(row.valid_at) >= point:
                raise ClaimLifecycleConflict(
                    "singleton assertion overlaps a simultaneous or future claim"
                )
            predecessors.append(row.record_id)
    return ClaimWriteResolution(
        existing.record_id if existing else incoming.record_id,
        existing is not None,
        point,
        incoming.valid_at,
        incoming.invalid_at,
        incoming.valid_until,
        tuple(predecessors),
    )


def resolution_properties(resolution: ClaimWriteResolution) -> dict[str, Any]:
    """The sole lifecycle timestamp authority for physical persistence."""
    return {
        "record_id": resolution.record_id,
        "valid_at": iso(resolution.valid_at),
        "valid_from": iso(resolution.valid_at),
        "invalid_at": iso(resolution.invalid_at),
        "valid_until": iso(resolution.valid_until),
    }


def apply_claim_to_rows(
    rows: list[ClaimRow], row: ClaimRow, resolution: ClaimWriteResolution
) -> list[ClaimRow]:
    """Update a virtual preparation set or a staged memory store identically."""
    result = []
    for previous in rows:
        if previous.pot_id == row.pot_id and previous.record_id == resolution.record_id:
            continue
        if (
            previous.pot_id == row.pot_id
            and previous.record_id in resolution.superseded_record_ids
        ):
            previous = replace(
                previous,
                invalid_at=resolution.effective_at,
                properties={
                    **previous.properties,
                    "superseded_by_object": row.object_key,
                    "supersession_reason": "singleton_predicate",
                    "last_mutation_id": row.mutation_id,
                },
            )
        result.append(previous)
    return [*result, row]
