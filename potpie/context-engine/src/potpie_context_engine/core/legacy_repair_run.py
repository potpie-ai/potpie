"""Immutable manifest binding a reviewed repair to ordinary prepared children."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, TYPE_CHECKING

from .legacy_repair_policy import LegacyFinding
from .mutation_execution import MutationPersistenceError, request_fingerprint

if TYPE_CHECKING:
    from .graph_plans import GraphMutationPlanRecord


@dataclass(frozen=True, slots=True)
class RepairChild:
    plan_id: str
    mutation_id: str
    batch_fingerprint: str
    expected_version: int


@dataclass(frozen=True, slots=True)
class RepairRun:
    repair_id: str
    pot_id: str
    request_fingerprint: str
    base_revision: int
    prepared_at: datetime
    expires_at: datetime
    children: tuple[RepairChild, ...]
    findings: tuple[LegacyFinding, ...]
    scanned_records: int
    repairable_records: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or any(
                not isinstance(value, str) or not value.strip()
                for value in (self.repair_id, self.pot_id, self.request_fingerprint)
            )
        ):
            raise MutationPersistenceError("invalid repair manifest identity/schema")
        if (
            any(
                type(value) is not int or value < 0
                for value in (
                    self.base_revision,
                    self.scanned_records,
                    self.repairable_records,
                )
            )
            or self.repairable_records > self.scanned_records
        ):
            raise MutationPersistenceError("invalid repair manifest counts/revision")
        if (
            any(
                not isinstance(value, datetime) or value.utcoffset() is None
                for value in (self.prepared_at, self.expires_at)
            )
            or self.expires_at <= self.prepared_at
        ):
            raise MutationPersistenceError("invalid repair manifest lifetime")
        if self.blocked and self.children:
            raise MutationPersistenceError(
                "blocking repair cannot have executable children"
            )
        identities = set()
        for index, child in enumerate(self.children):
            if (
                any(
                    not isinstance(value, str) or not value
                    for value in (
                        child.plan_id,
                        child.mutation_id,
                        child.batch_fingerprint,
                    )
                )
                or type(child.expected_version) is not int
                or child.expected_version != self.base_revision + index
            ):
                raise MutationPersistenceError("invalid repair child binding")
            for identity in (child.plan_id, child.mutation_id):
                if identity in identities:
                    raise MutationPersistenceError("duplicate repair child identity")
                identities.add(identity)

    @property
    def blocked(self) -> bool:
        return any(finding.blocking for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "prepared_at": self.prepared_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "children": [asdict(child) for child in self.children],
            "findings": [asdict(finding) for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RepairRun:
        try:
            data = dict(raw)
            for name in ("prepared_at", "expires_at"):
                data[name] = datetime.fromisoformat(data[name])
            data["children"] = tuple(RepairChild(**item) for item in data["children"])
            data["findings"] = tuple(LegacyFinding(**item) for item in data["findings"])
            return cls(**data)
        except (TypeError, ValueError, KeyError) as exc:
            raise MutationPersistenceError("malformed repair manifest") from exc

    def validate_children(self, records: Sequence[GraphMutationPlanRecord]) -> None:
        if len(records) != len(self.children):
            raise MutationPersistenceError("repair manifest child is missing")
        targets = set()
        total = 0
        for index, (child, record) in enumerate(zip(self.children, records)):
            if record is None:
                raise MutationPersistenceError("repair manifest child is missing")
            record.validate_prepared()
            if (
                record.prepared is None
                or record.lowered_batch is None
                or (
                    record.pot_id,
                    record.plan_id,
                    record.mutation_id,
                    record.prepared.batch_fingerprint,
                    record.prepared.expected_version,
                    record.created_at,
                    record.expires_at,
                    record.original_payload,
                )
                != (
                    self.pot_id,
                    child.plan_id,
                    child.mutation_id,
                    child.batch_fingerprint,
                    child.expected_version,
                    self.prepared_at,
                    self.expires_at,
                    child_request(self.repair_id, index),
                )
                or not record.lowered_batch.record_repairs
            ):
                raise MutationPersistenceError(
                    "repair child differs from frozen manifest"
                )
            for effect in record.lowered_batch.record_repairs:
                if effect.before.fingerprint in targets:
                    raise MutationPersistenceError(
                        "duplicate repair target across children"
                    )
                targets.add(effect.before.fingerprint)
                total += 1
        if not self.blocked and total != self.repairable_records:
            raise MutationPersistenceError("repair manifest effect count differs")


def child_request(repair_id: str, index: int) -> dict[str, Any]:
    return {
        "kind": "legacy_repair",
        "policy_version": 1,
        "repair_id": repair_id,
        "index": index,
        "idempotency_key": f"legacy-repair:{repair_id}:{index}",
    }


def repair_request_fingerprint(*, chunk_size: int) -> str:
    return request_fingerprint(
        {"kind": "legacy_repair", "policy_version": 1, "chunk_size": chunk_size}
    )


def validate_repair_reservation(
    run: RepairRun,
    records: Sequence[GraphMutationPlanRecord],
    existing: RepairRun | None,
) -> bool:
    """Return whether to insert; a prior identity permanently binds its policy."""
    if existing is not None:
        if existing.request_fingerprint != run.request_fingerprint:
            raise MutationPersistenceError(
                "repair identity is bound to a different request"
            )
        return False
    run.validate_children(records)
    return True
