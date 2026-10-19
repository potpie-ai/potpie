"""Versioned identities and immutable metadata for prepared graph execution.

Batch fingerprint version 1 deliberately retains the historical encoding.
Normalization happens on a copy after fingerprint validation, never on an old
receipt or on the caller's persisted prepared batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .reconciliation import MutationBatch, MutationResult


FINGERPRINT_VERSION = 1
PREPARATION_SCHEMA_VERSION = 4
EXECUTION_SCHEMA_VERSION = 2


class MutationPersistenceError(ValueError):
    """Persisted identity or execution metadata cannot be trusted."""


def request_fingerprint(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("expected_subgraph_versions", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=20).hexdigest()


def batch_fingerprint(batch: MutationBatch) -> str:
    from .graph_plans import mutation_batch_to_dict

    encoded = json.dumps(
        mutation_batch_to_dict(batch),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=20).hexdigest()


def batch_has_work(batch: MutationBatch) -> bool:
    return bool(
        batch.reset_pot
        or (
            batch.snapshot_import is not None
            and (
                batch.snapshot_import.get("entities")
                or batch.snapshot_import.get("claims")
            )
        )
        or batch.entity_upserts
        or batch.edge_upserts
        or batch.edge_deletes
        or batch.invalidations
        or batch.record_repairs
    )


def execution_matches_batch(
    result: MutationResult | None,
    batch: MutationBatch,
    *,
    pot_id: str,
    mutation_id: str,
    prepared: PreparedMutation | None = None,
    allow_legacy: bool = False,
) -> bool:
    """Validate execution binding at workflow boundaries, including recovery."""
    if result is None or result.ok is not True or result.mutation_id != mutation_id:
        return False
    execution = result.execution
    if execution is None:
        return allow_legacy and prepared is None
    if (
        execution.pot_id != pot_id
        or execution.batch_fingerprint != batch_fingerprint(batch)
        or execution.version_after
        != execution.version_before + int(batch_has_work(batch))
    ):
        return False
    return prepared is None or (
        execution.request_fingerprint == prepared.request_fingerprint
        and execution.version_before == prepared.expected_version
    )


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise MutationPersistenceError("execution timestamp must be an ISO string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MutationPersistenceError("invalid execution timestamp") from exc
    if parsed.tzinfo is None:
        raise MutationPersistenceError("execution timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _identities(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise MutationPersistenceError(
            "claim identities must be an array of nonempty strings"
        )
    return tuple(values)


def _identity(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise MutationPersistenceError("mutation identity is incomplete")


def _aware(value: datetime) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise MutationPersistenceError("mutation timestamp must include a timezone")


@dataclass(frozen=True, slots=True)
class PreparedMutation:
    pot_id: str
    mutation_id: str
    request_fingerprint: str
    batch_fingerprint: str
    expected_version: int
    prepared_at: datetime
    target_claim_record_ids: tuple[str, ...] = ()
    schema_version: int = PREPARATION_SCHEMA_VERSION
    fingerprint_version: int = FINGERPRINT_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or type(self.fingerprint_version) is not int
            or self.schema_version not in {1, 2, 3, PREPARATION_SCHEMA_VERSION}
            or self.fingerprint_version != FINGERPRINT_VERSION
        ):
            raise MutationPersistenceError("unsupported prepared mutation version")
        for value in (
            self.pot_id,
            self.mutation_id,
            self.request_fingerprint,
            self.batch_fingerprint,
        ):
            _identity(value)
        if type(self.expected_version) is not int or self.expected_version < 0:
            raise MutationPersistenceError("invalid prepared revision")
        _aware(self.prepared_at)
        object.__setattr__(
            self, "target_claim_record_ids", _identities(self.target_claim_record_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "prepared_at": self.prepared_at.isoformat(),
            "target_claim_record_ids": list(self.target_claim_record_ids),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreparedMutation:
        try:
            return cls(
                pot_id=raw["pot_id"],
                mutation_id=raw["mutation_id"],
                request_fingerprint=raw["request_fingerprint"],
                batch_fingerprint=raw["batch_fingerprint"],
                expected_version=raw["expected_version"],
                prepared_at=_utc(raw["prepared_at"]),
                target_claim_record_ids=_identities(
                    raw.get("target_claim_record_ids", ())
                ),
                schema_version=raw["schema_version"],
                fingerprint_version=raw["fingerprint_version"],
            )
        except (KeyError, TypeError) as exc:
            raise MutationPersistenceError("malformed prepared mutation") from exc


@dataclass(frozen=True, slots=True)
class ExecutionMetadata:
    pot_id: str
    applied_at: datetime
    version_before: int
    version_after: int
    batch_fingerprint: str
    request_fingerprint: str | None = None
    affected_claim_record_ids: tuple[str, ...] = ()
    schema_version: int = EXECUTION_SCHEMA_VERSION
    fingerprint_version: int = FINGERPRINT_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or type(self.fingerprint_version) is not int
            or self.schema_version != EXECUTION_SCHEMA_VERSION
            or self.fingerprint_version != FINGERPRINT_VERSION
        ):
            raise MutationPersistenceError("unsupported execution receipt version")
        _identity(self.pot_id)
        _identity(self.batch_fingerprint)
        if self.request_fingerprint is not None:
            _identity(self.request_fingerprint)
        before, after = self.version_before, self.version_after
        if (
            type(before) is not int
            or type(after) is not int
            or before < 0
            or after not in (before, before + 1)
        ):
            raise MutationPersistenceError("invalid execution revision transition")
        _aware(self.applied_at)
        object.__setattr__(
            self,
            "affected_claim_record_ids",
            _identities(self.affected_claim_record_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "applied_at": self.applied_at.isoformat(),
            "affected_claim_record_ids": list(self.affected_claim_record_ids),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExecutionMetadata:
        try:
            return cls(
                pot_id=raw["pot_id"],
                applied_at=_utc(raw["applied_at"]),
                version_before=raw["version_before"],
                version_after=raw["version_after"],
                batch_fingerprint=raw["batch_fingerprint"],
                request_fingerprint=raw.get("request_fingerprint"),
                affected_claim_record_ids=_identities(
                    raw.get("affected_claim_record_ids", ())
                ),
                schema_version=raw["schema_version"],
                fingerprint_version=raw["fingerprint_version"],
            )
        except (KeyError, TypeError) as exc:
            raise MutationPersistenceError("malformed execution receipt") from exc


@dataclass(frozen=True, slots=True)
class MutationCapabilities:
    atomic_writes: bool = False
    revision_check: bool = False
    execution_lookup: bool = False
    durable_receipts: bool = False
