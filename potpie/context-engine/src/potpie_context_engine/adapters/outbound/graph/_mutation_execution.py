"""Apply-once execution receipts for built-in graph mutation adapters."""

from __future__ import annotations

from potpie_context_engine.core.mutation_execution import PREPARATION_SCHEMA_VERSION

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import threading
from typing import Any

from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.mutation_execution import (
    ExecutionMetadata,
    MutationPersistenceError,
    batch_fingerprint,
    batch_has_work,
)
from potpie_context_engine.core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_engine.core.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
)

ExecutionKey = tuple[str, str]


class MutationExecutionReuseError(ValueError):
    """One reserved mutation id was presented with different batch content."""


class MutationExecutionAmbiguousError(RuntimeError):
    """An owned apply raised after execution began and cannot be retried safely."""


@dataclass(frozen=True, slots=True)
class CompletedMutationExecution:
    pot_id: str
    mutation_id: str
    batch_fingerprint: str
    result: MutationResult


@dataclass(slots=True)
class _InFlight:
    batch_fingerprint: str
    completed: threading.Event


def mutation_batch_fingerprint(plan: MutationBatch) -> str:
    return batch_fingerprint(plan)


def validate_prepared_execution(
    plan: MutationBatch,
    *,
    pot_id: str,
    mutation_id: str,
    provenance: ProvenanceContext | None,
) -> None:
    prepared = provenance.prepared if provenance else None
    closures = (*plan.edge_deletes, *plan.invalidations)
    if (prepared is None or prepared.schema_version < 3) and any(
        op.closures is not None for op in closures
    ):
        raise MutationExecutionReuseError(
            "frozen closures require a version-3 prepared mutation"
        )
    if (
        prepared is not None
        and prepared.schema_version >= 3
        and any(op.closures is None for op in closures)
    ):
        raise MutationExecutionReuseError(
            "prepared correction is missing its frozen closures"
        )
    if (prepared is None or prepared.schema_version < 2) and any(
        edge.resolution is not None for edge in plan.edge_upserts
    ):
        raise MutationExecutionReuseError(
            "frozen assertion resolutions require a version-2 prepared mutation"
        )
    if prepared is not None and (
        prepared.pot_id != pot_id
        or prepared.mutation_id != mutation_id
        or prepared.batch_fingerprint != batch_fingerprint(plan)
    ):
        raise MutationExecutionReuseError(
            "prepared execution identity or batch was changed"
        )
    if (
        prepared is not None
        and prepared.schema_version >= 2
        and any(edge.resolution is None for edge in plan.edge_upserts)
    ):
        raise MutationExecutionReuseError(
            "prepared assertion is missing its frozen incarnation resolution"
        )


def validate_new_preparation(provenance: ProvenanceContext | None) -> None:
    """Call only after authoritative lookup has proved no receipt exists."""
    if (
        provenance
        and provenance.prepared
        and provenance.prepared.schema_version < PREPARATION_SCHEMA_VERSION
    ):
        raise MutationExecutionReuseError(
            "migration_required: unexecuted legacy preparation needs a fresh identity"
        )


def complete_execution(
    result: MutationResult,
    plan: MutationBatch,
    *,
    pot_id: str,
    version_before: int,
    provenance: ProvenanceContext | None = None,
    record_ids: Iterable[str] = (),
    applied_at: datetime | None = None,
) -> MutationResult:
    """Called inside the backend transaction, before publishing its receipt."""
    validate_prepared_execution(
        plan, pot_id=pot_id, mutation_id=result.mutation_id, provenance=provenance
    )
    prepared = provenance.prepared if provenance else None
    if prepared is not None and prepared.expected_version != version_before:
        raise GraphMutationVersionConflict(
            expected=prepared.expected_version, current=version_before
        )
    if not result.ok:
        raise ValueError("cannot create a completion receipt for a failed execution")
    return replace(
        result,
        execution=ExecutionMetadata(
            pot_id=pot_id,
            applied_at=applied_at or datetime.now(timezone.utc),
            version_before=version_before,
            version_after=version_before + int(result.ok and batch_has_work(plan)),
            batch_fingerprint=batch_fingerprint(plan),
            request_fingerprint=prepared.request_fingerprint if prepared else None,
            affected_claim_record_ids=tuple(sorted(set(record_ids))),
        ),
    )


def validate_execution_replay(
    result: MutationResult, *, provenance: ProvenanceContext | None
) -> None:
    """Validate request binding before returning an already completed result.

    Legacy receipts have unknown request metadata; their original batch digest
    remains authoritative. A new prepared execution must retain its request hash.
    """
    prepared = provenance.prepared if provenance else None
    execution = result.execution
    if (
        prepared is not None
        and execution is not None
        and (
            execution.request_fingerprint != prepared.request_fingerprint
            or execution.batch_fingerprint != prepared.batch_fingerprint
            or execution.pot_id != prepared.pot_id
            or result.mutation_id != prepared.mutation_id
        )
    ):
        raise MutationExecutionReuseError(
            "execution identity was already bound to a different request"
        )


class MutationExecutionRegistry:
    """Thread-safe apply-once registry with same-id fingerprint validation."""

    def __init__(
        self,
        completed: Iterable[CompletedMutationExecution] = (),
    ) -> None:
        self._lock = threading.Lock()
        self._completed: dict[ExecutionKey, CompletedMutationExecution] = {
            (item.pot_id, item.mutation_id): deepcopy(item) for item in completed
        }
        self._in_flight: dict[ExecutionKey, _InFlight] = {}
        self._ambiguous: dict[ExecutionKey, tuple[str, str]] = {}

    def lookup(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ) -> MutationExecutionLookup:
        fingerprint = mutation_batch_fingerprint(plan)
        key = (expected_pot_id, mutation_id)
        with self._lock:
            in_flight = self._in_flight.get(key)
            if in_flight is not None:
                _require_same_fingerprint(key, fingerprint, in_flight.batch_fingerprint)
                return MutationExecutionLookup(
                    state=MutationExecutionState.in_flight.value,
                    mutation_id=mutation_id,
                    batch_fingerprint=fingerprint,
                )
            completed = self._completed.get(key)
            if completed is not None:
                _require_same_fingerprint(
                    key,
                    fingerprint,
                    completed.batch_fingerprint,
                )
                return MutationExecutionLookup(
                    state=MutationExecutionState.completed.value,
                    mutation_id=mutation_id,
                    batch_fingerprint=fingerprint,
                    result=deepcopy(completed.result),
                )
            ambiguous = self._ambiguous.get(key)
            if ambiguous is not None:
                stored_fingerprint, detail = ambiguous
                _require_same_fingerprint(key, fingerprint, stored_fingerprint)
                return MutationExecutionLookup(
                    state=MutationExecutionState.ambiguous.value,
                    mutation_id=mutation_id,
                    batch_fingerprint=fingerprint,
                    detail=detail,
                )
        return MutationExecutionLookup(
            state=MutationExecutionState.absent.value,
            mutation_id=mutation_id,
            batch_fingerprint=fingerprint,
        )

    def execute(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
        operation: Callable[[], MutationResult],
        on_completed: Callable[[], None] | None = None,
    ) -> MutationResult:
        fingerprint = mutation_batch_fingerprint(plan)
        key = (expected_pot_id, mutation_id)
        while True:
            owner, event, cached = self._claim(key, fingerprint)
            if cached is not None:
                return cached
            if not owner:
                event.wait()
                continue
            try:
                result = operation()
            except BaseException as exc:
                self._fail(key, fingerprint, exc)
                raise
            self._finish(key, fingerprint, result)
            try:
                if result.ok and on_completed is not None:
                    on_completed()
            finally:
                self._signal(key)
            return deepcopy(result)

    async def execute_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
        operation: Callable[[], Awaitable[MutationResult]],
    ) -> MutationResult:
        fingerprint = mutation_batch_fingerprint(plan)
        key = (expected_pot_id, mutation_id)
        while True:
            owner, event, cached = self._claim(key, fingerprint)
            if cached is not None:
                return cached
            if not owner:
                await asyncio.to_thread(event.wait)
                continue
            try:
                result = await operation()
            except BaseException as exc:
                self._fail(key, fingerprint, exc)
                raise
            self._finish(key, fingerprint, result)
            self._signal(key)
            return deepcopy(result)

    def completed(self) -> tuple[CompletedMutationExecution, ...]:
        with self._lock:
            return tuple(deepcopy(item) for item in self._completed.values())

    def replace_completed(
        self,
        completed: Iterable[CompletedMutationExecution],
    ) -> None:
        """Replace the durable snapshot when no local execution is active."""

        replacement = {
            (item.pot_id, item.mutation_id): deepcopy(item) for item in completed
        }
        with self._lock:
            if self._in_flight:
                raise RuntimeError(
                    "cannot replace mutation receipts while an execution is in flight"
                )
            self._completed = replacement
            self._ambiguous.clear()

    def discard_pot(self, pot_id: str) -> None:
        with self._lock:
            for key in [key for key in self._completed if key[0] == pot_id]:
                self._completed.pop(key, None)
            for key in [key for key in self._ambiguous if key[0] == pot_id]:
                self._ambiguous.pop(key, None)

    def _claim(
        self,
        key: ExecutionKey,
        fingerprint: str,
    ) -> tuple[bool, threading.Event, MutationResult | None]:
        with self._lock:
            in_flight = self._in_flight.get(key)
            if in_flight is not None:
                _require_same_fingerprint(key, fingerprint, in_flight.batch_fingerprint)
                return False, in_flight.completed, None
            completed = self._completed.get(key)
            if completed is not None:
                _require_same_fingerprint(
                    key,
                    fingerprint,
                    completed.batch_fingerprint,
                )
                return False, threading.Event(), deepcopy(completed.result)
            ambiguous = self._ambiguous.get(key)
            if ambiguous is not None:
                stored_fingerprint, detail = ambiguous
                _require_same_fingerprint(key, fingerprint, stored_fingerprint)
                raise MutationExecutionAmbiguousError(detail)
            event = threading.Event()
            self._in_flight[key] = _InFlight(
                batch_fingerprint=fingerprint,
                completed=event,
            )
            return True, event, None

    def _finish(
        self,
        key: ExecutionKey,
        fingerprint: str,
        result: MutationResult,
    ) -> None:
        with self._lock:
            if result.ok:
                self._completed[key] = CompletedMutationExecution(
                    pot_id=key[0],
                    mutation_id=key[1],
                    batch_fingerprint=fingerprint,
                    result=deepcopy(result),
                )
            else:
                self._ambiguous[key] = (
                    fingerprint,
                    result.error or "mutation apply returned a non-success result",
                )

    def _fail(
        self,
        key: ExecutionKey,
        fingerprint: str,
        exc: BaseException,
    ) -> None:
        with self._lock:
            self._ambiguous[key] = (
                fingerprint,
                f"{type(exc).__name__}: {exc}",
            )
        self._signal(key)

    def _signal(self, key: ExecutionKey) -> None:
        with self._lock:
            in_flight = self._in_flight.pop(key, None)
        if in_flight is not None:
            in_flight.completed.set()


def execution_receipts_to_json(
    receipts: Iterable[CompletedMutationExecution],
) -> list[dict[str, Any]]:
    return [
        {
            "pot_id": item.pot_id,
            "mutation_id": item.mutation_id,
            "batch_fingerprint": item.batch_fingerprint,
            "result": _mutation_result_to_dict(item.result),
        }
        for item in receipts
    ]


def execution_receipts_from_json(
    raw: object,
) -> tuple[CompletedMutationExecution, ...]:
    if not isinstance(raw, list):
        raise MutationPersistenceError("execution receipts must be a list")
    receipts: list[CompletedMutationExecution] = []
    seen: set[ExecutionKey] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise MutationPersistenceError("malformed execution receipt")
        pot_id, mutation_id, fingerprint = (
            item.get(name) for name in ("pot_id", "mutation_id", "batch_fingerprint")
        )
        if any(
            not isinstance(value, str) or not value
            for value in (pot_id, mutation_id, fingerprint)
        ):
            raise MutationPersistenceError("invalid execution receipt identity")
        result = _mutation_result_from_dict(item.get("result"))
        if (
            not pot_id
            or not mutation_id
            or not fingerprint
            or result is None
            or result.mutation_id != mutation_id
        ):
            raise MutationPersistenceError("incomplete execution receipt identity")
        if (pot_id, mutation_id) in seen:
            raise MutationPersistenceError("duplicate execution receipt identity")
        if result.execution is not None and (
            result.execution.pot_id != pot_id
            or result.execution.batch_fingerprint != fingerprint
        ):
            raise MutationPersistenceError(
                "execution receipt metadata differs from its identity"
            )
        seen.add((pot_id, mutation_id))
        receipts.append(
            CompletedMutationExecution(pot_id, mutation_id, fingerprint, result)
        )
    return tuple(receipts)


def _require_same_fingerprint(
    key: ExecutionKey,
    received: str,
    stored: str,
) -> None:
    if received != stored:
        raise MutationExecutionReuseError(
            f"mutation_id {key[1]!r} for pot {key[0]!r} was already used "
            "with a different mutation batch"
        )


def _mutation_result_to_dict(result: MutationResult) -> dict[str, Any]:
    summary = result.mutation_summary
    return {
        **(
            {"administrative_summary": dict(result.administrative_summary)}
            if result.administrative_summary
            else {}
        ),
        **(
            {"execution": result.execution.to_dict()}
            if result.execution is not None
            else {}
        ),
        "ok": result.ok,
        "mutation_id": result.mutation_id,
        "mutation_summary": {
            "entity_upserts_applied": summary.entity_upserts_applied,
            "edge_upserts_applied": summary.edge_upserts_applied,
            "edge_deletes_applied": summary.edge_deletes_applied,
            "invalidations_applied": summary.invalidations_applied,
            "stamp_counts": dict(summary.stamp_counts),
        },
        "error": result.error,
        "reconciliation_errors": [dict(item) for item in result.reconciliation_errors],
        "downgrades": [dict(item) for item in result.downgrades],
    }


def _mutation_result_from_dict(raw: object) -> MutationResult | None:
    if not isinstance(raw, Mapping):
        return None
    summary_raw = raw.get("mutation_summary")
    if not isinstance(summary_raw, Mapping):
        return None
    if (
        raw.get("ok") is not True
        or not isinstance(raw.get("mutation_id"), str)
        or not raw["mutation_id"]
    ):
        raise MutationPersistenceError(
            "completion receipt must describe a successful identified execution"
        )
    counts = {}
    for name in (
        "entity_upserts_applied",
        "edge_upserts_applied",
        "edge_deletes_applied",
        "invalidations_applied",
    ):
        value = summary_raw.get(name, 0)
        if type(value) is not int or value < 0:
            raise MutationPersistenceError("invalid execution mutation count")
        counts[name] = value
    stamps = summary_raw.get("stamp_counts", {})
    if not isinstance(stamps, Mapping) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in stamps.items()
    ):
        raise MutationPersistenceError("invalid execution stamp counts")
    errors, downgrades = raw.get("reconciliation_errors", []), raw.get("downgrades", [])
    if any(
        not isinstance(items, (list, tuple))
        or any(not isinstance(item, Mapping) for item in items)
        for items in (errors, downgrades)
    ):
        raise MutationPersistenceError("invalid execution diagnostic metadata")
    if raw.get("error") is not None and not isinstance(raw["error"], str):
        raise MutationPersistenceError("invalid execution error")
    administrative = raw.get("administrative_summary", {})
    if not isinstance(administrative, Mapping) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in administrative.items()
    ):
        raise MutationPersistenceError("invalid administrative execution counts")
    return MutationResult(
        execution=(
            ExecutionMetadata.from_dict(raw["execution"])
            if "execution" in raw
            else None
        ),
        ok=True,
        administrative_summary=dict(administrative),
        mutation_id=raw["mutation_id"],
        mutation_summary=MutationSummary(**counts, stamp_counts=dict(stamps)),
        error=raw.get("error"),
        reconciliation_errors=[dict(item) for item in errors],
        downgrades=[dict(item) for item in downgrades],
    )


__all__ = [
    "CompletedMutationExecution",
    "MutationExecutionAmbiguousError",
    "MutationExecutionRegistry",
    "MutationExecutionReuseError",
    "execution_receipts_from_json",
    "execution_receipts_to_json",
    "mutation_batch_fingerprint",
]
