"""Apply-once execution receipts for built-in graph mutation adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import threading
from typing import Any

from potpie_context_core.graph_plans import mutation_batch_to_dict
from potpie_context_core.ports.graph.mutation import (
    MutationExecutionLookup,
    MutationExecutionState,
)
from potpie_context_core.reconciliation import (
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
    encoded = json.dumps(
        mutation_batch_to_dict(plan),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=20).hexdigest()


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
        return ()
    receipts: list[CompletedMutationExecution] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        pot_id = str(item.get("pot_id") or "")
        mutation_id = str(item.get("mutation_id") or "")
        fingerprint = str(item.get("batch_fingerprint") or "")
        result = _mutation_result_from_dict(item.get("result"))
        if pot_id and mutation_id and fingerprint and result is not None:
            receipts.append(
                CompletedMutationExecution(
                    pot_id=pot_id,
                    mutation_id=mutation_id,
                    batch_fingerprint=fingerprint,
                    result=result,
                )
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
    return MutationResult(
        ok=bool(raw.get("ok")),
        mutation_id=str(raw.get("mutation_id") or ""),
        mutation_summary=MutationSummary(
            entity_upserts_applied=int(summary_raw.get("entity_upserts_applied") or 0),
            edge_upserts_applied=int(summary_raw.get("edge_upserts_applied") or 0),
            edge_deletes_applied=int(summary_raw.get("edge_deletes_applied") or 0),
            invalidations_applied=int(summary_raw.get("invalidations_applied") or 0),
            stamp_counts={
                str(key): int(value)
                for key, value in dict(summary_raw.get("stamp_counts") or {}).items()
            },
        ),
        error=str(raw.get("error") or "") or None,
        reconciliation_errors=[
            dict(item)
            for item in raw.get("reconciliation_errors") or ()
            if isinstance(item, Mapping)
        ],
        downgrades=[
            dict(item)
            for item in raw.get("downgrades") or ()
            if isinstance(item, Mapping)
        ],
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
