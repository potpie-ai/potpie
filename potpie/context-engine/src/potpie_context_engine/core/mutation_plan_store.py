"""Process-local plan reservation for volatile runtimes and standalone writes.

Writers coordinating a pot must share this instance. Persistence and cross-process
coordination require a store implementing the same atomic reservation port.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import threading
from .graph_plans import GraphMutationPlanRecord
from .mutation_execution import request_fingerprint as _request_fingerprint

@dataclass(slots=True)
class InMemoryGraphPlanStore:
    _records: dict[tuple[str, str], GraphMutationPlanRecord] = field(
        default_factory=dict
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def save(self, record: GraphMutationPlanRecord) -> None:
        with self._lock:
            self._records[(record.pot_id, record.plan_id)] = record

    def reserve_idempotency(
        self,
        *,
        record: GraphMutationPlanRecord,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> tuple[GraphMutationPlanRecord, bool]:
        with self._lock:
            for (pot_id, _), existing in self._records.items():
                if pot_id != record.pot_id:
                    continue
                key = str(
                    existing.original_payload.get("idempotency_key") or ""
                ).strip()
                if key != idempotency_key or not existing.reserves_idempotency:
                    continue
                if (
                    _request_fingerprint(existing.original_payload)
                    != request_fingerprint
                ):
                    raise ValueError(
                        f"idempotency_key {idempotency_key!r} is already bound to "
                        f"plan {existing.plan_id!r} with different content"
                    )
                return existing, False
            self._records[(record.pot_id, record.plan_id)] = record
            return record, True

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        with self._lock:
            return self._records.get((pot_id, plan_id))

    def compare_and_set(
        self,
        *,
        expected: GraphMutationPlanRecord,
        replacement: GraphMutationPlanRecord,
    ) -> bool:
        key = (expected.pot_id, expected.plan_id)
        if key != (replacement.pot_id, replacement.plan_id):
            raise ValueError("plan compare-and-set cannot change plan identity")
        with self._lock:
            if self._records.get(key) != expected:
                return False
            self._records[key] = replacement
            return True

    def list(
        self,
        *,
        pot_id: str,
        plan_id: str | None = None,
        mutation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphMutationPlanRecord, ...]:
        with self._lock:
            records = [
                record
                for (record_pot, _), record in self._records.items()
                if record_pot == pot_id
                and (plan_id is None or record.plan_id == plan_id)
                and (mutation_id is None or record.mutation_id == mutation_id)
                and (since is None or record.created_at >= since)
                and (until is None or record.created_at <= until)
            ]
        records.sort(key=lambda record: record.created_at, reverse=True)
        return tuple(records[:limit] if limit is not None else records)

    async def save_async(self, record: GraphMutationPlanRecord) -> None:
        self.save(record)

    async def reserve_idempotency_async(self, **kwargs):
        return self.reserve_idempotency(**kwargs)

    async def get_async(
        self, *, pot_id: str, plan_id: str
    ) -> GraphMutationPlanRecord | None:
        return self.get(pot_id=pot_id, plan_id=plan_id)

    async def compare_and_set_async(
        self,
        *,
        expected: GraphMutationPlanRecord,
        replacement: GraphMutationPlanRecord,
    ) -> bool:
        return self.compare_and_set(expected=expected, replacement=replacement)

    async def list_async(self, **kwargs):
        return self.list(**kwargs)


