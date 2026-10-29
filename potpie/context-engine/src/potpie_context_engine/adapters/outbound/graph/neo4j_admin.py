"""Prepare entity repair batches for Neo4j's canonical atomic mutation path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
import uuid

from potpie_context_engine.core.definition import GraphDefinition
from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.ports.graph.analytics import RepairReport
from potpie_context_engine.core.reconciliation_validation import (
    validate_reconciliation_plan,
)

from .backends.claim_query_analytics import ClaimQueryAnalytics
from .document_key_repair import DOCUMENT_KEY_TARGET, wants_document_key_repair
from .entity_label_repair import wants_entity_label_repair
from .entity_summary_repair import wants_entity_summary_repair
from .entity_repair import prepare_entity_repairs

_ENTITY_SCAN = (
    "MATCH (e:Entity {group_id:$pot}) WHERE e.entity_key > $after "
    "RETURN e.entity_key AS key, labels(e) AS labels, properties(e) AS props "
    "ORDER BY e.entity_key LIMIT 100000"
)


class Neo4jAnalytics(ClaimQueryAnalytics):
    """Combine requested entity repairs into one checked, receipt-bearing batch."""

    def __init__(self, claim_query: Any, mutation: Any, definition: GraphDefinition):
        super().__init__(claim_query)
        self._mutation = mutation
        self._definition = definition

    def repair(self, pot_id: str, *, targets: Sequence[str] = ()) -> RepairReport:
        from .backends.neo4j_backend import _run_sync

        repaired = _run_sync(self._repair_async(pot_id, targets))
        findings = ()
        if wants_document_key_repair(targets):
            findings = super().repair(pot_id, targets=(DOCUMENT_KEY_TARGET,)).findings
        parts = [
            f"repaired {count} {target.replace('_', ' ')}"
            for target, count in repaired.items()
        ]
        parts.extend(finding.detail for finding in findings if finding.detail)
        return RepairReport(
            pot_id=pot_id,
            targets=tuple(targets),
            repaired=repaired,
            findings=findings,
            detail=", ".join(parts)
            if parts
            else "analytics are computed on read from claim_query; nothing to rebuild",
        )

    async def _repair_async(
        self, pot_id: str, targets: Sequence[str]
    ) -> dict[str, int]:
        summaries = wants_entity_summary_repair(targets)
        labels = wants_entity_label_repair(targets)
        if not summaries and not labels:
            return {}
        if not self._mutation.atomic_mutations_supported:
            raise NotImplementedError(
                "Neo4j entity repair requires native atomic mutations"
            )
        # A revision read before selection plus CAS at execution rejects any
        # concurrent canonical write during preparation. Conflicts publish no
        # effects or receipts; only those unexecuted preparations are reselected.
        for attempt in range(3):
            version = await self._mutation.current_version_async(pot_id)
            rows = await self._entity_rows(pot_id)
            plan, repaired = prepare_entity_repairs(
                rows, definition=self._definition, targets=targets
            )
            if not plan.entity_upserts:
                return repaired
            validate_reconciliation_plan(plan, pot_id, definition=self._definition)
            mutation_id = uuid.uuid4().hex
            fingerprint = batch_fingerprint(plan)
            provenance = ProvenanceContext(
                mutation_id=mutation_id,
                prepared=PreparedMutation(
                    pot_id=pot_id,
                    mutation_id=mutation_id,
                    request_fingerprint=fingerprint,
                    batch_fingerprint=fingerprint,
                    expected_version=version,
                    prepared_at=datetime.now(timezone.utc),
                ),
            )
            try:
                await self._mutation.compare_and_apply_async(
                    plan,
                    expected_pot_id=pot_id,
                    expected_version=version,
                    provenance_context=provenance,
                )
            except GraphMutationVersionConflict:
                if attempt == 2:
                    raise
            else:
                return repaired
        raise AssertionError("unreachable repair preparation state")

    async def _entity_rows(self, pot_id: str) -> list[Mapping[str, Any]]:
        driver = self._mutation._get_writer()._new_driver()
        if driver is None:
            raise RuntimeError("neo4j_unavailable")
        try:
            async with driver.session() as session:

                async def read(transaction):
                    rows, after = [], ""
                    while True:
                        result = await transaction.run(
                            _ENTITY_SCAN, pot=pot_id, after=after
                        )
                        page = await result.data()
                        await result.consume()
                        if not page:
                            return rows
                        rows.extend(page)
                        after = page[-1]["key"]

                return await session.execute_read(read)
        finally:
            await driver.close()
