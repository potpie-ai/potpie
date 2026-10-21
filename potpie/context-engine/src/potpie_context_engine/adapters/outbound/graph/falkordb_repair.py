"""Select one FalkorDB entity repair delta and publish it with revision CAS."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import uuid

from redis.exceptions import ResponseError

from potpie_context_engine.core.errors import GraphMutationVersionConflict
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.journal_context import (
    JournalWriteContext,
    journal_write_context,
)
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.ports.graph.analytics import RepairReport

from .backends.claim_query_analytics import ClaimQueryAnalytics
from .document_key_repair import DOCUMENT_KEY_TARGET, wants_document_key_repair
from .entity_label_repair import wants_entity_label_repair
from .entity_repair import prepare_entity_repairs
from .entity_summary_repair import wants_entity_summary_repair

_SCAN = (
    "MATCH (e:Entity {group_id:$gid}) WHERE e.entity_key > $after "
    "RETURN e.entity_key AS key,labels(e) AS labels,properties(e) AS props "
    "ORDER BY e.entity_key LIMIT 100000"
)


class FalkorDBRepair(ClaimQueryAnalytics):
    def __init__(self, claim_query, graph_provider, mutation, definition):
        super().__init__(claim_query)
        self._graph_provider, self._mutation, self._definition = (
            graph_provider,
            mutation,
            definition,
        )

    def repair(self, pot_id, *, targets=()):
        repaired = self._repair_entities(pot_id, targets)
        report = (
            super().repair(pot_id, targets=(DOCUMENT_KEY_TARGET,))
            if wants_document_key_repair(targets)
            else RepairReport(pot_id=pot_id, targets=tuple(targets))
        )
        parts = [
            f"repaired {count} {target.replace('_', ' ')}"
            for target, count in repaired.items()
        ]
        parts.extend(finding.detail for finding in report.findings if finding.detail)
        return replace(
            report,
            targets=tuple(targets),
            repaired=repaired,
            detail=", ".join(parts) or report.detail,
        )

    def _repair_entities(self, pot_id, targets):
        if not (
            wants_entity_summary_repair(targets) or wants_entity_label_repair(targets)
        ):
            return {}
        self._mutation._require_native()
        for attempt in range(3):
            version = self._mutation.current_version(pot_id)
            plan, repaired = prepare_entity_repairs(
                self._entity_rows(pot_id),
                definition=self._definition,
                targets=targets,
            )
            if not plan.entity_upserts:
                return repaired
            mutation_id = uuid.uuid4().hex
            fingerprint = batch_fingerprint(plan)
            context = ProvenanceContext(
                mutation_id=mutation_id,
                prepared=PreparedMutation(
                    pot_id,
                    mutation_id,
                    fingerprint,
                    fingerprint,
                    version,
                    datetime.now(timezone.utc),
                ),
            )
            try:
                with journal_write_context(
                    JournalWriteContext(origin="repair", required_access="admin")
                ):
                    self._mutation.compare_and_apply(
                        plan,
                        expected_pot_id=pot_id,
                        expected_version=version,
                        provenance_context=context,
                    )
            except GraphMutationVersionConflict:
                if attempt == 2:
                    raise
            else:
                return repaired
        raise AssertionError("unreachable repair preparation state")

    def _entity_rows(self, pot_id):
        graph, after = self._graph_provider(), ""
        while True:
            try:
                rows = graph.ro_query(
                    _SCAN, params={"gid": pot_id, "after": after}
                ).result_set
            except ResponseError as exc:
                if "empty key" not in str(exc).lower():
                    raise
                return
            if not rows:
                return
            for key, labels, props in rows:
                yield {"key": key, "labels": labels, "props": props}
            after = rows[-1][0]
