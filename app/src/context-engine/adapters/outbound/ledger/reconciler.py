"""``DeterministicEventReconciler`` — deterministic events → claims.

THIS IS THE PARKED-DECISION SEAM. How normalized events become graph claims —
deterministic extraction, an LLM reconciliation agent, or a hybrid — is an
explicitly undecided low-level choice.

    TODO(decide): LLM-vs-deterministic reconciliation strategy.
    The existing pydantic reconciliation agent
    (``adapters/outbound/reconciliation/``) and the deterministic extractors
    (``domain/deterministic_extractors.py``) are both candidate bodies for this
    port. This impl uses a trivial deterministic lowering so the
    ``ledger pull --apply`` → reconcile → claims path runs end to end; swapping
    in the real strategy must not change this port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from domain.context_events import EventRef
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.ports.graph.mutation import GraphMutationPort
from domain.ports.ledger.client import LedgerEvent
from domain.ports.ledger.reconciler import ReconcileResult
from domain.reconciliation import ReconciliationPlan


@dataclass(slots=True)
class DeterministicEventReconciler:
    """Lower each event to one generic claim through the backend mutation port."""

    mutation: GraphMutationPort

    def reconcile(
        self, *, pot_id: str, events: Sequence[LedgerEvent]
    ) -> ReconcileResult:
        written = 0
        for event in events:
            plan = self._lower(pot_id, event)
            result = self.mutation.apply(plan, expected_pot_id=pot_id)
            written += result.mutation_summary.edge_upserts_applied
        return ReconcileResult(
            pot_id=pot_id,
            events_in=len(events),
            claims_written=written,
            detail="deterministic reconciler (see TODO(decide))",
        )

    def _lower(self, pot_id: str, event: LedgerEvent) -> ReconciliationPlan:
        # TODO(decide)/(stage-N): real extraction maps event.kind + payload to
        # ontology entities/predicates. This impl emits one provenance-stamped edge.
        event_key = f"event:{event.provider}:{event.event_id}"
        subject = str(event.payload.get("subject") or event_key)
        target = str(event.payload.get("object") or f"source:{event.source_id}")
        fact = str(event.payload.get("fact") or f"{event.provider} {event.kind}")
        return ReconciliationPlan(
            event_ref=EventRef(
                event_id=event.event_id, source_system=event.provider, pot_id=pot_id
            ),
            summary=fact,
            entity_upserts=[EntityUpsert(entity_key=subject, labels=("Event",))],
            edge_upserts=[
                EdgeUpsert(
                    edge_type="RELATES_TO",
                    from_entity_key=subject,
                    to_entity_key=target,
                    properties={
                        "fact": fact,
                        "source_system": event.provider,
                        "source_ref": event.event_id,
                        "valid_at": event.occurred_at,
                    },
                )
            ],
        )


__all__ = ["DeterministicEventReconciler"]
