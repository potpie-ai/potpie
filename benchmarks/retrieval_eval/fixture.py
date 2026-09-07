"""Build the deterministic 148-claim Ledgerly retrieval fixture."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.intelligence.local_embedder import (
    HashingEmbedder,
)
from potpie_context_engine.adapters.outbound.resources.index import build_resource_index
from potpie_context_engine.application.services.agent_context import AgentContextService
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.read_orchestrator import (
    ReadOrchestrator,
)
from potpie_context_engine.application.services.resource_facade import ResourceFacade
from potpie_context_engine.domain.ranking import (
    Candidate,
    RankedItem,
    RankingService,
    TaskContext,
)
from potpie_context_engine.testing import InMemoryResourceStore
from potpie_context_core.ports.claim_query import ClaimRow
from potpie_context_core.ports.resource_index import ResourceIndexPort
from potpie_context_core.semantic_mutations import SemanticMutationRequest

DEEP_QUERY_POT_ID = "golden:deep-query"
FIXTURE_AS_OF = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
FIXTURE_WRITTEN_AT = "2026-09-03T10:00:00Z"

_ROOT = Path(__file__).parent / "fixtures" / "deep_query"
_PLANS = _ROOT / "plans"
_RESOURCES = _ROOT / "resources"


def seed_demo_distractors(service: DefaultGraphService, *, pot_id: str) -> None:
    """Give each three-answer demo family six competing claims.

    Top-five recall over only three candidates cannot detect bad ranking.
    These are ordinary project facts, seeded before the labeled answers so
    returning insertion order also fails the retrieval contract.
    """
    groups = (
        (
            "Preference",
            "preference",
            "POLICY_APPLIES_TO",
            "preferences",
            "preference",
            (
                (
                    "timezone-aware",
                    "Store timestamps with timezone information and convert to UTC at persistence boundaries.",
                ),
                (
                    "parameterized-sql",
                    "Bind database query parameters instead of concatenating user input into SQL.",
                ),
                (
                    "bounded-pagination",
                    "List endpoints must paginate with a bounded page size and a continuation cursor.",
                ),
                (
                    "secret-storage",
                    "Load credentials from the operating system credential store, never from committed files.",
                ),
                (
                    "money-decimals",
                    "Represent monetary amounts with Decimal and explicit currency, avoiding floating point.",
                ),
                (
                    "atomic-writes",
                    "Write configuration updates to a temporary file and atomically rename it.",
                ),
            ),
        ),
        (
            "BugPattern",
            "bug_pattern",
            "REPRODUCES",
            "bugs",
            "agent_claim",
            (
                (
                    "expired-certificate",
                    "An expired TLS certificate prevents the gateway from connecting to the upstream service.",
                ),
                (
                    "missing-index",
                    "Customer lookup scans the entire account table because the email column has no index.",
                ),
                (
                    "file-descriptor-leak",
                    "Unclosed uploaded files exhaust the process file descriptor limit during large imports.",
                ),
                (
                    "pagination-gap",
                    "Deleting a record between paginated requests causes offset pagination to skip the next record.",
                ),
                (
                    "stale-permissions",
                    "Cached role assignments allow access after an administrator revokes a user's permissions.",
                ),
                (
                    "disk-full",
                    "The document export fails when temporary archive files fill the worker disk.",
                ),
            ),
        ),
    )
    operations = [
        {
            "op": "assert_claim",
            "subgraph": subgraph,
            "subject": {"key": f"{prefix}:{slug}", "type": entity_type},
            "predicate": predicate,
            "object": {"key": "service:payments-api", "type": "Service"},
            "truth": truth,
            "description": description,
        }
        for entity_type, prefix, predicate, subgraph, truth, claims in groups
        for slug, description in claims
    ]
    result = service.mutate(
        SemanticMutationRequest.parse({"pot_id": pot_id, "operations": operations})
    )
    if result.status != "applied":
        raise AssertionError(f"demo distractors were not applied: {result.to_dict()}")


_BEFORE_RESOURCES = (
    "01-baseline",
    "02-infra",
    "03-decisions",
    "04-preference-policy",
    "05-bugs",
    "06-timeline",
    "07-dup-service",
)
_AFTER_RESOURCES = (
    "08-doclinks",
    "08-records",
    "09-patch-notifier",
    "10-supersede-notifier-target",
    "10-supersedes-system-claim",
    "11-end-statement-tz-staging",
    "12-retract-notifier-postgres",
    "13-merge-dup-service",
    "14-transition-ses",
    "15-plaid-env-prod",
    "16-x",
    "17-end-recon-window",
)


@dataclass(slots=True)
class GoldenPot:
    """The built graph plus resources that must live for the test session."""

    service: DefaultGraphService
    context: AgentContextService
    backend: InMemoryGraphBackend
    resource_store: InMemoryResourceStore
    resource_index: ResourceIndexPort

    def close(self) -> None:
        close = getattr(self.resource_index, "close", None)
        if callable(close):
            close()


class _FrozenRankingService(RankingService):
    """Keep the fixture's 2026 event ordering stable as wall time advances."""

    def rank(
        self, candidates: Iterable[Candidate], context: TaskContext
    ) -> list[RankedItem]:
        return super().rank(
            candidates,
            dataclasses.replace(context, now=context.now or FIXTURE_AS_OF),
        )


def build_deep_query_golden_pot(*, index_home: Path) -> GoldenPot:
    """Replay the checked-in round-6 fixture through real write/read ports."""
    backend = InMemoryGraphBackend(embedder=HashingEmbedder())
    resource_index = build_resource_index("sqlite_fts", home=index_home)
    service = DefaultGraphService(backend=backend, resource_index=resource_index)

    # DefaultGraphService intentionally has no clock/ranker constructor knob.
    # The benchmark replaces only its orchestrator with the same production
    # readers and a frozen clock so recency does not decay as CI ages.
    service._orchestrator = ReadOrchestrator(
        claim_query=backend.claim_query,
        ranker=_FrozenRankingService(),
        reader_registry=service.definition.readers,
        resource_index=resource_index,
    )

    for name in _BEFORE_RESOURCES:
        _apply_plan(service, name)

    resource_store = InMemoryResourceStore()
    resources = ResourceFacade(
        store=resource_store,
        graph=service,
        claims=backend.claim_query,
        index=resource_index,
    )
    for slug, directory, source_ref, source_kind in (
        (
            "ledgerly-oncall-runbook",
            "runbook",
            "https://wiki.ledgerly.internal/oncall",
            "markdown",
        ),
        (
            "adr-0007-advisory-locks",
            "adr",
            "https://github.com/ledgerly-inc/ledgerly/blob/main/docs/adr/0007-advisory-locks.md",
            "markdown",
        ),
        (
            "infra-cost-2026-q3",
            "cost",
            "https://docs.google.com/spreadsheets/d/ledgerly-infra-q3",
            "spreadsheet",
        ),
    ):
        result = resources.import_dir(
            pot_id=DEEP_QUERY_POT_ID,
            slug=slug,
            source_dir=_RESOURCES / directory,
            source_ref=source_ref,
            source_kind=source_kind,
        )
        if result.graph is None or result.graph.status != "applied":
            raise AssertionError(f"resource {slug} did not reach the graph: {result}")
        if result.index is None or result.index.detail:
            raise AssertionError(f"resource {slug} did not reach the index: {result}")

    for name in _AFTER_RESOURCES:
        if name == "10-supersedes-system-claim":
            _add_system_claims(backend, name)
        else:
            _apply_plan(service, name)

    rows = [row for row in backend.claim_query.rows if row.pot_id == DEEP_QUERY_POT_ID]
    live = [row for row in rows if row.invalid_at is None]
    invalidated = [row for row in rows if row.invalid_at is not None]
    predicates = {row.predicate for row in rows}
    entity_keys = {key for row in rows for key in (row.subject_key, row.object_key)}
    entity_keys.update(
        key
        for pot_id, key in backend.claim_query.entity_label_index
        if pot_id == DEEP_QUERY_POT_ID
    )
    actual = (
        len(rows),
        len(live),
        len(invalidated),
        len(entity_keys),
        len(predicates),
    )
    expected = (148, 144, 4, 86, 27)
    if actual != expected:
        raise AssertionError(
            "deep-query fixture drifted: "
            "claims/live/invalidated/entities/predicates="
            f"{actual}, expected={expected}"
        )

    return GoldenPot(
        service=service,
        context=AgentContextService(graph=service, pots=object(), skills=object()),
        backend=backend,
        resource_store=resource_store,
        resource_index=resource_index,
    )


def _apply_plan(service: DefaultGraphService, name: str) -> None:
    payload = json.loads((_PLANS / f"{name}.json").read_text(encoding="utf-8"))
    payload["pot_id"] = DEEP_QUERY_POT_ID
    for operation in payload["operations"]:
        if operation["op"] in {"assert_claim", "link_entities"}:
            operation.setdefault("valid_from", FIXTURE_WRITTEN_AT)
            operation.setdefault("observed_at", FIXTURE_WRITTEN_AT)
    request = SemanticMutationRequest.parse(
        payload,
        approved_by="golden-fixture",
        allow_review_required=True,
    )
    result = service.mutate(request)
    if result.status != "applied":
        raise AssertionError(f"fixture plan {name} was not applied: {result.to_dict()}")


def _add_system_claims(backend: InMemoryGraphBackend, name: str) -> None:
    """Replay writer-internal edges absent from the reference backend.

    ``SUPERSEDES`` is deliberately excluded from the public semantic mutation
    vocabulary. Canonical persistent writers create it while applying a
    supersession invalidation; the in-memory conformance backend currently
    records the invalidation and replacement but not this bookkeeping edge.
    Keep the captured store shape exact without teaching the public validator
    to accept a writer-internal predicate.
    """
    payload = json.loads((_PLANS / f"{name}.json").read_text(encoding="utf-8"))
    for claim in payload["claims"]:
        subject = str(claim["subject_key"])
        object_key = str(claim["object_key"])
        description = str(claim["description"])
        backend.claim_query.add(
            ClaimRow(
                pot_id=DEEP_QUERY_POT_ID,
                predicate=str(claim["predicate"]),
                subject_key=subject,
                object_key=object_key,
                valid_at=FIXTURE_AS_OF,
                evidence_strength="stated",
                source_ref=str(claim["source_ref"]),
                fact=description,
                claim_key=(
                    f"claim:{DEEP_QUERY_POT_ID}:infra_topology:"
                    f"{subject}:SUPERSEDES:{object_key}:fixture"
                ),
                subgraph="infra_topology",
                truth=str(claim["truth"]),
                confidence=0.85,
                description=description,
                observed_at=FIXTURE_AS_OF,
                source_refs=(str(claim["source_ref"]),),
            )
        )


__all__ = [
    "DEEP_QUERY_POT_ID",
    "FIXTURE_AS_OF",
    "GoldenPot",
    "build_deep_query_golden_pot",
]
