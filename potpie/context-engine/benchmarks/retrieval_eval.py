"""Retrieval eval harness — recall@k / MRR on a golden set (Graph V1.5 R7).

Retrieval relevance, not graph shape, gates three of the four use cases. This
harness makes that relevance a *number*: a golden set of (query → expected
claim) cases is run against a backend and scored with ``recall@k`` and ``MRR``,
separate from end-to-end answer quality. Any embedding/weight change can then
report its delta (and CI can gate on it).

The harness is backend-agnostic: it seeds claims through ``DefaultGraphService``
and queries through the same read path, so it measures the *real* retrieval
stack (embedder + claim store + ranker), not a mock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from application.services.graph_service import DefaultGraphService
from domain.ports.claim_query import ClaimQueryFilter


@dataclass(frozen=True, slots=True)
class GoldenClaim:
    """One claim to seed, identified by its subject key."""

    subject_key: str
    subject_type: str
    predicate: str
    object_key: str
    object_type: str
    subgraph: str
    truth: str
    description: str


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One retrieval case: a query that should surface ``expected`` first."""

    query: str
    predicate: str
    expected_subject_key: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate retrieval metrics over a golden set."""

    recall_at_1: float
    recall_at_5: float
    mrr: float
    n: int
    ranks: Mapping[str, int | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "recall@1": round(self.recall_at_1, 3),
            "recall@5": round(self.recall_at_5, 3),
            "mrr": round(self.mrr, 3),
            "n": self.n,
        }


# --- Golden corpus (seeded from the four use cases) -------------------------
# Paraphrase-heavy on purpose: the query shares little or no exact wording with
# the claim, so lexical overlap fails and only real vector recall succeeds.

GOLDEN_CLAIMS: tuple[GoldenClaim, ...] = (
    GoldenClaim(
        "preference:wrap-retries", "Preference", "POLICY_APPLIES_TO",
        "service:payments-api", "Service", "preferences", "preference",
        "wrap every outbound HTTP call to a third party in a tenacity retry with exponential backoff",
    ),
    GoldenClaim(
        "preference:no-print", "Preference", "POLICY_APPLIES_TO",
        "service:payments-api", "Service", "preferences", "preference",
        "never use bare print statements for diagnostics; emit structured logs instead",
    ),
    GoldenClaim(
        "preference:four-space", "Preference", "POLICY_APPLIES_TO",
        "service:payments-api", "Service", "preferences", "preference",
        "indent python source with four spaces, never tabs",
    ),
    GoldenClaim(
        "bug_pattern:refund-deadlock", "BugPattern", "REPRODUCES",
        "service:payments-api", "Service", "bugs", "agent_claim",
        "payment deadlock when two refunds settle the same order concurrently, a lock-ordering race",
    ),
    GoldenClaim(
        "bug_pattern:timezone-off-by-one", "BugPattern", "REPRODUCES",
        "service:reporting", "Service", "bugs", "agent_claim",
        "daily report totals shift by a day near month boundaries due to naive UTC truncation",
    ),
    GoldenClaim(
        "bug_pattern:webhook-replay", "BugPattern", "REPRODUCES",
        "service:payments-api", "Service", "bugs", "agent_claim",
        "duplicate charges when stripe retries a webhook and the handler is not idempotent",
    ),
)

GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        "add automatic retries to flaky external API requests",
        "POLICY_APPLIES_TO", "preference:wrap-retries", "retry paraphrase",
    ),
    GoldenCase(
        "stop debugging with console output, use the logger",
        "POLICY_APPLIES_TO", "preference:no-print", "logging paraphrase",
    ),
    GoldenCase(
        "concurrent settle causes a hang on refunds",
        "REPRODUCES", "bug_pattern:refund-deadlock", "deadlock paraphrase",
    ),
    GoldenCase(
        "report numbers are wrong at the end of the month",
        "REPRODUCES", "bug_pattern:timezone-off-by-one", "tz paraphrase",
    ),
    GoldenCase(
        "stripe sends the same event twice and we double charge",
        "REPRODUCES", "bug_pattern:webhook-replay", "idempotency paraphrase",
    ),
)


def seed_golden(service: DefaultGraphService, *, pot_id: str) -> None:
    """Seed the golden claims through the real write path (embed-on-write)."""
    from domain.semantic_mutations import SemanticMutationRequest

    ops = [
        {
            "op": "assert_claim",
            "subgraph": c.subgraph,
            "subject": {"key": c.subject_key, "type": c.subject_type},
            "predicate": c.predicate,
            "object": {"key": c.object_key, "type": c.object_type},
            "truth": c.truth,
            "description": c.description,
        }
        for c in GOLDEN_CLAIMS
    ]
    service.mutate(SemanticMutationRequest.parse({"pot_id": pot_id, "operations": ops}))


def _rank_of(
    service: DefaultGraphService, *, pot_id: str, case: GoldenCase, k: int
) -> int | None:
    rows = service.backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=pot_id,
            predicate_in=(case.predicate,),
            fact_query=case.query,
            limit=max(k * 3, 15),
        )
    )
    for idx, row in enumerate(rows, start=1):
        if row.subject_key == case.expected_subject_key:
            return idx
    return None


def evaluate(
    service: DefaultGraphService,
    *,
    pot_id: str,
    cases: Sequence[GoldenCase] = GOLDEN_CASES,
    k: int = 5,
) -> EvalReport:
    """Run the golden cases and compute recall@1 / recall@5 / MRR."""
    ranks: dict[str, int | None] = {}
    hit1 = 0
    hitk = 0
    mrr_sum = 0.0
    for case in cases:
        rank = _rank_of(service, pot_id=pot_id, case=case, k=k)
        ranks[case.expected_subject_key] = rank
        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= k:
                hitk += 1
            mrr_sum += 1.0 / rank
    n = len(cases) or 1
    return EvalReport(
        recall_at_1=hit1 / n,
        recall_at_5=hitk / n,
        mrr=mrr_sum / n,
        n=len(cases),
        ranks=ranks,
    )


__all__ = [
    "EvalReport",
    "GOLDEN_CASES",
    "GOLDEN_CLAIMS",
    "GoldenCase",
    "GoldenClaim",
    "evaluate",
    "seed_golden",
]
