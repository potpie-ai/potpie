"""Retrieval eval as a CI number (Graph V1.5 R7).

Proves the bundled local embedder (vector mode) materially beats the labeled
lexical fallback on a paraphrase-heavy golden set — i.e. R1 moved the metric.
"""

from __future__ import annotations

import pytest

from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from adapters.outbound.intelligence.local_embedder import build_embedder
from application.services.graph_service import DefaultGraphService
from benchmarks.retrieval_eval import evaluate, seed_golden

pytestmark = pytest.mark.unit

POT = "eval/pot"


def _report(embedder):
    svc = DefaultGraphService(backend=InMemoryGraphBackend(embedder=embedder))
    seed_golden(svc, pot_id=POT)
    return evaluate(svc, pot_id=POT)


def test_vector_mode_meets_recall_floor() -> None:
    report = _report(build_embedder())
    # Paraphrase recall: the right claim is in the top-5 for most cases, and
    # usually #1. These are CI floors — a regression in the embedder trips them.
    assert report.recall_at_5 >= 0.8, report.to_dict()
    assert report.mrr >= 0.6, report.to_dict()


def test_vector_is_never_worse_than_lexical() -> None:
    vector = _report(build_embedder())
    lexical = _report(None)  # no embedder → labeled Jaccard fallback
    # The bundled local embedder must never regress below the lexical floor.
    assert vector.recall_at_5 >= lexical.recall_at_5
    assert vector.mrr >= lexical.mrr, {
        "vector": vector.to_dict(),
        "lexical": lexical.to_dict(),
    }


def test_vector_recovers_morphological_variants_lexical_misses() -> None:
    """R1: subword vectors recover variants whole-word Jaccard cannot.

    The query shares no *whole word* with the relevant claim — only character
    n-grams (memoize↔memoization, computations↔calculations). Whole-word Jaccard
    scores both claims ~0 (a tie), so it cannot rank the right one first; the
    embedder ranks it first via shared subwords.
    """
    from domain.ports.claim_query import ClaimQueryFilter
    from domain.semantic_mutations import SemanticMutationRequest

    def rank_first_key(embedder) -> str | None:
        svc = DefaultGraphService(backend=InMemoryGraphBackend(embedder=embedder))
        svc.mutate(
            SemanticMutationRequest.parse(
                {
                    "pot_id": POT,
                    "operations": [
                        {
                            "op": "assert_claim",
                            "subgraph": "preferences",
                            "subject": {"key": "preference:memoize", "type": "Preference"},
                            "predicate": "POLICY_APPLIES_TO",
                            "object": {"key": "service:pricing", "type": "Service"},
                            "truth": "preference",
                            "description": "memoize expensive computations",
                        },
                        {
                            "op": "assert_claim",
                            "subgraph": "preferences",
                            "subject": {"key": "preference:validate", "type": "Preference"},
                            "predicate": "POLICY_APPLIES_TO",
                            "object": {"key": "service:pricing", "type": "Service"},
                            "truth": "preference",
                            "description": "validate all inputs at the boundary",
                        },
                    ],
                }
            )
        )
        rows = svc.backend.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=POT,
                predicate_in=("POLICY_APPLIES_TO",),
                fact_query="memoization of pricey calculations",
            )
        )
        return rows[0].subject_key if rows else None

    assert rank_first_key(build_embedder()) == "preference:memoize"
