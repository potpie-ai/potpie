"""ANN over-fetch sizing for selective claim filters (resources P9)."""

from __future__ import annotations

import pytest

from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.adapters.outbound.graph.canonical_claim_query import (
    vector_candidate_k,
    vector_filter_is_selective,
)

pytestmark = pytest.mark.unit


def test_selective_filters_raise_ann_candidate_k() -> None:
    assert vector_candidate_k(10, selective=False) == 50
    assert vector_candidate_k(10, selective=True) == 250
    assert vector_candidate_k(20, selective=True) == 500


def test_vector_filter_is_selective_for_predicate_or_subgraph_exclusion() -> None:
    base = ClaimQueryFilter(pot_id="p")
    assert vector_filter_is_selective(base) is False
    assert vector_filter_is_selective(
        ClaimQueryFilter(pot_id="p", predicate_in=("RESOLVED",))
    )
    assert vector_filter_is_selective(
        ClaimQueryFilter(pot_id="p", subgraph_not_in=("knowledge",))
    )
