"""ResourcesReader: the include family that does not read the claim store."""

from __future__ import annotations

import pytest

from potpie_context_core.agent_context_port import (
    DEFAULT_INTENT_INCLUDES,
    READER_BACKED_INCLUDES,
)
from potpie_context_core.ports.resource_index import (
    MATCH_MODE_DISABLED,
    MATCH_MODE_HYBRID,
    MATCH_MODE_LEXICAL,
    ChunkHit,
    IndexCapabilities,
    IndexSearchResult,
)
from potpie_context_engine.application.readers._common import ReadRequest
from potpie_context_engine.application.readers.resources import ResourcesReader
from potpie_context_engine.application.services.envelope_builder import (
    INCLUDE_RANK_WEIGHT,
)
from potpie_context_engine.domain.ranking import RankingService


def hit(
    seq,
    *,
    similarity=None,
    lexical_rank=None,
    semantic_rank=None,
    rank=1,
    term_coverage=None,
):
    return ChunkHit(
        resource_id=f"potpie://res/q3-review/liability/{seq:04d}",
        doc="q3-review",
        section="liability",
        seq=seq,
        document_key="document:q3-review",
        section_key="docsection:q3-review:liability",
        section_title="12. Limitation of Liability",
        label="cap on damages",
        snippet="…liability shall not exceed the fees paid…",
        chars=3980,
        revision=3,
        score=0.5,
        rank=rank,
        match_mode=MATCH_MODE_HYBRID,
        source_ref="file:///q3.pdf",
        lexical_rank=lexical_rank,
        semantic_rank=semantic_rank,
        term_coverage=term_coverage,
        similarity=similarity,
    )


class FakeIndex:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def capabilities(self):
        return IndexCapabilities(profile="fake", lexical=True, snippets=True)

    def search(self, *, pot_id, query, limit=12, doc=None):
        self.calls.append(
            {"pot_id": pot_id, "query": query, "limit": limit, "doc": doc}
        )
        return self.result


def reader(result):
    index = FakeIndex(result)
    return ResourcesReader(index=index, ranker=RankingService()), index


def test_payload_carries_the_graph_keys_and_the_fetch_command():
    rdr, _ = reader(
        IndexSearchResult(
            profile="sqlite_hybrid",
            match_mode=MATCH_MODE_HYBRID,
            hits=(hit(2, similarity=0.6, lexical_rank=2, semantic_rank=1),),
        )
    )
    response = rdr.read(ReadRequest(pot_id="p", query="liability cap"))
    assert len(response.items) == 1
    payload = response.items[0].candidate.payload
    assert payload["kind"] == "resource_chunk"
    # Derived locally from the id — the whole reason the graph tie is free.
    assert payload["document_key"] == "document:q3-review"
    assert payload["section_key"] == "docsection:q3-review:liability"
    assert payload["fetch"].endswith("potpie://res/q3-review/liability/0002")
    assert payload["retrieval"]["match_mode"] == MATCH_MODE_HYBRID
    assert payload["retrieval"]["lexical_rank"] == 2
    assert payload["retrieval"]["semantic_rank"] == 1
    # A snippet, never a body: the two-call search→get path depends on it.
    assert "snippet" in payload and "text" not in payload


def test_a_disabled_index_is_reported_not_raised():
    rdr, _ = reader(
        IndexSearchResult(
            profile="none", match_mode=MATCH_MODE_DISABLED, detail="index is disabled"
        )
    )
    response = rdr.read(ReadRequest(pot_id="p", query="anything"))
    assert response.items == ()
    assert response.coverage_status == "empty"
    # The caller can tell "no matches" from "no index" without a second call.
    assert response.meta["match_mode"] == MATCH_MODE_DISABLED
    assert response.meta["detail"] == "index is disabled"


def test_a_read_with_no_query_returns_nothing_rather_than_filler():
    rdr, index = reader(IndexSearchResult(profile="sqlite_fts"))
    response = rdr.read(ReadRequest(pot_id="p", query="   "))
    assert response.items == () and response.coverage_status == "empty"
    # And it does not even ask: a corpus has no standing order.
    assert index.calls == []


def test_a_term_match_is_scored_by_its_lexical_rank_not_its_cosine():
    """The reader's vote to the ranker must be the fusion's *result*.

    This assertion used to read the other way — a lexical hit passed its raw
    cosine through, or ``None`` when it had none. End-to-end that inverted the
    ranking: ``CRDTs`` occurs in exactly one chunk, the lexical arm ranked it
    first, and its 0.11 cosine put it fourth in the envelope behind three
    chunks that merely drifted near the query in vector space. A hit both arms
    agreed on scored *worse* than a hit only one arm found.

    The measured number stays visible in the payload, so the diagnostic is not
    lost — only the vote changes.
    """
    rdr, _ = reader(
        IndexSearchResult(
            profile="sqlite_fts",
            match_mode=MATCH_MODE_LEXICAL,
            hits=(hit(0, lexical_rank=1),),
        )
    )
    response = rdr.read(ReadRequest(pot_id="p", query="ERR_QUOTA_EXCEEDED"))
    assert response.items[0].candidate.semantic_similarity == 1.0
    # The cosine the semantic arm did not measure is still reported as absent.
    assert response.items[0].candidate.payload["retrieval"]["similarity"] is None


def test_a_thin_term_match_is_discounted_rather_than_scored_as_certain():
    """Rank 1 of a bad pool must not read like rank 1 of a good one.

    Measured end-to-end: with rank as the only lexical signal, every query the
    corpus could not answer returned one identical, confident score, because on
    any query *somebody* ranks first.
    """
    full, thin = (
        reader(
            IndexSearchResult(
                profile="sqlite_fts",
                match_mode=MATCH_MODE_LEXICAL,
                hits=(hit(0, lexical_rank=1, term_coverage=coverage),),
            )
        )[0]
        .read(ReadRequest(pot_id="p", query="anything"))
        .items[0]
        .candidate.semantic_similarity
        for coverage in (1.0, 1 / 3)
    )
    assert full == 1.0
    assert thin == pytest.approx(1 / 3)


def test_a_low_cosine_term_match_outranks_a_high_cosine_drifter():
    """The exact end-to-end inversion, pinned as a unit test."""
    hits = (
        # The only chunk containing the query term, and a poor embedding match.
        hit(0, lexical_rank=1, similarity=0.11, semantic_rank=4, rank=1),
        # Nearer in vector space, no term match at all.
        hit(1, similarity=0.17, semantic_rank=1, rank=2),
    )
    rdr, _ = reader(
        IndexSearchResult(
            profile="sqlite_hybrid", match_mode=MATCH_MODE_HYBRID, hits=hits
        )
    )
    response = rdr.read(ReadRequest(pot_id="p", query="CRDTs"))
    assert response.items[0].candidate.payload["seq"] == 0


def test_the_relevance_floor_cuts_the_semantic_tail_and_spares_term_hits():
    hits = (
        hit(0, similarity=0.60, semantic_rank=1, rank=1),
        hit(1, similarity=0.05, semantic_rank=2, rank=2),  # far below the floor
        hit(2, similarity=0.02, lexical_rank=3, rank=3),  # exact term: survives
    )
    rdr, _ = reader(
        IndexSearchResult(
            profile="sqlite_hybrid", match_mode=MATCH_MODE_HYBRID, hits=hits
        )
    )
    response = rdr.read(ReadRequest(pot_id="p", query="liability"))
    kept = {item.candidate.payload["seq"] for item in response.items}
    assert kept == {0, 2}, "the unscored tail must go and the term hit must stay"


def test_scope_doc_narrows_to_one_document():
    rdr, index = reader(IndexSearchResult(profile="sqlite_fts"))
    rdr.read(ReadRequest(pot_id="p", query="x", scope={"doc": " q3-review "}))
    assert index.calls[0]["doc"] == "q3-review"


def test_scope_service_is_ignored_rather_than_silently_honoured():
    rdr, index = reader(IndexSearchResult(profile="sqlite_fts"))
    rdr.read(ReadRequest(pot_id="p", query="x", scope={"service": "billing"}))
    assert index.calls[0]["doc"] is None


def test_resources_is_advertised_backed_and_demoted():
    assert "resources" in READER_BACKED_INCLUDES
    # Bare ``potpie search`` must reach document text, or a phrase that appears
    # in a document and in no summary reads as "not found".
    assert "resources" in DEFAULT_INTENT_INCLUDES["unknown"]
    assert "resources" in DEFAULT_INTENT_INCLUDES["docs"]
    # Below ``docs``, so a corpus cannot crowd project memory out of a mixed
    # envelope.
    assert INCLUDE_RANK_WEIGHT["resources"] < INCLUDE_RANK_WEIGHT["docs"]
    assert INCLUDE_RANK_WEIGHT["resources"] < INCLUDE_RANK_WEIGHT["decisions"]
