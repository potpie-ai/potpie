"""ResourcesReader — document *payloads*, reached through the index.

Every other P9 reader queries the claim store. This one does not, and the
difference is the point: ``docs`` returns what an agent *wrote about* a
document (one summary per section, the only index that existed before), while
``resources`` returns the document's own text. A fact no summary mentions is
unreachable through ``docs`` and reachable here — which is the failure the
resource-search design ranks first.

The two families are complementary, not redundant, so both stay: ``docs`` for
"which document covers this, and what did we say about it", ``resources`` for
"which passage actually says it". A hit carries ``document_key`` and
``section_key``, computed from the resource id by pure string manipulation, so
crossing back to the graph costs no second retrieval call.

A hit is a ~240-char snippet plus the chunk id, never a chunk body: twelve
4,000-char chunks would be ~48k chars in one envelope, and the two-call
``search`` → ``resource get`` path exists precisely so an agent spends that
budget only on the chunk it chose.

Degradation is labeled, never silent. An index that is switched off or cannot
load answers ``match_mode == "disabled"`` with zero hits, and this reader
reports that in ``meta`` rather than raising — the read has other include
families to answer, and an unindexed deployment is a configuration fact, not an
error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from potpie_context_core.ports.resource_index import (
    LEXICAL_RANK_DECAY,
    MATCH_MODE_DISABLED,
    SIMILARITY_BLEND,
    ChunkHit,
    IndexSearchResult,
    ResourceIndexPort,
)
from potpie_context_engine.application.readers._common import (
    RELEVANCE_FLOOR_FRACTION,
    RELEVANCE_FLOOR_MINIMUM,
    ReadRequest,
    ReadResponse,
    coverage_status_from_count,
    rank_candidates,
)
from potpie_context_engine.domain.ranking import Candidate, RankingService


@dataclass(slots=True)
class ResourcesReader:
    """Answers ``--include resources`` from a :class:`ResourceIndexPort`."""

    index: ResourceIndexPort
    ranker: RankingService
    claim_query: Any = None
    """Accepted and unused, so the orchestrator builds every reader alike."""

    family: str = "resources"

    def read(self, req: ReadRequest) -> ReadResponse:
        query = (req.query or "").strip()
        if not query:
            # A resource corpus has no standing order — no recency, no
            # strength, nothing that makes one passage the answer to no
            # question. Returning the first N chunks of whatever happens to be
            # imported would be filler occupying slots a scoped family could
            # have used.
            return ReadResponse(
                family=self.family,
                items=(),
                coverage_status="empty",
                meta={
                    "match_mode": MATCH_MODE_DISABLED,
                    "candidate_pool": 0,
                    "detail": "resources needs a query; none was given",
                },
            )

        result = self.index.search(
            pot_id=req.pot_id,
            query=query,
            # Over-fetch so the floor below has a tail to cut and still leaves
            # a full page behind.
            limit=max(req.max_items * 2, req.max_items),
            doc=_requested_doc(req),
        )
        hits = _hits_clearing_relevance_floor(result.hits)
        ranked = rank_candidates(
            service=self.ranker,
            candidates=[_candidate(hit, result) for hit in hits],
            req=req,
        )
        return ReadResponse(
            family=self.family,
            items=tuple(ranked),
            coverage_status=coverage_status_from_count(
                found=len(ranked), requested=req.max_items
            ),
            meta={
                # The retrieval that actually ran, so a caller can tell "no
                # matches" from "no index" without a second call.
                "match_mode": result.match_mode,
                "profile": result.profile,
                "candidate_pool": len(result.hits),
                "lexical_candidates": result.lexical_candidates,
                "semantic_candidates": result.semantic_candidates,
                "detail": result.detail,
            },
        )


def _requested_doc(req: ReadRequest) -> str | None:
    """A single-document filter, when the caller scoped to one.

    ``doc`` is the only scope key this family understands. Service and repo
    scopes are claim-graph concepts and a chunk carries no edges, so honouring
    them here would mean silently returning everything.

    Reachable through ``SearchRequest.scope`` only: ``potpie search`` exposes
    ``--include``, ``--intent`` and ``--pot`` and has no ``--scope``, so a CLI
    caller who knows they want one document cannot say so. Stated here rather
    than left implied, because an earlier draft of this docstring cited a
    ``--scope doc=`` flag that does not exist."""
    value = req.scope.get("doc") or req.scope.get("document")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _hits_clearing_relevance_floor(hits: Sequence[ChunkHit]) -> list[ChunkHit]:
    """Drop the semantic tail the query gives no reason to return.

    A k-NN returns *k* rows whether or not any of them is an answer, so without
    a floor a twelve-item envelope is padded to twelve with whatever the corpus
    happens to hold — the failure that put a cost spreadsheet in front of a
    database-incident question.

    Two rules, the same two ``docs`` applies and for the same reasons:

    - A hit the **lexical** arm found survives regardless. An exact identifier
      is the strongest relevance signal there is and the weakest embedding one:
      ``ERR_QUOTA_EXCEEDED`` is one token to FTS5 and noise to MiniLM.
    - A **semantic-only** hit is judged against a floor derived from the pool's
      own best similarity, not an absolute threshold. Similarity here is
      ``1 - cosine distance``, which is not a calibrated probability and whose
      usable range moves with the embedder and the corpus; what is stable is
      that a real answer sits clear of the tail behind it.
    """
    if not hits:
        return []
    best = max((hit.similarity or 0.0) for hit in hits)
    floor = max(best * RELEVANCE_FLOOR_FRACTION, RELEVANCE_FLOOR_MINIMUM)
    return [
        hit
        for hit in hits
        if hit.lexical_rank is not None or (hit.similarity or 0.0) >= floor
    ]


def _candidate(hit: ChunkHit, result: IndexSearchResult) -> Candidate:
    return Candidate(
        # The chunk id is already globally unique and is what ``resource get``
        # takes, so it doubles as the candidate key with nothing invented.
        candidate_key=hit.resource_id,
        payload=_payload(hit, result),
        # Verbatim source text, quoted rather than paraphrased: the strongest
        # evidence class the ranker knows. Cross-family crowding is handled
        # once, by the envelope builder's rank demotion, rather than by
        # understating what a chunk is here.
        strength="attested",
        semantic_similarity=_relevance(hit, calibrated=result.similarity_calibrated),
    )


def _relevance(hit: ChunkHit, *, calibrated: bool = False) -> float | None:
    """The relevance the ranker should order this hit by.

    ``semantic_similarity`` is the ranker's only relevance channel, so whatever
    goes here *is* the index's vote. Handing it ``hit.similarity`` — the raw
    cosine — looked natural and was wrong: it forwards one **input** of the
    fusion instead of its result, and silently overrules the arm that is right
    far more often.

    The failure it caused is worth stating exactly, because it inverts the
    obvious intuition. ``CRDTs`` appears in exactly one chunk of the corpus.
    The lexical arm ranked that chunk first; its embedding cosine was 0.11,
    lower than three chunks that merely drift near the query in vector space,
    so the ranker demoted the only real answer to fourth. A hit matched by
    *both* arms scored worse than a hit matched by one — and worse still than a
    lexical-only hit, which returns ``None`` here and collects the ranker's
    neutral 0.5 default.

    So a hit the lexical arm found is scored by its **lexical rank**. That is
    the same judgement :func:`_hits_clearing_relevance_floor` already makes one
    step earlier when it exempts lexical hits from the similarity floor — "an
    exact identifier is the strongest relevance signal there is and the weakest
    embedding one". This makes scoring agree with filtering instead of
    contradicting it. Semantic-only hits are untouched: their measured cosine
    is the honest number and stays.

    What the paragraph above got *half* right is that the cosine is an input to
    be weighed rather than a verdict to be obeyed. Forwarding it whole inverts
    the ranking; discarding it leaves the score purely ordinal, and an ordinal
    score cannot tell "nothing beat this" from "this answers the question" —
    measured, that is a corpus where unanswerable queries top out *above* the
    weakest correct answer. So when both arms scored a chunk, the two are
    blended at :data:`SIMILARITY_BLEND`, which is measured rather than assumed
    and is deliberately short of 1.0 for exactly the reason the ``CRDTs``
    failure gives.

    A chunk **only** the lexical arm found keeps the pure rank-and-coverage
    score: no cosine was measured for it, and substituting a neutral stand-in
    would be inventing a signal, the same mistake as treating unknown coverage
    as zero. The asymmetry is real but small in practice — measured over 190
    questions, 189 of the top hits were scored by both arms — and it grows only
    as a corpus outgrows the candidate pool.

    ``calibrated`` is what keeps this honest across deployments. The blend is an
    improvement only where the cosine means something: with a real sentence
    encoder it moves top-1 from 0.700 to 0.721, and with the bundled hashing
    embedder — potpie's OSS default, which ships without a model download — it
    moves top-1 from 0.705 *down* to 0.658. So an uncalibrated index keeps the
    ordinal score, and the tuning that helps a hosted deployment cannot quietly
    degrade the default one.
    """
    if hit.lexical_rank is None:
        return hit.similarity
    rank_score = LEXICAL_RANK_DECAY / (LEXICAL_RANK_DECAY - 1.0 + hit.lexical_rank)
    # Rank alone says only that nothing beat this chunk, and on a query the
    # corpus cannot answer *somebody* still ranks first — which is how every
    # unanswerable query came back at one identical, confident score. Coverage
    # is the second half of the signal: how much of what was asked for is
    # actually here. Unknown coverage means unmeasured, so it does not discount.
    coverage = 1.0 if hit.term_coverage is None else hit.term_coverage
    ordinal = rank_score * coverage
    if hit.similarity is None or not calibrated:
        return ordinal
    return (1.0 - SIMILARITY_BLEND) * ordinal + SIMILARITY_BLEND * hit.similarity


def _payload(hit: ChunkHit, result: IndexSearchResult) -> dict[str, Any]:
    return {
        "kind": "resource_chunk",
        "resource_id": hit.resource_id,
        "doc": hit.doc,
        "section": hit.section,
        "seq": hit.seq,
        # Derivable from the id, and handed over anyway: an agent that has to
        # know the key grammar to cross into the graph is an agent that will
        # spend a call finding out.
        "document_key": hit.document_key,
        "section_key": hit.section_key,
        "section_title": hit.section_title,
        "label": hit.label,
        "snippet": hit.snippet,
        "chars": hit.chars,
        "revision": hit.revision,
        "source_ref": hit.source_ref,
        "retrieval": {
            "profile": result.profile,
            "match_mode": hit.match_mode,
            "rank": hit.rank,
            "lexical_rank": hit.lexical_rank,
            "semantic_rank": hit.semantic_rank,
            "similarity": hit.similarity,
            # Both halves of the lexical judgement are published, because
            # "ranked first" and "contains one of your five words" are very
            # different answers and only the pair distinguishes them.
            "term_coverage": hit.term_coverage,
        },
        # The next command, spelled out. ``snippet`` is a window; this is how
        # the agent gets the passage.
        "fetch": f"potpie resource get {hit.resource_id}",
    }


__all__ = ["ResourcesReader"]
