"""``ResourceIndexPort`` — the swappable retrieval index over stored chunks.

A document imported into Potpie splits in two: bytes to
:mod:`~potpie_context_core.ports.resource_store`, structure to the graph. Until
this port existed the *only* index into those bytes was one agent-authored
summary per section, so a fact no summary mentions was unreachable and nothing
reported the miss.

This port is that index, modelled on ``GraphBackend``: a capability bundle plus
narrow methods, so the retrieval *algorithm* and its *storage* are both
swappable while the CLI surface, the envelope shape, and the agent flow stay
fixed. An unbuilt capability fails closed with ``CapabilityNotImplemented``
rather than returning ``None`` — see
``adapters/outbound/resources/index/_unimplemented.py``.

Two properties the rest of the design leans on:

*The index is derived, never a source of truth.* Files are the truth; every row
here can be recomputed from them, which is why there is no backup, export, or
schema-version negotiation — ``resource index rebuild`` is the recovery path.

*Pending work is a row state, not a message.* Chunks land with their vectors
``NULL`` and a single writer thread drains them, so "what is pending" is a
``SELECT``, a crash is a no-op, and no broker, scheduler, or worker process is
introduced. :meth:`ResourceIndexPort.drain` is that loop's one step.

The DTOs live in ``potpie-context-core`` because the local daemon reconstructs
them by module path when a call crosses the RPC boundary (``potpie/daemon/rpc.py``
allows core classes only), so every field here must survive a
``cls(**decoded_fields)`` round trip: no ``init=False`` fields, and tuples
rather than sets for sequences.

See ``docs/context-graph/resources.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from potpie_context_core.ports.resource_store import Chunk, DocumentManifest

# --- Contract constants -----------------------------------------------------

#: How a degraded profile says so, reusing the vocabulary the codebase already
#: has for claim retrieval (``ports/graph/semantic``, ``in_memory_reader``). A
#: second word for the same idea would be drift, so callers render one set of
#: labels whatever produced them.
MATCH_MODE_LEXICAL = "lexical"
MATCH_MODE_SEMANTIC = "semantic"
MATCH_MODE_HYBRID = "hybrid"
MATCH_MODE_DISABLED = "disabled"

MATCH_MODES: tuple[str, ...] = (
    MATCH_MODE_LEXICAL,
    MATCH_MODE_SEMANTIC,
    MATCH_MODE_HYBRID,
    MATCH_MODE_DISABLED,
)

#: A hit carries a window of the chunk, never the chunk. Twelve 4,000-char
#: chunks would put ~48k chars in one envelope; a snippet plus the id preserves
#: the two-call ``search`` → ``resource get`` path.
SNIPPET_TARGET_CHARS = 240

#: Sub-window size for embedding. MiniLM-L6-v2 truncates at 256 tokens and a
#: 4,000-char chunk is ~792, so embedding whole chunks would silently index
#: their first third and discard the rest. This is correctness, not tuning.
EMBED_WINDOW_CHARS = 800
EMBED_WINDOW_OVERLAP_CHARS = 160

#: How many texts one :meth:`ResourceIndexPort.drain` step embeds. Bounded so a
#: drain is interruptible: the daemon hard-kills 10s after SIGTERM and a batch
#: that does not finish is simply left pending.
DEFAULT_DRAIN_BUDGET = 256

#: Reciprocal-rank-fusion constant. 60 is the value the RRF paper uses and the
#: one every implementation defaults to; it is here so the two arms fuse the
#: same way in every profile.
RRF_K = 60

#: Converts a lexical rank into the [0, 1] relevance the envelope ranker orders
#: by: ``K / (K - 1 + rank)``, so rank 1 scores 1.0, rank 2 scores 0.9, and rank
#: 10 scores 0.5 — the ranker's own neutral default, which is the right place
#: for the tail of a lexical result set to land.
#:
#: Distinct from :data:`RRF_K` on purpose. RRF fuses two rankings *inside* the
#: index and 60 deliberately flattens them so neither arm dominates. This maps
#: one ranking *out* of the index, where flattening is the opposite of what is
#: wanted: 60 would compress ranks 1-10 into 0.983-0.870, a spread far too
#: narrow to survive a weighted mean against recency and strength.
LEXICAL_RANK_DECAY = 10.0

#: How much of a lexical hit's relevance is its measured cosine, when the
#: semantic arm scored the same chunk. The rest stays rank-and-coverage.
#:
#: Rank-only relevance is *ordinal*: rank 1 scores 1.0 whether the chunk answers
#: the query or merely beat everything else, which is why unanswerable queries
#: used to come back at the same confident score as correct ones. Cosine is the
#: only bounded, measured [0, 1] signal in the pipeline, so it is what turns
#: "nothing beat this" into "this is actually close to what was asked".
#:
#: 0.75 is measured, not chosen for roundness — see
#: ``docs/context-graph/retrieval-tuning.md``. Over a 316-chunk corpus and 190
#: labelled questions the region 0.65-0.80 is a plateau, and both ends of it are
#: far better than either extreme. **1.0 is not the limit of a good thing**: at
#: pure cosine, top-1 collapses from 0.72 to 0.54, which is the same failure
#: :func:`~...readers.resources._relevance` documents — an exact identifier is
#: the strongest lexical signal and the weakest embedding one, so the remaining
#: 0.25 of rank-and-coverage is what keeps ``ERR_QUOTA_EXCEEDED`` findable.
SIMILARITY_BLEND = 0.75

# --- Error codes ------------------------------------------------------------
# Stable strings the CLI maps to exit codes and ``--json`` error payloads.

RESOURCE_INDEX_UNAVAILABLE = "resource_index_unavailable"
RESOURCE_INDEX_QUERY_INVALID = "resource_index_query_invalid"
RESOURCE_INDEX_PROFILE_UNKNOWN = "resource_index_profile_unknown"
RESOURCE_INDEX_WRITE_FAILED = "resource_index_write_failed"


class ResourceIndexError(ValueError):
    """A resource-index operation failed with a stable, reportable ``code``.

    Mirrors ``ResourceStoreError`` exactly, including the ``ValueError`` base:
    the daemon's error payload has a ``ValueError`` branch that forwards
    ``detail`` and ``recommended_next_action``, while anything else is reduced
    to a bare message on its way across the hop.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        detail: str | None = None,
        recommended_next_action: str | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.recommended_next_action = recommended_next_action
        super().__init__(message)

    def __reduce__(self) -> tuple[Any, ...]:
        # ``BaseException.__reduce__`` would hand ``self.args`` — just the
        # message — back to a two-argument ``__init__``. Spell the round trip
        # out so copy and pickle work.
        return (
            self.__class__,
            (self.code, str(self)),
            {
                "detail": self.detail,
                "recommended_next_action": self.recommended_next_action,
            },
        )


# --- Capabilities -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexCapabilities:
    """Static declaration of what a profile really answers.

    A copy of ``BackendCapabilities`` down to the ``implemented()`` helper,
    including the rule it enforces: a slot declared ``False`` is backed by a
    stub that raises ``CapabilityNotImplemented`` naming
    ``resource_index.<profile>.<capability>.<method>``, so a gap is
    attributable in telemetry rather than an ``AttributeError`` at a call site.
    """

    profile: str
    lexical: bool = False
    """BM25 / term match over chunk text."""

    semantic: bool = False
    """Vector similarity over embedded sub-windows and section summaries."""

    hybrid: bool = False
    """Fuses both arms in one call (implies ``lexical`` and ``semantic``)."""

    snippets: bool = False
    """Can return a matching window, not just an id."""

    incremental: bool = False
    """Can index one document without a full rebuild."""

    def implemented(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("lexical", "semantic", "hybrid", "snippets", "incremental")
            if getattr(self, name)
        )

    def match_mode(self) -> str:
        """The strongest mode this profile can answer in.

        The single place the capability bundle is turned into the ``match_mode``
        vocabulary, so a profile cannot describe itself one way in ``status``
        and another way in a search payload.
        """
        if self.hybrid and self.lexical and self.semantic:
            return MATCH_MODE_HYBRID
        if self.semantic:
            return MATCH_MODE_SEMANTIC
        if self.lexical:
            return MATCH_MODE_LEXICAL
        return MATCH_MODE_DISABLED


# --- DTOs -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkHit:
    """One chunk the index matched, plus the graph keys it ties back to.

    ``document_key`` and ``section_key`` are computed locally from the resource
    id — both slug grammars forbid ``:`` and ``/``, so the mapping is total and
    bidirectional — which is what makes the graph tie free. An agent moves
    straight to ``graph neighborhood --entity <section_key>`` with no second
    retrieval round trip.
    """

    resource_id: str
    doc: str
    section: str
    seq: int
    document_key: str
    section_key: str
    section_title: str
    label: str
    snippet: str
    chars: int
    revision: int
    score: float
    rank: int
    match_mode: str
    source_ref: str | None = None
    lexical_rank: int | None = None
    """Rank in the lexical arm, when that arm returned this chunk."""

    semantic_rank: int | None = None
    """Rank in the semantic arm, when that arm returned this chunk."""

    similarity: float | None = None
    """Cosine similarity to the query, when the semantic arm scored this chunk.

    ``None`` for a lexical-only hit. This is a *diagnostic*, not the reader's
    vote to the ranker — see :func:`~...readers.resources._relevance` for why
    forwarding it as the vote inverted the ranking."""

    term_coverage: float | None = None
    """Fraction of the query's content terms this chunk actually contains.

    The missing half of "how good is this lexical match". A rank says only that
    nothing beat this chunk; on a query no chunk answers, rank 1 still goes to
    somebody. Coverage is what separates the two: ``ERR_CARD_DECLINED`` is one
    term and the containing chunk scores 1.0, while "how do I reset my Okta
    password" leaves any chunk matching only "reset" at 0.33 — so an absent
    answer reads as weak instead of reading as certain.

    ``None`` when no arm measured it (a semantic-only hit, or an index with no
    lexical arm), which readers must treat as "unknown", never as zero."""


@dataclass(frozen=True, slots=True)
class IndexSearchResult:
    """A search's hits plus how they were found.

    ``match_mode`` is the labeled-degradation channel: a profile that can only
    answer lexically says ``"lexical"``, and one that cannot answer at all says
    ``"disabled"`` and returns zero hits — never an error, so a caller's
    coverage report stays legible instead of the whole read failing.
    """

    profile: str
    match_mode: str = MATCH_MODE_DISABLED
    hits: tuple[ChunkHit, ...] = ()
    lexical_candidates: int = 0
    semantic_candidates: int = 0
    detail: str | None = None

    similarity_calibrated: bool = False
    """Whether ``ChunkHit.similarity`` means "close in meaning", not just "ranks".

    Travels with the result rather than being looked up, because the reader has
    no way to reach the embedder and should not grow one. ``False`` is the
    conservative default: a profile that has not said so is read as ordinal, and
    a reader that ignores this field behaves exactly as it did before the field
    existed. See :data:`SIMILARITY_BLEND` for what turns on it."""


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What one :meth:`ResourceIndexPort.index_document` call wrote.

    ``pending_embeddings`` is the count this call *left* for the drain, not a
    total: the lexical postings are written inline (measured 0.68s per 10k
    chunks) and the vectors are deliberately deferred, so a caller reporting
    "import finished" while embeddings are outstanding is describing the design
    rather than a partial failure.
    """

    doc: str
    profile: str
    sections: int = 0
    chunks: int = 0
    windows: int = 0
    pending_embeddings: int = 0
    removed_chunks: int = 0
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DrainReport:
    """One pass of the drain loop over rows whose embedding is still ``NULL``."""

    profile: str
    embedded: int = 0
    remaining: int = 0
    batches: int = 0
    elapsed_ms: int = 0
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceIndexStatus:
    """Whether the index can answer right now, and what it holds.

    ``potpie resource index status`` and ``potpie doctor`` are the callers, so
    like ``ResourceStoreStatus`` this never raises: an index that cannot answer
    says so with ``ready=False`` and a ``detail``, because a diagnostic that
    throws is useless in exactly the situation it exists for.

    ``replica`` and ``shared_store`` exist for one honest reason. The index is
    process-local while the file store may be a shared volume, so a document
    imported through one replica is invisible to another until that replica
    rebuilds. Rather than pretend otherwise, the status names the replica it
    was answered by and flags when the store underneath is shared — the
    deployment constraint is then visible in a diagnostic instead of surfacing
    as a document that is silently unfindable half the time.
    """

    profile: str
    ready: bool
    capabilities: tuple[str, ...] = ()
    match_mode: str = MATCH_MODE_DISABLED
    documents: int = 0
    chunks: int = 0
    windows: int = 0
    pending_embeddings: int = 0
    embedder: str | None = None
    dimensions: int | None = None
    location: str | None = None
    replica: str | None = None
    shared_store: bool = False
    detail: str | None = None


# --- Port -------------------------------------------------------------------


@runtime_checkable
class ResourceIndexPort(Protocol):
    """Pot-scoped retrieval index over the document store's chunks."""

    def capabilities(self) -> IndexCapabilities:
        """Declare what this profile answers, without triggering an error."""
        ...

    def search(
        self,
        *,
        pot_id: str,
        query: str,
        limit: int = 12,
        doc: str | None = None,
    ) -> IndexSearchResult:
        """Return the best chunks for ``query``, one item per chunk.

        Runs every retrieval arm the profile declares and fuses them by
        reciprocal rank. A profile that declares nothing returns zero hits with
        ``match_mode == "disabled"`` rather than raising: an unindexed
        deployment must degrade in a *labeled* way, not fail the read."""
        ...

    def index_document(
        self,
        *,
        pot_id: str,
        manifest: DocumentManifest,
        chunks: tuple[Chunk, ...],
    ) -> IndexReport:
        """Absorb one document's chunks, replacing any prior revision.

        Lexical postings are written inline; vectors are left ``NULL`` for the
        drain. Idempotent by ``(pot_id, resource_id)``, so re-indexing a
        document is a replace rather than a duplicate."""
        ...

    def drop_document(self, *, pot_id: str, slug: str) -> bool:
        """Remove one document's rows. Dropping what is absent is a no-op."""
        ...

    def purge_pot(self, pot_id: str) -> bool:
        """Remove every row the pot owns, for pot reset and teardown."""
        ...

    def drain(
        self, *, pot_id: str | None = None, budget: int = DEFAULT_DRAIN_BUDGET
    ) -> DrainReport:
        """Embed up to ``budget`` pending windows and report what is left.

        The whole background story: pending work is ``embedding IS NULL``, so
        this is resumable across restarts, idempotent under concurrent callers
        that serialize on the writer, and observable by counting."""
        ...

    def status(self, *, pot_id: str | None = None) -> ResourceIndexStatus:
        """Report readiness and counts for ``potpie doctor``. Never raises."""
        ...


__all__ = [
    "DEFAULT_DRAIN_BUDGET",
    "EMBED_WINDOW_CHARS",
    "EMBED_WINDOW_OVERLAP_CHARS",
    "MATCH_MODES",
    "MATCH_MODE_DISABLED",
    "MATCH_MODE_HYBRID",
    "MATCH_MODE_LEXICAL",
    "MATCH_MODE_SEMANTIC",
    "RESOURCE_INDEX_PROFILE_UNKNOWN",
    "RESOURCE_INDEX_QUERY_INVALID",
    "RESOURCE_INDEX_UNAVAILABLE",
    "RESOURCE_INDEX_WRITE_FAILED",
    "RRF_K",
    "SIMILARITY_BLEND",
    "SNIPPET_TARGET_CHARS",
    "ChunkHit",
    "DrainReport",
    "IndexCapabilities",
    "IndexReport",
    "IndexSearchResult",
    "ResourceIndexError",
    "ResourceIndexPort",
    "ResourceIndexStatus",
]
