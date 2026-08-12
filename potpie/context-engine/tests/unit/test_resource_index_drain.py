"""The drain loop: derived state, not a job queue.

These tests pin the properties that make a plain thread sufficient — no
broker, no scheduler, no delivery guarantee — because every one of them is a
property somebody could reasonably "improve away".
"""

from __future__ import annotations

import threading
import time

import pytest

from potpie_context_core.ports.resource_index import DrainReport, IndexCapabilities
from potpie_context_engine.adapters.outbound.resources.index import build_resource_index
from potpie_context_engine.adapters.outbound.resources.index.drain import (
    ResourceIndexDrain,
)
from potpie_context_engine.adapters.outbound.resources.index.sqlite_fts import (
    embed_windows,
    fts_match_expression,
    reciprocal_rank_fusion,
    term_coverage,
)
from potpie_context_core.ports.resource_index import (
    EMBED_WINDOW_CHARS,
    EMBED_WINDOW_OVERLAP_CHARS,
)
from potpie_context_engine.testing.conformance import build_conformance_document


class FakeIndex:
    """An index whose backlog the test controls."""

    def __init__(self, *, semantic=True, pending=5, fail=False):
        self.caps = IndexCapabilities(
            profile="fake", lexical=True, semantic=semantic, hybrid=semantic
        )
        self.pending = pending
        self.fail = fail
        self.passes = 0
        self.seen = threading.Event()

    def capabilities(self):
        return self.caps

    def drain(self, *, pot_id=None, budget=256):
        self.passes += 1
        self.seen.set()
        if self.fail:
            raise RuntimeError("embedder exploded")
        embedded = min(self.pending, budget)
        self.pending -= embedded
        return DrainReport(
            profile="fake", embedded=embedded, remaining=self.pending, batches=1
        )


def test_the_thread_does_not_start_for_a_profile_with_nothing_to_drain():
    """A lexical-only host must not hold a waking thread forever."""
    drain = ResourceIndexDrain(index=FakeIndex(semantic=False))
    assert drain.start() is False
    assert drain.running is False


def test_the_thread_drains_on_start_without_waiting_for_the_idle_tick():
    index = FakeIndex(pending=3)
    drain = ResourceIndexDrain(index=index, idle_interval=30.0)
    assert drain.start() is True
    try:
        assert index.seen.wait(timeout=5.0), "the first pass must not wait 30s"
        deadline = time.monotonic() + 5.0
        while index.pending and time.monotonic() < deadline:
            time.sleep(0.01)
        assert index.pending == 0
    finally:
        drain.stop()
    assert drain.running is False


def test_a_failing_pass_does_not_kill_the_loop():
    """The loop outlives any one failure; a broken embedder must not need a restart."""
    index = FakeIndex(fail=True)
    drain = ResourceIndexDrain(index=index, idle_interval=0.05)
    assert drain.start() is True
    try:
        assert index.seen.wait(timeout=5.0)
        deadline = time.monotonic() + 5.0
        while index.passes < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert index.passes >= 2, "the loop stopped after a failed pass"
    finally:
        drain.stop()


def test_starting_twice_is_idempotent():
    index = FakeIndex(pending=0)
    drain = ResourceIndexDrain(index=index)
    try:
        assert drain.start() is True
        assert drain.start() is True
        assert drain.running is True
    finally:
        drain.stop()


def test_a_crash_leaves_the_work_pending_and_the_next_drain_resumes(tmp_path):
    """The property a queue would provide, provided by a row state instead."""
    manifest, chunks = build_conformance_document()
    first = build_resource_index("sqlite_hybrid", home=tmp_path)
    if not first.capabilities().semantic:
        pytest.skip("sqlite-vec is unavailable in this interpreter")
    first.index_document(pot_id="p", manifest=manifest, chunks=chunks)
    assert first.status(pot_id="p").pending_embeddings > 0
    # Simulate the process dying mid-backlog: drop the handle entirely.
    first.close()

    resumed = build_resource_index("sqlite_hybrid", home=tmp_path)
    assert resumed.status(pot_id="p").pending_embeddings > 0
    resumed.drain(pot_id="p")
    assert resumed.status(pot_id="p").pending_embeddings == 0


def test_changing_the_embedder_marks_every_vector_pending(tmp_path):
    """Invalidation is the entire migration story for a derived index."""

    class Embedder:
        def __init__(self, name, dimensions):
            self.name = name
            self.dimensions = dimensions

        def embed(self, text):
            return tuple([0.1] * self.dimensions)

        def embed_many(self, texts):
            return [self.embed(t) for t in texts]

    from potpie_context_engine.adapters.outbound.resources.index.sqlite_hybrid import (
        SqliteHybridResourceIndex,
    )

    manifest, chunks = build_conformance_document()
    index = SqliteHybridResourceIndex(home=tmp_path, embedder=Embedder("model-a", 8))
    if not index.capabilities().semantic:
        pytest.skip("sqlite-vec is unavailable in this interpreter")
    index.index_document(pot_id="p", manifest=manifest, chunks=chunks)
    index.drain(pot_id="p")
    assert index.status(pot_id="p").pending_embeddings == 0
    index.close()

    swapped = SqliteHybridResourceIndex(home=tmp_path, embedder=Embedder("model-b", 8))
    # Opening is what reconciles: vectors from two models do not share a space,
    # so the only honest options are re-embed or refuse, and the drain already
    # knows how to re-embed.
    assert swapped.status(pot_id="p").pending_embeddings > 0


# --- the pure pieces --------------------------------------------------------


def test_hostile_query_text_becomes_a_legal_match_expression():
    """FTS5 MATCH is a query language; user text is not. Quote, never pass through."""
    assert fts_match_expression("ERR_QUOTA_EXCEEDED") == '"ERR_QUOTA_EXCEEDED"*'
    # ``:`` would be a column filter and raise; ``NEAR`` is an operator. The
    # quoting is what neutralizes them; the trailing ``*`` is ours, applied
    # outside the quotes, so it never turns user text into an operator.
    assert fts_match_expression("col:value") == '"col" OR "value"*'
    assert fts_match_expression('NEAR("x")') == '"NEAR"* OR "x"'
    assert fts_match_expression('"unbalanced') == '"unbalanced"*'
    # Nothing tokenizable is not an empty query — it is *no* query.
    assert fts_match_expression("--- ***") is None
    assert fts_match_expression("") is None


def test_function_words_are_dropped_so_a_question_does_not_match_everything():
    """Measured, not stylistic: OR-ing 'why/did/we' retrieves the whole corpus.

    On the e2e answer key this alone cost more than half of top-1 — the one
    chunk containing ``CRDTs`` ranked fifth behind chunks whose only claim was
    the word "we".
    """
    assert fts_match_expression("why did we reject CRDTs") == '"reject"* OR "CRDTs"*'
    assert fts_match_expression("how do I roll back the payments service") == (
        '"roll"* OR "back"* OR "payments"* OR "service"*'
    )
    # A query that is *only* function words keeps them: retrieving loosely beats
    # retrieving nothing, and the semantic arm is the better judge there anyway.
    assert fts_match_expression("what is it") == '"what"* OR "is" OR "it"'
    # Words an operator might genuinely be searching for are not function words,
    # however short. "not cumulative" means it.
    assert fts_match_expression("not cumulative") == '"not" OR "cumulative"*'


def test_term_coverage_separates_a_real_answer_from_a_lucky_rank():
    """Rank says "nothing beat this"; coverage says "how much of it is here".

    Without the second number every unanswerable query came back at one
    identical, confident score, because somebody always ranks first.
    """
    text = "The API returns ERR_QUOTA_EXCEEDED with a retry-after header."
    assert term_coverage(text, ("ERR_QUOTA_EXCEEDED",)) == 1.0
    # Two of "okta", "password", "reset" are absent: a weak match, scored weak.
    assert term_coverage(text, ("reset", "okta", "password")) == pytest.approx(0.0)
    assert term_coverage(text, ("retry", "header", "okta")) == pytest.approx(2 / 3)
    # Nothing to measure is unknown, not zero — readers must not discount it.
    assert term_coverage(text, ()) is None


def test_coverage_tolerates_morphology_but_not_coincidence():
    """``fail``/``failover`` must count; ``cat``/``catastrophic`` must not.

    Strict token equality alone dropped a correct top-1 answer out of the
    result set: the runbook writes ``failover`` as one word.
    """
    assert term_coverage("Promote the standby during failover.", ("fail",)) == 1.0
    assert term_coverage("Service credits accrue monthly.", ("credit",)) == 1.0
    assert term_coverage("Roll back the deployment.", ("deploy",)) == 1.0
    # Under the prefix minimum, so it stays exact — no "up" covering "upstream".
    assert term_coverage("The upstream service is slow.", ("up",)) == 0.0
    assert term_coverage("A catastrophic outage.", ("cat",)) == 0.0
    # The documented price of not stemming: a prefix cannot cross a spelling
    # change. Pinned so nobody "fixes" the docstring to over-claim again — the
    # first two drafts of that docstring claimed both of these, wrongly.
    assert term_coverage("Webhook retries use backoff.", ("retry",)) == 0.0
    assert term_coverage("Termination for convenience.", ("terminate",)) == 0.0


def test_underscores_stay_inside_one_token():
    """``get_user_id`` is one identifier, not three words."""
    assert fts_match_expression("get_user_id") == '"get_user_id"*'


def test_long_enough_terms_get_a_prefix_so_retrieval_can_reach_inflections():
    """Prefix matching has to reach SQLite, not just the coverage denominator.

    Before this, every token was an exact phrase, so ``fail`` could not
    *retrieve* a chunk saying ``failover`` — while :func:`term_coverage` happily
    credited it as covered. Scoring knew about a match retrieval could not find.
    """
    assert fts_match_expression("failover") == '"failover"*'
    assert fts_match_expression("fail") == '"fail"*'
    # Short tokens stay exact: ``"of"*`` matches most of a corpus and buys
    # nothing, which is the same reason the coverage side has a floor.
    assert fts_match_expression("api v2") == '"api" OR "v2"'
    assert fts_match_expression("dns ttl cache") == '"dns" OR "ttl" OR "cache"*'
    # A prefix widens retrieval; it does not widen coverage, which still counts
    # whole tokens (plus its own >=4-char prefix rule). Under-counting there
    # only discounts confidence, never removes a hit.
    assert term_coverage("Webhook retries use backoff.", ("retry",)) == 0.0


def test_windows_cover_the_whole_chunk_with_overlap():
    """Whole-chunk embedding would index the first third and drop the rest."""
    body = "x" * (EMBED_WINDOW_CHARS * 3)
    spans = embed_windows(body)
    assert len(spans) > 1
    assert spans[0][0] == 0
    # The last window reaches the end of the text.
    assert spans[-1][0] + spans[-1][1] == len(body)
    # Consecutive windows overlap, so a fact on a boundary is whole in one.
    assert spans[1][0] < spans[0][0] + spans[0][1]
    assert spans[0][0] + spans[0][1] - spans[1][0] == EMBED_WINDOW_OVERLAP_CHARS
    # A short chunk is one window, and an empty one is none.
    assert embed_windows("short") == ((0, 5),)
    assert embed_windows("   ") == ()


def test_rrf_rewards_agreement_between_the_arms():
    """A chunk both arms found outranks one only a single arm ranked first."""
    fused = reciprocal_rank_fusion([10, 20, 30], [20, 40])
    assert fused[20][0] > fused[10][0]
    assert fused[20][1] == {0: 2, 1: 1}
    assert fused[40][1] == {1: 2}
