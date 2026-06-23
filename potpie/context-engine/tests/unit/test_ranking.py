"""Uniform ranker (rebuild plan P7)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


from potpie.context_engine.domain.ranking import (
    Candidate,
    RankingService,
    TaskContext,
    candidate_from_edge_record,
    truncate,
)


_NOW = datetime(2026, 5, 20, tzinfo=timezone.utc)


def _ctx(**overrides) -> TaskContext:
    return TaskContext(
        pot_id="pot-1",
        scope=overrides.pop("scope", {}),
        intent=overrides.pop("intent", None),
        freshness_preference=overrides.pop("freshness_preference", "balanced"),
        now=_NOW,
        **overrides,
    )


def _make_candidate(
    *,
    key: str,
    strength: str = "attested",
    valid_at: datetime | None = None,
    corroboration: int = 1,
    scope_overlap: float | None = None,
    semantic_similarity: float | None = None,
    coverage_status: str | None = None,
) -> Candidate:
    return Candidate(
        candidate_key=key,
        payload={"key": key},
        strength=strength,
        valid_at=valid_at,
        corroboration_count=corroboration,
        scope_overlap=scope_overlap,
        semantic_similarity=semantic_similarity,
        coverage_status=coverage_status,
    )


class TestStrengthDominantWhenAllElseEqual:
    def test_deterministic_beats_attested_beats_stated(self) -> None:
        service = RankingService()
        ranked = service.rank(
            [
                _make_candidate(key="a", strength="stated"),
                _make_candidate(key="b", strength="attested"),
                _make_candidate(key="c", strength="deterministic"),
            ],
            _ctx(),
        )
        order = [r.candidate.candidate_key for r in ranked]
        assert order == ["c", "b", "a"]


class TestRecencyMatters:
    def test_more_recent_wins_when_strength_equal(self) -> None:
        service = RankingService()
        ranked = service.rank(
            [
                _make_candidate(
                    key="old",
                    valid_at=_NOW - timedelta(days=200),
                ),
                _make_candidate(
                    key="new",
                    valid_at=_NOW - timedelta(days=2),
                ),
            ],
            _ctx(),
        )
        assert ranked[0].candidate.candidate_key == "new"

    def test_freshness_preference_shifts_halflife(self) -> None:
        service = RankingService()
        cand_recent = _make_candidate(key="recent", valid_at=_NOW - timedelta(days=5))
        cand_old = _make_candidate(key="old", valid_at=_NOW - timedelta(days=60))

        balanced = service.rank([cand_recent, cand_old], _ctx())
        fresh = service.rank(
            [cand_recent, cand_old], _ctx(freshness_preference="fresh")
        )
        # In 'fresh' mode the gap should *widen* — recent's recency
        # score stays high while old's drops faster.
        balanced_gap = (
            balanced[0].breakdown["recency"] - balanced[1].breakdown["recency"]
        )
        fresh_gap = fresh[0].breakdown["recency"] - fresh[1].breakdown["recency"]
        assert fresh_gap >= balanced_gap


class TestScopeOverlap:
    def test_scope_overlap_dominates_when_only_factor_differs(self) -> None:
        service = RankingService()
        ranked = service.rank(
            [
                _make_candidate(key="loose", scope_overlap=0.1),
                _make_candidate(key="tight", scope_overlap=0.95),
            ],
            _ctx(),
        )
        assert ranked[0].candidate.candidate_key == "tight"


class TestCorroborationBoost:
    def test_more_corroboration_outranks_single_source(self) -> None:
        service = RankingService()
        ranked = service.rank(
            [
                _make_candidate(key="single", corroboration=1),
                _make_candidate(key="triple", corroboration=3),
            ],
            _ctx(),
        )
        assert ranked[0].candidate.candidate_key == "triple"


class TestCoverageQualityDownweight:
    def test_empty_coverage_downweights(self) -> None:
        service = RankingService()
        ranked = service.rank(
            [
                _make_candidate(key="full", coverage_status="complete"),
                _make_candidate(key="empty", coverage_status="empty"),
            ],
            _ctx(),
        )
        assert ranked[0].candidate.candidate_key == "full"
        # Score gap should be material. Under the R3 weighted-sum rule a single
        # factor's contribution is its weight share, so the gap is smaller than
        # under the old geometric mean but still clearly separates the two.
        assert ranked[0].score - ranked[1].score >= 0.05


class TestSemanticSimilarity:
    def test_high_similarity_outranks_low(self) -> None:
        service = RankingService()
        ranked = service.rank(
            [
                _make_candidate(key="far", semantic_similarity=0.2),
                _make_candidate(key="near", semantic_similarity=0.9),
            ],
            _ctx(),
        )
        assert ranked[0].candidate.candidate_key == "near"


class TestNoVeto:
    def test_zero_soft_signal_does_not_collapse_strong_candidate(self) -> None:
        """R3: a zero semantic score must re-rank, never veto.

        A strong, recent, scope-matched, well-corroborated claim with a zero
        semantic-overlap score must still outrank a weak claim that merely has a
        neutral semantic score — the old geometric mean buried the strong one.
        """
        service = RankingService()
        ranked = service.rank(
            [
                _make_candidate(
                    key="strong_zero_semantic",
                    strength="deterministic",
                    scope_overlap=1.0,
                    corroboration=3,
                    semantic_similarity=0.0,
                ),
                _make_candidate(
                    key="weak_neutral",
                    strength="speculative",
                    scope_overlap=0.1,
                    corroboration=1,
                    semantic_similarity=0.5,
                ),
            ],
            _ctx(),
        )
        assert ranked[0].candidate.candidate_key == "strong_zero_semantic"
        # And it is not collapsed to ~0 by the zero factor.
        assert ranked[0].score > 0.4


class TestDeterministicOrdering:
    def test_same_inputs_same_order(self) -> None:
        service = RankingService()
        cands = [
            _make_candidate(key="a", strength="attested"),
            _make_candidate(key="b", strength="stated"),
            _make_candidate(key="c", strength="deterministic"),
        ]
        a = [r.candidate.candidate_key for r in service.rank(cands, _ctx())]
        b = [r.candidate.candidate_key for r in service.rank(cands, _ctx())]
        assert a == b


class TestCandidateFromEdgeRecord:
    def test_extracts_strength_and_valid_at(self) -> None:
        cand = candidate_from_edge_record(
            candidate_key="claim-1",
            edge_properties={
                "evidence_strength": "deterministic",
                "valid_at": "2026-05-15T00:00:00+00:00",
            },
            payload={"a": 1},
            scope_overlap=0.9,
            semantic_similarity=0.5,
            coverage_status="complete",
            corroboration_count=2,
        )
        assert cand.strength == "deterministic"
        assert cand.valid_at == datetime(2026, 5, 15, tzinfo=timezone.utc)
        assert cand.corroboration_count == 2
        assert cand.scope_overlap == 0.9

    def test_unparseable_valid_at_becomes_none(self) -> None:
        cand = candidate_from_edge_record(
            candidate_key="x",
            edge_properties={"evidence_strength": "stated", "valid_at": "garbage"},
        )
        assert cand.valid_at is None


class TestTruncate:
    def test_truncate_keeps_top_n(self) -> None:
        service = RankingService()
        ranked = service.rank(
            [
                _make_candidate(
                    key=str(i), strength="deterministic" if i == 0 else "stated"
                )
                for i in range(5)
            ],
            _ctx(),
        )
        keep = truncate(ranked, max_items=2)
        assert len(keep) == 2

    def test_truncate_zero_returns_empty(self) -> None:
        service = RankingService()
        ranked = service.rank([_make_candidate(key="a")], _ctx())
        assert truncate(ranked, max_items=0) == []
