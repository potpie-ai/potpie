"""BeliefDeriver + coverage-gap envelope confidence (rebuild plan P2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from domain.belief import (
    HIGH,
    MEDIUM,
    LOW,
    UNKNOWN,
    ClaimRecord,
    CoverageReport,
    coverage_cap,
    decay_weight,
    derive_belief,
    envelope_confidence,
    label_for_score,
    score_object,
    source_authority,
)


NOW = datetime(2026, 5, 20, tzinfo=timezone.utc)


def _claim(
    *,
    subject_key: str = "service:auth-svc",
    predicate: str = "DEPENDS_ON",
    object_key: str = "service:users-svc",
    source_system: str = "k8s-scanner",
    source_ref: str | None = "k8s/auth/networkpolicy.yaml",
    strength: str = "deterministic",
    observed_at: datetime | None = None,
    valid_at: datetime | None = None,
    verified_count: int = 0,
) -> ClaimRecord:
    return ClaimRecord(
        subject_key=subject_key,
        predicate=predicate,
        object_key=object_key,
        source_system=source_system,
        source_ref=source_ref,
        evidence_strength=strength,
        observed_at=observed_at or NOW,
        valid_at=valid_at or NOW,
        verified_count=verified_count,
    )


class TestDecayWeight:
    """Decay is linear; corroboration extends the effective TTL."""

    def test_fresh_claim_decay_is_full(self) -> None:
        w = decay_weight(observed_at=NOW, now=NOW, predicate="DEPENDS_ON")
        assert w == pytest.approx(1.0)

    def test_stale_claim_decays_to_zero(self) -> None:
        very_old = NOW - timedelta(days=365 * 5)
        w = decay_weight(observed_at=very_old, now=NOW, predicate="DEPENDS_ON")
        assert w == 0.0

    def test_corroboration_extends_ttl(self) -> None:
        observed = NOW - timedelta(days=30)
        single = decay_weight(
            observed_at=observed,
            now=NOW,
            predicate="DEPENDS_ON",
            corroboration_count=1,
        )
        triple = decay_weight(
            observed_at=observed,
            now=NOW,
            predicate="DEPENDS_ON",
            corroboration_count=3,
        )
        assert triple > single

    def test_missing_observation_treated_as_fresh(self) -> None:
        w = decay_weight(observed_at=None, now=NOW, predicate="DEPENDS_ON")
        assert w == 1.0


class TestSourceAuthority:
    def test_high_trust_sources_outweigh_default(self) -> None:
        assert source_authority("k8s-scanner") > 1.0
        assert source_authority("codeowners-scanner") > 1.0

    def test_ambient_sources_downweighted(self) -> None:
        assert source_authority("slack-message") < 1.0
        assert source_authority("pr-body-llm") < 1.0

    def test_unknown_source_neutral(self) -> None:
        assert source_authority("never-seen-this-before") == 1.0

    def test_none_source_neutral(self) -> None:
        assert source_authority(None) == 1.0


class TestScoreObject:
    """The scoring formula should produce intuitive label outcomes."""

    def test_single_deterministic_claim_is_high(self) -> None:
        claims = [_claim()]
        score = score_object(claims, now=NOW)
        assert label_for_score(score) == HIGH

    def test_two_sources_lift_label(self) -> None:
        single = score_object([_claim(strength="attested")], now=NOW)
        corroborated = score_object(
            [
                _claim(strength="attested", source_system="codeowners-scanner"),
                _claim(strength="attested", source_system="pr-body-llm"),
            ],
            now=NOW,
        )
        assert corroborated > single

    def test_stale_inferred_claim_is_unknown(self) -> None:
        old = NOW - timedelta(days=365 * 10)
        claims = [
            _claim(
                strength="inferred",
                source_system="pr-body-llm",
                observed_at=old,
                valid_at=old,
            )
        ]
        score = score_object(claims, now=NOW)
        assert label_for_score(score) == UNKNOWN

    def test_verification_bumps_label(self) -> None:
        # An attested + 1 source claim sits at the high/medium boundary; a
        # verification claim attached to its target should bump it to HIGH.
        claims = [_claim(strength="attested")]
        base = score_object(claims, now=NOW)
        verified = score_object(claims, now=NOW, verified_count=1)
        assert verified > base


class TestDeriveBelief:
    """End-to-end belief derivation over a set of claims."""

    def test_winner_picks_higher_score(self) -> None:
        claims = [
            # users-svc: corroborated, recent, deterministic — clear winner
            _claim(object_key="service:users-svc", source_system="k8s-scanner"),
            _claim(
                object_key="service:users-svc",
                source_system="codeowners-scanner",
            ),
            # billing-svc: single inferred LLM claim — also live, but weaker
            _claim(
                object_key="service:billing-svc",
                source_system="pr-body-llm",
                strength="inferred",
            ),
        ]
        belief = derive_belief(
            claims,
            subject_key="service:auth-svc",
            predicate="DEPENDS_ON",
            now=NOW,
        )
        assert belief.winner is not None
        assert belief.winner.object_key == "service:users-svc"
        # Two distinct sources for the winner.
        assert belief.winner.corroboration_count == 2
        # Both candidates surface so the agent sees the disagreement.
        assert {c.object_key for c in belief.candidates} == {
            "service:users-svc",
            "service:billing-svc",
        }

    def test_equal_recency_conflict_no_winner(self) -> None:
        # Two equally-strong claims for different objects at the same valid_at:
        # the deriver refuses to pick.
        common_time = NOW
        claims = [
            _claim(
                object_key="service:billing-svc",
                source_system="k8s-scanner",
                observed_at=common_time,
                valid_at=common_time,
            ),
            _claim(
                object_key="service:invoicing-svc",
                source_system="codeowners-scanner",
                observed_at=common_time,
                valid_at=common_time,
            ),
        ]
        belief = derive_belief(
            claims,
            subject_key="service:auth-svc",
            predicate="DEPENDS_ON",
            now=NOW,
        )
        assert belief.conflict_open is True
        assert belief.winner is None
        assert len(belief.candidates) == 2

    def test_no_claims_returns_empty_belief(self) -> None:
        belief = derive_belief(
            [],
            subject_key="service:auth-svc",
            predicate="DEPENDS_ON",
            now=NOW,
        )
        assert belief.winner is None
        assert belief.candidates == ()
        assert belief.conflict_open is False

    def test_unrelated_claims_filtered(self) -> None:
        claims = [
            _claim(predicate="OWNED_BY", object_key="person:alice"),
            _claim(predicate="DEPENDS_ON", object_key="service:users-svc"),
        ]
        belief = derive_belief(
            claims,
            subject_key="service:auth-svc",
            predicate="DEPENDS_ON",
            now=NOW,
        )
        assert belief.winner is not None
        assert belief.winner.object_key == "service:users-svc"
        # The OWNED_BY claim should not appear among candidates.
        assert all(c.object_key != "person:alice" for c in belief.candidates)


class TestCoverageGap:
    """F5: the envelope cannot read 'high' when planned families came back empty."""

    def test_complete_coverage_does_not_cap(self) -> None:
        report = CoverageReport(
            planned_families=frozenset({"topology", "owners"}),
            returning_families=frozenset({"topology", "owners"}),
        )
        assert coverage_cap(report) == HIGH

    def test_partial_coverage_caps_at_medium(self) -> None:
        report = CoverageReport(
            planned_families=frozenset({"topology", "owners", "policies"}),
            returning_families=frozenset({"topology", "owners"}),
        )
        # 2/3 ≈ partial → cap = medium
        assert coverage_cap(report) == MEDIUM

    def test_empty_coverage_caps_at_unknown(self) -> None:
        report = CoverageReport(
            planned_families=frozenset({"topology", "owners"}),
            returning_families=frozenset(),
        )
        assert coverage_cap(report) == UNKNOWN

    def test_envelope_confidence_capped_down_by_coverage(self) -> None:
        # All facts read high, but coverage is sparse → envelope is low/medium.
        labels = [HIGH, HIGH, HIGH]
        report = CoverageReport(
            planned_families=frozenset({"topology", "owners", "policies"}),
            returning_families=frozenset({"topology"}),
        )
        # 1/3 ≈ sparse → cap = low; min(high, low) = low
        assert envelope_confidence(labels, report) == LOW

    def test_envelope_confidence_complete_coverage_returns_best_label(self) -> None:
        labels = [HIGH, MEDIUM, LOW]
        report = CoverageReport(
            planned_families=frozenset({"a", "b"}),
            returning_families=frozenset({"a", "b"}),
        )
        assert envelope_confidence(labels, report) == HIGH

    def test_no_planned_families_treated_as_complete(self) -> None:
        # If the planner didn't expect specific families (degenerate
        # case), coverage doesn't cap.
        report = CoverageReport()
        assert envelope_confidence([HIGH], report) == HIGH
