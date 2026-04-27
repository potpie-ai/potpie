"""Pot status read model: capability matrix, payload conversions, maintenance derivation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domain.context_status import (
    DEFAULT_RESOLVER_CAPABILITIES,
    EventLedgerHealth,
    MaintenanceJob,
    ReconciliationLedgerHealth,
    ResolverCapability,
    SourceCapabilityMatrixEntry,
    StatusSource,
    build_source_capability_matrix,
    derive_maintenance_jobs,
    derive_pot_last_success_at,
    event_ledger_health_to_payload,
    maintenance_jobs_to_payload,
    reconciliation_ledger_health_to_payload,
    resolver_capabilities_to_payload,
    source_capabilities_for,
    source_capability_matrix_to_payload,
    status_source_to_payload,
)

pytestmark = pytest.mark.unit


def _source(**overrides) -> StatusSource:
    base = {
        "source_id": "s1",
        "pot_id": "p1",
        "source_kind": "github_repo",
        "provider": "github",
    }
    base.update(overrides)
    return StatusSource(**base)


# --- Capability matrix -----------------------------------------------------


class TestSourceCapabilitiesFor:
    def test_baseline_matches_default_when_no_resolver(self) -> None:
        caps = source_capabilities_for(_source())
        # references_only is available; everything else is unavailable with reasons.
        by_policy = {c.policy: c for c in caps}
        assert by_policy["references_only"].available is True
        assert by_policy["summary"].available is False
        assert by_policy["summary"].reason is not None

    def test_disabled_sync_flips_available_to_false(self) -> None:
        caps = source_capabilities_for(_source(sync_enabled=False))
        by_policy = {c.policy: c for c in caps}
        assert by_policy["references_only"].available is False
        assert "Source sync is disabled" in (by_policy["references_only"].reason or "")

    def test_resolver_advertised_policy_upgrades_baseline(self) -> None:
        class _Cap:
            def __init__(self, provider, kind, policies):
                self.provider = provider
                self.source_kind = kind
                self.policies = policies

        class _FakeResolver:
            def capabilities(self):
                return [_Cap("github", "github_repo", frozenset({"summary", "verify"}))]

        caps = source_capabilities_for(_source(), resolver=_FakeResolver())
        by_policy = {c.policy: c for c in caps}
        assert by_policy["summary"].available is True
        assert by_policy["summary"].reason is None
        assert by_policy["verify"].available is True
        # snippets stays at its default (not advertised).
        assert by_policy["snippets"].available is False

    def test_resolver_with_no_match_leaves_baseline(self) -> None:
        class _FakeResolver:
            def capabilities(self):
                return []

        caps = source_capabilities_for(_source(), resolver=_FakeResolver())
        # Same as no resolver — baseline.
        assert {c.policy for c in caps} == {c.policy for c in DEFAULT_RESOLVER_CAPABILITIES}

    def test_resolver_capability_map_skips_empty_provider_or_kind(self) -> None:
        # An entry with empty provider should be skipped — the resolver doesn't
        # know what it's advertising.
        class _Cap:
            def __init__(self, provider, kind, policies):
                self.provider = provider
                self.source_kind = kind
                self.policies = policies

        class _FakeResolver:
            def capabilities(self):
                return [_Cap("", "github_repo", frozenset({"summary"}))]

        caps = source_capabilities_for(_source(), resolver=_FakeResolver())
        by_policy = {c.policy: c for c in caps}
        # Skipped entry → baseline.
        assert by_policy["summary"].available is False


class TestBuildSourceCapabilityMatrix:
    def test_dedupes_by_provider_and_kind(self) -> None:
        s1 = _source(source_id="s1")
        s2 = _source(source_id="s2")  # same provider+kind
        s3 = _source(source_id="s3", source_kind="linear_team", provider="linear")
        out = build_source_capability_matrix([s1, s2, s3])
        # Two unique (provider, kind) pairs.
        assert len(out) == 2

    def test_case_insensitive_dedupe(self) -> None:
        s1 = _source(source_id="s1", provider="GitHub", source_kind="GitHub_Repo")
        s2 = _source(source_id="s2", provider="github", source_kind="github_repo")
        assert len(build_source_capability_matrix([s1, s2])) == 1

    def test_each_entry_carries_capabilities(self) -> None:
        out = build_source_capability_matrix([_source()])
        assert out[0].capabilities  # non-empty list of ResolverCapability


# --- Payload conversion ----------------------------------------------------


class TestPayloadConversion:
    def test_resolver_capabilities_to_payload(self) -> None:
        out = resolver_capabilities_to_payload(DEFAULT_RESOLVER_CAPABILITIES)
        assert isinstance(out, list)
        assert all("policy" in c and "available" in c for c in out)

    def test_source_capability_matrix_to_payload(self) -> None:
        entries = [
            SourceCapabilityMatrixEntry(
                provider="github",
                source_kind="github_repo",
                capabilities=list(DEFAULT_RESOLVER_CAPABILITIES),
            )
        ]
        out = source_capability_matrix_to_payload(entries)
        assert out == [
            {
                "provider": "github",
                "source_kind": "github_repo",
                "capabilities": resolver_capabilities_to_payload(DEFAULT_RESOLVER_CAPABILITIES),
            }
        ]

    def test_status_source_to_payload_serializes_datetimes(self) -> None:
        ts = datetime(2026, 4, 27, tzinfo=timezone.utc)
        source = _source(
            last_sync_at=ts,
            last_success_at=ts,
            last_error_at=ts,
            last_verified_at=ts,
        )
        payload = status_source_to_payload(source)
        for k in ("last_sync_at", "last_success_at", "last_error_at", "last_verified_at"):
            assert isinstance(payload[k], str)
            assert "2026-04-27" in payload[k]
        assert "capabilities" in payload

    def test_status_source_to_payload_preserves_none(self) -> None:
        payload = status_source_to_payload(_source())
        # When the datetime is None, the value stays None in the payload.
        assert payload["last_sync_at"] is None

    def test_event_ledger_health_to_payload(self) -> None:
        health = EventLedgerHealth(
            counts={"queued": 1, "processing": 2},
            last_success_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
            recent_errors=[{"k": "v"}],
        )
        payload = event_ledger_health_to_payload(health)
        assert payload["counts"] == {"queued": 1, "processing": 2}
        assert "2026-04-27" in payload["last_success_at"]
        assert payload["recent_errors"] == [{"k": "v"}]

    def test_event_ledger_health_to_payload_with_none_dates(self) -> None:
        payload = event_ledger_health_to_payload(EventLedgerHealth())
        assert payload["last_success_at"] is None
        assert payload["last_error_at"] is None

    def test_reconciliation_ledger_health_to_payload(self) -> None:
        ts = datetime(2026, 4, 27, tzinfo=timezone.utc)
        health = ReconciliationLedgerHealth(
            run_counts={"succeeded": 5},
            step_counts={"applied": 10},
            last_run_success_at=ts,
            last_run_failure_at=None,
            recent_failed_runs=[{"id": 1}],
            stuck_step_samples=[{"step": "x"}],
        )
        payload = reconciliation_ledger_health_to_payload(health)
        assert payload["run_counts"] == {"succeeded": 5}
        assert payload["step_counts"] == {"applied": 10}
        assert "2026-04-27" in payload["last_run_success_at"]
        assert payload["last_run_failure_at"] is None
        assert payload["recent_failed_runs"] == [{"id": 1}]
        assert payload["stuck_step_samples"] == [{"step": "x"}]


# --- Maintenance jobs ------------------------------------------------------


class TestDeriveMaintenanceJobs:
    def test_no_signals_yields_no_jobs(self) -> None:
        jobs = derive_maintenance_jobs(
            event_ledger=EventLedgerHealth(),
            reconciliation=ReconciliationLedgerHealth(),
        )
        assert jobs == []

    def test_event_errors_yield_replay_job(self) -> None:
        jobs = derive_maintenance_jobs(
            event_ledger=EventLedgerHealth(counts={"error": 3}),
            reconciliation=ReconciliationLedgerHealth(),
        )
        assert any(j.action == "events.replay" for j in jobs)
        replay = next(j for j in jobs if j.action == "events.replay")
        assert replay.params["event_count"] == 3
        assert replay.severity == "warning"

    def test_failed_runs_yield_retry_job(self) -> None:
        jobs = derive_maintenance_jobs(
            event_ledger=EventLedgerHealth(),
            reconciliation=ReconciliationLedgerHealth(run_counts={"failed": 2}),
        )
        assert any(j.action == "reconciliation.retry_failed_runs" for j in jobs)

    def test_stuck_steps_yield_retry_job(self) -> None:
        jobs = derive_maintenance_jobs(
            event_ledger=EventLedgerHealth(),
            reconciliation=ReconciliationLedgerHealth(stuck_step_samples=[{"a": 1}]),
        )
        assert any(j.action == "reconciliation.retry_stuck_steps" for j in jobs)

    def test_hard_conflicts_yield_resolve_job(self) -> None:
        jobs = derive_maintenance_jobs(
            event_ledger=EventLedgerHealth(),
            reconciliation=ReconciliationLedgerHealth(),
            open_conflicts=[
                {"auto_resolvable": False, "id": 1},
                {"auto_resolvable": True, "id": 2},  # ignored
            ],
        )
        assert any(j.action == "conflicts.resolve" for j in jobs)
        resolve = next(j for j in jobs if j.action == "conflicts.resolve")
        assert resolve.params["conflict_count"] == 1

    def test_no_hard_conflicts_no_resolve_job(self) -> None:
        # Only auto-resolvable conflicts → no manual job.
        jobs = derive_maintenance_jobs(
            event_ledger=EventLedgerHealth(),
            reconciliation=ReconciliationLedgerHealth(),
            open_conflicts=[{"auto_resolvable": True, "id": 1}],
        )
        assert not any(j.action == "conflicts.resolve" for j in jobs)


class TestMaintenanceJobsPayload:
    def test_payload_shape(self) -> None:
        jobs = [
            MaintenanceJob(action="x", reason="r", severity="warning", params={"a": 1})
        ]
        out = maintenance_jobs_to_payload(jobs)
        assert out == [
            {"action": "x", "reason": "r", "severity": "warning", "params": {"a": 1}}
        ]


# --- derive_pot_last_success_at -------------------------------------------


class TestDerivePotLastSuccessAt:
    def test_no_signals_returns_none(self) -> None:
        assert derive_pot_last_success_at([], EventLedgerHealth()) is None

    def test_picks_max_across_inputs(self) -> None:
        early = datetime(2026, 4, 1, tzinfo=timezone.utc)
        late = datetime(2026, 4, 27, tzinfo=timezone.utc)
        sources = [_source(last_success_at=early)]
        ledger = EventLedgerHealth(last_success_at=late)
        assert derive_pot_last_success_at(sources, ledger) == late

    def test_uses_sync_when_no_explicit_success_and_no_error(self) -> None:
        ts = datetime(2026, 4, 27, tzinfo=timezone.utc)
        sources = [_source(last_sync_at=ts, last_error=None)]
        ledger = EventLedgerHealth()
        assert derive_pot_last_success_at(sources, ledger) == ts

    def test_skips_sync_when_error_present(self) -> None:
        ts = datetime(2026, 4, 27, tzinfo=timezone.utc)
        sources = [_source(last_sync_at=ts, last_error="boom")]
        ledger = EventLedgerHealth()
        assert derive_pot_last_success_at(sources, ledger) is None


def test_resolver_capability_default_state() -> None:
    cap = ResolverCapability(policy="x", available=True)
    assert cap.reason is None
