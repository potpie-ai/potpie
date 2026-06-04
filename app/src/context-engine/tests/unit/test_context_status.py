"""Unit tests for the context_status read model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from domain.context_status import (
    DEFAULT_RESOLVER_CAPABILITIES,
    EventLedgerHealth,
    MaintenanceJob,
    ReconciliationLedgerHealth,
    ResolverCapability,
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


def _src(**overrides: Any) -> StatusSource:
    base: dict[str, Any] = {
        "source_id": "src_a",
        "pot_id": "pot_1",
        "source_kind": "repository",
        "provider": "github",
    }
    base.update(overrides)
    return StatusSource(**base)


def test_default_resolver_capabilities_have_only_references_only_available() -> None:
    available = {c.policy: c.available for c in DEFAULT_RESOLVER_CAPABILITIES}
    assert available["references_only"] is True
    assert available["summary"] is False
    assert available["verify"] is False
    assert available["snippets"] is False


def test_resolver_capabilities_to_payload_round_trips() -> None:
    payload = resolver_capabilities_to_payload(
        [ResolverCapability(policy="summary", available=False, reason="not_wired")]
    )
    assert payload == [
        {"policy": "summary", "available": False, "reason": "not_wired"}
    ]


def test_status_source_to_payload_serializes_datetimes() -> None:
    ts = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
    payload = status_source_to_payload(
        _src(last_sync_at=ts, last_success_at=ts, last_verified_at=ts)
    )
    assert payload["last_sync_at"] == ts.isoformat()
    assert payload["last_success_at"] == ts.isoformat()
    assert payload["last_verified_at"] == ts.isoformat()


def test_event_ledger_health_to_payload_handles_empty() -> None:
    out = event_ledger_health_to_payload(EventLedgerHealth())
    assert out == {
        "counts": {},
        "last_success_at": None,
        "last_error_at": None,
        "recent_errors": [],
    }


def test_event_ledger_health_to_payload_serializes_timestamps() -> None:
    ts = datetime(2026, 4, 21, 9, 30, tzinfo=timezone.utc)
    out = event_ledger_health_to_payload(
        EventLedgerHealth(
            counts={"queued": 1, "done": 4, "error": 2},
            last_success_at=ts,
            last_error_at=ts,
            recent_errors=[{"event_id": "e1", "error": "boom"}],
        )
    )
    assert out["counts"] == {"queued": 1, "done": 4, "error": 2}
    assert out["last_success_at"] == ts.isoformat()
    assert out["last_error_at"] == ts.isoformat()
    assert out["recent_errors"] == [{"event_id": "e1", "error": "boom"}]


def test_derive_pot_last_success_at_prefers_most_recent_across_sources_and_ledger() -> None:
    older = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
    newer = older + timedelta(hours=6)
    src_old = _src(source_id="s_old", last_success_at=older)
    src_new = _src(source_id="s_new", last_success_at=newer)
    ledger = EventLedgerHealth(last_success_at=older)
    assert derive_pot_last_success_at([src_old, src_new], ledger) == newer


def test_derive_pot_last_success_at_falls_back_to_last_sync_when_no_error() -> None:
    ts = datetime(2026, 4, 21, 0, 0, tzinfo=timezone.utc)
    src = _src(last_sync_at=ts, last_success_at=None)
    assert derive_pot_last_success_at([src], EventLedgerHealth()) == ts


def test_derive_pot_last_success_at_skips_last_sync_when_source_has_error() -> None:
    ts = datetime(2026, 4, 21, 0, 0, tzinfo=timezone.utc)
    src = _src(last_sync_at=ts, last_error="429 from upstream")
    assert derive_pot_last_success_at([src], EventLedgerHealth()) is None


def test_derive_pot_last_success_at_returns_none_when_nothing_known() -> None:
    assert derive_pot_last_success_at([], EventLedgerHealth()) is None


def test_source_capabilities_for_mirrors_default_when_sync_enabled() -> None:
    caps = source_capabilities_for(_src(sync_enabled=True))
    by_policy = {c.policy: c for c in caps}
    assert by_policy["references_only"].available is True
    assert by_policy["summary"].available is False


def test_source_capabilities_for_disables_all_when_sync_off() -> None:
    caps = source_capabilities_for(_src(sync_enabled=False))
    assert all(c.available is False for c in caps)
    assert all(c.reason == "Source sync is disabled." for c in caps if c.policy == "references_only")


def test_status_source_to_payload_includes_capabilities() -> None:
    payload = status_source_to_payload(_src())
    assert "capabilities" in payload
    policies = {c["policy"] for c in payload["capabilities"]}
    assert policies == {"references_only", "summary", "verify", "snippets"}


class _StubResolverAdvertiser:
    def __init__(self, entries: list[tuple[str, str, frozenset[str]]]) -> None:
        from domain.source_resolution import ResolverCapabilityEntry

        self._entries = [
            ResolverCapabilityEntry(provider=p, source_kind=k, policies=pols)
            for p, k, pols in entries
        ]

    def capabilities(self) -> Any:
        return self._entries


def test_source_capabilities_for_upgrades_via_resolver_advertised_policies() -> None:
    adv = _StubResolverAdvertiser(
        [("github", "repository", frozenset({"summary", "verify"}))]
    )
    caps = source_capabilities_for(
        _src(provider="github", source_kind="repository"),
        resolver=adv,
    )
    by_policy = {c.policy: c for c in caps}
    assert by_policy["summary"].available is True
    assert by_policy["summary"].reason is None
    assert by_policy["verify"].available is True
    # Snippets is NOT advertised by this resolver, so default placeholder remains.
    assert by_policy["snippets"].available is False


def test_source_capabilities_for_resolver_does_not_override_sync_off() -> None:
    adv = _StubResolverAdvertiser(
        [("github", "repository", frozenset({"summary"}))]
    )
    caps = source_capabilities_for(
        _src(provider="github", source_kind="repository", sync_enabled=False),
        resolver=adv,
    )
    assert all(c.available is False for c in caps)


def test_status_source_to_payload_passes_resolver() -> None:
    adv = _StubResolverAdvertiser(
        [("github", "repository", frozenset({"summary"}))]
    )
    payload = status_source_to_payload(
        _src(provider="github", source_kind="repository"),
        resolver=adv,
    )
    summary_cap = next(c for c in payload["capabilities"] if c["policy"] == "summary")
    assert summary_cap["available"] is True


def test_reconciliation_ledger_health_to_payload_empty() -> None:
    out = reconciliation_ledger_health_to_payload(ReconciliationLedgerHealth())
    assert out == {
        "run_counts": {},
        "step_counts": {},
        "last_run_success_at": None,
        "last_run_failure_at": None,
        "recent_failed_runs": [],
        "stuck_step_samples": [],
    }


def test_reconciliation_ledger_health_to_payload_serializes_timestamps() -> None:
    ts = datetime(2026, 4, 21, 9, 0, tzinfo=timezone.utc)
    h = ReconciliationLedgerHealth(
        run_counts={"succeeded": 3, "failed": 1},
        step_counts={"applied": 8, "failed": 2},
        last_run_success_at=ts,
        last_run_failure_at=ts,
        recent_failed_runs=[{"run_id": "r1", "error": "boom"}],
        stuck_step_samples=[{"step_id": "s1", "status": "applying"}],
    )
    out = reconciliation_ledger_health_to_payload(h)
    assert out["run_counts"] == {"succeeded": 3, "failed": 1}
    assert out["step_counts"] == {"applied": 8, "failed": 2}
    assert out["last_run_success_at"] == ts.isoformat()
    assert out["last_run_failure_at"] == ts.isoformat()
    assert out["recent_failed_runs"][0]["run_id"] == "r1"
    assert out["stuck_step_samples"][0]["status"] == "applying"


def test_build_source_capability_matrix_dedupes_by_provider_and_kind() -> None:
    sources = [
        _src(source_id="a", provider="github", source_kind="repository"),
        _src(source_id="b", provider="github", source_kind="repository"),
        _src(source_id="c", provider="linear", source_kind="workspace"),
    ]
    entries = build_source_capability_matrix(sources)
    keys = {(e.provider, e.source_kind) for e in entries}
    assert keys == {("github", "repository"), ("linear", "workspace")}


def test_source_capability_matrix_to_payload_includes_capabilities() -> None:
    sources = [_src(provider="github", source_kind="repository")]
    payload = source_capability_matrix_to_payload(
        build_source_capability_matrix(sources)
    )
    assert payload[0]["provider"] == "github"
    assert payload[0]["source_kind"] == "repository"
    policies = {c["policy"] for c in payload[0]["capabilities"]}
    assert policies == {"references_only", "summary", "verify", "snippets"}


def test_derive_maintenance_jobs_flags_event_errors_and_failed_runs() -> None:
    jobs = derive_maintenance_jobs(
        event_ledger=EventLedgerHealth(counts={"error": 3}),
        reconciliation=ReconciliationLedgerHealth(
            run_counts={"failed": 2},
            stuck_step_samples=[{"step_id": "s1"}],
        ),
        open_conflicts=[{"auto_resolvable": False, "issue_uuid": "i1"}],
    )
    actions = [j.action for j in jobs]
    assert actions == [
        "events.replay",
        "reconciliation.retry_failed_runs",
        "reconciliation.retry_stuck_steps",
        "conflicts.resolve",
    ]
    assert all(j.severity == "warning" for j in jobs)


def test_derive_maintenance_jobs_empty_when_healthy() -> None:
    jobs = derive_maintenance_jobs(
        event_ledger=EventLedgerHealth(counts={"done": 10}),
        reconciliation=ReconciliationLedgerHealth(run_counts={"succeeded": 10}),
        open_conflicts=[],
    )
    assert jobs == []


def test_derive_maintenance_jobs_ignores_auto_resolvable_conflicts() -> None:
    jobs = derive_maintenance_jobs(
        event_ledger=EventLedgerHealth(),
        reconciliation=ReconciliationLedgerHealth(),
        open_conflicts=[{"auto_resolvable": True, "issue_uuid": "i1"}],
    )
    assert jobs == []


def test_maintenance_jobs_to_payload_round_trips() -> None:
    payload = maintenance_jobs_to_payload(
        [MaintenanceJob(action="events.replay", reason="r", severity="warning", params={"n": 1})]
    )
    assert payload == [
        {"action": "events.replay", "reason": "r", "severity": "warning", "params": {"n": 1}}
    ]
