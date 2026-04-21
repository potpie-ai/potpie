"""CLI output helpers."""

import asyncio
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from adapters.inbound.cli.output import (
    DoctorSnapshot,
    configure_error_output,
    configure_cli_logging,
    emit_error,
    print_doctor_report,
    print_search_results,
)
from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter


def test_configure_cli_logging_no_crash() -> None:
    configure_cli_logging(verbose=False)
    assert logging.getLogger("neo4j").level == logging.ERROR
    configure_cli_logging(verbose=True)
    assert logging.getLogger("neo4j").level == logging.DEBUG


def test_print_doctor_json_mode(capsys) -> None:
    snap = DoctorSnapshot(
        context_graph_enabled=True,
        neo4j_effective_set=True,
        neo4j_source="legacy",
        pot_maps_set=False,
        active_pot_id=None,
        potpie_api_key_env=False,
        potpie_stored_token=False,
        potpie_base_url=None,
        potpie_port_hint=None,
        database_url_set=False,
        github_token_set=False,
        summary_lines=["ok"],
    )
    print_doctor_report(snap, as_json=True)
    out = capsys.readouterr().out
    assert '"context_graph_enabled": true' in out
    assert '"neo4j_source": "legacy"' in out
    assert '"potpie_auth_ok": null' in out


def test_print_search_results_with_temporal(capsys) -> None:
    rows = [
        {
            "uuid": "abc",
            "name": "Edge",
            "summary": "fact text",
            "valid_at": "2024-06-01T12:00:00+00:00",
            "invalid_at": None,
            "created_at": "2024-05-01T00:00:00+00:00",
        }
    ]
    print_search_results(rows, as_json=False, with_temporal=True)
    out = capsys.readouterr().out
    assert "2024-06-01" in out
    assert "valid" in out and "expired" in out
    assert "created_at" in out


def test_print_search_results_lifecycle_tag(capsys) -> None:
    rows = [
        {
            "uuid": "x",
            "name": "E",
            "summary": "Roll out tracing",
            "lifecycle_status": "planned",
        }
    ]
    print_search_results(rows, as_json=False)
    out = capsys.readouterr().out
    assert "[planned]" in out
    assert "Roll out tracing" in out


def test_print_search_results_conflict_tag(capsys) -> None:
    rows = [
        {
            "uuid": "e1",
            "name": "Ledger",
            "summary": "Stored in MongoDB",
            "conflict_with_rows": [2],
        }
    ]
    print_search_results(rows, as_json=False)
    out = capsys.readouterr().out
    assert "[!] conflict with row 2" in out
    assert "MongoDB" in out


def test_print_search_results_compact_temporal_default(capsys) -> None:
    rows = [
        {
            "uuid": "abc",
            "name": "Edge",
            "summary": "fact text",
            "valid_at": "2024-06-01T12:00:00+00:00",
            "invalid_at": None,
        }
    ]
    print_search_results(rows, as_json=False, with_temporal=False)
    out = capsys.readouterr().out
    assert "valid" in out and "expired" in out
    assert "created_at" not in out


def test_print_search_results_json_ignores_temporal_flag(capsys) -> None:
    rows = [{"uuid": "u", "valid_at": "2024-01-01T00:00:00+00:00"}]
    print_search_results(rows, as_json=True, with_temporal=False)
    out = capsys.readouterr().out
    assert '"valid_at"' in out


def test_emit_error_json_mode(capsys) -> None:
    configure_error_output(as_json=True)
    try:
        emit_error("Bad thing", "use a better thing", hint="try again")
        err = capsys.readouterr().err
        assert '"ok": false' in err
        assert '"title": "Bad thing"' in err
        assert '"hint": "try again"' in err
    finally:
        configure_error_output(as_json=False)


def test_print_search_results_provenance_line(capsys) -> None:
    rows = [
        {
            "uuid": "edge-1",
            "name": "Fact",
            "summary": "Ledger uses append-only writes",
            "source_refs": ["adr-0042"],
            "reference_time": "2025-04-10T12:00:00+00:00",
            "episode_uuid": "df605b8d-aaaa-bbbb-cccc-ddddeeeeffff",
        }
    ]
    print_search_results(rows, as_json=False, show_provenance=True)
    out = capsys.readouterr().out
    assert "source: adr-0042" in out
    assert "ref: 2025-04-10" in out
    assert "episode: df605b8d" in out


def test_print_search_results_no_provenance_flag(capsys) -> None:
    rows = [
        {
            "uuid": "edge-1",
            "name": "Fact",
            "summary": "text",
            "source_refs": ["adr-0042"],
            "reference_time": "2025-04-10T00:00:00+00:00",
            "episode_uuid": "df605b8d-aaaa-bbbb-cccc-ddddeeeeffff",
        }
    ]
    print_search_results(rows, as_json=False, show_provenance=False)
    out = capsys.readouterr().out
    assert "source: adr-0042" not in out
    assert "ref:" not in out


def test_print_search_results_human_uses_cards(capsys) -> None:
    rows = [
        {
            "uuid": "abc",
            "name": "Edge",
            "fact": "A long but readable fact that should render inside a result card.",
        }
    ]
    print_search_results(rows, as_json=False)
    out = capsys.readouterr().out
    assert "1. Edge" in out
    assert "uuid:" in out


def test_graphiti_failure_reason_disabled() -> None:
    settings = MagicMock()
    settings.is_enabled.return_value = False
    settings.neo4j_uri.return_value = None
    settings.neo4j_user.return_value = None
    settings.neo4j_password.return_value = None
    adapter = GraphitiEpisodicAdapter(settings)
    assert adapter.failure_reason() == "context_graph_disabled"


def test_graphiti_maintenance_methods_create_client_inside_sync_run(monkeypatch) -> None:
    adapter = object.__new__(GraphitiEpisodicAdapter)
    adapter._enabled = True
    adapter._init_error = None
    adapter._thread_local = threading.local()

    def get_graphiti_inside_running_loop():
        asyncio.get_running_loop()
        return SimpleNamespace(driver=object())

    async def relabel_nodes_from_edges(driver, pot_id):
        return {"ok": True, "pot_id": pot_id, "driver": driver is not None}

    async def classify_modified_edges_for_group(driver, pot_id, *, dry_run=True):
        return {
            "ok": True,
            "pot_id": pot_id,
            "driver": driver is not None,
            "dry_run": dry_run,
        }

    monkeypatch.setattr(adapter, "_get_graphiti", get_graphiti_inside_running_loop)
    monkeypatch.setattr(
        "adapters.outbound.graphiti.apply_canonical_labels.relabel_nodes_from_edges",
        relabel_nodes_from_edges,
    )
    monkeypatch.setattr(
        "adapters.outbound.graphiti.classify_modified_edges.classify_modified_edges_for_group",
        classify_modified_edges_for_group,
    )

    assert adapter.relabel_nodes_from_edges_for_pot("pot-1") == {
        "ok": True,
        "pot_id": "pot-1",
        "driver": True,
    }
    assert adapter.classify_modified_edges_for_pot("pot-1", dry_run=False) == {
        "ok": True,
        "pot_id": "pot-1",
        "driver": True,
        "dry_run": False,
    }
