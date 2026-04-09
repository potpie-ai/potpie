"""CLI output helpers."""

import logging
from unittest.mock import MagicMock

from adapters.inbound.cli.output import (
    DoctorSnapshot,
    configure_cli_logging,
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
    assert "valid_at" in out or "2024-05-01" in out


def test_print_search_results_json_ignores_temporal_flag(capsys) -> None:
    rows = [{"uuid": "u", "valid_at": "2024-01-01T00:00:00+00:00"}]
    print_search_results(rows, as_json=True, with_temporal=False)
    out = capsys.readouterr().out
    assert '"valid_at"' in out


def test_graphiti_failure_reason_disabled() -> None:
    settings = MagicMock()
    settings.is_enabled.return_value = False
    settings.neo4j_uri.return_value = None
    settings.neo4j_user.return_value = None
    settings.neo4j_password.return_value = None
    adapter = GraphitiEpisodicAdapter(settings)
    assert adapter.failure_reason() == "context_graph_disabled"
