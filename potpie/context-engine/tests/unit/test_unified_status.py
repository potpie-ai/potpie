"""Unified status / doctor report."""

import json

from potpie_context_engine.adapters.inbound.cli.ui.output import DoctorSnapshot, print_unified_status_report


def test_unified_status_json(capsys) -> None:
    snap = DoctorSnapshot(
        context_graph_enabled=True,
        neo4j_effective_set=False,
        pot_maps_set=False,
        active_pot_id=None,
        potpie_api_key_env=True,
        potpie_stored_token=False,
        potpie_base_url="http://127.0.0.1:8001",
        potpie_port_hint=None,
        database_url_set=False,
        github_token_set=False,
        potpie_health_ok=True,
        potpie_auth_ok=True,
        summary_lines=[],
    )
    print_unified_status_report(
        snap,
        as_json=True,
        quick=False,
        pot_id="pot-1",
        pot_status={"ready": True},
    )
    out = json.loads(capsys.readouterr().out)
    assert out["doctor"]["potpie_health_ok"] is True
    assert out["pot_id"] == "pot-1"
    assert out["pot_status"]["ready"] is True
