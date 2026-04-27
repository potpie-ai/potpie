"""CLI ingest output for reconciliation_rejected."""

from adapters.inbound.cli.output import print_ingest_result


def test_print_ingest_rejection_plain(capsys) -> None:
    print_ingest_result(
        {
            "status": "reconciliation_rejected",
            "event_id": "3d6ab2c2-4f43-4a5f-8b3c-12bee7386613",
            "episode_uuid": None,
            "errors": [
                {"entity": "adr:0042", "issue": "unknown canonical labels: ADR"},
                {"entity": "DECIDED_BY", "issue": "unknown canonical edge type"},
            ],
            "downgrades": [],
        },
        as_json=False,
    )
    err = capsys.readouterr().err
    assert "reconciliation" in err.lower()
    assert "3d6ab2c2" in err
    assert "adr:0042" in err
    assert "unknown canonical labels" in err
    assert "DECIDED_BY" in err
    assert "Hint:" in err and "ontology" in err


def test_print_ingest_success_shows_downgrade_hint(capsys) -> None:
    print_ingest_result(
        {
            "status": "applied",
            "episode_uuid": "ep-1",
            "event_id": "e1",
            "downgrades": [{"kind": "edge_type", "from": "X", "to": "RELATED_TO"}],
        },
        as_json=False,
    )
    out = capsys.readouterr().out
    assert "ep-1" in out
    assert "downgrade" in out.lower()


def test_print_ingest_rejection_json(capsys) -> None:
    payload = {
        "status": "reconciliation_rejected",
        "event_id": "e1",
        "errors": [{"entity": "x", "issue": "y"}],
        "downgrades": [],
    }
    print_ingest_result(payload, as_json=True)
    out = capsys.readouterr().out
    assert '"status": "reconciliation_rejected"' in out
    assert "e1" in out
