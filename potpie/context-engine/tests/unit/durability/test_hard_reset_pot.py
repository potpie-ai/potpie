"""hard_reset_pot use case."""

from unittest.mock import MagicMock, call
from potpie_context_engine.core.graph_journal import JournalState

from potpie_context_engine.application.use_cases.hard_reset_pot import hard_reset_pot


def test_hard_reset_success() -> None:
    parent = MagicMock()
    backend = MagicMock()
    backend.mutation.reset_pot.return_value = {
        "pot_id": "pot-1",
        "ok": True,
        "episodic": {"ok": True},
        "structural": {
            "ok": True,
            "entity_deleted": 1,
            "file_deleted": 0,
            "node_deleted": 0,
        },
    }
    parent.attach_mock(backend, "backend")
    ledger = MagicMock()
    ledger.delete_all_for_pot.return_value = 3
    parent.attach_mock(ledger, "ledger")

    out = hard_reset_pot(backend, "pot-1", ledger=ledger)

    assert out["ok"] is True
    assert out["ledger_rows_deleted"] == 3
    assert parent.mock_calls.index(
        call.ledger.delete_all_for_pot("pot-1")
    ) < parent.mock_calls.index(call.backend.mutation.reset_pot("pot-1"))


def test_hard_reset_returns_graph_failure() -> None:
    backend = MagicMock()
    backend.mutation.reset_pot.return_value = {
        "pot_id": "pot-1",
        "ok": False,
        "error": "bad",
    }
    ledger = MagicMock()

    out = hard_reset_pot(backend, "pot-1", ledger=ledger)

    assert out["ok"] is False
    assert out["error"] == "bad"
    ledger.delete_all_for_pot.assert_called_once_with("pot-1")


def test_hard_reset_without_ledger() -> None:
    backend = MagicMock()
    backend.mutation.reset_pot.return_value = {"pot_id": "pot-1", "ok": True}

    out = hard_reset_pot(backend, "pot-1", ledger=None)

    assert out["ok"] is True
    assert "ledger_rows_deleted" not in out


def test_hard_reset_with_reconciliation_ledger() -> None:
    parent = MagicMock()
    backend = MagicMock()
    backend.mutation.reset_pot.return_value = {"pot_id": "pot-1", "ok": True}
    parent.attach_mock(backend, "backend")
    ledger = MagicMock()
    ledger.delete_all_for_pot.return_value = 3
    parent.attach_mock(ledger, "ledger")
    reco = MagicMock()
    reco.delete_all_for_pot.return_value = 5
    parent.attach_mock(reco, "reco")

    out = hard_reset_pot(
        backend,
        "pot-1",
        ledger=ledger,
        reconciliation_ledger=reco,
    )

    assert out["ok"] is True
    assert out["reconciliation_rows_deleted"] == 5
    assert out["ledger_rows_deleted"] == 3
    i_reco = parent.mock_calls.index(call.reco.delete_all_for_pot("pot-1"))
    i_led = parent.mock_calls.index(call.ledger.delete_all_for_pot("pot-1"))
    i_reset = parent.mock_calls.index(call.backend.mutation.reset_pot("pot-1"))
    assert i_reco < i_led < i_reset


def test_hard_reset_purges_resources_after_successful_graph_reset():
    parent = MagicMock()
    backend = MagicMock()
    backend.mutation.reset_pot.return_value = {"pot_id": "pot-1", "ok": True}
    parent.attach_mock(backend, "backend")
    resources = MagicMock()
    resources.purge_pot.return_value = True
    parent.attach_mock(resources, "resources")

    out = hard_reset_pot(backend, "pot-1", resources=resources)

    assert out["ok"] is True
    assert out["resources_purged"] is True
    assert parent.mock_calls.index(
        call.backend.mutation.reset_pot("pot-1")
    ) < parent.mock_calls.index(call.resources.purge_pot("pot-1"))


def test_hard_reset_skips_resource_purge_when_graph_reset_fails():
    backend = MagicMock()
    backend.mutation.reset_pot.return_value = {
        "pot_id": "pot-1",
        "ok": False,
        "error": "bad",
    }
    resources = MagicMock()

    out = hard_reset_pot(backend, "pot-1", resources=resources)

    assert out["ok"] is False
    resources.purge_pot.assert_not_called()
    assert "resources_purged" not in out


def test_hard_reset_is_refused_before_any_ledger_row_while_journaling() -> None:
    """The graph reset refuses a pot whose journal is capturing; checking first
    keeps the ledgers from being cleared under a graph that stays."""
    backend = MagicMock()
    backend.journal.journal_state.return_value = JournalState(
        pot_id="pot-1", generation="g1"
    )
    ledger = MagicMock()
    reconciliation_ledger = MagicMock()
    resources = MagicMock()

    out = hard_reset_pot(
        backend,
        "pot-1",
        ledger=ledger,
        reconciliation_ledger=reconciliation_ledger,
        resources=resources,
    )

    assert out["ok"] is False
    assert out["error"] == "journal_capture_active"
    ledger.delete_all_for_pot.assert_not_called()
    reconciliation_ledger.delete_all_for_pot.assert_not_called()
    backend.mutation.reset_pot.assert_not_called()
    resources.purge_pot.assert_not_called()
