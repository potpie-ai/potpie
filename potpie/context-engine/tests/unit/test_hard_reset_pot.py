"""hard_reset_pot use case."""

from unittest.mock import MagicMock, call

from application.use_cases.hard_reset_pot import hard_reset_pot


def test_hard_reset_success() -> None:
    parent = MagicMock()
    context_graph = MagicMock()
    context_graph.reset_pot.return_value = {
        "pot_id": "pot-1",
        "ok": True,
        "episodic": {"ok": True},
        "structural": {"ok": True, "entity_deleted": 1, "file_deleted": 0, "node_deleted": 0},
    }
    parent.attach_mock(context_graph, "context_graph")
    ledger = MagicMock()
    ledger.delete_all_for_pot.return_value = 3
    parent.attach_mock(ledger, "ledger")

    out = hard_reset_pot(context_graph, "pot-1", ledger=ledger)

    assert out["ok"] is True
    assert out["ledger_rows_deleted"] == 3
    assert parent.mock_calls.index(call.ledger.delete_all_for_pot("pot-1")) < parent.mock_calls.index(
        call.context_graph.reset_pot("pot-1")
    )


def test_hard_reset_returns_graph_failure() -> None:
    context_graph = MagicMock()
    context_graph.reset_pot.return_value = {"pot_id": "pot-1", "ok": False, "error": "bad"}
    ledger = MagicMock()

    out = hard_reset_pot(context_graph, "pot-1", ledger=ledger)

    assert out["ok"] is False
    assert out["error"] == "bad"
    ledger.delete_all_for_pot.assert_called_once_with("pot-1")


def test_hard_reset_without_ledger() -> None:
    context_graph = MagicMock()
    context_graph.reset_pot.return_value = {"pot_id": "pot-1", "ok": True}

    out = hard_reset_pot(context_graph, "pot-1", ledger=None)

    assert out["ok"] is True
    assert "ledger_rows_deleted" not in out


def test_hard_reset_with_reconciliation_ledger() -> None:
    parent = MagicMock()
    context_graph = MagicMock()
    context_graph.reset_pot.return_value = {"pot_id": "pot-1", "ok": True}
    parent.attach_mock(context_graph, "context_graph")
    ledger = MagicMock()
    ledger.delete_all_for_pot.return_value = 3
    parent.attach_mock(ledger, "ledger")
    reco = MagicMock()
    reco.delete_all_for_pot.return_value = 5
    parent.attach_mock(reco, "reco")

    out = hard_reset_pot(
        context_graph,
        "pot-1",
        ledger=ledger,
        reconciliation_ledger=reco,
    )

    assert out["ok"] is True
    assert out["reconciliation_rows_deleted"] == 5
    assert out["ledger_rows_deleted"] == 3
    i_reco = parent.mock_calls.index(call.reco.delete_all_for_pot("pot-1"))
    i_led = parent.mock_calls.index(call.ledger.delete_all_for_pot("pot-1"))
    i_reset = parent.mock_calls.index(call.context_graph.reset_pot("pot-1"))
    assert i_reco < i_led < i_reset
