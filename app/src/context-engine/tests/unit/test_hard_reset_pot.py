"""hard_reset_pot use case."""

from unittest.mock import MagicMock, call

from application.use_cases.hard_reset_pot import hard_reset_pot


def test_hard_reset_success() -> None:
    parent = MagicMock()
    episodic = MagicMock()
    episodic.reset_pot.return_value = {"ok": True}
    parent.attach_mock(episodic, "episodic")
    structural = MagicMock()
    structural.reset_pot.return_value = {"ok": True, "entity_deleted": 1, "file_deleted": 0, "node_deleted": 0}
    parent.attach_mock(structural, "structural")
    ledger = MagicMock()
    ledger.delete_all_for_pot.return_value = 3
    parent.attach_mock(ledger, "ledger")

    out = hard_reset_pot(episodic, structural, "pot-1", ledger=ledger)

    assert out["ok"] is True
    assert out["ledger_rows_deleted"] == 3
    assert parent.mock_calls.index(call.ledger.delete_all_for_pot("pot-1")) < parent.mock_calls.index(
        call.episodic.reset_pot("pot-1")
    )
    assert parent.mock_calls.index(call.episodic.reset_pot("pot-1")) < parent.mock_calls.index(
        call.structural.reset_pot("pot-1")
    )


def test_hard_reset_stops_on_episodic_failure() -> None:
    episodic = MagicMock()
    episodic.reset_pot.return_value = {"ok": False, "error": "bad"}
    structural = MagicMock()
    ledger = MagicMock()

    out = hard_reset_pot(episodic, structural, "pot-1", ledger=ledger)

    assert out["ok"] is False
    assert out["error"] == "bad"
    structural.reset_pot.assert_not_called()
    ledger.delete_all_for_pot.assert_called_once_with("pot-1")


def test_hard_reset_without_ledger() -> None:
    episodic = MagicMock()
    episodic.reset_pot.return_value = {"ok": True}
    structural = MagicMock()
    structural.reset_pot.return_value = {"ok": True}

    out = hard_reset_pot(episodic, structural, "pot-1", ledger=None)

    assert out["ok"] is True
    assert "ledger_rows_deleted" not in out


def test_hard_reset_with_reconciliation_ledger() -> None:
    parent = MagicMock()
    episodic = MagicMock()
    episodic.reset_pot.return_value = {"ok": True}
    parent.attach_mock(episodic, "episodic")
    structural = MagicMock()
    structural.reset_pot.return_value = {"ok": True}
    parent.attach_mock(structural, "structural")
    ledger = MagicMock()
    ledger.delete_all_for_pot.return_value = 3
    parent.attach_mock(ledger, "ledger")
    reco = MagicMock()
    reco.delete_all_for_pot.return_value = 5
    parent.attach_mock(reco, "reco")

    out = hard_reset_pot(
        episodic,
        structural,
        "pot-1",
        ledger=ledger,
        reconciliation_ledger=reco,
    )

    assert out["ok"] is True
    assert out["reconciliation_rows_deleted"] == 5
    assert out["ledger_rows_deleted"] == 3
    i_reco = parent.mock_calls.index(call.reco.delete_all_for_pot("pot-1"))
    i_led = parent.mock_calls.index(call.ledger.delete_all_for_pot("pot-1"))
    i_epi = parent.mock_calls.index(call.episodic.reset_pot("pot-1"))
    assert i_reco < i_led < i_epi
