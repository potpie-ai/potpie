"""CLI wiring for ``potpie why`` / ``potpie lineage capture``."""

from __future__ import annotations

from types import SimpleNamespace

from potpie.cli.commands import lineage


def test_service_treats_resolve_pot_id_as_a_string(monkeypatch) -> None:
    monkeypatch.setattr(lineage, "get_root_runtime", lambda: object())
    monkeypatch.setattr(
        lineage, "resolve_pot_id", lambda host, pot: "acme/demo-pot-id"
    )
    monkeypatch.setattr(lineage, "_graph_recorder", lambda pot: None)
    monkeypatch.setattr(
        lineage.LineageService,
        "for_pot",
        classmethod(lambda cls, pot_id, **kwargs: SimpleNamespace(pot_id=pot_id)),
    )

    pot_id, service = lineage._service(None, fail_open=True)

    assert pot_id == "acme/demo-pot-id"
    assert service.pot_id == "acme/demo-pot-id"
