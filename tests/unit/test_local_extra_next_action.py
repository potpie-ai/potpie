"""A missing FalkorDBLite names a repair that works on the platform at hand."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import sys

import pytest

from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
    _local_extra_next_action,
    build_falkordb_graph,
)
from potpie_context_engine.core.errors import CapabilityNotImplemented


def test_windows_is_not_told_to_install_an_extra_that_skips_it() -> None:
    text = _local_extra_next_action("Windows")

    assert "pip install" not in text
    assert "CONTEXT_ENGINE_BACKEND=embedded" in text
    assert "Windows" in text


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
def test_elsewhere_the_extra_is_the_repair(system: str) -> None:
    text = _local_extra_next_action(system)

    assert "pip install 'potpie-context-engine[local]'" in text
    assert "CONTEXT_ENGINE_BACKEND=embedded" in text


class _LiteSettings:
    def falkordb_graph_name(self) -> str:
        return "context"

    def falkordb_mode(self) -> str:
        return "lite"

    def falkordb_lite_path(self) -> str:
        raise AssertionError("the store path is never reached without the driver")


def test_a_missing_driver_is_a_typed_gap_with_the_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "redislite", None)
    monkeypatch.setitem(sys.modules, "redislite.falkordb_client", None)

    with pytest.raises(CapabilityNotImplemented) as raised:
        build_falkordb_graph(_LiteSettings())

    assert raised.value.capability == "graph.falkordb_lite.embedded_store"
    assert raised.value.recommended_next_action == _local_extra_next_action()
