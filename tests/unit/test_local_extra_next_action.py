"""The repair for a missing local graph driver is not the same sentence everywhere."""

from __future__ import annotations

from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
    _local_extra_next_action,
)


def test_windows_is_not_told_to_install_an_extra_it_already_has() -> None:
    """``potpie[local]`` is marked out of Windows entirely, so `pip install
    'potpie[local]'` there installs nothing and leads straight back here."""
    text = _local_extra_next_action("Windows")
    assert "pip install" not in text
    assert "potpie host use managed" in text
    assert "Windows" in text


def test_elsewhere_the_extra_is_the_repair() -> None:
    text = _local_extra_next_action("Darwin")
    assert "pip install 'potpie[local]'" in text
    assert "potpie host use managed" in text
