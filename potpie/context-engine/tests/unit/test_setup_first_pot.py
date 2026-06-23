"""Post-setup first pot naming."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie.context_engine.adapters.inbound.cli.ui import interactive_prompts, setup_ux
from potpie.context_engine.adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from potpie.context_engine.bootstrap.host_wiring import build_host_shell


@pytest.fixture()
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    shell = build_host_shell(backend=InMemoryGraphBackend())
    from potpie.context_engine.adapters.inbound.cli.commands import _common

    _common.set_host(shell)
    return shell


def test_maybe_prompt_first_pot_creates_and_registers_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    monkeypatch.setattr(
        interactive_prompts,
        "prompt_first_pot_name",
        lambda **_k: "my-pot",
    )
    monkeypatch.setattr(
        "potpie.context_engine.adapters.inbound.cli.ui.potpie_logo_anim.play_setup_finish",
        lambda *_a, **_k: None,
    )

    setup_ux._maybe_prompt_first_pot(repo=repo, default_pot_name="default")

    active = host.pots.active_pot()
    assert active is not None
    assert active.name == "my-pot"
    sources = host.pots.list_sources(pot_id=active.pot_id)
    assert any(s.kind == "repo" for s in sources)
