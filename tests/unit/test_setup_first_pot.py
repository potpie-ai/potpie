"""Post-setup first pot naming."""

from __future__ import annotations

from pathlib import Path

import pytest

from potpie.cli.ui import interactive_prompts, setup_ux


@pytest.fixture()
def runtime(root_test_runtime):
    from potpie.cli.commands import _common

    _common.set_cli_runtime(root_test_runtime)
    return root_test_runtime


def test_maybe_prompt_first_pot_creates_and_registers_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    monkeypatch.setattr(
        interactive_prompts,
        "prompt_first_pot_name",
        lambda **_k: "my-pot",
    )
    monkeypatch.setattr(
        "potpie.cli.ui.potpie_logo_anim.play_setup_finish",
        lambda *_a, **_k: None,
    )

    setup_ux._maybe_prompt_first_pot(repo=repo, default_pot_name="default")

    from potpie.runtime.async_bridge import run_sync
    from potpie.runtime.contracts import PotInfoRequest, SourceListRequest

    active = run_sync(lambda: runtime.engine.pots.info(PotInfoRequest()))
    assert active is not None
    assert active.name == "my-pot"
    sources = run_sync(
        lambda: runtime.engine.sources.list(SourceListRequest(pot_id=active.pot_id))
    ).items
    assert any(s.kind == "repo" for s in sources)
