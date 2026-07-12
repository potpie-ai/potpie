from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.composition import build_engine_components
from potpie_context_engine.domain.ports.services.graph_service import GraphReadRequest

pytestmark = pytest.mark.unit

FIXTURE_HOME = Path(__file__).parent / "fixtures" / "current_engine_home"


def test_current_engine_home_fixture_loads_without_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "engine-home"
    shutil.copytree(FIXTURE_HOME, home)
    before = {path.name: path.read_bytes() for path in home.iterdir() if path.is_file()}
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))

    host = build_engine_components(
        backend=EmbeddedGraphBackend(home=home),
        data_dir=home,
    )

    active = host.pots.active_pot()
    sources = host.pots.list_sources(pot_id="pot_existing")
    infra = host.graph.read(
        GraphReadRequest(
            pot_id="pot_existing",
            subgraph="infra_topology",
            view="service_neighborhood",
            scope={"service": "payments-api"},
            depth=2,
        )
    )
    timeline = host.graph.read(
        GraphReadRequest(
            pot_id="pot_existing",
            subgraph="recent_changes",
            view="timeline",
            limit=10,
        )
    )
    cursor = host.ledger.cursors.get(pot_id="pot_existing", source_id="github")

    assert active is not None
    assert (active.pot_id, active.name, active.active) == (
        "pot_existing",
        "existing-project",
        True,
    )
    assert [(source.source_id, source.kind, source.location) for source in sources] == [
        (
            "src_existing",
            "repo",
            "https://github.com/potpie-ai/potpie.git",
        )
    ]
    assert host.pots.repo_default(repo="https://github.com/potpie-ai/potpie.git") == (
        "pot_existing"
    )
    assert infra.items
    infra_payload = infra.to_dict()
    assert "service:ledger-api" in json.dumps(infra_payload)
    assert infra_payload["source_refs"] == ["repo:manifest"]
    assert timeline.items
    assert "deployed payments api version 2" in json.dumps(timeline.to_dict())
    assert cursor is not None and cursor.token == "7"
    assert {
        path.name: path.read_bytes() for path in home.iterdir() if path.is_file()
    } == before
