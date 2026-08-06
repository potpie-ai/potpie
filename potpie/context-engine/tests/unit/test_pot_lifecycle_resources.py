"""P5 — pot teardown purges the resource store (R8)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
    LocalResourceStore,
    pot_dir_name,
)
from potpie_context_engine.application.services.pot_management import (
    LocalPotManagementService,
)
from potpie_context_engine.application.use_cases.hard_reset_pot import hard_reset_pot
from potpie_context_engine.testing import write_import_directory

pytestmark = pytest.mark.unit

POT_NAME = "lifecycle"
DOC = "q3-review"


def _seed_document(store: LocalResourceStore, pot_id: str, root: Path) -> Path:
    source = write_import_directory(
        root,
        [
            {
                "slug": "capacity",
                "title": "Capacity",
                "summary": "headroom",
                "ordinal": 0,
                "content_hash": "capacity-1",
                "chunks": [{"label": "opening", "text": "alpha"}],
            }
        ],
        source_ref="file:///q3.pdf",
        source_kind="pdf",
    )
    store.import_dir(pot_id=pot_id, slug=DOC, source_dir=source)
    return store.home / "resources" / pot_dir_name(pot_id)


def test_reset_pot_leaves_no_files_under_the_pot_resource_tree(tmp_path):
    """Characterization: after reset, ``<home>/resources/<pot_dir>/`` is gone."""
    home = tmp_path / "home"
    pot_store = LocalPotStore(home=home)
    resources = LocalResourceStore(home=home)
    pots = LocalPotManagementService(
        store=pot_store,
        backend=InMemoryGraphBackend(),
        resources=resources,
    )
    pot = pots.create_pot(name=POT_NAME, use=True)
    pot_root = _seed_document(resources, pot.pot_id, tmp_path / "import")
    assert pot_root.is_dir()
    assert any(pot_root.rglob("*.txt"))

    pots.reset_pot(ref=pot.pot_id, confirm=True)

    assert not pot_root.exists()
    assert resources.purge_pot(pot.pot_id) is False


def test_archive_pot_tears_down_graph_and_resources(tmp_path):
    home = tmp_path / "home"
    pot_store = LocalPotStore(home=home)
    resources = LocalResourceStore(home=home)
    pots = LocalPotManagementService(
        store=pot_store,
        backend=InMemoryGraphBackend(),
        resources=resources,
    )
    pot = pots.create_pot(name=POT_NAME, use=True)
    pot_root = _seed_document(resources, pot.pot_id, tmp_path / "import")

    archived = pots.archive_pot(ref=pot.pot_id)

    assert archived.pot.archived is True
    assert archived.resources_purged is True
    assert not pot_root.exists()


def test_teardown_reports_no_purge_on_a_pot_that_held_no_resources(tmp_path):
    """P1-2: ``resources_purged`` is the store's answer, not a literal True."""
    home = tmp_path / "home"
    pots = LocalPotManagementService(
        store=LocalPotStore(home=home),
        backend=InMemoryGraphBackend(),
        resources=LocalResourceStore(home=home),
    )
    pot = pots.create_pot(name=POT_NAME, use=True)

    assert pots.reset_pot(ref=pot.pot_id, confirm=True).resources_purged is False
    assert pots.archive_pot(ref=pot.pot_id).resources_purged is False


def test_teardown_reports_unknown_purge_when_no_resource_store_is_wired(tmp_path):
    """Nothing to purge is not the same answer as purged nothing."""
    home = tmp_path / "home"
    pots = LocalPotManagementService(
        store=LocalPotStore(home=home),
        backend=InMemoryGraphBackend(),
    )
    pot = pots.create_pot(name=POT_NAME, use=True)

    assert pots.reset_pot(ref=pot.pot_id, confirm=True).resources_purged is None


def test_remove_source_does_not_touch_resources(tmp_path):
    """Decision: source remove is registration-only (ignore resources)."""
    home = tmp_path / "home"
    pot_store = LocalPotStore(home=home)
    resources = LocalResourceStore(home=home)
    pots = LocalPotManagementService(
        store=pot_store,
        backend=InMemoryGraphBackend(),
        resources=resources,
    )
    pot = pots.create_pot(name=POT_NAME, use=True)
    source = pots.add_source(
        pot_id=pot.pot_id, kind="repo", location="github.com/acme/x"
    )
    pot_root = _seed_document(resources, pot.pot_id, tmp_path / "import")

    pots.remove_source(pot_id=pot.pot_id, source_id=source.source_id)

    assert pot_root.is_dir()
    assert resources.list(pot_id=pot.pot_id, slug=DOC)
    assert pots.list_sources(pot_id=pot.pot_id) == []


def test_hard_reset_purges_resources_after_successful_graph_reset():
    parent = MagicMock()
    context_graph = MagicMock()
    context_graph.reset_pot.return_value = {"pot_id": "pot-1", "ok": True}
    parent.attach_mock(context_graph, "context_graph")
    resources = MagicMock()
    resources.purge_pot.return_value = True
    parent.attach_mock(resources, "resources")

    out = hard_reset_pot(context_graph, "pot-1", resources=resources)

    assert out["ok"] is True
    assert out["resources_purged"] is True
    assert parent.mock_calls.index(
        call.context_graph.reset_pot("pot-1")
    ) < parent.mock_calls.index(call.resources.purge_pot("pot-1"))


def test_hard_reset_skips_resource_purge_when_graph_reset_fails():
    context_graph = MagicMock()
    context_graph.reset_pot.return_value = {
        "pot_id": "pot-1",
        "ok": False,
        "error": "bad",
    }
    resources = MagicMock()

    out = hard_reset_pot(context_graph, "pot-1", resources=resources)

    assert out["ok"] is False
    resources.purge_pot.assert_not_called()
    assert "resources_purged" not in out
