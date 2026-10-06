"""Document resources across the local runtime: teardown, search and lifecycle.

Pot teardown (``pot reset`` / ``pot archive``) goes through the typed
``reset_context`` operation, so that is where the resource store is purged —
only after the graph reset succeeded, so a failed reset can never leave live
claims citing chunk ids that no longer exist.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from potpie.agent_context import AgentContextService
from potpie.cli.commands import _common, pots, resource
from potpie.pots.local_service import LocalPotManagementService
from potpie.pots.local_store import LocalPotStore
from potpie.runtime.composition import build_local_runtime
from potpie.runtime.local_engine import LocalEngineOperations
from potpie_context_engine import ContextIdentity, Success
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
    LocalResourceStore,
    pot_dir_name,
)
from potpie_context_engine.application.services.resource_facade import ResourceFacade
from potpie_context_engine.core.ports.agent_context import ResolveRequest, SearchRequest
from potpie_context_engine.requests import (
    ResetContextRequest,
    SearchRequest as EngineSearchRequest,
)
from potpie_context_engine.testing import write_import_directory

pytestmark = pytest.mark.unit

POT_NAME = "lifecycle"
DOC = "q3-review"


def _write_document(root: Path, *, text: str = "alpha") -> Path:
    return write_import_directory(
        root,
        [
            {
                "slug": "capacity",
                "title": "Capacity",
                "summary": "headroom",
                "ordinal": 0,
                "content_hash": "capacity-1",
                "chunks": [{"label": "opening", "text": text}],
            }
        ],
        source_ref="file:///q3.pdf",
        source_kind="pdf",
    )


def _seed_document(store: LocalResourceStore, pot_id: str, root: Path) -> Path:
    store.import_dir(pot_id=pot_id, slug=DOC, source_dir=_write_document(root))
    return store.home / "resources" / pot_dir_name(pot_id)


def _operations(*, reset_ok: bool = True, resources=None) -> LocalEngineOperations:
    mutation = SimpleNamespace(
        reset_pot=MagicMock(return_value={"pot_id": "pot-1", "ok": reset_ok})
    )
    return LocalEngineOperations(
        SimpleNamespace(backend=SimpleNamespace(mutation=mutation), resources=resources)
    )


# --- reset_context purges the resource store, after the graph -----------------


@pytest.mark.anyio
async def test_reset_leaves_no_files_under_the_pot_resource_tree(tmp_path) -> None:
    store = LocalResourceStore(home=tmp_path / "home")
    pot_root = _seed_document(store, "pot-1", tmp_path / "import")
    assert any(pot_root.rglob("*.txt"))

    outcome = await _operations(resources=ResourceFacade(store=store)).reset_context(
        ContextIdentity("pot-1"), ResetContextRequest()
    )

    assert isinstance(outcome, Success)
    assert outcome.value.reset is True
    assert outcome.value.resources_purged is True
    assert not pot_root.exists()


@pytest.mark.anyio
async def test_reset_reports_no_purge_on_a_pot_that_held_no_resources(
    tmp_path,
) -> None:
    """``resources_purged`` is the store's answer, not a literal ``True``."""
    store = LocalResourceStore(home=tmp_path / "home")

    outcome = await _operations(resources=ResourceFacade(store=store)).reset_context(
        ContextIdentity("pot-1"), ResetContextRequest()
    )

    assert outcome.value.resources_purged is False


@pytest.mark.anyio
async def test_reset_reports_unknown_purge_when_no_resource_store_is_composed() -> None:
    """Nothing to purge is not the same answer as purged nothing."""
    outcome = await _operations(resources=None).reset_context(
        ContextIdentity("pot-1"), ResetContextRequest()
    )

    assert outcome.value.reset is True
    assert outcome.value.resources_purged is None


@pytest.mark.anyio
async def test_a_failed_graph_reset_keeps_the_documents(tmp_path) -> None:
    store = LocalResourceStore(home=tmp_path / "home")
    pot_root = _seed_document(store, "pot-1", tmp_path / "import")

    outcome = await _operations(
        reset_ok=False, resources=ResourceFacade(store=store)
    ).reset_context(ContextIdentity("pot-1"), ResetContextRequest())

    assert outcome.value.reset is False
    assert outcome.value.resources_purged is None
    assert any(pot_root.rglob("*.txt"))


def test_remove_source_does_not_touch_resources(tmp_path) -> None:
    """``source remove`` is registration-only: documents stay until ``resource
    rm`` or pot teardown."""
    home = tmp_path / "home"
    resources = LocalResourceStore(home=home)
    service = LocalPotManagementService(
        store=LocalPotStore(home=home), backend=InMemoryGraphBackend()
    )
    pot = service.create_pot(name=POT_NAME, use=True)
    source = service.add_source(
        pot_id=pot.pot_id, kind="repo", location="github.com/acme/x"
    )
    pot_root = _seed_document(resources, pot.pot_id, tmp_path / "import")

    service.remove_source(pot_id=pot.pot_id, source_id=source.source_id)

    assert pot_root.is_dir()
    assert resources.list(pot_id=pot.pot_id, slug=DOC)
    assert service.list_sources(pot_id=pot.pot_id) == []


# --- the real local runtime -----------------------------------------------------


@pytest.fixture()
def runtime(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
    composition = build_local_runtime(backend=InMemoryGraphBackend())
    _common.set_runtime(composition)
    yield composition
    composition.close()
    _common.set_json(False)


def _import(tmp_path: Path, *, text: str = "alpha") -> dict:
    _common.set_json(True)
    result = CliRunner().invoke(
        resource.resource_app,
        ["import", str(_write_document(tmp_path / "in", text=text)), "--doc", DOC],
    )
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_archive_purges_the_pots_documents(runtime, tmp_path) -> None:
    pot = runtime.root.pots.create_pot(name=POT_NAME, use=True)
    _import(tmp_path)
    pot_root = tmp_path / "home" / "resources" / pot_dir_name(pot.pot_id)
    assert any(pot_root.rglob("*.txt"))

    result = CliRunner().invoke(pots.pot_app, ["archive", POT_NAME, "--confirm"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["archived"] is True
    assert payload["resources_purged"] is True
    assert not pot_root.exists()


def test_pot_reset_says_documents_were_cleared(runtime, tmp_path) -> None:
    runtime.root.pots.create_pot(name=POT_NAME, use=True)
    _import(tmp_path)
    _common.set_json(False)

    result = CliRunner().invoke(pots.pot_app, ["reset", POT_NAME, "--confirm"])

    assert result.exit_code == 0, result.stdout
    assert "stored documents" in result.stdout


def test_bare_search_reaches_ingested_documents(runtime, tmp_path) -> None:
    """A phrase lifted out of an ingested document is found by a bare search.

    Bare ``search`` resolves intent ``unknown``, whose families include both
    halves of a document: section summaries (``docs``) and chunk text
    (``resources``). The phrase here appears in the text and in no summary.
    """
    runtime.root.pots.create_pot(name=POT_NAME, use=True)
    _import(tmp_path, text="The limitation of liability cap is twelve months of fees.")

    envelope = _common.run_engine_operation(
        _common.get_engine_client(POT_NAME).search(
            EngineSearchRequest(query="limitation of liability cap")
        )
    )

    assert envelope.intent == "unknown"
    assert {"docs", "resources"} <= {report.include for report in envelope.coverage}
    hits = [item for item in envelope.items if item.include == "resources"]
    assert hits and hits[0].payload["doc"] == DOC


def test_runtime_starts_no_thread_until_asked_and_closes_cleanly(runtime) -> None:
    drain = runtime.engine.resources.drain
    assert drain.running is False

    runtime.start_background_work()
    runtime.start_background_work()
    expected = bool(runtime.engine.resources.index.capabilities().semantic)
    assert drain.running is expected

    runtime.close()
    runtime.close()
    assert drain.running is False
    assert not any(
        thread.name == drain.name and thread.is_alive()
        for thread in threading.enumerate()
    )


# --- the agent-facing ``docs`` filter ------------------------------------------


def _agent_context(graph) -> AgentContextService:
    return AgentContextService(graph=graph, pots=MagicMock(), skills=MagicMock())


@pytest.mark.parametrize(
    ("include", "expected"),
    [
        (("docs",), ("docs", "resources")),
        (("Docs", "decisions"), ("docs", "decisions", "resources")),
        (("docs", "resources"), ("docs", "resources")),
        (("resources",), ("resources",)),
        ((), ()),
    ],
)
def test_docs_include_also_searches_document_text(include, expected) -> None:
    graph = MagicMock()

    _agent_context(graph).search(SearchRequest(pot_id="p", query="q", include=include))
    _agent_context(graph).resolve(ResolveRequest(pot_id="p", task="t", include=include))

    assert graph.search.call_args.args[0].include == expected
    assert graph.resolve.call_args.args[0].include == expected


# --- composition and diagnostics -------------------------------------------------


def test_an_unknown_index_profile_degrades_instead_of_breaking_every_command(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo still never answers as the default index: it reports itself."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_RESOURCE_INDEX", "sqlite_vec")
    composition = build_local_runtime(backend=InMemoryGraphBackend())
    try:
        status = composition.engine.resources.index_status(pot_id="p")
    finally:
        composition.close()

    assert status.ready is False
    assert "sqlite_vec" in (status.detail or "")


def test_doctor_reports_the_store_and_its_index(runtime, tmp_path) -> None:
    from potpie.cli import main as cli_main

    runtime.root.pots.create_pot(name=POT_NAME, use=True)
    _import(tmp_path)

    result = CliRunner().invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["resources"]["available"] is True
    assert payload["resources"]["documents"] == 1
    assert payload["resource_index"]["available"] is True
    assert payload["resource_index"]["chunks"] == 1


def test_doctor_without_an_active_pot_reports_the_gap(runtime) -> None:
    from potpie.cli import main as cli_main

    result = CliRunner().invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["resources"]["available"] is False
    assert "no active pot" in payload["resources"]["detail"]


@pytest.mark.anyio
async def test_daemon_shutdown_closes_the_drain_after_releasing_leases() -> None:
    """Leases first, so no operation is mid-write when the index closes; the
    close itself must never be what fails a shutdown."""
    from potpie.daemon import __main__ as daemon_main

    order: list[str] = []

    async def release() -> object:
        order.append("leases")
        return Success(None)

    class _Composition:
        def close(self) -> None:
            order.append("drain and index")
            raise RuntimeError("index already closed")

    outcome = await daemon_main._then_close_composition(release, _Composition())()

    assert outcome == Success(None)
    assert order == ["leases", "drain and index"]
