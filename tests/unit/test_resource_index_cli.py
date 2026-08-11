"""``potpie resource index`` — the three verbs and the shapes they emit.

The CLI surface is fixed across every profile, which is the point of putting a
port under it: these tests pin the contract an agent reads, not the retrieval
behaviour (that is the conformance suite's job).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, resource
from potpie_context_core.ports.resource_index import (
    DrainReport,
    IndexReport,
    ResourceIndexStatus,
)

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_json(False)
    _common.set_host(None)


class FakeResources:
    def __init__(self, status=None):
        self.status_report = status or ResourceIndexStatus(
            profile="sqlite_hybrid",
            ready=True,
            capabilities=("lexical", "semantic", "hybrid", "snippets", "incremental"),
            match_mode="hybrid",
            documents=2,
            chunks=9,
            windows=31,
            pending_embeddings=0,
            embedder="local-hashing-v1",
            dimensions=256,
            location="/home/index/resources.sqlite3",
            replica="host:1",
        )
        self.builds: list[dict] = []
        self.rebuilds: list[dict] = []

    def index_status(self, *, pot_id=None):
        return self.status_report

    def index_build(self, *, pot_id=None, budget=256, wait=False):
        self.builds.append({"pot_id": pot_id, "wait": wait})
        return DrainReport(
            profile="sqlite_hybrid", embedded=12, remaining=0, batches=1, elapsed_ms=84
        )

    def index_rebuild(self, *, pot_id, doc=None):
        self.rebuilds.append({"pot_id": pot_id, "doc": doc})
        return (
            IndexReport(
                doc=doc or "q3-review",
                profile="sqlite_hybrid",
                sections=2,
                chunks=3,
                windows=7,
                pending_embeddings=7,
            ),
        )


class FakeHost:
    def __init__(self, resources):
        self.resources = resources


def _app(host):
    """Bind the fake host and turn on ``--json``.

    ``--json`` is a root-level flag on the real CLI, so a sub-app invoked
    directly never sees it; the tests set the mode the way the root callback
    would."""
    _common.set_host(host)
    _common.set_json(True)
    return resource.resource_app


def _json(result):
    return json.loads(result.stdout)


def test_status_reports_declared_capabilities_and_the_backlog(monkeypatch):
    host = FakeHost(FakeResources())
    monkeypatch.setattr(_common, "resolve_pot_id", lambda *a, **k: "pot-1")
    monkeypatch.setattr(resource, "resolve_pot_id", lambda *a, **k: "pot-1")
    result = runner.invoke(_app(host), ["index", "status"])
    assert result.exit_code == 0, result.stdout
    payload = _json(result)
    assert payload["profile"] == "sqlite_hybrid"
    assert payload["match_mode"] == "hybrid"
    assert "semantic" in payload["capabilities"]
    assert payload["chunks"] == 9 and payload["pending_embeddings"] == 0


def test_status_names_the_backlog_as_the_next_action(monkeypatch):
    """A pending backlog means search is lexical — the caller must be told."""
    resources = FakeResources(
        ResourceIndexStatus(
            profile="sqlite_hybrid",
            ready=True,
            capabilities=("lexical", "semantic", "hybrid"),
            match_mode="hybrid",
            documents=1,
            chunks=4,
            pending_embeddings=17,
        )
    )
    monkeypatch.setattr(resource, "resolve_pot_id", lambda *a, **k: "pot-1")
    result = runner.invoke(_app(FakeHost(resources)), ["index", "status"])
    payload = _json(result)
    assert "17" in payload["recommended_next_action"]
    assert "build" in payload["recommended_next_action"]


def test_build_wait_is_passed_through(monkeypatch):
    resources = FakeResources()
    monkeypatch.setattr(resource, "resolve_pot_id", lambda *a, **k: "pot-1")
    result = runner.invoke(_app(FakeHost(resources)), ["index", "build", "--wait"])
    assert result.exit_code == 0, result.stdout
    assert resources.builds == [{"pot_id": "pot-1", "wait": True}]
    assert _json(result)["embedded"] == 12


def test_build_for_one_doc_re_derives_that_document_first(monkeypatch):
    """``--doc`` has to mean something on a document the index never saw."""
    resources = FakeResources()
    monkeypatch.setattr(resource, "resolve_pot_id", lambda *a, **k: "pot-1")
    result = runner.invoke(
        _app(FakeHost(resources)),
        ["index", "build", "--doc", "q3-review"],
    )
    assert result.exit_code == 0, result.stdout
    assert resources.rebuilds == [{"pot_id": "pot-1", "doc": "q3-review"}]


def test_rebuild_requires_confirmation(monkeypatch):
    resources = FakeResources()
    monkeypatch.setattr(resource, "resolve_pot_id", lambda *a, **k: "pot-1")
    result = runner.invoke(_app(FakeHost(resources)), ["index", "rebuild"])
    assert result.exit_code != 0
    assert resources.rebuilds == []
    payload = _json(result)
    assert payload["code"] == "confirmation_required"
    assert "--confirm" in payload["recommended_next_action"]


def test_rebuild_reports_what_it_re_derived(monkeypatch):
    resources = FakeResources()
    monkeypatch.setattr(resource, "resolve_pot_id", lambda *a, **k: "pot-1")
    result = runner.invoke(
        _app(FakeHost(resources)),
        ["index", "rebuild", "--confirm"],
    )
    assert result.exit_code == 0, result.stdout
    payload = _json(result)
    assert payload["document_count"] == 1
    assert payload["chunk_count"] == 3
    assert payload["pending_embeddings"] == 7
    # Pending work after a rebuild is expected, not a failure — the next action
    # says how to finish it now rather than warning about it.
    assert "build" in payload["recommended_next_action"]
