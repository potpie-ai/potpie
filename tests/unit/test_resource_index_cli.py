"""``potpie resource index`` — the three verbs and the shapes they emit.

The CLI surface is fixed across every profile, which is the point of putting a
port under it: these tests pin the contract an agent reads through the typed
engine boundary, not the retrieval behaviour (that is the conformance suite's
job).
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, resource
from potpie_context_engine.core.ports.resource_index import (
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
    _common.set_runtime(None)


class _Pot:
    pot_id = "pot-1"
    name = "default"
    active = True
    archived = False


class _Pots:
    def active_pot(self):
        return _Pot()

    def list_pots(self):
        return [_Pot()]

    def list_repo_sources(self):
        return []

    def repo_default(self, *, repo):
        return None


class FakeResources:
    def __init__(self, status=None, *, backlog=(0,)):
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
            location="index/resources.sqlite3",
            replica="host:1",
        )
        self.backlog = list(backlog)
        self.builds: list[dict] = []
        self.rebuilds: list[dict] = []

    def index_status(self, *, pot_id=None):
        return self.status_report

    def index_build(self, *, pot_id=None, budget=256, wait=False):
        self.builds.append({"pot_id": pot_id, "wait": wait})
        remaining = self.backlog.pop(0) if self.backlog else 0
        return DrainReport(
            profile="sqlite_hybrid",
            embedded=12,
            remaining=remaining,
            batches=1,
            elapsed_ms=84,
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


def _app(resources):
    """Bind fake engine services and turn on ``--json``.

    ``--json`` is a root-level flag on the real CLI, so a sub-app invoked
    directly never sees it; the tests set the mode the way the root callback
    would."""
    _common.set_runtime(
        SimpleNamespace(
            pots=_Pots(),
            backend=SimpleNamespace(profile="in_memory"),
            resources=resources,
        )
    )
    _common.set_json(True)
    return resource.resource_app


def _json(result):
    return json.loads(result.stdout)


def test_status_reports_declared_capabilities_and_the_backlog():
    result = runner.invoke(_app(FakeResources()), ["index", "status"])

    assert result.exit_code == 0, result.stdout
    payload = _json(result)
    assert payload["profile"] == "sqlite_hybrid"
    assert payload["match_mode"] == "hybrid"
    assert "semantic" in payload["capabilities"]
    assert payload["chunks"] == 9 and payload["pending_embeddings"] == 0


def test_status_names_the_backlog_as_the_next_action():
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

    payload = _json(runner.invoke(_app(resources), ["index", "status"]))

    assert "17" in payload["recommended_next_action"]
    assert "build" in payload["recommended_next_action"]


def test_status_of_an_unready_index_names_the_config_key():
    """``resource_index`` is a real config key now, so the repair can name it."""
    resources = FakeResources(
        ResourceIndexStatus(profile="none", ready=False, detail="index is off")
    )

    payload = _json(runner.invoke(_app(resources), ["index", "status"]))

    assert payload["ready"] is False
    assert "potpie config set resource_index" in payload["recommended_next_action"]


def test_build_drains_one_bounded_batch_per_call():
    resources = FakeResources()

    result = runner.invoke(_app(resources), ["index", "build"])

    assert result.exit_code == 0, result.stdout
    assert resources.builds == [{"pot_id": "pot-1", "wait": False}]
    assert _json(result)["embedded"] == 12


def test_build_wait_keeps_draining_until_nothing_is_pending():
    """``--wait`` loops bounded calls, so no one call outlives a request
    deadline however large the backlog is."""
    resources = FakeResources(backlog=(30, 4, 0))

    result = runner.invoke(_app(resources), ["index", "build", "--wait"])

    assert result.exit_code == 0, result.stdout
    assert [row["wait"] for row in resources.builds] == [False, False, False]
    payload = _json(result)
    assert payload["embedded"] == 36
    assert payload["batches"] == 3
    assert payload["remaining"] == 0


def test_build_wait_stops_when_a_batch_embeds_nothing():
    """A batch that embeds nothing with work outstanding is a failing embedder;
    looping on it would hang the command instead of reporting it."""

    class Stalled(FakeResources):
        def index_build(self, *, pot_id=None, budget=256, wait=False):
            self.builds.append({"pot_id": pot_id, "wait": wait})
            return DrainReport(
                profile="sqlite_hybrid", embedded=0, remaining=9, detail="embedder down"
            )

    resources = Stalled()

    result = runner.invoke(_app(resources), ["index", "build", "--wait"])

    assert result.exit_code == 0, result.stdout
    assert len(resources.builds) == 1
    assert _json(result)["detail"] == "embedder down"


def test_build_for_one_doc_re_derives_that_document_first():
    """``--doc`` has to mean something on a document the index never saw."""
    resources = FakeResources()

    result = runner.invoke(_app(resources), ["index", "build", "--doc", "q3-review"])

    assert result.exit_code == 0, result.stdout
    assert resources.rebuilds == [{"pot_id": "pot-1", "doc": "q3-review"}]


def test_rebuild_requires_confirmation():
    resources = FakeResources()

    result = runner.invoke(_app(resources), ["index", "rebuild"])

    assert result.exit_code != 0
    assert resources.rebuilds == []
    payload = _json(result)
    assert payload["code"] == "confirmation_required"
    assert "--confirm" in payload["recommended_next_action"]


def test_rebuild_reports_what_it_re_derived():
    resources = FakeResources()

    result = runner.invoke(_app(resources), ["index", "rebuild", "--confirm"])

    assert result.exit_code == 0, result.stdout
    payload = _json(result)
    assert payload["document_count"] == 1
    assert payload["chunk_count"] == 3
    assert payload["pending_embeddings"] == 7
    # Pending work after a rebuild is expected, not a failure — the next action
    # says how to finish it now rather than warning about it.
    assert "build" in payload["recommended_next_action"]
    assert resources.rebuilds == [{"pot_id": "pot-1", "doc": None}]
