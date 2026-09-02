"""One command, one answer: the CLI asks the host once for each fact it needs.

Pot resolution, the pot header, and the empty-pot guidance used to re-fetch
the same pot list, source list, and data-plane counts independently — eight
host calls for ``graph read`` and eighteen for ``graph status --json``, of
which one was the command's real call. On a managed host each of those is a
full round trip, so the fan-out was the floor of every command's wall time.

These tests pin the budget with a host that counts what it is asked, and the
two properties that make a per-process memo safe: a write in the same process
is seen by the next read, and the RPC client rides one connection.
"""

from __future__ import annotations

import json

import httpx
import pytest
import typer
from typer.testing import CliRunner

from potpie.cli import host_snapshot
from potpie.cli.commands import _common, graph, pots, query
from potpie.daemon.client import DaemonRpcClient
from potpie_context_core.agent_envelope import AgentEnvelope
from potpie_context_core.errors import ContextEngineDisabled
from potpie_context_core.ports.graph.backend import BackendCapabilities
from potpie_context_core.ports.graph_service import (
    DataPlaneStatus,
    GraphEntityCandidate,
    GraphEntitySearchResult,
    GraphReadResult,
)
from tests._rpc_fakes import install_rpc_session

pytestmark = pytest.mark.unit

_REPO = "github.com/acme/shop"


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(_common, "_current_git_remote", lambda cwd: _REPO)
    yield
    _common.set_json(False)
    _common.set_host(None)


class _Pot:
    def __init__(self, pot_id: str, name: str, active: bool = False) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = active
        self.archived = False


class _Source:
    kind = "repo"

    def __init__(self, name: str) -> None:
        self.source_id = f"src_{name}"
        self.name = name
        self.location = name


class _RepoSourceRow:
    def __init__(self, pot: _Pot, source: _Source) -> None:
        self.pot_id = pot.pot_id
        self.pot_name = pot.name
        self.name = source.name
        self.location = source.location


class _Pots:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.p1 = _Pot("p1", "shop", active=True)
        self.p2 = _Pot("p2", "shop-fork")
        self.pots = [self.p1, self.p2]
        self.sources = {"p1": [_Source(_REPO)], "p2": [_Source(_REPO)]}
        self.defaults = {_REPO: "p1"}

    def list_pots(self):
        self.log.append("pots.list_pots")
        return list(self.pots)

    def active_pot(self):
        self.log.append("pots.active_pot")
        return next((pot for pot in self.pots if pot.active), None)

    def repo_default(self, *, repo):
        self.log.append("pots.repo_default")
        return self.defaults.get(repo)

    def list_repo_sources(self):
        self.log.append("pots.list_repo_sources")
        return [
            _RepoSourceRow(pot, source)
            for pot in self.pots
            for source in self.sources.get(pot.pot_id, [])
        ]

    def list_sources(self, *, pot_id):
        self.log.append("pots.list_sources")
        return list(self.sources.get(pot_id, []))

    # Writes answer with *fresh* records, as a host over the wire does. A fake
    # that mutated the objects it had already handed out would let a stale memo
    # look fresh, and the invalidation tests below would prove nothing.

    def use_pot(self, *, ref):
        self.log.append("pots.use_pot")
        self.pots = [
            _Pot(pot.pot_id, pot.name, active=ref in (pot.pot_id, pot.name))
            for pot in self.pots
        ]
        return next(pot for pot in self.pots if pot.active)

    def rename_pot(self, *, ref, new_name):
        self.log.append("pots.rename_pot")
        self.pots = [
            _Pot(
                pot.pot_id,
                new_name if ref in (pot.pot_id, pot.name) else pot.name,
                active=pot.active,
            )
            for pot in self.pots
        ]
        return next(pot for pot in self.pots if pot.name == new_name)

    def set_repo_default(self, *, repo, pot_id):
        self.log.append("pots.set_repo_default")
        self.defaults[repo] = pot_id


class _Graph:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def data_plane_status(self, pot_id):
        self.log.append("graph.data_plane_status")
        return DataPlaneStatus(
            pot_id=pot_id,
            backend_profile="memory",
            backend_ready=True,
            reader_backed_includes=("features",),
            counts={"claims": 3, "entities": 2},
            match_mode="lexical",
        )

    def read(self, request):
        self.log.append("graph.read")
        return GraphReadResult(
            view=request.view,
            subgraph=request.subgraph,
            items=({"key": "feature:checkout", "fact": "checkout is a feature"},),
            graph_contract_version="v1.5",
            ontology_version="2026-06-graph",
        )

    def search_entities(self, request):
        self.log.append("graph.search_entities")
        return GraphEntitySearchResult(
            entities=(
                GraphEntityCandidate(
                    key="feature:checkout", labels=("Feature",), score=0.9
                ),
            ),
            match_mode="lexical",
            graph_contract_version="v1.5",
            ontology_version="2026-06-graph",
        )


class _Backend:
    profile = "memory"

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def capabilities(self):
        self.log.append("backend.capabilities")
        return BackendCapabilities(profile="memory")


class _QualityBody:
    def to_dict(self):
        return {"status": "ok", "metrics": {}}


class _Workbench:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def quality(self, **kwargs):
        self.log.append("graph_workbench.quality")
        return _QualityBody()


class _AgentContext:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def search(self, request):
        self.log.append("agent_context.search")
        return AgentEnvelope(
            pot_id=request.pot_id, intent="unknown", items=(), coverage=()
        )


class _Host:
    """A host that logs every call the CLI makes to it."""

    profile = "local"

    def __init__(self) -> None:
        self.log: list[str] = []
        self.pots = _Pots(self.log)
        self.graph = _Graph(self.log)
        self.backend = _Backend(self.log)
        self.graph_workbench = _Workbench(self.log)
        self.agent_context = _AgentContext(self.log)


def _installed() -> _Host:
    host = _Host()
    _common.set_host(host)
    return host


def _query_app() -> typer.Typer:
    app = typer.Typer()
    query.register(app)
    return app


# --- the budgets ---------------------------------------------------------------


def test_graph_read_asks_the_host_three_times() -> None:
    """Resolution (repo default + the pot it names) and the read. Nothing else."""
    host = _installed()

    result = CliRunner().invoke(
        graph.graph_app, ["read", "--subgraph", "features", "--view", "feature_context"]
    )

    assert result.exit_code == 0, result.output
    assert host.log.count("graph.read") == 1
    assert len(host.log) <= 3, host.log
    assert "graph.data_plane_status" not in host.log
    assert "pots.list_sources" not in host.log


def test_the_read_header_names_the_pot_without_counting_it() -> None:
    """The counts were the two host calls the header cost; they moved behind
    ``--verbose``, where the header still spells them out."""
    _installed()

    result = CliRunner().invoke(
        graph.graph_app, ["read", "--subgraph", "features", "--view", "feature_context"]
    )

    first_line = result.output.splitlines()[0]
    assert first_line.startswith("pot=shop (p1)")
    assert "claims=" not in first_line


def test_the_verbose_header_still_carries_the_counts() -> None:
    host = _installed()
    _common.set_verbose(True)
    try:
        header = _common.pot_scope_human(host, "p1")
    finally:
        _common.set_verbose(False)

    assert header == "pot=shop (p1) sources=1 claims=3 entities=2"


def test_graph_status_json_asks_the_host_at_most_six_times() -> None:
    """Resolution, one data-plane status shared by the header, the guidance and
    the payload, the capabilities, the source count, the quality summary."""
    host = _installed()
    _common.set_json(True)

    result = CliRunner().invoke(graph.graph_app, ["status"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result"]["pot"]["id"] == "p1"
    assert host.log.count("graph.data_plane_status") == 1
    assert host.log.count("pots.list_pots") == 1
    assert len(host.log) <= 6, host.log


def test_search_asks_the_host_three_times() -> None:
    host = _installed()

    result = CliRunner().invoke(_query_app(), ["search", "checkout"])

    assert result.exit_code == 0, result.output
    assert host.log.count("agent_context.search") == 1
    assert len(host.log) <= 3, host.log


def test_search_entities_skips_the_guidance_when_it_found_something() -> None:
    host = _installed()

    result = CliRunner().invoke(graph.graph_app, ["search-entities", "checkout"])

    assert result.exit_code == 0, result.output
    assert host.log.count("graph.search_entities") == 1
    assert "graph.data_plane_status" not in host.log
    assert len(host.log) <= 3, host.log


# --- the memo's two safety properties ------------------------------------------


def test_an_answer_is_reused_only_for_the_same_host_object() -> None:
    """A fake built per test must not inherit a previous fake's answers through a
    recycled ``id()``: the memo checks identity, not just the key."""
    first = _Host()
    assert _common._list_pots(first)[0].name == "shop"
    second = _Host()
    second.pots.p1.name = "other"

    assert _common._list_pots(second)[0].name == "other"
    assert first.log == ["pots.list_pots"]
    assert second.log == ["pots.list_pots"]


def test_a_failed_call_is_not_remembered() -> None:
    host = _Host()
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first try fails")
        return "answer"

    with pytest.raises(RuntimeError):
        host_snapshot.memoized(host, "x", (), _flaky)
    assert host_snapshot.memoized(host, "x", (), _flaky) == "answer"
    assert calls["n"] == 2


def test_a_write_in_the_same_process_is_seen_by_the_next_read() -> None:
    """``pot use`` then a header, one process: the header must name the new pot."""
    host = _installed()
    assert _common._active_pot(host).pot_id == "p1"
    assert _common.pot_scope_human(host, "p1") == "pot=shop (p1)"

    result = CliRunner().invoke(pots.pot_app, ["use", "p2"])

    assert result.exit_code == 0, result.output
    assert _common._active_pot(host).pot_id == "p2"


def test_a_rename_in_the_same_process_is_seen_by_the_next_header() -> None:
    host = _installed()
    assert _common.pot_scope_human(host, "p1") == "pot=shop (p1)"

    host.pots.rename_pot(ref="p1", new_name="store")
    # The host changed underneath the memo, so the header is stale until a
    # write through the CLI — or the RPC client — clears it. Pinned as the
    # counter-example the next assertion depends on.
    assert _common.pot_scope_human(host, "p1") == "pot=shop (p1)"
    _common.invalidate_host_snapshot()

    assert _common.pot_scope_human(host, "p1") == "pot=store (p1)"


def test_injecting_a_host_forgets_the_previous_one_s_answers() -> None:
    host = _installed()
    _common._list_pots(host)
    assert host_snapshot.snapshot_entry_count() == 1

    _common.set_host(_Host())

    assert host_snapshot.snapshot_entry_count() == 0


# --- the RPC client --------------------------------------------------------------


class _Endpoint:
    def __init__(self) -> None:
        self.reads = 0

    def discovery(self) -> dict[str, str]:
        self.reads += 1
        return {"base_url": "http://daemon.test", "token": "tok"}


def _ok(url: str, **_: object) -> httpx.Response:
    return httpx.Response(
        200, json={"ok": True, "result": []}, request=httpx.Request("POST", url)
    )


def test_the_client_rides_one_session_and_reads_discovery_once(monkeypatch) -> None:
    opened = install_rpc_session(monkeypatch, _ok)
    endpoint = _Endpoint()
    client = DaemonRpcClient(daemon=endpoint)

    for _ in range(3):
        client.call("pots", "list_pots")

    assert len(opened) == 1
    assert opened[0].posts == 3
    assert endpoint.reads == 1


def test_a_connection_error_drops_the_session_and_rereads_discovery(
    monkeypatch,
) -> None:
    """A daemon that restarted mid-process may have moved; the next call must
    find it rather than keep posting to the address it used to have."""
    answers = iter(["refuse", "ok"])

    def _post(url: str, **_: object) -> httpx.Response:
        if next(answers) == "refuse":
            raise httpx.ConnectError("refused", request=httpx.Request("POST", url))
        return _ok(url)

    opened = install_rpc_session(monkeypatch, _post)
    endpoint = _Endpoint()
    client = DaemonRpcClient(daemon=endpoint)

    with pytest.raises(ContextEngineDisabled):
        client.call("pots", "list_pots")
    client.call("pots", "list_pots")

    assert len(opened) == 2
    assert opened[0].closed is True
    assert endpoint.reads == 2


def test_a_write_over_rpc_clears_the_memo_and_a_read_does_not(monkeypatch) -> None:
    install_rpc_session(monkeypatch, _ok)
    client = DaemonRpcClient(daemon=_Endpoint())
    host = _Host()

    _common._list_pots(host)
    client.call("pots", "list_pots")
    assert host_snapshot.snapshot_entry_count() == 1

    client.call("pots", "use_pot", ref="p2")
    assert host_snapshot.snapshot_entry_count() == 0
