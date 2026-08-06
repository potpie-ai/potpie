"""Unit tests for the FalkorDB GraphWriterPort adapter.

No live FalkorDB: an injected fake graph captures queries and feeds canned
results, so we exercise the ``enabled`` gate, the ``reset_pot`` contract
(client-side batched delete, same result-dict shape as Neo4j), the unnamed
best-effort index DDL, the async-shim record mapping, and that the reused
``cypher.py`` mutation path issues the expected MERGE.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from potpie_context_engine.adapters.outbound.graph.falkordb_writer import (
    _OWNED_SERVERS,
    FalkorDBGraphWriter,
    _aof_dir,
    _aof_is_ready,
    _died_without_saving,
    _ensure_lite_durability,
    _lite_server_config,
    _records_from_result,
    _refuse_untrustworthy_state,
    _remember_owned_server,
    build_falkordb_graph,
    shutdown_embedded_servers,
)
from potpie_context_core.errors import GraphSubstrateUnavailable
from potpie_context_core.api import (
    DEFAULT_GRAPH_DEFINITION,
    EdgeTypeSpec,
    EntityTypeSpec,
    GraphExtension,
    IdentityClass,
)
from potpie_context_core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)

pytestmark = pytest.mark.unit


class _FakeResult:
    def __init__(self, header=None, result_set=None):
        self.header = header or []
        self.result_set = result_set or []


class _FakeGraph:
    """Captures queries; answers count() and absorbs DETACH DELETE."""

    def __init__(self, count: int = 0, *, raise_on_index: bool = False):
        self.queries: list[tuple[str, dict]] = []
        self._count = count
        self._deleted = False
        self._raise_on_index = raise_on_index

    def query(self, cypher: str, params=None):
        self.queries.append((cypher, params or {}))
        if "CREATE INDEX" in cypher and self._raise_on_index:
            raise RuntimeError("already indexed")
        if "DETACH DELETE" in cypher:
            self._deleted = True
            return _FakeResult()
        if "count(n) AS cnt" in cypher:
            val = 0 if self._deleted else self._count
            return _FakeResult(header=[[1, "cnt"]], result_set=[[val]])
        return _FakeResult()


class _FakeSettings:
    def __init__(
        self, *, enabled=True, url="redis://localhost:6379", name="g", mode="server"
    ):
        self._enabled = enabled
        self._url = url
        self._name = name
        self._mode = mode

    def is_enabled(self) -> bool:
        return self._enabled

    def falkordb_url(self):
        return self._url

    def falkordb_graph_name(self) -> str:
        return self._name

    def falkordb_mode(self) -> str:
        return self._mode

    def falkordb_lite_path(self) -> str:
        return ".potpie/test/falkordb.db"


class _FakeEmbedder:
    name = "fake-embedder"
    dimensions = 3

    def embed(self, text: str) -> tuple[float, ...]:
        return (0.1, 0.2, 0.3)

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


def test_records_from_result_maps_columns() -> None:
    res = _FakeResult(
        header=[[1, "key"], [1, "labels"]], result_set=[["s", ["Entity", "Service"]]]
    )
    recs = _records_from_result(res)
    assert recs == [{"key": "s", "labels": ["Entity", "Service"]}]


def test_records_from_result_empty() -> None:
    assert _records_from_result(_FakeResult()) == []


def test_enabled_false_when_context_graph_disabled() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(enabled=False), graph=_FakeGraph())
    assert w.enabled is False


def test_enabled_false_when_unconfigured() -> None:
    # Server mode with no url and no injected graph → not configured.
    w = FalkorDBGraphWriter(_FakeSettings(url=None, mode="server"))
    assert w.enabled is False


def test_enabled_true_when_lite_mode() -> None:
    # Lite is the default local path: needs no url/graph to be enabled.
    w = FalkorDBGraphWriter(_FakeSettings(url=None, mode="lite"))
    assert w.enabled is True


def test_enabled_true_when_url_set() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(url="redis://localhost:6379"))
    assert w.enabled is True


def test_enabled_true_when_graph_injected() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(url=None), graph=_FakeGraph())
    assert w.enabled is True


async def test_reset_pot_disabled_returns_error() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(enabled=False), graph=_FakeGraph())
    out = await w.reset_pot("pot-1")
    assert out == {"ok": False, "error": "context_graph_disabled"}


async def test_reset_pot_rejects_invalid_pot_id() -> None:
    w = FalkorDBGraphWriter(_FakeSettings(), graph=_FakeGraph(count=3))
    out = await w.reset_pot("bad pot id")
    assert out["ok"] is False
    assert out["error"].startswith("invalid_pot_id")


async def test_reset_pot_success_shape() -> None:
    graph = _FakeGraph(count=3)
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    out = await w.reset_pot("pot-1")
    assert out == {
        "ok": True,
        "group_id_nodes_before": 3,
        "group_id_nodes_remaining": 0,
    }
    assert any("DETACH DELETE" in q for q, _ in graph.queries)


async def test_ensure_indexes_best_effort_swallows_errors() -> None:
    graph = _FakeGraph(raise_on_index=True)
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    # Even though every CREATE INDEX raises, ensure_indexes must not propagate.
    assert await w.ensure_indexes() is True
    assert sum("CREATE INDEX" in q for q, _ in graph.queries) == 4


async def test_ensure_indexes_uses_unnamed_form() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    await w.ensure_indexes()
    index_qs = [q for q, _ in graph.queries if "CREATE INDEX" in q]
    # FalkorDB rejects named indexes + IF NOT EXISTS; assert neither is used.
    for q in index_qs:
        assert "IF NOT EXISTS" not in q
        assert q.startswith("CREATE INDEX FOR")


async def test_ensure_indexes_creates_falkordb_vector_index_with_embedder_dim() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph, embedder=_FakeEmbedder())
    await w.ensure_indexes()
    vector_qs = [q for q, _ in graph.queries if "CREATE VECTOR INDEX" in q]
    assert len(vector_qs) == 1
    assert "RELATES_TO" in vector_qs[0]
    assert "fact_embedding" in vector_qs[0]
    assert "dimension:3" in vector_qs[0]
    assert "similarityFunction:'cosine'" in vector_qs[0]


async def test_upsert_entities_issues_merge_via_shim() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    prov = ProvenanceRef(pot_id="p1", source_event_id="e1")
    n = await w.upsert_entities(
        "p1",
        [EntityUpsert(entity_key="service:web", labels=("Entity",), properties={})],
        prov,
    )
    assert n == 1
    assert any("MERGE (e:Entity" in q for q, _ in graph.queries)


async def test_upsert_entities_removes_stale_labels_in_one_query() -> None:
    graph = _FakeGraph()
    writer = FalkorDBGraphWriter(_FakeSettings(), graph=graph)

    await writer.upsert_entities(
        "p1",
        [
            EntityUpsert(
                entity_key="environment:production",
                labels=("Entity", "Environment"),
                properties={},
            )
        ],
        ProvenanceRef(pot_id="p1", source_event_id="e1"),
    )

    removals = [query for query, _ in graph.queries if " REMOVE e:" in query]
    assert len(removals) == 1
    assert "Environment" not in removals[0]
    assert removals[0].count(":") > 1


async def test_upsert_entities_empty_is_noop() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph)
    n = await w.upsert_entities(
        "p1", [], ProvenanceRef(pot_id="p1", source_event_id="e1")
    )
    assert n == 0
    assert graph.queries == []


async def test_writer_uses_bound_extension_labels_and_predicates() -> None:
    definition = DEFAULT_GRAPH_DEFINITION.extend(
        GraphExtension(
            name="widgets",
            version="1",
            entity_types={
                "Widget": EntityTypeSpec(
                    label="Widget",
                    category="topology",
                    description="Widget.",
                    identity_class=IdentityClass.SLUG_ALIAS,
                    key_prefix="widget",
                    identity_policy="widget:<slug>",
                )
            },
            edge_types={
                "CONNECTS_WIDGET": EdgeTypeSpec(
                    edge_type="CONNECTS_WIDGET",
                    description="Connects widgets.",
                    allowed_pairs=(("Widget", "Widget"),),
                )
            },
        )
    )
    graph = _FakeGraph()
    writer = FalkorDBGraphWriter(
        _FakeSettings(),
        graph=graph,
        definition=definition,
    )
    provenance = ProvenanceRef(pot_id="p1", source_event_id="e1")

    await writer.upsert_entities(
        "p1",
        [
            EntityUpsert("widget:a", ("Widget",), {}),
            EntityUpsert("widget:b", ("Widget",), {}),
        ],
        provenance,
    )
    written = await writer.upsert_edges(
        "p1",
        [EdgeUpsert("CONNECTS_WIDGET", "widget:a", "widget:b")],
        provenance,
    )

    assert written == 1
    assert any("SET e:Widget" in query for query, _ in graph.queries)
    assert any(
        params.get("predicate") == "CONNECTS_WIDGET" for _, params in graph.queries
    )


async def test_writer_removes_builtin_label_conflicting_with_extension_key() -> None:
    definition = DEFAULT_GRAPH_DEFINITION.extend(
        GraphExtension(
            name="widgets",
            version="1",
            entity_types={
                "Widget": EntityTypeSpec(
                    label="Widget",
                    category="topology",
                    description="Widget.",
                    identity_class=IdentityClass.SLUG_ALIAS,
                    key_prefix="widget",
                    identity_policy="widget:<slug>",
                )
            },
        )
    )
    graph = _FakeGraph()
    writer = FalkorDBGraphWriter(
        _FakeSettings(),
        graph=graph,
        definition=definition,
    )

    await writer.upsert_entities(
        "p1",
        [
            EntityUpsert(
                "widget:a",
                ("Entity", "Service", "Widget"),
                {},
            )
        ],
        ProvenanceRef(pot_id="p1", source_event_id="e1"),
    )

    removals = [query for query, _ in graph.queries if " REMOVE e:" in query]
    assert len(removals) == 1
    assert ":Service" in removals[0]
    assert ":Widget" not in removals[0]
    assert any("SET e:Widget" in query for query, _ in graph.queries)


async def test_writer_never_embeds_unsafe_registered_labels_in_cypher() -> None:
    unsafe = EntityTypeSpec(
        label="Bad-Label",
        category="topology",
        description="Deliberately malformed test label.",
        identity_class=IdentityClass.SLUG_ALIAS,
        key_prefix="bad",
        identity_policy="bad:<slug>",
    )
    definition = SimpleNamespace(entity_types={"Bad-Label": unsafe})
    graph = _FakeGraph()
    writer = FalkorDBGraphWriter(
        _FakeSettings(),
        graph=graph,
        definition=definition,
    )

    await writer.upsert_entities(
        "p1",
        [EntityUpsert("bad:a", ("Bad-Label",), {})],
        ProvenanceRef(pot_id="p1", source_event_id="e1"),
    )

    assert graph.queries
    assert all("Bad-Label" not in query for query, _ in graph.queries)


async def test_writer_uses_bound_extension_predicate_for_invalidation() -> None:
    definition = DEFAULT_GRAPH_DEFINITION.extend(
        GraphExtension(
            name="widgets",
            version="1",
            entity_types={
                "Widget": EntityTypeSpec(
                    label="Widget",
                    category="topology",
                    description="Widget.",
                    identity_class=IdentityClass.SLUG_ALIAS,
                    key_prefix="widget",
                    identity_policy="widget:<slug>",
                )
            },
            edge_types={
                "CONNECTS_WIDGET": EdgeTypeSpec(
                    edge_type="CONNECTS_WIDGET",
                    description="Connects widgets.",
                    allowed_pairs=(("Widget", "Widget"),),
                )
            },
        )
    )
    graph = _FakeGraph()
    writer = FalkorDBGraphWriter(
        _FakeSettings(),
        graph=graph,
        definition=definition,
    )

    await writer.invalidate(
        "p1",
        [
            InvalidationOp(
                target_entity_key=None,
                target_edge=("CONNECTS_WIDGET", "widget:a", "widget:b"),
                reason="superseded",
            )
        ],
        ProvenanceRef(pot_id="p1", source_event_id="e1"),
    )

    assert any(
        params.get("predicate") == "CONNECTS_WIDGET" for _, params in graph.queries
    )


async def test_upsert_edges_writes_falkordb_vecf32_embedding() -> None:
    graph = _FakeGraph()
    w = FalkorDBGraphWriter(_FakeSettings(), graph=graph, embedder=_FakeEmbedder())
    prov = ProvenanceRef(pot_id="p1", source_event_id="e1")
    n = await w.upsert_edges(
        "p1",
        [
            EdgeUpsert(
                edge_type="DEPENDS_ON",
                from_entity_key="service:web",
                to_entity_key="service:auth",
                properties={"fact": "web depends on auth"},
            )
        ],
        prov,
    )

    assert n == 1
    vector_writes = [item for item in graph.queries if "vecf32($embedding)" in item[0]]
    assert len(vector_writes) == 1
    _, params = vector_writes[0]
    assert params["embedding"] == [0.1, 0.2, 0.3]
    assert params["embedding_model"] == "fake-embedder"
    assert params["embedding_dim"] == 3


def test_build_falkordb_graph_server_mode_requires_url() -> None:
    # Server mode with no URL must fail loudly, not silently fall back to Lite.
    with pytest.raises(RuntimeError, match="server mode requires a URL"):
        build_falkordb_graph(_FakeSettings(url=None, mode="server"))


def test_enabled_false_server_mode_no_url_even_with_provider() -> None:
    # The container always injects a shared graph_provider; the enabled gate must
    # still report False for an unsatisfiable config (server mode, no URL), so it
    # never disagrees with what build_falkordb_graph can actually honor.
    w = FalkorDBGraphWriter(
        _FakeSettings(url=None, mode="server"),
        graph_provider=lambda: _FakeGraph(),
    )
    assert w.enabled is False


# --- Lite AOF durability ----------------------------------------------------
# The embedded profile persists only on a clean shutdown unless AOF is on, so
# these pin the two decisions that make enabling it non-destructive. Both were
# found the hard way: each of the wrong answers silently discards a populated
# pot on the next startup.


def _lite_db(tmp_path) -> str:
    return str(tmp_path / "context_graph" / "falkordb.db")


def test_lite_server_config_withholds_aof_until_the_manifest_exists(tmp_path) -> None:
    # Startup ``appendonly yes`` makes the AOF authoritative and Redis reads a
    # missing AOF as an empty dataset — so requesting it on a db that only has
    # an RDB wipes every claim. Nothing is requested until an AOF is complete.
    path = _lite_db(tmp_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert _lite_server_config(path) == {}

    open(path, "wb").write(b"fake-rdb")
    assert _lite_server_config(path) == {}, "an RDB alone must not request AOF"


def test_lite_server_config_requests_aof_once_the_manifest_exists(tmp_path) -> None:
    path = _lite_db(tmp_path)
    aof = _aof_dir(path)
    os.makedirs(aof, exist_ok=True)
    open(os.path.join(aof, "appendonly.aof.manifest"), "w").write("file x seq 1\n")
    assert _lite_server_config(path) == {
        "appendonly": "yes",
        "appendfsync": "always",
    }


def test_interrupted_migration_falls_back_to_the_rdb(tmp_path) -> None:
    # ``CONFIG SET appendonly yes`` makes the directory and a temp incr file
    # immediately, then rewrites in the background. A process killed in that
    # window leaves the directory with no manifest. Keying readiness on the
    # directory would load that empty AOF over a good RDB and lose the pot;
    # keying on the manifest retries the migration instead.
    path = _lite_db(tmp_path)
    aof = _aof_dir(path)
    os.makedirs(aof, exist_ok=True)
    open(os.path.join(aof, "temp-appendonly.aof.incr"), "w").write("")

    assert os.path.isdir(aof)
    assert _aof_is_ready(path) is False
    assert _lite_server_config(path) == {}


def test_ensure_lite_durability_is_a_no_op_once_the_aof_is_complete(tmp_path) -> None:
    path = _lite_db(tmp_path)
    aof = _aof_dir(path)
    os.makedirs(aof, exist_ok=True)
    open(os.path.join(aof, "appendonly.aof.manifest"), "w").write("file x seq 1\n")

    calls: list[tuple[str, str]] = []
    db = SimpleNamespace(
        connection=SimpleNamespace(
            config_set=lambda k, v: calls.append((k, v)),
            info=lambda _section: {"aof_rewrite_in_progress": 0},
        )
    )
    _ensure_lite_durability(db, path)
    assert calls == [], "a migrated store must not be reconfigured on every open"


def test_ensure_lite_durability_enables_aof_and_waits_for_the_rewrite(
    tmp_path,
) -> None:
    # The rewrite is asynchronous. Returning before the manifest lands lets the
    # caller write claims and exit with the AOF still a temp file — reaching
    # neither a finished AOF nor (on a crash) the RDB.
    path = _lite_db(tmp_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    calls: list[tuple[str, str]] = []
    polls = {"n": 0}

    def _info(_section: str) -> dict[str, int]:
        polls["n"] += 1
        if polls["n"] >= 3:  # rewrite finishes on the third poll
            aof = _aof_dir(path)
            os.makedirs(aof, exist_ok=True)
            open(os.path.join(aof, "appendonly.aof.manifest"), "w").write("x")
            return {"aof_rewrite_in_progress": 0}
        return {"aof_rewrite_in_progress": 1}

    db = SimpleNamespace(
        connection=SimpleNamespace(
            config_set=lambda k, v: calls.append((k, v)), info=_info
        )
    )
    _ensure_lite_durability(db, path)

    assert ("appendfsync", "always") in calls
    assert ("appendonly", "yes") in calls
    assert _aof_is_ready(path), "must not return before the manifest exists"


def test_ensure_lite_durability_warns_but_never_blocks_startup(tmp_path) -> None:
    # A store that will not take the setting is exactly as usable as before, so
    # this degrades rather than bricking the CLI.
    path = _lite_db(tmp_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("CONFIG SET is disabled")

    db = SimpleNamespace(
        connection=SimpleNamespace(config_set=_boom, info=_boom),
    )
    _ensure_lite_durability(db, path)  # must not raise
    assert _aof_is_ready(path) is False


# --- Stale substrate ---------------------------------------------------------
# AOF shrinks the window in which writes are lost; it does not stop a *read*
# from serving whatever survived as if it were the whole graph. These pin the
# detector and the refusal, which is the half that turns a crash into confident
# wrong answers.


def _write_settings(path: str, *, pid: int | None) -> None:
    """Reproduce redislite's handshake file, with a pid that may be dead."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pidfile = path + ".pid"
    if pid is not None:
        with open(pidfile, "w") as handle:
            handle.write(str(pid))
    with open(path + ".settings", "w") as handle:
        json.dump({"pidfile": pidfile, "unixsocket": path + ".sock"}, handle)


def _dead_pid() -> int:
    """A pid guaranteed not to be running: run a child to completion, reuse its id.

    A real pid rather than a made-up one, so the liveness check under test is
    the actual ``os.kill(pid, 0)`` probe and not a stub of it.
    """
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def test_clean_shutdown_leaves_nothing_to_detect(tmp_path) -> None:
    # redislite removes the settings file when its owning process exits
    # cleanly, so its absence is the normal, healthy state.
    path = _lite_db(tmp_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert _died_without_saving(path) is False
    _refuse_untrustworthy_state(path)  # must not raise


def test_a_live_server_is_not_an_unclean_shutdown(tmp_path) -> None:
    # Attaching to a running server is the whole point of the settings file.
    path = _lite_db(tmp_path)
    _write_settings(path, pid=os.getpid())
    assert _died_without_saving(path) is False
    _refuse_untrustworthy_state(path)  # must not raise


def test_settings_left_behind_by_a_dead_server_is_an_unclean_shutdown(
    tmp_path,
) -> None:
    path = _lite_db(tmp_path)
    _write_settings(path, pid=_dead_pid())
    assert _died_without_saving(path) is True


def test_a_settings_file_with_no_pidfile_counts_as_unclean(tmp_path) -> None:
    # Conservative on purpose: the alternative is guessing in the direction
    # that loses data quietly.
    path = _lite_db(tmp_path)
    _write_settings(path, pid=None)
    assert _died_without_saving(path) is True


def test_unclean_shutdown_without_an_aof_refuses_to_serve(tmp_path) -> None:
    # The data on disk may predate writes that were reported as committed.
    # Answering from it would be a confident wrong answer, which is worse for a
    # memory product than an error.
    path = _lite_db(tmp_path)
    _write_settings(path, pid=_dead_pid())

    with pytest.raises(GraphSubstrateUnavailable) as exc:
        _refuse_untrustworthy_state(path)

    # The refusal is only useful if it names the way out.
    assert ".settings" in (exc.value.recommended_next_action or "")


def test_unclean_shutdown_with_a_complete_aof_serves_normally(tmp_path) -> None:
    # The AOF replays every committed write, so there is nothing to warn about
    # — this is the case the durability fix turned from data loss into a
    # non-event, and it must not now become a refusal.
    path = _lite_db(tmp_path)
    _write_settings(path, pid=_dead_pid())
    aof = _aof_dir(path)
    os.makedirs(aof, exist_ok=True)
    open(os.path.join(aof, "appendonly.aof.manifest"), "w").write("file x seq 1\n")

    _refuse_untrustworthy_state(path)  # must not raise


# --- Leaked embedded servers -------------------------------------------------


def test_only_servers_this_process_started_are_shut_down(tmp_path) -> None:
    """Never stop a server we merely attached to — it belongs to another process.

    redislite sets ``cleanupregistry`` on the client that spawned the server;
    an attaching client leaves it False. Shutting that one down would take the
    daemon's graph out from under it.
    """
    _OWNED_SERVERS.clear()
    attached = SimpleNamespace(connection=SimpleNamespace(cleanupregistry=False))
    _remember_owned_server(attached, str(tmp_path / "attached.db"))
    assert _OWNED_SERVERS == {}


def test_shutdown_stops_owned_servers_and_forgets_them(tmp_path) -> None:
    """``daemon stop`` must actually stop the server it started.

    redislite's own atexit hook declines to shut down a server with more than
    one connection open, and any process that has issued concurrent queries —
    every daemon — has a pool by then. So the server outlives the daemon,
    reparented to init, holding the db file until reboot.
    """
    _OWNED_SERVERS.clear()
    calls: list[dict] = []
    owner = SimpleNamespace(
        connection=SimpleNamespace(
            cleanupregistry=True,
            shutdown=lambda **kwargs: calls.append(kwargs),
        )
    )
    _remember_owned_server(owner, str(tmp_path / "owned.db"))

    assert shutdown_embedded_servers() == 1
    assert calls == [{"save": True, "now": True, "force": True}]
    # Idempotent: a second stop has nothing left to do.
    assert shutdown_embedded_servers() == 0


def test_shutdown_survives_a_server_that_is_already_gone(tmp_path) -> None:
    _OWNED_SERVERS.clear()

    def _boom(**_kwargs):
        raise ConnectionError("server already gone")

    owner = SimpleNamespace(
        connection=SimpleNamespace(cleanupregistry=True, shutdown=_boom)
    )
    _remember_owned_server(owner, str(tmp_path / "gone.db"))

    assert shutdown_embedded_servers() == 0  # counted only what actually stopped
    assert _OWNED_SERVERS == {}
