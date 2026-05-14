"""ContextReaderRegistry: routing, manifest, and merge behaviour."""

from __future__ import annotations

import pytest

from application.services.context_reader_registry import ContextReaderRegistry
from domain.context_reader import ReaderCapability, ReaderCost, ReaderResult
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphScope,
    ContextGraphStrategy,
)


class _FakeReader:
    """Programmable in-memory reader for routing tests."""

    def __init__(
        self,
        family: str,
        *,
        intents: frozenset[str] = frozenset(),
        requires_scope: frozenset[str] = frozenset(),
        result: object = None,
        compat: bool = False,
        explode: bool = False,
    ) -> None:
        self._family = family
        self._intents = intents
        self._requires_scope = requires_scope
        self._result = result if result is not None else [{"family": family}]
        self._compat = compat
        self._explode = explode
        self.calls: list[ContextGraphQuery] = []

    def family(self) -> str:
        return self._family

    def capability(self) -> ReaderCapability:
        return ReaderCapability(
            family=self._family,
            description=f"fake reader {self._family}",
            intents=self._intents,
            requires_scope=self._requires_scope,
            cost=ReaderCost(label="cheap"),
            backend="structural",
            compat=self._compat,
        )

    def read(self, request: ContextGraphQuery) -> ReaderResult:
        self.calls.append(request)
        if self._explode:
            raise RuntimeError(f"{self._family} kaboom")
        rows = self._result if isinstance(self._result, list) else [self._result]
        return ReaderResult(
            family=self._family,
            result=rows,
            count=len(rows) if isinstance(rows, list) else None,
            compat=self._compat,
        )


def test_register_and_lookup_by_family() -> None:
    reg = ContextReaderRegistry()
    a = _FakeReader("decisions")
    b = _FakeReader("owners")
    reg.register(a)
    reg.register(b)

    assert reg.get("DECISIONS") is a  # case-insensitive
    assert reg.get("missing") is None
    assert reg.families() == ("decisions", "owners")


def test_double_register_rejects() -> None:
    reg = ContextReaderRegistry()
    reg.register(_FakeReader("decisions"))
    with pytest.raises(ValueError):
        reg.register(_FakeReader("decisions"))


def test_explicit_include_routes_to_reader() -> None:
    reg = ContextReaderRegistry()
    reader = _FakeReader("decisions", result=[{"id": "d1"}])
    reg.register(reader)

    out = reg.execute(
        ContextGraphQuery(pot_id="p1", include=["decisions"])
    )

    assert out.kind == "decisions"
    assert out.result == [{"id": "d1"}]
    assert out.error is None
    assert reader.calls and reader.calls[0].pot_id == "p1"


def test_unknown_include_token_becomes_fallback_not_error() -> None:
    reg = ContextReaderRegistry()
    reg.register(_FakeReader("decisions"))

    out = reg.execute(
        ContextGraphQuery(pot_id="p1", include=["decisions", "bogus_family"])
    )

    fallbacks = out.meta.get("fallbacks") or []
    assert any(
        f["family"] == "bogus_family" and f["reason"] == "unsupported_include"
        for f in fallbacks
    )
    assert out.kind == "decisions"  # decisions still ran


def test_missing_required_scope_emits_fallback() -> None:
    reg = ContextReaderRegistry()
    reg.register(
        _FakeReader("owners", requires_scope=frozenset({"file_path"}))
    )

    out = reg.execute(
        ContextGraphQuery(pot_id="p1", include=["owners"])
    )

    assert out.error == "unsupported_context_graph_query"
    fb = out.meta.get("fallbacks") or []
    assert any(f["family"] == "owners" and f["reason"] == "missing_scope" for f in fb)


def test_strategy_auto_selects_semantic_search_when_query_present() -> None:
    reg = ContextReaderRegistry()
    semantic = _FakeReader("semantic_search", result=[{"uuid": "s1"}])
    reg.register(semantic)

    out = reg.execute(
        ContextGraphQuery(
            pot_id="p1",
            query="auth",
            strategy=ContextGraphStrategy.HYBRID,
        )
    )

    assert semantic.calls
    assert out.kind == "semantic_search"


def test_aggregate_goal_without_include_routes_to_graph_overview() -> None:
    reg = ContextReaderRegistry()
    overview = _FakeReader("graph_overview", result={"totals": {"entities": 0}})
    reg.register(overview)

    out = reg.execute(
        ContextGraphQuery(pot_id="p1", goal=ContextGraphGoal.AGGREGATE)
    )

    assert overview.calls
    assert out.kind == "graph_overview"


def test_neighborhood_goal_without_include_routes_to_project_graph() -> None:
    reg = ContextReaderRegistry()
    pg = _FakeReader("project_graph", result={"nodes": []})
    reg.register(pg)

    out = reg.execute(
        ContextGraphQuery(pot_id="p1", goal=ContextGraphGoal.NEIGHBORHOOD)
    )

    assert pg.calls
    assert out.kind == "project_graph"


def test_timeline_goal_picks_change_history_for_code_scoped_anchor() -> None:
    reg = ContextReaderRegistry()
    change = _FakeReader("change_history", result=[{"pr": 1}])
    timeline = _FakeReader("timeline", result={"activities": []})
    reg.register(change)
    reg.register(timeline)

    out = reg.execute(
        ContextGraphQuery(
            pot_id="p1",
            goal=ContextGraphGoal.TIMELINE,
            scope=ContextGraphScope(file_path="app/auth.py"),
        )
    )

    assert change.calls and not timeline.calls
    assert out.kind == "change_history"


def test_timeline_goal_picks_timeline_for_actor_anchor() -> None:
    reg = ContextReaderRegistry()
    change = _FakeReader("change_history", result=[{"pr": 1}])
    timeline = _FakeReader("timeline", result={"activities": []})
    reg.register(change)
    reg.register(timeline)

    out = reg.execute(
        ContextGraphQuery(
            pot_id="p1",
            goal=ContextGraphGoal.TIMELINE,
            scope=ContextGraphScope(user="alice"),
        )
    )

    assert timeline.calls and not change.calls
    assert out.kind == "timeline"


def test_multi_family_includes_merge_into_multi_envelope() -> None:
    reg = ContextReaderRegistry()
    sem = _FakeReader("semantic_search", result=[{"uuid": "s1"}])
    dec = _FakeReader("decisions", result=[{"id": "d1"}])
    reg.register(sem)
    reg.register(dec)

    out = reg.execute(
        ContextGraphQuery(
            pot_id="p1",
            query="auth",
            strategy=ContextGraphStrategy.HYBRID,
            include=["semantic_search", "decisions"],
        )
    )

    assert out.kind == "multi"
    assert set(out.result.keys()) == {"semantic_search", "decisions"}
    assert out.meta.get("merge") == "multi"


def test_reader_exception_becomes_fallback_not_raise() -> None:
    reg = ContextReaderRegistry()
    reg.register(_FakeReader("semantic_search", explode=True))
    reg.register(_FakeReader("decisions", result=[{"id": "d1"}]))

    out = reg.execute(
        ContextGraphQuery(
            pot_id="p1",
            query="auth",
            strategy=ContextGraphStrategy.HYBRID,
            include=["semantic_search", "decisions"],
        )
    )

    assert out.kind == "multi"
    assert "decisions" in out.result
    assert "semantic_search" not in out.result
    fallbacks = out.meta.get("fallbacks") or []
    assert any(
        fb["family"] == "semantic_search" and fb["reason"] == "executor_error"
        for fb in fallbacks
    )


def test_compat_reader_stamps_meta() -> None:
    reg = ContextReaderRegistry()
    reg.register(
        _FakeReader(
            "pr_diff",
            requires_scope=frozenset({"pr_number"}),
            result=[{"file_path": "a.py"}],
            compat=True,
        )
    )

    out = reg.execute(
        ContextGraphQuery(
            pot_id="p1",
            include=["pr_diff"],
            scope=ContextGraphScope(pr_number=5),
        )
    )

    assert out.kind == "pr_diff"
    assert out.meta.get("compat") is True
    assert out.meta["legs"][0]["compat"] is True


def test_manifest_reflects_registered_readers() -> None:
    reg = ContextReaderRegistry()
    reg.register(_FakeReader("decisions", intents=frozenset({"feature"})))
    reg.register(
        _FakeReader("owners", requires_scope=frozenset({"file_path"}))
    )

    manifest = {entry.family: entry for entry in reg.manifest()}

    assert set(manifest) == {"decisions", "owners"}
    assert manifest["decisions"].intents == ("feature",)
    assert manifest["owners"].requires_scope == ("file_path",)


def test_third_reader_smoke_test_release_notes_loads() -> None:
    """Phase 3 smoke test: ``ReleaseNotesReader`` registers without
    touching ``application/`` or ``domain/``. Adding the reader was a
    single-file change under ``adapters/outbound/readers/``."""
    from unittest.mock import MagicMock

    from adapters.outbound.readers import ReleaseNotesReader

    reg = ContextReaderRegistry()
    reg.register(ReleaseNotesReader(structural=MagicMock()))

    families = {entry.family for entry in reg.manifest()}
    assert "release_notes" in families
