"""Pot-archive sandbox-cleanup dispatcher — env gating + bare-cache cross-check."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _row(
    *,
    pot_id: str = "pot-1",
    owner: str = "acme",
    repo: str = "widgets",
    provider_host: str = "github.com",
    added_by_user_id: str = "u1",
) -> SimpleNamespace:
    return SimpleNamespace(
        pot_id=pot_id,
        owner=owner,
        repo=repo,
        provider_host=provider_host,
        added_by_user_id=added_by_user_id,
    )


class _Query:
    def __init__(self, *, rows: list[object]) -> None:
        self.rows = rows

    def filter(self, *_a, **_kw):
        return self

    def order_by(self, *_a):
        return self

    def all(self) -> list[object]:
        return self.rows


def _make_db(*, pot_rows: list[object], other_rows: list[object]) -> MagicMock:
    """Two distinct queries: pot rows, then other-pot rows for GC cross-check."""
    db = MagicMock()
    queries = iter([_Query(rows=pot_rows), _Query(rows=other_rows)])
    db.query.side_effect = lambda _model: next(queries)
    return db


class TestGcBareFlag:
    def test_default_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.modules.context_graph.archive_pot_cleanup import (
            gc_bare_on_pot_delete_enabled,
        )

        monkeypatch.delenv("CONTEXT_ENGINE_GC_BARE_ON_POT_DELETE", raising=False)
        assert gc_bare_on_pot_delete_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        from app.modules.context_graph.archive_pot_cleanup import (
            gc_bare_on_pot_delete_enabled,
        )

        monkeypatch.setenv("CONTEXT_ENGINE_GC_BARE_ON_POT_DELETE", val)
        assert gc_bare_on_pot_delete_enabled() is True


class TestDispatchPotSandboxCleanup:
    def test_no_rows_short_circuits(self) -> None:
        from app.modules.context_graph.archive_pot_cleanup import (
            dispatch_pot_sandbox_cleanup,
        )

        db = MagicMock()
        db.query.return_value = _Query(rows=[])
        with patch("threading.Thread") as thr:
            dispatch_pot_sandbox_cleanup(db, pot_id="pot-empty")
        thr.assert_not_called()

    def test_no_user_id_short_circuits(self) -> None:
        from app.modules.context_graph.archive_pot_cleanup import (
            dispatch_pot_sandbox_cleanup,
        )

        db = MagicMock()
        db.query.return_value = _Query(
            rows=[_row(added_by_user_id=None)]  # type: ignore[arg-type]
        )
        with patch("threading.Thread") as thr:
            dispatch_pot_sandbox_cleanup(db, pot_id="pot-1")
        thr.assert_not_called()

    def test_dispatches_thread_with_gc_false_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.modules.context_graph.archive_pot_cleanup import (
            dispatch_pot_sandbox_cleanup,
        )

        monkeypatch.delenv("CONTEXT_ENGINE_GC_BARE_ON_POT_DELETE", raising=False)
        db = MagicMock()
        db.query.return_value = _Query(rows=[_row()])
        with patch("threading.Thread") as thr:
            dispatch_pot_sandbox_cleanup(db, pot_id="pot-1")
        thr.assert_called_once()
        assert thr.call_args.kwargs["daemon"] is True

    def test_gc_caches_true_when_no_other_pot_references(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.modules.context_graph import archive_pot_cleanup as mod

        monkeypatch.setenv("CONTEXT_ENGINE_GC_BARE_ON_POT_DELETE", "1")
        captured: dict[str, object] = {}

        class FakeClient:
            async def destroy_pot_sandbox(self, **kwargs):  # noqa: ANN003
                captured.update(kwargs)
                return {"workspaces": 1, "repo_caches": 0}

        # Pot's own row, no rows for any other pot — GC eligible.
        db = _make_db(pot_rows=[_row()], other_rows=[])
        with patch.object(mod, "asyncio") as fake_async, patch(
            "app.modules.intelligence.tools.sandbox.client.get_sandbox_client",
            return_value=FakeClient(),
        ), patch("threading.Thread") as thr:
            # Run the dispatch.
            mod.dispatch_pot_sandbox_cleanup(db, pot_id="pot-1")
            # Execute the runner inline to verify the gc_caches arg.
            runner = thr.call_args.kwargs["target"]
            fake_async.run.side_effect = (
                lambda coro: __import__("asyncio").get_event_loop().run_until_complete(coro)
            )
            runner()
        assert captured.get("delete_repo_caches") is True
        assert captured.get("user_id") == "u1"
        assert captured.get("project_id") == "pot-1"

    def test_gc_caches_false_when_another_pot_references(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.modules.context_graph import archive_pot_cleanup as mod

        monkeypatch.setenv("CONTEXT_ENGINE_GC_BARE_ON_POT_DELETE", "1")
        captured: dict[str, object] = {}

        class FakeClient:
            async def destroy_pot_sandbox(self, **kwargs):  # noqa: ANN003
                captured.update(kwargs)
                return {"workspaces": 0, "repo_caches": 0}

        # Another pot still references the same repo.
        db = _make_db(
            pot_rows=[_row()],
            other_rows=[_row(pot_id="pot-other")],
        )
        with patch.object(mod, "asyncio") as fake_async, patch(
            "app.modules.intelligence.tools.sandbox.client.get_sandbox_client",
            return_value=FakeClient(),
        ), patch("threading.Thread") as thr:
            mod.dispatch_pot_sandbox_cleanup(db, pot_id="pot-1")
            runner = thr.call_args.kwargs["target"]
            fake_async.run.side_effect = (
                lambda coro: __import__("asyncio").get_event_loop().run_until_complete(coro)
            )
            runner()
        # Shared cache, GC must be off even though the flag was on.
        assert captured.get("delete_repo_caches") is False
