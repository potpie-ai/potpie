"""Reconciliation agent receives the pot-scoped sandbox tool builder."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.modules.context_graph.wiring import (
    _attach_agent_tools,
    _build_pot_sandbox_resolver,
)

pytestmark = pytest.mark.unit


def _fake_repo_row(
    *,
    pot_id: str = "pot-1",
    owner: str = "Acme",
    repo: str = "Widget",
    default_branch: str | None = "main",
    remote_url: str | None = "https://github.com/Acme/Widget",
    added_by_user_id: str = "user-42",
    provider_host: str = "github.com",
) -> object:
    return SimpleNamespace(
        pot_id=pot_id,
        owner=owner,
        repo=repo,
        default_branch=default_branch,
        remote_url=remote_url,
        added_by_user_id=added_by_user_id,
        provider_host=provider_host,
    )


class _StubQuery:
    def __init__(self, *, rows: list[object]) -> None:
        self._rows = rows
        self.filter_calls: list[tuple] = []

    def filter(self, *args, **_kwargs) -> "_StubQuery":
        self.filter_calls.append(args)
        return self

    def order_by(self, *_args, **_kwargs) -> "_StubQuery":
        return self

    def first(self) -> object | None:
        return self._rows[0] if self._rows else None

    def all(self) -> list[object]:
        return list(self._rows)


class _StubDb:
    def __init__(self, *, rows: list[object]) -> None:
        self.rows = rows

    def query(self, _model) -> _StubQuery:
        return _StubQuery(rows=list(self.rows))


class TestPotSandboxResolver:
    def test_returns_config_for_single_repo_pot(self) -> None:
        row = _fake_repo_row()
        db = _StubDb(rows=[row])
        with patch(
            "app.modules.intelligence.tools.sandbox.client._resolve_auth",
            return_value=SimpleNamespace(token="ghs_TOKEN", kind="app"),
        ):
            resolver = _build_pot_sandbox_resolver(db)  # type: ignore[arg-type]
            cfg = resolver("pot-1")
        assert cfg is not None
        assert cfg.user_id == "user-42"
        assert cfg.pot_id == "pot-1"
        assert cfg.provider_host == "github.com"
        assert len(cfg.repos) == 1
        attachment = cfg.repos[0]
        assert attachment.full_name == "Acme/Widget"
        assert attachment.default_branch == "main"
        assert attachment.repo_url == "https://github.com/Acme/Widget"
        assert attachment.auth_token == "ghs_TOKEN"
        assert attachment.auth_kind == "app"

    def test_loads_all_repos_for_multi_repo_pot(self) -> None:
        rows = [
            _fake_repo_row(owner="orgA", repo="alpha", added_by_user_id="u1"),
            _fake_repo_row(owner="orgB", repo="beta", added_by_user_id="u2"),
        ]
        db = _StubDb(rows=rows)
        with patch(
            "app.modules.intelligence.tools.sandbox.client._resolve_auth",
            side_effect=lambda uid, _repo: SimpleNamespace(token=f"tok-{uid}", kind="app"),
        ):
            resolver = _build_pot_sandbox_resolver(db)  # type: ignore[arg-type]
            cfg = resolver("pot-1")
        assert cfg is not None
        # First attacher owns the sandbox container.
        assert cfg.user_id == "u1"
        assert [r.full_name for r in cfg.repos] == ["orgA/alpha", "orgB/beta"]
        assert [r.auth_token for r in cfg.repos] == ["tok-u1", "tok-u2"]
        assert [r.auth_kind for r in cfg.repos] == ["app", "app"]

    def test_returns_none_when_pot_has_no_repos(self) -> None:
        db = _StubDb(rows=[])
        with patch(
            "app.modules.intelligence.tools.sandbox.client._resolve_auth",
            return_value=SimpleNamespace(token=None, kind="none"),
        ):
            resolver = _build_pot_sandbox_resolver(db)  # type: ignore[arg-type]
            assert resolver("pot-empty") is None

    def test_defaults_branch_to_main_when_missing(self) -> None:
        row = _fake_repo_row(default_branch=None)
        db = _StubDb(rows=[row])
        with patch(
            "app.modules.intelligence.tools.sandbox.client._resolve_auth",
            return_value=SimpleNamespace(token=None, kind="none"),
        ):
            cfg = _build_pot_sandbox_resolver(db)("pot-1")  # type: ignore[arg-type]
        assert cfg is not None
        assert cfg.repos[0].default_branch == "main"


class _RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[list] = []

    def add_extra_tools(self, builders: list) -> None:
        self.calls.append(list(builders))


def _noop_source_for_repo(_repo_name: str) -> SimpleNamespace:
    return SimpleNamespace()


@contextmanager
def _isolate_sandbox_surface():
    """GitHub/Linear/web surfaces are tested elsewhere; isolate sandbox wiring."""
    with (
        patch(
            "adapters.outbound.connectors.github.agent_tools.build_github_tools",
            side_effect=RuntimeError("patched for test"),
        ),
        patch(
            "adapters.outbound.connectors.linear.agent_tools.build_linear_tools",
            side_effect=RuntimeError("patched for test"),
        ),
        patch(
            "app.modules.context_graph.agent_web_tools.build_web_tools",
            side_effect=RuntimeError("patched for test"),
        ),
    ):
        yield


class TestAttachSandboxTools:
    def test_attaches_one_builder_to_agent(self) -> None:
        agent = _RecordingAgent()
        db = _StubDb(rows=[_fake_repo_row()])
        with _isolate_sandbox_surface():
            _attach_agent_tools(
                agent,
                db,  # type: ignore[arg-type]
                source_for_repo=_noop_source_for_repo,
            )
        assert len(agent.calls) == 1
        assert len(agent.calls[0]) == 1
        assert callable(agent.calls[0][0])

    def test_noop_when_agent_is_none(self) -> None:
        _attach_agent_tools(
            None,
            _StubDb(rows=[]),  # type: ignore[arg-type]
            source_for_repo=_noop_source_for_repo,
        )

    def test_env_flag_disables_attachment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_DISABLE_SANDBOX_TOOLS", "1")
        agent = _RecordingAgent()
        with _isolate_sandbox_surface():
            _attach_agent_tools(
                agent,
                _StubDb(rows=[]),  # type: ignore[arg-type]
                source_for_repo=_noop_source_for_repo,
            )
        assert agent.calls == []

    def test_agent_without_add_extra_tools_is_silent(self) -> None:
        sentinel = SimpleNamespace()
        _attach_agent_tools(
            sentinel,
            _StubDb(rows=[]),  # type: ignore[arg-type]
            source_for_repo=_noop_source_for_repo,
        )
