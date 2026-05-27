"""C-5: Potpie wiring binds GitHub agent tools to repos attached to the pot."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from adapters.outbound.connectors.github.agent_tools import build_github_tools
from app.modules.context_graph.wiring import _attach_agent_tools

pytestmark = pytest.mark.unit


def _fake_repo_row(
    *,
    pot_id: str = "pot-1",
    owner: str = "acme",
    repo: str = "widgets",
) -> SimpleNamespace:
    return SimpleNamespace(
        pot_id=pot_id,
        owner=owner,
        repo=repo,
    )


class _StubQuery:
    def __init__(self, *, rows: list[object]) -> None:
        self._rows = rows

    def filter(self, *_args, **_kwargs) -> "_StubQuery":
        return self

    def all(self) -> list[object]:
        return list(self._rows)


class _StubDb:
    def __init__(self, *, rows: list[object]) -> None:
        self.rows = rows

    def query(self, _model) -> _StubQuery:
        return _StubQuery(rows=list(self.rows))


class _RecordingAgent:
    def __init__(self) -> None:
        self.builders: list = []

    def add_extra_tools(self, builders: list) -> None:
        self.builders.extend(builders)


class _State:
    def __init__(self, pot_id: str) -> None:
        self.pot_id = pot_id


class _FakePort:
    def get_pull_request(self, repo_name, pr_number, include_diff=False):
        return {"repo_name": repo_name, "pr_number": pr_number, "ok": True}


@contextmanager
def _capture_github_allowlist():
    """Run ``_attach_agent_tools`` with only the GitHub surface wired."""
    captured: dict[str, object] = {}

    def _fake_build_github(source_for_repo, *, allowed_repos_for_pot=None):
        captured["allowed_repos_for_pot"] = allowed_repos_for_pot
        return lambda _state: []

    with (
        patch(
            "adapters.outbound.agent_tools.sandbox.build_sandbox_tools",
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
        patch(
            "adapters.outbound.connectors.github.agent_tools.build_github_tools",
            side_effect=_fake_build_github,
        ),
    ):
        yield captured


def test_wiring_passes_pot_attached_repos_to_github_tools() -> None:
    rows = [
        _fake_repo_row(owner="Acme", repo="Alpha"),
        _fake_repo_row(owner="org", repo="beta"),
    ]
    agent = _RecordingAgent()
    with _capture_github_allowlist() as captured:
        _attach_agent_tools(
            agent,
            _StubDb(rows=rows),  # type: ignore[arg-type]
            source_for_repo=lambda _: _FakePort(),
        )

    allow_fn = captured["allowed_repos_for_pot"]
    assert callable(allow_fn)
    assert allow_fn("pot-1") == {"acme/alpha", "org/beta"}  # type: ignore[operator]
    assert len(agent.builders) == 1


def test_wiring_allowlist_blocks_foreign_repo_before_github_api() -> None:
    rows = [_fake_repo_row(owner="acme", repo="widgets")]
    agent = _RecordingAgent()
    with _capture_github_allowlist() as captured:
        _attach_agent_tools(
            agent,
            _StubDb(rows=rows),  # type: ignore[arg-type]
            source_for_repo=lambda _: _FakePort(),
        )

    allow_fn = captured["allowed_repos_for_pot"]
    calls: list[str] = []

    def source_for_repo(repo_name: str) -> _FakePort:
        calls.append(repo_name)
        return _FakePort()

    tools = build_github_tools(
        source_for_repo,
        allowed_repos_for_pot=allow_fn,  # type: ignore[arg-type]
    )(_State("pot-1"))
    by_name = {t.name: t.function for t in tools}

    out = by_name["github_get_pull_request"]("acme/widgets", 3)
    assert out == {"repo_name": "acme/widgets", "pr_number": 3, "ok": True}
    assert calls == ["acme/widgets"]

    blocked = by_name["github_get_pull_request"]("victim/private", 1)
    assert blocked == {"error": "unknown_repo", "repo_name": "victim/private"}
    assert calls == ["acme/widgets"]
