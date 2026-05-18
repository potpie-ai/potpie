"""Git-history tools wired by ``build_sandbox_tools`` for the agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from adapters.outbound.agent_tools.sandbox import (
    PotSandboxConfig,
    RepoAttachment,
    build_sandbox_tools,
)

pytestmark = pytest.mark.unit


@dataclass
class _Handle:
    workspace_id: str
    local_path: Any = None


@dataclass
class _ExecResult:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""


class _FakeClient:
    def __init__(self) -> None:
        self.acquired: list[str] = []
        self.exec_log: list[list[str]] = []
        self.exec_queue: list[_ExecResult] = []
        self.released = 0

    async def acquire_session(self, **kwargs: Any) -> _Handle:
        self.acquired.append(kwargs["repo"])
        return _Handle(workspace_id=f"ws-{kwargs['repo']}")

    async def release_session(
        self, handle: _Handle, *, destroy_runtime: bool = False
    ) -> None:
        self.released += 1

    async def exec(self, handle: _Handle, cmd, **_kwargs):
        self.exec_log.append(list(cmd))
        if self.exec_queue:
            return self.exec_queue.pop(0)
        return _ExecResult(exit_code=0, stdout=b"", stderr=b"")


def _state() -> SimpleNamespace:
    """Mimic the agent's ``_BatchRunState`` (only fields the builder reads)."""
    return SimpleNamespace(pot_id="pot-1", repo_name=None, cleanup_callbacks=[])


def _build_tools(
    *, repos: list[tuple[str, str]], client: _FakeClient
):
    cfg = PotSandboxConfig(
        user_id="u",
        pot_id="pot-1",
        provider_host="github.com",
        repos=[
            RepoAttachment(owner=o, repo=r, default_branch="main")
            for o, r in repos
        ],
    )

    async def _factory() -> _FakeClient:
        return client

    def _resolver(_pot_id: str) -> PotSandboxConfig:
        return cfg

    builder = build_sandbox_tools(
        client_factory=_factory, pot_resolver=_resolver
    )
    tools = builder(_state())
    by_name = {t.name: t for t in tools}
    return by_name


def _get_func(tool: Any) -> Any:
    # pydantic_ai.Tool stores the callable on either ``function`` or
    # ``func`` depending on version; cover both.
    return getattr(tool, "function", None) or getattr(tool, "func")


@pytest.mark.asyncio
class TestSandboxListRepos:
    async def test_lists_attached_repos(self) -> None:
        client = _FakeClient()
        tools = _build_tools(
            repos=[("a", "x"), ("b", "y")], client=client
        )
        out = await _get_func(tools["sandbox_list_repos"])()
        assert out["pot_id"] == "pot-1"
        assert [r["repo"] for r in out["repos"]] == ["a/x", "b/y"]
        # Listing the repos doesn't materialise any workspace.
        assert client.acquired == []


@pytest.mark.asyncio
class TestAmbiguousAndUnknownRepo:
    async def test_ambiguous_repo_on_multi_repo_pot(self) -> None:
        client = _FakeClient()
        tools = _build_tools(
            repos=[("a", "x"), ("b", "y")], client=client
        )
        out = await _get_func(tools["sandbox_read_file"])(path="README.md")
        assert out.get("error") == "ambiguous_repo"
        assert "a/x" in out["available"] and "b/y" in out["available"]

    async def test_unknown_repo_returns_structured_error(self) -> None:
        client = _FakeClient()
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_read_file"])(
            path="README.md", repo="not/here"
        )
        assert out.get("error") == "unknown_repo"


@pytest.mark.asyncio
class TestSandboxGitLog:
    async def test_issues_correct_command_shape(self) -> None:
        client = _FakeClient()
        client.exec_queue.append(
            _ExecResult(
                exit_code=0,
                stdout=(
                    b"abc123\x1fAlice\x1falice@example.com\x1f"
                    b"2026-05-01T10:00:00+00:00\x1fInitial commit\n"
                    b"def456\x1fBob\x1fbob@example.com\x1f"
                    b"2026-05-02T12:00:00+00:00\x1fFix bug\n"
                ),
            )
        )
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_git_log"])(
            since="2 weeks ago", limit=10
        )
        cmd = client.exec_log[0]
        assert cmd[0:3] == ["git", "log", "--max-count=10"]
        assert any(arg.startswith("--pretty=format:") for arg in cmd)
        assert "--since=2 weeks ago" in cmd
        assert out["count"] == 2
        assert out["commits"][0]["commit"] == "abc123"
        assert out["commits"][1]["author"] == "Bob"

    async def test_caps_limit_to_500(self) -> None:
        client = _FakeClient()
        tools = _build_tools(repos=[("a", "x")], client=client)
        await _get_func(tools["sandbox_git_log"])(limit=10_000)
        assert "--max-count=500" in client.exec_log[0]


@pytest.mark.asyncio
class TestSandboxCheckout:
    async def test_unknown_ref_classified(self) -> None:
        client = _FakeClient()
        client.exec_queue = [
            _ExecResult(
                exit_code=128,
                stderr=b"fatal: couldn't find remote ref bogus",
            ),
        ]
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_checkout"])(ref="bogus")
        assert out["error"]
        assert out["kind"] == "unknown_ref"
        # Only the fetch was attempted (checkout never ran).
        assert len(client.exec_log) == 1

    async def test_happy_path_fetch_then_checkout_then_rev_parse(self) -> None:
        client = _FakeClient()
        client.exec_queue = [
            _ExecResult(exit_code=0),  # fetch
            _ExecResult(exit_code=0),  # checkout
            _ExecResult(exit_code=0, stdout=b"deadbeef\n"),  # rev-parse
        ]
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_checkout"])(ref="main")
        assert out == {"repo": "a/x", "ref": "main", "head_sha": "deadbeef"}
        assert client.exec_log[0][:3] == ["git", "fetch", "origin"]
        assert client.exec_log[1][:3] == ["git", "checkout", "--detach"]
        assert client.exec_log[2] == ["git", "rev-parse", "HEAD"]

    async def test_force_flag_passed_to_checkout(self) -> None:
        client = _FakeClient()
        client.exec_queue = [
            _ExecResult(exit_code=0),
            _ExecResult(exit_code=0),
            _ExecResult(exit_code=0, stdout=b"abc\n"),
        ]
        tools = _build_tools(repos=[("a", "x")], client=client)
        await _get_func(tools["sandbox_checkout"])(ref="main", force=True)
        assert "--force" in client.exec_log[1]

    async def test_checkout_serialized_per_repo(self) -> None:
        """Two concurrent checkouts on the same repo must serialise."""
        client = _FakeClient()

        # The fake's exec returns immediately; serialisation comes from the
        # facade's per-repo asyncio.Lock. We assert exec ordering: every
        # checkout's three calls (fetch, checkout, rev-parse) land
        # contiguously in client.exec_log — no interleaving.
        client.exec_queue = [
            _ExecResult(exit_code=0),
            _ExecResult(exit_code=0),
            _ExecResult(exit_code=0, stdout=b"sha-A\n"),
            _ExecResult(exit_code=0),
            _ExecResult(exit_code=0),
            _ExecResult(exit_code=0, stdout=b"sha-B\n"),
        ]
        tools = _build_tools(repos=[("a", "x")], client=client)
        checkout = _get_func(tools["sandbox_checkout"])

        results = await asyncio.gather(
            checkout(ref="alpha"),
            checkout(ref="beta"),
        )
        # Both completed successfully.
        assert all("head_sha" in r for r in results)
        # Each three-call block stays contiguous.
        cmds = [c[2] if len(c) > 2 else "" for c in client.exec_log]
        # The two blocks: ['alpha', '--detach' or 'alpha', 'HEAD'] +
        # ['beta', '--detach' or 'beta', 'HEAD'] OR the reverse order.
        # Verify no interleaving: refs in fetch positions (indices 0, 3) are
        # different, and same ref appears in the checkout positions (1, 4).
        fetch_refs = {client.exec_log[0][3], client.exec_log[3][3]}
        assert fetch_refs == {"alpha", "beta"}
        # The ref on index 1 (first checkout) matches the ref on index 0
        # (first fetch).
        assert client.exec_log[0][3] == client.exec_log[1][3]
        # Similarly for the second block.
        assert client.exec_log[3][3] == client.exec_log[4][3]


@pytest.mark.asyncio
class TestSandboxGitDiff:
    async def test_diff_with_paths(self) -> None:
        client = _FakeClient()
        client.exec_queue = [
            _ExecResult(exit_code=0, stdout=b"diff --git a/x b/x\n")
        ]
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_git_diff"])(
            base="HEAD~5", head="HEAD", paths=["src/a.py", "src/b.py"]
        )
        assert out["base"] == "HEAD~5"
        assert out["head"] == "HEAD"
        cmd = client.exec_log[0]
        assert "git" in cmd and "diff" in cmd
        assert "HEAD~5..HEAD" in cmd
        # Paths follow a `--` separator.
        idx = cmd.index("--")
        assert cmd[idx + 1 :] == ["src/a.py", "src/b.py"]


class _BlowupClient(_FakeClient):
    """Fake client whose ``acquire_session`` blows up like a transient Daytona
    failure. Mirrors what we saw in production: connection reset by peer
    bubbling out of ``client.create()``."""

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    async def acquire_session(self, **kwargs: Any) -> _Handle:  # type: ignore[override]
        raise self._exc


@pytest.mark.asyncio
class TestSandboxUnavailable:
    """Containment at the tool boundary: an infra failure during
    ``facade.acquire()`` must surface as a structured ``sandbox_unavailable``
    payload, not propagate out of the tool and abort the agent run."""

    async def test_list_dir_contains_acquire_failure(self) -> None:
        client = _BlowupClient(RuntimeError("connection reset by peer"))
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_list_dir"])(path=".")
        assert out["kind"] == "sandbox_unavailable"
        assert out["transient"] is True
        assert "connection reset" in out["error"]
        assert out["path"] == "."

    async def test_read_file_contains_acquire_failure(self) -> None:
        client = _BlowupClient(RuntimeError("snapshot pull failed"))
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_read_file"])(path="README.md")
        assert out["kind"] == "sandbox_unavailable"
        assert out["transient"] is True
        assert out["path"] == "README.md"

    async def test_search_contains_acquire_failure(self) -> None:
        client = _BlowupClient(RuntimeError("daytona 503"))
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_search"])(pattern="needle")
        assert out["kind"] == "sandbox_unavailable"
        assert out["transient"] is True
        assert out["pattern"] == "needle"

    async def test_git_log_contains_acquire_failure(self) -> None:
        client = _BlowupClient(RuntimeError("daytona create failed"))
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_git_log"])()
        assert out["kind"] == "sandbox_unavailable"
        assert out["transient"] is True

    async def test_checkout_contains_acquire_failure(self) -> None:
        client = _BlowupClient(RuntimeError("workspace provider unreachable"))
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_checkout"])(ref="main")
        assert out["kind"] == "sandbox_unavailable"
        assert out["transient"] is True
        assert out["ref"] == "main"

    async def test_repo_resolution_errors_take_priority(self) -> None:
        # An unknown_repo error must come back as `unknown_repo`, not as
        # `sandbox_unavailable` — the third except clause is strictly a
        # fallback for non-LookupError failures.
        client = _BlowupClient(RuntimeError("should not be hit"))
        tools = _build_tools(repos=[("a", "x")], client=client)
        out = await _get_func(tools["sandbox_read_file"])(
            path="x", repo="not/here"
        )
        assert out.get("error") == "unknown_repo"
