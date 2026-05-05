"""End-to-end coverage of the Potpie-bot identity + remote auth wiring.

This pins the contracts that make "every sandbox action attributed to
the Potpie bot" actually work:

* Commits default to the configured ``BotIdentityProvider`` author —
  ``GIT_AUTHOR_NAME``/``EMAIL`` end up on the commit object.
* The runtime spec stamps the same identity as fallback so raw
  ``sandbox_shell git commit`` is also bot-attributed.
* Explicit ``author=(...)`` on ``SandboxClient.commit`` overrides the
  default.
* ``SandboxClient.push`` resolves a fresh token from
  ``RemoteAuthProvider`` per call and prepends
  ``-c http.<host>.extraheader='AUTHORIZATION: bearer …'`` so push to a
  remote that requires auth works without ever persisting the token to
  ``.git/config``.

All tests use a hermetic on-disk fixture upstream — no network, no
GitHub. The auth providers are stub implementations of the new ports
so the tests don't depend on Potpie's full DB / GitHub-App machinery.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import AsyncIterator

import pytest

from sandbox import (
    Author,
    CommandKind,
    RemoteAuth,
    RepoIdentity,
    SandboxClient,
    SandboxSettings,
    WorkspaceMode,
    build_sandbox_container,
)


REPO_NAME = "owner/test-repo"


# ----------------------------------------------------------------------
# Stub provider implementations
# ----------------------------------------------------------------------
class _StubBotIdentity:
    """Minimal :class:`BotIdentityProvider` for the tests.

    Always returns the same author regardless of repo; the production
    Potpie adapter does the same thing (bot identity is repo-agnostic
    by design).
    """

    def __init__(self, name: str = "test-bot[bot]", email: str = "bot@example.test"):
        self._author = Author(name=name, email=email)
        self.calls: list[tuple[str, str | None]] = []

    async def identity_for_repo(
        self, *, repo: RepoIdentity, user_id: str | None = None
    ) -> Author:
        self.calls.append((repo.repo_name, user_id))
        return self._author


class _StubRemoteAuth:
    """Capture-and-return :class:`RemoteAuthProvider` for tests."""

    def __init__(self, token: str = "tok_test_123", kind: str = "app"):
        self._auth = RemoteAuth(token=token, kind=kind)
        self.calls: list[tuple[str, str | None]] = []

    async def auth_for_remote(
        self, *, repo: RepoIdentity, user_id: str | None = None
    ) -> RemoteAuth:
        self.calls.append((repo.repo_name, user_id))
        return self._auth


# ----------------------------------------------------------------------
# Fixture: build a client wired with the stubs
# ----------------------------------------------------------------------
@pytest.fixture
async def client_with_bot(
    repos_base: Path, metadata_path: Path
) -> AsyncIterator[tuple[SandboxClient, _StubBotIdentity, _StubRemoteAuth]]:
    bot = _StubBotIdentity()
    auth = _StubRemoteAuth()
    settings = SandboxSettings(
        provider="local",
        runtime="local_subprocess",
        repos_base_path=str(repos_base),
        metadata_path=str(metadata_path),
        local_allow_write=True,
    )
    container = build_sandbox_container(
        settings,
        bot_identity_provider=bot,
        remote_auth_provider=auth,
    )
    yield SandboxClient.from_container(container), bot, auth


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


# ======================================================================
# Bot identity on commits
# ======================================================================
class TestCommitIdentity:
    """``client.commit`` must default the author/committer to the
    configured :class:`BotIdentityProvider` when no explicit ``author=``
    is passed.

    This is the load-bearing "every commit goes to the bot" property —
    a regression here puts user emails on PR commits instead of the bot.
    """

    async def test_commit_defaults_to_bot_identity(
        self,
        client_with_bot: tuple[SandboxClient, _StubBotIdentity, _StubRemoteAuth],
        upstream_repo: Path,
    ) -> None:
        client, bot, _ = client_with_bot
        handle = await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-bot", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="bot-id",
        )
        await client.write_file(handle, "agent.txt", b"hello\n")
        sha = await client.commit(handle, "agent commit")
        assert len(sha) == 40

        worktree = Path(handle.local_path)
        # Author and committer both stamped from the bot identity.
        assert _git(["log", "-1", "--format=%an"], worktree) == "test-bot[bot]"
        assert _git(["log", "-1", "--format=%ae"], worktree) == "bot@example.test"
        assert _git(["log", "-1", "--format=%cn"], worktree) == "test-bot[bot]"
        assert _git(["log", "-1", "--format=%ce"], worktree) == "bot@example.test"

        # And the provider was invoked with the right repo identity.
        assert bot.calls, "bot identity provider was never asked"
        called_repo, called_user = bot.calls[-1]
        assert called_repo == REPO_NAME
        assert called_user == "u1"

    async def test_explicit_author_overrides_bot(
        self,
        client_with_bot: tuple[SandboxClient, _StubBotIdentity, _StubRemoteAuth],
        upstream_repo: Path,
    ) -> None:
        """A caller passing ``author=`` must win over the provider —
        this is the override seam for "act as the user" workflows.
        """
        client, _, _ = client_with_bot
        handle = await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-override", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="override",
        )
        await client.write_file(handle, "x.txt", b"x\n")
        await client.commit(
            handle, "user commit",
            author=("Real User", "user@example.com"),
        )
        worktree = Path(handle.local_path)
        assert _git(["log", "-1", "--format=%an"], worktree) == "Real User"
        assert _git(["log", "-1", "--format=%ae"], worktree) == "user@example.com"

    async def test_runtime_env_carries_identity_for_raw_shell_commits(
        self,
        client_with_bot: tuple[SandboxClient, _StubBotIdentity, _StubRemoteAuth],
        upstream_repo: Path,
    ) -> None:
        """The service stamps ``GIT_AUTHOR_*`` / ``GIT_COMMITTER_*``
        into the runtime spec env at ``get_or_create_runtime`` time.

        That covers the case where the agent runs ``git commit`` via
        ``sandbox_shell`` (raw exec) instead of ``client.commit``.
        Without this, raw shell commits would fall back to
        ``git config user.email`` (often empty in containers, or the
        host operator's identity).
        """
        client, _, _ = client_with_bot
        handle = await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-rawcommit", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="rawcommit",
        )
        # Touch a file via raw shell, then stage+commit via raw shell
        # — no helper involved.
        write = await client.exec(
            handle,
            ["sh", "-c", "echo raw > raw.txt"],
            command_kind=CommandKind.WRITE,
        )
        assert write.exit_code == 0
        add = await client.exec(
            handle, ["git", "add", "-A"], command_kind=CommandKind.WRITE
        )
        assert add.exit_code == 0
        commit = await client.exec(
            handle,
            ["git", "commit", "-m", "raw shell commit"],
            command_kind=CommandKind.WRITE,
        )
        assert commit.exit_code == 0, commit.stderr.decode()

        worktree = Path(handle.local_path)
        assert _git(["log", "-1", "--format=%an"], worktree) == "test-bot[bot]"
        assert _git(["log", "-1", "--format=%ae"], worktree) == "bot@example.test"


# ======================================================================
# Remote auth on push
# ======================================================================
class TestPushAuthInjection:
    """``client.push`` must resolve a token from
    :class:`RemoteAuthProvider` *per call* and inject it via
    ``-c http.<host>.extraheader=…`` rather than persisting it.

    The bare clone's ``origin`` URL was scrubbed at clone time on
    purpose. Push has to re-acquire its own credential.
    """

    async def test_push_consults_remote_auth_provider(
        self,
        client_with_bot: tuple[SandboxClient, _StubBotIdentity, _StubRemoteAuth],
        upstream_repo: Path,
    ) -> None:
        """The provider's ``auth_for_remote`` is called and the result
        is fresh per call (not cached at acquire time).
        """
        client, _, auth = client_with_bot
        handle = await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-push", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="push",
        )
        await client.write_file(handle, "p.txt", b"push test\n")
        await client.commit(handle, "before push")

        # Push should succeed because the local upstream allows
        # `receive.denyCurrentBranch=updateInstead` (set by the fixture).
        before_calls = len(auth.calls)
        await client.push(handle)
        # The provider was queried at least once during push.
        assert len(auth.calls) > before_calls
        called_repo, called_user = auth.calls[-1]
        assert called_repo == REPO_NAME
        assert called_user == "u1"

        # Each push re-resolves; calling push twice doubles the calls.
        await client.write_file(handle, "p.txt", b"push test 2\n")
        await client.commit(handle, "second push")
        before_second = len(auth.calls)
        await client.push(handle)
        assert len(auth.calls) > before_second, (
            "push must re-resolve auth on every call — installation tokens "
            "expire in 1h and acquire-time caching would silently break "
            "long-running conversations"
        )

    async def test_push_does_not_persist_token_to_git_config(
        self,
        client_with_bot: tuple[SandboxClient, _StubBotIdentity, _StubRemoteAuth],
        upstream_repo: Path,
        repos_base: Path,
    ) -> None:
        """The token must travel via ``-c …`` flags only — never landing
        in any ``.git/config`` on disk. This is the security claim:
        process-wide bot tokens stay out of persistent state where a
        future user of the same cache could read them.
        """
        client, _, _ = client_with_bot
        handle = await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-no-persist", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="no-persist",
        )
        await client.write_file(handle, "p.txt", b"x\n")
        await client.commit(handle, "x")
        await client.push(handle)

        # Walk every .git/config under .repos/ and assert no token leak.
        token_marker = "tok_test_123"
        for config_file in repos_base.rglob(".git/config"):
            assert token_marker not in config_file.read_text(), (
                f"token leaked into {config_file}"
            )
        # Bare repo's config too.
        for config_file in repos_base.rglob("config"):
            if "config.tmp" in config_file.name:
                continue
            try:
                contents = config_file.read_text()
            except (UnicodeDecodeError, PermissionError):
                continue
            assert token_marker not in contents, f"token leaked into {config_file}"

    async def test_push_works_without_remote_auth_provider(
        self, repos_base: Path, metadata_path: Path, upstream_repo: Path
    ) -> None:
        """When no :class:`RemoteAuthProvider` is wired (e.g. the dev
        default), push still works against repos that don't require
        auth — the absence of a provider degrades to "push without
        injecting a header", which is what every existing public-repo
        flow relies on.
        """
        settings = SandboxSettings(
            provider="local", runtime="local_subprocess",
            repos_base_path=str(repos_base),
            metadata_path=str(metadata_path),
            local_allow_write=True,
        )
        container = build_sandbox_container(settings)  # no auth provider
        client = SandboxClient.from_container(container)
        handle = await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-noauth", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="noauth",
        )
        await client.write_file(handle, "x.txt", b"x\n")
        await client.commit(
            handle, "no auth", author=("X", "x@x.com"),
        )
        # Local fixture upstream accepts pushes without auth.
        await client.push(handle)


# ======================================================================
# Provider-less defaults still work
# ======================================================================
class TestNoProviderRegression:
    """Sanity: a SandboxClient with no bot/remote-auth providers wired
    behaves the same way it did before this feature existed.

    This guards against accidentally making the providers mandatory.
    """

    async def test_commit_with_explicit_author_only(
        self, repos_base: Path, metadata_path: Path, upstream_repo: Path
    ) -> None:
        settings = SandboxSettings(
            provider="local", runtime="local_subprocess",
            repos_base_path=str(repos_base),
            metadata_path=str(metadata_path),
            local_allow_write=True,
        )
        container = build_sandbox_container(settings)
        client = SandboxClient.from_container(container)
        handle = await client.acquire_session(
            user_id="u1", project_id="p1",
            repo=REPO_NAME, repo_url=str(upstream_repo),
            branch="agent/edits-noprovider", base_ref="main", create_branch=True,
            mode=WorkspaceMode.EDIT, conversation_id="noprovider",
        )
        await client.write_file(handle, "y.txt", b"y\n")
        # No bot identity provider, so the caller must pass author=.
        sha = await client.commit(
            handle, "explicit author", author=("Y", "y@y.com")
        )
        assert len(sha) == 40
        worktree = Path(handle.local_path)
        assert _git(["log", "-1", "--format=%ae"], worktree) == "y@y.com"
