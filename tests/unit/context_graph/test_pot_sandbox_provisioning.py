"""Prewarm dispatcher — env disable, no-user short-circuit, thread shape."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


class TestPrewarmDisabled:
    def test_default_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.modules.context_graph.pot_sandbox_provisioning import (
            prewarm_disabled,
        )

        monkeypatch.delenv("CONTEXT_ENGINE_DISABLE_SANDBOX_PREWARM", raising=False)
        assert prewarm_disabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        from app.modules.context_graph.pot_sandbox_provisioning import (
            prewarm_disabled,
        )

        monkeypatch.setenv("CONTEXT_ENGINE_DISABLE_SANDBOX_PREWARM", val)
        assert prewarm_disabled() is True


class TestDispatchPotRepoPrewarm:
    def test_no_user_id_short_circuits(self) -> None:
        from app.modules.context_graph.pot_sandbox_provisioning import (
            dispatch_pot_repo_prewarm,
        )

        with patch("threading.Thread") as thr:
            dispatch_pot_repo_prewarm(
                user_id=None,
                pot_id="pot-1",
                owner="acme",
                repo="widgets",
                default_branch="main",
                repo_url=None,
            )
        thr.assert_not_called()

    def test_disabled_env_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.modules.context_graph.pot_sandbox_provisioning import (
            dispatch_pot_repo_prewarm,
        )

        monkeypatch.setenv("CONTEXT_ENGINE_DISABLE_SANDBOX_PREWARM", "1")
        with patch("threading.Thread") as thr:
            dispatch_pot_repo_prewarm(
                user_id="u1",
                pot_id="pot-1",
                owner="acme",
                repo="widgets",
                default_branch="main",
                repo_url=None,
            )
        thr.assert_not_called()

    def test_dispatches_daemon_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.modules.context_graph.pot_sandbox_provisioning import (
            dispatch_pot_repo_prewarm,
        )

        monkeypatch.delenv("CONTEXT_ENGINE_DISABLE_SANDBOX_PREWARM", raising=False)
        with patch("threading.Thread") as thr:
            dispatch_pot_repo_prewarm(
                user_id="u1",
                pot_id="pot-1",
                owner="acme",
                repo="widgets",
                default_branch="main",
                repo_url="https://github.com/acme/widgets",
            )
        thr.assert_called_once()
        assert thr.call_args.kwargs["daemon"] is True
        # Name should encode pot + repo for ops diagnostics.
        assert "pot-1" in thr.call_args.kwargs["name"]
        assert "acme" in thr.call_args.kwargs["name"]


class TestResolveDefaultBranch:
    """Cover the discovery helper that replaced the hardcoded `main` fallback.

    The crash in production was: repo with default ``master`` got prewarmed
    against ``main`` (the old hard fallback) and ``worktree add main`` died
    with ``invalid reference: main``. Discovery must parse symref output and
    refuse to guess on failure.
    """

    def test_parses_ls_remote_symref(self) -> None:
        from app.modules.context_graph.pot_sandbox_provisioning import (
            _resolve_default_branch,
        )

        completed = SimpleNamespace(
            returncode=0,
            stdout="ref: refs/heads/master\tHEAD\nabc123\tHEAD\n",
            stderr="",
        )
        with patch(
            "app.modules.context_graph.pot_sandbox_provisioning.subprocess.run",
            return_value=completed,
        ):
            branch = _resolve_default_branch(
                repo_url="https://github.com/acme/widgets.git",
                owner="acme",
                repo="widgets",
                token=None,
            )
        assert branch == "master"

    def test_returns_none_on_nonzero_exit(self) -> None:
        from app.modules.context_graph.pot_sandbox_provisioning import (
            _resolve_default_branch,
        )

        completed = SimpleNamespace(returncode=128, stdout="", stderr="boom")
        with patch(
            "app.modules.context_graph.pot_sandbox_provisioning.subprocess.run",
            return_value=completed,
        ):
            assert (
                _resolve_default_branch(
                    repo_url=None, owner="acme", repo="widgets", token=None
                )
                is None
            )

    def test_returns_none_on_timeout(self) -> None:
        from app.modules.context_graph.pot_sandbox_provisioning import (
            _resolve_default_branch,
        )

        with patch(
            "app.modules.context_graph.pot_sandbox_provisioning.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=15),
        ):
            assert (
                _resolve_default_branch(
                    repo_url=None, owner="acme", repo="widgets", token=None
                )
                is None
            )

    def test_embeds_token_into_https_url(self) -> None:
        """Token must be carried in the auth url so private repos resolve."""
        from app.modules.context_graph.pot_sandbox_provisioning import (
            _resolve_default_branch,
        )

        completed = SimpleNamespace(
            returncode=0,
            stdout="ref: refs/heads/main\tHEAD\n",
            stderr="",
        )
        with patch(
            "app.modules.context_graph.pot_sandbox_provisioning.subprocess.run",
            return_value=completed,
        ) as run:
            _resolve_default_branch(
                repo_url="https://github.com/acme/private.git",
                owner="acme",
                repo="private",
                token="ghs_redacted",
            )
        invoked_url = run.call_args.args[0][3]
        assert invoked_url.startswith("https://x-access-token:ghs_redacted@github.com/")
