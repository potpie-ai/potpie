"""Pluggable per-pot token resolver — wiring honors set_pot_token_resolver."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _row(*, owner: str, repo: str, user: str = "u1") -> SimpleNamespace:
    return SimpleNamespace(
        owner=owner,
        repo=repo,
        provider_host="github.com",
        default_branch="main",
        remote_url=f"https://github.com/{owner}/{repo}",
        added_by_user_id=user,
        created_at=None,
    )


class _Q:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def filter(self, *_a, **_kw):
        return self

    def order_by(self, *_a):
        return self

    def all(self) -> list[object]:
        return self.rows


def _db(rows: list[object]) -> MagicMock:
    db = MagicMock()
    db.query.return_value = _Q(rows)
    return db


class TestPluggableTokenResolver:
    def test_custom_resolver_used_for_each_attached_repo(self) -> None:
        from app.modules.context_graph import wiring

        custom_calls: list[tuple[str | None, str | None]] = []

        def _resolver(user_id, repo_name):
            custom_calls.append((user_id, repo_name))
            return f"tok-{repo_name}", "custom_app"

        original = wiring._POT_TOKEN_RESOLVER
        wiring.set_pot_token_resolver(_resolver)
        try:
            build = wiring._build_pot_sandbox_resolver(
                _db([_row(owner="a", repo="x"), _row(owner="b", repo="y", user="u2")])
            )
            cfg = build("pot-1")
        finally:
            wiring.set_pot_token_resolver(None)
            assert wiring._POT_TOKEN_RESOLVER is original

        assert cfg is not None
        assert {r.full_name for r in cfg.repos} == {"a/x", "b/y"}
        # First attacher (oldest row) owns the user_id; per-row tokens still
        # pull from each row's added_by_user_id.
        assert cfg.user_id == "u1"
        assert {(c[0], c[1]) for c in custom_calls} == {("u1", "a/x"), ("u2", "b/y")}
        tokens = {r.full_name: (r.auth_token, r.auth_kind) for r in cfg.repos}
        assert tokens["a/x"] == ("tok-a/x", "custom_app")
        assert tokens["b/y"] == ("tok-b/y", "custom_app")

    def test_resolver_exception_does_not_break_resolver(self) -> None:
        from app.modules.context_graph import wiring

        def _boom(_uid, _repo):
            raise RuntimeError("resolver crash")

        wiring.set_pot_token_resolver(_boom)
        try:
            build = wiring._build_pot_sandbox_resolver(
                _db([_row(owner="a", repo="x")])
            )
            cfg = build("pot-1")
        finally:
            wiring.set_pot_token_resolver(None)

        assert cfg is not None
        attachment = cfg.repos[0]
        # Token resolution failed but the attachment still surfaces.
        assert attachment.auth_token is None
        assert attachment.auth_kind == "error"

    def test_empty_pot_returns_none(self) -> None:
        from app.modules.context_graph import wiring

        build = wiring._build_pot_sandbox_resolver(_db([]))
        assert build("pot-1") is None

    def test_default_resolver_calls_resolve_auth(self) -> None:
        """The default resolver delegates to the existing token chain.

        We patch the chain so the test stays hermetic — the assertion is
        that the wiring's default path actually goes through ``_resolve_auth``
        (not the legacy ``_resolve_auth_token``), preserving the auth kind.
        """
        from unittest.mock import patch

        from app.modules.context_graph import wiring

        # Ensure default sink in place.
        wiring.set_pot_token_resolver(None)

        with patch(
            "app.modules.intelligence.tools.sandbox.client._resolve_auth",
        ) as resolve_auth:
            resolve_auth.return_value = SimpleNamespace(token="t1", kind="app")
            build = wiring._build_pot_sandbox_resolver(
                _db([_row(owner="a", repo="x")])
            )
            cfg = build("pot-1")
        assert cfg is not None
        attachment = cfg.repos[0]
        assert attachment.auth_token == "t1"
        assert attachment.auth_kind == "app"
        resolve_auth.assert_called_once_with("u1", "a/x")
