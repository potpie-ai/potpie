"""M-2: ``attach_repo_to_pot`` SSRF guard on ``provider_host``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.modules.context_graph.attach_repo_to_pot import (
    _allowed_provider_hosts,
    attach_repo_to_pot,
)

pytestmark = pytest.mark.unit


def _fake_pot(*, pot_id: str = "pot-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=pot_id,
        primary_repo_name=None,
        user_id="owner-1",
    )


class _Query:
    def __init__(self, *, result: object | None) -> None:
        self.result = result

    def filter(self, *_args, **_kwargs) -> "_Query":
        return self

    def first(self) -> object | None:
        return self.result


def _make_db(*, query_results: list[object | None]) -> MagicMock:
    db = MagicMock()
    queries = iter(query_results)

    def _query(_model) -> _Query:
        return _Query(result=next(queries, None))

    db.query.side_effect = _query
    return db


class TestAllowedProviderHosts:
    def test_default_includes_github_com(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CONTEXT_ENGINE_ALLOWED_PROVIDER_HOSTS", raising=False)
        assert _allowed_provider_hosts() == {"github.com"}

    def test_env_adds_github_enterprise_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CONTEXT_ENGINE_ALLOWED_PROVIDER_HOSTS",
            " github.enterprise.corp , GHE.EXAMPLE.COM ",
        )
        hosts = _allowed_provider_hosts()
        assert "github.com" in hosts
        assert "github.enterprise.corp" in hosts
        assert "ghe.example.com" in hosts


class TestAttachRepoProviderHostGuard:
    def test_rejects_internal_provider_host(self) -> None:
        db = _make_db(query_results=[_fake_pot()])
        with pytest.raises(ValueError, match="provider_host not allowed"):
            attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="169.254.169.254",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url=None,
                default_branch=None,
                submitted_by_user_id="u",
            )

    def test_rejects_arbitrary_external_provider_host(self) -> None:
        db = _make_db(query_results=[_fake_pot()])
        with pytest.raises(ValueError, match="provider_host not allowed"):
            attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="evil.example.com",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url=None,
                default_branch=None,
                submitted_by_user_id="u",
            )

    def test_rejects_localhost_provider_host(self) -> None:
        db = _make_db(query_results=[_fake_pot()])
        with pytest.raises(ValueError, match="provider_host not allowed"):
            attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="localhost",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url=None,
                default_branch=None,
                submitted_by_user_id="u",
            )

    def test_rejects_internal_remote_url_even_when_provider_host_allowed(self) -> None:
        db = _make_db(query_results=[_fake_pot()])
        with pytest.raises(ValueError, match="remote_url not allowed"):
            attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.com",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url="http://169.254.169.254/latest/meta-data",
                default_branch=None,
                submitted_by_user_id="u",
            )

    def test_rejects_arbitrary_remote_url_host(self) -> None:
        db = _make_db(query_results=[_fake_pot()])
        with pytest.raises(ValueError, match="remote_url not allowed"):
            attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.com",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url="https://evil.example.com/acme/widgets.git",
                default_branch=None,
                submitted_by_user_id="u",
            )

    @pytest.mark.parametrize(
        "remote_url",
        [
            "http://github.com/acme/widgets.git",
            "ftp://github.com/acme/widgets.git",
            "file:///etc/passwd",
            "https://github.com./acme/widgets.git",
            "https://githu\u0432.com/acme/widgets.git",
        ],
    )
    def test_rejects_unsafe_or_confusable_remote_urls(self, remote_url: str) -> None:
        db = _make_db(query_results=[_fake_pot()])
        with pytest.raises(ValueError, match="remote_url not allowed"):
            attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.com",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url=remote_url,
                default_branch=None,
                submitted_by_user_id="u",
            )

    def test_allows_github_remote_url(self) -> None:
        db = _make_db(query_results=[_fake_pot(), None])
        source = SimpleNamespace(id="src-github")
        with patch(
            "app.modules.context_graph.attach_repo_to_pot.mirror_repository_into_sources",
            return_value=source,
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._dispatch_prewarm"
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._emit_bootstrap_event",
            return_value="evt-1",
        ):
            result = attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="GITHUB.COM",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url="https://github.com/acme/widgets.git",
                default_branch=None,
                submitted_by_user_id="u",
            )
        assert result.repository.provider_host == "github.com"
        assert result.repository.remote_url == "https://github.com/acme/widgets.git"

    def test_rejects_remote_url_that_does_not_match_provider_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CONTEXT_ENGINE_ALLOWED_PROVIDER_HOSTS",
            "github.com, github.enterprise.corp",
        )
        db = _make_db(query_results=[_fake_pot()])
        with pytest.raises(ValueError, match="remote_url not allowed"):
            attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.com",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url="https://github.enterprise.corp/acme/widgets.git",
                default_branch=None,
                submitted_by_user_id="u",
            )

    def test_allows_scp_like_github_ssh_remote_url(self) -> None:
        db = _make_db(query_results=[_fake_pot(), None])
        source = SimpleNamespace(id="src-ssh")
        with patch(
            "app.modules.context_graph.attach_repo_to_pot.mirror_repository_into_sources",
            return_value=source,
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._dispatch_prewarm"
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._emit_bootstrap_event",
            return_value="evt-1",
        ):
            result = attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.com",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url="git@github.com:acme/widgets.git",
                default_branch=None,
                submitted_by_user_id="u",
            )
        assert result.repository.remote_url == "git@github.com:acme/widgets.git"

    def test_allows_github_enterprise_host_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "CONTEXT_ENGINE_ALLOWED_PROVIDER_HOSTS", "github.mycompany.com"
        )
        db = _make_db(query_results=[_fake_pot(), None])
        source = SimpleNamespace(id="src-ghe")
        with patch(
            "app.modules.context_graph.attach_repo_to_pot.mirror_repository_into_sources",
            return_value=source,
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._dispatch_prewarm"
        ), patch(
            "app.modules.context_graph.attach_repo_to_pot._emit_bootstrap_event",
            return_value="evt-1",
        ):
            result = attach_repo_to_pot(
                db,
                pot_id="pot-1",
                provider="github",
                provider_host="github.mycompany.com",
                owner="acme",
                repo="widgets",
                external_repo_id=None,
                remote_url=None,
                default_branch=None,
                submitted_by_user_id="u",
            )
        assert result.repository.provider_host == "github.mycompany.com"
