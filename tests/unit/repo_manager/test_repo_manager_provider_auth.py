from types import SimpleNamespace

import pytest

from app.modules.repo_manager.repo_manager import RepoManager


pytestmark = pytest.mark.unit


def test_derive_repo_url_uses_gitbucket_base(monkeypatch, tmp_path):
    monkeypatch.setenv("CODE_PROVIDER", "gitbucket")
    monkeypatch.setenv("CODE_PROVIDER_BASE_URL", "http://localhost:8080/api/v3")
    manager = RepoManager(repos_base_path=str(tmp_path))

    clone_url = manager._derive_repo_url("root/potpie")

    assert clone_url == "http://localhost:8080/root/potpie.git"


def test_build_authenticated_url_uses_gitbucket_basic_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("CODE_PROVIDER", "gitbucket")
    monkeypatch.setenv("GITBUCKET_USERNAME", "root")
    monkeypatch.setenv("GITBUCKET_PASSWORD", "secret")
    manager = RepoManager(repos_base_path=str(tmp_path))

    clone_url = manager._build_authenticated_url(
        "http://localhost:8080/root/potpie.git", None
    )

    assert clone_url == "http://root:secret@localhost:8080/root/potpie.git"


def test_build_authenticated_url_prefers_gitbucket_env_basic_over_token(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CODE_PROVIDER", "gitbucket")
    monkeypatch.setenv("GITBUCKET_USERNAME", "root")
    monkeypatch.setenv("GITBUCKET_PASSWORD", "secret")
    manager = RepoManager(repos_base_path=str(tmp_path))

    clone_url = manager._build_authenticated_url(
        "http://localhost:8080/root/potpie.git", "some_token"
    )

    assert clone_url == "http://root:secret@localhost:8080/root/potpie.git"


def test_ensure_bare_repo_raises_gitbucket_specific_auth_error(monkeypatch, tmp_path):
    monkeypatch.setenv("CODE_PROVIDER", "gitbucket")
    monkeypatch.setenv("CODE_PROVIDER_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("CODE_PROVIDER_TOKEN", "dummy_token")
    manager = RepoManager(repos_base_path=str(tmp_path))

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stderr="fatal: Authentication failed for 'http://localhost:8080/root/potpie.git/'",
            stdout="",
        )

    monkeypatch.setattr("app.modules.repo_manager.repo_manager.subprocess.run", fake_run)

    with pytest.raises(RuntimeError) as exc:
        manager.ensure_bare_repo("root/potpie", ref="main", user_id="defaultuser")

    assert "GitBucket authentication failed" in str(exc.value)
