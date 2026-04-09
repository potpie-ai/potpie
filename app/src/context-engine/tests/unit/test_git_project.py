"""CLI git → pot scope resolution helpers."""

import pytest

from adapters.inbound.cli import git_project as gp
from adapters.inbound.cli.git_project import (
    parse_owner_repo_from_remote,
    resolve_pot_id_for_repo,
    resolve_pot_id_from_git_cwd,
)


@pytest.fixture(autouse=True)
def _no_cli_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid loading the real repo ``.env`` during tests (would change env expectations)."""
    monkeypatch.setattr(gp, "load_cli_env", lambda: None)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("git@github.com:acme/app.git", "acme/app"),
        ("git@github.com:acme/app", "acme/app"),
        ("https://github.com/acme/app.git", "acme/app"),
        ("https://github.com/acme/app", "acme/app"),
        ("ssh://git@github.com/acme/app.git", "acme/app"),
    ],
)
def test_parse_owner_repo_from_remote(url: str, expected: str) -> None:
    assert parse_owner_repo_from_remote(url) == expected


def test_resolve_repo_to_pot_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CONTEXT_ENGINE_REPO_TO_POT",
        '{"acme/app":"proj-1"}',
    )
    monkeypatch.delenv("CONTEXT_ENGINE_POTS", raising=False)
    assert resolve_pot_id_for_repo("acme/app") == "proj-1"
    assert resolve_pot_id_for_repo("Acme/App") == "proj-1"


def test_resolve_from_pots_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_REPO_TO_POT", raising=False)
    monkeypatch.setenv(
        "CONTEXT_ENGINE_POTS",
        '{"proj-2":"acme/app"}',
    )
    assert resolve_pot_id_for_repo("acme/app") == "proj-2"


def test_resolve_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_REPO_TO_POT", raising=False)
    monkeypatch.delenv("CONTEXT_ENGINE_POTS", raising=False)
    monkeypatch.setattr(gp, "get_active_pot_id", lambda: "")
    assert resolve_pot_id_for_repo("acme/app") is None


def test_resolve_from_git_uses_active_pot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_REPO_TO_POT", raising=False)
    monkeypatch.delenv("CONTEXT_ENGINE_POTS", raising=False)
    uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    monkeypatch.setattr(gp, "get_active_pot_id", lambda: uid)
    monkeypatch.setattr(
        gp,
        "get_git_origin_remote_url",
        lambda _cwd=None: "https://github.com/acme/app.git",
    )
    pid, err = resolve_pot_id_from_git_cwd()
    assert pid == uid
    assert err == ""


def test_resolve_from_git_prefers_active_pot_over_env_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CONTEXT_ENGINE_REPO_TO_POT",
        '{"acme/app":"pot-from-env"}',
    )
    uid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    monkeypatch.setattr(gp, "get_active_pot_id", lambda: uid)
    monkeypatch.setattr(
        gp,
        "get_git_origin_remote_url",
        lambda _cwd=None: "https://github.com/acme/app.git",
    )
    pid, err = resolve_pot_id_from_git_cwd()
    assert pid == uid
    assert err == ""


def test_resolve_prefers_active_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CONTEXT_ENGINE_REPO_TO_POT",
        '{"acme/app":"pot-from-env"}',
    )
    uid = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    monkeypatch.setattr(gp, "get_active_pot_id", lambda: uid)
    monkeypatch.setattr(gp, "get_git_origin_remote_url", lambda _cwd=None: None)
    pid, err = resolve_pot_id_from_git_cwd()
    assert pid == uid
    assert err == ""


def test_resolve_active_non_uuid_without_alias_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_REPO_TO_POT", raising=False)
    monkeypatch.delenv("CONTEXT_ENGINE_POTS", raising=False)
    monkeypatch.setattr(gp, "get_active_pot_id", lambda: "not-a-uuid-or-alias")
    monkeypatch.setattr(gp, "get_git_origin_remote_url", lambda _cwd=None: None)
    pid, err = resolve_pot_id_from_git_cwd()
    assert pid is None
    assert "Unknown pot" in err


def test_resolve_from_git_errors_without_origin_or_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_REPO_TO_POT", raising=False)
    monkeypatch.delenv("CONTEXT_ENGINE_POTS", raising=False)
    monkeypatch.setattr(gp, "get_active_pot_id", lambda: "")
    monkeypatch.setattr(gp, "get_git_origin_remote_url", lambda _cwd=None: "https://github.com/acme/app.git")
    pid, err = resolve_pot_id_from_git_cwd()
    assert pid is None
    assert "No pot for repository" in err
    assert "pot use" in err
