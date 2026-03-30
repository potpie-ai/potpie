"""CLI git → project resolution helpers."""

import pytest

from adapters.inbound.cli import git_project as gp
from adapters.inbound.cli.git_project import (
    parse_owner_repo_from_remote,
    resolve_project_id_for_repo,
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


def test_resolve_repo_to_project_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CONTEXT_ENGINE_REPO_TO_PROJECT",
        '{"acme/app":"proj-1"}',
    )
    monkeypatch.delenv("CONTEXT_ENGINE_PROJECTS", raising=False)
    assert resolve_project_id_for_repo("acme/app") == "proj-1"
    assert resolve_project_id_for_repo("Acme/App") == "proj-1"


def test_resolve_from_projects_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_REPO_TO_PROJECT", raising=False)
    monkeypatch.setenv(
        "CONTEXT_ENGINE_PROJECTS",
        '{"proj-2":"acme/app"}',
    )
    assert resolve_project_id_for_repo("acme/app") == "proj-2"


def test_resolve_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_REPO_TO_PROJECT", raising=False)
    monkeypatch.delenv("CONTEXT_ENGINE_PROJECTS", raising=False)
    assert resolve_project_id_for_repo("acme/app") is None


def test_potpie_error_detail_json() -> None:
    class R:
        def json(self):
            return {"detail": "Invalid user_id"}

        text = ""

    assert gp._potpie_error_detail(R()) == "Invalid user_id"


def test_try_potpie_project_list_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POTPIE_API_URL", raising=False)
    monkeypatch.delenv("POTPIE_BASE_URL", raising=False)
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(gp, "get_stored_api_key", lambda: "")
    pid, err = gp._try_potpie_project_list("acme/app")
    assert pid is None and err == ""


def test_try_potpie_project_list_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POTPIE_API_URL", "http://localhost:8000")
    monkeypatch.setenv("POTPIE_API_KEY", "k")

    class Resp:
        status_code = 200
        text = "[]"

        def json(self):
            return [{"id": "proj-1", "repo_name": "acme/app", "status": "x"}]

    def fake_get(*_a, **_k):
        return Resp()

    monkeypatch.setattr("httpx.get", fake_get)
    pid, err = gp._try_potpie_project_list("acme/app")
    assert pid == "proj-1"
    assert err == ""


def test_try_potpie_project_list_disambiguate_by_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POTPIE_API_URL", "http://localhost:8000")
    monkeypatch.setenv("POTPIE_API_KEY", "k")

    class Resp:
        status_code = 200
        text = "[]"

        def json(self):
            return [
                {"id": "a", "repo_name": "acme/app", "branch_name": "main", "status": "x"},
                {"id": "b", "repo_name": "acme/app", "branch_name": "feat/context-engine", "status": "x"},
            ]

    monkeypatch.setattr("httpx.get", lambda *_a, **_k: Resp())
    monkeypatch.setattr(gp, "get_git_current_branch", lambda _cwd=None: "feat/context-engine")
    pid, err = gp._try_potpie_project_list("acme/app")
    assert pid == "b"
    assert err == ""


def test_try_potpie_project_list_multiple_no_branch_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old API without branch_name: keep ambiguous-repo error."""
    monkeypatch.setenv("POTPIE_API_URL", "http://localhost:8000")
    monkeypatch.setenv("POTPIE_API_KEY", "k")

    class Resp:
        status_code = 200
        text = "[]"

        def json(self):
            return [
                {"id": "a", "repo_name": "acme/app", "status": "x"},
                {"id": "b", "repo_name": "acme/app", "status": "x"},
            ]

    monkeypatch.setattr("httpx.get", lambda *_a, **_k: Resp())
    monkeypatch.setattr(gp, "get_git_current_branch", lambda _cwd=None: "main")
    pid, err = gp._try_potpie_project_list("acme/app")
    assert pid is None
    assert "Multiple Potpie projects matched" in err
    assert "branch_name" in err


def test_potpie_headers_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POTPIE_API_KEY", "k")
    monkeypatch.setattr(gp, "get_stored_api_key", lambda: "stored")
    h, err = gp._potpie_request_headers()
    assert err == ""
    assert h == {"X-API-Key": "k"}


def test_potpie_headers_stored_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POTPIE_API_KEY", raising=False)
    monkeypatch.setattr(gp, "get_stored_api_key", lambda: "stored")
    h, err = gp._potpie_request_headers()
    assert err == ""
    assert h == {"X-API-Key": "stored"}


def test_try_potpie_project_list_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POTPIE_API_URL", "http://localhost:8000")
    monkeypatch.setenv("POTPIE_API_KEY", "k")

    class Resp:
        status_code = 200
        text = "[]"

        def json(self):
            return [{"id": "other", "repo_name": "other/repo", "status": "x"}]

    monkeypatch.setattr("httpx.get", lambda *_a, **_k: Resp())
    pid, err = gp._try_potpie_project_list("acme/app")
    assert pid is None
    assert "No Potpie project" in err
