"""CLI command, auth flow, read client, and read flow coverage for GitLab."""

from __future__ import annotations

import io
import types
from typing import Any

import httpx
import pytest
import typer
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli as cli_main
from adapters.inbound.cli.auth import gitlab_auth as gl_auth
from adapters.inbound.cli.auth import gitlab_commands as gl_cmds
from adapters.inbound.cli.auth import gitlab_read as gl_read
from adapters.outbound.cli_auth import credentials_store as cs
from adapters.outbound.cli_auth.gitlab_client import GitLabAuthErrorKind
from adapters.outbound.cli_auth.gitlab_read_client import (
    GitLabReadError,
    fetch_gitlab_issues,
    fetch_gitlab_merge_requests,
    fetch_gitlab_projects,
)
from adapters.outbound.cli_auth.http import AuthHttpError
from tests._auth_fakes import FakeAuthHttpClient

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_creds(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.setattr(
        cs, "integration_secrets_path", lambda: tmp_path / "integration_secrets.json",
    )


def _save_gitlab(*, host: str = "gitlab.com", token: str = "glpat-test") -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": f"https://{host}",
            "instance_host": host,
            "personal_access_token": token,
        },
        account={"username": "jane", "id": "42"},
    )


def _patch_read_http(monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]) -> None:
    from adapters.outbound.cli_auth import gitlab_read_client as gl_read_client

    fake = FakeAuthHttpClient(responses)
    monkeypatch.setattr(
        gl_read_client,
        "AuthHttpClient",
        lambda **_kwargs: fake,
    )


# ─── gitlab_auth helpers ────────────────────────────────────────────────────


def test_auth_failure_message_all_kinds() -> None:
    base = gl_auth._auth_failure_message(None, "https://gitlab.com")
    assert "Could not authenticate" in base
    invalid = gl_auth._auth_failure_message(
        GitLabAuthErrorKind.INVALID_CREDENTIALS, "https://gitlab.com",
    )
    assert "Invalid personal access token" in invalid
    scopes = gl_auth._auth_failure_message(
        GitLabAuthErrorKind.INSUFFICIENT_SCOPES, "https://gitlab.com",
    )
    assert "missing required scopes" in scopes
    unreachable = gl_auth._auth_failure_message(
        GitLabAuthErrorKind.INSTANCE_UNREACHABLE, "https://git.corp.com",
    )
    assert "Cannot reach" in unreachable


def test_detect_gitlab_from_git_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "adapters.inbound.cli.repo_location.current_git_remote",
        lambda _cwd: "gitlab.corp.com/acme/api",
    )
    assert gl_auth._detect_gitlab_from_git_remote() == "https://gitlab.corp.com"


def test_detect_gitlab_from_git_remote_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "adapters.inbound.cli.repo_location.current_git_remote",
        lambda _cwd: "github.com/acme/api",
    )
    assert gl_auth._detect_gitlab_from_git_remote() is None


def test_guard_typer_prompt_maps_abort_to_keyboard_interrupt() -> None:
    import click

    def _abort() -> None:
        raise click.Abort()

    with pytest.raises(KeyboardInterrupt):
        gl_auth._guard_typer_prompt(_abort)


def test_open_gitlab_pat_page_browser_open_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gl_auth, "_wait_for_enter_or_auto_open", lambda: None)
    monkeypatch.setattr(gl_auth.webbrowser, "open", lambda *_a, **_k: False)

    gl_auth._open_gitlab_pat_page("https://gitlab.com")

    out = capsys.readouterr().out
    assert "Could not open a browser" in out
    assert "personal_access_tokens" in out


def test_run_gitlab_pat_auth_already_connected(tmp_path, capsys) -> None:
    _save_gitlab()
    gl_auth.run_gitlab_pat_auth()
    out = capsys.readouterr().out
    assert "already connected" in out


def test_run_gitlab_pat_auth_non_tty_requires_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gl_auth.sys.stdin, "isatty", lambda: False)
    with pytest.raises(typer.Exit) as exc:
        gl_auth.run_gitlab_pat_auth(force=True)
    assert exc.value.exit_code == 1


def test_run_gitlab_pat_auth_supplied_credentials_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        gl_auth,
        "verify_instance_access",
        lambda *_a, **_k: (True, None, {"id": 1, "username": "jane"}),
    )
    monkeypatch.setattr(
        gl_auth,
        "verify_read_api_scope",
        lambda *_a, **_k: (True, None),
    )
    gl_auth.run_gitlab_pat_auth(
        force=True,
        instance="https://gitlab.corp.com",
        token="glpat-abc",
    )
    assert "Connected GitLab" in capsys.readouterr().out


def test_run_gitlab_pat_auth_verify_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gl_auth,
        "verify_instance_access",
        lambda *_a, **_k: (False, GitLabAuthErrorKind.INVALID_CREDENTIALS, {}),
    )
    with pytest.raises(typer.Exit) as exc:
        gl_auth.run_gitlab_pat_auth(
            force=True,
            instance="https://gitlab.com",
            token="bad",
        )
    assert exc.value.exit_code == gl_auth.EXIT_AUTH


def test_run_gitlab_pat_auth_scope_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gl_auth,
        "verify_instance_access",
        lambda *_a, **_k: (True, None, {"id": 1, "username": "jane"}),
    )
    monkeypatch.setattr(
        gl_auth,
        "verify_read_api_scope",
        lambda *_a, **_k: (False, GitLabAuthErrorKind.INSUFFICIENT_SCOPES),
    )
    with pytest.raises(typer.Exit) as exc:
        gl_auth.run_gitlab_pat_auth(
            force=True,
            instance="https://gitlab.com",
            token="glpat-abc",
        )
    assert exc.value.exit_code == gl_auth.EXIT_AUTH


def test_run_gitlab_pat_auth_storage_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gl_auth,
        "verify_instance_access",
        lambda *_a, **_k: (True, None, {"id": 1, "username": "jane"}),
    )
    monkeypatch.setattr(
        gl_auth,
        "verify_read_api_scope",
        lambda *_a, **_k: (True, None),
    )
    monkeypatch.setattr(
        gl_auth,
        "save_gitlab_credentials",
        lambda *_a, **_k: (_ for _ in ()).throw(
            cs.ProviderCredentialError("disk full")
        ),
    )
    with pytest.raises(typer.Exit) as exc:
        gl_auth.run_gitlab_pat_auth(
            force=True,
            instance="https://gitlab.com",
            token="glpat-abc",
        )
    assert exc.value.exit_code == gl_auth.EXIT_AUTH


def test_wait_for_enter_or_auto_open_select_error_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sleeps: list[float] = []

    def _bad_select(*_args, **_kwargs):
        raise OSError("select failed")

    monkeypatch.setattr(gl_auth.select, "select", _bad_select)
    monkeypatch.setattr(gl_auth.time, "sleep", lambda s: sleeps.append(s))

    gl_auth._wait_for_enter_or_auto_open(seconds=1, input_stream=io.StringIO(""))

    assert sleeps == [1]
    assert "browser opens in 1s" in capsys.readouterr().out


# ─── gitlab_read_client ─────────────────────────────────────────────────────


def test_fetch_gitlab_projects_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_gitlab()
    _patch_read_http(
        monkeypatch,
        [
            httpx.Response(
                200,
                json=[
                    {
                        "id": 7,
                        "path_with_namespace": "acme/api",
                        "name": "API",
                        "visibility": "private",
                        "web_url": "https://gitlab.com/acme/api",
                        "default_branch": "main",
                    },
                    "skip-me",
                ],
            ),
        ],
    )
    rows = fetch_gitlab_projects()
    assert len(rows) == 1
    assert rows[0]["path_with_namespace"] == "acme/api"


def test_fetch_gitlab_projects_non_list_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_gitlab()
    _patch_read_http(monkeypatch, [httpx.Response(200, json={"error": "nope"})])
    assert fetch_gitlab_projects() == []


def test_fetch_gitlab_merge_requests_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_gitlab()
    _patch_read_http(
        monkeypatch,
        [
            httpx.Response(
                200,
                json=[
                    {
                        "iid": 3,
                        "title": "Fix bug",
                        "state": "opened",
                        "author": {"username": "jane"},
                        "target_branch": "main",
                        "source_branch": "fix",
                        "web_url": "https://gitlab.com/acme/api/-/merge_requests/3",
                        "updated_at": "2026-01-01",
                    }
                ],
            ),
        ],
    )
    rows = fetch_gitlab_merge_requests(7)
    assert rows[0]["iid"] == 3
    assert rows[0]["author"] == "jane"


def test_fetch_gitlab_issues_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_gitlab()
    _patch_read_http(
        monkeypatch,
        [
            httpx.Response(
                200,
                json=[
                    {
                        "iid": 9,
                        "title": "Bug",
                        "state": "opened",
                        "assignee": {"username": "bob"},
                        "labels": ["bug"],
                        "web_url": "https://gitlab.com/acme/api/-/issues/9",
                        "updated_at": "2026-01-02",
                    }
                ],
            ),
        ],
    )
    rows = fetch_gitlab_issues(7)
    assert rows[0]["iid"] == 9
    assert rows[0]["assignee"] == "bob"


def test_get_json_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from adapters.outbound.cli_auth import gitlab_read_client as gl_read_client

    _save_gitlab()

    def _raise(*_a, **_k):
        raise AuthHttpError("connection refused")

    fake = types.SimpleNamespace(get=_raise, close=lambda: None)
    monkeypatch.setattr(gl_read_client, "AuthHttpClient", lambda **_k: fake)
    with pytest.raises(GitLabReadError, match="request failed"):
        fetch_gitlab_projects()

    _patch_read_http(monkeypatch, [httpx.Response(401)])
    with pytest.raises(GitLabReadError, match="expired or revoked"):
        fetch_gitlab_projects()

    _patch_read_http(monkeypatch, [httpx.Response(403)])
    with pytest.raises(GitLabReadError, match="required scopes"):
        fetch_gitlab_projects()

    _patch_read_http(monkeypatch, [httpx.Response(500)])
    with pytest.raises(GitLabReadError, match="HTTP 500"):
        fetch_gitlab_projects()

    _patch_read_http(monkeypatch, [httpx.Response(200, content=b"not-json")])
    with pytest.raises(GitLabReadError, match="non-JSON"):
        fetch_gitlab_projects()


# ─── gitlab_read flow ───────────────────────────────────────────────────────


def test_run_gitlab_select_flow_with_project_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_gitlab()
    monkeypatch.setattr(gl_read.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        gl_read,
        "fetch_gitlab_projects",
        lambda **_k: [{"id": 7, "path_with_namespace": "acme/api", "name": "API"}],
    )
    monkeypatch.setattr(
        gl_read,
        "fetch_gitlab_merge_requests",
        lambda *_a, **_k: [{"iid": 1, "title": "MR"}],
    )
    monkeypatch.setattr(
        gl_read,
        "fetch_gitlab_issues",
        lambda *_a, **_k: [{"iid": 2, "title": "Issue"}],
    )
    result = gl_read.run_gitlab_select_flow(project_path="acme/api", limit=5)
    assert result["workspace_key"] == "acme/api"
    assert len(result["merge_requests"]) == 1
    assert len(result["issues"]) == 1


def test_run_gitlab_select_flow_with_saved_default_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_gitlab()
    cs.save_gitlab_workspace_prefs(default_project="acme/api")
    monkeypatch.setattr(gl_read.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        gl_read,
        "fetch_gitlab_projects",
        lambda **_k: [{"id": 7, "path_with_namespace": "acme/api", "name": "API"}],
    )
    monkeypatch.setattr(
        gl_read,
        "fetch_gitlab_merge_requests",
        lambda *_a, **_k: [{"iid": 1, "title": "MR"}],
    )
    monkeypatch.setattr(
        gl_read,
        "fetch_gitlab_issues",
        lambda *_a, **_k: [{"iid": 2, "title": "Issue"}],
    )
    result = gl_read.run_gitlab_select_flow(limit=5)
    assert result["workspace_key"] == "acme/api"


def test_run_gitlab_select_flow_requires_terminal_without_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_gitlab()
    monkeypatch.setattr(gl_read.sys.stdin, "isatty", lambda: False)
    with pytest.raises(GitLabReadError, match="requires a terminal"):
        gl_read.run_gitlab_select_flow()


def test_prompt_project_invalid_then_valid(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts = iter(["nope", "1"])

    monkeypatch.setattr(gl_read.typer, "prompt", lambda *_a, **_k: next(prompts))
    picked = gl_read._prompt_project(
        [{"path_with_namespace": "acme/api", "name": "API"}],
    )
    assert picked["path_with_namespace"] == "acme/api"
    assert "Enter a number" in capsys.readouterr().out


# ─── gitlab_commands via CliRunner ──────────────────────────────────────────


def test_gitlab_login_cli_invokes_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    def _fake_auth(**kwargs: Any) -> None:
        called.update(kwargs)

    monkeypatch.setattr(gl_cmds, "run_gitlab_pat_auth", _fake_auth)
    result = runner.invoke(cli_main.app, ["gitlab", "login", "--force"])
    assert result.exit_code == 0, result.stdout
    assert called.get("force") is True


def test_gitlab_logout_cli_success() -> None:
    _save_gitlab()
    result = runner.invoke(cli_main.app, ["gitlab", "logout"])
    assert result.exit_code == 0, result.stdout
    assert cs.get_gitlab_credentials() == {}


def test_gitlab_logout_cli_json_when_not_authenticated() -> None:
    result = runner.invoke(cli_main.app, ["--json", "gitlab", "logout"])
    assert result.exit_code == 0, result.stdout
    assert '"cleared_stale": true' in result.stdout


def test_gitlab_ls_cli_lists_instances() -> None:
    _save_gitlab()
    result = runner.invoke(cli_main.app, ["gitlab", "ls"])
    assert result.exit_code == 0, result.stdout
    assert "gitlab.com" in result.stdout


def test_gitlab_ls_cli_json() -> None:
    _save_gitlab()
    result = runner.invoke(cli_main.app, ["--json", "gitlab", "ls"])
    assert result.exit_code == 0, result.stdout
    assert '"instances"' in result.stdout


def test_gitlab_ls_cli_empty() -> None:
    result = runner.invoke(cli_main.app, ["gitlab", "ls"])
    assert result.exit_code == 0, result.stdout
    assert "none" in result.stdout.lower()


def test_gitlab_repos_cli_lists_projects(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_gitlab()
    monkeypatch.setattr(
        gl_cmds,
        "fetch_gitlab_projects",
        lambda **_k: [
            {"path_with_namespace": "acme/api", "visibility": "private"},
        ],
    )
    result = runner.invoke(cli_main.app, ["gitlab", "repos"])
    assert result.exit_code == 0, result.stdout
    assert "acme/api" in result.stdout


def test_gitlab_repos_cli_not_connected() -> None:
    result = runner.invoke(cli_main.app, ["gitlab", "repos"])
    assert result.exit_code == gl_cmds.EXIT_UNAVAILABLE


def test_gitlab_select_cli_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gl_cmds,
        "run_gitlab_select_flow",
        lambda **_k: {
            "product": "gitlab",
            "workspace_key": "acme/api",
            "workspace_name": "API",
            "merge_requests": [{"iid": 1, "title": "MR", "author": "jane"}],
            "issues": [
                {
                    "iid": 2,
                    "title": "Bug",
                    "assignee": "bob",
                    "labels": ["bug"],
                    "web_url": "https://gitlab.com/acme/api/-/issues/2",
                }
            ],
        },
    )
    result = runner.invoke(
        cli_main.app,
        ["--json", "gitlab", "select", "--project", "acme/api"],
    )
    assert result.exit_code == 0, result.stdout
    assert '"merge_requests"' in result.stdout


def test_prompt_instance_url_defaults_to_gitlab_com(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gl_auth.typer, "prompt", lambda *_a, **_k: "gitlab.com")
    assert gl_auth._prompt_instance_url() == "https://gitlab.com"


def test_prompt_pat_rejects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gl_auth.typer, "prompt", lambda *_a, **_k: "   ")
    with pytest.raises(typer.Exit) as exc:
        gl_auth._prompt_pat()
    assert exc.value.exit_code == 1


def test_gitlab_logout_cli_with_instance_json() -> None:
    _save_gitlab()
    result = runner.invoke(
        cli_main.app,
        ["--json", "gitlab", "logout", "--instance", "gitlab.com"],
    )
    assert result.exit_code == 0, result.stdout
    assert '"instance": "gitlab.com"' in result.stdout


def test_gitlab_logout_cli_stale_plain_message() -> None:
    result = runner.invoke(cli_main.app, ["gitlab", "logout"])
    assert result.exit_code == 0, result.stdout
    assert "No active session" in result.stdout


def test_gitlab_logout_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_kwargs: Any) -> None:
        raise cs.ProviderCredentialError("locked")

    monkeypatch.setattr(gl_cmds, "clear_gitlab_credentials", _boom)
    result = runner.invoke(cli_main.app, ["gitlab", "logout"])
    assert result.exit_code == gl_cmds.EXIT_AUTH


def test_gitlab_ls_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> list[dict[str, Any]]:
        raise cs.CredentialStoreError("read failed")

    monkeypatch.setattr(gl_cmds, "list_gitlab_instances", _boom)
    result = runner.invoke(cli_main.app, ["gitlab", "ls"])
    assert result.exit_code == gl_cmds.EXIT_UNAVAILABLE


def test_gitlab_ls_cli_shows_active_instance() -> None:
    _save_gitlab()
    result = runner.invoke(cli_main.app, ["gitlab", "ls"])
    assert result.exit_code == 0, result.stdout
    assert "(active)" in result.stdout
    assert "(jane)" in result.stdout


def test_gitlab_repos_cli_json_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_gitlab()
    monkeypatch.setattr(gl_cmds, "fetch_gitlab_projects", lambda **_k: [])
    result = runner.invoke(cli_main.app, ["--json", "gitlab", "repos"])
    assert result.exit_code == 0, result.stdout
    assert '"count": 0' in result.stdout


def test_gitlab_repos_cli_plain_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_gitlab()
    monkeypatch.setattr(gl_cmds, "fetch_gitlab_projects", lambda **_k: [])
    result = runner.invoke(cli_main.app, ["gitlab", "repos"])
    assert result.exit_code == 0, result.stdout
    assert "(none)" in result.stdout


def test_gitlab_repos_cli_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _save_gitlab()

    def _fail(**_kwargs: Any) -> list[dict[str, Any]]:
        raise GitLabReadError("token expired")

    monkeypatch.setattr(gl_cmds, "fetch_gitlab_projects", _fail)
    result = runner.invoke(cli_main.app, ["gitlab", "repos"])
    assert result.exit_code == gl_cmds.EXIT_UNAVAILABLE


def test_gitlab_select_cli_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(**_kwargs: Any) -> dict[str, Any]:
        raise GitLabReadError("project missing")

    monkeypatch.setattr(gl_cmds, "run_gitlab_select_flow", _fail)
    result = runner.invoke(cli_main.app, ["gitlab", "select", "--project", "acme/api"])
    assert result.exit_code == gl_cmds.EXIT_UNAVAILABLE


def test_gitlab_select_cli_plain_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gl_cmds,
        "run_gitlab_select_flow",
        lambda **_k: {
            "workspace_key": "acme/api",
            "workspace_name": "API",
            "merge_requests": [
                {
                    "iid": 1,
                    "title": "MR",
                    "author": "jane",
                    "source_branch": "feat",
                    "target_branch": "main",
                    "web_url": "https://gitlab.com/acme/api/-/merge_requests/1",
                }
            ],
            "issues": [],
        },
    )
    result = runner.invoke(cli_main.app, ["gitlab", "select", "--project", "acme/api"])
    assert result.exit_code == 0, result.stdout
    assert "Open merge requests" in result.stdout
    assert "No open issues" in result.stdout


def test_gitlab_select_cli_plain_with_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gl_cmds,
        "run_gitlab_select_flow",
        lambda **_k: {
            "workspace_key": "acme/api",
            "workspace_name": "API",
            "merge_requests": [],
            "issues": [
                {
                    "iid": 5,
                    "title": "Crash",
                    "assignee": "alice",
                    "labels": ["bug", "p1"],
                    "web_url": "https://gitlab.com/acme/api/-/issues/5",
                }
            ],
        },
    )
    result = runner.invoke(cli_main.app, ["gitlab", "select", "--project", "acme/api"])
    assert result.exit_code == 0, result.stdout
    assert "Open issues" in result.stdout
    assert "Labels: bug, p1" in result.stdout
    assert "Assignee: alice" in result.stdout
