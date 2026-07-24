"""Unit tests for GitLab CLI auth commands and credential store integration."""

from __future__ import annotations


import pytest
import typer

from adapters.outbound.cli_auth import credentials_store as cs
from adapters.outbound.cli_auth.gitlab_read_client import (
    GitLabReadError,
)

pytestmark = pytest.mark.unit


# ─── credential store round-trip ───────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_creds(tmp_path, monkeypatch):
    """Redirect credential files to a temp directory."""
    monkeypatch.setattr(cs, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(cs, "credentials_path", lambda: tmp_path / "credentials.json")
    monkeypatch.setattr(
        cs,
        "integration_secrets_path",
        lambda: tmp_path / "integration_secrets.json",
    )


def test_save_and_get_gitlab_credentials() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.corp.com",
            "instance_host": "gitlab.corp.com",
            "personal_access_token": "glpat-abc123",
            "stored_at": 1710000000.0,
        },
        account={"id": "42", "username": "jane"},
    )

    creds = cs.get_gitlab_credentials()
    assert creds["personal_access_token"] == "glpat-abc123"
    assert creds.get("instance_host") == "gitlab.corp.com"


def test_save_gitlab_credentials_requires_pat() -> None:
    with pytest.raises(cs.ProviderCredentialError, match="personal access token"):
        cs.save_gitlab_credentials(
            {"instance_url": "https://gitlab.com", "personal_access_token": ""},
        )


def test_get_gitlab_credentials_empty_when_not_stored() -> None:
    assert cs.get_gitlab_credentials() == {}


def test_clear_gitlab_credentials() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-abc",
        },
    )
    assert cs.get_gitlab_credentials() != {}
    cs.clear_gitlab_credentials()
    assert cs.get_gitlab_credentials() == {}


def test_multi_instance_save_and_list() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-saas",
        },
        account={"username": "alice"},
    )
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://git.corp.com",
            "instance_host": "git.corp.com",
            "personal_access_token": "glpat-corp",
        },
        account={"username": "alice-corp"},
    )
    instances = cs.list_gitlab_instances()
    assert len(instances) == 2
    hosts = {i["instance_host"] for i in instances}
    assert "gitlab.com" in hosts
    assert "git.corp.com" in hosts

    active_count = sum(1 for i in instances if i.get("active"))
    assert active_count == 1


def test_clear_specific_instance() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-saas",
        },
    )
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://git.corp.com",
            "instance_host": "git.corp.com",
            "personal_access_token": "glpat-corp",
        },
    )
    cs.clear_gitlab_credentials(instance_host="gitlab.com")
    instances = cs.list_gitlab_instances()
    assert len(instances) == 1
    assert instances[0]["instance_host"] == "git.corp.com"


def test_get_gitlab_credentials_for_specific_host() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-saas",
        },
    )
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://git.corp.com",
            "instance_host": "git.corp.com",
            "personal_access_token": "glpat-corp",
        },
    )
    creds = cs.get_gitlab_credentials(instance_host="gitlab.com")
    assert creds["personal_access_token"] == "glpat-saas"


def test_save_gitlab_workspace_prefs() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-abc",
        },
    )
    cs.save_gitlab_workspace_prefs(default_project="acme/api")
    creds = cs.get_gitlab_credentials()
    workspaces = creds.get("workspaces") or {}
    assert workspaces.get("default_project") == "acme/api"


def test_save_gitlab_credentials_preserves_workspace_prefs_on_relogin() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-old",
            "stored_at": 1710000000.0,
        },
    )
    cs.save_gitlab_workspace_prefs(default_project="acme/api")
    first = cs.get_gitlab_credentials()
    created_at = first.get("created_at")

    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-new",
            "stored_at": 1710000100.0,
        },
        account={"username": "jane"},
    )
    creds = cs.get_gitlab_credentials()
    assert creds["personal_access_token"] == "glpat-new"
    assert creds.get("workspaces", {}).get("default_project") == "acme/api"
    if created_at is not None:
        assert creds.get("created_at") == created_at


def test_save_gitlab_workspace_prefs_unknown_host() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-abc",
        },
    )
    with pytest.raises(cs.ProviderCredentialError, match="not connected"):
        cs.save_gitlab_workspace_prefs(
            instance_host="unknown.corp.com",
            default_project="acme/api",
        )


def test_save_gitlab_workspace_prefs_not_connected() -> None:
    with pytest.raises(cs.ProviderCredentialError, match="not connected"):
        cs.save_gitlab_workspace_prefs(default_project="acme/api")


# ─── integration_status for gitlab ────────────────────────────────────────


def test_get_integration_status_gitlab_not_authenticated() -> None:
    status = cs.get_integration_status("gitlab")
    assert status["provider"] == "gitlab"
    assert status["authenticated"] is False
    assert status["auth_type"] == "personal_access_token"


def test_get_integration_status_gitlab_authenticated() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.corp.com",
            "instance_host": "gitlab.corp.com",
            "personal_access_token": "glpat-abc",
            "stored_at": 1710000000.0,
        },
        account={"username": "jane", "email": "jane@corp.com"},
    )
    status = cs.get_integration_status("gitlab")
    assert status["provider"] == "gitlab"
    assert status["authenticated"] is True
    assert status["auth_type"] == "personal_access_token"
    assert status["login"] == "jane"
    assert status["instance_host"] == "gitlab.corp.com"


# ─── get_integration_tokens for gitlab ────────────────────────────────────


def test_get_integration_tokens_gitlab() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-tok",
        },
    )
    tokens = cs.get_integration_tokens("gitlab")
    assert tokens["auth_type"] == "personal_access_token"
    assert tokens["personal_access_token"] == "glpat-tok"


def test_get_integration_tokens_gitlab_empty() -> None:
    tokens = cs.get_integration_tokens("gitlab")
    assert tokens == {}


# ─── clear_integration_tokens for gitlab ──────────────────────────────────


def test_clear_integration_tokens_gitlab() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-tok",
        },
    )
    cs.clear_integration_tokens("gitlab")
    assert cs.get_gitlab_credentials() == {}


# ─── list_integration_providers includes gitlab ───────────────────────────


def test_list_integration_providers_includes_gitlab() -> None:
    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-tok",
        },
    )
    providers = cs.list_integration_providers()
    assert "gitlab" in providers


# ─── gitlab_read_client credential loading ────────────────────────────────


def test_load_gitlab_read_credentials_raises_when_not_connected() -> None:
    from adapters.outbound.cli_auth.gitlab_read_client import (
        load_gitlab_read_credentials,
    )

    with pytest.raises(GitLabReadError, match="not connected"):
        load_gitlab_read_credentials()


def test_load_gitlab_read_credentials_raises_when_no_token() -> None:
    from adapters.outbound.cli_auth.gitlab_read_client import (
        load_gitlab_read_credentials,
    )

    cs.save_gitlab_credentials(
        {
            "instance_url": "https://gitlab.com",
            "instance_host": "gitlab.com",
            "personal_access_token": "glpat-tok",
        },
    )
    # Simulate token missing by clearing the secret but keeping metadata
    cs._delete_file_secret("GitLab PAT", cs._gitlab_pat_secret("gitlab.com"))
    with pytest.raises(GitLabReadError, match="not connected"):
        load_gitlab_read_credentials()


# ─── _norm_gitlab_host ────────────────────────────────────────────────────


def test_norm_gitlab_host_bare() -> None:
    assert cs._norm_gitlab_host("gitlab.corp.com") == "gitlab.corp.com"


def test_norm_gitlab_host_url() -> None:
    assert cs._norm_gitlab_host("https://gitlab.corp.com") == "gitlab.corp.com"


def test_norm_gitlab_host_empty_defaults() -> None:
    assert cs._norm_gitlab_host("") == "gitlab.com"


# ─── GitLab login browser countdown UX ─────────────────────────────────────


def test_wait_for_enter_or_auto_open_returns_when_enter_is_pressed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import io

    from potpie.cli.auth import gitlab_auth as gl_auth

    input_stream = io.StringIO("\n")
    monkeypatch.setattr(
        gl_auth.select,
        "select",
        lambda _read, _write, _error, _timeout: ([input_stream], [], []),
    )
    monkeypatch.setattr(
        gl_auth.time,
        "sleep",
        lambda _seconds: pytest.fail("enter should skip the countdown sleep"),
    )

    gl_auth._wait_for_enter_or_auto_open(seconds=10, input_stream=input_stream)

    out = capsys.readouterr().out
    assert "Press Enter to open now, or browser opens in 10s" in out


def test_wait_for_enter_or_auto_open_times_out_on_same_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import io

    from potpie.cli.auth import gitlab_auth as gl_auth

    monkeypatch.setattr(
        gl_auth.select,
        "select",
        lambda _read, _write, _error, _timeout: ([], [], []),
    )

    gl_auth._wait_for_enter_or_auto_open(seconds=2, input_stream=io.StringIO(""))

    out = capsys.readouterr().out
    assert "\r\033[KPress Enter to open now, or browser opens in 2s" in out
    assert "\r\033[KPress Enter to open now, or browser opens in 1s" in out


def test_open_gitlab_pat_page_ctrl_c_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from potpie.cli.auth import gitlab_auth as gl_auth

    monkeypatch.setattr(
        gl_auth.webbrowser,
        "open",
        lambda _url: pytest.fail("cancelled login must not open the browser"),
    )

    def _cancel() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(gl_auth, "_wait_for_enter_or_auto_open", _cancel)

    with pytest.raises(typer.Exit) as exc:
        gl_auth._open_gitlab_pat_page("https://gitlab.com")
    assert exc.value.exit_code == gl_auth.EXIT_CANCELLED
    assert "\nGitLab login cancelled." in capsys.readouterr().out


def test_run_gitlab_pat_auth_opens_browser_after_countdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from potpie.cli.auth import gitlab_auth as gl_auth

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gl_auth.sys.stdin, "isatty", lambda: True)
    opened: list[str] = []
    monkeypatch.setattr(
        gl_auth.webbrowser,
        "open",
        lambda url, **_kwargs: opened.append(url) or True,
    )
    monkeypatch.setattr(gl_auth, "_wait_for_enter_or_auto_open", lambda: None)
    monkeypatch.setattr(
        gl_auth, "_prompt_instance_url", lambda default="": "gitlab.com"
    )
    monkeypatch.setattr(gl_auth, "_prompt_pat", lambda: "glpat-test")
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

    gl_auth.run_gitlab_pat_auth(force=True)

    assert opened == ["https://gitlab.com/-/user_settings/personal_access_tokens"]
