"""Live E2E checks for CLI Atlassian auth (Jira and Confluence).

REVERT: delete ``tests/integration/test_cli_auth_atlassian_e2e.py`` and remove the
``cli_auth_e2e`` marker from ``pyproject.toml``.

Opt-in only — does not run in default pytest/CI. Uses an isolated ``XDG_CONFIG_HOME`` so
your real Potpie credentials are not modified.

Enable:
  export RUN_CLI_AUTH_E2E=1

From the repository root:

  uv run pytest tests/integration/test_cli_auth_atlassian_e2e.py -v

Required env (see below). Runtime settings may merge repo ``potpie/.env`` only
when the bootstrap environment resolves to ``dev``.

Environment variables
---------------------
RUN_CLI_AUTH_E2E=1
  Master switch (required).

Jira (real Atlassian API):
  CLI_AUTH_E2E_JIRA_EMAIL
  CLI_AUTH_E2E_JIRA_API_TOKEN
  CLI_AUTH_E2E_ATLASSIAN_SITE_SUBDOMAIN   e.g. myteam (for myteam.atlassian.net)

Confluence (same site/token as Jira is typical):
  CLI_AUTH_E2E_CONFLUENCE_EMAIL           (optional; defaults to Jira email)
  CLI_AUTH_E2E_CONFLUENCE_API_TOKEN       (optional; defaults to Jira token)
  CLI_AUTH_E2E_ATLASSIAN_SITE_SUBDOMAIN   (shared)
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.cli_auth_e2e,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _require_e2e_enabled() -> None:
    if not _truthy("RUN_CLI_AUTH_E2E"):
        pytest.skip("Set RUN_CLI_AUTH_E2E=1 to run CLI auth E2E tests")


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _atlassian_site_subdomain() -> str:
    return _env("CLI_AUTH_E2E_ATLASSIAN_SITE_SUBDOMAIN")


def _reset_cli_env_loader() -> None:
    import potpie.runtime.env_bootstrap as env_bootstrap

    env_bootstrap._loaded = False


def _run_potpie(
    *args: str,
    env: dict[str, str],
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    cmd = ["uv", "run", "potpie", *args]
    return subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture()
def isolated_cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Isolated credentials dir + dev-only repo .env merge via runtime settings."""
    _require_e2e_enabled()
    xdg = tmp_path / "xdg"
    xdg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    _reset_cli_env_loader()
    from potpie.runtime.settings import (
        ensure_runtime_environment_loaded,
    )

    ensure_runtime_environment_loaded()
    merged = os.environ.copy()
    merged["XDG_CONFIG_HOME"] = str(xdg)
    merged["PYTHONPATH"] = str(_REPO_ROOT)
    return merged


def _jira_creds() -> tuple[str, str, str] | None:
    email = _env("CLI_AUTH_E2E_JIRA_EMAIL")
    token = _env("CLI_AUTH_E2E_JIRA_API_TOKEN")
    subdomain = _atlassian_site_subdomain()
    if not (email and token and subdomain):
        return None
    return email, token, subdomain


def _confluence_creds() -> tuple[str, str, str] | None:
    jira = _jira_creds()
    email = _env("CLI_AUTH_E2E_CONFLUENCE_EMAIL") or (jira[0] if jira else "")
    token = _env("CLI_AUTH_E2E_CONFLUENCE_API_TOKEN") or (jira[1] if jira else "")
    subdomain = _atlassian_site_subdomain()
    if not (email and token and subdomain):
        return None
    return email, token, subdomain


def _parse_json_stdout(proc: subprocess.CompletedProcess[str]) -> Any:
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout)


def _login_product_via_cli(
    product: str,
    *,
    email: str,
    api_token: str,
    site_subdomain: str,
    env: dict[str, str],
) -> None:
    proc = _run_potpie(
        "--json",
        "auth",
        product,
        "login",
        "--force",
        "--email",
        email,
        "--api-token",
        api_token,
        "--site-subdomain",
        site_subdomain,
        env=env,
    )
    payload = _parse_json_stdout(proc)
    assert payload.get("ok") is True


def test_e2e_jira_api_token_login_and_list_projects(
    isolated_cli_env: dict[str, str],
) -> None:
    """Login with API token (real Atlassian) then list projects via CLI."""
    creds = _jira_creds()
    if creds is None:
        pytest.skip(
            "Set CLI_AUTH_E2E_JIRA_EMAIL, CLI_AUTH_E2E_JIRA_API_TOKEN, "
            "and CLI_AUTH_E2E_ATLASSIAN_SITE_SUBDOMAIN"
        )
    email, api_token, subdomain = creds

    _login_product_via_cli(
        "jira",
        email=email,
        api_token=api_token,
        site_subdomain=subdomain,
        env=isolated_cli_env,
    )

    proc = _run_potpie("--json", "auth", "jira", "ls", "-n", "5", env=isolated_cli_env)
    payload = _parse_json_stdout(proc)
    assert payload.get("ok") is True
    assert isinstance(payload.get("projects"), list)


def test_e2e_confluence_api_token_login_and_list_spaces(
    isolated_cli_env: dict[str, str],
) -> None:
    """Login with API token (real Atlassian) then list spaces via CLI."""
    creds = _confluence_creds()
    if creds is None:
        pytest.skip("Set Confluence/Jira E2E env vars (see module docstring)")
    email, api_token, subdomain = creds

    _login_product_via_cli(
        "confluence",
        email=email,
        api_token=api_token,
        site_subdomain=subdomain,
        env=isolated_cli_env,
    )

    proc = _run_potpie(
        "--json", "auth", "confluence", "ls", "-n", "5", env=isolated_cli_env
    )
    payload = _parse_json_stdout(proc)
    assert payload.get("ok") is True
    assert isinstance(payload.get("spaces"), list)
