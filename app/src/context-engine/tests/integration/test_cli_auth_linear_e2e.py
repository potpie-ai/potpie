"""Live E2E checks for Linear CLI OAuth (opt-in).


REVERT: delete this file and remove the ``cli_auth_e2e`` marker from ``pyproject.toml``.

Opt-in only — does not run in default pytest/CI. Uses an isolated ``XDG_CONFIG_HOME`` so
your real Potpie credentials are not modified.

Enable:
  export RUN_CLI_AUTH_E2E=1

From ``potpie/app/src/context-engine``:

  uv run pytest tests/integration/test_cli_auth_linear_e2e.py -v

Required env (see below). ``load_cli_env()`` still merges repo ``potpie/.env`` for
``LINEAR_CLIENT_ID`` and related vars.

Environment variables
---------------------
RUN_CLI_AUTH_E2E=1
  Master switch (required).



Linear (verify path without browser OAuth):
  LINEAR_CLIENT_ID                        (from .env)
  CLI_AUTH_E2E_LINEAR_ACCESS_TOKEN        (optional; seeds keychain in isolated config)
  CLI_AUTH_E2E_LINEAR_REFRESH_TOKEN       (optional)

Linear interactive OAuth (opens browser — local only):
  CLI_AUTH_E2E_RUN_LINEAR_LOGIN=1         (optional; skipped by default)
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

_CONTEXT_ENGINE_ROOT = Path(__file__).resolve().parents[2]


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _require_e2e_enabled() -> None:
    if not _truthy("RUN_CLI_AUTH_E2E"):
        pytest.skip("Set RUN_CLI_AUTH_E2E=1 to run CLI auth E2E tests")


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _parse_json_stdout(proc: subprocess.CompletedProcess[str]) -> Any:
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout)


def test_e2e_linear_status_and_verify_with_seeded_tokens(isolated_cli_env: dict[str, str]) -> None:
    """Seed Linear tokens from env, then run status --verify against Linear API."""
    if not _env("LINEAR_CLIENT_ID"):
        pytest.skip("LINEAR_CLIENT_ID not set (load potpie/.env or export it)")

    access = _env("CLI_AUTH_E2E_LINEAR_ACCESS_TOKEN")
    if not access:
        pytest.skip("Set CLI_AUTH_E2E_LINEAR_ACCESS_TOKEN (or run interactive login test)")

    from adapters.inbound.cli.credentials_store import save_integration_tokens

    tokens: dict[str, Any] = {
        "access_token": access,
        "expires_at": 9999999999.0,
    }
    refresh = _env("CLI_AUTH_E2E_LINEAR_REFRESH_TOKEN")
    if refresh:
        tokens["refresh_token"] = refresh

    save_integration_tokens("linear", tokens)

    proc = _run_potpie("--json", "auth", "status", "--verify", env=isolated_cli_env)
    payload = _parse_json_stdout(proc)
    integrations = payload.get("integrations") or []
    linear = next((row for row in integrations if row.get("provider") == "linear"), None)
    assert linear is not None
    assert linear.get("authenticated") is True
    assert linear.get("verified") is True


@pytest.mark.skipif(
    not _truthy("CLI_AUTH_E2E_RUN_LINEAR_LOGIN"),
    reason="Set CLI_AUTH_E2E_RUN_LINEAR_LOGIN=1 for interactive OAuth (opens browser)",
)
def test_e2e_linear_oauth_login_interactive(isolated_cli_env: dict[str, str]) -> None:
    """Optional: full Linear OAuth via CLI (requires browser; local developer only)."""
    if not _env("LINEAR_CLIENT_ID"):
        pytest.skip("LINEAR_CLIENT_ID not set")

    proc = _run_potpie("auth", "linear", "login", "--force", env=isolated_cli_env, timeout=300)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    verify = _run_potpie("--json", "auth", "status", "--verify", env=isolated_cli_env)
    payload = _parse_json_stdout(verify)
    linear = next(
        (row for row in payload.get("integrations") or [] if row.get("provider") == "linear"),
        None,
    )
    assert linear is not None
    assert linear.get("authenticated") is True
