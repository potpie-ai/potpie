"""Live E2E checks for Linear CLI OAuth (opt-in).


REVERT: delete this file and remove the ``cli_auth_e2e`` marker from ``pyproject.toml``.

Opt-in only — does not run in default pytest/CI. Uses an isolated ``XDG_CONFIG_HOME`` and
a disposable file keyring so your real Potpie credentials and system keychain are not modified.

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
from collections.abc import Iterator
from typing import Any

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.cli_auth_e2e,
]

_CONTEXT_ENGINE_ROOT = Path(__file__).resolve().parents[2]
_E2E_KEYRING_BACKEND = "adapters.inbound.cli.e2e_keyring.E2EKeyring"


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


def _activate_e2e_keyring() -> None:
    import keyring.core as keyring_core

    keyring_core.set_keyring(keyring_core.load_keyring(os.environ["PYTHON_KEYRING_BACKEND"]))


def _run_potpie(
    *args: str,
    env: dict[str, str],
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    cmd = ["uv", "run", "potpie", *args]
    return subprocess.run(
        cmd,
        cwd=str(_CONTEXT_ENGINE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture()
def isolated_cli_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, str]]:
    """Isolated credentials dir + repo .env merged via load_cli_env."""
    _require_e2e_enabled()
    import adapters.inbound.cli.env_bootstrap as env_bootstrap

    saved_loaded = env_bootstrap._loaded
    saved_environ = os.environ.copy()

    xdg = tmp_path / "xdg"
    xdg.mkdir(parents=True, exist_ok=True)
    keyring_file = xdg / "e2e_keyring.json"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("POTPIE_E2E_KEYRING_FILE", str(keyring_file))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", _E2E_KEYRING_BACKEND)
    env_bootstrap._loaded = False
    from adapters.inbound.cli.env_bootstrap import load_cli_env

    load_cli_env()
    _activate_e2e_keyring()
    merged = os.environ.copy()
    merged["XDG_CONFIG_HOME"] = str(xdg)
    merged["POTPIE_E2E_KEYRING_FILE"] = str(keyring_file)
    merged["PYTHON_KEYRING_BACKEND"] = _E2E_KEYRING_BACKEND
    merged["PYTHONPATH"] = str(_CONTEXT_ENGINE_ROOT)
    yield merged
    os.environ.clear()
    os.environ.update(saved_environ)
    env_bootstrap._loaded = saved_loaded
    import keyring.core as keyring_core

    keyring_core._keyring_backend = None


def test_e2e_linear_status_and_verify_with_seeded_tokens(isolated_cli_env: dict[str, str]) -> None:
    """Seed Linear tokens from env, then run status --verify against Linear API."""
    _require_e2e_enabled()
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
    _require_e2e_enabled()
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
