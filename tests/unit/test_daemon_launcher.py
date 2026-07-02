from __future__ import annotations

import json
import pathlib

import pytest

from potpie.cli.telemetry.preferences import (
    TelemetryPreferences,
    save_preferences,
)
from potpie.daemon.process import launcher
from potpie.runtime import settings as runtime_settings


@pytest.fixture(autouse=True)
def _clear_runtime_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    monkeypatch.delenv("POTPIE_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(runtime_settings, "load_distribution_defaults", lambda: {})


def test_start_detached_enables_telemetry_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    env = _captured_child_env(monkeypatch, tmp_path)

    assert env["POTPIE_TELEMETRY_DISABLED"] == "0"


def test_start_detached_applies_persisted_cli_telemetry_preference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    save_preferences(TelemetryPreferences(enabled=False))

    env = _captured_child_env(monkeypatch, tmp_path)

    assert env["POTPIE_TELEMETRY_DISABLED"] == "1"


def test_start_detached_env_block_wins_over_persisted_enable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")
    save_preferences(TelemetryPreferences(enabled=True))

    env = _captured_child_env(monkeypatch, tmp_path)

    assert env["POTPIE_TELEMETRY_DISABLED"] == "1"


def _captured_child_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> dict[str, str]:
    captured: dict[str, dict[str, str]] = {}

    class _FakePopen:
        pid = 4242

        def __init__(
            self,
            _args: list[str],
            *,
            stdout: object,
            stderr: int,
            start_new_session: bool,
            close_fds: bool,
            env: dict[str, str],
        ) -> None:
            del stdout, stderr, start_new_session, close_fds
            captured["env"] = env
            home = pathlib.Path(env["CONTEXT_ENGINE_HOME"])
            (home / "discovery.json").write_text(
                json.dumps(
                    {
                        "transport": "http",
                        "base_url": "http://127.0.0.1:1",
                        "token": "secret",
                        "pid": self.pid,
                        "log_file": str(home / "daemon.log"),
                    }
                ),
                encoding="utf-8",
            )

        def poll(self) -> None:
            return None

    monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)

    result = launcher.start_detached(tmp_path / "home", ready_timeout_s=1)

    assert result["pid"] == 4242
    return captured["env"]
