from __future__ import annotations

import json
from pathlib import Path

import pytest

from potpie_context_engine.contracts import EngineStatusRequest

from potpie.daemon.client import DaemonEngineClient
from potpie.runtime import LocalEngineClient, ProductSettings, create_runtime
from potpie.runtime.errors import RuntimeDaemonUnavailable


def test_product_settings_runtime_precedence(tmp_path: Path) -> None:
    data_dir = tmp_path / "state"
    data_dir.mkdir()
    data_dir.joinpath("config.json").write_text(
        json.dumps({"runtime_mode": "in-process", "backend": "embedded"}),
        encoding="utf-8",
    )
    environ = {"POTPIE_HOME": str(data_dir), "POTPIE_RUNTIME_MODE": "daemon"}

    from_env = ProductSettings.load(environ=environ)
    from_override = ProductSettings.load(runtime_override="in-process", environ=environ)
    from_persisted = ProductSettings.load(environ={"POTPIE_HOME": str(data_dir)})
    by_default = ProductSettings.load(environ={"HOME": str(tmp_path / "home")})

    assert from_env.runtime_mode == "daemon"
    assert from_override.runtime_mode == "in-process"
    assert from_persisted.runtime_mode == "in-process"
    assert by_default.runtime_mode == "daemon"


def test_invalid_runtime_mode_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid Potpie runtime mode"):
        ProductSettings.load(
            environ={"HOME": str(tmp_path), "POTPIE_RUNTIME_MODE": "automatic"}
        )


@pytest.mark.asyncio
async def test_in_process_runtime_uses_local_engine(tmp_path: Path) -> None:
    runtime = create_runtime(
        settings=ProductSettings(
            data_dir=tmp_path,
            runtime_mode="in-process",
            backend="in_memory",
        )
    )

    assert isinstance(runtime.engine, LocalEngineClient)
    await runtime.aclose()


@pytest.mark.asyncio
async def test_daemon_default_never_falls_back_when_unavailable(tmp_path: Path) -> None:
    runtime = create_runtime(
        settings=ProductSettings(
            data_dir=tmp_path,
            runtime_mode="daemon",
            backend="embedded",
        )
    )

    assert isinstance(runtime.engine, DaemonEngineClient)
    with pytest.raises(RuntimeDaemonUnavailable) as caught:
        await runtime.engine.context.status(EngineStatusRequest())

    assert caught.value.code == "RUNTIME_DAEMON_UNAVAILABLE"
    assert caught.value.recommended_command == "potpie daemon start"
