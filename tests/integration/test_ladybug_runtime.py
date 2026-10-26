"""Native Ladybug persistence through the actual local daemon lifecycle."""

from __future__ import annotations

# ruff: noqa: S101 - acceptance assertions.
import asyncio
import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_ladybug_setup_daemon_record_and_retry_survive_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if importlib.util.find_spec("ladybug") is None:
        if os.getenv("POTPIE_REQUIRE_LADYBUG") == "1":
            pytest.fail("required native Ladybug acceptance needs the ladybug package")
        pytest.skip("optional ladybug package is not installed")

    from potpie_context_engine import Success
    from potpie_context_engine.core.lifecycle import SetupPlan
    from potpie_context_engine.requests import (
        DataPlaneStatusRequest,
        RecordRequest,
        SearchRequest,
    )

    from potpie.daemon.discovery import load_daemon_connection
    from potpie.daemon.lifecycle import Daemon
    from potpie.runtime import ContextSelector, DaemonEngineClient, HttpDaemonTransport
    from potpie.runtime.composition import build_local_runtime

    home = tmp_path / "runtime"
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(home))
    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "ladybug")
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "daemon")
    monkeypatch.setenv("CONTEXT_ENGINE_EMBEDDER", "hashing")
    monkeypatch.setenv("CONTEXT_ENGINE_RESOURCE_INDEX", "sqlite_fts")
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "production")
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    monkeypatch.delenv("CONTEXT_ENGINE_LADYBUG_PATH", raising=False)
    monkeypatch.delenv("LADYBUG_PATH", raising=False)

    composition = build_local_runtime()
    try:
        setup = composition.root.setup.run(
            SetupPlan(
                backend="ladybug",
                host_mode="in_process",
                repo=None,
                defer_skills=True,
                embeddings="local",
            )
        )
        assert setup.ok, setup.to_dict()
        pot = composition.root.pots.active_pot()
        assert pot is not None
        pot_id = pot.pot_id
    finally:
        composition.close()

    request = RecordRequest(
        record_type="feature_note",
        summary="Ladybug Windows durable daemon restart acceptance",
        scope={"service": "ladybug-runtime"},
        idempotency_key="ladybug-runtime-restart",
    )

    async def exercise():
        connection = load_daemon_connection(home)
        transport = HttpDaemonTransport(
            endpoint=connection.discovery.endpoint,
            bearer_token=connection.bearer_token,
        )
        client = DaemonEngineClient(
            selector=ContextSelector(kind="explicit", value=pot_id),
            transport=transport,
            expected_instance_id=connection.discovery.instance_id,
        )
        try:
            handshake = await client.handshake()
            assert isinstance(handshake, Success), handshake
            recorded = await client.record(request)
            assert isinstance(recorded, Success), recorded
            assert recorded.value.accepted, recorded.value
            searched = await client.search(
                SearchRequest(query=request.summary, include=("raw_graph",))
            )
            assert isinstance(searched, Success), searched
            assert request.summary in [
                item.payload.get("fact") for item in searched.value.items
            ]
            status = await client.data_plane_status(DataPlaneStatusRequest())
            assert isinstance(status, Success), status
            assert status.value.backend_profile == "ladybug"
            assert status.value.backend_ready
            assert status.value.counts["claims"] == 1
            return connection.discovery.instance_id, recorded.value
        finally:
            await transport.close()

    daemon = Daemon(home=home, in_process=False, startup_timeout_s=30)
    try:
        daemon.ensure(SetupPlan(backend="ladybug", embeddings="local"))
        first_instance, first = asyncio.run(exercise())
        assert first.status == "recorded"
        assert first.mutations_applied == 1
        assert first.metadata["mutation_id"]
        daemon.restart()
        restarted_instance, replay = asyncio.run(exercise())
        assert restarted_instance != first_instance
        assert replay.status == "duplicate"
        assert replay.mutations_applied == 0
        assert replay.record_id == first.record_id
        assert replay.metadata["mutation_id"] == first.metadata["mutation_id"]
        assert replay.metadata["plan_id"] == first.metadata["plan_id"]
        assert (home / "context_graph" / "ladybug.db").is_file()
    finally:
        daemon.stop()
