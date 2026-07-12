from __future__ import annotations

from pathlib import Path
from typing import get_type_hints
from unittest.mock import patch

import pytest

from potpie_context_engine import (
    ContextEngine,
    EngineClient,
    EngineConfig,
    EngineDependencies,
    create_engine,
)
from potpie_context_engine.client import ContextClient
from potpie_context_engine.contracts import (
    EmptyRequest,
    EngineStatusRequest,
    PotCreateRequest,
    ProvisionApplyRequest,
    ProvisionInspectRequest,
    ResolveRequest,
)


def test_public_exports_are_declared() -> None:
    import potpie_context_engine

    assert potpie_context_engine.__all__ == [
        "ContextEngine",
        "EngineClient",
        "EngineConfig",
        "EngineDependencies",
        "create_engine",
    ]
    assert get_type_hints(ContextClient.resolve)["request"] is ResolveRequest
    assert EngineClient is not None


def test_persistent_config_requires_an_explicit_data_directory() -> None:
    with pytest.raises(ValueError, match="requires data_dir"):
        EngineConfig(storage_mode="persistent", data_dir=None, backend="embedded")

    config = EngineConfig.persistent(data_dir="state")

    assert config.data_dir == Path("state")
    assert config.backend == "embedded"


@pytest.mark.asyncio
async def test_in_memory_engine_never_discovers_the_user_home() -> None:
    with patch.object(Path, "home", side_effect=AssertionError("home accessed")):
        engine = create_engine(EngineConfig.in_memory())
        assert isinstance(engine, ContextEngine)
        created = await engine.pots.create(PotCreateRequest(name="demo", use=True))
        status = await engine.context.status(EngineStatusRequest())
        resolved = await engine.context.resolve(
            ResolveRequest(pot_id=created.pot_id, task="understand the project")
        )
        pots = await engine.pots.list(EmptyRequest())
        provision = await engine.provision.inspect(ProvisionInspectRequest())
        await engine.aclose()

    assert status.pot_id == created.pot_id
    assert status.pot_name == "demo"
    assert status.backend == "in_memory"
    assert resolved.pot_id == created.pot_id
    assert pots.count == 1
    assert provision.data_dir is None


@pytest.mark.asyncio
async def test_persistent_engine_uses_only_the_caller_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "engine-data"
    with patch.object(Path, "home", side_effect=AssertionError("home accessed")):
        engine = create_engine(EngineConfig.persistent(data_dir=data_dir))
        report = await engine.provision.apply(ProvisionApplyRequest())
        created = await engine.pots.create(PotCreateRequest(name="persisted"))
        await engine.aclose()

    assert report.ok is True
    assert created.name == "persisted"
    assert data_dir.joinpath("pots.json").is_file()
    assert data_dir.joinpath("graph.json").is_file()


@pytest.mark.asyncio
async def test_optional_http_application_factory_is_injected() -> None:
    marker = object()
    engine = create_engine(
        EngineConfig.in_memory(),
        EngineDependencies(http_application_factory=lambda _engine: marker),
    )

    assert engine.create_http_application() is marker
    await engine.aclose()
