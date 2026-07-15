from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from potpie_context_engine import EngineConfig, create_engine
from potpie_context_engine.contracts import (
    EmptyRequest,
    EngineStatusRequest,
    GraphCatalogRequest,
    GraphReadRequest,
    LedgerStatusRequest,
    PotCreateRequest,
    ProvisionInspectRequest,
    RegisterRepoSourceRequest,
    ResolveRequest,
    SourceAddRequest,
    SourceListRequest,
    TimelineRecentRequest,
)

from potpie.daemon.main import create_app
from potpie.daemon.rpc import (
    ENGINE_RPC_REGISTRY,
    RPC_PROTOCOL_VERSION,
    dispatch_rpc,
)


def _payload(method: str, params: dict, request_id: str = "request-1") -> dict:
    return {
        "protocol_version": RPC_PROTOCOL_VERSION,
        "request_id": request_id,
        "method": method,
        "params": params,
    }


def test_registry_is_explicit_engine_only_and_fully_typed() -> None:
    methods = ENGINE_RPC_REGISTRY.methods()

    assert len(methods) == 47
    assert all(method.startswith("engine.") for method in methods)
    assert "engine.context.resolve" in methods
    assert "engine.provision.apply" in methods
    for method in methods:
        spec = ENGINE_RPC_REGISTRY.get(method)
        assert spec.request_type is not None
        assert spec.result_type is not None
        assert spec.request_adapter is not None
        assert spec.result_adapter is not None


@pytest.mark.asyncio
async def test_dispatch_rejects_protocol_unknown_method_and_malformed_params(
    tmp_path: Path,
) -> None:
    engine = create_engine(EngineConfig.in_memory())

    mismatch = await dispatch_rpc(
        engine,
        {
            "protocol_version": "0",
            "request_id": "mismatch",
            "method": "engine.context.status",
            "params": {},
        },
    )
    unknown = await dispatch_rpc(engine, _payload("engine.auth.login", {}))
    malformed = await dispatch_rpc(
        engine, _payload("engine.context.resolve", {"pot_id": 3})
    )
    await engine.aclose()

    assert mismatch["error"]["code"] == "RPC_PROTOCOL_MISMATCH"
    assert unknown["error"]["code"] == "RPC_METHOD_NOT_FOUND"
    assert malformed["error"]["code"] == "RPC_INVALID_PARAMS"


@pytest.mark.asyncio
async def test_local_and_registry_dispatch_have_capability_parity() -> None:
    engine = create_engine(EngineConfig.in_memory())
    create_request = PotCreateRequest(name="parity", use=True)
    local_pot = await engine.pots.create(create_request)
    source_request = SourceAddRequest(
        pot_id=local_pot.pot_id,
        kind="repo",
        location="acme/widgets",
    )
    source_response = await dispatch_rpc(
        engine,
        _payload(
            "engine.sources.add",
            ENGINE_RPC_REGISTRY.encode_request("engine.sources.add", source_request),
            "source-add",
        ),
    )
    assert source_response["ok"] is True
    register_request = RegisterRepoSourceRequest(
        pot_id=local_pot.pot_id,
        location="github.com/acme/widgets",
        make_default=True,
    )
    register_response = await dispatch_rpc(
        engine,
        _payload(
            "engine.sources.register_repo",
            ENGINE_RPC_REGISTRY.encode_request(
                "engine.sources.register_repo", register_request
            ),
            "source-register",
        ),
    )
    assert register_response["ok"] is True

    cases = (
        ("engine.pots.list", EmptyRequest(), await engine.pots.list(EmptyRequest())),
        (
            "engine.context.status",
            EngineStatusRequest(pot_id=local_pot.pot_id),
            await engine.context.status(EngineStatusRequest(pot_id=local_pot.pot_id)),
        ),
        (
            "engine.context.resolve",
            ResolveRequest(pot_id=local_pot.pot_id, task="parity"),
            await engine.context.resolve(
                ResolveRequest(pot_id=local_pot.pot_id, task="parity")
            ),
        ),
        (
            "engine.graph.catalog",
            GraphCatalogRequest(pot_id=local_pot.pot_id),
            await engine.graph.catalog(GraphCatalogRequest(pot_id=local_pot.pot_id)),
        ),
        (
            "engine.graph.read",
            GraphReadRequest(
                pot_id=local_pot.pot_id,
                subgraph="recent_changes",
                view="timeline",
            ),
            await engine.graph.read(
                GraphReadRequest(
                    pot_id=local_pot.pot_id,
                    subgraph="recent_changes",
                    view="timeline",
                )
            ),
        ),
        (
            "engine.ledger.status",
            LedgerStatusRequest(),
            await engine.ledger.status(LedgerStatusRequest()),
        ),
        (
            "engine.timeline.recent",
            TimelineRecentRequest(pot_id=local_pot.pot_id),
            await engine.timeline.recent(
                TimelineRecentRequest(pot_id=local_pot.pot_id)
            ),
        ),
        (
            "engine.provision.inspect",
            ProvisionInspectRequest(pot_id=local_pot.pot_id),
            await engine.provision.inspect(
                ProvisionInspectRequest(pot_id=local_pot.pot_id)
            ),
        ),
    )
    source_list = SourceListRequest(pot_id=local_pot.pot_id)
    cases += (
        (
            "engine.sources.list",
            source_list,
            await engine.sources.list(source_list),
        ),
    )

    for index, (method, request, local_result) in enumerate(cases):
        response = await dispatch_rpc(
            engine,
            _payload(
                method,
                ENGINE_RPC_REGISTRY.encode_request(method, request),
                f"parity-{index}",
            ),
        )
        assert response["ok"] is True, response
        remote_result = ENGINE_RPC_REGISTRY.decode_result(method, response["result"])
        assert remote_result == local_result

    await engine.aclose()


def test_daemon_http_exposes_healthz_and_typed_rpc_only(tmp_path: Path) -> None:
    engine = create_engine(EngineConfig.in_memory())
    app = create_app(
        token="secret",
        base_url="http://127.0.0.1:1",
        pid=123,
        log_file=str(tmp_path / "daemon.log"),
        engine=engine,
        data_dir=tmp_path,
    )

    with TestClient(app) as client:
        health = client.get("/healthz")
        unauthorized = client.post("/rpc", json=_payload("engine.context.status", {}))
        response = client.post(
            "/rpc",
            headers={"Authorization": "Bearer secret"},
            json=_payload("engine.context.status", {}),
        )
        removed_attr = client.post(
            "/attr",
            headers={"Authorization": "Bearer secret"},
            json={"surface": "backend", "name": "profile"},
        )

    assert health.json()["protocol_version"] == "1"
    assert unauthorized.status_code == 401
    assert response.json()["ok"] is True
    assert removed_attr.status_code == 404
