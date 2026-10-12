"""What an embedding host needs from the serving kit beyond the wire contract.

A real multi-tenant host keeps plans and inbox items in an async database
pool, builds the agent door over its own graph runtime, and drives the
endpoint from its own contract tests. Each of those is pinned here against a
variant of the reference host.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

import potpie.runtime.hosting as hosting
import potpie.runtime.hosting_testing as hosting_testing
from potpie.runtime.hosting import (
    ENGINE_OPERATION_FAILED,
    AgentContextService,
    ContextIdentity,
    ContextSelector,
    EngineOperation,
    EngineServices,
    Failure,
    ServiceEngineOperations,
)
from potpie.runtime.hosting_testing import (
    PROTOCOL_VERSION,
    DestructiveIntent,
    EngineOperationRequest,
    FailureResponse,
    HandshakePayload,
    HandshakeRequest,
    HandshakeResult,
    SuccessResponse,
    decode_response,
    encode_request,
)
from potpie_context_engine.api import CapabilityNotImplemented
from potpie_context_engine.core.ports.agent_context import StatusRequest
from potpie_context_engine.requests import (
    CommitRequest,
    CommitStatusRequest,
    InboxAddRequest,
    InboxClaimRequest,
    InboxListRequest,
    ProcessingStatusRequest,
    ProposeRequest,
    ReadRequest,
    RecordRequest,
    ResetContextRequest,
    ResolveRequest,
)
from tests.support.reference_host import ReferenceHost

pytestmark = pytest.mark.unit

SHOP = ContextSelector("explicit", "pot_shop")

_DEPENDENCY = {
    "operations": [
        {
            "op": "assert_claim",
            "subject": {"key": "service:billing", "type": "Service"},
            "predicate": "DEPENDS_ON",
            "object": {"key": "service:ledger", "type": "Service"},
            "truth": "agent_claim",
            "description": "Billing calls the ledger to post charges.",
        }
    ]
}


def _engine(operation: EngineOperation, payload: object) -> dict[str, Any]:
    return encode_request(
        EngineOperationRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=f"request-{operation.value}",
            operation=operation,
            selector=SHOP,
            payload=payload,  # type: ignore[arg-type]
        )
    )


def _value(response) -> Any:
    body = response.json()
    assert body["outcome"]["ok"] is True, body
    return body["outcome"]["value"]


# --- async-only plan and inbox stores ---------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("async_stores", [False, True], ids=["sync", "async-only"])
async def test_workbench_operations_run_on_async_only_host_stores(
    tmp_path: Path, async_stores: bool
) -> None:
    host = ReferenceHost.build(tmp_path, async_stores=async_stores)
    try:
        proposal = _value(
            await host.post(
                _engine(EngineOperation.PROPOSE, ProposeRequest(mutation=_DEPENDENCY))
            )
        )
        commit = _value(
            await host.post(
                _engine(
                    EngineOperation.COMMIT, CommitRequest(plan_id=proposal["plan_id"])
                )
            )
        )
        status = _value(
            await host.post(
                _engine(
                    EngineOperation.COMMIT_STATUS,
                    CommitStatusRequest(plan_id=proposal["plan_id"]),
                )
            )
        )
        added = _value(
            await host.post(
                _engine(
                    EngineOperation.INBOX_ADD,
                    InboxAddRequest(summary="Confirm the ledger owner."),
                )
            )
        )
        claimed = _value(
            await host.post(
                _engine(
                    EngineOperation.INBOX_CLAIM,
                    InboxClaimRequest(item_id=added["item"]["item_id"]),
                )
            )
        )
        listed = _value(
            await host.post(_engine(EngineOperation.INBOX_LIST, InboxListRequest()))
        )
    finally:
        await host.close()

    assert commit["ok"] is True and status["mutation_id"] == commit["mutation_id"]
    assert claimed["item"]["claimed_by"] == "alice"
    assert [item["item_id"] for item in listed["items"]] == [added["item"]["item_id"]]
    if async_stores:
        assert {"save", "compare_and_set"} <= set(host.plan_store.calls)
        assert {"save", "compare_and_set", "list"} <= set(host.inbox_store.calls)


# --- the agent door over a host's own graph runtime ---------------------------------


@pytest.mark.anyio
async def test_the_kit_agent_door_resolves_and_records_on_async_only_stores(
    tmp_path: Path,
) -> None:
    host = ReferenceHost.build(tmp_path, async_stores=True, agent_door=True)
    try:
        resolved = _value(
            await host.post(
                _engine(EngineOperation.RESOLVE, ResolveRequest(task="ship billing"))
            )
        )
        recorded = _value(
            await host.post(
                _engine(
                    EngineOperation.RECORD,
                    RecordRequest(
                        record_type="decision",
                        summary="Billing posts charges through the ledger.",
                        details={"rationale": "The ledger owns every balance."},
                        idempotency_key="decision-ledger",
                    ),
                )
            )
        )
    finally:
        await host.close()

    assert isinstance(host.agent_context, AgentContextService)
    assert resolved["pot_id"] == "pot_shop"
    # The door filled the intent the caller left unset.
    assert resolved["metadata"]["intent_source"] == "inferred"
    assert recorded["accepted"] is True and recorded["status"] == "recorded"
    # An idempotent record is a plan in the host's async-only store.
    assert "save" in host.plan_store.calls


def test_the_host_agent_door_needs_no_pot_or_skill_service(tmp_path: Path) -> None:
    host = ReferenceHost.build(tmp_path, agent_door=True)
    door = host.agent_context

    assert door.pots is None and door.skills is None
    assert door.graph is host.graph_runtime.graph
    assert door.workbench is host.graph_runtime.workbench
    with pytest.raises(CapabilityNotImplemented, match="agent_context.status"):
        door.status(StatusRequest(pot_id="pot_shop"))


# --- a host's contract tests speak the wire with the testing surface ---------------


@pytest.mark.anyio
async def test_a_contract_test_confirms_a_destructive_operation(
    tmp_path: Path,
) -> None:
    host = ReferenceHost.build(tmp_path)

    def reset(intent: DestructiveIntent | None) -> EngineOperationRequest:
        return EngineOperationRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id="reset-1",
            operation=EngineOperation.RESET_CONTEXT,
            selector=SHOP,
            payload=ResetContextRequest(),
            destructive_intent=intent,
        )

    confirmed = reset(
        DestructiveIntent(
            confirmed=True,
            operation=EngineOperation.RESET_CONTEXT.value,
            selector=SHOP,
            request_id="reset-1",
        )
    )
    unconfirmed = reset(None)
    try:
        answers = [
            decode_response(
                json.loads((await host.post(encode_request(request))).body),
                request=request,
            )
            for request in (confirmed, unconfirmed)
        ]
    finally:
        await host.close()

    done, refused = (answer.value for answer in answers)  # type: ignore[union-attr]
    assert isinstance(done, SuccessResponse)
    assert done.outcome.value.context_id == "pot_shop"
    assert isinstance(refused, FailureResponse)
    assert refused.outcome.error.code == "destructive_intent_invalid"


@pytest.mark.anyio
async def test_a_contract_test_reads_the_handshake_typed(tmp_path: Path) -> None:
    host = ReferenceHost.build(tmp_path)
    request = HandshakeRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="handshake-1",
        payload=HandshakePayload(),
    )
    try:
        reply = await host.post(encode_request(request))
    finally:
        await host.close()

    decoded = decode_response(json.loads(reply.body), request=request)
    result = decoded.value.outcome.value  # type: ignore[union-attr]
    assert isinstance(result, HandshakeResult)
    assert {"resolve", "reset_context", "host.pot_list"} <= set(result.capabilities)


def test_host_contract_tests_need_only_the_two_hosting_surfaces() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    root_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("potpie.")
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("potpie.")
    }

    assert root_imports == {"potpie.runtime.hosting", "potpie.runtime.hosting_testing"}
    assert "Provisional" in (hosting_testing.__doc__ or "")
    assert all(hasattr(hosting_testing, name) for name in hosting_testing.__all__)
    # One name, one home: the testing surface adds only the client half.
    assert set(hosting_testing.__all__).isdisjoint(hosting.__all__)


# --- a transport-neutral failure code ------------------------------------------------


class _Exploding:
    def read(self, request: Any) -> Any:
        raise RuntimeError("backend went away")

    def get_event(self, event_id: str) -> Any:
        raise RuntimeError("store went away")


@pytest.mark.anyio
async def test_an_unexpected_engine_failure_names_no_transport() -> None:
    exploding = _Exploding()
    operations = ServiceEngineOperations(
        EngineServices(
            pots=None,
            agent_context=None,
            graph=exploding,
            graph_workbench=None,
            backend=None,
            nudge=None,
            ingestion_events=exploding,
        )
    )
    context = ContextIdentity("pot_shop")

    read = await operations.read(context, ReadRequest(subgraph="core", view="v"))
    status = await operations.processing_status(
        context, ProcessingStatusRequest(event_id="event-1")
    )

    assert ENGINE_OPERATION_FAILED == "engine_operation_failed"
    for outcome, details in (
        (read, {"error_type": "RuntimeError"}),
        (status, {"operation": "processing_status", "error_type": "RuntimeError"}),
    ):
        assert isinstance(outcome, Failure)
        assert outcome.error.code == ENGINE_OPERATION_FAILED
        assert outcome.error.message == "the engine operation failed"
        assert outcome.error.details == details
