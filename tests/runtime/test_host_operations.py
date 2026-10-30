"""Host operations, extension operations and the ``graph_slice`` engine op.

A managed host serves ``host.*`` and ``ext.<vendor>.<op>`` operations beside
the engine catalog on the same envelope. The local daemon serves neither, so
neither changes its catalog fingerprint or capabilities.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from potpie.pots.contracts import PotAggregateStatus, PotInfo, SourceInfo
from potpie.runtime import (
    PROTOCOL_VERSION,
    AuthenticatedActor,
    AuthorizationScope,
    ContextResourceManager,
    ContextSelector,
    EngineOperation,
    OperationCoordinator,
    SafetyClass,
    SuccessResponse,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    operation_capabilities,
    operation_catalog_fingerprint,
)
from potpie.runtime.clients import TypedEngineOperationHandler, TypedOperationHandler
from potpie.runtime.commit_access import current_authorization_scope
from potpie.runtime.extensions import (
    HostExtension,
    extension_registry,
    operation_spec,
)
from potpie.runtime.host_operations import (
    HOST_OPERATION_CATALOG,
    GrantSetRequest,
    HostOperation,
    PotCounts,
    PotListRequest,
    PotListResult,
    PotUseRequest,
    SourceAddRequest,
    WhoamiRequest,
)
from potpie.runtime.local_engine import LocalEngineOperations
from potpie.runtime.protocol import HostOperationRequest
from potpie.runtime.resource_manager import PassThroughAuthenticator
from potpie.runtime.selection import CatalogSelectorResolver
from potpie_context_engine import (
    ContextEngine,
    ContextIdentity,
    EngineConfig,
    EngineDependencies,
    Failure,
    Success,
)
from potpie_context_engine.core.ports.graph.inspection import (
    GraphEdge,
    GraphNode,
    GraphSlice,
)
from potpie_context_engine.requests import GraphSliceRequest

pytestmark = pytest.mark.unit

ALICE = AuthenticatedActor(actor_id="alice")


# --- the catalog -----------------------------------------------------------------


EXPECTED = {
    "host.pot_list": ("shared_context_read", False),
    "host.pot_create": ("exclusive_resource_mutation", False),
    "host.pot_use": ("exclusive_resource_mutation", True),
    "host.repo_default_set": ("exclusive_resource_mutation", True),
    "host.repo_default_clear": ("exclusive_resource_mutation", False),
    "host.pot_status": ("shared_context_read", True),
    "host.source_list": ("shared_context_read", True),
    "host.source_add": ("exclusive_context_mutation", True),
    "host.whoami": ("shared_context_read", False),
    "host.grant_list": ("shared_context_read", True),
    "host.grant_set": ("exclusive_resource_mutation", True),
    "host.grant_revoke": ("exclusive_resource_mutation", True),
}


def test_the_host_catalog_has_the_twelve_operations_and_their_safety() -> None:
    actual = {
        spec.name: (spec.safety.value, spec.context_required)
        for spec in HOST_OPERATION_CATALOG.values()
    }

    assert actual == EXPECTED
    assert all(spec.family == "host" for spec in HOST_OPERATION_CATALOG.values())
    assert not any(spec.destructive for spec in HOST_OPERATION_CATALOG.values())


def test_host_operations_reuse_the_pot_contract_types() -> None:
    result_types = {
        spec.name: spec.result_type for spec in HOST_OPERATION_CATALOG.values()
    }

    assert result_types["host.pot_create"] is PotInfo
    assert result_types["host.pot_use"] is PotInfo
    assert result_types["host.source_add"] is SourceInfo
    assert result_types["host.pot_status"] is PotAggregateStatus


def test_the_local_daemon_catalog_does_not_carry_host_operations() -> None:
    capabilities = set(operation_capabilities())

    assert "graph_slice" in capabilities
    assert not any(name.startswith(("host.", "ext.")) for name in capabilities)
    # The fingerprint is computed from the engine and control catalogs only:
    # every host operation could change without touching it.
    from potpie.runtime import operations

    assert all(spec.operation in set(EngineOperation) for spec in operations._SPECS)
    assert len(operation_catalog_fingerprint()) == 64


def test_operation_spec_answers_for_every_family() -> None:
    extension = _echo_extension()

    assert operation_spec("resolve").safety is SafetyClass.SHARED_CONTEXT_READ
    assert (
        operation_spec("host.grant_set")
        is HOST_OPERATION_CATALOG[HostOperation.GRANT_SET]
    )
    assert operation_spec("ext.acme.echo", extensions=(extension,)) == extension.spec
    assert operation_spec("ext.acme.echo") is None
    assert operation_spec("nothing") is None


# --- extensions ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Echo:
    text: str = ""


@dataclass(frozen=True, slots=True)
class _Echoed:
    text: str = ""
    by: str = ""


def _echo_extension(**changes: Any) -> HostExtension:
    async def echo(request: _Echo, *, actor, context):
        assert context is None
        return _Echoed(text=request.text, by=actor.actor_id)

    values: dict[str, Any] = {
        "name": "ext.acme.echo",
        "request_type": _Echo,
        "result_type": _Echoed,
        "safety": SafetyClass.SHARED_CONTEXT_READ,
        "handler": echo,
    }
    values.update(changes)
    return HostExtension(**values)


@pytest.mark.parametrize(
    "name", ["host.echo", "ext.acme", "ext.Acme.echo", "ext.acme.echo.more", "echo"]
)
def test_extension_names_are_namespaced(name: str) -> None:
    with pytest.raises(ValueError, match="ext.<vendor>.<op>"):
        _echo_extension(name=name)


def test_extensions_need_dataclass_types_and_no_control_safety() -> None:
    with pytest.raises(TypeError, match="request type"):
        _echo_extension(request_type=dict)
    with pytest.raises(ValueError, match="control"):
        _echo_extension(safety=SafetyClass.DAEMON_LIFECYCLE_CONTROL)
    with pytest.raises(ValueError, match="resource conflict metadata"):
        _echo_extension(safety=SafetyClass.EXCLUSIVE_RESOURCE_MUTATION)
    with pytest.raises(ValueError, match="twice"):
        extension_registry([_echo_extension(), _echo_extension()])


# --- the wire --------------------------------------------------------------------


def _host_request(operation: str, payload: object, **changes: Any):
    values: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": f"request-{operation}",
        "operation": operation,
        "selector": ContextSelector("explicit", "pot_a"),
        "payload": payload,
    }
    values.update(changes)
    return HostOperationRequest(**values)


@pytest.mark.parametrize("operation", list(HostOperation))
def test_every_host_operation_round_trips_on_a_host_that_serves_it(
    operation: HostOperation,
) -> None:
    request = _host_request(operation, HOST_OPERATION_CATALOG[operation].request_type())

    document = encode_request(request)
    served = decode_request(document, host_operations=True)
    local = decode_request(document)

    assert document["operation"] == operation.value
    assert served == Success(request)
    assert isinstance(local, Failure)
    assert local.error.code == "operation_unknown"


def test_a_host_operation_request_checks_its_payload_type() -> None:
    with pytest.raises(ValueError, match="requires payload PotListRequest"):
        _host_request(HostOperation.POT_LIST, WhoamiRequest())
    with pytest.raises(ValueError, match="neither a host operation"):
        _host_request("pot_list", PotListRequest())


def test_extension_requests_decode_only_where_registered() -> None:
    extension = _echo_extension()
    request = _host_request("ext.acme.echo", _Echo(text="hi"))
    document = encode_request(request)

    registered = decode_request(document, extensions={extension.name: extension})
    unregistered = decode_request(document, host_operations=True)

    assert registered == Success(request)
    assert isinstance(unregistered, Failure)
    assert unregistered.error.code == "operation_unknown"


def test_unknown_host_request_fields_are_named() -> None:
    document = encode_request(_host_request(HostOperation.POT_LIST, PotListRequest()))
    document["payload"]["include_widgets"] = True

    decoded = decode_request(document, host_operations=True)

    assert isinstance(decoded, Failure)
    assert decoded.error.code == "request_fields_unsupported"
    assert decoded.error.details == {
        "operation": "host.pot_list",
        "fields": ("include_widgets",),
    }


def test_a_host_may_serve_a_subset_of_the_engine_catalog() -> None:
    from potpie.runtime import EngineOperationRequest

    request = EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="slice-1",
        operation=EngineOperation.GRAPH_SLICE,
        selector=ContextSelector("active"),
        payload=GraphSliceRequest(),
    )

    decoded = decode_request(
        encode_request(request), engine_operations={EngineOperation.RESOLVE}
    )

    assert isinstance(decoded, Failure)
    assert decoded.error.code == "operation_unknown"
    assert decoded.error.details == {"operation": "graph_slice"}


def test_host_results_decode_leniently_and_extension_results_stay_plain() -> None:
    request = _host_request(HostOperation.POT_LIST, PotListRequest(include_status=True))
    result = PotListResult(
        pots=(PotInfo(pot_id="pot_a", name="shop", active=True),),
        active_pot_id="pot_a",
        status={"pot_a": PotCounts(source_count=1, counts={"claims": 3})},
    )
    document = encode_response(
        SuccessResponse(
            protocol_version=PROTOCOL_VERSION,
            request_id=request.request_id,
            outcome=Success(result),
        )
    )
    document["outcome"]["value"]["pots"][0]["color"] = "teal"

    decoded = decode_response(document, request=request)
    echo = _host_request("ext.acme.echo", _Echo(text="hi"))
    plain = decode_response(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": echo.request_id,
            "outcome": {"ok": True, "value": {"text": "hi", "by": "alice"}},
        },
        request=echo,
    )

    assert isinstance(decoded, Success)
    assert decoded.value.outcome.value == result
    assert isinstance(plain, Success)
    assert plain.value.outcome.value.to_dict() == {"text": "hi", "by": "alice"}


# --- routing through the typed handler ---------------------------------------------


@dataclass
class _Catalog:
    pots: tuple[PotInfo, ...] = (PotInfo(pot_id="pot_a", name="shop"),)

    async def pots_for(self, actor):
        return self.pots if actor.actor_id == "alice" else ()

    async def active_for(self, actor):
        return None

    async def repo_default_for(self, actor, repo):
        return None

    async def repo_sources_for(self, actor):
        return ()


class _Authorizer:
    async def authorize(self, actor, operation, context):
        return Success(
            AuthorizationScope(
                actor_id=actor.actor_id,
                operation=operation,
                context=context,
                attributes={"role": "admin"},
            )
        )


class _NoComposer:
    async def fingerprint(self, context):  # pragma: no cover - never composed
        raise AssertionError("host operations compose no engine")

    async def compose(self, context, fingerprint):  # pragma: no cover
        raise AssertionError("host operations compose no engine")


@dataclass
class _HostHandler:
    calls: list[tuple[str, str, str | None, object]] = field(default_factory=list)
    scopes: list[AuthorizationScope | None] = field(default_factory=list)
    fail: Exception | None = None

    async def handle(self, operation, request, *, actor, context):
        self.calls.append(
            (
                operation.value,
                actor.actor_id,
                context.value if context else None,
                request,
            )
        )
        self.scopes.append(current_authorization_scope())
        if self.fail is not None:
            raise self.fail
        if operation is HostOperation.POT_USE:
            return PotInfo(pot_id=context.value, name="shop", active=True)
        return Success(PotListResult(pots=(PotInfo(pot_id="pot_a", name="shop"),)))


def _handler() -> TypedOperationHandler:
    manager = ContextResourceManager(
        resolver=CatalogSelectorResolver(_Catalog()),
        authenticator=PassThroughAuthenticator(),
        authorizer=_Authorizer(),
        composer=_NoComposer(),
    )
    return TypedOperationHandler(manager, coordinator=OperationCoordinator())


@pytest.mark.anyio
async def test_a_context_free_host_operation_is_authenticated_only() -> None:
    host = _HostHandler()

    outcome = await _handler().handle(
        _host_request(HostOperation.POT_LIST, PotListRequest()),
        authentication=ALICE,
        host_handler=host,
    )

    assert isinstance(outcome, Success)
    assert host.calls == [("host.pot_list", "alice", None, PotListRequest())]
    assert host.scopes == [None]


@pytest.mark.anyio
async def test_a_context_host_operation_is_resolved_and_authorized_first() -> None:
    host = _HostHandler()

    outcome = await _handler().handle(
        _host_request(HostOperation.POT_USE, PotUseRequest()),
        authentication=ALICE,
        host_handler=host,
    )

    assert outcome == Success(PotInfo(pot_id="pot_a", name="shop", active=True))
    assert host.calls[0][:3] == ("host.pot_use", "alice", "pot_a")
    scope = host.scopes[0]
    assert scope is not None
    assert scope.operation == "host.pot_use" and scope.attributes == {"role": "admin"}


@pytest.mark.anyio
async def test_selection_and_authentication_failures_never_reach_the_handler() -> None:
    host = _HostHandler()
    handler = _handler()

    unknown_pot = await handler.handle(
        _host_request(HostOperation.SOURCE_ADD, SourceAddRequest(kind="repo")),
        authentication=AuthenticatedActor(actor_id="carol"),
        host_handler=host,
    )
    anonymous = await handler.handle(
        _host_request(HostOperation.WHOAMI, WhoamiRequest()),
        authentication={"bearer": "forged"},
        host_handler=host,
    )

    assert isinstance(unknown_pot, Failure)
    assert unknown_pot.error.code == "pot_not_found"
    assert isinstance(anonymous, Failure)
    assert anonymous.error.category == "authentication"
    assert host.calls == []


@pytest.mark.anyio
async def test_a_host_handler_defect_is_a_typed_internal_error() -> None:
    host = _HostHandler(fail=RuntimeError("secret detail"))

    outcome = await _handler().handle(
        _host_request(HostOperation.GRANT_SET, GrantSetRequest(actor_id="bob")),
        authentication=ALICE,
        host_handler=host,
    )

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "operation_handler_failed"
    assert "secret" not in str(outcome.error)


@pytest.mark.anyio
async def test_unserved_host_and_extension_operations_are_unknown() -> None:
    handler = _handler()

    no_host = await handler.handle(
        _host_request(HostOperation.POT_LIST, PotListRequest()), authentication=ALICE
    )
    no_extension = await handler.handle(
        _host_request("ext.acme.echo", _Echo()), authentication=ALICE
    )

    for outcome in (no_host, no_extension):
        assert isinstance(outcome, Failure)
        assert outcome.error.code == "operation_unknown"


@pytest.mark.anyio
async def test_an_extension_runs_its_own_handler() -> None:
    outcome = await _handler().handle(
        _host_request("ext.acme.echo", _Echo(text="hi")),
        authentication=ALICE,
        extension=_echo_extension(),
    )

    assert outcome == Success(_Echoed(text="hi", by="alice"))


def test_the_former_handler_name_is_kept() -> None:
    assert TypedEngineOperationHandler is TypedOperationHandler


# --- graph_slice --------------------------------------------------------------------


class _Inspection:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def slice(self, *, pot_id, filter_):
        self.calls.append({"pot_id": pot_id, "filter": filter_})
        return GraphSlice(
            pot_id=pot_id,
            nodes=(GraphNode(key="service:a", labels=("Entity", "Service")),),
            edges=(GraphEdge(predicate="USES", from_key="service:a", to_key="db:b"),),
        )


@pytest.mark.anyio
async def test_graph_slice_reads_the_whole_graph_of_the_pot() -> None:
    inspection = _Inspection()
    operations = LocalEngineOperations(
        SimpleNamespace(backend=SimpleNamespace(inspection=inspection))
    )

    outcome = await operations.slice(
        ContextIdentity("pot_a"), GraphSliceRequest(include_invalidated=True, limit=10)
    )

    assert isinstance(outcome, Success)
    assert outcome.value.nodes[0].key == "service:a"
    query = inspection.calls[0]["filter"]
    assert inspection.calls[0]["pot_id"] == "pot_a"
    assert (query.pot_id, query.include_invalidated, query.limit) == ("pot_a", True, 10)


class _GraphWithoutSlice:
    async def catalog(self, context, request):  # pragma: no cover - unused
        return {}


@pytest.mark.anyio
async def test_an_engine_whose_graph_has_no_slice_answers_not_implemented() -> None:
    graph = _GraphWithoutSlice()
    engine = ContextEngine(
        context=ContextIdentity("pot_a"),
        config=EngineConfig(),
        dependencies=EngineDependencies(
            context=graph,  # type: ignore[arg-type]
            graph=graph,  # type: ignore[arg-type]
            workbench=graph,  # type: ignore[arg-type]
            ingestion=graph,  # type: ignore[arg-type]
            nudge=graph,  # type: ignore[arg-type]
        ),
    )

    outcome = await engine.slice(GraphSliceRequest())

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "not_implemented"
    assert outcome.error.details == {"operation": "graph_slice"}
