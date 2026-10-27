"""The serving kit's endpoint contract, against the reference multi-actor host.

Everything here goes through ``OperationEndpoint.handle(body, headers)`` with
JSON a remote client would send, so it pins the wire behaviour an embedding
host gets from the kit: authentication first, per-actor selection, typed
refusals and their HTTP statuses, ticketless compatibility, request limits,
caller-assertion rules and scope-derived commit access.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from potpie.runtime import (
    PROTOCOL_VERSION,
    ContextSelector,
    DaemonStatusPayload,
    DaemonStatusRequest,
    EngineOperation,
    EngineOperationRequest,
    HandshakePayload,
    HandshakeRequest,
    ShutdownPayload,
    ShutdownRequest,
    encode_request,
)
from potpie.runtime.host_operations import (
    GrantSetRequest,
    HostOperation,
    PotCreateRequest,
    PotListRequest,
    PotUseRequest,
    RepoDefaultSetRequest,
    SourceAddRequest,
    WhoamiRequest,
)
from potpie.runtime.protocol import HostOperationRequest
from potpie_context_engine.requests import (
    CommitRequest,
    DisableRollbackRequest,
    ExportSnapshotRequest,
    GraphSliceRequest,
    InboxClaimRequest,
    JournalStatusRequest,
    RecordRequest,
    ResolveRequest,
)
from tests.support.reference_host import (
    ALICE_KEY,
    BOB_KEY,
    CAROL_KEY,
    HOST_ID,
    HOST_VERSION,
    EchoRequest,
    ReferenceHost,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
async def host(tmp_path: Path):
    reference = ReferenceHost.build(tmp_path)
    yield reference
    await reference.close()


def _engine(
    operation: EngineOperation,
    payload: object,
    selector: ContextSelector = ContextSelector("explicit", "pot_shop"),
    *,
    request_id: str = "request-1",
) -> dict[str, Any]:
    return encode_request(
        EngineOperationRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=request_id,
            operation=operation,
            selector=selector,
            payload=payload,  # type: ignore[arg-type]
        )
    )


def _host(
    operation: str,
    payload: object,
    selector: ContextSelector = ContextSelector("explicit", "pot_shop"),
) -> dict[str, Any]:
    return encode_request(
        HostOperationRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=f"request-{operation}",
            operation=operation,
            selector=selector,
            payload=payload,
        )
    )


def _error(response) -> dict[str, Any]:
    body = response.json()
    assert body["outcome"]["ok"] is False, body
    return body["outcome"]["error"]


def _value(response) -> Any:
    body = response.json()
    assert body["outcome"]["ok"] is True, body
    return body["outcome"]["value"]


# --- authentication comes first -----------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("key", [None, "wrong-key"])
async def test_an_unauthenticated_request_is_refused_before_anything_is_decoded(
    host: ReferenceHost, key: str | None
) -> None:
    probe = _engine(EngineOperation.RESOLVE, ResolveRequest(), request_id="probe-1")
    probe["operation"] = "not.a.real.operation"

    response = await host.post(probe, key=key)

    assert response.status == 401
    error = _error(response)
    assert error["category"] == "authentication"
    assert response.json()["request_id"] == "probe-1"
    assert response.headers["X-Request-ID"] == "probe-1"
    assert host.agent_context.calls == []


@pytest.mark.anyio
async def test_an_unauthenticated_caller_cannot_tell_which_pots_exist(
    host: ReferenceHost,
) -> None:
    unauthenticated = [
        await host.post(
            _engine(
                EngineOperation.RESOLVE,
                ResolveRequest(),
                ContextSelector("explicit", ref),
            ),
            key=None,
        )
        for ref in ("pot_shop", "pot_missing")
    ]
    carol = [
        await host.post(
            _engine(
                EngineOperation.RESOLVE,
                ResolveRequest(),
                ContextSelector("explicit", ref),
            ),
            key=CAROL_KEY,
        )
        for ref in ("pot_shop", "pot_missing")
    ]

    assert [r.status for r in unauthenticated] == [401, 401]
    assert _error(unauthenticated[0]) == _error(unauthenticated[1])
    assert [_error(r)["code"] for r in carol] == ["pot_not_found", "pot_not_found"]


@pytest.mark.anyio
async def test_anonymous_access_exists_only_when_the_host_enables_it(
    tmp_path: Path,
) -> None:
    open_host = ReferenceHost.build(tmp_path / "open", allow_anonymous=True)
    closed_host = ReferenceHost.build(tmp_path / "closed")
    document = _host(HostOperation.WHOAMI, WhoamiRequest())

    allowed = await open_host.post(document, key=None)
    refused = await closed_host.post(document, key=None)

    assert _value(allowed)["subject"] == "anonymous"
    assert refused.status == 401
    assert _error(refused)["code"] == "authentication_required"


# --- handshake, status and control on a ticketless host ----------------------------


@pytest.mark.anyio
async def test_the_handshake_names_a_stable_host_and_everything_it_serves(
    host: ReferenceHost,
) -> None:
    response = await host.post(
        encode_request(
            HandshakeRequest(
                protocol_version=PROTOCOL_VERSION,
                request_id="handshake-1",
                payload=HandshakePayload(client_operation_catalog_fingerprint="0" * 64),
            )
        )
    )

    result = _value(response)
    assert result["instance_id"] == HOST_ID
    assert result["compatibility_ticket"] == "stateless"
    assert result["lifecycle_state"] == "ready"
    capabilities = set(result["capabilities"])
    assert {
        "resolve",
        "graph_slice",
        "host.pot_list",
        "ext.reference.echo",
    } <= capabilities
    assert "daemon.handshake" in capabilities and "daemon.status" in capabilities
    assert "daemon.shutdown" not in capabilities


@pytest.mark.anyio
async def test_operations_need_no_handshake_and_no_ticket(host: ReferenceHost) -> None:
    document = _engine(EngineOperation.RESOLVE, ResolveRequest(task="ship"))
    assert document["compatibility_ticket"] is None

    response = await host.post(document)

    assert response.status == 200
    assert _value(response)["pot_id"] == "pot_shop"


@pytest.mark.anyio
async def test_status_reports_the_host_build_and_shutdown_is_refused(
    host: ReferenceHost,
) -> None:
    status = await host.post(
        encode_request(
            DaemonStatusRequest(
                protocol_version=PROTOCOL_VERSION,
                request_id="status-1",
                payload=DaemonStatusPayload(),
            )
        )
    )
    shutdown = await host.post(
        encode_request(
            ShutdownRequest(
                protocol_version=PROTOCOL_VERSION,
                request_id="shutdown-1",
                payload=ShutdownPayload(),
            )
        )
    )

    assert _value(status)["version"] == HOST_VERSION
    assert _value(status)["instance_id"] == HOST_ID
    assert shutdown.status == 403
    assert _error(shutdown)["category"] == "authorization"
    assert _error(shutdown)["code"] == "host_control_refused"
    assert host.endpoint.lifecycle_state == "ready"


@pytest.mark.anyio
async def test_a_draining_host_refuses_new_work_as_retryable(
    host: ReferenceHost,
) -> None:
    await host.endpoint.begin_drain()

    response = await host.post(_engine(EngineOperation.RESOLVE, ResolveRequest()))

    assert response.status == 503
    error = _error(response)
    assert (error["code"], error["retry_posture"]) == ("daemon_draining", "safe")


# --- limits and precise refusals ---------------------------------------------------


@pytest.mark.anyio
async def test_a_body_over_the_limit_is_a_typed_413(tmp_path: Path) -> None:
    host = ReferenceHost.build(tmp_path, max_request_bytes=2048)

    response = await host.post(
        _engine(EngineOperation.RESOLVE, ResolveRequest(task="x" * 4096))
    )

    assert response.status == 413
    error = _error(response)
    assert error["code"] == "request_too_large"
    assert error["details"] == {"max_bytes": 2048}


@pytest.mark.anyio
async def test_unknown_fields_and_operations_are_named(tmp_path: Path) -> None:
    host = ReferenceHost.build(
        tmp_path, engine_operations={EngineOperation.RESOLVE, EngineOperation.RECORD}
    )
    extra_field = _engine(EngineOperation.RESOLVE, ResolveRequest())
    extra_field["payload"]["ranking"] = {"profile": "v9"}

    refused_field = await host.post(extra_field)
    refused_op = await host.post(
        _engine(EngineOperation.GRAPH_SLICE, GraphSliceRequest())
    )

    assert refused_field.status == 400
    assert _error(refused_field)["code"] == "request_fields_unsupported"
    assert _error(refused_field)["details"] == {
        "operation": "resolve",
        "fields": ["ranking"],
    }
    assert _error(refused_op)["code"] == "operation_unknown"
    assert _error(refused_op)["details"] == {"operation": "graph_slice"}


# --- roles, caller assertions and commit access -------------------------------------


@pytest.mark.anyio
async def test_a_reader_reads_but_cannot_write(host: ReferenceHost) -> None:
    read = await host.post(
        _engine(EngineOperation.RESOLVE, ResolveRequest()), key=BOB_KEY
    )
    write = await host.post(
        _engine(EngineOperation.RECORD, RecordRequest(record_type="note", summary="s")),
        key=BOB_KEY,
    )
    admin_write = await host.post(
        _engine(EngineOperation.RECORD, RecordRequest(record_type="note", summary="s"))
    )

    assert read.status == 200
    assert write.status == 403
    assert _error(write)["code"] == "role_insufficient"
    assert admin_write.status == 200
    assert host.agent_context.calls == [("resolve", "pot_shop"), ("record", "pot_shop")]


@pytest.mark.anyio
async def test_commit_access_follows_the_role_the_request_was_authorized_under(
    host: ReferenceHost,
) -> None:
    for pot_id in ("pot_lab", "pot_shop"):
        host.graph_runtime.backend.journal.activate(
            pot_id=pot_id, rollback_enabled=True
        )
    lab = ContextSelector("explicit", "pot_lab")

    status = await host.post(
        _engine(EngineOperation.JOURNAL_STATUS, JournalStatusRequest(), lab)
    )
    # Alice is only a writer on pot_lab: the authorizer lets a write through,
    # and the commit service, asking for admin, refuses it from her scope.
    disable = await host.post(
        _engine(EngineOperation.DISABLE_ROLLBACK, DisableRollbackRequest(), lab)
    )
    as_admin = await host.post(
        _engine(EngineOperation.DISABLE_ROLLBACK, DisableRollbackRequest())
    )

    assert _value(status)["document"]["ok"] is True
    assert _error(disable)["code"] == "commit_access_denied"
    assert _value(as_admin)["document"]["ok"] is True
    assert _value(as_admin)["document"]["state"]["rollback_enabled"] is False


@pytest.mark.anyio
async def test_caller_assertions_are_checked_against_the_credential(
    host: ReferenceHost,
) -> None:
    approval = await host.post(
        _engine(EngineOperation.COMMIT, CommitRequest(plan_id="p", approved_by="bob"))
    )
    claim = await host.post(
        _engine(
            EngineOperation.INBOX_CLAIM,
            InboxClaimRequest(item_id="i", claimed_by="bob"),
        )
    )
    host_path = await host.post(
        _engine(
            EngineOperation.EXPORT_SNAPSHOT, ExportSnapshotRequest(destination="out")
        )
    )

    assert approval.status == 403
    assert _error(approval)["code"] == "caller_assertion_mismatch"
    assert _error(approval)["details"] == {
        "operation": "commit",
        "field": "approved_by",
    }
    assert _error(claim)["details"] == {
        "operation": "inbox_claim",
        "field": "claimed_by",
    }
    assert _error(host_path)["code"] == "host_path_refused"


# --- host and extension operations ----------------------------------------------------


@pytest.mark.anyio
async def test_pot_listing_is_per_actor(host: ReferenceHost) -> None:
    listings = {
        name: _value(
            await host.post(_host(HostOperation.POT_LIST, PotListRequest()), key=key)
        )
        for name, key in (("alice", ALICE_KEY), ("bob", BOB_KEY), ("carol", CAROL_KEY))
    }

    assert [pot["pot_id"] for pot in listings["alice"]["pots"]] == [
        "pot_shop",
        "pot_lab",
    ]
    assert [pot["pot_id"] for pot in listings["bob"]["pots"]] == ["pot_shop"]
    assert listings["carol"]["pots"] == []


@pytest.mark.anyio
async def test_the_onboarding_sequence_runs_through_host_operations(
    host: ReferenceHost,
) -> None:
    created = await host.post(
        _host(HostOperation.POT_CREATE, PotCreateRequest(name="docs"))
    )
    used = await host.post(
        _host(
            HostOperation.POT_USE, PotUseRequest(), ContextSelector("explicit", "docs")
        )
    )
    defaulted = await host.post(
        _host(
            HostOperation.REPO_DEFAULT_SET,
            RepoDefaultSetRequest(repo="github.com/acme/shop"),
            ContextSelector("explicit", "docs"),
        )
    )
    source = await host.post(
        _host(
            HostOperation.SOURCE_ADD,
            SourceAddRequest(kind="repo", location="github.com/acme/shop"),
            ContextSelector("active"),
        )
    )
    listing = await host.post(
        _host(
            HostOperation.POT_LIST,
            PotListRequest(repo="github.com/acme/shop", include_status=True),
        )
    )
    resolved = await host.post(
        _engine(
            EngineOperation.RESOLVE,
            ResolveRequest(),
            ContextSelector("repository", "github.com/acme/shop"),
        )
    )

    assert _value(created)["pot_id"] == "pot_docs"
    assert _value(used)["active"] is True
    assert _value(defaulted) == {"repo": "github.com/acme/shop", "pot_id": "pot_docs"}
    assert _value(source)["kind"] == "repo"
    listed = _value(listing)
    assert listed["active_pot_id"] == "pot_docs"
    assert listed["repo_default_pot_id"] == "pot_docs"
    assert listed["status"]["pot_docs"]["source_count"] == 1
    assert _value(resolved)["pot_id"] == "pot_docs"
    # The handler saw the authenticated actor through current_actor_id().
    assert set(host.state.handled_by) == {"alice"}


@pytest.mark.anyio
async def test_sharing_is_admin_only_and_takes_effect(host: ReferenceHost) -> None:
    grant = _host(
        HostOperation.GRANT_SET, GrantSetRequest(actor_id="carol", role="reader")
    )

    by_reader = await host.post(grant, key=BOB_KEY)
    by_admin = await host.post(grant)
    carol_reads = await host.post(
        _engine(EngineOperation.RESOLVE, ResolveRequest()), key=CAROL_KEY
    )

    assert _error(by_reader)["code"] == "role_insufficient"
    assert _value(by_admin) == {
        "pot_id": "pot_shop",
        "actor_id": "carol",
        "role": "reader",
    }
    assert _value(carol_reads)["pot_id"] == "pot_shop"


@pytest.mark.anyio
async def test_an_extension_operation_is_served_with_the_actor(
    host: ReferenceHost,
) -> None:
    response = await host.post(
        _host("ext.reference.echo", EchoRequest(text="hi")), key=BOB_KEY
    )

    assert _value(response) == {"text": "hi", "by": "bob"}


@pytest.mark.anyio
async def test_graph_slice_serves_the_whole_graph(host: ReferenceHost) -> None:
    response = await host.post(
        _engine(EngineOperation.GRAPH_SLICE, GraphSliceRequest())
    )

    value = _value(response)
    assert value["pot_id"] == "pot_shop"
    assert value["nodes"] == [] and value["truncated"] is False


# --- the kit's import surface is enough -------------------------------------------


def test_the_reference_host_uses_only_the_hosting_surface() -> None:
    source = ROOT / "tests" / "support" / "reference_host.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    root_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "potpie" or node.module.startswith("potpie."))
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "potpie" or alias.name.startswith("potpie.")
    }

    assert root_imports == {"potpie.runtime.hosting"}


def test_the_hosting_surface_exports_only_what_it_imports() -> None:
    import potpie.runtime.hosting as hosting

    assert sorted(hosting.__all__) == sorted(set(hosting.__all__))
    assert all(hasattr(hosting, name) for name in hosting.__all__)
    json.dumps(hosting.__all__)
