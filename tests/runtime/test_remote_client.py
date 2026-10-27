"""The managed-host client: URL rules, the wire it sends, and version skew.

Skew is played against the reference host the way the design's end-to-end
skew step does: a host that lacks an operation, a host that lacks a request
field a newer client sends, and a newer host whose results carry fields this
client does not know. Unchanged calls keep working in every direction.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import httpx
import pytest

from potpie.runtime import ContextSelector, EngineOperation
from potpie.runtime import remote
from potpie.runtime.host_operations import (
    PotCreateRequest,
    PotListRequest,
    PotListResult,
    RepoDefaultSetRequest,
)
from potpie.runtime.remote import (
    HostNotConfigured,
    HttpsHostTransport,
    ManagedHostEndpoint,
    ManagedHostProbe,
    ManagedHostSettings,
    RemoteHostClient,
    RemoteHostError,
    connect,
    from_agent_request,
    without_fields,
)
from potpie_context_engine import Failure, Success
from potpie_context_engine.core.agent_envelope import AgentEnvelope
from potpie_context_engine.core.ports import agent_context as door
from potpie_context_engine.requests import (
    GraphSliceRequest,
    HistoryRequest,
    RecordRequest,
    ResolveRequest,
    SearchRequest,
    SubmitEventRequest,
)
from tests.support.reference_host import (
    ALICE_KEY,
    BOB_KEY,
    HOST_ID,
    EchoResult,
    ReferenceHost,
)

pytestmark = pytest.mark.unit

BASE = "https://your-host.example/context-graph"
SHOP = ContextSelector("explicit", "pot_shop")


# --- addresses ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "opt_in"),
    [
        ("https://your-host.example", None),
        ("https://your-host.example:8443/prefix/", None),
        ("http://localhost:8080", None),
        ("http://127.0.0.5:9000", None),
        ("http://[::1]:9000", None),
        ("http://graph.internal.example:8080", "graph.internal.example:8080"),
    ],
)
def test_accepted_urls(url: str, opt_in: str | None) -> None:
    endpoint = ManagedHostEndpoint(
        base_url=url, token="key", allow_insecure_host=opt_in
    )

    assert not endpoint.base_url.endswith("/")
    assert endpoint.operations_url == endpoint.base_url + "/v1/operations"


@pytest.mark.parametrize(
    ("url", "opt_in"),
    [
        ("http://your-host.example", None),
        ("http://graph.internal.example:8080", "graph.internal.example"),
        ("http://graph.internal.example:8080", "other.example:8080"),
        ("ftp://your-host.example", None),
        ("your-host.example", None),
    ],
)
def test_refused_urls(url: str, opt_in: str | None) -> None:
    with pytest.raises(ValueError):
        ManagedHostEndpoint(base_url=url, token="key", allow_insecure_host=opt_in)


def test_the_operations_url_joins_a_prefix_with_one_slash() -> None:
    endpoint = ManagedHostEndpoint(base_url=BASE + "///", token="key")

    assert endpoint.operations_url == BASE + "/v1/operations"
    assert endpoint.label == "Managed (your-host.example/context-graph)"


def test_an_empty_token_is_only_for_loopback() -> None:
    assert ManagedHostEndpoint(base_url="http://127.0.0.1:9", token="").token is None
    with pytest.raises(ValueError, match="loopback"):
        ManagedHostEndpoint(base_url=BASE, token="  ")


# --- the wire ------------------------------------------------------------------------


async def _resolve_via(transport: httpx.MockTransport, endpoint: ManagedHostEndpoint):
    connection = connect(
        url=endpoint.base_url,
        token=endpoint.token,
        allow_insecure_host=endpoint.allow_insecure_host,
        http_transport=transport,
    )
    try:
        return await connection.engine(SHOP).resolve(ResolveRequest(task="ship"))
    finally:
        await connection.aclose()


@pytest.mark.anyio
async def test_requests_carry_the_key_correlation_and_client_identity(
    tmp_path: Path,
) -> None:
    host = ReferenceHost.build(tmp_path)
    seen: list[httpx.Request] = []

    outcome = await _resolve_via(
        host.httpx_transport(seen=seen), ManagedHostEndpoint(BASE, token=ALICE_KEY)
    )

    assert isinstance(outcome, Success)
    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == BASE + "/v1/operations"
    assert request.headers["authorization"] == f"Bearer {ALICE_KEY}"
    assert request.headers["x-request-id"] == body["request_id"]
    assert request.headers["user-agent"].startswith("potpie/")
    # Ticketless and compact: no handshake came first, defaults stayed home.
    assert len(seen) == 1
    assert body["compatibility_ticket"] is None
    assert body["payload"] == {"task": "ship"}
    await host.close()


@pytest.mark.anyio
async def test_an_empty_loopback_token_sends_no_authorization_header(
    tmp_path: Path,
) -> None:
    host = ReferenceHost.build(tmp_path, allow_anonymous=True)
    seen: list[httpx.Request] = []

    outcome = await _resolve_via(
        host.httpx_transport(seen=seen),
        ManagedHostEndpoint("http://127.0.0.1:8765", token=None),
    )

    assert "authorization" not in seen[0].headers
    # Anonymous holds no grant on the pot, so selection says it is not there.
    assert isinstance(outcome, Failure) and outcome.error.code == "pot_not_found"
    await host.close()


@pytest.mark.anyio
async def test_a_key_over_opted_in_plain_http_warns_once_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = ReferenceHost.build(tmp_path)
    endpoint = ManagedHostEndpoint(
        "http://graph.internal.example:8080",
        token=ALICE_KEY,
        allow_insecure_host="graph.internal.example:8080",
    )

    for _ in range(2):
        assert isinstance(await _resolve_via(host.httpx_transport(), endpoint), Success)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("plain http") == 1
    await host.close()


@pytest.mark.anyio
async def test_a_refused_key_is_a_typed_authentication_failure(tmp_path: Path) -> None:
    host = ReferenceHost.build(tmp_path)

    outcome = await _resolve_via(
        host.httpx_transport(), ManagedHostEndpoint(BASE, token="not-a-key")
    )

    assert isinstance(outcome, Failure)
    assert outcome.error.category == "authentication"
    assert outcome.error.code == "authentication_failed"
    await host.close()


@pytest.mark.anyio
async def test_a_base_url_without_the_route_is_reported_as_such() -> None:
    def not_here(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="<html>no</html>")

    outcome = await _resolve_via(
        httpx.MockTransport(not_here), ManagedHostEndpoint(BASE, token="key")
    )

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "host_route_not_found"
    assert outcome.error.details["dispatched"] is False


# --- host operations, extensions and the probe ----------------------------------------


@pytest.mark.anyio
async def test_host_operations_and_extensions_round_trip(tmp_path: Path) -> None:
    host = ReferenceHost.build(tmp_path)
    connection = connect(
        url=BASE, token=ALICE_KEY, http_transport=host.httpx_transport()
    )
    client = connection.host

    handshake = await client.handshake()
    created = await client.pot_create(PotCreateRequest(name="docs", use=True))
    defaulted = await client.repo_default_set(
        ContextSelector("explicit", "docs"),
        RepoDefaultSetRequest(repo="github.com/a/b"),
    )
    listing = await client.pot_list(PotListRequest(repo="github.com/a/b"))
    who = await client.whoami()
    echo = await connection.extension(
        "ext.reference.echo", _EchoV1(text="hi"), EchoResult
    )

    assert isinstance(handshake, Success) and handshake.value.instance_id == HOST_ID
    assert isinstance(created, Success) and created.value.pot_id == "pot_docs"
    assert isinstance(defaulted, Success) and defaulted.value.pot_id == "pot_docs"
    assert isinstance(listing, Success) and isinstance(listing.value, PotListResult)
    assert listing.value.active_pot_id == "pot_docs"
    assert listing.value.repo_default_pot_id == "pot_docs"
    assert isinstance(who, Success) and who.value.subject == "alice"
    assert echo == Success(EchoResult(text="hi", by="alice"))
    await connection.aclose()
    await host.close()


@pytest.mark.anyio
async def test_the_handshake_ignores_a_different_catalog_fingerprint(
    tmp_path: Path,
) -> None:
    host = ReferenceHost.build(tmp_path)

    def other_build(document: dict[str, Any]) -> dict[str, Any]:
        value = document["outcome"].get("value")
        if isinstance(value, dict) and "operation_catalog_fingerprint" in value:
            value["operation_catalog_fingerprint"] = "0" * 64
        return document

    client = RemoteHostClient(
        _transport(host.httpx_transport(rewrite_response=other_build), ALICE_KEY)
    )

    assert isinstance(await client.handshake(), Success)
    await host.close()


@pytest.mark.anyio
async def test_the_probe_raises_on_failure_and_returns_pots(tmp_path: Path) -> None:
    host = ReferenceHost.build(tmp_path)
    good = ManagedHostProbe(
        ManagedHostEndpoint(BASE, token=BOB_KEY), http_transport=host.httpx_transport()
    )
    bad = ManagedHostProbe(
        ManagedHostEndpoint(BASE, token="nope"), http_transport=host.httpx_transport()
    )

    await good.handshake()
    listing = await good.pot_list()
    with pytest.raises(RemoteHostError) as refused:
        await bad.handshake()

    assert [pot.pot_id for pot in listing.pots] == ["pot_shop"]
    assert refused.value.error.code == "authentication_failed"
    await good.aclose()
    await bad.aclose()
    await host.close()


# --- version skew ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _EchoV1:
    text: str = ""


@dataclass(frozen=True, slots=True)
class _EchoV2:
    """A newer client's request: one optional field the host does not know."""

    text: str = ""
    shout: bool = False


@dataclass(frozen=True, slots=True)
class _OlderEchoResult:
    text: str = ""


def _transport(mock: httpx.MockTransport, key: str) -> HttpsHostTransport:
    return HttpsHostTransport(ManagedHostEndpoint(BASE, token=key), http_transport=mock)


@pytest.mark.anyio
async def test_an_operation_the_host_lacks_is_host_outdated(tmp_path: Path) -> None:
    older = ReferenceHost.build(
        tmp_path,
        engine_operations=set(EngineOperation) - {EngineOperation.GRAPH_SLICE},
        serve_host_operations=False,
    )
    connection = connect(
        url=BASE, token=ALICE_KEY, http_transport=older.httpx_transport()
    )

    sliced = await connection.engine(SHOP).graph_slice(GraphSliceRequest())
    listed = await connection.host.pot_list()
    unchanged = await connection.engine(SHOP).resolve(ResolveRequest())

    assert isinstance(sliced, Failure)
    assert sliced.error.code == "host_outdated"
    assert sliced.error.category == "protocol"
    assert sliced.error.details == {
        "host": "Managed (your-host.example/context-graph)",
        "operation": "graph_slice",
        "fields": (),
    }
    assert isinstance(listed, Failure)
    assert listed.error.details["operation"] == "host.pot_list"
    assert isinstance(unchanged, Success)
    await connection.aclose()
    await older.close()


@pytest.mark.anyio
async def test_a_field_the_host_lacks_is_host_outdated_and_can_be_dropped(
    tmp_path: Path,
) -> None:
    older = ReferenceHost.build(tmp_path)
    client = RemoteHostClient(_transport(older.httpx_transport(), ALICE_KEY))

    plain = await client.extension("ext.reference.echo", _EchoV2(text="a"), EchoResult)
    shouted = await client.extension(
        "ext.reference.echo", _EchoV2(text="a", shout=True), EchoResult
    )
    assert isinstance(shouted, Failure)
    retried = await client.extension(
        "ext.reference.echo",
        without_fields(_EchoV2(text="a", shout=True), shouted.error.details["fields"]),
        EchoResult,
    )

    # Leaving the new field at its default sends what an older client would.
    assert plain == Success(EchoResult(text="a", by="alice"))
    assert shouted.error.code == "host_outdated"
    assert shouted.error.details["fields"] == ("shout",)
    assert "'shout'" in shouted.error.message
    assert retried == Success(EchoResult(text="a", by="alice"))
    await older.close()


@pytest.mark.anyio
async def test_a_newer_clients_engine_field_is_named_by_the_older_host(
    tmp_path: Path,
) -> None:
    older = ReferenceHost.build(tmp_path)

    def newer_client(document: dict[str, Any]) -> dict[str, Any]:
        document["payload"]["include_widgets"] = True
        return document

    connection = connect(
        url=BASE,
        token=ALICE_KEY,
        http_transport=older.httpx_transport(rewrite_request=newer_client),
    )

    outcome = await connection.engine(SHOP).resolve(ResolveRequest())

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "host_outdated"
    assert outcome.error.details["operation"] == "resolve"
    assert outcome.error.details["fields"] == ("include_widgets",)
    await connection.aclose()
    await older.close()


@pytest.mark.anyio
async def test_a_newer_hosts_extra_result_fields_are_ignored(tmp_path: Path) -> None:
    newer = ReferenceHost.build(tmp_path)

    def newer_host(document: dict[str, Any]) -> dict[str, Any]:
        value = document["outcome"].get("value")
        if isinstance(value, dict):
            value["ranking_profile"] = "v9"
            for pot in value.get("pots", ()):
                pot["color"] = "teal"
        return document

    connection = connect(
        url=BASE,
        token=ALICE_KEY,
        http_transport=newer.httpx_transport(rewrite_response=newer_host),
    )

    resolved = await connection.engine(SHOP).resolve(ResolveRequest(task="t"))
    listed = await connection.host.pot_list()
    echoed = await connection.extension(
        "ext.reference.echo", _EchoV1(text="x"), _OlderEchoResult
    )

    assert isinstance(resolved, Success) and isinstance(resolved.value, AgentEnvelope)
    assert resolved.value.metadata == {"task": "t"}
    assert isinstance(listed, Success)
    assert [pot.name for pot in listed.value.pots] == ["shop", "lab"]
    assert echoed == Success(_OlderEchoResult(text="x"))
    await connection.aclose()
    await newer.close()


# --- helpers -------------------------------------------------------------------------


def test_without_fields_resets_nested_and_refuses_what_cannot_be_dropped() -> None:
    from potpie_context_engine.core.actor import Actor

    request = SubmitEventRequest(
        source_system="s", actor=Actor(user_id="u", surface="cli", client_name="c")
    )

    trimmed = without_fields(request, ["actor.client_name", "source_system"])

    assert trimmed.actor == Actor(user_id="u", surface="cli")
    assert trimmed.source_system == ""
    assert without_fields(HistoryRequest(include_claims=False), ["include_claims"]) == (
        HistoryRequest()
    )
    with pytest.raises(ValueError, match="no field"):
        without_fields(request, ["priority"])
    with pytest.raises(ValueError, match="no default"):
        without_fields(request, ["actor.user_id"])


def test_connect_without_a_url_reads_the_registry_hook(monkeypatch) -> None:
    monkeypatch.setattr(remote, "_registered_managed_host", lambda: None)
    with pytest.raises(HostNotConfigured):
        connect()
    with pytest.raises(ValueError, match="local runtime"):
        connect("local")

    monkeypatch.setattr(
        remote,
        "_registered_managed_host",
        lambda: ManagedHostSettings(base_url=BASE, token="registry-key"),
    )
    first = connect("managed")
    again = connect()

    assert first is again
    assert first.endpoint.base_url == BASE
    assert first.endpoint.token == "registry-key"


# --- long-lived threaded embedders ----------------------------------------------------


class _ClosingMock(httpx.MockTransport):
    """A mock host that records the event loop each pooled client is closed on."""

    def __init__(self, handler: Any) -> None:
        super().__init__(handler)
        self.closed_on: list[asyncio.AbstractEventLoop] = []

    async def aclose(self) -> None:
        self.closed_on.append(asyncio.get_running_loop())


def test_a_transport_used_from_several_loops_closes_every_client(
    tmp_path: Path,
) -> None:
    host = ReferenceHost.build(tmp_path)
    mock = _ClosingMock(host.httpx_transport().handler)
    transport = _transport(mock, ALICE_KEY)
    client = RemoteHostClient(transport)
    server_loop = asyncio.new_event_loop()
    server = threading.Thread(target=server_loop.run_forever, daemon=True)
    server.start()

    def on_server_loop(awaitable: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(awaitable, server_loop).result(30)

    try:
        # A long-lived loop on another thread, then two short-lived loops.
        assert isinstance(on_server_loop(client.whoami()), Success)
        assert isinstance(asyncio.run(client.whoami()), Success)
        assert mock.closed_on == []
        assert isinstance(asyncio.run(client.whoami()), Success)
        # The second short-lived loop closed the first one's client; the
        # long-lived loop's client is still its own and still serving.
        assert len(mock.closed_on) == 1
        assert isinstance(on_server_loop(client.whoami()), Success)
        assert len(mock.closed_on) == 1

        asyncio.run(transport.aclose())

        # Every client the three loops opened is closed, the live loop's on
        # that loop.
        assert len(mock.closed_on) == 3
        assert mock.closed_on.count(server_loop) == 1
        assert not transport.is_open
    finally:
        server_loop.call_soon_threadsafe(server_loop.stop)
        server.join(timeout=30)
        server_loop.close()
        asyncio.run(host.close())


def test_run_serves_engine_host_and_extension_calls_from_plain_threads(
    tmp_path: Path,
) -> None:
    host = ReferenceHost.build(tmp_path)
    connection = connect(
        url=BASE, token=ALICE_KEY, http_transport=host.httpx_transport()
    )
    answers: list[tuple[bool, str, EchoResult]] = []
    errors: list[BaseException] = []

    def servicer(index: int) -> None:
        try:
            resolved = connection.run(
                connection.engine(SHOP).resolve(ResolveRequest(task="ship"))
            )
            who = connection.run(connection.host.whoami(), timeout_s=30)
            echo = connection.run(
                connection.extension(
                    "ext.reference.echo", _EchoV1(text=str(index)), EchoResult
                )
            )
            answers.append(
                (isinstance(resolved, Success), who.value.subject, echo.value)
            )
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=servicer, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    assert sorted(answers, key=lambda row: row[2].text) == [
        (True, "alice", EchoResult(text=str(i), by="alice")) for i in range(6)
    ]
    connection.close()
    assert not connection.transport.is_open
    assert not connection.pots.client.transport.is_open
    remote._background.run(host.close())


def test_run_cancels_a_call_that_outlives_its_timeout() -> None:
    connection = connect(
        url=BASE,
        token="key",
        http_transport=httpx.MockTransport(lambda request: httpx.Response(404)),
    )
    finished = threading.Event()

    async def slow() -> None:
        try:
            await asyncio.sleep(30)
        finally:
            finished.set()

    with pytest.raises(TimeoutError):
        connection.run(slow(), timeout_s=0.05)

    assert finished.wait(timeout=5)


def test_run_refuses_to_block_its_own_loop() -> None:
    connection = connect(
        url=BASE,
        token="key",
        http_transport=httpx.MockTransport(lambda request: httpx.Response(404)),
    )

    async def nested() -> None:
        connection.run(asyncio.sleep(0))

    with pytest.raises(RuntimeError, match="its own event loop"):
        connection.run(nested())


@pytest.mark.parametrize(
    ("door_request", "engine_request"),
    [
        (
            door.ResolveRequest(
                pot_id="pot_shop", task="ship", include=("decisions",), max_items=3
            ),
            ResolveRequest(task="ship", include=("decisions",), max_items=3),
        ),
        (
            door.SearchRequest(pot_id="pot_shop", query="billing", mode="deep"),
            SearchRequest(query="billing", mode="deep"),
        ),
        (
            door.RecordRequest(
                pot_id="pot_shop",
                record_type="decision",
                summary="Use one pooled client",
                idempotency_key="k1",
            ),
            RecordRequest(
                record_type="decision",
                summary="Use one pooled client",
                idempotency_key="k1",
            ),
        ),
    ],
)
def test_an_agent_door_request_becomes_a_selector_and_an_engine_request(
    door_request: Any, engine_request: Any
) -> None:
    assert from_agent_request(door_request) == (SHOP, engine_request)
    # Every door field but the pot has an engine field, so nothing is dropped.
    assert {item.name for item in fields(door_request)} - {"pot_id"} == {
        item.name for item in fields(engine_request)
    }


def test_the_agent_door_adapter_refuses_what_it_cannot_carry() -> None:
    with pytest.raises(TypeError, match="StatusRequest"):
        from_agent_request(door.StatusRequest(pot_id="pot_shop"))  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="non-empty"):
        from_agent_request(door.SearchRequest(pot_id=" ", query="billing"))


@pytest.mark.anyio
async def test_an_agent_door_request_round_trips_to_the_host(tmp_path: Path) -> None:
    host = ReferenceHost.build(tmp_path)
    connection = connect(
        url=BASE, token=ALICE_KEY, http_transport=host.httpx_transport()
    )

    selector, request = from_agent_request(
        door.ResolveRequest(pot_id="pot_shop", task="ship")
    )
    outcome = await connection.engine(selector).resolve(request)

    assert isinstance(outcome, Success)
    assert isinstance(outcome.value, AgentEnvelope)
    await connection.aclose()
    await host.close()
