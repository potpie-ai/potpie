"""A client newer than its host: old features keep working, new ones say why not.

The codec rebuilds objects with a strict ``cls(**raw)``, so every field this
client's build added — however optional — used to reach an older host as
``ClaimQueryFilter.__init__() got an unexpected keyword argument
'include_retired'``, and every method it added as a validation error. Two rules
fix that from the client's side alone, because the older host cannot change:

* defaults stay off the wire, so a call that does not use a new field is the
  call an older client would have made;
* a refusal naming something this call sent is :class:`HostOutdated` — a
  capability answer with a repair — rather than a broken daemon or a caller
  mistake, and callers can retry an older feature without its new options.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from potpie.cli import hosts
from potpie.daemon.client import (
    DaemonRpcClient,
    HostOutdated,
    _raise_remote_error,
    _sent_names,
)
from potpie.daemon.http.ui.auth import UiAuth
from potpie.daemon.http.ui.router import build_ui_api_router
from potpie.daemon.rpc import decode, encode
from potpie_context_core.errors import ContextEngineDisabled
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.ports.graph.inspection import GraphSlice
from tests._rpc_fakes import install_rpc_session

_URL = "https://graph.example.com"
_LABEL = "Managed (graph.example.com)"
_TOKEN = "host-outdated-test-token"  # noqa: S105 - non-secret test fixture


class _Endpoint:
    def discovery(self) -> dict[str, str]:
        return {"base_url": _URL, "token": "key"}


def _client() -> DaemonRpcClient:
    return DaemonRpcClient(daemon=_Endpoint(), label=_LABEL)


def _envelope(code: str, message: str):
    def _post(url: str, **_: object) -> httpx.Response:
        body = {"ok": False, "error": {"code": code, "message": message}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    return _post


# --- defaults stay off the wire ---------------------------------------------


def test_a_default_filter_sends_only_what_an_older_client_would() -> None:
    wire = encode(ClaimQueryFilter(pot_id="p"), omit_defaults=True)

    assert wire["value"] == {"pot_id": "p"}


def test_a_field_the_call_uses_is_still_sent() -> None:
    wire = encode(
        ClaimQueryFilter(pot_id="p", include_invalidated=True, include_retired=True),
        omit_defaults=True,
    )

    assert wire["value"] == {
        "pot_id": "p",
        "include_invalidated": True,
        "include_retired": True,
    }


def test_omitted_defaults_decode_to_the_same_object() -> None:
    value = ClaimQueryFilter(pot_id="p", predicate_in=("OWNS",), limit=5)

    assert decode(encode(value, omit_defaults=True)) == value


def test_an_older_class_decodes_what_a_newer_client_sends() -> None:
    """The older host's view: its class lacks fields this build added."""

    @dataclass(frozen=True)
    class OlderFilter:
        pot_id: str
        include_invalidated: bool = False

    wire = encode(ClaimQueryFilter(pot_id="p"), omit_defaults=True)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        OlderFilter(**encode(ClaimQueryFilter(pot_id="p"))["value"])
    assert OlderFilter(**wire["value"]) == OlderFilter(pot_id="p")


def test_a_value_equal_but_not_the_default_type_is_sent() -> None:
    @dataclass(frozen=True)
    class Flags:
        flag: bool = False

    assert encode(Flags(flag=0), omit_defaults=True)["value"] == {"flag": 0}  # type: ignore[arg-type]


def test_responses_keep_every_field() -> None:
    """Only requests omit: other readers of an encoded value expect all of it."""
    assert "include_retired" in encode(ClaimQueryFilter(pot_id="p"))["value"]


def test_the_client_puts_no_default_on_the_wire(monkeypatch) -> None:
    seen: list[dict] = []

    def _post(url: str, **kwargs: object) -> httpx.Response:
        seen.append(kwargs["json"])  # type: ignore[arg-type]
        body = {"ok": True, "result": encode(GraphSlice(pot_id="p"))}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    install_rpc_session(monkeypatch, _post)

    result = _client().call(
        "backend.inspection",
        "slice",
        pot_id="p",
        filter_=ClaimQueryFilter(pot_id="p"),
    )

    assert result == GraphSlice(pot_id="p")
    assert "include_retired" not in json.dumps(seen[0])


# --- a refusal from an older host is a capability answer ---------------------


def test_an_unknown_field_the_call_sent_is_an_outdated_host(monkeypatch) -> None:
    install_rpc_session(
        monkeypatch,
        _envelope(
            "daemon_error",
            "ClaimQueryFilter.__init__() got an unexpected keyword argument "
            "'include_retired'",
        ),
    )

    with pytest.raises(HostOutdated) as raised:
        _client().call(
            "backend.inspection",
            "slice",
            pot_id="p",
            filter_=ClaimQueryFilter(pot_id="p", include_retired=True),
        )

    assert raised.value.argument == "include_retired"
    assert raised.value.capability == "backend.inspection.slice"
    assert "older Potpie" in str(raised.value)
    assert raised.value.recommended_next_action


def test_an_unknown_keyword_the_call_sent_is_an_outdated_host(monkeypatch) -> None:
    install_rpc_session(
        monkeypatch,
        _envelope(
            "daemon_error",
            "history() got an unexpected keyword argument 'include_claims'",
        ),
    )

    with pytest.raises(HostOutdated) as raised:
        _client().call("graph_workbench", "history", pot_id="p", include_claims=False)

    assert raised.value.argument == "include_claims"


@pytest.mark.parametrize(
    "message",
    [
        # A name this call never sent.
        "GraphService.read() got an unexpected keyword argument 'cursor'",
        # A name it did send, refused by a function it did not call: the host's
        # own code passing a bad keyword internally.
        "_build_reader() got an unexpected keyword argument 'pot_id'",
        # A field name it sent, refused by a class it did not send.
        "GraphSlice.__init__() got an unexpected keyword argument 'pot_id'",
    ],
)
def test_a_host_bug_stays_a_host_fault(message: str) -> None:
    """The guard that keeps a host's own TypeError from being retold as its age."""
    sent = frozenset({("", "pot_id"), ("", "request"), ("GraphReadRequest", "pot_id")})

    with pytest.raises(ContextEngineDisabled) as raised:
        _raise_remote_error(
            {"ok": False, "error": {"code": "daemon_error", "message": message}},
            surface="graph",
            method="read",
            sent=sent,
        )

    assert not isinstance(raised.value, HostOutdated)


def test_a_method_the_host_lacks_is_an_outdated_host() -> None:
    """A managed service's allowlist refusing a member it predates."""
    with pytest.raises(HostOutdated) as raised:
        _raise_remote_error(
            {
                "ok": False,
                "error": {
                    "code": "validation_error",
                    "message": "invalid RPC member: commits",
                },
            },
            label=_LABEL,
            surface="graph_workbench",
            method="commits",
        )

    assert raised.value.argument is None
    assert raised.value.capability == "graph_workbench.commits"


def test_a_missing_attribute_is_not_read_as_age() -> None:
    """A local daemon's ``AttributeError`` cannot tell a missing method from a bug
    inside one, so it is left as the fault it reports."""
    with pytest.raises(ContextEngineDisabled) as raised:
        _raise_remote_error(
            {
                "ok": False,
                "error": {
                    "code": "daemon_error",
                    "message": "'NoneType' object has no attribute 'status'",
                },
            },
            surface="agent_context",
            method="status",
        )

    assert not isinstance(raised.value, HostOutdated)


def test_a_private_member_refusal_stays_a_caller_mistake() -> None:
    with pytest.raises(ValueError) as raised:
        _raise_remote_error(
            {
                "ok": False,
                "error": {
                    "code": "validation_error",
                    "message": "invalid RPC member: _x",
                },
            },
            surface="graph_workbench",
            method="_x",
        )

    assert not isinstance(raised.value, HostOutdated)


def test_sent_names_are_keywords_and_fields_not_mapping_keys() -> None:
    names = _sent_names(
        encode((), omit_defaults=True),
        encode(
            {
                "pot_id": "p",
                "filter_": ClaimQueryFilter(pot_id="p", include_retired=True),
                "scope": {"repo": "potpie"},
            },
            omit_defaults=True,
        ),
    )

    assert names == {
        ("", "pot_id"),
        ("", "filter_"),
        ("", "scope"),
        ("ClaimQueryFilter", "pot_id"),
        ("ClaimQueryFilter", "include_retired"),
    }


# --- the explorer keeps what an older host has -------------------------------


@pytest.fixture
def explorer(monkeypatch, tmp_path):
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)

    def build(graph_workbench) -> TestClient:
        pot = SimpleNamespace(pot_id="p", name="p")
        host = SimpleNamespace(
            graph_workbench=graph_workbench,
            pots=SimpleNamespace(list_pots=lambda: [pot], active_pot=lambda: pot),
        )
        app = FastAPI()
        app.state.ui_auth = UiAuth(token=_TOKEN)
        app.include_router(build_ui_api_router(host), prefix="/ui")
        return TestClient(app, headers={"Authorization": f"Bearer {_TOKEN}"})

    yield build
    hosts.reset_for_tests()


def test_mutation_history_drops_an_option_an_older_host_lacks(explorer) -> None:
    calls: list[dict] = []

    def history(**kwargs):
        calls.append(kwargs)
        if "include_claims" in kwargs:
            raise HostOutdated(
                "graph_workbench.history", label=_LABEL, argument="include_claims"
            )
        return {"ok": True, "entries": [], "warnings": []}

    client = explorer(SimpleNamespace(history=history))

    response = client.get(
        "/ui/api/mutation-history", params={"host": "local", "pot": "p"}
    )

    assert response.status_code == 200
    assert calls == [
        {"pot_id": "p", "limit": 50, "include_claims": False},
        {"pot_id": "p", "limit": 50},
    ]


def test_commit_history_on_an_older_host_is_a_structured_501(explorer) -> None:
    def commits(**_):
        raise HostOutdated("graph_workbench.commits", label=_LABEL)

    client = explorer(SimpleNamespace(commits=commits))

    response = client.get("/ui/api/commits", params={"host": "local", "pot": "p"})

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert detail["status"] == "host_outdated"
    assert "older Potpie" in detail["message"]
