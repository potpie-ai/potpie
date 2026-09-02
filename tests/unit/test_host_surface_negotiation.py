"""The CLI learns what a host serves by asking it, not by reading its prose.

Two repositories keep a copy of the RPC allowlist and nothing negotiates between
them. When the copies drifted, a surface the managed service did not list came
back as ``validation_error: invalid RPC surface: resources`` — *the caller got it
wrong*, exit 1, no repair — and the client's only way to tell that apart from a
genuine bad argument was to match the first three words of someone else's error
message. The managed team can reword that sentence in a refactor and silently
break this client; nobody would find out until a user did.

``GET /surfaces`` (:mod:`potpie.daemon.surfaces`) and
:mod:`potpie.daemon.negotiation` replace the guess with an answer the host gives
about itself. What this module pins:

* a refusal on a surface the host **did not publish** is a capability gap
  whatever words it arrives in — the prose is no longer load-bearing;
* a refusal on a surface the host **did publish** is never retargeted, even when
  it is worded exactly like the legacy refusal — the contract now outranks the
  sentence, which is the half a string match could never do;
* the contract is negotiated **once per endpoint-and-credential**, so this costs
  one round trip per process and re-asks when a key rotates;
* a host that publishes **nothing** — every deployment older than the endpoint,
  including managed today — keeps working exactly as it does now.

The tests drive the real ``DaemonRpcClient`` with ``httpx`` stubbed at the same
seam the rest of the suite uses, so every layer underneath is the shipped one.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from potpie.cli import hosts
from potpie.cli import main as cli_main
from potpie.daemon import negotiation, surfaces
from potpie_context_core.errors import CapabilityNotImplemented
from tests._rpc_fakes import install_rpc_session

runner = CliRunner()

MANAGED_URL = "http://negotiated.test"
MANAGED_TOKEN = "managed-key"  # noqa: S105 - non-secret test fixture

#: A refusal that says nothing this client could pattern-match. The point of the
#: whole exercise: the managed service is free to word its refusals however it
#: likes, and a client that depends on the wording is one refactor from broken.
REWORDED_REFUSAL: dict[str, Any] = {
    "ok": False,
    "error": {
        "code": "validation_error",
        "message": "surface 'resources' is not part of this deployment",
        "detail": None,
        "recommended_next_action": None,
    },
}

#: The legacy wording, for the host that publishes no contract at all.
LEGACY_REFUSAL: dict[str, Any] = {
    "ok": False,
    "error": {
        "code": "validation_error",
        "message": "invalid RPC surface: resources",
        "detail": None,
        "recommended_next_action": None,
    },
}


class _Host:
    """A managed endpoint: what it *publishes*, and what it actually refuses.

    Those are two declarations on purpose — the drift between them is the whole
    subject. ``published`` is what ``GET /surfaces`` answers (``None`` for a host
    that has no such route, which is every deployment today); ``refuses`` is what
    the RPC door turns away, in whatever words ``refusal`` carries.

    Negotiation requests are counted, because "once per connection" is a property
    of the design and not an implementation detail: a client that re-asked on
    every refusal would turn one bad call into a storm of them.
    """

    def __init__(
        self,
        *,
        published: list[str] | None,
        refuses: frozenset[str] = frozenset(),
        refusal: dict[str, Any] | None = None,
        answers: dict[str, dict[str, Any]] | None = None,
        contract: int = 1,
    ) -> None:
        self.published = published
        self.refuses = refuses
        self.refusal = refusal if refusal is not None else REWORDED_REFUSAL
        self.answers = answers or {}
        self.contract = contract
        self.negotiations = 0
        self.tokens: list[str] = []
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("GET", url)
        self.negotiations += 1
        # The inventory of what a host serves rides the same credential as the
        # calls themselves; an unauthenticated probe would be a different
        # question with a different answer.
        self.tokens.append(str(kwargs["headers"]["Authorization"]))
        assert url.endswith("/surfaces"), url
        if self.published is None:
            return httpx.Response(404, request=request)
        return httpx.Response(
            200,
            json={"contract": self.contract, "surfaces": self.published},
            request=request,
        )

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        payload = kwargs.get("json") or {}
        surface = str(payload.get("surface"))
        target = f"{surface}.{payload.get('method') or payload.get('name')}"
        self.calls.append(target)
        request = httpx.Request("POST", url)
        if surface in self.refuses:
            return httpx.Response(200, json=self.refusal, request=request)
        if target in self.answers:
            return httpx.Response(200, json=self.answers[target], request=request)
        # Anything unmapped: a plain validation error, which is what a real host
        # sends for an argument it will not accept.
        return httpx.Response(
            200,
            json={
                "ok": False,
                "error": {
                    "code": "validation_error",
                    "message": f"no answer for {target}",
                },
            },
            request=request,
        )


@pytest.fixture
def managed(monkeypatch, tmp_path):
    """Route a managed host at a fake endpoint, with no state on this machine."""
    hosts.reset_for_tests()
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: (MANAGED_URL, MANAGED_TOKEN))
    yield monkeypatch
    hosts.reset_for_tests()


def _install(monkeypatch, host: _Host) -> Any:
    monkeypatch.setattr(httpx, "get", host.get)
    install_rpc_session(monkeypatch, host.post)
    return hosts.build_managed_host(MANAGED_URL, MANAGED_TOKEN)


def _without(*names: str) -> list[str]:
    return sorted(surfaces.RPC_SURFACES - frozenset(names))


def _ok(result: Any) -> dict[str, Any]:
    from potpie.daemon.rpc import encode

    return {"ok": True, "result": encode(result)}


def _pot(pot_id: str) -> Any:
    from potpie_context_engine.domain.ports.services.pot_management import PotInfo

    return PotInfo(pot_id=pot_id, name="default", active=True)


def _drifted(*names: str, refusal: dict[str, Any] | None = None) -> _Host:
    """A host that publishes its contract and refuses exactly what it omitted."""
    return _Host(published=_without(*names), refuses=frozenset(names), refusal=refusal)


# --- the negotiated contract decides ----------------------------------------


def test_a_refusal_is_a_capability_gap_because_the_host_did_not_list_the_surface(
    managed,
) -> None:
    """The message is unreadable to this client, and it does not need to read it.

    Before negotiation this envelope was a plain ``ValueError`` — exit 1, "the
    caller got it wrong", no next action — for asking a host about a surface it
    chose not to deploy.
    """
    host = _drifted("resources")
    shell = _install(managed, host)

    with pytest.raises(CapabilityNotImplemented) as raised:
        shell.resources.status()

    assert raised.value.capability == "resources"
    assert "--host local" in (raised.value.recommended_next_action or "")
    assert host.negotiations == 1


def test_a_prose_shaped_refusal_on_a_published_surface_stays_a_caller_mistake(
    managed,
) -> None:
    """The half a string match cannot do, and the reason it had to go.

    This host publishes ``graph``, so it does serve it, so this is a validation
    error about the *call* — whatever it happens to say. Matching on the sentence
    alone would report a bad argument as "this host does not implement graph" and
    send the user to another host to run a command that would fail there too.
    """
    host = _Host(
        published=sorted(surfaces.RPC_SURFACES),
        answers={
            "graph.read": {
                "ok": False,
                "error": {
                    "code": "validation_error",
                    "message": "invalid RPC surface: graph is not a view name",
                },
            }
        },
    )
    shell = _install(managed, host)

    with pytest.raises(ValueError) as raised:
        shell.graph.read()

    assert not isinstance(raised.value, CapabilityNotImplemented)


def test_an_unrecognised_envelope_from_an_unserved_surface_is_still_a_gap(
    managed,
) -> None:
    """Not every host refuses in an envelope this client knows how to read.

    Reported as ``daemon_error`` — "the service is broken" — this sends someone
    to investigate a service that is behaving correctly.
    """
    host = _drifted(
        "resources",
        refusal={"ok": False, "error": {"code": "daemon_error", "message": "nope"}},
    )
    shell = _install(managed, host)

    with pytest.raises(CapabilityNotImplemented) as raised:
        shell.resources.list()

    assert raised.value.capability == "resources"


def test_a_served_surface_keeps_its_own_domain_errors(managed) -> None:
    """The control: negotiation must not swallow the errors a host really means."""
    host = _Host(
        published=sorted(surfaces.RPC_SURFACES),
        answers={
            "pots.active_pot": {
                "ok": False,
                "error": {"code": "pot_not_found", "message": "No pot matching 'x'."},
            }
        },
    )
    shell = _install(managed, host)

    from potpie_context_core.errors import PotNotFound

    with pytest.raises(PotNotFound):
        shell.pots.active_pot()


# --- once per connection -----------------------------------------------------


def test_the_contract_is_negotiated_once_and_reused(managed) -> None:
    """One round trip for the process, not one per refused call."""
    host = _drifted("resources")
    shell = _install(managed, host)

    for _ in range(3):
        with pytest.raises(CapabilityNotImplemented):
            shell.resources.status()

    assert host.negotiations == 1
    assert len(host.calls) == 3


def test_a_healthy_call_never_asks_the_host_what_it_serves(managed) -> None:
    """The cost of this has to be nothing when nothing is wrong.

    The question is only worth asking when the answer changes what the caller is
    told, which is never on a call that succeeded.
    """
    host = _Host(
        published=sorted(surfaces.RPC_SURFACES),
        answers={"graph.catalog": _ok("falkordb")},
    )
    shell = _install(managed, host)

    assert shell.graph.catalog() == "falkordb"
    assert host.negotiations == 0


def test_a_rotated_credential_is_negotiated_again(managed) -> None:
    """The cache is keyed by the key: a 401 is one of the answers it stores.

    A contract cached against the address alone would keep the refusal recorded
    for a token that has since been replaced.
    """
    host = _drifted("resources")
    _install(managed, host)

    negotiation.negotiate(MANAGED_URL, MANAGED_TOKEN)
    negotiation.negotiate(MANAGED_URL, MANAGED_TOKEN)
    assert host.negotiations == 1

    negotiation.negotiate(MANAGED_URL, "rotated-key")
    assert host.negotiations == 2
    assert host.tokens == [f"Bearer {MANAGED_TOKEN}", "Bearer rotated-key"]


def test_doctor_and_the_rpc_client_read_one_negotiated_answer(managed) -> None:
    """Two caches would let the report and the error disagree about one host."""
    host = _drifted("resources")
    shell = _install(managed, host)

    assert hosts.advertised_surfaces(shell) == frozenset(host.published or [])
    with pytest.raises(CapabilityNotImplemented):
        shell.resources.status()

    assert host.negotiations == 1


# --- a host that publishes nothing -------------------------------------------


def test_a_silent_host_keeps_todays_behaviour(managed) -> None:
    """Every deployment older than the endpoint, which today includes managed.

    404 means "this host does not say", never "this host serves nothing": the
    call goes out, the refusal is read the way it always was, and the CLI is
    never stricter than the host it is talking to.
    """
    host = _Host(
        published=None, refuses=frozenset({"resources"}), refusal=LEGACY_REFUSAL
    )
    shell = _install(managed, host)

    with pytest.raises(CapabilityNotImplemented) as raised:
        shell.resources.status()

    assert raised.value.capability == "resources"
    assert "--host local" in (raised.value.recommended_next_action or "")


def test_a_silent_hosts_ordinary_validation_error_is_still_one(managed) -> None:
    host = _Host(
        published=None,
        answers={
            "graph.read": {
                "ok": False,
                "error": {
                    "code": "validation_error",
                    "message": "invalid slug 'a b'",
                },
            }
        },
    )
    shell = _install(managed, host)

    with pytest.raises(ValueError) as raised:
        shell.graph.read()

    assert not isinstance(raised.value, CapabilityNotImplemented)


def test_negotiation_never_raises_out_of_the_path_it_is_explaining(managed) -> None:
    """A probe that failed must not replace the error the caller is holding."""
    host = _Host(
        published=None, refuses=frozenset({"resources"}), refusal=LEGACY_REFUSAL
    )
    shell = _install(managed, host)

    def _explode(url: str, **_: object) -> httpx.Response:
        raise httpx.ConnectError("nope", request=httpx.Request("GET", url))

    managed.setattr(httpx, "get", _explode)

    with pytest.raises(CapabilityNotImplemented):
        shell.resources.status()


@pytest.mark.parametrize(
    "body",
    [
        {"contract": 2},
        {"contract": 1, "surfaces": "resources"},
        {"contract": 1, "surfaces": ["resources", 7]},
        ["resources"],
    ],
    ids=["no-list", "not-a-list", "not-all-strings", "not-an-object"],
)
def test_an_answer_this_build_cannot_read_is_unknown_not_empty(
    managed, body: Any
) -> None:
    """A future dialect must degrade to "does not say", never to "serves nothing".

    Read as an empty set, one unrecognised body would have this client refuse
    every command against a host that implements all of them.
    """
    managed.setattr(
        httpx,
        "get",
        lambda url, **_: httpx.Response(
            200, json=body, request=httpx.Request("GET", url)
        ),
    )

    assert negotiation.negotiate(MANAGED_URL, MANAGED_TOKEN).surfaces is None


# --- through the CLI ---------------------------------------------------------


def test_a_resource_command_degrades_on_a_host_that_reworded_its_refusal(
    managed,
) -> None:
    """End to end, and the user-visible point of all of it.

    ``resource --help`` is host-independent, so a refusal is the only place a
    user learns the feature is not deployed here. It must be exit 2 with a
    repair — and it must stay that way when the host stops saying "invalid RPC
    surface".
    """
    host = _drifted("resources")
    host.answers["pots.active_pot"] = _ok(_pot("pot_managed_1"))
    _install(managed, host)

    result = runner.invoke(
        cli_main.app,
        ["--json", "--host", "managed", "resource", "list", "--doc", "nope"],
    )

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "not_implemented"
    assert "--host local" in (payload["recommended_next_action"] or "")


def test_doctor_degrades_per_section_against_a_reworded_host(managed) -> None:
    """``doctor`` still produces a report, and still names what is missing."""
    _install(managed, _drifted("resources", "ledger"))

    result = runner.invoke(cli_main.app, ["--json", "--host", "managed", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    for name in ("resources", "resource_index", "ledger"):
        assert name in payload["degraded_sections"]
        assert payload["sections"][name]["status"] == "unavailable"
    # Reported from the host's own answer rather than inferred from failures.
    assert payload["host"]["missing_surfaces"] == ["ledger", "resources"]
    # And the blocks that never needed the host survive.
    assert payload["daemon"]["mode"]
    assert payload["cli_install"]["package_name"]
