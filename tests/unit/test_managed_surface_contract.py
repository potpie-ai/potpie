"""What a host owes the CLI, and what the CLI does when a host does not pay.

The daemon's RPC allowlist is maintained twice — this repo and the managed
service's own build — and nothing compared the copies. They drifted: the managed
copy has no ``resources``, so every ``potpie resource ...`` command against it
failed with ``validation_error: invalid RPC surface: resources`` (exit 1, "the
caller got it wrong", no next action) and ``potpie doctor``, which asks for
resource status unconditionally, produced nothing at all.

Three properties are pinned here, in the layer each one actually lives in:

* the allowlist is a declaration something can walk, so a surface added to the
  host facade without an entry fails here rather than at runtime on a machine we
  do not own;
* a host refusing a surface answers ``not_implemented`` — including when it is
  an older host that still says ``invalid RPC surface``, which the client
  translates, because we cannot deploy into that repository;
* ``doctor`` reports a gap per section and exits 0 with a report, and never
  loses the local blocks (daemon, install, host label) to a remote that did not
  answer.

The doctor tests deliberately drive the *real* ``DaemonRpcClient`` over a
monkeypatched ``httpx``, replaying envelopes recorded verbatim from the live
managed service. A ``MagicMock`` host is what let the previously-unwrapped calls
through the existing regression test: a mock never refuses anything.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from potpie.cli import hosts
from potpie.cli import main as cli_main
from potpie.daemon import client as daemon_client
from potpie.daemon import main as daemon_main
from potpie.daemon import surfaces
from potpie.daemon.client import RemoteHostShell, RemoteSurface
from potpie.daemon.rpc import encode
from potpie_context_core.errors import CapabilityNotImplemented
from potpie_context_core.ports.graph.backend import BackendCapabilities
from potpie_context_core.ports.graph.mutation import BackendReadiness
from potpie_context_core.ports.resource_store import ResourceStoreStatus

runner = CliRunner()

TOKEN = "daemon-token-for-tests"  # noqa: S105 - non-secret test fixture
BEARER = {"Authorization": f"Bearer {TOKEN}"}

MANAGED_URL = "http://managed.test"
MANAGED_TOKEN = "managed-key"  # noqa: S105 - non-secret test fixture

#: The refusal the live managed service sends for a surface its copy of the
#: allowlist does not list. Pasted from the CLI test report rather than
#: reconstructed, because the exact bytes are the contract this defends against.
LEGACY_SURFACE_REFUSAL = {
    "ok": False,
    "error": {
        "code": "validation_error",
        "message": "invalid RPC surface: resources",
        "detail": None,
        "recommended_next_action": None,
    },
}

#: The managed service's answer for a capability it does implement but has not
#: deployed. Also verbatim.
LEDGER_NOT_IMPLEMENTED = {
    "ok": False,
    "error": {
        "code": "not_implemented",
        "capability": "ledger.status",
        "message": (
            "Capability not implemented: ledger.status — managed ledger "
            "connectors are not deployed"
        ),
        "detail": "managed ledger connectors are not deployed",
        "recommended_next_action": None,
    },
}


# --- the declaration ---------------------------------------------------------


def test_every_surface_the_cli_calls_is_declared_by_the_daemon() -> None:
    """The intra-repo half of the drift guard.

    ``RemoteHostShell`` is the list of things the CLI can address; the allowlist
    is the list of things the daemon will route. A surface added to the first
    and not the second is refused at runtime, on someone's machine, in the
    vocabulary of a caller mistake. It should fail here instead.
    """
    shell = RemoteHostShell(rpc=SimpleNamespace(daemon=None))  # type: ignore[arg-type]
    addressed = {
        value._path
        for value in vars(shell).values()
        if isinstance(value, RemoteSurface)
    }
    addressed |= {f"backend.{name}" for name in RemoteSurface._NESTED}

    undeclared = sorted(addressed - surfaces.RPC_SURFACES)
    assert not undeclared, (
        f"the CLI addresses {undeclared} but the daemon will refuse them; add "
        "them to potpie.daemon.surfaces.RPC_SURFACES"
    )


def test_every_host_shell_field_is_declared_or_explicitly_denied() -> None:
    """A new host service must be a decision, not an omission.

    ``daemon`` and ``profile`` are withheld on purpose (process lifecycle the
    CLI drives locally, and a plain ``str``). Pinning them as *denied* rather
    than merely absent stops the next reader from "fixing" the gap by widening
    the allowlist.
    """
    undeclared = surfaces.undeclared_host_surfaces()

    assert undeclared == (), (
        f"HostShell field(s) {list(undeclared)} are neither in RPC_SURFACES nor "
        "in DENIED_SURFACES — decide which, and say why in DENIED_SURFACES"
    )


def test_the_denied_surfaces_stay_out_of_the_allowlist() -> None:
    assert not (surfaces.DENIED_SURFACES & surfaces.RPC_SURFACES)


# --- what the daemon puts on the wire ---------------------------------------


class _StubHost:
    """Enough of a host that ``create_app`` can be built without the engine."""

    profile = "in_memory"

    def __init__(self) -> None:
        self.backend = self


@pytest.fixture
def app(monkeypatch, tmp_path):
    # No `with TestClient(...)`: the lifespan writes this machine's real
    # pid/discovery files.
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(daemon_main, "build_host_shell", _StubHost)
    yield daemon_main.create_app(
        token=TOKEN,
        base_url="http://127.0.0.1:1",
        pid=123,
        log_file="/tmp/potpie-surface-test.log",  # noqa: S108 - fixed test path
    )
    hosts.reset_for_tests()


def test_the_daemon_advertises_its_surfaces_only_to_an_authorized_caller(app) -> None:
    """The answer to "what do you implement" is behind the daemon token.

    It is trivially safe content, but it sits on the same app as an
    unauthenticated ``/health``, and an inventory of what a process serves is
    not something to hand to anything that can reach the port.
    """
    authorized = TestClient(app, headers=BEARER).get("/surfaces")

    assert authorized.status_code == 200, authorized.text
    body = authorized.json()
    assert set(body["surfaces"]) == set(surfaces.RPC_SURFACES)
    assert body["contract"] == surfaces.SURFACE_CONTRACT_VERSION

    anonymous = TestClient(app)
    assert anonymous.get("/surfaces").status_code == 401
    assert (
        anonymous.get(
            "/surfaces", headers={"Authorization": "Bearer not-the-token"}
        ).status_code
        == 401
    )


def test_an_undeclared_surface_is_answered_as_a_capability_gap(app) -> None:
    """The envelope on the wire — which is the artifact the second repo copies.

    ``validation_error`` means *the caller got it wrong*; a client asking for a
    surface this host chose not to serve got it exactly right and has nothing to
    correct. The companion assertion is the control: a bad *member* really is a
    caller mistake and must stay one.
    """
    authorized = TestClient(app, headers=BEARER)

    refused = authorized.post(
        "/rpc", json={"surface": "resources_typo", "method": "status"}
    ).json()

    assert refused["ok"] is False
    assert refused["error"]["code"] == "not_implemented"
    assert refused["error"]["capability"] == "resources_typo"
    assert refused["error"]["recommended_next_action"]

    bad_member = authorized.post(
        "/rpc", json={"surface": "backend", "method": "_profile"}
    ).json()

    assert bad_member["error"]["code"] == "validation_error"


def test_a_legacy_invalid_surface_answer_becomes_a_capability_gap() -> None:
    """We do not control the managed repo, so the fix above ships nothing today.

    The input is the envelope that service actually sends, byte for byte.
    """
    with pytest.raises(CapabilityNotImplemented) as raised:
        daemon_client._raise_remote_error(
            LEGACY_SURFACE_REFUSAL,
            status_code=200,
            endpoint="http://svc/rpc",
            label="Managed (svc)",
        )

    assert raised.value.capability == "resources"
    assert "Managed (svc)" in (raised.value.detail or "")
    assert "--host local" in (raised.value.recommended_next_action or "")


def test_an_ordinary_validation_error_is_still_a_validation_error() -> None:
    """The control on the branch above: only the surface refusal is retargeted."""
    with pytest.raises(ValueError) as raised:
        daemon_client._raise_remote_error(
            {"error": {"code": "validation_error", "message": "invalid slug 'a b'"}}
        )

    assert not isinstance(raised.value, CapabilityNotImplemented)


# --- what a client makes of a host that does not answer ----------------------


def _managed_host() -> Any:
    return hosts.build_managed_host(MANAGED_URL, MANAGED_TOKEN)


def test_a_host_that_publishes_nothing_reads_as_unknown_not_empty(monkeypatch) -> None:
    """Silence is not an empty set.

    The managed service predates ``/surfaces`` and answers 404. A client that
    read that as "implements nothing" would refuse every command against a host
    that in fact implements everything.
    """
    hosts.reset_for_tests()
    host = _managed_host()

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **_: httpx.Response(404, request=httpx.Request("GET", url)),
    )
    assert hosts.advertised_surfaces(host) is None

    hosts.reset_for_tests()

    def _unreachable(url: str, **_: object) -> httpx.Response:
        raise httpx.ConnectError("nope", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", _unreachable)
    assert hosts.advertised_surfaces(host) is None
    hosts.reset_for_tests()


def test_a_host_that_publishes_its_surfaces_is_read_verbatim(monkeypatch) -> None:
    hosts.reset_for_tests()
    published = sorted(surfaces.RPC_SURFACES - {"resources"})

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **_: httpx.Response(
            200,
            json={"contract": 1, "surfaces": published},
            request=httpx.Request("GET", url),
        ),
    )

    assert hosts.advertised_surfaces(_managed_host()) == frozenset(published)
    hosts.reset_for_tests()


def test_a_host_with_no_transport_is_never_probed(monkeypatch) -> None:
    """An in-process host (and every ``MagicMock`` in the suite) has no endpoint.

    Formatting a mock's ``__getitem__`` into a URL would send a diagnostic off
    to open a socket against nonsense, which is why the discovery record is
    type-checked rather than duck-typed.
    """
    from unittest.mock import MagicMock

    hosts.reset_for_tests()

    def _never(*_args: object, **_kwargs: object) -> httpx.Response:
        raise AssertionError("a host without a real endpoint must not be probed")

    monkeypatch.setattr(httpx, "get", _never)

    assert hosts.advertised_surfaces(MagicMock()) is None
    assert hosts.advertised_surfaces(object()) is None
    hosts.reset_for_tests()


# --- doctor, over the real client -------------------------------------------


class _Dispatcher:
    """A managed host that answers per ``{surface, method}``, refusing the rest.

    Sits at ``httpx.post`` — the same seam ``tests/unit/test_daemon_rpc.py``
    uses — so every layer under it is the shipped one: the RPC envelope, the
    error decode, the host facade, the command.
    """

    def __init__(
        self,
        answers: dict[str, dict[str, Any]],
        *,
        refused: frozenset[str],
    ) -> None:
        self.answers = answers
        self.refused = refused
        self.seen: list[str] = []

    def __call__(self, url: str, **kwargs: Any) -> httpx.Response:
        payload = kwargs.get("json") or {}
        surface = str(payload.get("surface"))
        target = f"{surface}.{payload.get('method') or payload.get('name')}"
        self.seen.append(target)
        request = httpx.Request("POST", url)
        if surface in self.refused:
            body = {
                "ok": False,
                "error": {
                    "code": "validation_error",
                    "message": f"invalid RPC surface: {surface}",
                    "detail": None,
                    "recommended_next_action": None,
                },
            }
            return httpx.Response(200, json=body, request=request)
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


def _pot(pot_id: str) -> Any:
    from potpie_context_engine.domain.ports.services.pot_management import PotInfo

    return PotInfo(pot_id=pot_id, name="default", active=True)


def _ok(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": encode(result)}


HEALTHY_ANSWERS: dict[str, dict[str, Any]] = {
    "backend.capabilities": _ok(
        BackendCapabilities(profile="falkordb", mutation=True, claim_query=True)
    ),
    "backend.profile": _ok("falkordb"),
    "backend.mutation.readiness": _ok(
        BackendReadiness(
            profile="falkordb", ready=True, capability_ready={"mutation": True}
        )
    ),
    "pots.active_pot": _ok(_pot("pot_managed_1")),
    "pots.list_repo_sources": _ok([]),
    "ledger.status": LEDGER_NOT_IMPLEMENTED,
}


@pytest.fixture
def managed_cli(monkeypatch, tmp_path):
    """Route ``--host managed`` at a fake endpoint, with no local state at all.

    ``CONTEXT_ENGINE_HOME`` is moved as well as the registry: ``doctor`` asks
    the *local* daemon for its process status even when the active host is
    remote, and that must land in a temp home rather than on this machine's
    real one.
    """
    hosts.reset_for_tests()
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: (MANAGED_URL, MANAGED_TOKEN))
    # No `/surfaces` on the fake host unless a test says otherwise: that is the
    # deployed managed service's behaviour today.
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **_: httpx.Response(404, request=httpx.Request("GET", url)),
    )
    yield monkeypatch
    hosts.reset_for_tests()


def _run_doctor(monkeypatch, dispatcher: _Dispatcher) -> dict[str, Any]:
    monkeypatch.setattr(httpx, "post", dispatcher)
    result = runner.invoke(cli_main.app, ["--json", "--host", "managed", "doctor"])
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_doctor_reports_a_gap_per_section_instead_of_dying_on_one(
    managed_cli,
) -> None:
    """The recorded managed shape: no ``resources`` surface, no deployed ledger."""
    payload = _run_doctor(
        managed_cli,
        _Dispatcher(HEALTHY_ANSWERS, refused=frozenset({"resources"})),
    )

    assert payload["ok"] is False
    assert payload["degraded_sections"] == ["ledger", "resource_index", "resources"]
    assert payload["sections"]["resources"]["status"] == "unavailable"
    assert payload["sections"]["daemon"]["status"] == "ok"
    assert payload["sections"]["backend_capabilities"]["status"] == "ok"
    # The refusal reaches the report as a capability gap, in the host's terms —
    # not as "invalid RPC surface", which names a routing concept the reader has
    # no way to act on.
    assert "RPC surface" not in payload["sections"]["resources"]["detail"]
    assert "resources" in payload["sections"]["resources"]["detail"]
    # A degraded block keeps its whole key set, so an agent reading a field gets
    # None rather than a KeyError on precisely the hosts this report is for.
    assert payload["resources"]["documents"] is None
    assert payload["resources"]["kind"] is None
    assert payload["resource_index"]["chunks"] is None
    # And everything healthy still answered.
    assert payload["backend_capabilities"]
    assert payload["backend_ready"] is True
    assert payload["active_pot"] == "pot_managed_1"


def test_doctor_survives_a_host_that_refuses_every_surface(managed_cli) -> None:
    """The regression for the calls the previous round left bare.

    Before this, a host that does not serve ``backend`` took the whole command
    down with ``{"code": "validation_error", "message": "invalid RPC surface:
    backend"}`` at exit 1 — losing the daemon status, the CLI install block and
    the host label, none of which needed the remote host at all.
    """
    payload = _run_doctor(
        managed_cli,
        _Dispatcher(
            {},
            refused=frozenset(
                {"backend", "pots", "resources", "ledger", "agent_context"}
            ),
        ),
    )

    assert payload["ok"] is False
    for name in (
        "backend_capabilities",
        "backend_profile",
        "active_pot",
        "backend_readiness",
        "ledger",
        "resources",
        "resource_index",
    ):
        assert payload["sections"][name]["status"] == "unavailable", name
    # The blocks that never needed a host survive.
    assert payload["daemon"]["mode"]
    assert payload["cli_install"]["package_name"]
    assert payload["host"]["label"]
    assert payload["backend_capabilities"] == []
    # Never a pointer back at the command the operator is already running.
    assert "potpie doctor" not in (payload["recommended_next_action"] or "")


def test_doctor_still_answers_when_the_local_daemon_is_not_running(
    monkeypatch, tmp_path
) -> None:
    """The common case, and the sharpest one: ``doctor`` told you to run doctor.

    With no daemon the first probe raised ``ContextEngineDisabled`` straight out
    of the command, so the error boundary rendered exit 2 and its generic repair
    — "check backend/daemon readiness with 'potpie doctor'" — for someone who
    had just run ``potpie doctor``.
    """
    hosts.reset_for_tests()
    # The suite pins every CLI unit test to an in-process host; this one is
    # *about* the daemon transport, so it opts back into it.
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "daemon")
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_endpoint", lambda: None)

    def _refuse(url: str, **_: object) -> httpx.Response:
        raise httpx.ConnectError(
            "connection refused", request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", _refuse)

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["sections"]["backend_capabilities"]["status"] == "unavailable"
    assert payload["daemon"]["up"] is False
    assert payload["cli_install"]["package_name"]
    assert "potpie doctor" not in (payload["recommended_next_action"] or "")
    hosts.reset_for_tests()


def test_doctor_reports_missing_surfaces_only_when_the_host_says(managed_cli) -> None:
    """``missing_surfaces`` is a subtraction, and it is only valid against an
    answer. Computed from silence it would accuse every deployed host of serving
    nothing."""
    silent = _run_doctor(
        managed_cli, _Dispatcher(HEALTHY_ANSWERS, refused=frozenset({"resources"}))
    )

    assert silent["host"]["advertised_surfaces"] is None
    assert silent["host"]["missing_surfaces"] is None

    hosts.reset_for_tests()
    managed_cli.setattr(
        httpx,
        "get",
        lambda url, **_: httpx.Response(
            200,
            json={
                "contract": 1,
                "surfaces": sorted(surfaces.RPC_SURFACES - {"resources"}),
            },
            request=httpx.Request("GET", url),
        ),
    )

    speaking = _run_doctor(
        managed_cli, _Dispatcher(HEALTHY_ANSWERS, refused=frozenset({"resources"}))
    )

    assert speaking["host"]["missing_surfaces"] == ["resources"]
    assert "resources" not in speaking["host"]["advertised_surfaces"]


def test_doctor_human_output_names_the_degraded_sections(managed_cli) -> None:
    """Human mode is never a subset of ``--json``: an operator who did not pass
    ``--json`` must still learn that two sections are missing."""
    managed_cli.setattr(
        httpx, "post", _Dispatcher(HEALTHY_ANSWERS, refused=frozenset({"resources"}))
    )

    result = runner.invoke(cli_main.app, ["--host", "managed", "doctor"])

    assert result.exit_code == 0, result.stdout
    flat = " ".join(result.stdout.split())
    assert "degraded:" in flat
    assert "resources" in flat
    # The local blocks are printed whether or not the remote answered.
    assert "daemon:" in flat


def test_doctor_stays_green_when_every_section_answers(managed_cli) -> None:
    """The control: ``ok`` must mean something, so it has to be reachable."""
    answers = dict(HEALTHY_ANSWERS)
    answers["ledger.status"] = _ok_ledger()
    answers["resources.status"] = _ok(
        ResourceStoreStatus(kind="local", ready=True, location="/data", documents=2)
    )
    answers["resources.index_status"] = _ok(_index_status())

    payload = _run_doctor(managed_cli, _Dispatcher(answers, refused=frozenset()))

    assert payload["degraded_sections"] == []
    assert payload["ok"] is True
    assert payload["resources"]["documents"] == 2
    assert payload["recommended_next_action"] is None


def _ok_ledger() -> dict[str, Any]:
    from potpie_context_engine.domain.ports.ledger.client import LedgerHealth

    return _ok(LedgerHealth(available=True, binding="none"))


def _index_status() -> Any:
    from potpie_context_core.ports.resource_index import ResourceIndexStatus

    return ResourceIndexStatus(profile="lexical", ready=True)


# --- the resource group ------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["resource", "list", "--doc", "nope"],
        ["resource", "get", "potpie://res/nope/s/1"],
        ["resource", "rm", "nope", "--confirm"],
    ],
    ids=["list", "get", "rm"],
)
def test_resource_commands_refuse_with_a_capability_answer(
    managed_cli, argv: list[str]
) -> None:
    """A host without the surface is not a caller mistake.

    All four verbs are advertised identically on every host — ``resource --help``
    is host-independent — so this refusal is the only place a user can learn the
    feature is not there. Today it said "invalid RPC surface: resources" at exit
    1 with no next action.
    """
    managed_cli.setattr(
        httpx, "post", _Dispatcher(HEALTHY_ANSWERS, refused=frozenset({"resources"}))
    )

    result = runner.invoke(cli_main.app, ["--json", "--host", "managed", *argv])

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "not_implemented"
    assert "RPC surface" not in payload["message"]
    assert "--host local" in (payload["recommended_next_action"] or "")


def test_an_index_capability_gap_keeps_its_own_repair(capsys, monkeypatch) -> None:
    """The control on the rewording above.

    The engine's own gaps are named ``resource_index.<profile>....`` and mean
    something far more specific — switch the index profile and rebuild. Matching
    on a loose ``resources`` prefix would replace that repair with "use the local
    host", which is not the fix and would send the reader nowhere.
    """
    import typer

    from potpie.cli.commands import _common
    from potpie.cli.commands.resource import _resource_contract

    monkeypatch.setattr(_common, "_state", dict(_common._state, json=True))

    with pytest.raises(typer.Exit):
        with _resource_contract():
            raise CapabilityNotImplemented(
                "resource_index.none.write.index_document",
                detail="the 'none' resource index has not implemented it yet",
                recommended_next_action="Set CONTEXT_ENGINE_RESOURCE_INDEX to sqlite_fts",
            )

    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "not_implemented"
    assert payload["recommended_next_action"] == (
        "Set CONTEXT_ENGINE_RESOURCE_INDEX to sqlite_fts"
    )
