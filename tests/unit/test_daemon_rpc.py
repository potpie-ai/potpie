from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from domain.lifecycle import SetupPlan
from host.shell import HostShell
from potpie.daemon import main as daemon_main
from potpie.daemon.client import RemoteSurface
from potpie.daemon.rpc import (
    RPC_SURFACES,
    TYPE_KEY,
    decode,
    encode,
    validate_rpc_attr,
    validate_rpc_method,
)


def test_daemon_rpc_roundtrips_domain_dataclasses() -> None:
    plan = SetupPlan(
        backend="embedded",
        repo="potpie",
        pot="default",
        agent="claude",
        assume_yes=True,
    )

    assert decode(encode(plan)) == plan


def test_daemon_rpc_rejects_non_domain_class_references() -> None:
    with pytest.raises(TypeError, match="RPC class not allowed"):
        decode(
            {
                TYPE_KEY: "dataclass",
                "class": "os:stat_result",
                "value": {},
            }
        )


def test_daemon_rpc_rejects_unregistered_dataclasses() -> None:
    @dataclass(frozen=True)
    class PrivatePayload:
        value: str

    with pytest.raises(TypeError, match="RPC class not allowed"):
        encode(PrivatePayload(value="secret"))


def test_daemon_rpc_rejects_unregistered_domain_references() -> None:
    with pytest.raises(TypeError, match="RPC class not allowed"):
        decode(
            {
                TYPE_KEY: "dataclass",
                "class": "domain.errors:CapabilityNotImplemented",
                "value": {},
            }
        )


def test_daemon_rpc_contract_allows_known_methods_and_attrs() -> None:
    validate_rpc_method("graph", "read")
    validate_rpc_method("backend.mutation", "readiness")
    validate_rpc_attr("backend", "profile")


def test_daemon_rpc_top_level_contract_tracks_host_shell_surfaces() -> None:
    expected = {field.name for field in fields(HostShell)} - {"daemon", "profile"}
    actual = {surface for surface in RPC_SURFACES if "." not in surface}

    assert actual == expected


def test_daemon_rpc_contract_rejects_unknown_members() -> None:
    with pytest.raises(ValueError, match="invalid RPC method: graph.delete_all"):
        validate_rpc_method("graph", "delete_all")
    with pytest.raises(ValueError, match="invalid RPC attribute: graph.profile"):
        validate_rpc_attr("graph", "profile")
    with pytest.raises(ValueError, match="invalid RPC surface: graph.private"):
        validate_rpc_method("graph.private", "read")


def test_remote_surface_only_exposes_registered_members() -> None:
    class _Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, tuple, dict]] = []

        def call(self, surface: str, method: str, *args, **kwargs):
            self.calls.append((surface, method, args, kwargs))
            return "called"

        def attr(self, surface: str, name: str):
            return f"{surface}.{name}"

    client = _Client()
    backend = RemoteSurface(client, "backend")

    assert backend.profile == "backend.profile"
    assert backend.mutation.readiness("pot-1") == "called"
    assert client.calls == [
        ("backend.mutation", "readiness", ("pot-1",), {}),
    ]
    with pytest.raises(AttributeError, match="has no RPC member"):
        backend.delete_all()


def test_daemon_http_rpc_rejects_unregistered_method(monkeypatch, tmp_path) -> None:
    host = SimpleNamespace(
        backend=SimpleNamespace(profile="test"),
        graph=SimpleNamespace(delete_all=lambda: "should not run"),
    )
    monkeypatch.setattr(daemon_main, "build_potpie_host_shell", lambda: host)
    monkeypatch.setattr(daemon_main, "default_home", lambda: tmp_path)

    app = daemon_main.create_app(
        token="secret",
        base_url="http://127.0.0.1:1",
        pid=123,
        log_file=str(tmp_path / "daemon.log"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/rpc",
            headers={"Authorization": "Bearer secret"},
            json={"surface": "graph", "method": "delete_all"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "validation_error",
            "message": "invalid RPC method: graph.delete_all",
        },
    }
