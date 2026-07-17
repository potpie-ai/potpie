"""build_daemon_config derives the in-code daemon manifest (no TOML file)."""

from __future__ import annotations

import pathlib

from potpie_context_engine.host.daemon_runtime.config import (
    ComponentEntry,
    DaemonConfig,
    ReadyProbeEntry,
    ServiceEntry,
    ShellSettings,
    TransportEntry,
    build_daemon_config,
)


def test_defaults_one_uds_transport_and_context_component(tmp_path: pathlib.Path):
    cfg = build_daemon_config(tmp_path)
    assert isinstance(cfg, DaemonConfig)
    assert cfg.shell.data_dir == str(tmp_path)
    assert [(t.type, t.bind) for t in cfg.transports] == [
        ("http", f"unix:{tmp_path}/daemon.sock")
    ]
    assert cfg.services == []
    assert [c.type for c in cfg.components] == ["context_graph"]
    assert cfg.components[0].requires_services == []


def test_custom_services_and_components_pass_through(tmp_path: pathlib.Path):
    svc = ServiceEntry(
        name="graph-db",
        backend="external",
        config={},
        ready=ReadyProbeEntry(kind="tcp", target="127.0.0.1:7687"),
        endpoint="bolt://127.0.0.1:7687",
    )
    comp = ComponentEntry(
        type="context_graph", requires_services=["graph-db"], config={"k": "v"}
    )
    cfg = build_daemon_config(
        tmp_path, services=[svc], components=[comp], log_level="debug"
    )
    assert cfg.shell == ShellSettings(data_dir=str(tmp_path), log_level="debug")
    assert cfg.services[0].name == "graph-db"
    assert cfg.components[0].requires_services == ["graph-db"]


def test_transport_entry_and_settings_are_frozen():
    t = TransportEntry(type="http", bind="unix:/x/daemon.sock")
    assert t.type == "http"
