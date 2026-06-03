"""In-code daemon config: the manifest the ``DaemonRuntime`` runs.

Unlike a user-facing config file, the daemon manifest (transports/services/components)
is *derived in code* at start time from the home dir + active backend profile — see
``build_daemon_config``. This keeps the daemon dependency-pure (no TOML parser) and
avoids a second config file colliding with ``ConfigService``'s ``config.json``.
"""
from __future__ import annotations
import pathlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ShellSettings:
    data_dir: str
    log_level: str = "info"


@dataclass(frozen=True)
class TransportEntry:
    type: str
    bind: str


@dataclass(frozen=True)
class ReadyProbeEntry:
    kind: str
    target: str
    interval_s: float = 0.5
    timeout_s: float = 30.0


@dataclass(frozen=True)
class ServiceEntry:
    name: str
    backend: str
    config: dict[str, Any]
    ready: ReadyProbeEntry
    endpoint: str
    restart: str = "on_failure"
    depends_on: list[str] = field(default_factory=list)
    data_dir: str | None = None


@dataclass(frozen=True)
class ComponentEntry:
    type: str
    requires_services: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DaemonConfig:
    shell: ShellSettings
    transports: list[TransportEntry]
    services: list[ServiceEntry]
    components: list[ComponentEntry]


def build_daemon_config(
    home: pathlib.Path,
    *,
    log_level: str = "info",
    services: list[ServiceEntry] | None = None,
    components: list[ComponentEntry] | None = None,
) -> DaemonConfig:
    """Derive the daemon manifest from ``home`` (+ optional managed services/components).

    Defaults to a single HTTP transport over a Unix socket at ``<home>/daemon.sock``
    and the ``context_graph`` component (which serves ``context.*`` by delegating to the
    in-process ``HostShell`` — no managed services required for V1).
    """
    home = pathlib.Path(home)
    services = services or []
    if components is None:
        components = [ComponentEntry(type="context_graph", requires_services=[], config={})]
    return DaemonConfig(
        shell=ShellSettings(data_dir=str(home), log_level=log_level),
        transports=[TransportEntry(type="http", bind=f"unix:{home}/daemon.sock")],
        services=services,
        components=components,
    )
