"""Local daemon lifecycle port used by setup and host status surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict

from domain.lifecycle import SetupPlan, StepResult


class DaemonDiscovery(TypedDict, total=False):
    transport: str
    base_url: str
    token: str
    pid: int
    log_file: str
    backend: str


class DaemonStatus(TypedDict):
    up: bool
    mode: Literal["in_process", "detached"]
    home: str
    detail: str
    pid: NotRequired[int | None]
    url: NotRequired[str]
    backend: NotRequired[str]


class DaemonHealth(TypedDict, total=False):
    live: bool
    ok: bool
    mode: Literal["in_process", "detached", "daemon"]
    pid: int
    backend: str


class DaemonInstallResult(TypedDict):
    installed: bool
    detail: str


class DaemonStartResult(TypedDict):
    pid: int
    url: str
    backend: NotRequired[str]
    log_file: NotRequired[str]


class DaemonStopResult(TypedDict):
    detail: str


class DaemonRestartResult(DaemonStartResult, total=False):
    started: DaemonStartResult


class DaemonLifecyclePort(Protocol):
    home: Path
    in_process: bool

    def discovery(self) -> DaemonDiscovery | None: ...
    def status(self) -> DaemonStatus: ...
    def health(self) -> DaemonHealth: ...
    def logs(self) -> list[str]: ...
    def ensure(self, plan: SetupPlan | None = None) -> StepResult: ...
    def install(self) -> DaemonInstallResult: ...
    def start(self, *, backend: str | None = None) -> DaemonStartResult: ...
    def stop(self) -> DaemonStopResult: ...
    def restart(self) -> DaemonRestartResult: ...


__all__ = [
    "DaemonDiscovery",
    "DaemonHealth",
    "DaemonInstallResult",
    "DaemonLifecyclePort",
    "DaemonRestartResult",
    "DaemonStartResult",
    "DaemonStatus",
    "DaemonStopResult",
]
