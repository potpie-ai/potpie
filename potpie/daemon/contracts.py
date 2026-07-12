"""Root-owned daemon process and discovery contracts."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


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


__all__ = [
    "DaemonDiscovery",
    "DaemonHealth",
    "DaemonInstallResult",
    "DaemonRestartResult",
    "DaemonStartResult",
    "DaemonStatus",
    "DaemonStopResult",
]
