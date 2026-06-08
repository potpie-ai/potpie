"""Managed-service value objects: how the daemon describes a service it supervises."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class ReadyProbe:
    kind: str  # "tcp" | "http" | "cmd"
    target: str  # "host:port" | url | shell argv (json-encoded)
    interval_s: float = 0.5
    timeout_s: float = 30.0


class RestartPolicy(Enum):
    NO = "no"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"


@dataclass
class ServiceSpec:
    name: str
    backend: str  # registry key
    config: dict[str, Any]  # backend-specific
    ready: ReadyProbe
    endpoint: str  # connection string consumers receive
    restart: RestartPolicy = RestartPolicy.ON_FAILURE
    depends_on: list[str] = field(default_factory=list)
    data_dir: str | None = None  # optional per-service writable dir
