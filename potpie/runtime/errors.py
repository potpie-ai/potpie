"""Stable root-runtime errors surfaced by CLI and MCP boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class RuntimeBoundaryError(RuntimeError):
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    retryable: bool = False
    recommended_command: str | None = None

    def __str__(self) -> str:
        return self.message


class RuntimeDaemonUnavailable(RuntimeBoundaryError):
    def __init__(self, message: str = "The Potpie daemon is not reachable.") -> None:
        super().__init__(
            code="RUNTIME_DAEMON_UNAVAILABLE",
            message=message,
            retryable=True,
            recommended_command="potpie daemon start",
        )


class RpcProtocolMismatch(RuntimeBoundaryError):
    def __init__(self, *, actual: str | None) -> None:
        super().__init__(
            code="RPC_PROTOCOL_MISMATCH",
            message="The running Potpie daemon uses an incompatible RPC protocol.",
            details={"expected": "1", "actual": actual},
            retryable=True,
            recommended_command="potpie daemon restart",
        )


class DaemonRpcFailure(RuntimeBoundaryError):
    pass


__all__ = [
    "DaemonRpcFailure",
    "RpcProtocolMismatch",
    "RuntimeBoundaryError",
    "RuntimeDaemonUnavailable",
]
