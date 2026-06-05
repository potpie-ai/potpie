"""Operation contract — neutral, transport-agnostic ops served identically over any protocol."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable
from pydantic import BaseModel


class AuthRequirement(Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    NONE = "none"


class OpKind(Enum):
    UNARY = "unary"
    SERVER_STREAM = "server_stream"  # reserved; V1 ships UNARY only


class OperationError(Exception):
    """Uniform error for any operation handler. Transport adapters map ``code`` onto protocol-native error shapes."""

    def __init__(
        self,
        code: str,
        message: str,
        detail: dict[str, Any] | None = None,
        recommended_next_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.recommended_next_action = recommended_next_action


@dataclass(frozen=True)
class Principal:
    """Identity of the caller produced by transport auth."""

    name: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationContext:
    principal: Principal
    request_id: str
    deadline: float | None = None
    deps: Any = None  # opaque; component-supplied wired deps


@dataclass(frozen=True)
class OperationSpec:
    name: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel, OperationContext], Awaitable[BaseModel]]
    summary: str
    mutating: bool = False
    auth: AuthRequirement = AuthRequirement.REQUIRED
    kind: OpKind = OpKind.UNARY


class OperationRegistry:
    def __init__(self) -> None:
        self._ops: dict[str, OperationSpec] = {}

    def register(self, op: OperationSpec) -> None:
        if op.name in self._ops:
            raise ValueError(f"operation already registered: {op.name!r}")
        self._ops[op.name] = op

    def get(self, name: str) -> OperationSpec:
        return self._ops[name]

    def all(self) -> list[OperationSpec]:
        return list(self._ops.values())
