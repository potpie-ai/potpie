"""Transport-neutral outcomes for the public Context Engine boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, Mapping, TypeAlias, TypeVar


RetryPosture = Literal["safe", "unsafe", "unknown", "not_applicable"]


@dataclass(frozen=True, slots=True)
class DomainError:
    """A context-domain input, state, capability, or invariant failure."""

    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "not_applicable"
    category: Literal["domain"] = "domain"


@dataclass(frozen=True, slots=True)
class DependencyError:
    """A declared engine-owned dependency failed while serving an operation."""

    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "unknown"
    category: Literal["dependency"] = "dependency"


@dataclass(frozen=True, slots=True)
class EngineLifecycleError:
    """Construction, closure, or engine-lifetime enforcement failed."""

    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    recommended_next_action: str | None = None
    retry_posture: RetryPosture = "not_applicable"
    category: Literal["engine_lifecycle"] = "engine_lifecycle"


EngineError: TypeAlias = DomainError | DependencyError | EngineLifecycleError

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Success(Generic[T]):
    """Successful public operation outcome."""

    value: T
    ok: Literal[True] = True


@dataclass(frozen=True, slots=True)
class Failure:
    """Failed public operation outcome."""

    error: EngineError
    ok: Literal[False] = False


Outcome: TypeAlias = Success[T] | Failure


__all__ = [
    "DependencyError",
    "DomainError",
    "EngineError",
    "EngineLifecycleError",
    "Failure",
    "Outcome",
    "RetryPosture",
    "Success",
]
