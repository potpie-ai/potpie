"""Trusted execution context, never accepted from client mutation fields."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JournalWriteContext:
    origin: str = "graph"
    required_access: str = "write"
    resource_operation_id: str | None = None
    resource_owner: str | None = None
    plan_id: str | None = None


_DEFAULT_CONTEXT = JournalWriteContext()
_CONTEXT: ContextVar[JournalWriteContext | None] = ContextVar(
    "graph_journal_write_context", default=None
)


def current_journal_context() -> JournalWriteContext:
    return _CONTEXT.get() or _DEFAULT_CONTEXT


@contextmanager
def journal_write_context(context: JournalWriteContext):
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)
