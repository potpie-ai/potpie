"""Fence authoritative resource workflows before their first content write."""

import uuid
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from potpie_context_engine.core.graph_journal import JournalError, JournalState
from potpie_context_engine.core.journal_context import (
    JournalWriteContext,
    current_journal_context,
    journal_write_context,
)

Workflow = TypeVar("Workflow", bound=Callable[..., Any])


def journal_resource_workflow(method: Workflow) -> Workflow:
    @wraps(method)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        pot_id = kwargs["pot_id"]
        journal = self.journal
        state = journal.journal_state(pot_id) if journal is not None else None
        if state is None:
            return method(self, *args, **kwargs)
        context = current_journal_context()
        if state.resource_guard:
            guard = state.resource_guard
            if (context.resource_operation_id, context.resource_owner) != (
                guard.operation_id,
                guard.owner,
            ):
                raise JournalError(
                    "resource operation is incomplete; reconcile it with a fenced recovery worker"
                )
            operation_id, owner = guard.operation_id, guard.owner
        else:
            operation_id, owner = uuid.uuid4().hex, uuid.uuid4().hex
            journal.begin_resource(
                pot_id=pot_id, operation_id=operation_id, owner=owner
            )
        with journal_write_context(
            JournalWriteContext(
                origin="resource",
                required_access=context.required_access,
                resource_operation_id=operation_id,
                resource_owner=owner,
            )
        ):
            result = method(self, *args, **kwargs)
            graph = getattr(result, "graph", None)
            successful = (
                (graph is None or graph.ok)
                and not getattr(result, "review_marker_errors", ())
                and not getattr(result, "missing_claim_keys", ())
            )
            if successful:
                journal.complete_resource(
                    pot_id=pot_id, operation_id=operation_id, owner=owner
                )
            # Errors deliberately leave the durable guard incomplete. A Python
            # finally block or an elapsed lease must never clear it.
            return result

    return cast(Workflow, guarded)


JOURNAL_CAPTURE_ACTIVE = "journal_capture_active"


def journal_capture_active(journal: Any, pot_id: str) -> bool:
    """Whether ``journal`` is capturing commits for ``pot_id``."""
    if journal is None:
        return False
    return isinstance(journal.journal_state(pot_id), JournalState)


def refuse_while_journaling(journal: Any, pot_id: str, operation: str) -> None:
    """Refuse a whole-pot teardown before any part of it runs.

    A pot reset, archive or purge would discard the pot's journal coverage
    mid-generation, and no operation retires a journal generation yet. The
    graph reset and the document purge both refuse on their own; checking
    once, first, is what keeps a teardown from stopping half-way (graph
    cleared, documents kept, or ledgers cleared and graph kept).
    """
    if journal_capture_active(journal, pot_id):
        raise JournalError(
            f"{operation} is refused while graph journal capture is active "
            "for this pot; it would discard the pot's commit history",
            code=JOURNAL_CAPTURE_ACTIVE,
        )


def reject_journal_admin(self, pot_id: str, operation: str) -> None:
    refuse_while_journaling(self.journal, pot_id, operation)


def reconcile_resource_operation(
    facade, *, pot_id, operation_id, expected_owner, verify_worker_stopped, workflow
):
    """Internal recovery: reconcile bytes, graph and markers via the normal workflow."""
    owner = uuid.uuid4().hex
    facade.journal.recover_resource_guard(
        pot_id=pot_id,
        operation_id=operation_id,
        expected_owner=expected_owner,
        new_owner=owner,
        verify_worker_stopped=verify_worker_stopped,
    )
    with journal_write_context(
        JournalWriteContext(
            origin="resource",
            required_access="admin",
            resource_operation_id=operation_id,
            resource_owner=owner,
        )
    ):
        return workflow()
