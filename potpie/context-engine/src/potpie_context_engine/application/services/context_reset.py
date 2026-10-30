"""Graph reset receipt recovery and the separate resource-purge outcome."""

from .resource_journal import refuse_while_journaling
from potpie_context_engine.core.mutation_execution import (
    ExecutionMetadata,
    MutationCapabilities,
    batch_fingerprint,
    execution_matches_batch,
)
from potpie_context_engine.core.ports.graph.mutation import (
    MutationExecutionState,
    MutationExecutionLookup,
)
from potpie_context_engine.core.reconciliation import MutationBatch
from potpie_context_engine.results import ResetContextResult


def reset_context(
    backend, resources, *, pot_id: str, mutation_id: str | None
) -> ResetContextResult:
    """Recover graph execution before evaluating current administrative policy.

    Resource purge is a separate store operation. A completed execution replay
    never repeats that purge: newly imported resources must survive old retries.
    """
    result = None
    replayed = False
    if mutation_id is not None:
        capabilities = getattr(backend.mutation, "capabilities", MutationCapabilities())
        lookup = (
            backend.mutation.lookup_execution(
                MutationBatch(reset_pot=True),
                expected_pot_id=pot_id,
                mutation_id=mutation_id,
            )
            if capabilities.execution_lookup
            else None
        )
        if (
            isinstance(lookup, MutationExecutionLookup)
            and lookup.state == MutationExecutionState.completed.value
        ):
            receipt = lookup.result
            if (
                lookup.mutation_id == mutation_id
                and lookup.batch_fingerprint
                == batch_fingerprint(MutationBatch(reset_pot=True))
                and execution_matches_batch(
                    receipt,
                    MutationBatch(reset_pot=True),
                    pot_id=pot_id,
                    mutation_id=mutation_id,
                )
            ):
                result = {
                    "ok": True,
                    "mutation_id": receipt.mutation_id,
                    "execution": (
                        receipt.execution.to_dict() if receipt.execution else None
                    ),
                }
                replayed = True
        if result is None and (
            not isinstance(lookup, MutationExecutionLookup)
            or lookup.mutation_id != mutation_id
            or lookup.state != MutationExecutionState.absent.value
        ):
            return ResetContextResult(
                context_id=pot_id,
                reset=False,
                mutation_id=mutation_id,
                resources_status=(
                    "unknown_on_recovery" if resources is not None else "not_configured"
                ),
                detail="Reset execution is unresolved; no reset or resource purge was attempted.",
            )
    if result is None:
        refuse_while_journaling(getattr(backend, "journal", None), pot_id, "pot reset")
        result = backend.mutation.reset_pot(pot_id, mutation_id=mutation_id)

    reset_ok = bool(result.get("ok", True))
    execution = (
        ExecutionMetadata.from_dict(result["execution"])
        if result.get("execution") is not None
        else None
    )
    purged, detail = None, result.get("error")
    if resources is None:
        resource_status = "not_configured"
    elif replayed:
        resource_status = "unknown_on_recovery"
    elif not reset_ok:
        resource_status = "not_attempted"
    else:
        try:
            purged = bool(resources.purge_pot(pot_id))
            resource_status = "completed"
        except Exception as exc:
            resource_status, detail = (
                "failed",
                f"Graph reset completed; resource purge failed: {exc}",
            )
    return ResetContextResult(
        context_id=pot_id,
        reset=reset_ok,
        mutation_id=result.get("mutation_id"),
        execution=execution,
        resources_purged=purged,
        resources_status=resource_status,
        detail=detail,
    )
