"""Domain-level errors; translate to HTTP/CLI at adapter boundaries."""


class ContextEngineError(Exception):
    """Base for all context-engine domain errors."""


class ContextEngineDisabled(ContextEngineError):
    """Feature flag off or graph clients unavailable."""


class GraphSubstrateUnavailable(ContextEngineDisabled):
    """The graph store cannot be trusted to answer, so it must not answer.

    Distinct from "the backend is down": the store here is *reachable* and will
    happily serve a dataset that may be missing recent writes — the worst
    outcome for a memory product, because the answer looks complete. Raised
    instead of opening the store, and carries the operator's recovery step so
    the CLI's error contract can print something actionable rather than a
    generic unavailability message.
    """

    def __init__(self, message: str, *, recommended_next_action: str | None = None):
        super().__init__(message)
        self.recommended_next_action = recommended_next_action


class PotTeardownFailed(ContextEngineDisabled):
    """A pot's graph wipe did not happen, so the rest of the teardown must not.

    Tearing a pot down destroys two stores, and only one order is safe: the
    resource tree may go once the claims citing its chunks are gone. Mutation
    adapters report an unreachable store by *returning* ``{"ok": False,
    "error": ...}`` rather than raising, so a caller that ignores the return
    purges the chunk files anyway and leaves the pot worse than it found it —
    live claims whose evidence no longer exists — while reporting success.
    Raised instead of continuing, carrying the adapter's own error text so the
    CLI's unavailability contract prints what actually failed.
    """

    def __init__(self, message: str, *, recommended_next_action: str | None = None):
        super().__init__(message)
        self.recommended_next_action = recommended_next_action


class AlreadyIngested(ContextEngineError):
    """Source was already recorded in the ingestion ledger."""


class UpstreamError(ContextEngineError):
    """External dependency failed (GitHub, Neo4j, Postgres)."""


class BridgeError(ContextEngineError):
    """Failed to write structural graph bridges."""


class PotNotFound(ContextEngineError):
    """Host could not resolve pot_id."""


class PotArchived(ContextEngineError):
    """The pot exists, but archiving it was the end of its life.

    Distinct from :class:`PotNotFound` because the two need different repairs:
    a ref nobody recognises is a typo, while this one resolved — the pot is
    listed under ``pot list --archived`` and its graph and resource tree were
    torn down when it was archived. Selecting it, writing to it, or routing a
    repo into it can only produce an empty answer from a pot the operator
    believes still holds their project's memory.

    The flag used to be write-only: nothing in the product read it, so archived
    pots kept appearing in ``pot list``, kept being selectable, kept accepting
    claims, and a repo default pointing at one kept routing every scoped read
    and write into it.
    """

    def __init__(self, message: str, *, recommended_next_action: str | None = None):
        super().__init__(message)
        self.recommended_next_action = recommended_next_action


class PotNameConflict(ContextEngineError):
    """A pot name (or a name shadowing a pot id) is already taken.

    Pot names are the refs humans type, and ``rename`` enforced nothing: two
    pots could end up sharing one, after which every bare-ref resolution picked
    an arbitrary one of them — including ``pot reset <name> --confirm``, which
    then destroyed whichever it happened to find first. A name equal to another
    pot's *id* is the same defect wearing a different hat, since refs resolve
    against both.
    """

    def __init__(self, message: str, *, recommended_next_action: str | None = None):
        super().__init__(message)
        self.recommended_next_action = recommended_next_action


class MutationBatchValidationError(ContextEngineError):
    """A :class:`~potpie_context_core.reconciliation.MutationBatch` failed structural validation.

    When validation is driven by ontology checks, ``structured_issues`` carries
    ``{"entity", "issue"}`` rows aligned with :func:`potpie_context_core.reconciliation_issues.validation_lines_to_issues`.
    """

    def __init__(
        self,
        message: str,
        *,
        structured_issues: tuple[dict[str, str], ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.structured_issues = structured_issues or ()


# Back-compat alias (Step 5a rename).
ReconciliationPlanValidationError = MutationBatchValidationError


class ReconciliationApplyError(ContextEngineError):
    """Deterministic apply step failed after validation."""


class CapabilityNotImplemented(ContextEngineError):
    """A port/capability is wired into the skeleton but has no real
    implementation yet.

    The architectural skeleton wires *every* Protocol to at least a dummy
    adapter, so no public surface dead-ends in a bare ``NotImplementedError``.
    When an inbound adapter hits an unbuilt capability it catches this and
    renders the structured not-implemented contract (CLI exit ``2``/``3``,
    ``context_status`` ``not_implemented``, JSON ``code``/``message``/
    ``recommended_next_action``) instead of a traceback.

    ``capability`` is a dotted slot name (e.g. ``graph.inspection.path``) so
    logs/telemetry can attribute the gap to a specific boundary.
    """

    def __init__(
        self,
        capability: str,
        *,
        detail: str | None = None,
        recommended_next_action: str | None = None,
    ) -> None:
        self.capability = capability
        self.detail = detail
        self.recommended_next_action = recommended_next_action
        message = f"Capability not implemented: {capability}"
        if detail:
            message = f"{message} — {detail}"
        super().__init__(message)
