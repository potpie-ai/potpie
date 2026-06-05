"""Domain-level errors; translate to HTTP/CLI at adapter boundaries."""


class ContextEngineError(Exception):
    """Base for all context-engine domain errors."""


class ContextEngineDisabled(ContextEngineError):
    """Feature flag off or graph clients unavailable."""


class AlreadyIngested(ContextEngineError):
    """Source was already recorded in the ingestion ledger."""


class UpstreamError(ContextEngineError):
    """External dependency failed (GitHub, Neo4j, Postgres)."""


class BridgeError(ContextEngineError):
    """Failed to write structural graph bridges."""


class PotNotFound(ContextEngineError):
    """Host could not resolve pot_id."""


class ReconciliationPlanValidationError(ContextEngineError):
    """Reconciliation plan failed validation.

    When validation is driven by ontology checks, ``structured_issues`` carries
    ``{"entity", "issue"}`` rows aligned with :func:`domain.reconciliation_issues.validation_lines_to_issues`.
    """

    def __init__(
        self,
        message: str,
        *,
        structured_issues: tuple[dict[str, str], ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.structured_issues = structured_issues or ()


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
