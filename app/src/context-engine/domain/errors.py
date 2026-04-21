"""Domain-level errors; translate to HTTP/CLI at adapter boundaries."""


class ContextEngineError(Exception):
    """Base for all context-engine domain errors."""


class ContextEngineDisabled(ContextEngineError):
    """Feature flag off or graph clients unavailable."""


class AlreadyIngested(ContextEngineError):
    """Source was already recorded in the ingestion ledger."""


class UpstreamError(ContextEngineError):
    """External dependency failed (GitHub, Neo4j, Graphiti, Postgres)."""


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

