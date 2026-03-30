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

