"""Per-tool factories that the reconciliation agent composes at run time.

Each builder takes the batch run state plus its own infrastructure
dependencies (e.g. a sandbox client) and returns a list of pydantic-ai
``Tool`` callables for that batch run. Source-specific builders live next
to their connector — e.g. ``adapters.outbound.connectors.github.agent_tools``.
"""
