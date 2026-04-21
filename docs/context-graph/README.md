# Context Graph Docs

The canonical combined context graph architecture is:

- [Context Graph Architecture](graph.md)
- [Context Graph Features And Functionalities](features-and-functionalities.md)
- [Context Graph Planning Next Steps](planning-next-steps.md)
- [Unified Graphiti Application Architecture](unified-graphiti-application-architecture.md)
- [Context Engine Test Harness And Findings](testing-and-bugs.md)

`graph.md` remains the canonical product architecture. The unified Graphiti application doc is the focused implementation plan for collapsing the application layer onto one Graphiti-backed graph port.

The architecture set now combines:

- the original `docs/context-graph/graph.md` architecture
- the pot, context-wrap, source-reference, and agent-integration requirements
- the minimal agent context port: `context_resolve`, `context_search`, `context_record`, and `context_status`
- the current `app/src/context-engine` implementation review
- the gap analysis and recommended next steps

Keep product-level architecture updates in `docs/context-graph/graph.md`. Keep implementation migration details for the one-port Graphiti application layer in `docs/context-graph/unified-graphiti-application-architecture.md`.

Use `docs/context-graph/planning-next-steps.md` for the current refactor plan that separates stable agent/API contracts from UI, ingestion automation, and operator/admin surfaces; moves the product model toward source-first pot management; and wires deeper status, source resolver, and verification behavior behind the existing four-tool agent port.

## Implementation Status

Phase 1, the canonical ontology foundation, is implemented in:

- [`app/src/context-engine/domain/ontology.py`](../../app/src/context-engine/domain/ontology.py)
- [`app/src/context-engine/application/use_cases/reconciliation_validation.py`](../../app/src/context-engine/application/use_cases/reconciliation_validation.py)

The ontology module is the code-level catalog for public canonical labels, edge types, allowed relationships, required properties, lifecycle/status validation, and the current ontology version. Generic structural reconciliation mutations are validated against this catalog before they can be applied.

Phase 2, source references, freshness, and verification metadata, is implemented in:

- [`app/src/context-engine/domain/source_references.py`](../../app/src/context-engine/domain/source_references.py)
- [`app/src/context-engine/domain/intelligence_models.py`](../../app/src/context-engine/domain/intelligence_models.py)
- [`app/src/context-engine/application/services/context_resolution.py`](../../app/src/context-engine/application/services/context_resolution.py)
- [`app/src/context-engine/adapters/inbound/http/api/v1/context/router.py`](../../app/src/context-engine/adapters/inbound/http/api/v1/context/router.py)
- [`app/src/context-engine/adapters/inbound/mcp/server.py`](../../app/src/context-engine/adapters/inbound/mcp/server.py)

The resolver still exposes source behavior through `context_resolve`, not a separate source tool. Requests can now pass `mode`, `source_policy`, `include`, `exclude`, and scoped `source_refs`. Responses include normalized `source_refs`, a `freshness` report, explicit `fallbacks`, and recommended verification actions when source-backed context is missing or unverified.

Phase 3, the minimal agent context port, is implemented in:

- [`app/src/context-engine/domain/agent_context_port.py`](../../app/src/context-engine/domain/agent_context_port.py)
- [`app/src/context-engine/domain/intelligence_policy.py`](../../app/src/context-engine/domain/intelligence_policy.py)
- [`app/src/context-engine/application/services/context_resolution.py`](../../app/src/context-engine/application/services/context_resolution.py)
- [`app/src/context-engine/adapters/inbound/http/api/v1/context/router.py`](../../app/src/context-engine/adapters/inbound/http/api/v1/context/router.py)
- [`app/src/context-engine/adapters/inbound/mcp/server.py`](../../app/src/context-engine/adapters/inbound/mcp/server.py)

The agent-facing MCP surface is now the four-tool port: `context_resolve`, `context_search`, `context_record`, and `context_status`. Specialized structural reads remain available as internal HTTP query endpoints for compatibility, but they are no longer registered as separate MCP tools. `context_resolve` now accepts richer scope fields, `budget`, and `as_of`, and returns the common agent envelope with `answer`, `facts`, `evidence`, `source_refs`, `coverage`, `freshness`, `fallbacks`, and `recommended_next_actions`.

Phase 4, project map expansion, is implemented as the first canonical project-map read path in:

- [`app/src/context-engine/domain/intelligence_models.py`](../../app/src/context-engine/domain/intelligence_models.py)
- [`app/src/context-engine/domain/intelligence_policy.py`](../../app/src/context-engine/domain/intelligence_policy.py)
- [`app/src/context-engine/domain/ports/intelligence_provider.py`](../../app/src/context-engine/domain/ports/intelligence_provider.py)
- [`app/src/context-engine/adapters/outbound/intelligence/hybrid_graph.py`](../../app/src/context-engine/adapters/outbound/intelligence/hybrid_graph.py)
- [`app/src/context-engine/adapters/outbound/neo4j/structural.py`](../../app/src/context-engine/adapters/outbound/neo4j/structural.py)

`context_resolve` now plans and returns a `project_map` family for project-wide orientation: purpose, repositories, services, components, features, docs, deployments, runbooks, local workflows, scripts, config references, preferences, and agent instructions. This is exposed through existing `intent`, `scope`, and `include` parameters rather than new public tools. The structural adapter reads compact canonical nodes and relationship references only; full source payloads remain behind source refs and external resolvers.

Phase 5, debugging memory and prior fixes, is implemented as the first reusable debugging read path in:

- [`app/src/context-engine/domain/intelligence_models.py`](../../app/src/context-engine/domain/intelligence_models.py)
- [`app/src/context-engine/domain/intelligence_policy.py`](../../app/src/context-engine/domain/intelligence_policy.py)
- [`app/src/context-engine/adapters/outbound/intelligence/hybrid_graph.py`](../../app/src/context-engine/adapters/outbound/intelligence/hybrid_graph.py)
- [`app/src/context-engine/adapters/outbound/neo4j/structural.py`](../../app/src/context-engine/adapters/outbound/neo4j/structural.py)

`context_resolve` now plans and returns `debugging_memory` for `prior_fixes`, `diagnostic_signals`, `incidents`, and `alerts`. It reads compact canonical `Fix`, `BugPattern`, `Investigation`, `DiagnosticSignal`, `Incident`, and `Alert` records plus relationship references for affected scope, signals, and related changes. `context_record` accepts debugging-oriented record types such as `fix`, `bug_pattern`, `investigation`, `diagnostic_signal`, and `incident_summary`; reconciliation is responsible for turning those records into canonical graph mutations.

Phase 6, agent instructions, skills, and operating workflows, is implemented in:

- [`app/src/context-engine/domain/agent_context_port.py`](../../app/src/context-engine/domain/agent_context_port.py)
- [`app/src/context-engine/adapters/inbound/http/api/v1/context/router.py`](../../app/src/context-engine/adapters/inbound/http/api/v1/context/router.py)
- [`app/src/context-engine/adapters/inbound/mcp/server.py`](../../app/src/context-engine/adapters/inbound/mcp/server.py)
- [`app/src/context-engine/adapters/inbound/cli/templates/agent_bundle/AGENTS.md`](../../app/src/context-engine/adapters/inbound/cli/templates/agent_bundle/AGENTS.md)
- [`app/src/context-engine/adapters/inbound/cli/templates/agent_bundle/.agents/skills/potpie-agent-context/SKILL.md`](../../app/src/context-engine/adapters/inbound/cli/templates/agent_bundle/.agents/skills/potpie-agent-context/SKILL.md)

The code now exposes a stable agent port manifest and `context_resolve` recipes for feature, debugging, review, operations, docs, and onboarding workflows. `context_status` returns the manifest and recommended recipe for an optional intent, MCP tool descriptions steer agents to the four-tool port, and generated `AGENTS.md` plus repo-local skills explain how to gather bounded context without introducing one-off tools for every context type.

Phase 7, quality, drift management, and scale, is implemented as the first graph-quality policy and response layer in:

- [`app/src/context-engine/domain/graph_quality.py`](../../app/src/context-engine/domain/graph_quality.py)
- [`app/src/context-engine/domain/source_references.py`](../../app/src/context-engine/domain/source_references.py)
- [`app/src/context-engine/domain/ontology.py`](../../app/src/context-engine/domain/ontology.py)
- [`app/src/context-engine/application/services/context_resolution.py`](../../app/src/context-engine/application/services/context_resolution.py)

`context_resolve` and `context_status` now return a `quality` report with freshness/source-sync metrics, quality issues, source-of-truth policy, freshness TTL policy, and recommended maintenance jobs such as `verify_entity`, `refresh_scope`, `resync_source_scope`, and `expire_stale_facts`. The ontology includes first-pass `QualityIssue`, `MaintenanceJob`, and `MaterializedAccessPath` entities plus edges for `FLAGS`, `REPAIRS`, and `MATERIALIZES`, so future drift and housekeeping workflows can write canonical graph quality state instead of burying it in logs.
