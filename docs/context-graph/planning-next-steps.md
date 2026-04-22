# Context Graph Planning Next Steps

## Goal

Refactor the current context graph feature into a cleaner product and architecture shape:

- Keep the agent surface small and stable.
- Make sources the primary product model for pot data scope.
- Treat graph data as source-reference-first memory, not copied source payloads.
- Separate stable agent/API contracts from UI, ingestion automation, and operator endpoints.
- Wire readiness, freshness, quality, and source verification deeply enough that agents can trust the response envelope.
- Treat compatibility routes and provider-specific write branches as temporary
  migration scaffolding. The target system has one Graphiti-backed graph layer
  and generic ontology-validated graph mutations.

This plan builds on [`graph.md`](graph.md) and the clarified feature contract in [`features-and-functionalities.md`](features-and-functionalities.md).

## Desired End State

### Agent Surface

MCP remains limited to:

- `context_resolve`
- `context_search`
- `context_record`
- `context_status`

Feature, debugging, review, operations, docs, onboarding, planning, refactor, test, and security workflows remain `context_resolve` recipes. Do not add one-off public tools such as `context_get_feature_context`, `context_get_source`, or `context_get_debugging_context`.

### HTTP Surface

The HTTP API is grouped by intent:

| Group | Purpose | Example Routes |
| --- | --- | --- |
| Stable agent/API-client | External agents, SDKs, automation clients | `query/context-graph`, `record`, `status`, `ingest` |
| UI/application | Potpie product flows | `pots`, `sources`, `members`, `invitations`, UI raw ingest, event reads |
| Ingestion/automation | Webhooks, scheduled sync, worker control | `sync`, `ingest-pr`, `events/reconcile`, replay |
| Operator/admin | Repair, reset, graph hygiene | conflicts, reset, maintenance |

### Source Model

`ContextGraphPotSource` becomes the primary source-of-truth for what a pot is connected to. Repositories remain a source subtype with code-graph behavior during the migration; they should not define the broader source model.

Supported source classes should converge on one shape:

- `source_id`
- `pot_id`
- `source_kind`
- `provider`
- `provider_host`
- `scope_json`
- `sync_enabled`
- `sync_mode`
- `last_sync_at`
- `last_success_at`
- `last_error_at`
- `last_error`
- resolver capability metadata

### Source-Backed Context

`source_policy` remains the public extension point:

- `references_only`: compact graph facts and refs.
- `summary`: bounded resolver summaries.
- `verify`: source-of-truth checks and verification state.
- `snippets`: bounded snippets with permissions and budget enforcement.

No separate public `context_source` tool is introduced.

## Current Gaps To Close

1. **Feature docs and API docs over-flatten the API surface.**  
   They list stable product APIs, app APIs, ingestion APIs, and operator APIs together.

2. **`context_status` is too shallow.**  
   It returns first-pass readiness but does not yet summarize source rows, event-ledger health, last successful ingestion, resolver capability state, or last verification.

3. **`source_policy` is stronger as a contract than as an implementation.**  
   The path mostly returns refs and fallbacks today. Real source summary, verification, and snippet resolvers need to be wired.

4. **PR diff detail is too close to graph read semantics.**  
   Full diffs should be fetched from source systems through bounded source-backed modes, while graph storage keeps compact summaries, changed symbols, touched files, decisions, and refs.

5. **Repositories and sources overlap.**  
   GitHub repository routes are useful, but the product concept should be source-first.

6. **`events/ingest` is a compatibility alias but appears primary in some docs.**  
   New docs and clients should use `events/reconcile`.

7. **Operator endpoints need explicit treatment.**  
   Reset, conflict resolution, and maintenance jobs should have admin authorization, audit expectations, and separate docs.

## Refactor Plan

### Phase 1: Documentation And Contract Cleanup

Status: started by this document and the updated feature reference.

Tasks:

- Keep `features-and-functionalities.md` organized by API surface group.
- Keep `graph.md` as the architecture source of truth.
- Keep `unified-graphiti-application-architecture.md` focused on implementation migration.
- Mark `/events/ingest` as compatibility-only.
- Reframe `pr_diff` as source-backed artifact detail, not durable graph payload.
- Correct CLI docs and feature docs around `potpie add`, `potpie ingest`, and `potpie pot repo add`.

Acceptance criteria:

- A reader can identify which endpoints are for agents, UI, ingestion automation, and operators.
- A reader understands that the four-tool MCP port is the only agent tool surface.
- A reader understands that full source payloads are fetched through source resolvers, not copied wholesale into the graph.

### Phase 2: Status API Deepening

Make `POST /api/v2/context/status` the trust and readiness API for agents, CLI, UI, and SDKs.

Tasks:

- Add a status read model that can access pot sources, repositories, event ledgers, and resolver capabilities.
- Include attached sources and their sync status.
- Include last successful ingestion per source and pot.
- Include queued, processing, done, and error event counts.
- Include most recent event errors.
- Include resolver capability availability for `references_only`, `summary`, `verify`, and `snippets`.
- Include source verification state and last verification timestamps when known.
- Preserve the current `agent_port` manifest and recommended recipe behavior.

Acceptance criteria:

- `context_status` can tell an agent whether graph memory is safe for orientation, needs verification, or is degraded.
- UI can use the same endpoint to drive source readiness indicators.
- CLI `doctor` can optionally display deeper context status after pot resolution.

### Phase 3: Source-First Pot Model

Make sources the primary product-level data-scope model.

Tasks:

- Audit all routes that create repository rows and source rows.
- Make `/pots/{pot_id}/sources/*` the preferred app path for source attachment.
- Keep `/pots/{pot_id}/repositories` only until CLI GitHub and webhook clients
  can use source-first routing.
- Add source capability metadata needed by context resolution and status.
- Ensure deleting a source consistently updates repository routing when the source is a GitHub repository.
- Update CLI docs to explain repository compatibility versus source-first model.

Acceptance criteria:

- New UI/source integrations do not need to create repository-shaped records unless they are actually code repositories.
- GitHub repositories still work for existing CLI and webhook routes.
- Future sources such as docs, Slack, incidents, deployments, and alerts fit the same source model.

### Phase 4: Source Resolver Port

Status: implemented. `domain.ports.source_resolver.SourceResolverPort` is wired
into `ContextResolutionService` and the host container; default
`NullSourceResolver` plus a `CompositeSourceResolver` dispatching to
`GitHubPullRequestResolver` and `DocumentationUriResolver` are registered in
host wiring. Capability advertisement upgrades `context_status` per
`(provider, source_kind)`.

Add a source resolver layer behind `context_resolve`.

Tasks:

- Define a source resolver port that accepts source refs, source policy, scope, budget, and caller auth context.
- Implement resolver capability discovery for `context_status`.
- Implement resolver fallback reasons: unavailable, unsupported source type, permission denied, stale token, source unreachable, budget exceeded, and no source refs.
- Start with GitHub PR/source refs and documentation URI refs.
- Keep fetched payloads bounded and avoid storing full payloads in graph nodes by default.

Acceptance criteria:

- `source_policy=summary` can return compact source-backed summaries for supported refs.
- `source_policy=verify` can mark whether a fact was checked against a source of truth.
- `source_policy=snippets` can return small bounded snippets only when requested.
- Unsupported source policies produce clear fallbacks, not silent success.

### Phase 5: PR Diff Refactor

Status: implemented. Diff snippets now flow through
`artifact={kind:"pr", identifier:N}` plus `source_policy=summary`/`snippets`,
served by `GitHubPullRequestResolver`. Artifact-derived source refs carry
`source_system="github"` so the composite resolver dispatches them. The
legacy `include=["pr_diff"]` graph-query branch is annotated as
compatibility-only — the graph keeps changed files, touched symbols, PR/review
summaries, and decisions, but full diff text is no longer a documented graph
family.

Move detailed PR diff behavior behind source-backed artifact resolution.

Tasks:

- Keep changed files, touched symbols, PR summaries, review summaries, and decisions in the graph.
- Stop documenting full `pr_diff` as a normal queryable graph family.
- Route diff detail through `artifact={kind:"pr"}` plus `source_policy=summary` or `source_policy=snippets`.
- Add budget enforcement for diff snippets.
- Add tests that high-level review context still works without fetching full diffs.

Acceptance criteria:

- Review agents can get useful PR context without loading full diffs by default.
- Detailed diff access is explicit, bounded, permission-aware, and source-backed.

### Phase 6: Event API Clarification

Status: implemented. ``POST /events/reconcile`` is the canonical path. The
alias ``POST /events/ingest`` is hidden from OpenAPI
(``include_in_schema=False``, ``deprecated=True``), attaches
``Deprecation: true`` plus an RFC 7234 ``Warning`` header and a ``Link``
header pointing at the successor route, and increments a process-local
``_EVENTS_INGEST_ALIAS_CALLS`` counter (exposed via
``events_ingest_alias_call_count``) with a ``logging.WARNING`` on each
call. Docs route readers to ``/events/reconcile``.

Make normalized event reconciliation the single primary event write path.

Tasks:

- Keep `/events/reconcile` as the documented event submission path.
- Keep `/events/ingest` as hidden compatibility alias until clients migrate.
- Add deprecation notes or metrics for alias usage if needed.
- Align CLI/event docs around event inspection and replay.

Acceptance criteria:

- New clients only use `/events/reconcile`.
- Existing compatibility clients continue to work.
- Docs no longer promote two equivalent event-write APIs.

### Phase 7: Operator Surface Hardening

Status: implemented. Operator routes (``/reset``, ``/conflicts/list``,
``/conflicts/resolve``, ``/maintenance/classify-modified-edges``) now carry
the OpenAPI tag ``context:operator`` and a ``[operator]`` prefix on their
summaries, so they surface as a distinct group in docs/SDKs rather than
alongside agent/product routes. Every destructive or graph-mutating call
emits a structured audit log via the dedicated
``context_engine.operator_audit`` logger (see
``_audit_operator_action`` in ``router.py``), with action, pot_id, actor
identity, ``dry_run`` flag, and per-action details. Dry-run behavior of
``/maintenance/classify-modified-edges`` is now explicit in the route
description and audited on every invocation. The CLI ``potpie pot
hard-reset`` now prompts for destructive confirmation unless ``--yes`` is
passed, and the help text calls the command out as destructive and
operator/admin.

Separate operator/admin actions from normal product APIs.

Tasks:

- Review auth checks for reset, maintenance, and conflict resolution.
- Add audit events for destructive or graph-mutating operator actions.
- Document dry-run behavior for maintenance routes.
- Consider moving operator routes under a clearer prefix or tagging them separately in OpenAPI.
- Ensure CLI destructive commands clearly label risk and require explicit pot scope.

Acceptance criteria:

- Reset and maintenance actions are visibly admin/operator workflows.
- Conflict resolution has an audit trail.
- OpenAPI or docs do not make operator routes look like everyday agent features.

### Phase 8: Test And Migration Coverage

Status: implemented. Contract coverage for the stable agent surface lives in
`tests/unit/test_agent_surface_contract.py`: it introspects the live
`FastMCP` instance to assert the tool set is exactly
`{context_resolve, context_search, context_record, context_status}`,
pins `context_ingest_episode` as a non-tool module-level helper, validates
every `CONTEXT_RESOLVE_RECIPES` entry against `CONTEXT_INCLUDE_VALUES` and
the allowed `mode`/`source_policy` vocabularies (with no smuggled-in
per-intent `tool` field), asserts `context_port_manifest()` advertises
exactly the four tools and mirrors the recipe catalog, and confirms
`context_recipe_for_intent` returns a defensive copy so callers cannot
mutate the shared include list. Existing coverage already addressed the
other Phase 8 items: `tests/unit/test_context_status.py` covers readiness
states and resolver-capability upgrades; `tests/unit/test_source_resolvers.py`
covers source-policy fallbacks (`RESOLVER_UNAVAILABLE`,
`UNSUPPORTED_SOURCE_POLICY`, `UNSUPPORTED_SOURCE_TYPE`, `BUDGET_EXCEEDED`)
across the null/composite/github/docs resolvers;
`tests/unit/test_context_resolution.py` Phase-5 block covers review-intent
context working without diff fetch under `references_only` plus PR artifact
routing through `summary`/`snippets`; `tests/unit/test_events_ingest_alias.py`
pins the deprecated-alias contract; and
`tests/unit/test_operator_audit.py` pins the operator tag/audit contract.

Tasks:

- Unit-test `context_status` readiness states.
- Unit-test source policy fallbacks.
- Unit-test source resolver behavior for supported and unsupported refs.
- Test that MCP still exposes only the four-tool port.
- Test that `context_resolve` recipes do not require new public tools.
- Test that PR review context works with source-backed diff detail disabled.
- Add doc examples to smoke tests where practical.

Acceptance criteria:

- Contract tests make accidental endpoint/tool sprawl visible.
- Refactors can proceed without breaking CLI/MCP assumptions.

## Recommended Work Order

1. Land documentation and contract cleanup.
2. Deepen `context_status`, because it improves every consumer without changing write paths.
3. Normalize source model and source capability metadata.
4. Add source resolver port and fallbacks.
5. Move PR diff detail behind source-backed artifact resolution.
6. De-emphasize compatibility event aliases.
7. Harden operator routes.
8. Add broader source-specific ingestion and verification jobs.

## Non-Goals

- Do not add a public tool for every context family.
- Do not turn the graph into a full copy of PR diffs, docs, Slack threads, logs, or incident payloads.
- Do not make `context_search` the default agent entrypoint.
- Do not require agents to know Graphiti, Cypher, Neo4j labels, or source-specific schemas.
- Do not block the simple CLI workflows while refactoring the source-first model.

## Decision Summary

The feature should become smaller at the public agent boundary and richer behind that boundary. The right direction is not more endpoints for more context types. The right direction is stronger `context_resolve`, richer source refs, better status/quality reporting, source-backed verification and snippets, and a source-first pot model that lets Potpie add new integrations without changing the agent contract.
