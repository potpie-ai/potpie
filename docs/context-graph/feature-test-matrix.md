# Context Graph Feature / Test Matrix

## Purpose

This doc is the canonical mapping between the product features documented in [`features-and-functionalities.md`](features-and-functionalities.md) and the tests that cover them. It is meant to:

1. Make it easy to see, per feature, **what is implemented and tested today**.
2. Make it easy to see **what is intentionally not yet covered** (gaps + planned work).
3. Give reviewers a quick way to spot-check that a change ships with corresponding tests.

Notation in the tables below:

- Tests under `app/src/context-engine/tests/` are part of the embedded `context-engine` package (portable layer).
- Tests under `tests/unit/context_graph/` and `tests/unit/intelligence/` are part of the Potpie host (host wiring layer).
- Path prefixes are abbreviated to keep the tables readable:
  - `ce/` &rarr; `app/src/context-engine/tests/`
  - `host/` &rarr; `tests/`

## How to use this doc

- Adding or extending a feature: locate the row, add the test file in the right column, and check the gap list to see whether the work also closes any "planned" item.
- Reviewing a PR: the reviewer should be able to find the changed area in this matrix and see at least one test row that matches.
- Removing a feature: delete the row plus the tests it points to and update the gap section.

## Feature 1 — Context Pot Management

Pot tenancy, membership, invitations, and active-pot selection.

| Capability | Implementation entrypoint | Tests |
| --- | --- | --- |
| Pot CRUD and membership roles | `app/modules/context_graph/pot_*` | `host/unit/context_graph/test_pot_member_roles.py` |
| Active pot selection / aliases / git-remote resolution (CLI) | `adapters/inbound/cli/credentials_store.py`, `git_project.py` | `ce/unit/test_credentials_store.py`, `ce/unit/test_git_project.py` |
| API-key access to pots | `app/modules/auth/api_key_deps.py` | `host/unit/auth/test_api_key_service.py` |

## Feature 2 — Source Management

Source-first attachment of repositories, Linear teams, and other external systems to a pot.

| Capability | Tests |
| --- | --- |
| Pot-source service (scope hash, scope JSON, row &rarr; status conversion) | `host/unit/context_graph/test_pot_sources_service.py` |
| Source references (compact provenance refs on facts) | `ce/unit/test_source_references.py`, `host/unit/context_graph/test_source_references.py` |
| Source resolution (`references_only`, `summary`, `verify`, `snippets`) | `ce/unit/test_source_resolution.py`, `host/unit/context_graph/test_source_resolution.py`, `ce/unit/test_source_resolvers.py` |
| Linear source: plan, resolver, webhook normalization | `ce/unit/test_linear_issue_plan.py`, `ce/unit/test_linear_issue_resolver.py`, `ce/unit/test_linear_webhook_normalize.py`, `host/unit/integrations/test_linear_sync_graph.py` |

## Feature 3 — Repository And Codebase Ingestion

`POST /api/v2/context/sync` and the worker-side backfill that builds structural project context.

| Capability | Tests |
| --- | --- |
| Pot-wide and per-repo backfill orchestration | `ce/unit/test_backfill_pot.py` |
| Async ingestion plan and queue handoff | `ce/unit/test_ingestion_async_plan.py`, `ce/unit/test_queue_factory.py`, `ce/unit/test_hatchet_env_bootstrap.py` |
| Ingestion DB / step status surfacing | `ce/unit/test_ingestion_db_status.py`, `ce/unit/test_ingestion_step_db_status.py`, `host/unit/context_graph/test_ingestion_db_status.py` |
| Ingestion event models | `ce/unit/test_ingestion_event_models.py` |

## Feature 4 — Pull Request Ingestion

`POST /api/v2/context/ingest-pr` and the GitHub-webhook driven path.

| Capability | Tests |
| --- | --- |
| Generic PR plan / mutation / merged-PR application | `ce/unit/test_reconciliation.py` (`test_github_pr_plan_builds_generic_mutations`, `test_apply_reconciliation_plan_applies_generic_pr_mutations`, `test_ingest_merged_pull_request_applies_through_context_graph`) |
| PR / commit episode formatting | `host/unit/context_graph/test_episode_formatters.py` |
| Review-thread grouping into discussions | `host/unit/context_graph/test_review_thread_grouper.py` |
| PR-aware extraction edges | `ce/unit/test_extraction_edges.py`, `host/unit/context_graph/test_extraction_edges.py` |

## Feature 5 — Raw Context Ingestion

`POST /api/v2/context/ingest` for notes, docs, links, fixes, decisions, preferences, etc.

| Capability | Tests |
| --- | --- |
| Episode ingest and CLI argument resolution | `ce/unit/test_ingest_episode.py`, `ce/unit/test_ingest_args.py` |
| HTTP 422 / rejection rendering | `ce/integration/test_ingest_http_422.py`, `ce/unit/test_ingest_rejection_render.py` |
| Sync vs queued receipt (episode_uuid + job_id) | `ce/unit/test_run_raw_episode_ingestion.py`, `ce/unit/test_wait_ingestion_event.py` |

## Feature 6 — Durable Event And Reconciliation Pipeline

The event ledger plus `validate_reconciliation_plan` and `apply_reconciliation_plan`.

| Capability | Tests |
| --- | --- |
| Reconcile + replay event paths | `ce/unit/test_reconciliation.py` |
| Reconciliation issues / flags surfaced to callers | `ce/unit/test_reconciliation_issues.py`, `host/unit/context_graph/test_reconciliation_issues.py`, `host/unit/context_graph/test_reconciliation_flags.py` |
| Validation edge cases | `ce/unit/test_reconciliation_validation_edge_cases.py` |
| Entity canonicalization (key normalization, synonym merge, edge rewrite, dedupe) | `ce/unit/test_entity_canonicalization.py`, `host/unit/context_graph/test_entity_canonicalization.py` |
| Label inference and soft-ontology downgrade | `ce/unit/test_label_inference.py`, `ce/unit/test_soft_downgrade.py`, `host/unit/context_graph/test_canonical_label_inference.py` |
| Deterministic extractors | `ce/unit/test_extractors.py`, `host/unit/context_graph/test_deterministic_extractors.py` |
| Deprecated `events/ingest` alias signalling + counter | `ce/unit/test_events_ingest_alias.py` |
| Apply-step provenance contract | `ce/unit/test_apply_episode_provenance.py`, `ce/unit/test_provenance_contract.py` |
| Ontology catalog, lifecycle, classifier passes | `ce/unit/test_ontology.py`, `ce/unit/test_ontology_lifecycle.py`, `ce/unit/test_ontology_classifier.py`, `ce/unit/test_ontology_classifier_pass.py`, `host/unit/context_graph/test_ontology_classifier.py`, `host/unit/context_graph/test_ontology_helpers.py` |

## Feature 7 — Unified Graph Querying (`POST /query/context-graph`)

One read API serving semantic, exact, traversal, temporal, aggregate, and answer goals.

| Capability | Tests |
| --- | --- |
| Adapter routing (semantic / timeline / sync-answer guards / preset compilation) | `ce/unit/test_context_graph_query.py` |
| Graph writer mutations | `ce/unit/test_context_graph_writer.py` |
| Query planner | `ce/unit/test_graph_query_planner.py` |
| Context resolution bundles (intent / scope / mode / source_policy / budget; PR review include set; debugging memory; verify reporting) | `ce/unit/test_context_resolution.py` |
| Temporal / supersession reads | `ce/unit/test_temporal_search.py`, `ce/integration/test_temporal_supersede.py` |
| Causal multi-hop and merge bridges | `ce/unit/test_causal_search_merge.py`, `ce/unit/test_causal_structural_window.py`, `ce/integration/test_causal_multihop.py` |
| Semantic-fallback bridge for empty structural reads | `ce/unit/test_semantic_fallback_bridge.py` |
| Search and provenance scoping | `ce/unit/test_search_provenance.py`, `ce/unit/test_context_events_scope.py` |
| Answer synthesis (LLM summary path with `meta.answer_summary_source`) | `ce/unit/test_answer_synthesis.py` |
| Edge collapse golden snapshot | `ce/unit/test_edge_collapse_golden.py` |

## Feature 8 — Agent Memory Recording (`POST /record`) and Agent Surface

| Capability | Tests |
| --- | --- |
| Agent context port: manifest, recipes, intent fallback, copy-not-share | `ce/unit/test_agent_context_port.py`, `host/unit/context_graph/test_agent_context_port.py` |
| MCP surface contract: exactly the 4 tools, recipes are `context_resolve` shapes, no private ingest helper | `ce/unit/test_agent_surface_contract.py` |
| Context-aware ingestion agent | `ce/unit/test_context_aware_ingestion_agent.py` |
| Intelligence policy (scope merging, capability gating, mode/budget, signals passthrough) | `ce/unit/test_intelligence_policy.py`, `host/unit/context_graph/test_intelligence_policy.py` |
| Intelligence signal extraction (PR / file path / symbol / intent / raw query) | `host/unit/context_graph/test_intelligence_signals.py` |
| Chat-agent context wiring (host integration of `context_resolve`) | `host/unit/intelligence/test_chat_agent_context.py` |

## Feature 9 — Status, Quality, Readiness (`POST /status`)

| Capability | Tests |
| --- | --- |
| Status payload shaping: resolver capabilities, source serialization, ledger health, derived `last_success_at` | `ce/unit/test_context_status.py`, `host/unit/context_graph/test_context_status.py` |
| Graph quality: staleness flags, source-access degradation, source-type-aware family policies, family-conflict detection (contradiction / supersession / chose-vs-migrated) | `ce/unit/test_graph_quality.py`, `host/unit/context_graph/test_graph_quality.py` |

## Feature 10 — Conflict Management

| Capability | Tests |
| --- | --- |
| Family conflict detection across reconciliation output | `ce/integration/test_family_conflict_detection.py` |
| Edge collapse golden behavior used by conflict resolution | `ce/unit/test_edge_collapse_golden.py` |
| Quality-issue surfacing in status / resolve responses | `ce/unit/test_graph_quality.py` |

## Feature 11 — Maintenance and Operator Surface

| Capability | Tests |
| --- | --- |
| `classify-modified-edges` (dry-run by default; double env-flag write gate) | `ce/integration/test_classify_modified_edges.py` |
| Hard-reset pot graph | `ce/unit/test_hard_reset_pot.py` |
| Operator audit logger record fields | `ce/unit/test_operator_audit.py` |

## Cross-cutting

| Concern | Tests |
| --- | --- |
| HTTP API client (used by CLI / SDK) | `ce/unit/test_potpie_context_api_client.py` |
| MCP project-access guards | `ce/unit/test_mcp_project_access.py` |
| CLI output rendering and skill commands | `ce/unit/test_cli_output.py`, `ce/unit/test_cli_skill_commands.py` |
| CLI environment bootstrap | `ce/unit/test_env_bootstrap.py` |
| Agent installer (AGENTS.md / SKILL.md scaffold) | `ce/unit/test_agent_installer.py` |
| Bootstrap container wiring | `ce/unit/test_bootstrap_container.py` |
| Postgres session lifecycle | `ce/unit/test_postgres_session.py` |
| Benchmark harness (dataset / evaluator / runner) | `ce/unit/benchmarks/test_benchmark_dataset.py`, `ce/unit/benchmarks/test_benchmark_evaluator.py`, `ce/unit/benchmarks/test_benchmark_runner.py` |

## Gaps and Planned Work

Items below are listed in the product doc but **do not yet have a dedicated automated test**. They are tracked here so reviewers can see what is intentionally unfinished and what should be addressed in follow-up MRs.

### Endpoint coverage not yet exercised end-to-end

- `POST /api/v2/context/conflicts/list` and `POST /api/v2/context/conflicts/resolve` — only the underlying detection / quality pieces are tested today; the operator HTTP routes (auth, audit fields, dry-run negotiation) need their own integration test.
- `POST /api/v2/context/reset` — operator HTTP path / audit fields.
- `POST /api/v2/context/events/replay` — happy path and idempotency.
- `POST /api/v2/context/pots/{pot_id}/ingest/raw` — UI raw ingest path beyond what the agent CLI exercises.
- `GET /api/v2/context/events/{event_id}` and `GET /api/v2/context/pots/{pot_id}/events` — event inspection responses (pagination + `source_channel` / `source_system` columns).

### Source policy maturity

- `source_policy=summary` with a real resolver (currently exercised mostly via fallback paths).
- `source_policy=snippets` budget clamping (`max_chars_per_item`, `max_total_chars`, `max_snippets_per_ref`) end-to-end.
- `source_policy=verify` updating verification state once a resolver is wired.

### Status / readiness deepening

- Coverage and freshness numbers driven by the live source-resolver capability matrix.
- Recommended-recipe selection for less common intents (`planning`, `refactor`, `test`, `security`).

### Cross-source reconciliation

- Multi-source episodes that mix GitHub, Linear, and raw notes inside a single reconciliation run.
- Synonym table extension paths (`SYNONYMS`) — covered structurally but no fixture for cross-spelling merges across sources.

### CLI / MCP parity

- `potpie status`, `potpie resolve`, `potpie overview`, `potpie record` parity with the MCP `context_*` tools beyond unit coverage of their argument plumbing.
- MCP tool descriptions / manifest exposure tested through an actual MCP transport.

### Authorization and tenancy

- Pot membership / role enforcement at the HTTP layer for every mutating route (currently covered at the role-matrix layer).
- `INTERNAL_ADMIN_SECRET` impersonation path with `X-User-Id` (covered indirectly through `api_key_deps.py`; deserves a dedicated test).

### Operator audit coverage

- `context_engine.operator_audit` record assertions for every destructive route, not just hard-reset and classify-modified-edges.

### Compatibility shims (slated for removal)

- `POST /api/v2/context/events/ingest` deprecation alias has alias / counter tests; **plan**: remove the route after known clients migrate, then delete `ce/unit/test_events_ingest_alias.py`.
- Repository-routing endpoints under `/api/v2/context/pots/{pot_id}/repositories*` are transitional and should be deleted (along with their mirroring logic) once the source-first migration is complete.

## Update protocol

When you add a feature or test:

1. Add the test path to the right row above. Prefer the abbreviation prefixes.
2. If the change closes a gap, remove the entry from "Gaps and Planned Work".
3. If the change introduces new surface area, add a new row rather than overloading an existing one.
4. Keep this doc in lockstep with [`features-and-functionalities.md`](features-and-functionalities.md): if a feature is added there, it must appear here, even if only as a "no tests yet" entry under "Gaps and Planned Work".
