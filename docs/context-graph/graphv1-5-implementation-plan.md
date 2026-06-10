# Graph V1.5 Implementation Plan

Last reviewed: 2026-06-09.

This plan implements the in-between graph surface before the full Graph V2
workbench. The goal is to give harness agents a V2-shaped way to discover,
query, and directly mutate the graph now, while keeping the existing V1
`context_*` and top-level CLI wrappers working.

The in-between surface is **Graph Surface Lite**:

```bash
potpie graph catalog
potpie graph read
potpie graph search-entities
potpie graph mutate
```

Deferred until the full workbench:

```bash
potpie graph describe
potpie graph propose
potpie graph commit
potpie graph history
potpie graph inbox
```

Deferring `inbox` (and the plan store behind `propose`/`commit`) means V1.5
persists nothing un-committed. There is no server-side pending queue: a
`review_required` result is returned to the harness, which retains the proposed
mutation and either re-submits it with approval or drops it. Ambiguous findings
become low-authority `agent_claim`s or observations, never a Potpie-held inbox
item. The inbox / maintainer-handoff workflow arrives with V2.

`graph mutate` is direct apply, but it is not raw graph CRUD. It accepts semantic
mutation operations, validates schema/ontology/evidence/risk, lowers to today's
`ReconciliationPlan`, and applies through `GraphMutationPort`. Later,
`graph mutate --dry-run` becomes `graph propose`, and direct `graph mutate`
becomes `propose + auto-commit` as a convenience wrapper.

## Implementation Review Findings (2026-06-09)

Status: uncommitted implementation reviewed from the beginning of this plan
through the current V1.5 work. Treat this section as the fix queue. Each item
should be closed with either a targeted code/test change or an explicit scope
decision recorded in this document.

| ID | Area | Severity | Status |
|---|---|---:|---|
| IR-01 | Step 4 semantic validator holes | High | Fixed in working tree |
| IR-02 | Step 5/6 provenance drop and Step 5a rename incomplete | High | Fixed in working tree |
| IR-03 | Step 6a read surface not assembled | High | Fixed in working tree |
| IR-04 | Step 6a infra environment filtering bug | High | Fixed in working tree |
| IR-05 | Step 7 CLI contract gaps | Medium | Fixed in working tree |
| IR-06 | Step 8 local `context_record` validation bypass | High | Fixed in working tree |
| IR-07 | Step 9 Neo4j/FalkorDB metadata and ANN incomplete | High | Fixed in working tree |
| IR-08 | Step 10/11 deterministic ingest paths bypass harness | High | Closed by scope removal |
| IR-09 | Step 12a nudge event spelling mismatch | Medium | Fixed in working tree |
| IR-10 | Step 13 template test failure | Medium | Open |
| IR-11 | R1/R2 retrieval card and backend gap | Medium | Open |
| IR-12 | R5/R6 retriever consolidation not implemented | Medium | Open |

### IR-01: Step 4 semantic validator holes

Problem:

- `append_event` validates verb, timestamp, and description, but it does not
  validate entity references in `subject`, `actor`, `targets`, or `mentions`.
  A probe with a bad actor such as `{key: "repo:not-person", type: "Person"}`
  and a target such as `{key: "service:x", type: "Nope"}` validated
  successfully.
- `end_relation_validity` accepts `subject + predicate` without an `object`.
  The validator allows it, and the lowerer falls back to broad entity
  invalidation. A probe lowered this into an invalidation for
  `("service:payments-api", None)`, which can retire more than the intended
  relation.
- Review-required operations such as `supersede_claim` skip endpoint and entity
  validation. A probe with garbage references returned `review_required` with no
  validation issues.

Evidence:

- `potpie/context-engine/application/services/semantic_mutation_validator.py:207`
- `potpie/context-engine/application/services/semantic_mutation_validator.py:291`
- `potpie/context-engine/application/services/semantic_mutation_validator.py:305`
- `potpie/context-engine/application/services/semantic_mutation_lowering.py:267`

Close criteria:

- [x] Validate all entity references in event payloads before lowering.
- [x] Require `end_relation_validity` to target an exact `claim_key`, edge, or
      `(subject, predicate, object)` relation unless broad invalidation is
      explicitly requested and approved.
- [x] Run the same shape and endpoint validation for `review_required` ops before
      returning the review decision.
- [x] Add regression tests for invalid event actor/target refs, missing
      `end_relation_validity.object`, and malformed `supersede_claim` refs.

### IR-02: Step 5/6 provenance drop and Step 5a rename incomplete

Problem:

- The semantic lowerer creates `plan.provenance`, but
  `DefaultGraphService.mutate` applies only `plan.batch`. It does not pass
  `provenance_context=plan.provenance` into the mutation port, so applied
  mutations can lose the event/provenance context that Step 6 depends on.
- Step 5a is only partially reflected in the implementation. Live structural
  write types and older deterministic write paths still expose `ReconciliationPlan` and
  `ReconciliationResult` naming even though this tier is now a deterministic
  mutation batch, not LLM reconciliation.
- Non-event structural writes still risk carrying fabricated event provenance
  because the provenance boundary is not cleanly named.

Evidence:

- `potpie/context-engine/application/services/graph_service.py:344`
- `potpie/context-engine/domain/reconciliation.py:70`
- `potpie/context-engine/application/services/ingest_service.py:65`

Close criteria:

- [x] Pass `provenance_context=plan.provenance` when applying lowered semantic
      mutations.
- [x] Rename the deterministic structural write DTOs and service plumbing to
      `MutationBatch` / `MutationResult` or add aliases with a clear migration
      plan and no new code using reconciliation names.
- [x] Keep LLM reconciliation names only in the optional agentic reconciliation
      path.
- [x] Add a mutate-path test that verifies provenance reaches the mutation port.

Fix applied:

- `DefaultGraphService.mutate()` now passes the semantic lowerer's
  `plan.provenance` into `GraphMutationPort.apply(...)`, so actor/source context
  reaches the write boundary.
- `ProvenanceContext` now carries optional `source_event_id` and
  `source_system`, letting non-event writes preserve source provenance without
  fabricating an `EventRef`.
- Active graph mutation plumbing now uses `MutationBatch` / `MutationResult`
  names directly. Back-compat aliases (`ReconciliationPlan`,
  `ReconciliationResult`, `apply_reconciliation_plan`) remain only as migration
  shims for older callers/tests.
- Regression coverage added for semantic mutate provenance forwarding and
  no-`EventRef` provenance stamping in `apply_mutation_batch`.

### IR-03: Step 6a read surface not assembled

Problem:

- `GraphViewSpec` declares `inline_relations` and `ranking_inputs`, but the
  catalog output currently omits `ranking_inputs`.
- `GraphService.read` records `inline_relations` only in response metadata.
  It does not assemble V2-style per-entity relation payloads.
- The CLI read payload still emits flat `items`, and concrete readers return
  flat claim rows. The one-call UC2/UC4 shape described by Step 6a is therefore
  not met: callers still have to join or infer related entities themselves.

Evidence:

- `potpie/context-engine/domain/graph_views.py:58`
- `potpie/context-engine/application/services/graph_service.py:198`
- `potpie/context-engine/adapters/inbound/cli/commands/graph.py:361`
- `potpie/context-engine/application/readers/prior_bugs.py:45`
- `potpie/context-engine/application/readers/infra_topology.py:50`

Close criteria:

- [x] Include `ranking_inputs` in the catalog contract.
- [x] Either return assembled entity payloads with `relations[]` for views that
      advertise `inline_relations`, or explicitly mark inline relation assembly
      deferred and adjust this plan/templates/tests accordingly.
- [x] Add at least one contract test for a graph read response that proves
      inline relation metadata is actionable, not just descriptive.

Resolution:

- `GraphViewSpec.to_catalog_entry()` now includes `ranking_inputs`.
- `DefaultGraphService.read()` assembles graph-read results for views that
  declare `inline_relations` into entity payloads with `relations[]`; the flat
  claim envelope remains unchanged for `context_resolve` / `context_search`.
- Contract coverage now asserts both catalog `ranking_inputs` and an
  `infra_topology.service_neighborhood` read with an actionable
  `DEPENDS_ON` inline relation.

### IR-04: Step 6a infra environment filtering bug

Problem:

- `InfraTopologyReader.read` filters out rows only when a row has a string
  environment that differs from the requested environment. Rows with no
  environment property pass an environment-filtered query.
- A probe with an unqualified edge
  `service:ledger-api DEPENDS_ON service:queue` plus a staging edge returned
  `("service:queue", None)` from a `environment="prod"` query.
- This weakens environment-scoped topology reads and can leak generic or
  ambiguous dependencies into production-specific answers.

Evidence:

- `potpie/context-engine/application/readers/infra_topology.py:53`

Close criteria:

- [x] For environment-filtered reads, exclude rows with missing or different
      environment unless the caller explicitly asks for unqualified edges.
- [x] Decide whether unqualified topology edges are allowed in V1.5 and document
      the rule in the view contract.
- [x] Add tests for prod, staging, and unqualified dependency rows.

Resolution:

- `InfraTopologyReader` now normalizes environment filters and excludes rows
  whose environment qualifier is missing or different by default.
- The same environment filter is applied during BFS expansion, not only after
  traversal, so unqualified or staging edges cannot pull unrelated prod rows into
  an environment-scoped answer.
- V1.5 allows unqualified topology edges only when the caller explicitly sets
  `include_unqualified_environment=true` in the read scope. The
  `infra_topology.service_neighborhood` view contract now documents that rule.
- Regression coverage now covers prod/staging rows, unqualified opt-in, and
  traversal leakage through unqualified or wrong-environment rows.

### IR-05: Step 7 CLI contract gaps

Problem:

- `graph catalog --subgraph` is accepted but ignored. A probe of
  `catalog(subgraph="bugs")` returned all eight views.
- `graph mutate` emits rejected/error JSON for semantic validation failures, but
  the CLI path appears to keep a successful process exit for those failures.
  Parse errors fail, semantic rejections do not.
- `_parse_scope` silently drops malformed `--scope` pairs instead of failing or
  surfacing a warning.

Evidence:

- `potpie/context-engine/application/services/graph_service.py:159`
- `potpie/context-engine/adapters/inbound/cli/commands/graph.py:31`
- `potpie/context-engine/adapters/inbound/cli/commands/graph.py:152`
- `potpie/context-engine/adapters/inbound/cli/commands/graph.py:331`

Close criteria:

- [x] Make `catalog --subgraph` filter the catalog or remove the option until it
      is implemented.
- [x] Return non-zero exit status for semantic validation rejection and mutation
      errors.
- [x] Treat malformed `--scope` values as input errors, or document and test the
      ignore behavior if it is intentional.

Resolution:

- `DefaultGraphService.catalog()` now filters views by requested subgraph and
  rejects unknown subgraphs with a validation error.
- `graph mutate` preserves the rejected/error result JSON shape but exits with
  validation status `1` whenever `SemanticMutationResult.ok` is false.
- Graph CLI `--scope` parsing now rejects malformed entries instead of silently
  dropping them.
- Regression coverage added for catalog subgraph filtering, non-zero mutation
  rejection exit, and malformed scope rejection before service dispatch.

### IR-06: Step 8 local `context_record` validation bypass

Problem:

- `record_to_semantic_request` catches `ContextRecordValidationError` and
  continues, so invalid structured payloads silently downgrade into generic
  records and still write claims.
- A probe with `bug_pattern` and `details={"kind": 123}` was accepted and
  recorded a claim.
- The managed HTTP `record_durable_context` path validates more strictly, so
  local HostShell behavior diverges from managed behavior.

Evidence:

- `potpie/context-engine/application/services/record_to_semantic.py:48`

Close criteria:

- [x] Preserve summary-only legacy records, but reject malformed structured
      `details` for known record types.
- [x] Remove the structured downgrade path for schema-shaped known records so
      malformed structured details cannot silently claim acceptance.
- [x] Add local HostShell tests that cover malformed known record details and a
      valid legacy summary-only record.

Resolution:

- Local `record_to_semantic_request()` now preserves legacy summary-only records
  but validates schema-shaped structured details for known record types instead
  of swallowing `ContextRecordValidationError`.
- Schema detection is based on each structured record type's declared detail
  fields, so generic adapter metadata such as `confidence`, `visibility`, and
  `text` does not force strict structured validation.
- Malformed structured details now reject before any graph write; there is no
  structured downgrade path left for schema-shaped known records.
- Regression coverage now includes the exact malformed `bug_pattern`
  `details={"kind": 123}` HostShell path and a valid summary-only known record.

### IR-07: Step 9 Neo4j/FalkorDB metadata and ANN incomplete

Resolution:

- `ClaimRow` V1.5 metadata is now first-class on Neo4j, FalkorDB,
  in-memory, and embedded backends.
- Neo4j/FalkorDB share the canonical row parser, which lifts `claim_key`,
  `subgraph`, `truth`, `confidence`, `description`, `environment`,
  `observed_at`, `valid_until`, `mutation_id`, `source_refs`, `evidence`,
  `graph_contract_version`, and `ontology_version` out of relationship
  properties into typed `ClaimRow` fields.
- Contract metadata is no longer used as a legacy `properties` fallback; the
  reader extras bag is limited to non-contract extras and computed annotations.
- `evidence_strength` is derived from V1.5 `truth` for ranking instead of being
  treated as source metadata.
- Neo4j and FalkorDB semantic reads use native relationship vector indexes when
  an embedder is configured, with explicit lexical fallback coverage.

Evidence:

- `potpie/context-engine/adapters/outbound/graph/neo4j_reader.py:13`
- `potpie/context-engine/adapters/outbound/graph/neo4j_reader.py:76`
- `potpie/context-engine/adapters/outbound/graph/falkordb_reader.py:75`

Close criteria:

- [x] Parse all V1.5 claim metadata from Neo4j relationship properties into
      `ClaimRow`.
- [x] Parse the same V1.5 metadata from FalkorDB rows into `ClaimRow`.
- [x] Wire Neo4j semantic reads to the relationship vector index when embeddings
      are configured.
- [x] Label lexical fallback behavior explicitly and cover both vector and
      fallback paths in tests.

### IR-08: Step 10/11 deterministic ingest paths bypass harness

Problem:

- Local code/config ingestion and timeline update paths created graph updates without
  harness judgment.
- The former Event Ledger apply mode also let local event pulls write graph claims through a
  built-in reconciler.
- Managed HTTP `/record` is no longer part of this local V1.5 scope.
- These paths conflict with the principle that intelligence and durable write
  decisions live in the harness.

Evidence:

- Removed the local ingest CLI group and local ingest services.
- Removed local code/config ingestion adapters, registry, and port.
- Removed automatic timeline update service and CLI commands.
- Removed Event Ledger apply mode and the built-in event reconciler.

Close criteria:

- [x] No local code/config ingestion command remains in the CLI.
- [x] No local automatic timeline update command remains in the CLI.
- [x] Event Ledger pull is read-only and cannot apply graph mutations.
- [x] Harness-facing `context_record` / `graph mutate` remain the canonical
      durable write path.

### IR-09: Step 12a nudge event spelling mismatch

Problem:

- This plan says `graph nudge --event pre-edit`, but the implementation accepts
  `pre_edit`.
- CLI help advertises underscore spelling, while the plan uses dash spelling.
- A probe showed `pre_edit` works and `pre-edit` returns
  `{ok: false, silent: true, detail: "unknown event"}`.

Resolution:

- Keep underscore event names as the canonical internal contract.
- Accept dash aliases at the nudge service boundary, so `pre-edit` canonicalizes
  to `pre_edit` before policy lookup.
- Update CLI help to document that dash aliases are accepted.
- Update the bundled hook adapter's direct-event passthrough to canonicalize dash
  aliases too.

Evidence:

- `potpie/context-engine/domain/nudge.py`
- `potpie/context-engine/adapters/inbound/cli/commands/graph.py`
- `potpie/context-engine/application/services/nudge_service.py`
- `potpie/context-engine/adapters/inbound/cli/templates/claude_plugin/hooks/potpie_nudge.py`
- `potpie/context-engine/tests/unit/test_nudge_service.py`
- `potpie/context-engine/tests/unit/test_nudge_adapter.py`
- `potpie/context-engine/tests/unit/test_graph_cli_contract.py`

Close criteria:

- [x] Accept both `pre-edit` and `pre_edit` aliases, or update all docs, plans,
      templates, and tests to use one spelling.
- [x] Add a regression test for whichever spelling contract is chosen.

### IR-10: Step 13 template test failure

Problem:

- A focused test run produced `74 passed, 1 failed`.
- Failing command:
  `uv run pytest tests/unit/test_agent_templates_v15.py tests/unit/test_nudge_adapter.py tests/unit/test_retrieval_eval.py tests/unit/test_ranking.py tests/unit/test_scope_match.py`
- Failure: `test_no_stale_include_names_anywhere` flags `recent_changes` inside
  `recent_changes.timeline`. That appears to be a valid view prefix, but not a
  valid include name, so the test may be over-broad unless the template is using
  it as an include token.

Evidence:

- `potpie/context-engine/tests/unit/test_agent_templates_v15.py:58`
- `potpie/context-engine/adapters/inbound/cli/templates/agent_bundle/.agents/skills/potpie-graph/SKILL.md:42`

Close criteria:

- [x] Inspect the template usage and decide whether `recent_changes.timeline` is
      a valid view reference or a stale include leak.
- [x] If it is valid, narrow the stale-include test so it checks include tokens,
      not arbitrary view names.
- [x] Keep `recent_changes.timeline` as a valid view reference; update templates
      only where examples implied repo scope was required.
- [x] Re-run the focused test command and require it to pass.

### IR-11: R1/R2 retrieval card and backend gap

Problem:

- In-memory and embedded paths have embed-on-write behavior and a local hashing
  embedder, but Neo4j/FalkorDB still rely on token-overlap style matching.
- `build_retrieval_card` includes description, fact, subject, predicate, object,
  and scope, but it does not include arbitrary structured retrieval fields such
  as `prescription`, `symptom_signature`, or `fix_steps` unless an agent repeats
  those fields in `description`.
- The lowerer copies only limited structured scope data such as `code_scope` into
  edge properties, so structured fields can be lost before retrieval scoring.

Evidence:

- `potpie/context-engine/domain/retrieval_card.py:28`
- `potpie/context-engine/application/services/semantic_mutation_lowering.py:333`
- `potpie/context-engine/application/services/semantic_mutation_lowering.py:388`

Close criteria:

- [ ] Decide which structured fields are first-class retrieval text in V1.5.
- [ ] Ensure the lowerer preserves those fields on claim properties.
- [ ] Ensure the read/eval path builds the same retrieval card that write-time
      embedding uses.
- [ ] Implement or explicitly defer the Neo4j ANN path from R1.

### IR-12: R5/R6 retriever consolidation not implemented

Problem:

- Four concrete reader classes still remain:
  `CodingPreferencesReader`, `InfraTopologyReader`, `TimelineReader`, and
  `PriorBugsReader`.
- `ReadOrchestrator` fans out to per-include readers, and `EnvelopeBuilder`
  concatenates and sorts independent outputs.
- This is not yet one ANN/retriever pass with per-family caps and a global
  relevance floor, so R5/R6 are not implemented even if the V1-style read path
  still functions.

Evidence:

- `potpie/context-engine/application/services/read_orchestrator.py:56`
- `potpie/context-engine/application/services/envelope_builder.py:71`

Close criteria:

- [ ] Decide whether R5/R6 are required for V1.5 completion.
- [ ] If yes, add the shared `ClaimRetriever` path and migrate use-case readers
      onto parameterized view specs.
- [ ] If no, mark R5/R6 deferred in this plan and remove them from V1.5 done
      criteria.

## Non-Negotiables

- Agents never touch the physical store. Writes go through semantic mutations
  lowered to `GraphMutationPort`; reads go through the typed Query Surface
  (retrieve / filter / traverse — see the Query Surface section). The prohibition
  is on exposing the *store schema and query language* (`EntityUpsert`,
  `EdgeUpsert`, Cypher, SQL), not on query expressiveness — expressiveness comes
  from the three typed axes. A raw-Cypher escape hatch may exist for operators
  only, off the agent tool surface.
- Keep all durable writes behind semantic validation and `GraphMutationPort`.
- Keep intelligence in harness skills. Potpie validates, lowers, commits, and
  audits.
- Make ontology keys, truth classes, evidence, claim metadata, and read views
  V2-compatible before adding richer write workflows.
- Keep MCP `context_*` tools unchanged for now. Tests currently pin exactly four
  MCP tools. This means the new `graph catalog/read/search-entities/mutate`
  surface is CLI-only in V1.5; MCP parity is a deliberate later step, not part of
  this milestone.
- Implement local CLI first through `HostShell`. Managed HTTP can follow once the
  local contract is stable.

## Current Code Map

| Area | Current Code | What It Means |
|---|---|---|
| Host facade | `potpie/context-engine/host/shell.py` | CLI and MCP bind through `HostShell`. Add Graph Surface Lite behind `host.graph`, not in adapters. |
| Local composition | `potpie/context-engine/bootstrap/host_wiring.py` | Wires `DefaultGraphService`, `AgentContextService`, backends, and read-only ledger access. |
| CLI graph group | `potpie/context-engine/adapters/inbound/cli/commands/graph.py` | Already has `status`, `inspect`, `export`, `import`, `repair`. Add `catalog`, `read`, `search-entities`, `mutate` here. |
| V1 wrappers | `potpie/context-engine/adapters/inbound/cli/commands/query.py`, `adapters/inbound/mcp/server.py` | Keep `resolve`, `search`, `record`, `context_*` as compatibility wrappers. |
| Graph data plane | `potpie/context-engine/application/services/graph_service.py` | `record()` currently lowers directly to `ReconciliationPlan`; this is the main V1.5 write-path change. |
| Graph service port | `potpie/context-engine/domain/ports/services/graph_service.py` | Extend this protocol with Graph Surface Lite DTOs/methods. |
| Read trunk | `potpie/context-engine/application/services/read_orchestrator.py` | Existing `intent/include` read path. `graph read` should map V2-style views onto this first. |
| Include vocabulary | `potpie/context-engine/domain/agent_context_port.py` | Current V1 include contract and four-tool manifest. Add separate view map instead of overloading this file. |
| Ontology | `potpie/context-engine/domain/ontology.py` | Good declarative base. Needs V1.5/V2 key patterns, truth classes, claim fields, and catalog output. |
| Structured records | `potpie/context-engine/domain/context_records.py` | Useful record payload validators. Reuse them when `context_record` lowers to semantic mutations. |
| Current write DTO | `potpie/context-engine/domain/reconciliation.py` | Keep as the backend-lowering format, but rename `ReconciliationPlan` → `MutationBatch` (Step 5a) — it reconciles nothing, it is a mutation batch. Move the vestigial agentic types (`ReconciliationRequest`, `EvidenceRef`) out with the LLM path (Step 11). |
| Mutation primitives | `potpie/context-engine/domain/graph_mutations.py` | Existing structural primitives plus provenance context. Add semantic mutation DTOs separately. |
| Validation | `potpie/context-engine/application/services/reconciliation_validation.py` | Structural validation exists. Add semantic validation before lowering, then keep structural validation as a backstop. |
| Backend write port | `potpie/context-engine/domain/ports/graph/mutation.py` | Canonical backend write door. Do not widen it for agent-facing operations. |
| Backend read port | `potpie/context-engine/domain/ports/claim_query.py` | Current `ClaimRow` lacks first-class V1.5 claim fields. Add optional fields and keep properties backward-compatible. |
| Local backends | `adapters/outbound/graph/backends/in_memory_backend.py`, `embedded_backend.py` | Conformance reference and local default. Update these first. |
| Neo4j writer/reader | `adapters/outbound/graph/cypher.py`, `neo4j_reader.py` | Already store flexible edge properties. Add V1.5 claim metadata to properties and parse them back. |
| Use-case readers | `application/readers/{prior_bugs,coding_preferences,infra_topology,timeline_reader}.py` | ~80% identical boilerplate over `find_claims`. Collapse into one parameterized retriever (R5). |
| Ranker | `domain/ranking.py` | Weighted geometric mean with a `1e-6` floor; lets a zero semantic score veto a strong candidate. Split hard filters from soft scores (R3). |
| Semantic match (stub) | `adapters/outbound/graph/{neo4j_reader,in_memory_reader,falkordb_reader}.py` | `_embedding_score` is Jaccard token overlap, not vectors. Replace with embed-on-write + ANN (R1). |
| Vector index (unused) | `adapters/outbound/graph/cypher.py` | `claim_fact_embeddings` index exists but `fact_embedding` is never written or queried. Wire it (R1). |
| Semantic port | `domain/ports/graph/semantic.py` | `SemanticSearchPort` shape is right; needs a real embedder-backed implementation. |
| Service-side reconciliation | `adapters/outbound/reconciliation/*`, `application/use_cases/process_batch.py` | Existing LLM reconciliation agent conflicts with the V1.5 principle. Park it as opt-in, not canonical. |
| Tests | `tests/conformance/*`, `tests/unit/test_agent_surface_contract.py`, `test_record_types.py`, `test_reconciliation_validation_edge_cases.py` | Add Graph Surface Lite tests without breaking current four-tool MCP tests. |

## Target Surface Contract

### `potpie graph catalog`

Purpose: let agents discover the graph contract without loading docs.

`catalog` accepts an optional `--task "..."` argument now, but in V1.5 it returns
the same static contract regardless. V2 turns `--task` into a subgraph/view
ranker; accepting (and ignoring) it now keeps that change additive instead of a
signature break for skills already calling `catalog`.

Minimum JSON:

```json
{
  "ok": true,
  "graph_contract_version": "v1.5",
  "ontology_version": "2026-06-graph",
  "commands": ["catalog", "read", "search-entities", "mutate"],
  "truth_classes": ["authoritative_fact", "source_observation", "agent_claim", "user_decision", "preference", "timeline_event", "quality_finding"],
  "views": [
    {
      "name": "bugs.prior_occurrences",
      "v1_include": "prior_bugs",
      "backed": true
    }
  ],
  "mutation_operations": ["append_event", "upsert_entity", "link_entities", "assert_claim", "end_relation_validity", "retract_claim"],
  "review_required_operations": ["supersede_claim", "merge_duplicate_entities"],
  "deferred_operations": ["patch_entity", "transition_state"],
  "entity_types": [],
  "predicates": []
}
```

`mutation_operations` lists only what V1.5 can actually apply.
`supersede_claim` and `merge_duplicate_entities` are surfaced separately as
`review_required_operations`: they always return `review_required` and have no
V1.5 approval path (no plan store, no identity resolver), so advertising them as
applicable would lie. `patch_entity` and `transition_state` are part of the V2
vocabulary but `deferred_operations` here — V1.5 models state changes as new
claims/events, not in-place edits. This keeps the catalog honest about coverage
rather than advertising dead ends.

### `potpie graph read`

Purpose: V2-style read request over the existing read trunk.

Example:

```bash
potpie --json graph read --view bugs.prior_occurrences --query "refund race" --pot local/default
potpie --json graph read --view recent_changes.timeline --time-window 7d --limit 20
potpie --json timeline recent --time-window 7d --limit 20
```

V1.5 implementation maps `bugs.prior_occurrences` to `include=prior_bugs` and
returns the same envelope shape as `context_resolve`, plus graph contract
metadata. Timeline reads default to deduped event rows sorted by occurrence time;
`--format raw` exposes the underlying relation payload.

The read envelope must also carry a `subgraph_versions` block (a single
monotonic counter or a per-subgraph stub is fine in V1.5). V2's optimistic
concurrency — `propose`/`commit` against `expected_subgraph_versions` — depends
on reads exposing versions. Reserving the field now keeps that change additive
rather than a read-contract break later.

### `potpie graph search-entities`

Purpose: narrow entity/claim lookup for identity resolution before writes.

Example:

```bash
potpie graph search-entities "payments api" --type Service --limit 10 --json
```

V1.5 can implement this with the current semantic/claim query projections:
search matching claim facts, collect subject/object entities, include labels and
supporting claims. Do not promise a perfect entity index yet.

### `potpie graph mutate`

Purpose: validate and directly apply semantic graph mutations.

Command:

```bash
potpie graph mutate --file mutation.json --json
potpie graph mutate --file mutation.json --dry-run --json
```

Canonical payload shape should be batch-shaped even when there is one operation:

```json
{
  "graph_contract_version": "v1.5",
  "pot_id": "local/default",
  "idempotency_key": "mutation:preference:pytest-fixtures",
  "created_by": {
    "surface": "cli",
    "harness": "codex"
  },
  "operations": [
    {
      "op": "link_entities",
      "subgraph": "infra_topology",
      "subject": {
        "key": "service:local:payments-api",
        "type": "Service",
        "properties": {
          "name": "payments-api"
        }
      },
      "predicate": "DEPENDS_ON",
      "object": {
        "key": "service:local:ledger-api",
        "type": "Service",
        "properties": {
          "name": "ledger-api"
        }
      },
      "truth": "authoritative_fact",
      "confidence": 0.95,
      "evidence": [
        {
          "source_ref": "github:pr:412",
          "authority": "external_system"
        }
      ],
      "valid_from": "2026-06-08T00:00:00+05:30"
    }
  ]
}
```

Expected direct-apply response:

```json
{
  "ok": true,
  "status": "applied",
  "risk": "low",
  "auto_committed": true,
  "mutation_id": "mutation:01J...",
  "operations_accepted": 1,
  "operations_applied": 1,
  "mutations_applied": {
    "entity_upserts": 2,
    "edge_upserts": 1,
    "invalidations": 0
  },
  "warnings": []
}
```

Expected `--dry-run` response:

```json
{
  "ok": true,
  "status": "validated",
  "would_apply": true,
  "risk": "low",
  "operations_accepted": 1,
  "preview": {
    "entity_upserts": 2,
    "edge_upserts": 1,
    "invalidations": 0
  },
  "warnings": []
}
```

## Query Surface

This is the place to define and evolve how agents *ask* the graph. A graph is
only as useful as its query surface, so it is worth treating as a first-class
contract rather than a side effect of the read command.

The read side is not one command — it is three query axes. The four target use
cases reduce to three distinct query shapes, and a single flat lookup covers
only one of them. The non-negotiable above is "no raw store," not "no rich
queries": expressiveness lives in these three typed, pot-scoped,
backend-portable axes.

| Axis | V1.5 delivery | What it does | Carries |
|---|---|---|---|
| **Retrieve** | `graph read --view` | Ranked semantic read over a named view; returns entities *with their immediate relations inline* (a shallow join). | UC1 surface-on-intent, UC3 timeline-by-relevance, UC4 symptom recall |
| **Filter** | `graph search-entities` | Typed structured lookup by entity type, predicate, scope, time window, truth class, strength, **and edge qualifiers** (e.g. `environment=prod`). The `WHERE`-clause half. | UC2 "which adapter in which env", enumerations, identity resolution before writes |
| **Traverse** | `infra_topology.service_neighborhood` view (`depth`, `direction`) | Bounded, depth-limited, predicate-typed neighborhood walk. | UC2 dependency blast-radius, UC4 bug→fix→PR chains |

Per use case, the exact query each one reduces to:

| Use case | Reduces to | Axes used |
|---|---|---|
| 1. Preferences (surface-on-intent) | semantic match (intent → preference) + hierarchical scope filter (repo→dir→file) + rank by strength/recency | Retrieve (the ranking is the work, not the filter) |
| 2. Infra / dependency graph | multi-hop dependency walk + edge-qualifier filter (env) | Traverse + Filter |
| 3. Timeline | time-range scan + scope filter + order-by-time | Filter (Retrieve when correlating by relevance) |
| 4. Bug recall | semantic match (symptom → prior bug) + 1–2 hop join to fix/PR | Retrieve (+ inline relations) |

Two design facts fall out of this table:

- **For UC1 and UC4 the filter is only the `WHERE` clause; the value is in the
  ranking.** "Build a payments feature" will not keyword-match a preference
  titled "wrap external calls in tenacity retry," and "refund race timeout" will
  not keyword-match "payment deadlock on concurrent settle." These two use cases
  succeed or fail on the Retrieval Hardening track (R1–R7), not on query-language
  expressiveness. Put the energy there.
- **Traverse is a real axis, not a view detail.** A flat
  `(subject, predicate, object)` filter is 1-hop only; UC2 blast-radius is
  transitive closure and UC4 is a 1–2 hop join. V1.5 delivers traversal *through*
  traversal-backed views (`service_neighborhood` with a `depth` param;
  `prior_occurrences` returning the bug with its `RESOLVED_BY`/`FIXES` relations
  inline) so no generic primitive is needed yet. V2 promotes traversal to a
  composable `graph neighborhood` op once views prove insufficient — keeping the
  change additive, not a contract break.

**Edge qualifiers are part of identity, not just axis-2 filtering.** Filtering on
`environment=prod` needs relations to carry an `environment` qualifier, and —
decided for V1.5 — that qualifier joins the edge identity / singleton /
supersession key when present: `(subject, predicate, object, environment)`.
Without it, adding a staging edge supersedes the prod edge under the same
`(subject, predicate)` and the infra subgraph collapses every environment into
one ambiguous topology; with it, prod and staging coexist and "what depends on X
in prod" is a single qualified traversal. The qualifier rides on both bindings
(`USES`/datastore/adapter/config) and service dependencies (`DEPENDS_ON`). Step 1
extends the identity key; Step 9 stamps the qualifier on infra edges.

**Boundary (deliberate).** These three axes cover UC1–UC4 and nothing wider.
Open-ended graph analytics — arbitrary shortest path across mixed predicates,
centrality/PageRank, cycle detection, unbounded recursive aggregation — is out
of scope on the agent surface. This is a project-*memory* graph for
retrieval-into-context, not a graph-analytics engine. If an operator genuinely
needs that shape, the raw-Cypher escape hatch (operators only, off the agent
tool surface) is where it lives — never the agent query surface.

## Trigger Model

The Query Surface says *what* an agent can ask; the trigger model says *when it
fires and who pays*. A surface nobody calls at the right moment is dead weight,
and the four use cases all silently assume read-before-act and write-back-after-
learning. This section makes that invocation explicit — and does it entirely on
the user's existing subscription, with no API token.

### Principle: the billing rule is the architecture rule

The only way to use model intelligence for free is in the user's interactive
session (skills and subagents run on their Claude subscription). Anything that
runs as a side process — hooks, headless `claude -p`, the Agent SDK, an
LLM-calling MCP server — needs its own API key. So the split is hard:

- **Reasoning** (fact vs claim, which entity, which mutation, what to capture)
  runs **in-session** as skills/subagents → user's subscription → free.
- **Triggering** runs as **mechanical hooks** that only ever shell out to the
  deterministic `potpie graph` CLI. A hook never calls a model.
- **Retrieval** uses a **local embedding model bundled in Potpie** — hooks can't
  borrow the session and we will not pay for an embeddings API. Zero external
  dependency.
- **Ingestion** is harness-led. Source events, repo links, tickets, and docs are
  read by the in-session agent/harness, which decides what durable facts exist
  and writes semantic mutations. Hooks and background paths do not infer or write
  graph facts from source material.

There is no background/autonomous reasoning in V1.5. The one place a token would
appear — an out-of-session maintenance pass — is deferred.

### Retrieval cards are agent-authored

Because the embedding model is local and small, recall quality comes from the
*input*, not the model. Every node and edge written through `mutate` carries an
agent-authored `description` — a natural-language retrieval card the in-session
model writes specifically so the entity/claim surfaces for the queries it should
match. Potpie embeds that card locally on write (R1/R2). The agent is the
intelligence that makes search work; Potpie is the deterministic index. Skills
must instruct the agent to write descriptions for *retrieval*, not for display —
include the symptoms, synonyms, and scope a future searcher would use.

### Mechanism: one nudge brain, thin dumb hooks

Trigger policy lives in a single new command, not in fragile per-harness shell
scripts:

```bash
potpie graph nudge --event <event> --scope <key>|--path <file> --session <id> --json
```

It returns:

```json
{
  "inject_context": "compact, ranked context to add to the session (or null)",
  "instruction":    "optional short directive for the agent, e.g. capture this fix (or null)",
  "silent":         false
}
```

A hook is then a three-line adapter: forward the event + path from the harness
hook payload, call `potpie graph nudge`, inject its `inject_context` /
`instruction` into the session (or do nothing when `silent`). Potpie owns *what
to read, when, and whether to prompt a write*; the harness owns only the wiring.
The same adapter shape works for Claude Code, Codex, and Cursor.

The nudge has two directions:

- **Inject-data (read):** `nudge` runs a `graph read` under the hood and injects
  ranked results. The agent gets context whether or not it thought to ask.
- **Inject-instruction (write):** `nudge` injects a short "consider recording X"
  directive; the in-session agent decides and calls `graph mutate`. No hook ever
  reasons.

### Event → nudge map

| Harness event | `nudge` does | UC | Direction |
|---|---|---|---|
| `SessionStart` | inject repo baseline (active decisions, repo-level prefs) and optionally prompt harness-led source-history ingest | 1/2 read, 3 optional write | data/instruction |
| `PreToolUse(Write\|Edit)` | inject preferences + known bug-patterns scoped to the file path | 1, 4 | data |
| `PreToolUse(Bash: deploy/infra cmd)` | inject env-qualified service neighborhood + adapter bindings | 2 | data |
| `PostToolUse(Bash test/run **fails**)` | inject prior-occurrence matches by symptom + recent changes for scope | 4, 3 | data |
| `PostToolUse(test **red→green**)` | nudge: "you resolved `<error>` after editing `<files>` — record the bug+fix if non-obvious" | 4 | instruction |
| `Stop` (end of task) | nudge: "capture durable learnings (new prefs, decisions, fixes)" | 1, 4 | instruction |

The red→green and `Stop` nudges are the compounding engine: they are what make
the graph accumulate, without a hook ever calling a model.

### Potpie provides vs. harness wires

**Potpie (V1.5 local runtime):**

1. `potpie graph nudge` — the event→action policy brain (view selection,
   thresholds, session-dedup).
2. Fast, token-budgeted `graph read` suited to injection (compact,
   source-ref-first).
3. A bundled **local embedding model** — retrieval needs no API key.
4. Idempotent `graph mutate` (idempotency key) so repeated nudge-driven writes
   do not duplicate.
5. A per-session injection ledger so the same context is not injected twice.

**Harness (thin):**

1. Claude Code: a **Potpie plugin** bundling `hooks.json` (events → `graph
   nudge`) and the **skills/subagents** that do the free in-session reasoning.
   Plugin skills/subagents run on the subscription; plugin hooks are mechanical
   CLI calls. Both free.
2. Codex / Cursor: the same `nudge` / `read` / `mutate` calls behind their own
   thin adapters — deferred past V1.5.

### Noise control

A nudge that fires constantly gets ignored, so the policy brain must:

- return `silent: true` when nothing scores above threshold;
- never inject the same preference/bug twice in a session (the injection ledger);
- keep payloads within a token budget (top-K, compact);
- emit write-instructions only on strong signals (red→green, explicit decision
  language, end-of-task) — not on every `Stop`.

### V1.5 scope

- **In:** the six Potpie primitives above, the Claude Code plugin (hooks +
  skills), and the four nudge events that carry UC1–UC4.
- **Deferred:** Codex/Cursor adapters, a richer event taxonomy, and any
  background/autonomous pass (the one path that would cost a token).

## Implementation Steps

### 0. Add Contract Tests First

Goal: lock the intended V1.5 behavior before moving code.

Add tests:

- `potpie/context-engine/tests/unit/test_graph_surface_lite_contract.py`
- `potpie/context-engine/tests/unit/test_semantic_mutations.py`
- `potpie/context-engine/tests/conformance/test_graph_surface_lite_e2e.py`

Test expectations:

- `graph catalog` reports `graph_contract_version=v1.5`, ontology version,
  views, truth classes, and supported mutation ops.
- `graph read --view bugs.prior_occurrences` reaches the same data as
  `context_resolve(include=("prior_bugs",))`.
- `graph mutate --dry-run` validates and does not write.
- `graph mutate` writes low-risk operations through `GraphMutationPort`.
- `context_record` and `graph mutate` produce claims with the same V1.5 metadata.
- MCP still exposes exactly the four existing `context_*` tools until explicitly
  changed.

Done when the new tests fail for the expected missing implementation and the
existing suite still imports.

### 1. Add Versioned Graph Contract Constants

Goal: stop scattering contract names, versions, and truth vocabularies.

Add a small domain module, for example:

- `potpie/context-engine/domain/graph_contract.py`

Include:

- `GRAPH_CONTRACT_VERSION = "v1.5"`
- `ONTOLOGY_VERSION = "2026-06-graph"`
- `TruthClass` enum
- `SemanticMutationOp` enum
- `MutationRisk` enum
- supported source authority strings
- canonical entity key pattern helpers
- the edge identity key, which includes `environment` when present:
  `(subject, predicate, object, environment)` (see Query Surface). The
  singleton/supersession key derives from the same tuple, so an env-qualified
  edge never supersedes its counterpart in another environment.

Then update `domain/ontology.py` to import or mirror `ONTOLOGY_VERSION`.

Important cleanup: current code and docs disagree on some key patterns. For
example, docs use `bug-pattern:<scope>:<slug>`, while `domain/ontology.py` uses
`key_prefix="bug_pattern"`. Decide now and make the code canonical before more
facts are written. Prefer the Graph V2 docs' readable key patterns and add
normalization aliases for legacy prefixes where needed.

Done when:

- one module owns graph contract constants;
- ontology version in code matches catalog output;
- tests assert key prefix consistency for public entity types.

### 2. Add V2-Style Read View Map

Goal: make V1 reads forward-compatible without replacing readers.

Add a module such as:

- `potpie/context-engine/domain/graph_views.py`

Suggested shape:

```python
@dataclass(frozen=True, slots=True)
class GraphViewSpec:
    name: str
    subgraph: str
    view: str
    v1_include: str
    description: str
    backed: bool
```

Initial map:

| View | V1 Include | Backed |
|---|---|---:|
| `bugs.prior_occurrences` | `prior_bugs` | yes |
| `recent_changes.timeline` | `timeline` | yes |
| `infra_topology.service_neighborhood` | `infra_topology` | yes |
| `preferences.active_preferences` | `coding_preferences` | yes |
| `admin.inspection_slice` | `raw_graph` | yes |
| `decisions.active_decisions` | `decisions` | no |
| `ownership.owner_context` | `owners` | no |
| `docs.reference_context` | `docs` | no |

Wire this into:

- `DefaultGraphService.read(...)`
- `graph catalog`
- status metadata if useful

Done when `graph read --view <name>` can route through `ReadOrchestrator` and
unsupported views return an honest `not_implemented` response.

### 3. Add Semantic Mutation DTOs

Goal: introduce the public write contract above the structural mutation batch
(`ReconciliationPlan`, renamed to `MutationBatch` in Step 5a). Two tiers:
public/agent layer `SemanticMutation*`; internal/backend layer `MutationBatch`.

Add:

- `potpie/context-engine/domain/semantic_mutations.py`

Recommended DTOs:

- `GraphEntityRef`
- `GraphEvidenceRef`
- `SemanticMutation`
- `SemanticMutationRequest`
- `SemanticMutationValidationIssue`
- `SemanticMutationPlan`
- `SemanticMutationResult`

Keep the public payload batch-shaped:

```json
{
  "pot_id": "...",
  "operations": []
}
```

Support a temporary single-operation alias only at the CLI parsing boundary:

```json
{
  "operation": "assert_claim",
  "...": "..."
}
```

Normalize that into `operations=[...]` immediately.

Minimum operation support for first implementation:

- `upsert_entity`
- `link_entities`
- `assert_claim`
- `append_event`
- `end_relation_validity`
- `retract_claim`

Return `review_required` for these until the review workflow exists:

- `supersede_claim`
- `merge_duplicate_entities`
- broad topology removals
- multi-claim invalidations

Defer `patch_entity` and `transition_state` to V2. V1.5 models state changes as
new claims/events (e.g. a `VERIFIED` claim, or an `append_event`), not in-place
property/state edits. The catalog advertises them under `deferred_operations` so
the absence is honest, not silent.

Every entity ref and claim-bearing op carries an agent-authored `description` —
the natural-language retrieval card the local embedder indexes (see Trigger
Model / R2). It is required for entities and link/assert ops, optional for pure
events. Validation should warn (not reject) on an empty or display-only
description, because recall quality depends on the agent writing it for search:
symptoms, synonyms, and scope a future query would use.

Done when mutation JSON can parse into typed DTOs without importing adapters or
backends.

### 4. Add Semantic Validator And Risk Policy

Goal: reject unsafe or ungrounded mutations before they become structural plans.

Add:

- `potpie/context-engine/application/services/semantic_mutation_validator.py`

Validation rules:

- `graph_contract_version` is supported.
- `pot_id` is present.
- each `op` is known.
- entity `type` exists in `ENTITY_TYPES`.
- entity key prefix matches the entity type's canonical key policy.
- predicate exists in `EDGE_TYPES`.
- predicate endpoint rules allow subject/object labels.
- `truth` is a supported truth class.
- durable writes include evidence or an explicit low-authority truth class.
- `confidence` is between `0.0` and `1.0`.
- timestamps parse as ISO 8601.
- no operation can hard-delete a fact.
- risky operations are classified as `review_required`, not silently applied.

Risk policy:

| Risk | Auto Apply In V1.5 | Examples |
|---|---:|---|
| `low` | yes | evidence-backed `link_entities`, narrow preferences, low-authority harness observations |
| `medium` | only with explicit approval flag | user decisions, state transitions, retractions with evidence |
| `high` | no | entity merge, broad supersession, topology removal, unknown ontology |

V1.5 can support `--allow-review-required --approved-by <user-ref>` for local
CLI, but default `graph mutate` should only apply low-risk validated plans.

Done when invalid semantic payloads fail with structured issues and no backend
write is attempted.

### 5. Add Semantic Lowering To `ReconciliationPlan`

Goal: keep backend ports stable while changing the public write contract.

Add:

- `potpie/context-engine/application/services/semantic_mutation_lowering.py`

Lowering rules:

- `upsert_entity` -> `EntityUpsert`
- `link_entities` -> subject/object `EntityUpsert`s plus one `EdgeUpsert`
- `assert_claim` with entity object -> same as `link_entities` with claim metadata
- `assert_claim` with value object -> create an `Observation` or `Document`
  object entity and link to it; do not invent authoritative facts from raw text
- `append_event` -> `Activity` entity plus `TOUCHED`, `PERFORMED`, or `MENTIONS`
  edges when targets are present
- `end_relation_validity` -> `InvalidationOp` or `GraphMutationPort.invalidate`
  target, never hard delete
- `retract_claim` -> invalidation with reason and evidence

Every lowered `EdgeUpsert.properties` should carry:

- `claim_key`
- `subgraph`
- `truth`
- `confidence`
- `source_refs`
- `evidence`
- `valid_at` or `valid_from`
- `valid_until`
- `observed_at`
- `created_by`
- `graph_contract_version`
- `ontology_version`
- `idempotency_key`
- `fact`

Use deterministic `claim_key` generation. A practical first identity:

```text
claim:<pot_id>:<subgraph>:<subject_key>:<predicate>:<object_key-or-value-hash>:<source-ref-or-idempotency-hash>
```

Done when a semantic mutation can produce a `MutationBatch` (the renamed
`ReconciliationPlan`, see Step 5a) and `ProvenanceContext` without writing.

### 5a. Rename `ReconciliationPlan` → `MutationBatch` (Same Pass As Step 5)

Goal: name the two write tiers coherently and stop "reconciliation" from naming
two different things.

`ReconciliationPlan` no longer reconciles anything. It is a typed, atomic batch
of `entity_upserts` / `edge_upserts` / `edge_deletes` / `invalidations` applied
through the one write door — its own docstring says "plans are entity/edge
upserts, deletes, and invalidations." The name is vestigial from the old
agentic-reconciliation design. Reserve "reconciliation" for genuine source-state
convergence: the ledger `reconcile()` path and V2's `reconcile_snapshot`.

Do this in the **same pass as Step 5** — that is the one moment you are already
introducing the public `SemanticMutation*` layer and touching the lowering, so
naming both tiers together is cheap. Do not ship it as a standalone churn PR
(~33 files plus the whole `reconcil*` surface).

Rename:

- `ReconciliationPlan` → `MutationBatch` (the public `SemanticMutationPlan` keeps
  its name; `MutationBatch` is the internal tier it lowers into).
- `apply_reconciliation_plan` → `apply_mutation_batch` (`adapters/outbound/graph/apply_plan.py`).
- `ReconciliationResult` → `MutationResult`; `MutationSummary` stays as-is.
- `ReconciliationPlanValidationError` → `MutationBatchValidationError` (`domain/errors.py`).
- `application/services/reconciliation_validation.py` → `mutation_batch_validation.py`.
- Drop the `IngestionPlan = ReconciliationPlan` alias. Keep a thin
  `ReconciliationPlan = MutationBatch` shim for one iteration only if external
  imports still need it, then delete.

Shape cleanup (do while renaming):

- Relax `event_ref` and `summary` from required to optional. They force an event
  frame onto non-event writes — `graph_service._lower_record` fabricates
  `EventRef(source_system="agent")` purely to record a preference.
- Carry write provenance through `ProvenanceContext`, not as required batch
  fields.

Done when the backend write door applies a `MutationBatch` (via
`apply_mutation_batch`), no non-event write fabricates an `EventRef`, and
"reconciliation" appears in the code only on the snapshot/source-convergence
path.

### 6. Extend `GraphService` With Graph Surface Lite Methods

Goal: give CLI a stable service API and keep adapters thin.

Update:

- `potpie/context-engine/domain/ports/services/graph_service.py`
- `potpie/context-engine/application/services/graph_service.py`

Add methods:

```python
def catalog(self, request: GraphCatalogRequest) -> GraphCatalogResult: ...
def read(self, request: GraphReadRequest) -> AgentEnvelope: ...
def search_entities(self, request: GraphEntitySearchRequest) -> GraphEntitySearchResult: ...
def mutate(self, request: SemanticMutationRequest) -> SemanticMutationResult: ...
```

Implementation:

- `catalog` derives from `domain/ontology.py`, `domain/graph_views.py`, contract
  constants, and backend capabilities.
- `read` maps `view` to V1 include, calls `ReadOrchestrator.resolve`, and stamps
  a `subgraph_versions` block onto the envelope (stub/monotonic in V1.5).
- `search_entities` uses `backend.semantic.search` or `claim_query.find_claims`
  and returns entity candidates plus supporting claims.
- `mutate` parses, validates, risk-classifies, lowers, and either dry-runs or
  applies through `backend.mutation.apply`.

Keep `resolve`, `search`, and `record` in the same service. Do not add a second
graph stack.

Done when service-level tests pass without invoking Typer or MCP.

### 6a. Wire The Three Query Axes

Goal: make the Query Surface real, not just a named-view facade that fails UC2.

The bare `read` + `search-entities` above only cover the Retrieve and Filter
axes shallowly. Close the three gaps the use-case analysis surfaced:

- **Inline relations (Retrieve).** `read` views must return each entity with its
  immediate relations attached, so a use case's join needs no second call. Mirror
  the V2 read-response `relations[]` shape.
- **Edge qualifiers (Filter).** Extend `search_entities` / the claim query to
  filter on predicate and the `environment` qualifier, not just entity type +
  text. Because the qualifier is part of identity (Step 1), this filters over
  distinct claims rather than disambiguating one ambiguous edge. Requires the
  qualifier stamped on infra edges in Step 9.
- **Bounded traversal (Traverse).** Make `infra_topology.service_neighborhood` do
  a real depth-limited, direction-aware walk (`depth ≤ K`, predicate-typed,
  `environment`-filtered), implemented over the backend's native traversal
  (variable-length path on Neo4j, recursive CTE on Postgres, BFS on embedded) —
  never raw Cypher across the agent boundary. Do **not** add a generic
  `graph neighborhood` command in V1.5; that is the V2 promotion.

Each of the four use-case views must declare a contract — inputs (scope params),
assembled output (entities + which relations are inlined), and ranking inputs —
not just a name. The minimum set:

| View | Inputs | Assembled output (inline) | Ranking inputs |
|---|---|---|---|
| `preferences.active_preferences` | `repo`, `path`/`scope`, `query` | preference entities + `POLICY_APPLIES_TO` scope, walked up the scope hierarchy (repo→dir→file) | semantic match, strength, recency, **truth class** |
| `bugs.prior_occurrences` | `query` (symptom), optional `service`/`repo`, `time_window` | bug pattern + `REPRODUCES` scope + `RESOLVED_BY` fix/PR + verification, all inline | semantic symptom match, recency, resolution status |
| `recent_changes.timeline` | optional `scope`, `time_window` | project-pot activity/PR/ticket events across registered repo sources, ordered by time; scope narrows when explicitly supplied | recency, scope overlap |
| `infra_topology.service_neighborhood` | `service`, `depth`, `direction`, `environment` | the depth-bounded neighborhood with `DEPENDS_ON`/`USES` edges and their `environment` qualifier | n/a (deterministic walk) |

Done when UC2 ("what depends on ledger-api, in prod, two hops out") and UC4
("has this symptom been seen before, and what fixed it") are each answerable
through the surface in one call, without a raw store query.

### 7. Add CLI Commands Under `potpie graph`

Goal: expose the V2-shaped in-between surface to agents.

Update:

- `potpie/context-engine/adapters/inbound/cli/commands/graph.py`

Add:

```bash
potpie graph catalog [--pot] [--subgraph]
potpie graph read --view <subgraph.view> [--query "..."] [--scope key:value] [--since ISO] [--until ISO] [--time-window 7d] [--format events|raw|jsonl] [--limit N] [--pot]
potpie timeline recent [--time-window 7d] [--service <name>] [--limit N] [--pot]
potpie graph search-entities <query> [--type Service] [--limit N] [--pot]
potpie graph mutate --file mutation.json [--dry-run] [--allow-review-required] [--approved-by user:...] [--pot]
```

Adapter rules:

- CLI reads JSON files and stdin, then passes DTOs to `host.graph`.
- CLI does not validate ontology itself.
- CLI does not call backend ports directly except existing admin commands.
- CLI JSON output includes `graph_contract_version` and `ontology_version`.

Done when a local user can run `catalog`, `read`, `search-entities`, and
`mutate --dry-run` against the embedded backend.

### 8. Rewire `context_record` Through Semantic Mutations

Goal: make V1 compatibility writes use the same path as `graph mutate`.

Update:

- `DefaultGraphService.record(...)`
- `domain/context_records.py` only if payload shape gaps are found
- `domain/ports/agent_context.py` to add optional receipt metadata if needed

Flow:

1. Normalize `record_type`.
2. Validate structured payload with `validate_record_payload`.
3. Convert the record into a `SemanticMutationRequest`.
4. Call `self.mutate(...)`.
5. Return the existing `RecordReceipt` shape with extra metadata:
   - `graph_contract_version`
   - `ontology_version`
   - `mutation_id`
   - `claim_keys`
   - `subgraph`
   - `truth`
   - `risk`
   - `auto_committed`

Initial record mappings:

| Record Type | Semantic Operation | Truth | Subgraph/View |
|---|---|---|---|
| `preference`, `policy` | `assert_claim` / `link_entities` with `POLICY_APPLIES_TO` | `preference` | `preferences` |
| `bug_pattern` | `assert_claim` with `REPRODUCES` | `agent_claim` | `bugs` |
| `fix` | `assert_claim` with `RESOLVED` | `agent_claim` | `bugs` |
| `verification` | `append_event` or `assert_claim` with `VERIFIED` | `timeline_event` / `agent_claim` | `bugs` |
| `decision` | `assert_claim` with `DECIDED` | `user_decision` | `decisions` |
| free-form note types | `assert_claim` to `Document`/`Observation` or `review_required` | `agent_claim` / `source_observation` | relevant or `admin` |

Backward compatibility:

- Old callers that provide only `summary` should still get a receipt.
- If evidence is insufficient, use low-authority `agent_claim` or return
  `review_required`, depending on risk.
- Do not let `context_record` bypass semantic validation.

Done when existing record -> resolve tests pass and new tests prove
`context_record` and `graph mutate` use the same lowerer.

### 9. Stamp V1.5 Claim Metadata In Backends

Goal: make written claims future-readable without a migration.

Update:

- `domain/ports/claim_query.py`
- `adapters/outbound/graph/backends/in_memory_backend.py`
- `adapters/outbound/graph/backends/embedded_backend.py`
- `adapters/outbound/graph/neo4j_reader.py`
- `adapters/outbound/graph/cypher.py`
- `adapters/outbound/graph/falkordb_reader.py` if FalkorDB remains supported

Add optional `ClaimRow` fields or reliably expose these via `properties`:

- `claim_key`
- `subgraph`
- `truth`
- `confidence`
- `source_refs`
- `evidence`
- `description` (agent-authored retrieval card)
- `environment` (qualifier; part of the identity key when present — Step 1)
- `valid_until`
- `observed_at`
- `mutation_id`
- `graph_contract_version`
- `ontology_version`

Preferred implementation:

- Add optional `ClaimRow` fields for first-class reads.
- Keep all fields mirrored in `properties` for compatibility.
- Embed the agent-authored `description` with the bundled local model on write
  and persist the vector for ANN (R1). No external embeddings service.
- Stamp `environment` on infra edges so axis-2 filtering and the
  `service_neighborhood` walk are environment-correct.
- Update JSON dump/load for embedded backend.
- Update Neo4j/FalkorDB readers to parse edge properties into `ClaimRow`.

Done when a claim written through `graph mutate` can be read back with V1.5
metadata (including its local embedding and `environment`) on in-memory and
embedded backends.

### 10. Remove Local Deterministic Ingestion

Goal: keep durable graph updates behind harness judgment.

Removed:

- local code/config ingestion CLI commands;
- local ingestion services, adapters, registry, and port;
- automatic timeline update commands;
- Event Ledger apply mode and the built-in event reconciler.

Done when the only local durable write surfaces are `context_record` and
`graph mutate`, and Event Ledger pull is read-only.

### 11. Park Service-Side Reconciliation As Non-Canonical

Goal: align implementation with "intelligence lives in the harness."

Current conflict:

- `adapters/outbound/reconciliation/factory.py` enables the pydantic-deep
  planner by default when installed.
- `pydantic_deep_agent.py` owns a graph mutation loop.
- managed HTTP `context_record` submits events into the reconciliation pipeline.

Implementation direction:

- Make LLM reconciliation opt-in, not default.
- Change `CONTEXT_ENGINE_AGENT_PLANNER_ENABLED` default to false when this does
  not break a required deployment path.
- Prefer deterministic event processors for source events.
- For ambiguous event content, create observations or pending work rather than
  canonical rich facts.
- Keep old reconciliation classes available for experiments/tests, but do not
  document them as the canonical graph update path.
- Move the agentic-only types — `ReconciliationRequest`, `EvidenceRef`,
  `LlmReconciliationPlan` and its convert — out of `domain/reconciliation.py`
  into the parked LLM adapter package, so the renamed `MutationBatch` module
  (Step 5a) holds only the live structural write tier.

Recommended sequence:

1. Finish local CLI Graph Surface Lite first.
2. Rewire local `context_record`.
3. Add deterministic handling for managed `context_record`, or route it to the
   same semantic mutation path.
4. Flip planner default to opt-in.

Done when canonical graph updates do not require Potpie-owned LLM reconciliation.

### 12. Update Skills And Agent Templates

Goal: make harness instructions match the new responsibility split.

Update:

- `adapters/inbound/cli/templates/agent_bundle/AGENTS.md`
- `adapters/inbound/cli/templates/claude_bundle/CLAUDE.md`
- built-in skill catalog if it references old recipes

Template changes:

- Tell agents to start with `potpie graph catalog` for graph-aware workflows.
- Use `potpie graph read` for V2-style reads where available.
- Use `potpie graph search-entities` before creating/linking non-authoritative
  entities.
- Use `potpie graph mutate --dry-run` before direct apply for anything not
  obviously low-risk.
- Keep `context_resolve`, `context_search`, and `context_record` documented as
  compatibility wrappers.
- Remove promises that Potpie will infer rich ontology updates from prose.
- Require a `description` written **for retrieval** on every entity/claim —
  symptoms, synonyms, and scope a future searcher would use — since the local
  embedder indexes exactly that text (Trigger Model / R2).
- Define how the agent responds to a nudge: act on `inject_context` as graph
  truth, and treat a write `instruction` as a prompt to decide truth class,
  resolve identity, and call `graph mutate` — never as an auto-write.

Done when installed templates no longer advertise stale include names like
`purpose`, `feature_map`, or `prior_fixes` unless those names are actually
supported, and the skills tell agents to write retrieval-grade descriptions and
to handle nudges.

### 12a. Add The Nudge Brain, Local Embeddings, And Session Ledger

Goal: make triggering deterministic and free (Trigger Model). All policy lives in
Potpie; hooks stay dumb.

Add:

- `potpie graph nudge --event <event> --scope|--path --session <id> --json`,
  returning `{ inject_context, instruction, silent }`. It selects the view for
  the event, runs a token-budgeted `graph read`, applies the relevance threshold,
  and emits a write `instruction` only on strong signals.
- A **bundled local embedding model** behind the existing embedder seam, default
  on the OSS path so retrieval needs no API key. ANN over the persisted vectors
  from Step 9 / R1.
- A **per-session injection ledger** (keyed by `--session`) so the same
  preference/bug is not injected twice.
- Optional nudge instruction for harness-led source-history ingest. The harness
  performs source reads and writes semantic mutations; `graph nudge` itself does
  not ingest or write graph facts.

Non-negotiable: `graph nudge` makes **no model calls** and does not write graph
facts. The only intelligence is the local embedder (similarity, not generation)
and the in-session agent that consumes the nudge.

Done when `graph nudge --event pre-edit --path <f>` returns ranked
`inject_context` for a matching scope, `silent: true` when nothing matches, and
never injects a duplicate within one `--session`.

### 12b. Ship The Claude Code Plugin (Hooks + Skills)

Goal: wire the events to the brain on the user's subscription, with zero tokens.

Add a Potpie plugin bundling:

- `hooks.json` mapping `SessionStart`, `PreToolUse(Write|Edit)`,
  `PreToolUse(Bash …)`, `PostToolUse(Bash …)`, and `Stop` to thin adapters that
  forward the event payload to `potpie graph nudge` and inject its output. The
  adapters contain no logic beyond field-forwarding.
- The **skills/subagents** from Step 12, which do the reasoning in-session.

Both halves run free: plugin skills/subagents on the subscription, plugin hooks
as mechanical CLI calls. No `ANTHROPIC_API_KEY` anywhere in the path.

Done when, on Claude Code, editing a file under a scoped path injects relevant
preferences without the agent asking, a red→green test run prompts a fix capture,
and nothing in the loop calls a model outside the session.

### 13. Add End-To-End Acceptance Tests

Goal: prove the new surface is usable before implementing full V2.

Add or update tests:

- `tests/conformance/test_graph_surface_lite_e2e.py`
- `tests/conformance/test_host_shell_end_to_end.py`
- `tests/conformance/test_graph_backend_conformance.py`
- `tests/unit/test_agent_surface_contract.py`
- `tests/unit/test_record_types.py`
- `tests/unit/test_read_orchestrator.py`
- `tests/unit/test_reconciliation_validation_edge_cases.py`

Required scenarios:

1. `catalog` shows backed and planned views.
2. `read` returns data for a backed view.
3. `search-entities` finds entities from a prior mutation.
4. `mutate --dry-run` validates without writing.
5. `mutate` applies low-risk `link_entities`.
6. `mutate` rejects invalid endpoint pairs.
7. `mutate` returns `review_required` for high-risk operations.
8. `context_record` uses the semantic mutation path.
9. embedded backend persists V1.5 metadata across CLI processes.
10. MCP still exposes exactly four compatibility tools.
11. `graph nudge` returns ranked `inject_context` on a scope match, `silent` on
    none, and no duplicate within one `--session`.
12. an agent-authored `description` is embedded by the local model on write and a
    paraphrased query retrieves it (no external embedder).
13. `graph nudge`/`ingest` make zero model calls (assert no API client is
    constructed on that path).

Suggested focused run:

```bash
python3 -m pytest \
  potpie/context-engine/tests/unit/test_graph_surface_lite_contract.py \
  potpie/context-engine/tests/unit/test_semantic_mutations.py \
  potpie/context-engine/tests/conformance/test_graph_surface_lite_e2e.py \
  potpie/context-engine/tests/conformance/test_host_shell_end_to_end.py
```

## Retrieval Hardening

The Graph Surface Lite read path — `graph read` and the `context_resolve`
compatibility wrapper — routes through `ReadOrchestrator` → per-include readers
→ `ClaimQueryPort.find_claims` → `RankingService`. Three of the four target use
cases (coding preferences surfacing on new code, prior-bug recall by symptom,
timeline correlation) are gated on retrieval *relevance*, not on graph shape or
surface vocabulary. Adding `catalog`/`read`/`mutate` does not move these use
cases unless the retrieval layer underneath improves. Today that layer is a
stub. This section makes it first-class. It runs as a parallel track to the
numbered surface steps above.

### Current Retrieval Reality

| Observation | Evidence | Consequence |
|---|---|---|
| "Semantic similarity" is Jaccard token overlap, not vectors. | `_embedding_score` in `neo4j_reader.py`, `in_memory_reader.py`, `falkordb_reader.py` | Paraphrase ("retry is flaky" vs "wrap calls in tenacity backoff") scores ~0. UC1/UC4 miss. |
| The vector index exists but is dead. | `claim_fact_embeddings` created in `cypher.py`; `fact_embedding` never written or queried. | No real semantic recall path; the index is cosmetic. |
| Candidate generation is a full-partition scan. | `_FIND_CLAIMS_CYPHER` matches all `:RELATES_TO` of the predicate set, then scores in Python. | Won't scale; relevance is decided only after pulling everything. |
| The ranker lets the weakest signal veto. | `RankingService._combine` is a weighted geometric mean with a `1e-6` floor. | A zero lexical-overlap score collapses an otherwise strong, recent, scoped candidate. |
| Scope match is flat exact-string equality. | `_scope_overlap` in `coding_preferences.py` | `file_path` exact match is near-useless; no `repo › service › path-prefix › symbol` hierarchy. |
| `resolve` returns N independently-ranked lists. | `ReadOrchestrator.resolve` fan-out + `EnvelopeBuilder`. | Cross-family candidates are never compared; unbacked families add empty noise. |

### R1. Embed On Write, Retrieve By ANN

Goal: replace token overlap with real vector retrieval over the index that
already exists.

- Add an `EmbedderPort` (domain) with an API-backed default impl and a
  deterministic no-op fallback. Composition wires it in `host_wiring.py`.
- On write (canonical writer / semantic lowering, Steps 5 + 9), embed the
  claim's retrieval card (R2) and persist `fact_embedding`.
- On read, embed the query once and use `db.index.vector.queryRelationships`
  over `claim_fact_embeddings` for top-k candidate generation, replacing the
  MATCH-all + Python loop in `neo4j_reader.find_claims`.
- Keep Jaccard as an explicit, *named* fallback for the no-embedder OSS path.
  Surface the active match mode (`vector` | `lexical`) in `graph status` and
  `context_status` so empty results are debuggable, not mysterious.

Done when a paraphrased query retrieves the right claim on Neo4j with an
embedder configured, and the lexical fallback is labeled rather than silent.

### R2. Embed The Retrieval Card, Not `fact` Alone

Goal: embed the text that actually carries the signal.

- The discriminating content lives in the structured payload — a `Preference`'s
  `prescription`, a `BugPattern`'s `symptom_signature` — not in `fact` alone.
- The card's lead text is the **agent-authored `description`** (Step 3 / Trigger
  Model): the in-session model writes it for search, which is what makes a small
  local embedder enough. Compose the canonical card as `description • subject •
  predicate • object • scope • prescription/symptom` and embed that.
- Query-side expansion (turning "add retry to the payments client" into a good
  retrieval query) belongs in the **harness skill**, not the daemon — consistent
  with "intelligence in the harness." Note this in Step 12 skill updates.

Done when the embedded text for preferences and bugs includes the structured
payload, and the card builder is shared by the write path and the eval read path.

### R3. Split Hard Filters From Soft Scores

Goal: stop a soft signal from silently vetoing candidates.

- Classify read inputs: **filters** (pot, predicate, validity window, scope
  membership) are binary and run in the query; **scorers** (semantic distance,
  recency, strength, corroboration) run after and only re-order.
- Replace the geometric mean in `RankingService._combine` with a weighted sum
  (or keep a geometric form only across scorers that are all monotone and never
  legitimately zero). Semantic distance from R1 is the primary recall signal;
  the rest re-rank.

Done when a strong, recent, scope-matched claim is not buried by a low
lexical-overlap score, and the rank-combination rule is single and documented.

### R4. Hierarchical Scope Matching

Goal: make scope match the way projects are actually shaped.

- Scope is a hierarchy — `repo › service › path-prefix › symbol` (and, for UC2,
  `environment`). Match by containment/prefix, not equality.
- A repo-wide preference must match a file in that repo; a `src/payments/**`
  rule must match `src/payments/client.py`.
- Implement once in the shared retriever (R5), not per reader.

Done when a repo- or path-prefix-scoped preference surfaces for a task scoped to
a file beneath it.

### R5. Unify Readers Into One Parameterized Retriever

Goal: make a use case a config row, not a new module — and implement R1–R4 once.

- `PriorBugsReader`, `CodingPreferencesReader`, `InfraTopologyReader`, and
  `TimelineReader` are ~80% identical: `find_claims(predicate set)` → scope
  overlap → `Candidate` → `rank` → coverage. They differ only by
  **(predicate set, scope keys, payload fields)** plus small hooks (e.g.
  prior_bugs folds in `VERIFIED` counts).
- Replace them with one `ClaimRetriever` parameterized from ontology/view rows;
  express the differences as a declarative spec plus optional post-hooks.
- This is the "verbs fixed, types are data" direction. Do R5 *before* layering
  the `graph_views.py` view vocabulary (Step 2) so a view configures one
  retriever instead of selecting among hand-written readers.

Done when adding a backed view needs a spec row and no new reader class, and
embeddings/scope/ranking live in exactly one place.

### R6. Single Ranked Retrieval Pass

Goal: one coherent context block, not N siloed lists.

- `resolve` should issue one ANN over the relevant predicate set, rank globally,
  apply per-family caps and a relevance floor, and dedupe — instead of
  concatenating N independently-ranked reader outputs.
- Unbacked families contribute nothing rather than empty noise.

Done when a `feature`-intent resolve returns one globally-ranked, thresholded,
deduplicated set with per-family caps, and a weak preference cannot outrank a
strong prior bug merely because they were ranked in separate readers.

### R7. Retrieval Eval Harness

Goal: tune with numbers, not vibes.

- Stand up a golden retrieval set (task → expected claim ids) and measure
  `recall@k` / `MRR` in CI, separate from end-to-end answer quality. Seed it from
  the existing 48-scenario corpus.
- Land this *first* so R1–R6 can each be shown to move the metric.

Done when retrieval quality is a CI number and any weight/embedding change
reports its delta.

## Recommended Build Order

1. Add `graph_contract.py` and `graph_views.py`.
2. Add `graph catalog`.
3. Add `graph read`.
4. Add semantic mutation DTOs.
5. Add validator and risk policy.
6. Add lowerer to the structural batch; rename `ReconciliationPlan` →
   `MutationBatch` in the same pass (Step 5a).
7. Add `graph mutate --dry-run`.
8. Add direct low-risk `graph mutate`.
9. Rewire `context_record`.
10. Stamp/read V1.5 claim metadata across backends.
11. Remove local deterministic ingestion paths and stale metadata.
12. Park service-side LLM reconciliation as opt-in.
13. Update templates/skills around use-case workflows.
14. Add the nudge brain, local embedder, and session ledger (Step 12a).
15. Ship the Claude Code plugin — hooks + skills (Step 12b).

This order gives useful agent-facing value early: `catalog` and `read` can ship
before writes, and `mutate --dry-run` can ship before direct apply. The trigger
model lands last because it composes the read/mutate/ingest primitives — but the
agent-authored `description` (Step 3) and local embedding (Step 9) must be in
place before it, since the nudge's injected context is only as good as what the
agent embedded on write.

Retrieval hardening runs as a parallel track, sequenced for provable gains:

1. R7 eval harness first — every later step must move `recall@k` / `MRR`.
2. R1 embed-on-write + ANN candidate generation; R2 retrieval card.
3. R3 + R4 filter/score split and hierarchical scope.
4. R5 unify readers, then R6 single ranked pass.

Do R5 before Step 2's `graph_views.py` so views configure one retriever rather
than multiplying hand-written readers.

## What To Avoid

- Do not add a raw `graph edge upsert` command.
- Do not expose the structural `MutationBatch` (formerly `ReconciliationPlan`)
  JSON as the agent contract.
- Do not keep calling the backend write batch a "reconciliation"; it reconciles
  nothing. Reserve that word for snapshot/source-state convergence.
- Do not add new MCP tools until intentionally changing the MCP surface tests.
- Do not let `context_record` keep a private direct-lowering path.
- Do not make `graph read` a second read engine; route through the existing
  `ReadOrchestrator`.
- Do not make service-side LLM reconciliation the only way to write canonical
  graph facts.
- Do not hard-delete ordinary claims; use validity ending, retraction, or
  supersession.
- Do not call the Jaccard token-overlap fallback "semantic"; report the active
  match mode (`vector` | `lexical`) in status so empty results are debuggable.
- Do not let a soft signal (semantic distance) veto a candidate; hard filters
  belong in the query, soft signals in the rank.
- Do not wire embeddings, ANN, or scope matching per reader; implement them once
  in the shared retriever.
- Do not put a model call in a hook or `graph nudge`. Reasoning is in-session
  only; those paths use the local embedder (similarity) and nothing generative,
  so the loop needs no API token.
- Do not add deterministic source/code ingestion that bypasses harness judgment.
- Do not let a hook carry trigger logic. The policy lives in `graph nudge`; hooks
  only forward the event payload and inject the result.
- Do not fire nudges without a relevance gate and the session dedup ledger; a
  nudge that always fires gets ignored.

## V1.5 Done Criteria

V1.5 is complete when:

- `potpie graph catalog/read/search-entities/mutate` work locally through
  `HostShell`.
- `graph mutate` accepts semantic mutations, validates them, and applies only
  allowed direct writes.
- `graph mutate --dry-run` previews validation and lowering without writing.
- `context_record` calls the same semantic mutation path.
- written claims carry V1.5 contract, ontology, truth, evidence, provenance, and
  validity metadata.
- local deterministic ingestion paths are removed; harness-authored
  `context_record` / `graph mutate` writes carry the metadata.
- service-side LLM reconciliation is not required for canonical graph writes.
- V2 can later add `describe/propose/commit/history/inbox` without changing
  entity keys, claim shape, or mutation payload semantics.
- retrieval uses real embeddings with a labeled lexical fallback, not Jaccard
  token overlap posing as semantic;
- retrieval quality is measured in CI (`recall@k` / `MRR`) on a golden set;
- the four use-case readers are one parameterized retriever with hierarchical
  scope and a single documented rank-combination rule.
- the backend write tier is named `MutationBatch` and applied via
  `apply_mutation_batch`; "reconciliation" names only source-state convergence,
  and no non-event write fabricates an `EventRef`.
- `graph nudge` injects scoped context and emits write-instructions
  deterministically, with a relevance gate and per-session dedup.
- the embedder is local and bundled by default; the whole trigger + retrieval
  loop runs with no `ANTHROPIC_API_KEY` — reasoning is the in-session
  subscription, similarity is the local model.
- a Claude Code plugin wires `SessionStart`/`PreToolUse`/`PostToolUse`/`Stop` to
  `graph nudge`, so preferences surface on edit and fixes are captured on
  red→green without the agent being asked.
