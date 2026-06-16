# Graph V2 Implementation Plan

Last reviewed: 2026-06-16.

This plan turns the current Graph V1.5 surface into the Graph V2 workbench
described in [`graphv2.md`](./graphv2.md) and
[`workbench-ontology.md`](./workbench-ontology.md). The important constraint is
that V2 is a surface, workflow, and ontology-contract hardening effort, not a
storage rewrite. The existing semantic mutation path, claim metadata, readers,
and backend ports are already the right foundation.

## Current Position

The installed and source-level contract is Graph V1.5:

```bash
potpie graph catalog
potpie graph read
potpie graph search-entities
potpie graph mutate
```

The live catalog reports `graph_contract_version=v1.5`, ontology
`2026-06-graph`, vector match mode, and canonical backed views for decisions,
debugging, recent changes, infra topology, features, admin inspection,
code topology, and knowledge. The public view names are now standardized:

| Canonical view | Internal reader route |
|---|---|
| `decisions.preferences_for_scope` | `coding_preferences` |
| `debugging.prior_occurrences` | `prior_bugs` |
| `recent_changes.timeline` | `timeline` |
| `infra_topology.service_neighborhood` | `infra_topology` |
| `features.feature_context` | `features` |
| `decisions.active_decisions` | `decisions` |
| `code_topology.ownership_by_path` | `owners` |
| `knowledge.document_context` | `docs` |
| `admin.inspection_slice` | `raw_graph` |

The catalog also reports these mutation partitions:

| Partition | Operations |
|---|---|
| Applicable | `append_event`, `upsert_entity`, `link_entities`, `assert_claim`, `end_relation_validity`, `retract_claim`, `patch_entity`, `transition_state`, `supersede_claim`, `merge_duplicate_entities` |
| Review-required | none |
| Deferred | none |

There are already extra operator commands in the CLI (`graph status`, `inspect`,
`repair`, `export`, `import`), but they are not advertised by the V1.5 catalog
and do not use the V2 workbench envelope yet.

## Code Survey

These are the code seams this plan should build on:

| Area | Current code | Notes |
|---|---|---|
| CLI graph commands | `potpie/context-engine/adapters/inbound/cli/commands/graph.py` | Typer commands for catalog/read/search-entities/mutate, plus status/inspect/repair/export/import. CLI owns output rendering and pot resolution. |
| Graph data-plane service | `potpie/context-engine/application/services/graph_service.py` | Current V1.5 implementation: catalog, read, search_entities, mutate, and data_plane_status. Keep this as the core data plane. |
| Graph service port/DTOs | `potpie/context-engine/domain/ports/services/graph_service.py` | Defines V1.5 DTOs. V2 needs new workbench DTOs or a wrapper facade rather than ad hoc dicts in CLI code. |
| Contract constants | `potpie/context-engine/domain/graph_contract.py` | Owns versions, truth classes, mutation operation partitions, source authorities, key helpers, and edge identity. |
| Read view catalog | `potpie/context-engine/domain/graph_views.py` | V1.5 view specs already include inputs, inline relations, ranking inputs, traversal flag, and backed status. V2 should extend this into executable subgraph/view contracts. |
| Semantic mutation DTOs | `potpie/context-engine/domain/semantic_mutations.py` | Agent-facing semantic write payload and current mutate result. V2 propose can reuse this parser. |
| Semantic validation | `potpie/context-engine/application/services/semantic_mutation_validator.py` | Already validates op shape, ontology, evidence/truth, endpoint rules, risk, and review decisions. This should become the proposal validator. |
| Semantic lowering | `potpie/context-engine/application/services/semantic_mutation_lowering.py` | Lowers accepted semantic ops into `MutationBatch` and provenance. V2 plans should persist this validated/lowered intent or a replay-safe representation. |
| Backend write door | `potpie/context-engine/domain/ports/graph/mutation.py` and `potpie/context-engine/adapters/outbound/graph/apply_plan.py` | There is one backend write door. V2 commit should call this by server-held plan id, not accept a mutation payload. |
| Backend bundle | `potpie/context-engine/domain/ports/graph/backend.py` | Keep V2 on the service surface; do not widen the backend into agent semantics. Backends remain mutation/query/semantic/inspection/analytics/snapshot capability bundles. |
| Inspection/traversal | `potpie/context-engine/domain/ports/graph/inspection.py` | Existing neighborhood/path/slice capability can back `graph neighborhood` and admin inspection. |
| Analytics/repair | `potpie/context-engine/domain/ports/graph/analytics.py` | Existing counts/freshness/quality/repair are a starting point for `graph status`, `graph quality`, and `graph repair`. |
| Quality policy | `potpie/context-engine/domain/graph_quality.py` | Existing quality report primitives can seed V2 quality findings, but they are not yet graph-workbench findings with lifecycle. |
| Current tests | `potpie/context-engine/tests/unit/test_graph_surface_lite_contract.py`, `tests/conformance/test_graph_surface_lite_e2e.py` | These should be extended, not replaced. They already lock V1.5 behavior and the exact four-tool MCP constraint. |

## Design Direction

Use a narrow Graph Workbench layer above the existing `GraphService`.

The current `DefaultGraphService` is the data plane: readers, search, semantic
validation/lowering, mutation application, and backend status. V2 commands need
cross-cutting command envelopes, status composition, plan persistence, inbox
persistence, quality findings, and legacy command routing. Putting all of that
directly into the CLI would duplicate rules; putting it into backend ports would
mix agent workflow into storage.

Recommended shape:

```text
CLI commands
  -> GraphWorkbenchService / GraphWorkbenchPort
      -> GraphService data plane
      -> PotManagementService
      -> SkillManager
      -> GraphPlanStorePort
      -> GraphInboxStorePort
      -> GraphBackend capability ports
```

The first implementation can be local-only and file-backed for plans/inbox.
Hosted storage can implement the same ports later.

## Non-Goals

- Do not hard-delete ordinary graph facts as part of V2. Keep soft validity,
  retraction, supersession, and merge history as the correction model.
- Do not expose raw Cypher, SQL, or backend-specific query APIs to agents.
- Do not make the CLI or daemon infer rich ontology updates from prose. Harnesses
  still read sources and propose structured semantic mutations.
- Do not migrate all existing entity keys in the first V2 cut. Keep the
  canonical underscore prefixes and reject non-canonical prefix aliases.
- Do not add compatibility aliases for renamed views or subgraphs. Old public
  names should fail clearly instead of resolving silently.
- Do not change MCP to a large graph API until the CLI workbench contract is
  stable. V1.5 deliberately keeps MCP at exactly four `context_*` tools.

## Phase 0: Lock The V2 Command Envelope

Goal: every `potpie graph ... --json` command has one stable response shape, even
for unimplemented commands.

### Status

Done on 2026-06-15 for the CLI workbench envelope.

- [x] Added `domain/graph_workbench.py` with the V2 envelope DTOs, command
  partitions, unsupported/error/recommended-action DTOs, and the initial
  workbench status DTO.
- [x] Added `application/services/graph_workbench.py` with shared request-id
  creation, success/error/not-implemented envelope builders, V1.5 result
  normalization, and catalog projection into the V2 command set.
- [x] Updated `potpie graph ... --json` commands to emit the V2 top-level
  envelope:
  `ok`, `command`, `request_id`, `pot_id`, `graph_contract_version`,
  `ontology_version`, `subgraph_versions`, `result`, `warnings`,
  `unsupported`, and `recommended_next_action`.
- [x] Added structured `not_implemented` stubs for `describe`, `neighborhood`,
  `propose`, `commit`, `history`, `inbox`, and `quality`.
- [x] Kept the V1.5 data plane intact. The outer workbench
  `graph_contract_version` is now `v2`; the wrapped catalog result reports
  `data_plane_graph_contract_version=v1.5`. The ontology version remains
  `2026-06-graph` until Phase 1 introduces the executable ontology contract.
- [x] Kept `mutate`, `inspect`, `mutation-template`, and `nudge` callable during
  the transition, but excluded them from the canonical V2 `catalog.commands`
  list and surfaced them under `legacy_commands` or admin command lists.
- [x] Updated the bundled nudge hook adapter to read the nested `result` object
  while still tolerating the old flat V1.5 nudge payload.

Known limitations after Phase 0:

- The command bodies for `describe`, `neighborhood`, `propose`, `commit`,
  `history`, `inbox`, and `quality` are intentionally not implemented yet; they
  return structured `not_implemented` envelopes with exit code 2.
- Read/search/mutate result bodies are only wrapped, not fully normalized into
  final V2 body contracts. That normalization remains Phase 2 / Phase 3 work.
- Typer parser failures that happen before a graph command function runs, such
  as invalid primitive option types, can still be Typer-level errors. Command
  validation and domain errors inside graph command execution use the V2
  envelope.
- `potpie timeline ...` and `potpie backend ...` are separate command groups and
  were not moved to the graph workbench envelope in Phase 0.

### Changes

1. Add V2 workbench DTOs in a new domain module, for example
   `domain/graph_workbench.py`:
   - `GraphCommandEnvelope`
   - `GraphCommandError`
   - `GraphUnsupported`
   - `GraphRecommendedAction`
   - `GraphWorkbenchStatus`
   - command result DTOs as they become concrete

2. Add a shared envelope builder in application code, for example
   `application/services/graph_workbench.py`:
   - creates `request_id`
   - stamps `pot_id`, `graph_contract_version`, `ontology_version`
   - includes `subgraph_versions`
   - wraps success under `result`
   - wraps failures under `error`
   - includes `warnings`, `unsupported`, and `recommended_next_action`

3. Update `adapters/inbound/cli/commands/graph.py` so graph commands render the
   workbench envelope instead of each command emitting a bespoke top-level shape.

4. Advertise the V2 command set once envelope coverage exists:
   - `status`
   - `catalog`
   - `describe`
   - `search-entities`
   - `read`
   - `neighborhood`
   - `propose`
   - `commit`
   - `history`
   - `inbox`
   - `quality`
   - admin/operator commands separately

5. Keep `mutate`, `inspect`, and existing shortcuts out of the canonical V2
   command set. If they remain callable during the transition, they should call
   the same workbench/data-plane path and clearly report a canonical replacement;
   do not add compatibility aliases for renamed views or key prefixes.

### Acceptance Criteria

- Every graph CLI command supports `--json` and emits:
  `ok`, `command`, `request_id`, `pot_id`, `graph_contract_version`,
  `ontology_version`, `subgraph_versions`, `result`, `warnings`,
  `unsupported`, and `recommended_next_action`.
- Errors use the same envelope with `ok=false` and structured `error`.
- Unbuilt V2 commands return `not_implemented`, not shell tracebacks or silent
  empty results.
- `graph catalog` advertises commands that are envelope-stable, even if some
  bodies are not implemented yet.

### Tests

- Add unit tests for envelope success/error shape.
- Add CLI tests for `status`, unknown view/subgraph, and `not_implemented`.
- Update graph surface contract tests to separate V1.5 legacy-command
  expectations from V2 workbench expectations.

Implemented coverage on 2026-06-15:

- `tests/unit/test_graph_cli_contract.py` now verifies V2 envelope success,
  validation-error, capability-not-implemented, legacy-mutate rejection, status,
  catalog command advertisement, and `describe` stub behavior.
- Existing V1.5 service-level tests remain in
  `tests/unit/test_graph_surface_lite_contract.py` and
  `tests/conformance/test_graph_surface_lite_e2e.py`; these continue to pin the
  data-plane catalog/read/search/mutate contract separately from the V2 CLI
  envelope.
- `tests/unit/test_nudge_adapter.py` verifies the hook adapter remains
  compatible after `graph nudge` became a workbench-envelope command.

## Phase 1: Make The Ontology Executable

Goal: replace the current loose view catalog with explicit subgraph/view
contracts that can power `catalog`, `describe`, validation, and skills.

### Status

Done on 2026-06-15 for the read-only executable ontology/discovery surface:

- [x] Canonicalized public graph view names:
  `decisions.preferences_for_scope`, `debugging.prior_occurrences`,
  `features.feature_context`, `code_topology.ownership_by_path`, and
  `knowledge.document_context`.
- [x] Removed compatibility-alias behavior for renamed views/subgraphs; old
  public names should fail clearly instead of resolving silently.
- [x] Standardized semantic mutation subgraph routing:
  `POLICY_APPLIES_TO` -> `decisions`, bug/fix predicates -> `debugging`, and
  ownership categories -> `code_topology`.
- [x] Standardized entity-key prefixes: underscore prefixes are canonical, and
  hyphenated prefixes are not normalized as aliases.
- [x] Updated active docs, CLI templates, nudge policies, and graph contract
  tests to use the canonical vocabulary.
- [x] Introduced `domain/graph_workbench_ontology.py` as the executable
  ontology/workbench contract module. It defines `SubgraphContract`,
  `ViewContract`, `EntityTypeContract`, `RelationTypeContract`,
  `MutationPolicy`, `SourceAuthorityPolicy`, `IdentityPolicy`, and
  `ExampleCommand`.
- [x] Wired `graph describe <subgraph> [--view <view>]` to return the V2
  workbench envelope with subgraph/view purpose, when-to-use guidance, scope and
  filter contracts, result shape, ranking inputs, inline relation contracts,
  entity identity rules, source authority rules, mutation policies, truth
  classes, and optional examples.
- [x] Wired `catalog --task` to a deterministic keyword ranker over the
  executable contracts. For example, `debug staging timeout after deployment`
  ranks debugging, recent changes, infra topology, and decisions before
  unrelated views.
- [x] Added contract tests that fail if a view points to an unsupported include
  or if advertised mutation policies drift from the graph contract partitions.

Why this was not implemented in Phase 0:

- Phase 0 deliberately stabilized the outer CLI envelope first. It made every
  graph command return a predictable V2 envelope, including structured
  `not_implemented` responses, without changing the ontology/catalog model.
- Phase 1 needed a new executable contract layer above `graph_views.py`,
  `graph_contract.py`, and `ontology.py`. Doing that before the envelope would
  have mixed protocol stabilization with ontology modeling and made errors
  harder for agents to consume consistently.

Known limitations after Phase 1:

- The executable contracts are static and read-only. They describe the current
  V1.5 data-plane readers; they do not change reader behavior.
- `catalog --task` is deterministic keyword ranking only. It intentionally uses
  no LLM logic and no backend reads.
- Mutation policies are availability contracts (`applicable`,
  `review_required`, `deferred`) derived from V1.5 partitions. Per-subgraph risk
  policy enforcement, proposal persistence, and approval lifecycle remain Phase
  3 work.
- Read result bodies are still the wrapped V1.5 bodies from Phase 0. Full
  contract-shaped read normalization remains Phase 2.

### Changes

1. Introduce a versioned ontology contract module, for example
   `domain/graph_workbench_ontology.py`.

2. Define these primitives as data classes or typed dicts:
   - `SubgraphContract`
   - `ViewContract`
   - `EntityTypeContract`
   - `RelationTypeContract`
   - `MutationPolicy`
   - `SourceAuthorityPolicy`
   - `IdentityPolicy`
   - `ExampleCommand`

3. Migrate `domain/graph_views.py` into the new contract shape without changing
   reader behavior initially. Current readers become canonical V2-backed
   contracts; do not keep compatibility aliases:

   | Canonical V2 view | Internal reader route |
   |---|---|
   | `decisions.preferences_for_scope` | `coding_preferences` |
   | `debugging.prior_occurrences` | `prior_bugs` |
   | `recent_changes.timeline` | `timeline` |
   | `infra_topology.service_neighborhood` | `infra_topology` |
   | `features.feature_context` | `features` |
   | `decisions.active_decisions` | `decisions` |
   | `code_topology.ownership_by_path` | `owners` |
   | `knowledge.document_context` | `docs` |
   | `admin.inspection_slice` | `raw_graph`, reserved for admin/operator use. |

4. Add `graph describe <subgraph> [--view <view>]`:
   - purpose
   - when to use
   - required and optional scope
   - result shape
   - ranking inputs
   - supported filters
   - entity identity rules
   - relation contracts
   - mutation policies
   - source authority rules
   - examples
   - canonical examples

5. Add `catalog --task` ranking. The first version can be deterministic keyword
   ranking over contract descriptions and examples; it does not need LLM logic.

### Key Policy

Use one canonical key vocabulary. Entity-key prefixes are the underscore forms
already wired in the ontology (`bug_pattern`, `api_contract`, etc.). Do not
accept hyphenated prefix aliases such as `bug-pattern` at parse/search
boundaries. Hyphens remain valid inside key bodies, for example
`service:payments-api`.

Recommended approach:

1. Do not mass-migrate existing stored keys in Phase 1.
2. Reject non-canonical prefixes at validation/identity boundaries.
3. Store external IDs as external IDs, not as alternate graph key prefixes.
4. Update docs, templates, and tests to use the canonical vocabulary only.

### Acceptance Criteria

- `graph describe` can teach an agent how to use every backed view without
  reading docs.
- `catalog --task "debug staging timeout after deployment"` ranks debugging,
  recent changes, infra topology, and decisions ahead of unrelated views.
- Existing reader behavior remains available behind the canonical V2 view names;
  old view names do not resolve.
- Contract tests fail if a view points to an unsupported include or advertises a
  mutation that validation cannot handle.

Implemented coverage on 2026-06-15:

- `tests/unit/test_graph_workbench_ontology.py` verifies `describe_contract`,
  deterministic task ranking, mutation-policy partition alignment, unsupported
  view includes, and invalid mutation-policy advertisements.
- `tests/unit/test_graph_cli_contract.py` verifies `catalog --task` ranking,
  successful `graph describe ... --view ... --examples`, and `graph describe`
  validation errors inside the V2 workbench envelope.
- Existing `tests/unit/test_graph_views.py` continues to pin canonical view
  names, reader include routing, backed-view derivation, traversal flags, and
  catalog entry shape.

Verification run:

```bash
uv run ruff check \
  potpie/context-engine/domain/graph_workbench_ontology.py \
  potpie/context-engine/application/services/graph_workbench.py \
  potpie/context-engine/adapters/inbound/cli/commands/graph.py \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py \
  potpie/context-engine/tests/unit/test_graph_workbench_ontology.py

uv run pytest -q \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py \
  potpie/context-engine/tests/unit/test_graph_workbench_ontology.py \
  potpie/context-engine/tests/unit/test_graph_views.py
```

Result: Ruff passed; focused tests passed (`33 passed`).

Additional regression run after formatting:

```bash
uv run pytest -q \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py \
  potpie/context-engine/tests/unit/test_graph_workbench_ontology.py \
  potpie/context-engine/tests/unit/test_graph_views.py \
  potpie/context-engine/tests/unit/test_graph_surface_lite_contract.py \
  potpie/context-engine/tests/conformance/test_graph_surface_lite_e2e.py \
  potpie/context-engine/tests/unit/test_nudge_adapter.py
```

Result: `121 passed`.

## Phase 2: Normalize Reads And Add Neighborhood

Goal: V2 reads should be contract-shaped, scoped, provenance-rich, and explicit
about unsupported or broadened queries.

### Status

Done on 2026-06-15 for the canonical CLI/service read surface and bounded
neighborhood traversal.

- [x] Changed `graph read` to the canonical hard-break syntax:
  `--subgraph <name> --view <view>`. Fully qualified `--view
  subgraph.view` is rejected with a V2 validation envelope.
- [x] Added `GraphReadResult` as the V2 read body contract so graph-workbench
  callers no longer parse `AgentEnvelope` implementation details.
- [x] Normalized read results around `items`, `coverage`, `freshness`,
  `quality`, and `source_refs`. Cross-cutting `unsupported` and
  `subgraph_versions` remain top-level workbench-envelope fields.
- [x] Enforced executable view-contract filters and required narrowing inputs.
  Unsupported filters return structured `unsupported` entries instead of being
  ignored and causing silent broad reads.
- [x] Implemented `graph neighborhood` through the inspection port, with
  depth, direction, predicate, and limit controls.
- [x] Promoted neighborhood controls into `GraphInspectionPort` and implemented
  them for in-memory and FalkorDB inspection projections.
- [x] Extended `search-entities` with subgraph, scope, truth, time-bound,
  environment, and external-id filters.
- [x] Updated graph UI and nudge service callers to consume the V2 read result
  shape directly.

Deliberate contract choices:

- No compatibility alias was kept for the old read syntax. Callers must use
  `--subgraph debugging --view prior_occurrences`, not
  `--view debugging.prior_occurrences`.
- The V2 workbench envelope owns `unsupported` and `subgraph_versions`; read
  result bodies do not duplicate those fields after CLI normalization.

### Changes

1. Change CLI read syntax to accept the V2 form:

   ```bash
   potpie graph read --subgraph debugging --view prior_occurrences --query "timeout"
   ```

2. Make each read result live under the V2 envelope's `result` key:
   - `items`
   - `coverage`
   - `unsupported`
   - `freshness`
   - `quality`
   - `subgraph_versions`
   - `source_refs`

3. Harden scope behavior:
   - required scope must be enforced by each `ViewContract`
   - optional scope must be documented
   - unsupported filters should be returned in `unsupported`, not ignored
   - reads must not silently broaden scope

4. Add `graph neighborhood` as the generic bounded traversal command:
   - backed by `GraphInspectionPort.neighborhood` where available
   - bounded by depth, direction, predicate filters, and result limits
   - returns `unsupported` when the active backend lacks inspection capability
   - remains separate from ordinary read views

5. Extend `search-entities` filters:
   - `--subgraph`
   - `--scope`
   - `--truth`
   - `--since` / `--until`
   - edge qualifier filters such as `--environment`
   - external-id matching when identity records are available; do not accept
     non-canonical graph key prefixes as aliases

### Acceptance Criteria

- Existing use cases still work: project preferences, infra topology, timeline,
  and bug/debug memory.
- Reads return normalized result shapes and no longer require callers to parse
  unrelated `AgentEnvelope` implementation details.
- `neighborhood` works on backends with inspection and fails with a structured
  `unsupported` response on backends without it.
- `search-entities` is useful before writes, not just a broad semantic claim
  search.

### Tests

- Add per-view read contract tests for required scope, unsupported filters, and
  result shape.
- Add conformance coverage for neighborhood on in-memory and embedded/FalkorDB
  backends where supported.
- Add regression tests that environment-qualified infra edges remain isolated.

Implemented coverage on 2026-06-15:

- `tests/unit/test_graph_cli_contract.py` verifies canonical read syntax,
  hard rejection of fully qualified `--view`, normalized timeline JSON, and
  `graph neighborhood` success/unsupported behavior.
- `tests/unit/test_graph_surface_lite_contract.py` verifies V2 read result
  shape, unsupported filter reporting, extended search filters, and
  environment-qualified infra isolation.
- `tests/unit/test_nudge_service.py` verifies the nudge service consumes
  `GraphReadResult`.
- `tests/conformance/test_graph_backend_conformance.py` verifies backend
  inspection capabilities still answer after the neighborhood port extension.

Verification run:

```bash
uv run ruff check \
  potpie/context-engine/domain/ports/services/graph_service.py \
  potpie/context-engine/domain/graph_workbench_ontology.py \
  potpie/context-engine/application/services/graph_service.py \
  potpie/context-engine/adapters/inbound/cli/commands/graph.py \
  potpie/context-engine/application/services/nudge_service.py \
  potpie/context-engine/adapters/inbound/http/ui/router.py \
  potpie/context-engine/adapters/outbound/graph/backends/in_memory_backend.py \
  potpie/context-engine/adapters/outbound/graph/falkordb_inspection.py \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py \
  potpie/context-engine/tests/unit/test_graph_surface_lite_contract.py \
  potpie/context-engine/tests/unit/test_nudge_service.py \
  potpie/context-engine/tests/conformance/test_graph_surface_lite_e2e.py

uv run pytest -q \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py \
  potpie/context-engine/tests/unit/test_graph_surface_lite_contract.py \
  potpie/context-engine/tests/unit/test_nudge_service.py \
  potpie/context-engine/tests/conformance/test_graph_surface_lite_e2e.py \
  potpie/context-engine/tests/conformance/test_graph_backend_conformance.py \
  potpie/context-engine/tests/unit/test_graph_workbench_ontology.py \
  potpie/context-engine/tests/unit/test_falkordb_inspection.py
```

Result: Ruff passed; focused tests passed (`93 passed`).

## Phase 3: Split Mutate Into Propose And Commit

Goal: `propose` validates and persists a plan; `commit` applies exactly that
server-created plan by id.

### Status

Done on 2026-06-15 for the local CLI workbench write path.

- [x] Added graph mutation-plan DTOs and JSON-safe serializers for lowered
  `MutationBatch` and `ProvenanceContext` records.
- [x] Added `GraphPlanStorePort` and a local JSON-backed plan store under the
  Potpie home for cross-process CLI propose/commit flows.
- [x] Added `GraphWorkbenchService.propose` to parse semantic mutation payloads,
  validate with the existing semantic validator, lower applicable operations,
  compute a diff, capture expected `_global` subgraph versions, classify risk,
  persist the plan, and return the V2 workbench envelope body.
- [x] Added `GraphWorkbenchService.commit` to load plans by `plan_id`, reject
  missing, terminal, expired, stale-version, unapproved approval-gated plans,
  and review-required plans, then apply accepted plans through
  `GraphMutationPort.apply`.
- [x] Wired `potpie graph propose --file ... --json` and
  `potpie graph commit <plan_id> --json`.
- [x] Reworked CLI `graph mutate` as a legacy transition wrapper over
  `propose + commit`; it no longer directly calls the data-plane mutate path for
  CLI writes.

Deliberate first-cut choices:

- The first conflict guard uses the existing `_global` claim-count version from
  Phase 2. Per-subgraph counters can replace this behind the same plan record
  field later.
- Low-risk validated plans can commit without approval. Medium- and high-risk
  applicable plans need `--approved-by`; genuinely review-required operations
  persist as plans but are not applied by the local commit path.
- Plan TTL defaults to one hour. The CLI accepts `--ttl` values such as `30m`,
  `1h`, `7d`, or `2w`.
- Plan history is currently stored in the plan store only. The `graph history`
  command remains Phase 4.

Known limitations after Phase 3:

- `GraphService.mutate` remains as the V1.5 data-plane/internal compatibility
  method for existing wrappers such as `context_record`. Full wrapper migration
  remains Phase 8.
- The local JSON plan store is suitable for local CLI use but has no
  transactional lease model. Hosted or multi-process installs still need a state
  DB implementation.
- Review-required or deferred operations are persisted and visible but
  intentionally do not commit until their validation, lowering, history, and
  approval workflow are implemented.

### Changes

1. Add a plan store port:

   ```text
   domain/ports/graph/plan_store.py
   adapters/outbound/graph/plan_stores/local_json.py
   adapters/outbound/postgres/graph_plan_store.py later
   ```

2. Add plan DTOs:
   - `GraphMutationProposal`
   - `GraphMutationPlanRecord`
   - `GraphMutationPlanStatus`
   - `GraphMutationDiff`
   - `GraphMutationApproval`

3. Implement `graph propose --file mutation.json`:
   - parse `SemanticMutationRequest`
   - validate with `validate_semantic_request`
   - lower with `lower_semantic_request` when applicable
   - compute preview/diff from the lowered `MutationBatch`
   - capture expected subgraph versions
   - classify risk
   - persist the plan with expiry
   - return `plan_id`, status, risk, diff, warnings, rejected operations, and
     recommended next action

4. Implement `graph commit <plan_id>`:
   - load the stored plan
   - reject expired, abandoned, invalid, already committed, or stale-version plans
   - require approval fields for medium/high risk according to mutation policy
   - apply through `backend.mutation.apply`
   - persist mutation id and final status
   - return committed diff, mutation id, claim keys, and history pointer

5. Rework `graph mutate`:
   - keep it out of the canonical V2 command catalog once `propose`/`commit`
     are available
   - if still callable for existing installs, route it through `propose +
     auto-commit` only when policy allows
   - output should direct callers to `propose`/`commit`; it must not accept
     renamed-view aliases or non-canonical key prefixes

6. Add version conflict checks:
   - start with the current `_global` claim-count version
   - evolve toward per-subgraph counters from analytics or claim query
   - plan records carry `expected_subgraph_versions`

### Plan Store Persistence

The plan store must persist enough to commit without trusting the agent to resend
the payload. The safest first cut is to persist:

- original semantic request payload
- validation issues
- accepted/rejected/review-required operation summaries
- lowered mutation batch in a JSON-safe form
- provenance context in a JSON-safe form
- expected subgraph versions
- diff/preview
- expiry and status

If serializing `MutationBatch` is awkward, add explicit `to_dict` / `from_dict`
methods for `EntityUpsert`, `EdgeUpsert`, `InvalidationOp`, `MutationBatch`, and
`ProvenanceContext` rather than reparsing arbitrary agent payload at commit time.

### Acceptance Criteria

- `propose` never writes graph facts.
- `commit` accepts only a `plan_id`, never a mutation payload.
- A stale graph version causes `commit` to return `conflict` with expected and
  current versions.
- `mutate` cannot bypass validation, risk policy, canonical naming, or the same
  write door.
- Review-required operations have a real queue/status path instead of disappearing
  into a harness-only response.

### Tests

- Unit tests for plan status lifecycle: proposed, validated, review_required,
  conflict, committed, expired, abandoned.
- CLI tests proving `commit` cannot accept raw payloads.
- Regression tests proving a changed graph between propose and commit blocks
  stale plans.
- Backend conformance tests proving committed plans still apply through the one
  mutation port.

Implemented coverage on 2026-06-15:

- `tests/unit/test_graph_workbench_plans.py` verifies propose persistence
  without graph writes, commit-by-plan-id, stale-version conflict, medium-risk
  approval, and local JSON plan-store round trip of lowered plans.
- `tests/unit/test_graph_cli_contract.py` verifies `graph propose`,
  `graph commit`, legacy `graph mutate` error routing, and that commit rejects
  raw payload options.
- Existing graph surface, workbench ontology, backend conformance, and nudge
  tests continue to pass with the workbench write path in place.

Verification run:

```bash
uv run ruff check \
  potpie/context-engine/domain/graph_plans.py \
  potpie/context-engine/domain/ports/graph/plan_store.py \
  potpie/context-engine/adapters/outbound/graph/plan_stores/local_json.py \
  potpie/context-engine/adapters/outbound/graph/plan_stores/__init__.py \
  potpie/context-engine/application/services/graph_workbench.py \
  potpie/context-engine/bootstrap/host_wiring.py \
  potpie/context-engine/host/shell.py \
  potpie/context-engine/host/daemon_client.py \
  potpie/context-engine/adapters/inbound/cli/commands/graph.py \
  potpie/context-engine/domain/graph_workbench_ontology.py \
  potpie/context-engine/tests/unit/test_graph_workbench_plans.py \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py

uv run pytest -q \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py \
  potpie/context-engine/tests/unit/test_graph_workbench_plans.py \
  potpie/context-engine/tests/unit/test_graph_surface_lite_contract.py \
  potpie/context-engine/tests/conformance/test_graph_surface_lite_e2e.py \
  potpie/context-engine/tests/conformance/test_graph_backend_conformance.py \
  potpie/context-engine/tests/unit/test_graph_workbench_ontology.py \
  potpie/context-engine/tests/unit/test_graph_views.py \
  potpie/context-engine/tests/unit/test_nudge_service.py \
  potpie/context-engine/tests/unit/test_daemon_lifecycle_runtime.py
```

Result: Ruff passed; focused tests passed (`112 passed`).

## Phase 4: Add History

Goal: agents and users can inspect what changed, why, and from which source.

### Status

Done on 2026-06-15 for the local CLI workbench history surface.

- [x] Added `domain/graph_history.py` with request, entry, and result DTOs for
  read-only plan/claim history.
- [x] Extended `GraphPlanStorePort` and the local JSON plan store with bounded
  plan listing by plan id, mutation id, and time window.
- [x] Added `GraphWorkbenchService.history` to combine persisted plan records
  with committed claim rows from `ClaimQueryPort`.
- [x] Added exact claim-query filters for claim key, subgraph, and mutation id
  across in-memory, Neo4j, and FalkorDB readers.
- [x] Wired `potpie graph history --json` for `--entity`, `--claim`,
  `--subgraph`, `--plan`, `--mutation`, `--since`, `--until`, and `--limit`.

Deliberate first-cut choices:

- History remains read-only. Corrections still flow through `propose` and
  `commit`.
- `--plan` can answer from the local plan store even when the active backend
  cannot answer claim history. Claim-history capability gaps return structured
  `unsupported` entries instead of pretending the graph is empty.
- The first implementation uses stored plan records plus current/invalidated
  claim rows. There is still no separate backend-native mutation ledger.

### Changes

1. Add `graph history` with filters:
   - `--entity <key>`
   - `--claim <claim_key>`
   - `--subgraph <name>`
   - `--plan <plan_id>`
   - `--mutation <mutation_id>`
   - `--since` / `--until`

2. First implementation can combine:
   - plan store records
   - mutation ids returned by `GraphMutationPort.apply`
   - claim properties already stamped by semantic lowering
   - source refs, evidence, truth, validity fields, and created_by metadata

3. Later implementation can add backend-native claim history queries if needed.

### Acceptance Criteria

- After `commit`, `graph history --plan <plan_id>` shows validation, approval,
  commit result, mutation id, claim keys, and source refs.
- `graph history --entity <key>` shows current and invalidated claims involving
  that entity when the backend can answer it.
- History reads do not mutate graph state.

Implemented coverage on 2026-06-15:

- `tests/unit/test_graph_workbench_plans.py` verifies history by plan, mutation,
  and entity; local JSON plan-store listing; invalidated claim visibility; and
  read-only behavior.
- `tests/unit/test_graph_cli_contract.py` verifies `graph history --plan`
  returns the V2 workbench envelope.
- Claim-query reader and backend conformance tests continue to pass after adding
  exact history filters.

Verification run:

```bash
uv run ruff check \
  potpie/context-engine/domain/graph_history.py \
  potpie/context-engine/domain/ports/graph/plan_store.py \
  potpie/context-engine/adapters/outbound/graph/plan_stores/local_json.py \
  potpie/context-engine/domain/ports/claim_query.py \
  potpie/context-engine/adapters/outbound/graph/in_memory_reader.py \
  potpie/context-engine/adapters/outbound/graph/canonical_claim_query.py \
  potpie/context-engine/adapters/outbound/graph/neo4j_reader.py \
  potpie/context-engine/adapters/outbound/graph/falkordb_reader.py \
  potpie/context-engine/application/services/graph_workbench.py \
  potpie/context-engine/adapters/inbound/cli/commands/graph.py \
  potpie/context-engine/tests/unit/test_graph_workbench_plans.py \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py

uv run pytest -q \
  potpie/context-engine/tests/unit/test_graph_workbench_plans.py \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py \
  potpie/context-engine/tests/unit/test_neo4j_claim_query.py \
  potpie/context-engine/tests/unit/test_falkordb_reader.py \
  potpie/context-engine/tests/conformance/test_graph_backend_conformance.py \
  potpie/context-engine/tests/conformance/test_graph_surface_lite_e2e.py \
  potpie/context-engine/tests/unit/test_graph_surface_lite_contract.py
```

Result: Ruff passed; focused and reader/conformance tests passed (`102 passed`).

## Phase 5: Add Inbox

Goal: uncertain or incomplete graph work has a first-class pending state without
becoming canonical facts.

### Status

Done on 2026-06-15 for the local CLI workbench inbox workflow.

- [x] Added `domain/graph_inbox.py` with inbox item, status, and result DTOs.
- [x] Added `GraphInboxStorePort` and a local JSON-backed inbox store under the
  Potpie home.
- [x] Wired `GraphWorkbenchService` inbox methods for add, list, show, claim,
  mark-applied, mark-rejected, and close.
- [x] Wired `potpie graph inbox add/list/show/claim/mark-applied/mark-rejected/close --json`.
- [x] Updated the bundled graph skill to route uncertain graph work to
  `graph inbox add` and canonical writes through `propose`/`commit`.

Deliberate first-cut choices:

- Inbox items are local workbench state only. They are not exposed through graph
  read views or claim query.
- Closing an inbox item records a plan id, mutation id, or rejection reason.
- `mark-applied` and `mark-rejected` are terminal specialized close operations;
  `close` is a generic terminal close for superseded or manually resolved work.

### Changes

1. Add an inbox store port:

   ```text
   domain/ports/graph/inbox_store.py
   adapters/outbound/graph/inbox_stores/local_json.py
   adapters/outbound/postgres/graph_inbox_store.py later
   ```

2. Add `graph inbox` subcommands:
   - `add`
   - `list`
   - `show`
   - `claim`
   - `mark-applied`
   - `mark-rejected`
   - `close`

3. Inbox item fields:
   - `item_id`
   - `pot_id`
   - `status`
   - `summary`
   - `details`
   - `evidence`
   - `source_refs`
   - `suspected_subgraphs`
   - `created_by`
   - `created_at`
   - `claimed_by`
   - `closed_by`
   - `linked_plan_id`
   - `linked_mutation_id`

4. Do not let inbox items appear in ordinary graph reads as facts. They are
   pending work only.

### Acceptance Criteria

- Agents can add an inbox item when they have evidence but not enough certainty
  to write a canonical claim.
- Inbox processing requires `catalog`/`describe`/`read`/`search-entities`, then
  `propose`/`commit`.
- Closing an inbox item records the plan/mutation or rejection reason.

Implemented coverage on 2026-06-15:

- `tests/unit/test_graph_workbench_inbox.py` verifies add persistence without
  graph fact writes, claim/apply lifecycle, rejected terminal behavior,
  close validation, list filters, and local JSON round trip.
- `tests/unit/test_graph_cli_contract.py` verifies V2 envelopes and option
  routing for inbox add, list, claim, mark-applied, mark-rejected, and close.
- Existing plan/history tests continue to pass with the optional inbox store
  wired into `GraphWorkbenchService`.

Verification run:

```bash
uv run ruff check \
  potpie/context-engine/domain/graph_inbox.py \
  potpie/context-engine/domain/ports/graph/inbox_store.py \
  potpie/context-engine/adapters/outbound/graph/inbox_stores/__init__.py \
  potpie/context-engine/adapters/outbound/graph/inbox_stores/local_json.py \
  potpie/context-engine/application/services/graph_workbench.py \
  potpie/context-engine/bootstrap/host_wiring.py \
  potpie/context-engine/adapters/inbound/cli/commands/graph.py \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py \
  potpie/context-engine/tests/unit/test_graph_workbench_inbox.py

uv run pytest -q \
  potpie/context-engine/tests/unit/test_graph_workbench_inbox.py \
  potpie/context-engine/tests/unit/test_graph_workbench_plans.py \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py
```

Result: Ruff passed; focused inbox/plan/CLI tests passed (`45 passed`).
Additional broader graph slice passed (`124 passed`).

## Phase 6: Add Quality Workflows

Goal: long-lived pots can be inspected and repaired without raw graph surgery.

### Status

Done on 2026-06-15 for the local CLI workbench quality surface.

- [x] Added V2 quality result DTOs in `domain/graph_quality.py` for bounded
  read-only findings with deterministic `finding_id`, severity, affected
  entities/claims, source refs, evidence, suggested action, metrics, filters,
  and unsupported capability reporting.
- [x] Added `GraphWorkbenchService.quality` with report dispatch for
  `summary`, `duplicate-candidates`, `stale-facts`, `conflicting-claims`,
  `orphan-entities`, `low-confidence`, and `projection-drift`.
- [x] Backed `summary` with cheap analytics counters/freshness/quality.
- [x] Backed bounded finding reports with `ClaimQueryPort` and
  `GraphInspectionPort` where available, including duplicate display-name
  candidates, stale validity markers, missing required evidence, low confidence,
  singleton/predicate-family conflicts, orphan entities with no live useful
  claims, invalid endpoint pairs, and inspection projection drift.
- [x] Wired `potpie graph quality <report> --json` subcommands.
- [x] Updated the bundled graph skill to route canonical repairs through
  `graph propose`/`graph commit` and uncertain repairs through `graph inbox add`.

Deliberate first-cut choices:

- Quality findings are ephemeral/read-only workbench output. There is no quality
  finding lifecycle store yet.
- Duplicate detection is conservative: it groups entities by label and
  normalized stored display name, not fuzzy semantic similarity.
- Orphan detection reports entities that are only connected by invalidated
  claims through the current claim-query surface. A future entity inventory port
  can extend this to completely standalone entity upserts.
- `graph repair` remains operator projection maintenance; semantic corrections
  still flow through proposals.

### Changes

1. Add `graph quality`:
   - `summary`
   - `duplicate-candidates`
   - `stale-facts`
   - `conflicting-claims`
   - `orphan-entities`
   - `low-confidence`
   - `projection-drift`

2. Start from existing analytics and `domain/graph_quality.py`, then add scanners
   over `ClaimQueryPort` and `GraphInspectionPort`:
   - duplicate entity candidates
   - claims with missing evidence where required
   - invalid endpoint pairs from older data
   - stale source refs
   - orphan entities with no useful claims
   - conflicting singleton predicates
   - projection/index drift

3. Quality findings are read-only until repaired through proposals:
   - no direct repair writes from `quality`
   - suggested repairs become semantic mutations or inbox items
   - risky repairs require review/approval

4. Keep `graph repair` as operator projection maintenance:
   - semantic index rebuild
   - entity summary rebuild
   - snapshot/projection repair
   - not semantic fact correction

### Acceptance Criteria

- `graph quality summary` is cheap and works on local default backends.
- Duplicate/stale/conflict/orphan reports return bounded, source-backed findings.
- Repairs that alter canonical facts flow through `propose`/`commit`.

Implemented coverage on 2026-06-15:

- `tests/unit/test_graph_workbench_quality.py` verifies summary, duplicate
  candidates, stale facts, low-confidence/missing-evidence findings, singleton
  conflicts, orphan entities, invalid endpoint pairs, and read-only behavior.
- `tests/unit/test_graph_cli_contract.py` verifies `graph quality summary` and
  `graph quality low-confidence` route through the V2 workbench envelope and
  pass report filters.

Verification run:

```bash
uv run ruff check \
  potpie/context-engine/domain/graph_quality.py \
  potpie/context-engine/application/services/graph_workbench.py \
  potpie/context-engine/adapters/inbound/cli/commands/graph.py \
  potpie/context-engine/tests/unit/test_graph_workbench_quality.py \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py

uv run pytest -q \
  potpie/context-engine/tests/unit/test_graph_workbench_quality.py \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py
```

Result: Ruff passed; focused quality/CLI tests passed (`38 passed`).

## Phase 7: Complete V2 Mutation Vocabulary

Goal: implement the V2 operations that were deferred or review-only in V1.5.

### Cuts

#### Phase 7A: Safe Entity Corrections

Implement the two previously deferred entity-local operations through the
plan-store workflow:

1. `patch_entity`
   - restrict to metadata fields allowed by entity contract
   - require expected entity version when available
   - do not overwrite retrieval-grade descriptions with weaker values

2. `transition_state`
   - only for entity/relation types with declared lifecycle states
   - require valid state transition contract
   - preserve previous state in history

Catalog promotion for 7A is allowed only after validation, lowering, commit,
history, and tests exist for both operations.

#### Phase 7B: Snapshot And High-Risk Corrections

Implement the remaining correction workflows after 7A lands:

1. `reconcile_snapshot`
   - only for trusted snapshot sources
   - distinguish complete vs partial snapshot
   - complete trusted snapshots may end disappeared facts
   - partial snapshots must not remove absent facts
   - large production removals require review

2. `supersede_claim`
   - implemented through plan store and history
   - end old claim validity and link to new claim

3. `merge_duplicate_entities`
   - identity-record merge with external IDs and merge history
   - never hard-delete the losing key in the first cut
   - redirect losing canonical keys only through explicit merge records, not
     open-ended alias lookup

This cut implemented `supersede_claim` and `merge_duplicate_entities`.
`reconcile_snapshot` remains deferred.

Result: Ruff passed; focused semantic mutation, contract, workbench, and
Graph Surface Lite tests passed (`160 passed`).

### Acceptance Criteria

- The catalog only moves an operation from deferred/review-required to applicable
  after validation, lowering, commit, history, and tests exist.
- High-risk operations require approval and survive conflict checks.
- Corrections are auditable through `history`.

## Phase 8: Skills, Templates, And Canonical Workflow

Goal: agent harnesses learn the canonical V2 workflow and stop advertising old
view names, subgraph names, or key-prefix aliases.

### Status

Done on 2026-06-15 for bundled skills, harness templates, command shortcuts, and
template drift tests. This phase did not change graph storage, ontology
lowering, backend ports, or MCP tool count.

### Implementation Plan

#### Phase 8A: Template Inventory And Drift Guards

1. Treat the bundled harness templates as the implementation surface:
   - `adapters/inbound/cli/templates/claude_plugin/skills/`
   - `adapters/inbound/cli/templates/agent_bundle/.agents/skills/`
   - `adapters/inbound/cli/templates/claude_bundle/.claude/skills/`
   - root harness docs: `AGENTS.md`, `CLAUDE.md`
   - Claude command templates: `potpie-feature.md`, `potpie-record.md`

2. Update the seven graph-use-case skills first:
   - `potpie-graph`
   - `potpie-project-preferences`
   - `potpie-infra-architecture`
   - `potpie-change-timeline`
   - `potpie-debug-memory`
   - `potpie-repo-baseline`
   - `potpie-source-ingestion`

3. Keep the adjacent CLI helper aligned with graph examples:
   - `potpie-cli`

4. Preserve deliberate bundle invariants:
   - `potpie-graph` stays byte-identical across `agent_bundle`,
     `claude_bundle`, and `claude_plugin`.
   - Skills present in both `agent_bundle` and `claude_plugin` should either be
     byte-identical or have an explicit test-documented reason for divergence.

#### Phase 8B: Normalize The Canonical Read Workflow

Update every recommended graph workflow to this V2 sequence:

```bash
potpie --json graph status
potpie --json graph catalog --task "<task>"
potpie --json graph describe <subgraph> --view <view> --examples
potpie --json graph search-entities "<name>" --type <Type> --limit 10
potpie --json graph read --subgraph <subgraph> --view <view> --scope <scope>
```

Concrete command migration rules:

- Use `--subgraph decisions --view preferences_for_scope`, not
  `--view decisions.preferences_for_scope`.
- Use `--subgraph debugging --view prior_occurrences`, not
  `--view bugs.prior_occurrences` or `--view debugging.prior_occurrences`.
- Use `--subgraph features --view feature_context`, not stale feature view names
  such as `features.provided`.
- Prefer `graph read --subgraph recent_changes --view timeline` in canonical
  workflow docs. `potpie timeline recent` can be mentioned only as a legacy or
  convenience shortcut outside the V2 workbench envelope.
- Keep canonical entity-key examples on underscore prefixes and current V2
  subgraph names only.

#### Phase 8C: Normalize The Canonical Write Workflow

Replace every recommended `context_record`, `graph mutate`, and
`graph mutate --dry-run` write path with:

```bash
potpie --json graph propose --file mutation.json
# inspect result.status, risk, diff, warnings, rejected_operations, plan_id
potpie --json graph commit <plan_id>
potpie --json graph history --plan <plan_id>
```

Write guidance rules:

- `graph propose` is the validation/preview step. Skills should not tell agents
  to use `graph mutate --dry-run`.
- `graph commit` accepts a server-created `plan_id`, not a resent payload.
- Medium- and high-risk applicable plans need
  `graph commit <plan_id> --approved-by <user-ref>` according to the plan result.
- Use `graph inbox add` when evidence exists but the harness cannot safely pick
  the canonical ontology update.
- Keep semantic mutation payload examples batch-shaped and data-plane compatible
  with current accepted mutation payload versions. The workbench envelope is V2,
  but the semantic payload parser still accepts the V1.5 data-plane contract.
- Tell skills to trust `graph catalog` for the current mutation operation
  partition instead of hard-coding stale deferred/review-required lists.
- Repairs from `graph quality` must become `propose`/`commit` plans or inbox
  items. `graph repair` remains operator projection maintenance only.

#### Phase 8D: Rewrite Use-Case Skill Recipes

1. `potpie-graph`
   - Make it the concise canonical contract for status, catalog, describe,
     search, read, propose, commit, history, inbox, and quality.
   - Remove stale statements that `patch_entity`, `transition_state`,
     `supersede_claim`, or `merge_duplicate_entities` are unavailable if the
     current catalog advertises them as applicable.
   - Keep retrieval-grade description, evidence, truth-class, and nudge
     guidance.

2. `potpie-project-preferences`
   - Read preferences through `decisions.preferences_for_scope` using
     `--subgraph decisions --view preferences_for_scope`.
   - Record new preferences with `graph propose` and `graph commit`.
   - Mention `context_record` only as an MCP compatibility fallback.

3. `potpie-debug-memory`
   - Read prior bugs through `--subgraph debugging --view prior_occurrences`.
   - Correlate recent changes through `recent_changes.timeline` and topology
     through `infra_topology.service_neighborhood`.
   - Record bug patterns, fixes, investigations, and verifications through
     proposals; use inbox for uncertain learnings.

4. `potpie-infra-architecture`
   - Use `--subgraph infra_topology --view service_neighborhood`.
   - Replace "run `--dry-run`" with "run `graph propose`, inspect the diff and
     risk, then commit with approval when required."

5. `potpie-change-timeline`
   - Make `append_event` proposals the canonical write path.
   - Keep top-level `timeline recent` only as a convenience shortcut, not the
     canonical V2 graph workflow.

6. `potpie-repo-baseline`
   - Replace `graph mutate --dry-run` with `graph propose`.
   - Query baselines back with canonical `--subgraph features --view
     feature_context` and `--subgraph infra_topology --view service_neighborhood`.
   - Preserve the no-scanner, authored-source-first, evidence/truth/description
     requirements.

7. `potpie-source-ingestion`
   - Make source ingestion end in `graph propose`/`graph commit` or inbox, not
     direct mutate.
   - Keep repository baseline and change-history as separate harness-led passes.

8. `potpie-cli`
   - Keep setup, search, source, pot scope, and graph workbench commands compact.
   - Fold CLI troubleshooting into this skill instead of pointing to helper
     skills.

#### Phase 8E: Update Static Template Tests

1. Rename or reword `test_agent_templates_v15.py` around the V2 template
   contract.

2. Add negative drift checks that fail if recommended template text contains:
   - `graph mutate --dry-run`
   - `graph mutate --file` outside an explicitly labeled legacy section
   - `context_record` outside an explicitly labeled MCP/legacy section
   - `--view <subgraph>.<view>` command examples without `--subgraph`
   - obsolete public view names such as `bugs.prior_occurrences`,
     `preferences.active_preferences`, or `features.provided`

3. Add positive checks that recommended skills include:
   - `graph status`
   - `graph catalog --task`
   - `graph describe`
   - `graph search-entities`
   - `graph read --subgraph`
   - `graph propose`
   - `graph commit`
   - `graph history`
   - `graph inbox`

4. Extend existing drift tests:
   - `test_claude_plugin_manifest.py` keeps `potpie-graph` identical across all
     bundles.
   - `test_repo_baseline_skill.py` verifies both bundle copies use the V2
     propose/commit workflow and canonical read syntax.
   - Add one shared-skill parity test for all skills shipped in both
     `agent_bundle` and `claude_plugin`.
   - Keep `test_agent_surface_contract.py` asserting exactly the four existing
     MCP tools until a separate MCP V2 decision is made.

### Acceptance Criteria

- Installed skills teach the V2 graph workflow as the recommended path:
  status, catalog, describe, search, read, propose, commit, verify.
- No recommended skill presents `graph mutate`, `graph mutate --dry-run`,
  `context_record`, or `timeline recent` as the canonical V2 workflow.
- Wrapper docs label `context_resolve`, `context_search`, `context_record`,
  `context_status`, and `graph mutate` as legacy or MCP compatibility paths only.
- Template examples use canonical subgraph/view syntax and current public names.
- Bundled skill copies do not drift silently across harness bundles.
- MCP still exposes exactly the intended four tools until a separate MCP V2
  decision is made.

### Tests

Run the focused template and agent-surface suite:

```bash
uv run pytest -q \
  potpie/context-engine/tests/unit/test_agent_templates_v15.py \
  potpie/context-engine/tests/unit/test_claude_plugin_manifest.py \
  potpie/context-engine/tests/unit/test_repo_baseline_skill.py \
  potpie/context-engine/tests/unit/test_agent_surface_contract.py \
  potpie/context-engine/tests/unit/test_agent_installer.py \
  potpie/context-engine/tests/unit/test_bundle_catalog.py \
  potpie/context-engine/tests/unit/test_skill_manager_global_targets.py
```

Implemented coverage on 2026-06-15:

- Updated `potpie-graph` across `agent_bundle`, `claude_bundle`, and
  `claude_plugin` to teach status, catalog, describe, read, search, propose,
  commit, history, inbox, and quality as the canonical V2 workflow.
- Updated use-case skills for preferences, infra, timeline, debug memory,
  source ingestion, and repo baseline to use `--subgraph --view` read syntax and
  `graph propose` / `graph commit` writes.
- Updated helper skills, root harness docs, and Claude command templates to stop
  advertising `graph mutate --dry-run`, fully-qualified `--view` args, obsolete
  public view names, and top-level timeline shortcuts as canonical.
- Kept `context_*` tools as MCP compatibility wrappers only, and kept the MCP
  surface test pinned to exactly four tools.
- Added drift tests for V2 write workflow, canonical view syntax,
  `context_record` compatibility framing, and shared skill bundle parity.

Verification run:

```bash
uv run pytest -q \
  potpie/context-engine/tests/unit/test_agent_templates_v15.py \
  potpie/context-engine/tests/unit/test_claude_plugin_manifest.py \
  potpie/context-engine/tests/unit/test_repo_baseline_skill.py \
  potpie/context-engine/tests/unit/test_agent_surface_contract.py \
  potpie/context-engine/tests/unit/test_agent_installer.py \
  potpie/context-engine/tests/unit/test_bundle_catalog.py \
  potpie/context-engine/tests/unit/test_skill_manager_global_targets.py
```

Result: focused template and skill tests passed (`75 passed`).

## Phase 9: Observability And Backend Conformance

Goal: V2 behavior is measurable and backend-independent.

### Status

Done on 2026-06-16 for local CLI workbench observability, readiness
actionability, and runnable-backend workbench conformance.

- [x] Added Graph V2 workbench spans and counters through the existing
  observability port for `graph.status`, `graph.catalog`, `graph.describe`,
  `graph.search_entities`, `graph.read`, `graph.propose`, `graph.commit`,
  `graph.history`, `graph.inbox`, and `graph.quality`.
- [x] Kept local OSS dark by default through the existing `NoOpObservability`;
  console/OTLP telemetry still requires explicit environment configuration.
- [x] Added `graph status` readiness next-action metadata that points operators
  at `potpie backend doctor` when backend readiness is false.
- [x] Extended `potpie doctor` JSON output with backend readiness,
  capability-readiness details, active pot, and an actionable next step.
- [x] Added workbench conformance coverage for runnable backend profiles
  (`in_memory`, `embedded`) across propose, commit, history, inbox, and quality.
  Existing capability-gated tests continue to prove unsupported inspection and
  snapshot commands fail closed with structured unsupported envelopes.

### Changes

1. Emit spans/counters listed in [`observability.md`](./observability.md):
   - `graph.status`
   - `graph.catalog`
   - `graph.describe`
   - `graph.search_entities`
   - `graph.read`
   - `graph.propose`
   - `graph.commit`
   - `graph.history`
   - `graph.inbox`
   - `graph.quality`

2. Add conformance tests for every backend profile that claims support:
   - in-memory
   - embedded
   - falkordb_lite
   - falkordb
   - hosted later

3. Preserve capability-gated behavior:
   - if a backend lacks inspection, `neighborhood` returns structured unsupported
   - if a backend lacks snapshot, export/import return structured unsupported
   - if semantic index is missing, reads/search report lexical mode

### Acceptance Criteria

- `potpie doctor` and `graph status` make readiness failures actionable.
- Backends cannot claim a capability without passing the workbench conformance
  test for that capability.
- Local OSS remains useful without hosted telemetry.

Implemented coverage on 2026-06-16:

- `tests/unit/test_graph_cli_contract.py` verifies V2 graph command telemetry
  for read, propose, inbox, and quality paths, plus `graph status` readiness
  next-action metadata.
- `tests/unit/test_cli_bootstrap_status.py` verifies `potpie doctor --json`
  includes backend readiness, capability readiness, active pot, and next action.
- `tests/conformance/test_graph_backend_conformance.py` verifies runnable
  backend profiles satisfy the workbench propose/commit/history/inbox/quality
  workflow.
- Existing observability unit tests continue to verify the no-op default,
  console adapter, runtime wiring, tracing, cost-metric bridge, readiness
  probes, and JSON logging.

Verification run:

```bash
uv run pytest -q \
  potpie/context-engine/tests/unit/test_graph_cli_contract.py \
  potpie/context-engine/tests/unit/test_cli_bootstrap_status.py \
  potpie/context-engine/tests/conformance/test_graph_backend_conformance.py \
  potpie/context-engine/tests/unit/test_observability.py \
  potpie/context-engine/tests/unit/test_observability_tracing.py \
  potpie/context-engine/tests/unit/test_observability_cost_metrics.py \
  potpie/context-engine/tests/unit/test_observability_operate.py
```

Result: focused observability/readiness/conformance tests passed (`86 passed`).

## Suggested Work Packets

These are the smallest useful implementation PRs. Status is as of
2026-06-16.

| Packet | Status | Scope | Main files |
|---|---|---|---|
| V2-00 Envelope | Done | Add workbench envelope DTO/builder and wrap `status`/`catalog` first. | `domain/graph_workbench.py`, `application/services/graph_workbench.py`, `adapters/inbound/cli/commands/graph.py`, tests |
| V2-01 Describe | Done | Add executable subgraph/view contracts and `graph describe`. | `domain/graph_workbench_ontology.py`, `domain/graph_views.py`, `application/services/graph_workbench.py`, tests |
| V2-02 Read Shape | Done | Add `--subgraph --view`, normalized read results, unsupported filter reporting. | `graph.py`, `graph_service.py`, readers, tests |
| V2-03 Neighborhood | Done | Promote bounded traversal out of `inspect` into `graph neighborhood`. | `graph.py`, `GraphInspectionPort` adapters, conformance tests |
| V2-04 Propose | Done | Persist validated/lowered mutation plans without applying them. | `semantic_mutations.py`, validator/lowerer serializers, new plan store, tests |
| V2-05 Commit | Done | Commit by `plan_id`, handle expiry/conflict/approval/history pointer. | plan store, `graph_workbench.py`, `graph.py`, mutation tests |
| V2-06 History | Done | Query plan/mutation/claim history. | plan store, claim query helpers, CLI, tests |
| V2-07 Inbox | Done | Add inbox storage and CLI workflow. | inbox store, `graph.py`, skills, tests |
| V2-08 Quality | Done | Add quality command and findings. | `domain/graph_quality.py`, analytics/claim query scanners, CLI, tests |
| V2-09A Entity Correction Ops | Done | Implement `patch_entity` and `transition_state` under plan workflow. | contract, ontology, validator, lowerer, plan/commit/history tests |
| V2-09B Snapshot And High-Risk Corrections | Done | Implement `supersede_claim` and `merge_duplicate_entities`; defer `reconcile_snapshot`. | contract, validator, lowerer, approval/history tests |
| V2-10 Skills | Done | Update bundled skills and canonical workflow docs. | skill templates, `CLAUDE.md`, tests |
| V2-11 Observability And Backend Conformance | Done | Add workbench telemetry and capability conformance coverage. | observability hooks, backend conformance tests, `graph status` |

## Current Next Cut

No remaining V2 workbench cut is currently planned. Keep `reconcile_snapshot`
deferred until trusted snapshot source policy is explicit.

## Resolved Decisions

1. **Workbench service boundary**: implemented as `GraphWorkbenchService`, a
   facade above the existing `GraphService` data plane and backend capability
   ports. The CLI stays thin and routes propose/commit through HostShell.

2. **Contract version bump**: the CLI workbench envelope reports V2. The
   underlying data-plane catalog remains `v1.5`, exposed as
   `data_plane_graph_contract_version`.

3. **Ontology naming**: canonical public subgraphs are `decisions`,
   `debugging`, `recent_changes`, `infra_topology`, `features`,
   `code_topology`, `knowledge`, and `admin`.

4. **Entity key conventions**: underscore key prefixes remain canonical.
   Hyphenated prefix aliases are rejected; external IDs stay separate from graph
   keys.

5. **Plan store location**: first cut uses local JSON under the Potpie home.
   Hosted or multi-process installs should add a state DB implementation with
   leases/transactions behind `GraphPlanStorePort`.

6. **Approval mechanics**: low-risk validated plans can commit locally without
   approval. Medium- and high-risk applicable plans require
   `graph commit <plan_id> --approved-by <user-ref>`. Review-required or
   deferred operations remain persisted but uncommitted until their validation,
   lowering, and history support exists.
