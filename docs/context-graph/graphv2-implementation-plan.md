# Graph V2 Implementation Plan

Last reviewed: 2026-06-15.

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
| Applicable | `append_event`, `upsert_entity`, `link_entities`, `assert_claim`, `end_relation_validity`, `retract_claim` |
| Review-required | `supersede_claim`, `merge_duplicate_entities` |
| Deferred | `patch_entity`, `transition_state` |

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

## Phase 3: Split Mutate Into Propose And Commit

Goal: `propose` validates and persists a plan; `commit` applies exactly that
server-created plan by id.

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

## Phase 4: Add History

Goal: agents and users can inspect what changed, why, and from which source.

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

## Phase 5: Add Inbox

Goal: uncertain or incomplete graph work has a first-class pending state without
becoming canonical facts.

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

## Phase 6: Add Quality Workflows

Goal: long-lived pots can be inspected and repaired without raw graph surgery.

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

## Phase 7: Complete V2 Mutation Vocabulary

Goal: implement the V2 operations that were deferred or review-only in V1.5.

### Order

1. `patch_entity`
   - restrict to metadata fields allowed by entity contract
   - require expected entity version when available
   - do not overwrite retrieval-grade descriptions with weaker values

2. `transition_state`
   - only for entity/relation types with declared lifecycle states
   - require valid state transition contract
   - preserve previous state in history

3. `reconcile_snapshot`
   - only for trusted snapshot sources
   - distinguish complete vs partial snapshot
   - complete trusted snapshots may end disappeared facts
   - partial snapshots must not remove absent facts
   - large production removals require review

4. `supersede_claim`
   - implemented through plan store and history
   - end old claim validity and link to new claim

5. `merge_duplicate_entities`
   - identity-record merge with external IDs and merge history
   - never hard-delete the losing key in the first cut
   - redirect losing canonical keys only through explicit merge records, not
     open-ended alias lookup

### Acceptance Criteria

- The catalog only moves an operation from deferred/review-required to applicable
  after validation, lowering, commit, history, and tests exist.
- High-risk operations require approval and survive conflict checks.
- Corrections are auditable through `history`.

## Phase 8: Skills, Templates, And Canonical Workflow

Goal: agent harnesses learn the canonical V2 workflow and stop advertising old
view names, subgraph names, or key-prefix aliases.

### Changes

1. Update bundled skills under
   `adapters/inbound/cli/templates/claude_plugin/skills/`:
   - `potpie-graph`
   - `potpie-project-preferences`
   - `potpie-infra-architecture`
   - `potpie-change-timeline`
   - `potpie-debug-memory`
   - `potpie-repo-baseline`
   - `potpie-source-ingestion`

2. V2 skill workflow:
   - `graph status`
   - `graph catalog --task`
   - `graph describe`
   - `graph search-entities`
   - `graph read`
   - `graph propose`
   - inspect plan
   - `graph commit`
   - verify with `history` or `read`

3. Document V1 wrappers only as legacy escape hatches, not as the recommended
   graph workflow:
   - `context_resolve`
   - `context_search`
   - `context_record`
   - `context_status`
   - `graph mutate`
   These wrappers must not introduce compatibility aliases for renamed V2 views
   or non-canonical key prefixes.

4. Do not add a second graph API in skills. Skills explain when to use the CLI;
   the CLI/service enforces graph integrity.

### Acceptance Criteria

- Installed skills do not mention obsolete shortcut workflows as canonical.
- Tests that check bundled skill drift are updated.
- MCP still exposes exactly the intended tools until a separate MCP V2 decision
  is made.

## Phase 9: Observability And Backend Conformance

Goal: V2 behavior is measurable and backend-independent.

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

## Suggested Work Packets

These are the smallest useful implementation PRs.

| Packet | Scope | Main files |
|---|---|---|
| V2-00 Envelope | Add workbench envelope DTO/builder and wrap `status`/`catalog` first. | `domain/graph_workbench.py`, `application/services/graph_workbench.py`, `adapters/inbound/cli/commands/graph.py`, tests |
| V2-01 Describe | Add executable subgraph/view contracts and `graph describe`. | `domain/graph_workbench_ontology.py`, `domain/graph_views.py`, `application/services/graph_workbench.py`, tests |
| V2-02 Read Shape | Add `--subgraph --view`, normalized read results, unsupported filter reporting. | `graph.py`, `graph_service.py`, readers, tests |
| V2-03 Neighborhood | Promote bounded traversal out of `inspect` into `graph neighborhood`. | `graph.py`, `GraphInspectionPort` adapters, conformance tests |
| V2-04 Propose | Persist validated/lowered mutation plans without applying them. | `semantic_mutations.py`, validator/lowerer serializers, new plan store, tests |
| V2-05 Commit | Commit by `plan_id`, handle expiry/conflict/approval/history pointer. | plan store, `graph_workbench.py`, `graph.py`, mutation tests |
| V2-06 History | Query plan/mutation/claim history. | plan store, claim query helpers, CLI, tests |
| V2-07 Inbox | Add inbox storage and CLI workflow. | inbox store, `graph.py`, skills, tests |
| V2-08 Quality | Add quality command and findings. | `domain/graph_quality.py`, analytics/claim query scanners, CLI, tests |
| V2-09 Deferred Ops | Implement patch/transition/reconcile/supersede/merge under plan workflow. | contract, validator, lowerer, plan/commit/history tests |
| V2-10 Skills | Update bundled skills and canonical workflow docs. | skill templates, `CLAUDE.md`, tests |

## Recommended First Cut

Start with V2-00 through V2-02:

1. Add the V2 envelope and wrap `graph status` plus `graph catalog`.
2. Add executable contracts and `graph describe`.
3. Normalize `graph read --subgraph --view`; keep `--view subgraph.view` only for
   canonical fully-qualified names.

That gives agents a stable discovery/read loop before changing write behavior.
Then implement V2-04 and V2-05 together, because `propose` without `commit` is
useful for validation but not enough to replace `mutate`.

## Open Decisions

1. **Workbench service boundary**: this plan recommends a `GraphWorkbenchService`
   facade over `GraphService`. If the team prefers extending `GraphService`
   directly, keep the same DTOs and envelope builder so the CLI stays thin.

2. **Contract version bump**: do not set `GRAPH_CONTRACT_VERSION = "v2"` until
   the envelope, command set, describe/read shape, propose, and commit are
   stable. Use `v2-beta` or advertise V2 support in catalog metadata if needed
   before the formal bump.

3. **Ontology naming**: canonicalize the current public graph surface into the
   broader V2 seed ontology: `preferences` moves under `decisions`, `bugs` moves
   under `debugging`, `docs` moves under `knowledge`, and `ownership` moves under
   `code_topology`.

4. **Entity key conventions**: current underscore key prefixes remain canonical.
   Do not add hyphenated prefix aliases; store external IDs separately from graph
   keys.

5. **Plan store location**: local JSON under the Potpie home is enough for the
   first local workbench. Hosted or multi-process installs should use a state DB
   implementation with leases/transactions.

6. **Approval mechanics**: decide whether medium/high-risk local commits use
   `--approved-by`, interactive confirmation, a signed local approval record, or
   all of the above. The plan store should model approval independently of CLI UX.
