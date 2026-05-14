# Context Engine Rebuild Plan

This is the active execution plan for getting the Context Engine to its target shape. It is opinionated about direction and intentionally vague about exact code — every phase begins with its own discovery step that re-validates the current state and commits to specifics.

## How to use this plan

- Phases are sequential. Each phase ends with explicit exit criteria. Do not start phase N+1 until phase N's exit criteria are met.
- **Every phase begins with its own discovery step.** Do not trust assumptions from earlier phases or earlier architecture reviews. Read the current code. Re-validate. Then commit to specifics. The plan describes *direction*, not implementation. Decisions belong in the phase that makes them, not in this file.
- **No backwards compatibility.** Delete code that doesn't fit. Delete docs that don't fit. Rename things to be clearer. The cost of breakage is low; the cost of a confused codebase and confused agents is high.
- **Doc updates are part of every phase, not an afterthought.** Docs are the long-term source of truth for "what is this project, where is it headed." Each phase ends by reconciling docs with what was built.
- If a phase's discovery reveals the original direction was wrong, **stop, write down what changed, update this plan, and replan**. This file is the working plan; rewrite phases as understanding deepens.

## Anti-goals (apply to every phase)

- No second graph store. No parallel graph abstraction. One graph, one substrate.
- No public agent tools beyond `context_resolve`, `context_search`, `context_record`, `context_status`. New use cases become parameters (`intent`, `include`, `scope`, `mode`, `source_policy`), not new tools.
- No "compatibility" code paths. If something is replaced, the old path is deleted in the same phase.
- No source-specific code (GitHub, Linear, Slack, …) in the application layer. Source-specific behavior lives in connectors.
- No full source payloads stored in the graph. References + compact canonical summaries only. Source systems remain authoritative for their own data.
- No new use case file unless it represents a real new verb. Don't grow `application/use_cases/`; collapse it.

## Target end-state (one paragraph)

A single hexagonal Context Engine with:

- One write path: every change to the graph is a `ReconciliationPlan` produced by either a deterministic source connector or the reconciliation agent, validated against the Potpie ontology, applied through Graphiti as the sole writer into Neo4j.
- One read path: a `ContextReader` registry where each evidence family (decisions, change history, debugging memory, project map, ownership, …) is an isolated, capability-declaring reader; the planner is a thin router.
- One integration contract: a `SourceConnector` plugin port that subsumes listing, webhook normalization, source resolution, and plan proposal. Adding GitHub, Linear, Slack, Notion, Sentry, PagerDuty is a connector module — never an application-layer change.
- Four agent tools, parameter-driven, returning the same envelope (`answer / evidence / source_refs / coverage / freshness / quality / fallbacks / open_conflicts / recommended_next_actions`).
- Lean, accurate docs that describe the system as it actually is, with a clear separation between vision, current architecture, and contributor onboarding.

---

## Phase 0 — Doc reset and source-of-truth lock

**Goal:** Make the docs match the system, strip dead pointers, and establish the small set of canonical docs that every later phase will keep current.

### Discovery
- List every file under `docs/context-graph/`. Confirm which referenced files in `README.md` actually exist on disk and which are broken pointers.
- Read `graph.md`, `features-and-functionalities.md`, `feature-test-matrix.md` end to end. For each section, decide: load-bearing vision (keep), accurate current architecture (keep, possibly trim), historical / aspirational / superseded (delete).
- Check `app/src/context-engine/` against the docs. Where does the doc claim something the code does not do? Where does the code do something the docs do not describe?

### Decisions to make this phase
- Final list of canonical docs (target: ≤ 5 files).
- Where the agent contract envelope lives (one file, one place).
- Where the canonical ontology lives in docs vs. code (recommend: code is the source of truth; docs link to it).

### Implementation
Recommended target structure (re-validate during discovery; adjust if the discovery reveals a better split):

- `docs/context-graph/README.md` — one-screen index; nothing else.
- `docs/context-graph/vision.md` — what the Context Engine is, what problem it solves, the pot model, source-reference-first principle, the four-tool agent port, anti-goals. Load-bearing only. No implementation detail.
- `docs/context-graph/architecture.md` — the system as it currently is: ports, ingestion path, query path, Graphiti's role, the Potpie ontology layer, the source resolver layer. Code is the source of truth; this doc explains the *why* and points to the files.
- `docs/context-graph/agent-contract.md` — the `context_resolve` / `context_search` / `context_record` / `context_status` request/response contracts and the include-family catalog. The single source of truth for agent integrators.
- `docs/context-graph/extending.md` — how to add a new source connector, a new context reader, a new include family. Empty stub now; filled in as Phase 2 and Phase 3 land.
- `docs/context-graph/plan.md` — this file.

### Cleanup
- Delete all docs that are not in the target structure. Do not soft-archive. Do not link them as "historical." Delete.
- Delete every reference to docs that no longer exist (today: `unified-graphiti-application-architecture.md`, `implementation-next-steps.md`, `planning-next-steps.md`, `timeline.md`, `testing-and-bugs.md`).
- Strip `graph.md` of every aspirational section that is not implemented and is not committed to. If something is "future direction," it goes into `vision.md` as a principle, not a spec.

### Exit criteria
- A new contributor can read the canonical docs in under 30 minutes and explain the system back.
- Every code path in `app/src/context-engine/` is either described in `architecture.md` or marked for deletion in a later phase.
- No dead links in the docs directory.

---

## Phase 1 — Single write path (Graphiti is the only writer)

**Goal:** Every write into the graph goes through Graphiti's APIs. Direct Neo4j Cypher writes are deleted. The `StructuralGraphPort` is either retired or restricted to reads.

### Discovery
- Audit every direct Neo4j writer. Search for `session.run` / `tx.run` / `MERGE ` / `CREATE ` outside of Graphiti adapters. List every file that mutates Neo4j without going through Graphiti.
- For each direct writer, decide: replace with `add_triplet` / `EntityNode.save` / `EntityEdge.save` / `add_episode`, or delete entirely if the mutation is no longer needed.
- Confirm Graphiti version and feature support (custom entity types, edge type maps, fact_triple ingestion, namespacing). Re-validate that the substrate can express every canonical mutation Potpie needs. If not, document the specific gap and decide: extend Graphiti via PR, escape via `add_triplet`, or carve a justified exception.
- Identify whether reads should also migrate now. Recommend yes if cheap, defer to a later phase if expensive.

### Decisions to make this phase
- The exact Graphiti API surface every canonical mutation lowers to. Document the mapping in `architecture.md`.
- Whether `StructuralGraphPort` is deleted entirely or kept read-only.
- Whether to adopt Graphiti's `fact_triple` path for deterministic source events (PRs, issues) up front or in a later cost-optimization phase. (Recommendation: not yet; do it once telemetry says it matters.)

### Implementation
- Implement the mutation lowering in one adapter (`adapters/outbound/graphiti/`).
- Migrate the apply step to call only that adapter.
- Migrate every place that currently writes to Neo4j directly.
- Keep validation (`reconciliation_validation`) in front of the lowering — Graphiti does not validate ontology.

### Cleanup
- Delete the Neo4j Cypher writer code paths.
- Delete `StructuralGraphPort` if the decision is to retire it; otherwise rename to `StructuralReadPort` to make the role explicit.
- Delete any tests that asserted dual-writer behavior.

### Docs to update
- `architecture.md`: write path section reflects single writer with Graphiti API mapping.
- `architecture.md`: add a short "what Graphiti does for us, what we do above it" subsection (already drafted in `graph.md` today; bring across cleanly).

### Exit criteria
- `grep -r "session.run\|tx.run" app/src/context-engine/adapters/outbound/` returns only Graphiti-adapter calls or read-only structural reads.
- The reconciliation apply step has exactly one downstream call site.
- No mutation in the system bypasses ontology validation.

---

## Phase 2 — SourceConnector registry

**Goal:** A single `SourceConnector` port replaces the scattered `SourceControlPort` / `IssueTrackerPort` / `LinearIssueFetcher` / per-source webhook normalizers / per-source planners. Adding a new source means writing one connector module and registering it. The application layer never imports a concrete source.

### Discovery
- Map every place a source name appears in non-adapter code. Files like `application/use_cases/ingest_merged_pr.py` importing `adapters/outbound/reconciliation/github_pr_plan.py` are the violations to fix.
- List the ports involved today: `SourceControlPort`, `IssueTrackerPort`, `LinearIssueFetcher`, `SourceResolverPort`, `PotSourceListingPort`, the webhook normalizers under `adapters/inbound/http/webhooks/`, and the planners under `adapters/outbound/reconciliation/`. Decide which collapse into the new contract and which remain.
- Pick a *third* source (recommend Slack channel ingestion or Notion docs) as a smoke test: build it inside the new contract from scratch. If the contract is right, the third source is small and self-contained.

### Decisions to make this phase
- The exact `SourceConnector` interface. Recommended starting shape (refine in discovery):
  ```
  SourceConnector:
    capabilities() -> SourceCapabilityManifest
    list_artifacts(scope) -> Iterable[SourceRef]
    normalize_webhook(payload, headers) -> ContextEvent | None
    fetch(ref, policy) -> SourceResolution
    propose_plan(event, evidence_view) -> ReconciliationPlan | None
  ```
- How connectors are registered (entry point, plugin loader, or container wiring). Pick the simplest that supports out-of-tree connectors later.
- Whether `propose_plan` is mandatory (deterministic connectors) or optional (some sources only contribute resolution, and the agent always plans). Recommendation: optional; the registry advertises capabilities.
- How per-pot source credentials and scope live in the connector boundary, not leaked into application code.

### Implementation
- Define the port and the registry.
- Migrate GitHub into a connector module. Then Linear. Then the third (smoke test) source.
- Webhook routing: inbound HTTP receives the raw payload, looks up the connector by source kind, calls `normalize_webhook`, submits the event through the existing ingestion pipeline.
- The reconciliation agent receives connector-proposed plans as candidate input, not as the final plan. Agent still adjudicates, validates, and apply happens as in Phase 1.

### Cleanup
- Delete `SourceControlPort`, `IssueTrackerPort`, `LinearIssueFetcher` if fully subsumed.
- Delete `adapters/outbound/reconciliation/github_pr_plan.py`, `linear_issue_plan.py` once their logic moves into connector modules. The `reconciliation/` directory either becomes generic (validation, ontology mapping helpers) or is folded into connectors.
- Delete `application/use_cases/normalize_linear_webhook.py`, `application/use_cases/ingest_merged_pr.py`, `application/use_cases/ingest_single_pr.py`. Webhook normalization is connector-side; sync merged-PR shortcut is gone (see Phase 4).
- Delete every direct import of a source name from `application/`.

### Docs to update
- `architecture.md`: integration model section now describes the connector registry, not per-port plumbing.
- `extending.md`: write the "how to add a SourceConnector" guide using the third source as the worked example.
- `agent-contract.md`: describe what the connector capability manifest exposes through `context_status` so agents can ask "what sources do I have access to."

### Exit criteria
- `grep -r "github\|linear\|slack" app/src/context-engine/application/` returns nothing.
- A new connector can be added without touching `application/` or `domain/`.
- `context_status` returns a registered-connector manifest.
- All three sources (GitHub, Linear, third) ingest, resolve, and contribute plans through the same code path.

---

## Phase 3 — ContextReader registry

**Goal:** Apply the same extensibility pattern to the read side. Each include family (`decisions`, `change_history`, `project_map`, `debugging_memory`, `ownership`, …) becomes a registered `ContextReader` with its own capability descriptor. `GraphQueryPlanner` becomes a thin router.

### Discovery
- Inventory every leg in today's `GraphQueryPlanner`. For each leg, identify: input scope, output shape, backend it queries (Graphiti vector / Neo4j Cypher / both), and merge policy.
- Decide whether reads also collapse onto Graphiti's API in this phase or stay split (depends on Phase 1 outcome).
- Pick one *new* reader to add as smoke test (recommend `release_notes` or `security_findings`). If the contract is right, adding it is small and self-contained.

### Decisions to make this phase
- The `ContextReader` interface. Starting shape (refine in discovery):
  ```
  ContextReader:
    family() -> str                  # e.g. "decisions"
    intents() -> set[Intent]         # which intents trigger this reader by default
    capability_descriptor() -> ReaderCapability
    read(query: ContextGraphQuery) -> ReaderResult
  ```
- How the planner composes readers given a `(intent, include, scope, mode, budget)` request. Recommendation: declarative — intents and includes map to readers, the planner only orchestrates concurrency, budget, and merge.
- How readers declare their cost so the planner can respect `mode` and `budget` deterministically.

### Implementation
- Define the port and the registry.
- Decompose `GraphQueryPlanner` into a router + N readers. Migrate one family at a time; ship after each migration so behavior is observable.
- Add the smoke-test reader.
- Keep the answer-synthesizer and source-resolver layers unchanged (they consume the bundle, not the readers directly).

### Cleanup
- Delete the per-leg methods on `GraphQueryPlanner` once each reader replaces one.
- Delete dead reader code paths that no longer match an intent or include.
- Verify the response envelope still matches `agent-contract.md` exactly.

### Docs to update
- `architecture.md`: read path section becomes "router + readers."
- `extending.md`: write the "how to add a ContextReader" guide using the smoke-test reader.
- `agent-contract.md`: refresh the include-family catalog from the registry, not by hand.

### Exit criteria
- `GraphQueryPlanner` (or its replacement) is under ~150 lines and contains no domain logic specific to any one family.
- Adding a new include family touches one new reader file and one registration call.
- The agent-contract catalog is generated from the registry in CI and the doc never drifts.

---

## Phase 4 — Application layer collapse and ingestion path unification

**Goal:** Reduce `application/use_cases/` to a small, named set of verbs. Delete the sync merged-PR shortcut. One ingestion path: ContextEvent → ledger → debounced batch → reconciliation agent → validated plan → apply.

### Discovery
- Inventory every file under `application/use_cases/` and `application/services/`. For each, decide: keep (real verb), merge (overlapping concept), or delete (replaced by the connector/reader registries from Phase 2/3).
- Identify the entry points that still call the sync merged-PR path. Decide what replaces them (recommend: connector-driven backfill that uses the same reconciliation pipeline as live webhooks).
- Identify which use cases exist only because the system used to have multiple ingestion paths.

### Decisions to make this phase
- The final list of use-case verbs. Target: ~6–8. Recommended starting set:
  - `submit_event` (the canonical inbound)
  - `dispatch_due_batches`
  - `process_batch`
  - `resolve_context`
  - `record_durable_context`
  - `report_status`
  - `backfill_pot`
  - `hard_reset_pot`
- What lives in `services/` vs. `use_cases/`. Recommend: services compose ports for one cohesive operation; use cases are top-level verbs called by inbound adapters.

### Implementation
- Collapse use cases to the agreed list.
- Route every previously-sync ingestion through the agent path. Backfill becomes "enumerate via connector → submit events → wait for batch completion."
- Make sure the work-event capture and ledger semantics survive the collapse intact.

### Cleanup
- Delete every use case not on the final list.
- Delete the sync merged-PR shortcut and its tests.
- Delete `IngestionSubmissionService` SQLAlchemy leak — port-only construction.
- Delete the deprecated HTTP routes (`/events/ingest` alias, etc.). No alias paths remain.

### Docs to update
- `architecture.md`: ingestion section reflects the single path. Remove every mention of "sync merged-PR" or "compatibility branch."
- `agent-contract.md`: HTTP route list reflects only the routes that remain.

### Exit criteria
- `application/use_cases/` contains only the agreed verbs.
- `grep -r "merged_pr\|sync_pr\|ingest_pr" app/src/context-engine/` returns only test fixtures or comments deliberately kept.
- A single sequence diagram in `architecture.md` describes the only ingestion path.

---

## Phase 5 — PolicyPort, cost telemetry, drift instrumentation

**Goal:** Make the system operable. Centralize policy. Make cost and drift visible. Earn the right to scale.

### Discovery
- Map every place auth, scope, or access policy is enforced today (`require_auth`, ad-hoc checks in routes, source-credential lookups in connectors). Identify the places that will accrete implicit logic if not centralized now.
- Identify where LLM calls happen on the read path (synthesis) and the write path (Graphiti extraction, reconciliation agent). Decide what to instrument: per-call cost, per-resolve cost, per-episode cost, per-batch cost, per-pot cost.
- Decide drift signals: source freshness, edge invalidation rate, conflict rate, reader empty-result rate, ontology validation failure rate.

### Decisions to make this phase
- The `PolicyPort` interface. Starting shape: `authorize(actor, resource, action, context) -> Decision`. Decisions carry reasons so the response envelope can surface "you can ask but I won't fetch source X."
- Telemetry emission target (OTel? Postgres telemetry table? Both?). Pick one primary.
- Drift thresholds that flip `quality.status` in the response envelope.

### Implementation
- Introduce `PolicyPort` as a domain port; keep one concrete adapter to start.
- Wire policy at: HTTP auth, source-resolver fetch, connector list/fetch, write-apply.
- Emit cost and drift metrics from the spots identified in discovery.
- Surface a compact cost summary in `meta` of the resolve envelope (opt-in), and a drift summary in `quality`.

### Cleanup
- Delete inline auth checks that are now PolicyPort-mediated.
- Delete logging-only "metrics" that aren't structured.

### Docs to update
- `architecture.md`: add an "operability" section.
- `agent-contract.md`: document the `meta.cost` and `quality.drift` fields.

### Exit criteria
- A single policy decision call site per resource type.
- Per-pot dashboards (or query-able tables) for cost and drift exist.
- Response envelope carries enough information for an agent to decide "this is too expensive / too stale" without a second call.

---

## Phase 6 — Final cleanup pass and contributor docs

**Goal:** The codebase, the docs, and the directory layout match the target end-state. A new contributor can ship their first connector or reader in a day.

### Discovery
- Re-walk every directory under `app/src/context-engine/`. Flag anything whose purpose is unclear, anything that no other module imports, anything that contradicts the canonical docs.
- Re-walk the docs. Flag any drift introduced by the prior phases.
- Audit ports: every port has at least one production adapter and at least one test adapter; every adapter implements exactly one port.

### Decisions to make this phase
- Names. Rename anything whose name is wrong. This is the last cheap window.
- Whether to introduce a `domain/contracts/` separation between request/response models and pure-domain types (recommendation: only if the discovery shows the current `domain/` is hard to navigate).

### Implementation
- Rename, reorganize, prune.
- Fill in `extending.md` with end-to-end walkthroughs.
- Make the canonical ontology browsable from the docs (link or generate).

### Cleanup
- Delete unused ports, unused models, unused use cases, unused tests.
- Delete commented-out code. Delete TODO markers older than this rebuild.
- Delete every `__init__.py` re-export that isn't load-bearing — agents and humans should find one path to a symbol, not three.

### Docs to update
- All five canonical docs reflect the final state.
- `README.md` lists the canonical docs and nothing else.
- A short top-level note states the version of the architecture this represents and the date.

### Exit criteria
- `architecture.md` matches the code, top to bottom.
- A new contributor reads the docs, builds a connector, and the only application-layer code they touch is one registration call.
- No file in `app/src/context-engine/` exists that an architecture reviewer cannot place in the system at a glance.

---

## Things deliberately not in this plan

- A frontend / UI rebuild. The UI is a consumer of the agent contract; it is not the contract.
- Performance work beyond the cost telemetry in Phase 5. Optimize once the shape is right and telemetry shows where to optimize.
- Replacing Graphiti. The substrate is the right call; revisit only if Phase 5 telemetry forces a `fact_triple`-only or self-hosted decision.
- Cross-pot federation. One pot, one tenant, one graph. Federation is a separate problem.

## Working agreements for every phase

- Discovery findings are written into the phase's PR description, not into the docs.
- Decisions made during discovery are written into `architecture.md` immediately.
- If a phase takes longer than a week of focused work, split it. If a phase is "almost done" for two weeks, it isn't almost done — close the loop or replan.
- Tests are written for the new contract, not the old one. Old tests are deleted alongside the code they protected.
