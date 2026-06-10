# Harness-Led Repo Ingestion Plan

## Purpose

Repo ingestion must stay harness-led. The Potpie CLI is the durable memory
surface: it registers sources, reads graph context, resolves graph identities,
validates semantic mutations, and applies them. The harness is the intelligence:
it decides what to inspect, reads selected source material, reasons about durable
facts, and writes graph mutations with evidence.

This plan stages the implementation so we can improve repo ingestion without
reintroducing deterministic code scanning or CLI-owned graph inference.

## Boundary

### Harness Owns

- Breaking down a repo-ingestion task into investigation steps.
- Selecting what docs, config, manifests, routes, entrypoints, PRs, and issues
  to inspect.
- Deciding whether a fact is durable enough to store.
- Choosing graph entities, predicates, truth class, confidence, evidence, and
  retrieval descriptions.
- Calling `potpie graph mutate --dry-run` before applying mutations.

### Potpie CLI Owns

- `potpie source add repo ...`: source registration only.
- `potpie source list`: source visibility.
- `potpie graph catalog`: graph contract discovery.
- `potpie graph search-entities`: identity resolution.
- `potpie graph read`: context retrieval.
- `potpie timeline recent`: project-wide timeline retrieval.
- `potpie graph mutate`: semantic mutation validation and application.
- `potpie graph repair`: deterministic graph hygiene, not repo interpretation.

### Non-Goals

- No CLI repo scanner.
- No deterministic file/config walker that mutates the graph.
- No CLI command that infers services, features, topology, preferences, or
  dependencies from a working tree.
- No graph updates from file names alone.
- No service-side model call that replaces harness reasoning for local agent
  workflows.

## Stage 0: Lock The Boundary

Goal: make the product contract unambiguous.

Implementation:

- Audit docs, CLI help text, skills, and playbooks for wording that implies
  "Potpie ingests a repo" or "CLI scans a repo".
- Replace with "the harness ingests; Potpie registers, reads, validates, and
  stores."
- Keep source commands explicit: source registration is not ingestion.
- Keep event/connector ingestion language scoped to queued source events, not
  local repo understanding.

Likely files:

- `docs/context-graph/README.md`
- `docs/context-graph/architecture.md`
- `docs/context-graph/cli-flow.md`
- `docs/context-graph/graphv1-5-implementation-plan.md`
- `potpie/context-engine/adapters/inbound/cli/commands/pots.py`
- agent skills under the skill bundle/catalog.

Acceptance:

- No user-facing doc says the CLI scans or understands a repo.
- `potpie source add repo ...` is documented as registration only.
- Any `ingest` wording clearly refers to harness-led work or queued connector
  events.

## Stage 1: Fix Node Summaries

Goal: every graph node is meaningful for graph browsing and future semantic
retrieval.

Implementation:

- Standardize node metadata:
  - `summary`: compact display/search summary.
  - `description`: richer retrieval card with synonyms, scope, and evidence
    context.
- Extend semantic entity refs or mutation lowering so every entity write can
  carry both fields.
- In the writer, derive missing `summary` from `description`, `name`, or
  `entity_key`.
- Preserve `description` as the higher-recall retrieval card.
- Expose entity properties from FalkorDB read/search paths.
- Update graph UI captions/details to show summaries.
- Add repair support for existing nodes with missing/empty summaries.

Likely files:

- `potpie/context-engine/domain/semantic_mutations.py`
- `potpie/context-engine/application/services/semantic_mutation_lowering.py`
- `potpie/context-engine/adapters/outbound/graph/cypher.py`
- `potpie/context-engine/adapters/outbound/graph/falkordb_reader.py`
- `potpie/context-engine/adapters/inbound/http/ui/router.py`
- `potpie/context-engine/tests/unit/test_context_graph_writer.py`
- `potpie/context-engine/tests/unit/test_falkordb_reader.py`

Acceptance:

- New entity writes do not produce empty summaries.
- `graph search-entities` returns entity `summary` and `description`.
- `graph read` inline entity payloads include useful summaries.
- UI node captions/details can show summaries.
- Existing nodes can be repaired without inventing repo facts.

## Stage 2: Define Harness Repo Baseline Workflow

Goal: make repo baseline ingestion a repeatable harness procedure.

Implementation:

- Create a harness skill/playbook for `repo_baseline_understanding`.
- The playbook should tell the harness to:
  1. Resolve the current pot.
  2. Register the repo source if missing.
  3. Read the graph catalog.
  4. Search for existing repo/service entities before writing.
  5. Read authored docs first.
  6. Inspect selected source files only when they are the source of truth for a
     durable fact.
  7. Write semantic mutations with summary, description, truth, confidence, and
     evidence.

Recommended source order:

- README and docs.
- ADRs, runbooks, architecture docs, deployment docs.
- Package/app manifests.
- CI/deploy workflows.
- Environment templates.
- Framework config.
- Route/API specs or visible route entrypoints.
- Service clients/adapters only when needed to confirm topology.
- Datastore usage only when needed to confirm durable infra facts.

Baseline memory families:

- Repository purpose.
- Application/service type.
- Primary functionality and feature areas.
- Services and deployable units.
- Environment/deploy shape.
- Service dependencies.
- API contracts/routes.
- Datastores and important integrations.
- Project preferences from explicit docs/config.

Acceptance:

- The harness can ingest a repo baseline from current repo, path, URL, or doc
  reference.
- The procedure is selective and evidence-led.
- The graph is updated through `graph mutate`, never through scanning.
- Mutation batches include useful summaries and retrieval descriptions.

## Stage 3: Separate Change-History Ingestion

Goal: keep timeline ingestion distinct from baseline repo understanding.

Implementation:

- Treat PR/issue ingestion as `repo_change_history_ingestion`.
- Keep or rename the existing one-shot repo history playbook so its scope is
  explicit.
- Do not use change-history ingestion to infer the whole architecture.
- Record:
  - merged PR activities,
  - issue activities,
  - clear fixes from merged PRs,
  - bug patterns when symptoms are explicit,
  - decisions only when rationale is explicit,
  - affected repo/service scopes when evidence supports them.

Likely files:

- `potpie/context-engine/domain/playbooks/repo_one_shot_ingestion.md`
- `potpie/context-engine/domain/event_playbooks.py`
- related tests for repo one-shot ingestion skills.

Acceptance:

- Recent-change queries are project-wide by default across all repos in the pot.
- Repo baseline and timeline facts are independently queryable.
- The change-history playbook no longer implies baseline repo understanding.

## Stage 4: Add Feature Or Capability Ontology

Goal: represent what a repo/service does without generic observations.

Implementation:

- Add a first-class entity type:
  - `Feature` or `Capability`.
- Add predicates such as:
  - `PROVIDES`: `Repository|Service -> Feature`
  - `IMPLEMENTED_IN`: `Feature -> Repository|CodeAsset`
  - `AFFECTS`: existing broad downstream impact predicate can continue to work.
- Keep low-level file/module mapping out of scope unless the harness has strong
  evidence and the ontology supports it.
- Update catalog, validator, readers, ranking, tests, and skill instructions.

Likely files:

- `potpie/context-engine/domain/ontology.py`
- `potpie/context-engine/application/services/semantic_mutation_validator.py`
- `potpie/context-engine/domain/graph_contract.py`
- graph reader tests and ontology tests.

Acceptance:

- The harness can store user-facing or system-facing functionality as durable
  graph structure.
- Queries like "what does this repo do?" return feature/capability nodes and
  supporting claims.
- Feature entities have compact summaries and retrieval descriptions.

## Stage 5: Improve CLI Ergonomics For Harnesses

Goal: reduce brittle shell usage without adding CLI intelligence.

Implementation:

- Allow easier source registration:
  - `potpie source add repo .`
  - `potpie source add repo current`
- Keep both as source registration only.
- Improve pot auto-resolution errors:
  - no active pot,
  - current repo matches multiple pots,
  - current repo has no registered source,
  - registered source path/remote mismatch.
- Improve JSON output for:
  - active/current pot resolution,
  - registered repo sources,
  - mutation dry-run warnings,
  - catalog examples.
- Consider template helpers only if they are schema helpers, not extractors:
  - `potpie graph mutation-template --kind repo-baseline`

Acceptance:

- Harnesses need fewer fragile commands.
- Errors give the next useful action.
- No CLI helper reads repo files or infers graph facts.

## Stage 6: Update And Reinstall Skills

Goal: make Codex, Claude Code, and other harnesses follow the same model.

Implementation:

- Update skills:
  - `potpie-source-ingestion`
  - `potpie-cli`
  - `potpie-graph`
  - `potpie-project-preferences`
  - `potpie-infra-architecture`
  - `potpie-change-timeline`
  - `potpie-debug-memory`
- Require every durable write to include:
  - compact entity summaries,
  - retrieval-grade descriptions,
  - evidence refs,
  - honest truth class.
- Reinstall skills to Claude Code and Codex.
- Add tests that check critical skill language is present.

Acceptance:

- Installed skills say repo ingestion is harness-led.
- Skills do not instruct agents to run code scanners.
- Skills explicitly require summaries, descriptions, and evidence.

## Stage 7: End-To-End Validation On `potpie-ui`

Goal: prove the workflow with a real multi-repo project memory task.

Test flow:

1. Start in `/Users/nandan/Desktop/Dev/potpie-ui`.
2. Resolve the current pot automatically.
3. Register the repo source if missing.
4. Run the harness-led baseline procedure.
5. Run change-history ingestion separately.
6. Query the graph:
   - "What does potpie-ui do?"
   - "What services does it depend on?"
   - "What features does it provide?"
   - "What changed recently in the project?"
   - "What project preferences apply when adding a frontend feature?"
7. Inspect graph nodes for useful summaries.

Acceptance:

- Answers come from graph memory, not from re-reading the repo.
- Timeline is project-wide unless explicitly narrowed.
- Repo/service/feature nodes have compact summaries.
- Retrieval descriptions contain useful search language.
- No deterministic scan path mutated the graph.

## Implementation Order

Recommended order:

1. Stage 0: boundary wording.
2. Stage 1: node summaries and property hydration.
3. Stage 2: baseline harness workflow.
4. Stage 3: change-history naming/scope cleanup.
5. Stage 5: CLI ergonomics.
6. Stage 6: skills reinstall.
7. Stage 4: feature/capability ontology.
8. Stage 7: end-to-end validation.

Stage 4 can move earlier if repo functionality becomes difficult to express
without first-class `Feature` or `Capability` nodes.

