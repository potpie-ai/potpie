# Extending The Context Graph

Last reviewed: 2026-05-28.

Extend the graph by adding to the core model, service modules, and ports. Keep
the daemon shell, CLI, and managed cloud adapter thin.

## Principles

- Keep the daemon shell thin: lifecycle, auth, config, logs, health, and
  transport only.
- Add pot/control-plane behavior to the Pot Management Service.
- Add graph data-plane behavior to the Graph Service.
- Add graph vocabulary in `domain/ontology.py`.
- Add read behavior through a reader and the read orchestrator.
- Add write behavior through structured records, scanners, or graph mutations.
- Add storage behavior as a `GraphBackend` (capability ports), not store-specific
  code reachable from readers or CLI.
- Add agent skills to the Skill Manager catalog; add harness support as an agent
  target adapter. Never as a fifth agent tool.
- Add cloud integrations in managed/event-ledger adapters, not in the local
  daemon core.
- Do not add new public agent tools unless the four-tool contract itself is
  intentionally being revised.

## Add A Reader

Use this when a new include family needs to be answered by the graph.

1. Add a reader under `app/src/context-engine/application/readers/`.
2. Read through `ClaimQueryPort`; do not query Neo4j, SQLite, or Postgres
   directly from the reader.
3. Register the include key in `ReadOrchestrator._routing`.
4. Add the include key to `READER_BACKED_INCLUDES` in
   `domain/agent_context_port.py`.
5. Add focused tests for routing, coverage, unsupported includes, and ranking.

Current reader-backed includes:

- `coding_preferences`
- `infra_topology`
- `timeline`
- `prior_bugs`
- `raw_graph`

## Add A Scanner

Use this when local repo files can deterministically produce context.

1. Implement `ConfigScannerPort` from `domain/ports/config_scanner.py`.
2. Put the adapter under `adapters/outbound/scanners/`.
3. Register it in `application/services/config_scanner_registry.py` or the
   relevant container wiring.
4. Route CLI use through `application/use_cases/scan_working_tree.py`.
5. Emit validated graph mutations or structured records.

Good scanner candidates: CODEOWNERS, dependency manifests, Kubernetes
manifests, OpenAPI specs, CI workflow files, service manifests, and runbook
indexes.

## Add A Record Type

Use this when agents need to write durable memory through `context_record`.

1. Add a row to `RECORD_TYPES` in `domain/ontology.py`.
2. Add a structured payload builder in `domain/context_records.py` if the
   record should be validated beyond free-form `summary/details`.
3. Add deterministic claim emission for local mode when possible.
4. Map the record to the include family that should read it back.
5. Add tests for validation, idempotency, claim emission, and reader retrieval.

Structured records are the preferred local write path. Raw-event reconciliation
is optional locally and normal in managed cloud.

## Add An Entity Or Predicate

Use this when the existing ontology cannot represent the fact.

1. Add the entity to `ENTITY_TYPES` or the predicate to `EDGE_TYPES` in
   `domain/ontology.py`.
2. Define identity, key prefix, allowed endpoint pairs, source-of-truth family,
   freshness TTL, and singleton behavior if relevant.
3. Update record emitters, scanners, or reconciliation validation that should
   produce the new fact.
4. Update readers only if the new fact changes agent-visible output.
5. Add coherence and validation tests.

Do not add an entity just because a payload has a field. Add an entity when it
needs identity, traversal, provenance, or lifecycle as a graph object.

## Add A Graph Backend

Use this to support a new physical store (the swappable graph layer). The
contract is the capability ladder in
[`architecture.md`](./architecture.md#graph-service-interfaces): three mandatory
ports, three derivable ones.

1. Implement the **three mandatory** ports from `domain/ports/graph/`:
   - `GraphMutationPort` — apply validated mutations/invalidations, reset, ready.
   - `ClaimQueryPort` — canonical claim reads + bulk entity-label lookup.
   - `SemanticSearchPort` — vector index/search over claim facts (see below).
2. Subclass `ClaimDerivedBackend` to inherit working defaults for the **three
   derivable** ports (inspection, analytics, snapshot) computed over
   `ClaimQueryPort`. Override one *only* when the store can beat the naive
   version (e.g. Neo4j inspection via Cypher).
3. Assemble a `GraphBackend`: set `name`, declare `capabilities`, and expose the
   ports. Register it as a storage profile in the container wiring.
4. Honor the invariants:
   - Preserve pot isolation on every query, write, index entry, and snapshot.
   - Preserve entity keys, predicates, source refs, valid/observed time,
     invalidation, and mutation ids.
   - Treat the canonical claim store as the only source of truth; any secondary
     store (index, vector, traversal copy) is a rebuildable projection.
   - A write is one transaction against the mutation port. If the profile spans
     stores, reconcile inside the adapter — write claims first, update
     projections best-effort, expose `analytics().repair()` for reindex. Never
     surface multi-store state to the application.
5. Pass the shared **conformance suite** (`tests/graph/conformance/`). A backend
   is not done until it does. The benchmark seed/read scenarios also run through
   it for envelope-equivalence.

Readers and agent tools must not know which store is underneath. For OSS V1,
prefer the embedded single-store profile before a Docker-required one (SQLite is
the first target); Postgres/pgvector, Chroma, and Neo4j are optional profiles
behind the same ports.

## Add A Vector Store Or Embedder

`SemanticSearchPort` is mandatory and vector-backed (every backend ships it).
Use this when you want to bind it to a different vector store or swap the
embedder — not to make it optional.

1. Implement `SemanticSearchPort` (`index`, `search`) against the vector store
   and keep `SEMANTIC` in the backend's `capabilities`.
2. Keep semantic search behind the Graph Service, not behind reader-specific
   database code. The read orchestrator routes all semantic retrieval through
   this port.
3. Treat the vector index as a projection of the canonical claims: rebuildable
   by re-embedding via `analytics().repair()`, never a second source of truth.
4. Return comparable similarity metadata (e.g.
   `properties["semantic_similarity"]`) so ranking stays store-neutral.
5. Keep a working **local embedder** as the default so vector search runs offline
   with no external/cloud key. An alternate or hosted embedder is configurable,
   but the bundled local default must remain.

Good store bindings: SQLite vector extensions (`sqlite-vec`), pgvector, Chroma,
Neo4j vector indexes, or a hosted vector service in managed cloud.

## Add A Skill

Use this to teach harnesses a new four-tool workflow. A skill is a portable
recipe, not engine code; it composes `context_resolve` / `search` / `record` /
`status` — it never adds a tool.

1. Author the skill as a catalog entry (start from
   `adapters/inbound/cli/templates/`): give it an `id`, `version`, `title`,
   `description`, the `intents` it applies to, the four-tool recipe, and harness
   compatibility.
2. Keep the skill harness-neutral. Harness-specific rendering belongs to the
   agent target adapter, not the skill body.
3. Map it to the intents that should recommend it, so `context_status` can nudge
   when it is missing for a harness.
4. Version it; updates flow through `potpie skills update`.
5. Do not put skill content in the graph — skills are agent config, not facts.

First-party skills ship in this repo; additional skills resolve from the public
OSS catalog through `SkillCatalogPort`.

## Add An Agent Target Adapter

Use this to support a new harness (e.g. a new editor/agent beyond Claude Code,
Codex, and the generic `AGENTS.md` target).

1. Implement `AgentTargetPort` (`installed`, `install`, `remove`) for the harness
   under the Skill Manager's adapters. Set `name` (`claude`, `codex`, …).
2. Encode the harness's skill format and install location (e.g. Claude Code uses
   `.claude/commands/`; the generic target uses `.agents/skills/` + `AGENTS.md`).
3. Make installs idempotent and marker-fenced so re-running is safe and an
   uninstall is clean.
4. Report installed skills and their versions so drift detection and the
   `context_status` nudge work for this harness.
5. Do not bake skill content into the adapter — render from the catalog skill.

Skills stay neutral; the target renders them. This is the same binding pattern as
a graph storage profile: add a target, not a new skill format per skill.

## Add Pot Management Behavior

Use this for pot CRUD, source registry, graph status, analytics, lifecycle, or
export/import behavior.

1. Put local behavior in the local Pot Management Service, backed by the local
   state DB.
2. Preserve the setup invariant: first local setup creates an active `default`
   pot, and commands without `--pot` use the active pot.
3. Keep managed user/team/role/source behavior under `app/modules/context_graph/`.
4. Call the Graph Service for graph status, inspection, analytics, reset, and
   snapshot operations.
5. Keep pot management out of the agent four-tool surface unless the agent only
   needs read-only readiness via `context_status`.
6. Add tests around pot isolation, migration behavior, and local/cloud adapter
   differences.

## Add Managed Or Webhook Integration

Use this for hosted sources, cloud sync, or webhook ingestion.

1. Keep source credentials and webhook receivers out of the local daemon.
2. Put managed Potpie API/DB/user/worker logic under `app/modules/context_graph/`
   or existing managed adapter boundaries.
3. Put connector adapters under `adapters/outbound/connectors/` when they are
   used by managed workers or event-ledger services.
4. Normalize webhook payloads into an event ledger before graph ingestion.
5. Let local users pull from the ledger explicitly and record into their graph.

The event ledger is operational input. The context graph remains the fact
store.

## Add A CLI Command

Prefer commands that call the daemon or application use cases. The target command
groups and output/error conventions live in
[`oss-self-serve-flow.md`](./oss-self-serve-flow.md); update that contract when a
new command changes the user or agent workflow.

Local commands should default to the local profile:

- `potpie setup`
- `potpie init` (advanced/scripted bootstrap)
- `potpie status`
- `potpie daemon ...`
- `potpie pot ...`
- `potpie source ...`
- `potpie resolve`
- `potpie search`
- `potpie record`
- `potpie ingest ...` (scan / status / runs / show / replay)
- `potpie graph ...` (status / inspect / export / import / repair)
- `potpie backend ...` (list / status / use / doctor)
- `potpie skills ...` (list / install / update / remove / status / add)

Cloud commands should be visibly cloud-scoped:

- `potpie cloud login`
- `potpie cloud push`
- `potpie cloud pull`
- `potpie cloud status`
- `potpie cloud skills sync`

Do not make a local command silently call the managed API.

## What Not To Extend

- Do not create a fifth public agent tool for a use case.
- Do not bypass `ReadOrchestrator` for agent-visible reads.
- Do not query a physical store directly from CLI/reader code; go through a
  capability port.
- Do not make a secondary store (vector, FTS, traversal copy) a second source of
  truth, or orchestrate a multi-store write from the application layer.
- Do not put Pot Management behavior in the daemon shell itself.
- Do not put source-provider credentials in the local daemon by default.
- Do not expose skill management as an agent tool; the only agent-visible surface
  is the advisory `skills` block in `context_status`.
- Do not install or sync skills without an explicit CLI action; never sync to
  cloud agents silently.
- Do not put skill content in the graph, or harness-specific rendering in a
  skill body (it belongs in the agent target adapter).
- Do not duplicate the ontology in docs, CLI enums, or cloud-only code.
- Do not preserve stale compatibility paths after a replacement is complete.
