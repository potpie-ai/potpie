# Extending The Context Graph

Last reviewed: 2026-05-28.

Extend the graph by adding to the core model and ports. Keep the local daemon,
CLI, MCP server, and managed cloud adapter thin.

## Principles

- Add graph vocabulary in `domain/ontology.py`.
- Add read behavior through a reader and the read orchestrator.
- Add write behavior through structured records, scanners, or graph mutations.
- Add storage behavior behind `GraphWriterPort` and `ClaimQueryPort`.
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

## Add A Local Storage Adapter

Use this for the local OSS store.

1. Implement `GraphWriterPort`.
2. Implement `ClaimQueryPort`.
3. Preserve pot isolation on every query and write.
4. Preserve entity keys, predicates, source refs, valid time, observed time,
   invalidation, and mutation ids.
5. Run the benchmark seed/read scenarios against the adapter.

Readers and agent tools must not know which store is underneath.

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

Prefer commands that call the daemon or application use cases.

Local commands should default to the local profile:

- `potpie init`
- `potpie status`
- `potpie daemon ...`
- `potpie resolve`
- `potpie search`
- `potpie record`
- `potpie ingest scan`

Cloud commands should be visibly cloud-scoped:

- `potpie cloud login`
- `potpie cloud push`
- `potpie cloud pull`
- `potpie cloud status`

Do not make a local command silently call the managed API.

## What Not To Extend

- Do not create a fifth public agent tool for a use case.
- Do not bypass `ReadOrchestrator` for agent-visible reads.
- Do not query a physical store directly from CLI/MCP code.
- Do not put source-provider credentials in the local daemon by default.
- Do not duplicate the ontology in docs, CLI enums, or cloud-only code.
- Do not preserve stale compatibility paths after a replacement is complete.
