# Context Graph Docs

Last reviewed: 2026-05-28.

The docs are local-first. Read the open-source self-serve path first, then the
managed cloud path as a hosted deployment of the same service modules and graph
model.

Terminology:

- **Context Graph** is the portable graph model and agent-facing surface.
- **Context Engine** is the current code/service package that implements it.
- **Pot** is the current workspace/tenant boundary. The name may change later,
  but the boundary should stay explicit.
- **Daemon shell** is the local OS background process. It hosts services; it is
  not itself the graph business layer.
- **Pot Management Service** is the control plane for pot CRUD, graph status,
  source registry, analytics, export/import, and lifecycle operations. This is
  the current name for the service that may also be described as graph
  management.
- **Graph Service** is the data plane for graph writes, graph queries, semantic
  search routing, readers, ranking, and agent envelopes.
- **Graph Backend** is the swappable graph layer the Graph Service depends on: a
  bundle of capability ports (mutation, claim query, inspection, analytics,
  semantic search, snapshot) bound to physical stores by a storage profile. The
  default profile is a single embedded store.
- **Skill Manager Service** owns the lifecycle of the agent **skills** that
  third-party harnesses (Claude Code, Codex, …) need to drive the four tools:
  catalog, install/update/remove into a harness, OSS-catalog download, cloud
  sync, and the advisory missing-skill nudge surfaced via `context_status`.
- **Skill** is a portable four-tool recipe; an **agent target** is the per-harness
  adapter that renders and installs a skill into that harness's format/location.

## Reading Order

| Doc | Purpose |
|---|---|
| [`vision.md`](./vision.md) | Product principles: one pot, one graph model, four agent tools. |
| [`oss-self-serve-flow.md`](./oss-self-serve-flow.md) | End-to-end OSS user/agent flow and target CLI/daemon contracts from install through ingestion, query, records, graph/backend operations, skills, and cloud sync. |
| [`architecture.md`](./architecture.md) | Service boundaries and interfaces, the graph backend capability model (how to abstract/swap the graph layer), the Skill Manager, local OSS first then managed cloud, graph model, export, and event ledger boundaries. |
| [`agent-contract.md`](./agent-contract.md) | Stable four-tool contract used by CLI, local daemon, and cloud API. |
| [`extending.md`](./extending.md) | How to add readers, scanners, record types, graph backends, skills, agent targets, Pot Management behavior, and cloud connectors. |
| [`observability.md`](./observability.md) | Tracing, metrics, logging, readiness, and local/cloud telemetry boundaries. |
| [`bench-plan.md`](./bench-plan.md) | Benchmark taxonomy, scoring, scenario tracking, and adapter validation. |

Code lives at [`app/src/context-engine/`](../../app/src/context-engine/).

Old rebuild/design plans are paused and intentionally not kept in this folder.
Use git history if historical detail is needed.

The current package still contains managed-first CLI/API paths. Treat those as
implementation starting points, not the final OSS UX. The target default is:

```bash
pip install potpie
potpie setup --repo . --agent claude --scan
potpie status
```

`potpie setup` creates and uses a local `default` pot on first run. Pass
`--pot <name>` only when the initial pot should use a different name.

Cloud commands should be explicit:

```bash
potpie cloud login
potpie cloud push
potpie cloud pull
potpie cloud status
potpie cloud skills sync
```

## Ontology Import Surface

Most graph vocabulary is derived from `domain.ontology`:

```python
from domain.ontology import (
    ENTITY_TYPES, EDGE_TYPES, RECORD_TYPES,
    CANONICAL_LABELS, CANONICAL_EDGE_TYPES, SYSTEM_EDGE_TYPES, ALL_EDGE_TYPES,
    SCOPE_LABELS, ACTIVITY_LABELS,
    ENTITY_PROJECT_MAP_FAMILY, ENTITY_DEBUGGING_FAMILY, ENTITY_FACT_FAMILY,
    FACT_FAMILY_FRESHNESS_TTL_HOURS, SOURCE_OF_TRUTH_POLICIES,
    ENTITY_TEXT_CLASSIFIERS, ENTITY_PROPERTY_SIGNATURES,
    EDGE_ENDPOINT_INFERRED_LABELS, PREDICATE_FAMILY_EDGE_NAMES,
    SINGLETON_EDGE_TYPES, STRUCTURAL_INCLUDES, PUBLIC_RECORD_TYPES,
    validate_entity_upsert, validate_edge_upsert, validate_structural_mutations,
    entity_spec, edge_spec, record_type_spec, record_types_for_include,
    advertised_include_families,
    is_canonical_entity_label, is_canonical_edge_type,
    project_map_family_for_label, debugging_family_for_label, fact_family_for_label,
    is_scope_label, is_activity_label, allowed_edge_types_between,
    canonical_entity_labels,
    normalize_edge_name, inferred_labels_for_episodic_edge_endpoint,
    predicate_family_for_edge_name, predicate_family_for_episodic_supersede,
)
```

If a doc conflicts with `vision.md`, `architecture.md`, or `agent-contract.md`,
update the doc. Do not add a second graph model or a second agent contract.
