# Context Graph Docs

Last reviewed: 2026-05-28.

The docs are local-first. Read the open-source self-serve path first, then the
managed cloud path as a hosted deployment of the same core graph engine.

Terminology: **Context Graph** is the portable graph model and agent-facing
surface. **Context Engine** is the current code/service package that implements
it.

## Reading Order

| Doc | Purpose |
|---|---|
| [`vision.md`](./vision.md) | Product principles: one pot, one graph model, four agent tools. |
| [`architecture.md`](./architecture.md) | Local OSS architecture first, then managed cloud, including daemon, storage, graph model, export, and event ledger boundaries. |
| [`agent-contract.md`](./agent-contract.md) | Stable four-tool contract used by CLI, MCP, local daemon, and cloud API. |
| [`extending.md`](./extending.md) | How to add readers, scanners, record types, storage adapters, and cloud connectors. |
| [`observability.md`](./observability.md) | Tracing, metrics, logging, readiness, and local/cloud telemetry boundaries. |
| [`bench-plan.md`](./bench-plan.md) | Benchmark taxonomy, scoring, scenario tracking, and adapter validation. |

Code lives at [`app/src/context-engine/`](../../app/src/context-engine/).

Old rebuild/design plans are paused and intentionally not kept in this folder.
Use git history if historical detail is needed.

The current package still contains managed-first CLI/API paths. Treat those as
implementation starting points, not the final OSS UX. The target default is:

```bash
pip install potpie
potpie init
potpie daemon install
potpie status
```

Cloud commands should be explicit:

```bash
potpie cloud login
potpie cloud push
potpie cloud status
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
