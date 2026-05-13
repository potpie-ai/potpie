# Context Engine docs

**Architecture version:** Phase 9 (extensibility-first ontology). Last reviewed 2026-05-12.

Five docs, one job each. Read in order if you're new.

| Doc | What it answers |
|---|---|
| [`vision.md`](./vision.md) | What the Context Engine is, why it exists, the principles that bind every other decision. |
| [`architecture.md`](./architecture.md) | How the system is built today, with file paths into the code. |
| [`agent-contract.md`](./agent-contract.md) | The four-tool agent port — request and response contracts. The single source of truth for any agent or skill that consumes the engine. |
| [`extending.md`](./extending.md) | How to add a source, a reader, a record type, an intent, **or an ontology entity / edge** — without touching the application layer. |
| [`plan.md`](./plan.md) | The rebuild plan. Phased, sequential, with discovery first in every phase. |

Code lives at [`app/src/context-engine/`](../../app/src/context-engine/).

**The ontology is declarative and spec-driven.** [`app/src/context-engine/domain/ontology.py`](../../app/src/context-engine/domain/ontology.py) declares every canonical entity and edge with the metadata its consumers need (project-map family, fact family, source-of-truth, freshness TTL, text classifier cues, property signatures, edge endpoint inference). All downstream modules (classifier, structural reader, hybrid graph, graph-quality policy, query helpers) **derive** their tables from the spec — no module hardcodes label strings. Adding or renaming an entity is a single-file edit. See [`extending.md`](./extending.md#adding-an-ontology-entity) for the recipe.

Public API (the surface every other doc and adapter consumes):

```python
from domain.ontology import (
    # Catalogs
    ENTITY_TYPES, EDGE_TYPES,
    CANONICAL_LABELS, CANONICAL_EDGE_TYPES,
    SCOPE_LABELS, ACTIVITY_LABELS,
    # Derived tables (built from spec at import time)
    INCLUDE_KEY_LABELS, ENTITY_PROJECT_MAP_FAMILY, ENTITY_DEBUGGING_FAMILY,
    FACT_FAMILY_FRESHNESS_TTL_HOURS, SOURCE_OF_TRUTH_POLICIES,
    ENTITY_TEXT_CLASSIFIERS, ENTITY_PROPERTY_SIGNATURES,
    EDGE_ENDPOINT_INFERRED_LABELS, PREDICATE_FAMILY_EDGE_NAMES,
    # Validation
    validate_entity_upsert, validate_edge_upsert, validate_structural_mutations,
    # Lookups
    entity_spec, edge_spec,
    is_canonical_entity_label, is_canonical_edge_type,
    labels_for_include_key,
    project_map_family_for_label, debugging_family_for_label, fact_family_for_label,
    is_scope_label, is_activity_label,
    allowed_edge_types_between, canonical_entity_labels,
    # Episodic
    normalize_graphiti_edge_name, inferred_labels_for_episodic_edge_endpoint,
    predicate_family_for_edge_name, predicate_family_for_episodic_supersede,
)
```

These five files plus `plan.md` are the entire doc set. Anything else that appears here without a phase landing it is doc rot.
