---
name: graph-mutation-plan
description: Compose canonical semantic operations for apply_graph_mutations using stable entity keys and source evidence.
version: "2.0.0"
tags: [mutation, graph, ontology, reference]
---

# Composing an `apply_graph_mutations` request

Call `apply_graph_mutations(plan, event_id, summary)` for an event in the batch.
The plan contains an `operations` array of semantic operations. The tool binds
it to the batch pot and supplies the event idempotency key and ingestion actor.
Reuse stable entity keys resolved by reads. Never reference another pot.

Each operation has an `op` and the fields for that operation. Use
`upsert_entity` to establish an entity, `link_entities` or `assert_claim` for
supported relationships, and `retract_claim` or `end_relation_validity` when
source evidence changes an existing claim. Read the canonical mutation tool
catalog for supported operations and required fields. Do not submit writer
batch fields such as `entity_upserts`, `edge_upserts`, or `episodes`.

Use the exact types, key prefixes, predicates and allowed endpoint pairs from
the ontology in your instructions. Preserve source evidence on each claim,
including its reference, truth classification and any environment or validity
bounds. When the source does not support a relationship, omit it.

```json
{
  "operations": [
    {
      "op": "upsert_entity",
      "subject": {"key": "service:billing", "type": "Service"}
    },
    {
      "op": "link_entities",
      "subgraph": "infra_topology",
      "subject": {"key": "service:billing", "type": "Service"},
      "predicate": "DEPENDS_ON",
      "object": {"key": "service:ledger", "type": "Service"},
      "truth": "source_observation",
      "description": "Billing invokes the ledger service",
      "evidence": [{"kind": "pull_request", "ref": "github:pr:o/r:482"}]
    }
  ]
}
```

Read before changing existing claims so retractions identify their exact claim
keys or record IDs. Keep failed remedies distinct from successful fixes and
record decisions only when their rationale is stated in the source.
