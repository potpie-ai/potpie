# 03 — Canonical node labels not applied

## Problem

`potpie search "ledger" --node-labels Decision` returned `No results` in a pot that explicitly ingested an ADR-style decision ("Migration to Postgres decided … Deciders: Alice, Ravi"). The `DECIDES_FOR` edge exists, but the **node** on either side is not tagged with the canonical label `Decision`. The CLI flag is effectively dead for most ontology labels.

## Proposal

Synthesise canonical node labels during reconciliation, driven by two signals:

### Signal 1 — Edge-pattern inference

Rules stored in `domain/ontology.py` as `(edge_label, role) -> canonical_label`:

| Edge | Role | Infer label on role node |
|------|------|---------------------------|
| `DECIDES_FOR` | target | `Decision` |
| `FIXES` | subject | `Release` / `Fix` |
| `CAUSED` | target | `Incident` |
| `OWNS` | target | `Component` / `Artifact` |
| `DEPLOYED_TO` | target | `Deployment` |
| `DEPRECATES` / `DECOMMISSIONS` | target | `LegacyArtifact` |

Applied in a reconciliation pass after extraction: for each new edge, look up the rule table and add labels as a *set* on the node (not overwrite — a node can be both `Component` and `Deployment` target).

### Signal 2 — Structured extractor hints

Graphiti's extractor already picks up proper nouns and noun phrases; extend the prompt/schema to emit an optional `canonical_type` per entity drawn from the ontology vocabulary. When present, trust it (modulo ontology validation); otherwise fall back to Signal 1.

## Files touched

- `app/src/context-engine/domain/ontology.py` — rule table `EDGE_TO_CANONICAL_LABEL`.
- `app/src/context-engine/application/use_cases/reconciliation_validation.py` — apply the rule table after structural mutations pass validation.
- `app/src/context-engine/adapters/outbound/graphiti/episodic.py` — add `canonical_type` to the extractor schema.
- `app/src/context-engine/adapters/outbound/neo4j/structural.py` — `MERGE` label-set additions (Neo4j supports multi-label nodes natively).
- Tests: `app/src/context-engine/tests/unit/test_label_inference.py`.

## Backfill

Same maintenance-job story as #02: `relabel_nodes_from_edges` runs over existing pots, idempotent, Cypher-level `MATCH (n)-[r:DECIDES_FOR]->(m) SET m:Decision`.

## Risks

- Over-labelling. A node that is target of `OWNS` isn't always a `Component`; it might be a `Release`. Mitigation: inference rules return *candidate* labels with confidence; a tie or low confidence drops to no label, not a wrong label.
- Graphiti's label set is not strict Neo4j labels today — check the adapter path to make sure `--node-labels` maps to something queried in Cypher, not just a property filter.

## Rollout

1. Rule table + inference pass, gated by `CONTEXT_ENGINE_INFER_LABELS=1`, new ingests only.
2. Verify `--node-labels Decision` returns the expected nodes in the test pot.
3. Backfill existing pots.
4. Document the rule table in `docs/context-graph/graph.md`.

## Done when

- `potpie search "ledger" --node-labels Decision` returns at least the ADR-derived decision node in the test fixture.
- `context_resolve intent=feature include=["decisions"]` can be answered from the graph without substring-matching the word "decided" in summaries.
