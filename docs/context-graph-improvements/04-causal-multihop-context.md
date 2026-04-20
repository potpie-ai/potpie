# 04 — Causal / multi-hop context missing from search and resolve

## Problem

"What caused the ledger migration decision?" returned post-migration effects (decommissioned cluster, migration event) in the top-5 but did **not** surface the March 2025 *MongoDB scaling pain* episode (write contention, 40+ minute aggregations). The actual cause is one semantic hop away and is lost to cosine similarity on the word "migration".

`search` is a flat semantic ranker. `context_resolve` has the machinery to plan multi-entity reads but doesn't currently do causal edge traversal for `intent=debugging`.

## Proposal

Two levels — the first is a quick search upgrade; the second is a `context_resolve` capability.

### Level 1 — Edge-expanded search (hybrid retrieval)

After the semantic top-K:

1. Take the top-3 result nodes.
2. Expand one hop over a fixed edge whitelist: `CAUSED`, `CAUSES`, `DECIDES_FOR` (reverse), `FIXES` (reverse), `PRECEDES`, `TRIGGERED_BY`.
3. Merge expanded results into the ranking with a decay factor (0.6× top-K score), dedupe by edge uuid.
4. Cap additions so the output never exceeds `--limit`.

This is a local change in `application/use_cases/query_context.py` and reads cheaply from Neo4j via `structural.py`.

### Level 2 — Causal recipe in `context_resolve`

Add a `causal_chain` include-type to `intelligence_policy.py`. When present in `include`:

- Plan a breadth-first walk from the focal entity backwards along `CAUSED`, `TRIGGERED_BY`, `PRECEDES`, bounded by (a) max depth from `budget`, (b) time window from `as_of ± window`.
- Return the chain as an ordered list under `facts.causal_chain`, each node carrying `reference_time`, `source_refs`, and `confidence`.

Wire `intent=debugging` recipe in `domain/agent_context_port.py` to include `causal_chain` by default.

## Files touched

- `app/src/context-engine/application/use_cases/query_context.py` — edge-expanded search.
- `app/src/context-engine/adapters/outbound/neo4j/structural.py` — new `expand_causal_neighbours(node_uuids, edge_types, depth)` method.
- `app/src/context-engine/domain/intelligence_models.py` — `CausalChainItem` schema.
- `app/src/context-engine/domain/intelligence_policy.py` — `causal_chain` planner.
- `app/src/context-engine/application/services/context_resolution.py` — populate `facts.causal_chain` when planned.
- `app/src/context-engine/domain/agent_context_port.py` — update `intent=debugging` recipe.
- `app/src/context-engine/adapters/inbound/cli/output.py` — surface a small "↳ because: …" line under each search result that was brought in via causal expansion.
- Tests: integration test that reproduces the exact failure above (6-episode fixture, assert scaling-pain episode appears in top-5 for causal query).

## Dependencies

- Works **much** better once #02 (edge types) and #03 (node labels) land, because causal edges aren't collapsed into `MODIFIED` and nodes are queryable by canonical label. Level 1 is still useful standalone.
- Auto-extraction of `CAUSED` edges is the open question: Graphiti's extractor sometimes produces it from prose like "because", "due to", "as a result of", but not consistently. Consider a fallback heuristic during reconciliation that creates a soft `CAUSED` edge (`confidence: low`) when two episodes share a subject and the later episode's reference_time is within N days of the earlier one and one cites the other's source.

## Risks

- Edge expansion inflates noise for short queries. Mitigation: expand only when top-1 score is above a threshold; otherwise trust semantic.
- Causal chains are only as good as the `CAUSED` edges present. Measure extraction recall on a fixed fixture.

## Rollout

1. Ship Level 1 behind `CONTEXT_ENGINE_CAUSAL_EXPAND=1`.
2. Evaluate on the 6-episode fixture and on a real pot with known incidents (e.g. the `potpie-ai/potpie` pot in dev).
3. Ship Level 2 once #03 lands (needs `Incident` and `Decision` labels).
4. Update `intent=debugging` recipe default.

## Done when

- `potpie search "what caused the ledger migration decision?"` surfaces the MongoDB scaling-pain episode in the top-5 without any `--node-labels` hint.
- `context_resolve intent=debugging scope=ledger-service` returns `facts.causal_chain` containing at least the `{scaling_pain → migration_decision → migration_completed}` sequence in order.
