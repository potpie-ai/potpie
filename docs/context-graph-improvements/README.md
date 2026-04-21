# Context Graph Improvements

Focused plans for gaps surfaced during the 2026-04-20 CLI test pass against a fresh pot of 6 temporal episodes. Each file is a standalone plan: problem, observed evidence, proposal, touched files, risks, rollout.

| # | File | Theme | Rough size |
|---|------|-------|-----------|
| 1 | [01-temporal-resolution-in-search.md](01-temporal-resolution-in-search.md) | Surface `valid_at` / `invalid_at` / `superseded` by default; auto-invalidate contradicted facts | M |
| 2 | [02-edge-type-collapse.md](02-edge-type-collapse.md) | Stop collapsing action verbs onto generic `MODIFIED`; add `lifecycle_status` | M |
| 3 | [03-canonical-node-labels.md](03-canonical-node-labels.md) | Make `--node-labels Decision` actually match extracted nodes | S |
| 4 | [04-causal-multihop-context.md](04-causal-multihop-context.md) | Follow `Incident → CAUSED → Decision` on `context_resolve`; expand search with incident edges | L |
| 5 | [05-provenance-in-cli-output.md](05-provenance-in-cli-output.md) | Show `source`, `reference_time`, `episode_uuid` in `search` by default | S |
| 6 | [06-conflict-surfacing.md](06-conflict-surfacing.md) | Detect and emit `open_conflicts` when facts contradict; integrate with quality report | M |

Post-implementation test findings (2026-04-21) and concrete fixes for the remaining gaps live in a separate folder: [`../context-graph-fixes/`](../context-graph-fixes/README.md).

## Reading order

If you only pick two: **#5** (provenance) is the cheapest visible win; **#1** (temporal) is the highest value-per-line because every downstream agent workflow silently trusts stale facts today.

**#1, #2, #6** are correlated — they share the extraction-and-reconciliation pipeline. If several are tackled together, touch `domain/graph_mutations.py` and `application/use_cases/reconciliation_validation.py` once rather than three times.

## Out of scope for this round

- Replacing Graphiti with a custom extractor.
- Adding new MCP tools (the four-tool port is deliberate — see `docs/context-graph/graph.md`).
- UI/front-end surfaces.
