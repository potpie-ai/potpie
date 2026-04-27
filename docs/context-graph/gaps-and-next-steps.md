# Context Graph Gaps And Next Steps

This file is retained as a pointer for older links.

The active code-reviewed migration plan is
[`implementation-next-steps.md`](implementation-next-steps.md). It supersedes
the older phase summary that previously lived here.

Current highest-priority gaps (see `implementation-next-steps.md` for the live
ranked list and shipped/open status):

- tighten extraction / ontology classification so plain decision episodes land
  as `Decision` + `DECIDED` / `SUPERSEDES` rather than a mass of `Feature` +
  `RELATES_TO` (partially addressed 2026-04-22 with the deterministic label
  inference pass in `reconciliation_validation.py`; further drift-to-canonical
  rewrites still open)
- extend the semantic→structural bridge beyond causal-expand — `get_decisions`
  and `get_change_history` now fall back to semantic seeds when scope is empty
  (shipped 2026-04-22); remaining family legs still to sweep
- add non-GitHub source ingestion and resolvers

Recently closed (2026-04-22): entity canonicalization at the head of
reconciliation validation, LLM-backed answer synthesis for `goal=answer`, CLI
parity with the skill surface (`status`, `resolve`, `overview`, `record`), and
`potpie ingest --sync` / `potpie event list` UX sweeps.

Use [`graph.md`](graph.md) for the product architecture and
[`unified-graphiti-application-architecture.md`](unified-graphiti-application-architecture.md)
for the one-graph application-layer target.
