# Context Graph Gaps And Next Steps

This file is retained as a pointer for older links.

The active code-reviewed migration plan is
[`implementation-next-steps.md`](implementation-next-steps.md). It supersedes
the older phase summary that previously lived here.

Current highest-priority gaps:

- remove GitHub PR compatibility from graph writes
- collapse application reads and writes behind the Graphiti-backed graph layer
- replace hardcoded query dispatch with query planning and presets
- make the Ingestion Agent context-aware and tool-capable
- expose provenance, freshness, source verification, and conflicts in every
  consumer-facing context envelope
- deepen `context_status` with source rows, ledger health, resolver
  capabilities, verification state, and maintenance recommendations
- add non-GitHub source ingestion and resolvers

Use [`graph.md`](graph.md) for the product architecture and
[`unified-graphiti-application-architecture.md`](unified-graphiti-application-architecture.md)
for the one-graph application-layer target.
