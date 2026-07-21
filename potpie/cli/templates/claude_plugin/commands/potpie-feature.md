Use Potpie context before feature work. Load `potpie-project-preferences` first,
and load `potpie-infra-architecture` when the change touches services,
deployment, adapters, or runtime behavior.

If the `potpie` CLI is available, read the graph directly:

```bash
potpie --json graph catalog --task "feature work in repo:<owner/repo>"
potpie --json graph describe decisions --view preferences_for_scope --examples
potpie graph read --subgraph decisions --view preferences_for_scope --scope repo:<owner/repo>
potpie graph read --subgraph decisions --view active_decisions --scope repo:<owner/repo>
potpie graph read --subgraph infra_topology --view service_neighborhood --scope service:<name> --depth 2
```

Expand the user's request into a good retrieval `--query`/`query` (add symptoms and
synonyms a future searcher would type). Inspect coverage, freshness, quality,
fallbacks, open conflicts, and source refs before relying on the result.
