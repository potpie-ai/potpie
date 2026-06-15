Use Potpie context before feature work. Load `potpie-project-preferences` first,
and load `potpie-infra-architecture` when the change touches services,
deployment, adapters, or runtime behavior.

If the `potpie` CLI is available, read the graph directly:

```bash
potpie --json graph read --view decisions.preferences_for_scope --scope repo:<owner/repo>
potpie --json graph read --view decisions.active_decisions --scope repo:<owner/repo>
potpie --json graph read --view infra_topology.service_neighborhood --scope service:<name> --depth 2
```

Otherwise, run `context_resolve` with this recipe (MCP):

```json
{"intent":"feature","include":["coding_preferences","infra_topology","decisions","owners","docs"],"mode":"fast","source_policy":"references_only"}
```

Expand the user's request into a good retrieval `--query`/`query` (add symptoms and
synonyms a future searcher would type). Inspect coverage, freshness, quality,
fallbacks, open conflicts, and source refs before relying on the result.
