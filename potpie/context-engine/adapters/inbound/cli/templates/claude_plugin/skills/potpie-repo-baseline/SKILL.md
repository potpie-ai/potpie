---
name: "potpie-repo-baseline"
version: "1"
recommended: true
description: "Use when establishing or refreshing a repository's baseline memory in Potpie: purpose, application type, features, services, environments, dependencies, API contracts, datastores, and explicit preferences. The harness reads selected authored sources and writes semantic graph mutations; Potpie registers, reads, validates, and stores — it never scans the repo."
---

# Potpie Repo Baseline Understanding

Repo ingestion is harness-led. You break the repo down into investigation
steps, read selected sources, decide which facts are durable, and write
semantic mutations with evidence. Potpie registers the source, serves the
graph contract, resolves identity, validates, and stores. There is no CLI
scanner and no deterministic file walker — if a fact reaches the graph, it is
because you read its source and asserted it.

Baseline understanding is distinct from change-history ingestion (merged PRs
and issues — see `potpie-change-timeline`). Run them separately; never infer
the architecture from PR titles.

## Procedure

1. **Resolve the pot.** `potpie --json pot info`. If no pot resolves, create
   one (`potpie pot create <name> --use`) or ask the user which pot to use.
2. **Register the repo source if missing.** Check `potpie --json source list`;
   if the repo is absent, run `potpie source add repo .` (or pass the path /
   `owner/repo`). Registration records metadata only — it does not ingest.
3. **Read the graph contract.** `potpie --json graph catalog` — entity types,
   predicates, views, mutation ops. Trust the catalog over this document.
4. **Search before writing.** For every entity you plan to assert, resolve its
   canonical key first: `potpie --json graph search-entities "<name>"
   --type <Type>`. Reuse existing keys; near-duplicate keys fragment memory.
5. **Read authored docs first**, in the source-priority order below. Stop at
   the cheapest source that is authoritative for the fact.
6. **Inspect source files only when they are the source of truth** for a
   durable fact (a route table, a service client, a deployment target). Read
   the specific file; do not walk the tree.
7. **Write one mutation batch** with `potpie --json graph mutate --file
   mutation.json --dry-run`, fix any issues, then apply without `--dry-run`.

## Source priority order

1. README and authored docs.
2. ADRs, runbooks, architecture docs, deployment docs.
3. Package/app manifests (`package.json`, `pyproject.toml`, `go.mod`, …).
4. CI/deploy workflows.
5. Environment templates (`.env.example`, helm values, …).
6. Framework config.
7. Route/API specs or visible route entrypoints.
8. Service clients/adapters — only to confirm topology.
9. Datastore usage — only to confirm durable infra facts.

## Baseline memory families

| Fact | Entities | Predicates |
|---|---|---|
| Repository purpose / app type | `Repository` | properties + claims on the repo node |
| Features / functionality | `Feature` | `PROVIDES` (repo/service → feature), `IMPLEMENTED_IN` (feature → repo/service/code) |
| Services / deployable units | `Service` | `DEFINED_IN` (service → repo) |
| Environments / deploy shape | `Environment`, `Cluster` | `DEPLOYED_TO`, `HOSTED_ON` |
| Service dependencies | `Service`, `Dependency` | `DEPENDS_ON`, `USES` |
| API contracts / routes | `APIContract` | `EXPOSES` |
| Datastores / integrations | `DataStore` | `USES` |
| Explicit preferences | `Preference` | `POLICY_APPLIES_TO` |

Query baselines back with `potpie --json graph read --view features.feature_context
--scope anchor_entity_key:<repo-key>` ("what does this repo do?") and
`--view infra_topology.service_neighborhood` (dependencies and deploy shape).

## Mutation requirements

Every entity write carries a compact `summary` (display/browse) **and** a
retrieval-grade `description` (symptoms, synonyms, and scope a future
searcher would type). Every claim carries an honest `truth` class,
`confidence`, and `evidence` refs:

- `source_observation` / `authoritative_fact` — the source explicitly says it;
  cite it (`repo:<repo>#README`, `repo:<repo>#package.json`,
  `repo:<repo>#route:<path>`, or a URL).
- `agent_claim` — your reasoned interpretation; lower authority, no evidence
  required but still cite what you read.

```json
{
  "pot_id": "<pot>",
  "idempotency_key": "baseline:<repo>:v1",
  "created_by": {"surface": "cli", "harness": "<harness>"},
  "operations": [
    {
      "op": "assert_claim",
      "subject": {"key": "repo:github.com/acme/shop", "type": "Repository",
                  "name": "shop",
                  "summary": "Next.js storefront for the Acme shop",
                  "description": "Customer-facing storefront web app: product catalog, cart, checkout, order history. Next.js + TypeScript; talks to payments-api and inventory-api. Searchable as: webshop, storefront, e-commerce frontend."},
      "predicate": "PROVIDES",
      "object": {"key": "feature:checkout", "type": "Feature",
                 "name": "checkout",
                 "summary": "Cart checkout and payment capture flow",
                 "description": "End-to-end checkout: cart review, address entry, payment capture via payments-api, order confirmation emails. Searchable as: buy flow, purchase, place order, payment page."},
      "truth": "source_observation",
      "confidence": 0.9,
      "description": "README 'Features' section lists checkout as a primary storefront capability backed by payments-api.",
      "evidence": [{"source_ref": "repo:acme/shop#README", "authority": "repository_metadata"}]
    }
  ]
}
```

## Anti-patterns

- Do not run or build a code scanner; no deterministic path may mutate the graph.
- Do not walk the file tree to discover modules, services, features, or
  dependencies; read the selected sources above.
- Do not create a service or feature because a directory has a suggestive name.
- Do not record dependencies just because a lockfile mentions them.
- Do not skip `graph search-entities` before writing.
- Do not omit `summary`, retrieval-grade `description`, `truth`, or evidence.
- Do not use change-history (PRs/issues) to infer the baseline architecture.
