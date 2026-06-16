---
name: potpie-repo-baseline
description: "Use when establishing or refreshing a repository's baseline memory in Potpie: purpose, application type, features, services, environments, deploy shape, dependencies, API contracts, datastores, integrations, and explicit preferences. The harness reads selected authored sources and writes graph workbench mutations."
---

# Potpie Repo Baseline

Use this skill when the user asks to ingest, refresh, or establish what a
repository is and how it works.

## Procedure

1. Resolve the pot and source:

```bash
potpie --json pot info
potpie --json source list
potpie source add repo .
```

Source registration records metadata only. It does not ingest.

2. Discover the live graph contract:

```bash
potpie --json graph catalog --task "repo baseline"
potpie --json graph describe features --view feature_context --examples
```

3. Read authored sources first. Stop at the cheapest source that is authoritative
   for the fact.
4. Inspect source files only when they are the source of truth for a durable fact,
   such as routes, service clients, adapters, deployment targets, or API
   contracts.
5. Resolve identity before writing:

```bash
potpie --json graph search-entities "<repo service feature>" --limit 10
```

6. Write one or more semantic mutation batches:

```bash
potpie graph mutation-template --kind repo-baseline
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id>
potpie --json graph history --plan <plan_id>
```

## Source Priority

1. README and authored docs.
2. ADRs, runbooks, architecture docs, deployment docs.
3. Package or app manifests.
4. CI/deploy workflows.
5. Environment templates.
6. Framework config.
7. Route/API specs or visible route entrypoints.
8. Service clients/adapters, only to confirm topology.
9. Datastore usage, only to confirm durable infra facts.

## Baseline Memory

Record source-backed repository purpose, app type, features, deployable services,
environments, deploy shape, dependencies, adapters, datastores, API contracts,
important integrations, ownership, and explicit coding preferences.

Represent capabilities as `Feature` entities; link repos or services to them
with `PROVIDES`, and use `IMPLEMENTED_IN` when a source locates implementation.

Use `source_observation` or `authoritative_fact` for explicit evidence. Use
`agent_claim` for lower-authority interpretation. Every entity and claim needs a
compact summary, retrieval-grade description, truth class, confidence, and source
refs when available.

Query the result back with:

```bash
potpie --json graph read --subgraph features --view feature_context --scope anchor_entity_key:<repo-key>
potpie --json graph read --subgraph infra_topology --view service_neighborhood --scope service:<service>
```

## Boundaries

Baseline capture is harness-led. Do not walk the tree or run scanner-driven graph
updates to invent modules, services, features, or dependencies. Do not record
dependencies just because a lockfile mentions them. Do not use PRs or issues to
infer baseline architecture; change history belongs in `potpie-change-timeline`.
