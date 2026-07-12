---
name: potpie-repo-baseline
description: "Use when establishing, refreshing, or deeply understanding a repository's baseline memory in Potpie: purpose, application type, features, services/modules, environments, deploy shape, dependencies, API contracts, datastores, integrations, ownership, and explicit preferences. The harness reads authored and code-adjacent sources, then writes graph workbench mutations."
---

# Potpie Repo Baseline

Use this skill when the user asks to ingest, refresh, establish, or deeply
understand what a repository is and how it works.

## Procedure

1. Resolve the pot and source:

```bash
potpie --json pot info
potpie --json source list
potpie source add repo . --pot <pot-id-or-name>
```

Source registration records metadata only. It does not ingest or scan.

2. Discover the live graph contract:

```bash
potpie --json graph status
potpie --json graph catalog --task "deep repo baseline"
potpie --json graph catalog --subgraph features
potpie --json graph catalog --subgraph infra_topology
potpie --json graph catalog --subgraph decisions
```

3. Create todos for the baseline lanes: docs/product, repo map,
   runtime/deploy, API/data/integrations, preferences/workflows, synthesis,
   identity resolution, write, verify.
4. Read authored sources first, then inspect source files that are authoritative
   for durable facts: routes, service clients, adapters, deployment targets,
   API contracts, model/datastore usage, and test/workflow commands.
5. Resolve identity before writing:

```bash
potpie graph search-entities "<repo service feature>" --limit 10
potpie graph search-entities "<service>" --type Service --environment prod --limit 10
```

6. Write one or more semantic mutation batches:

```bash
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

`graph mutation-template --kind repo-baseline` is an optional skeleton helper;
trust `graph catalog` and `graph describe ... --examples` for the live contract.

## Deep Baseline Mode

Use this mode whenever the user says "ingest repo", "deeply understand",
"baseline this repo", or asks for broad repo memory.

1. Product and docs:
   - README, docs, ADRs, runbooks, contributing guide, package metadata,
     public docs, linked websites.
   - Capture purpose, app type, domain vocabulary, explicit features,
     decisions, preferences, and workflows.
2. Repo map:
   - Manifests, top-level apps/packages, framework config, route/API
     entrypoints, generated API specs, tests.
   - Capture services/modules only when a source clearly supports them.
3. Runtime and deploy:
   - Dockerfiles, compose, Kubernetes, Terraform, CI workflows, deploy scripts,
     environment templates, feature flags.
   - Capture environments, deploy shape, config variables, release/test
     workflows, and ownership when explicit.
4. API, data, and integrations:
   - Service clients, adapters, auth providers, queues, datastore/model usage,
     external API clients.
   - Capture APIContract, DataStore, Adapter, Dependency, and integration facts
     with file/doc evidence.
5. Preferences and local workflows:
   - Contribution docs, lint/test configs, Makefiles, package scripts,
     PR templates, code comments that explicitly state policy.
   - Record only reusable explicit preferences.
6. Synthesis:
   - Build an evidence matrix before writing. Split high-confidence facts from
     uncertain observations. Use inbox for useful but weak findings.

## Source Priority

1. README and authored docs.
2. ADRs, runbooks, architecture docs, deployment docs.
3. Package/app manifests and top-level workspace definitions.
4. CI/deploy workflows and environment templates.
5. Framework config and route/API specs.
6. Visible route/API entrypoints.
7. Service clients/adapters, only to confirm topology.
8. Datastore/model usage, only to confirm durable infra facts.
9. Tests and fixtures, only to confirm workflows, API behavior, or feature
   intent that is otherwise explicit.

## Baseline Memory

Record source-backed repository purpose, app type, features, deployable services
or modules, environments, deploy shape, dependencies, adapters, datastores,
API contracts, important integrations, ownership, local workflows, and explicit
coding preferences.

Use canonical entity families when writing: `Repository`, `Service`, `Feature`,
`Environment`, `DataStore`, `APIContract`, `Dependency`, and `Preference`.
Represent capabilities as `Feature` entities; link repos or services with
`PROVIDES`, and use `IMPLEMENTED_IN` when a source locates implementation.
Use topology relations such as `DEFINED_IN`, `DEPLOYED_TO`, `DEPENDS_ON`,
`USES`, `EXPOSES`, `USES_ADAPTER`, `CONFIGURES`, and `OWNED_BY` only when the
source supports the relation. Link explicit preferences with
`POLICY_APPLIES_TO`.

Use `authoritative_fact` for explicit source-of-truth evidence, such as docs,
deployment config, API specs, or source files that define behavior. Use
`source_observation` for direct observations that are not necessarily policy.
Use `agent_claim` for lower-authority synthesis. Every entity and claim needs a
compact summary, retrieval-grade description, confidence, truth class, source
authority, and source refs when available.

## Mutation Requirements

Before proposing:

- Resolve existing repository, service, feature, dependency, and owner entities.
- Prefer updating/linking existing entities over creating near-duplicates.
- Include evidence for `authoritative_fact` and `source_observation`.
- Keep one mutation file to one coherent family or source slice when possible.
- Put low-confidence but useful findings into `graph inbox add`.

After committing:

```bash
potpie graph read --subgraph features --view feature_context --scope anchor_entity_key:<repo-key> --limit 50
potpie graph read --subgraph infra_topology --view service_neighborhood --scope service:<service> --depth 2 --direction both --limit 50
potpie --json graph quality duplicate-candidates --limit 20
potpie --json graph quality low-confidence --limit 20
potpie --json graph quality conflicting-claims --limit 20
```

## Boundaries

Baseline capture is harness-led. Do not run scanner-driven graph updates or
legacy ingestion commands to invent modules, services, features, or
dependencies. Do not record dependencies just because a lockfile mentions them.
Do not infer baseline architecture from PR titles or issue status; change
history and change-history facts belong in `potpie-change-timeline`.
Local file inspection is allowed and expected, but the harness must read,
interpret, and cite the evidence before writing semantic facts.
