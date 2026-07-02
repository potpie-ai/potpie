---
name: potpie-infra-architecture
description: "Use for project infra and architecture context: environments, adapters, runtime configuration, deployments, service dependencies, datastores, API contracts, ownership, incidents, and dependency blast radius."
---

# Potpie Infra Architecture

Use this skill when the task touches environments, adapters, deployment behavior,
runtime config, service dependencies, production incidents, or architecture
changes.

## Fast Path

Start from the service, environment, adapter, or dependency named by the task.
Resolve identity before assuming keys:

```bash
potpie graph search-entities "<service env adapter dependency>" --limit 10
```

Read the service neighborhood with the graph workbench:

```bash
potpie graph read \
  --subgraph infra_topology \
  --view service_neighborhood \
  --scope service:<service-name> \
  --environment <dev|staging|prod|preview> \
  --depth 2 \
  --direction both \
  --limit 20
```

Omit `--environment` only when the task is environment-agnostic. For broad
architecture work, run `potpie --json graph describe infra_topology --view
service_neighborhood --examples` before choosing the read shape.

## Apply Results

Use infra context for blast radius, where-to-look decisions, deployment changes,
adapter selection, and incident debugging. Preserve environment qualifiers; a
staging dependency is not proof of a production dependency.

Look for explicit topology facts: `DEFINED_IN`, `DEPLOYED_TO`, `DEPENDS_ON`,
`USES`, `EXPOSES`, `HOSTED_ON`, and `OWNED_BY`.

## Record Architecture

Record only source-backed topology or carefully labeled agent inferences. Use
`authoritative_fact` when evidence is an explicit source of truth such as
deployment config, service manifest, infra doc, ADR, or user statement. Use
`agent_claim` for lower-authority interpretation.

Use the workbench write flow:

```bash
potpie --json graph catalog --task "record infra architecture"
potpie graph search-entities "<service>" --type Service --limit 10
potpie --json graph describe infra_topology --view service_neighborhood --examples
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

Every durable infra fact needs an environment when the fact differs by
environment, evidence when available, and a retrieval-grade description.

Architecture capture is harness-led: inspect authoritative sources and write
semantic facts. Do not use scanner-driven graph updates or infer topology from
directory names, imports, or package files alone.

## MCP Fallback

Use this only when the `potpie` CLI is unavailable:

```json
{"intent":"operations","include":["infra_topology","timeline","owners"],"mode":"balanced","source_policy":"summary"}
```

Include families are MCP-only vocabulary; in the CLI these are served by the
graph views `infra_topology.service_neighborhood`, `recent_changes.timeline`,
and `code_topology.ownership_by_path` — `graph read` does not accept include
family names.
