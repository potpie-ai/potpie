---
name: "potpie-infra-architecture"
version: "2"
recommended: true
description: "Use for project infra and architecture context: environments, adapters, deployments, service dependencies, datastores, API contracts, ownership, and dependency blast radius."
---

# Potpie Infra Architecture

Use this skill when the task touches deployments, environments, adapters,
runtime configuration, service dependencies, production incidents, or
architecture changes.

## Read First

Start from the service or environment named by the task:

```bash
potpie --json graph read \
  --view infra_topology.service_neighborhood \
  --scope service:<service-name> \
  --depth 2 \
  --direction both \
  --environment prod \
  --limit 20
```

MCP equivalent:

```json
{"intent":"operations","include":["infra_topology","timeline","owners"],"mode":"balanced","source_policy":"summary"}
```

For feature work, include infra alongside preferences:

```json
{"intent":"feature","include":["coding_preferences","infra_topology","decisions","owners","docs"],"mode":"fast","source_policy":"references_only"}
```

## What To Look For

Capture and query these architecture facts when they are explicit in source
material:

- Service to repository: `DEFINED_IN`.
- Service to environment: `DEPLOYED_TO`.
- Service to service: `DEPENDS_ON`.
- Service to datastore or dependency: `USES`.
- Service to API contract: `EXPOSES`.
- Environment to cluster/platform: `HOSTED_ON`.
- Service or repo ownership: `OWNED_BY`.

Always preserve environment qualifiers (`dev`, `staging`, `prod`, preview) when a
dependency differs by environment. A prod dependency should not silently replace a
staging dependency.

## Record Architecture

The harness must read and understand the source first. Do not infer topology by
walking a file tree or by seeing package names alone; no scan or deterministic
config walker writes topology to the graph. Use `authoritative_fact`
only when the evidence is an explicit source of truth such as a deployment config,
infra doc, service manifest, ADR, or user statement. Use `agent_claim` when you are
making a reasoned but lower-authority inference.

Example mutation shape:

```json
{
  "graph_contract_version": "v1.5",
  "pot_id": "local/default",
  "idempotency_key": "mutation:infra:payments-depends-ledger:prod",
  "created_by": {"surface": "cli", "harness": "codex"},
  "operations": [
    {
      "op": "link_entities",
      "subgraph": "infra_topology",
      "subject": {"key": "service:payments-api", "type": "Service", "properties": {"name": "payments-api"}},
      "predicate": "DEPENDS_ON",
      "object": {"key": "service:ledger-api", "type": "Service", "properties": {"name": "ledger-api"}},
      "truth": "authoritative_fact",
      "confidence": 0.95,
      "environment": "prod",
      "description": "payments-api calls ledger-api in prod to post settlements; ledger-api failures surface as settlement posting timeouts and refund completion delays.",
      "evidence": [{"source_ref": "github:pr:412", "authority": "external_system"}]
    }
  ]
}
```

Run `--dry-run` before applying any infra mutation.
