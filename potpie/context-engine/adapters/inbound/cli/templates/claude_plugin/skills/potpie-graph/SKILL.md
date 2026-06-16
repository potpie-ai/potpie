---
name: potpie-graph
description: "Use for explicit Potpie graph workbench operations or when another Potpie skill needs advanced graph usage: catalog, describe, read named views, search-entities, mutation-template, propose, commit, bulk apply, history, inbox, quality, corrections, and graph nudges. Prefer use-case skills for ordinary code, debug, infra, timeline, or ingestion context."
---

# Potpie Graph Workbench

The graph workbench is the CLI surface for project memory. You read named views,
resolve identity, propose validated semantic mutations, commit accepted plans,
and inspect quality. Potpie validates and stores; the agent is the harness that
decides what source material means.

Always pass `--json` for machine-readable workbench commands.

## Start Here

For non-trivial graph work, discover the live contract instead of relying on old
examples:

```bash
potpie --json graph status
potpie --json graph catalog --task "<task>"
potpie --json graph describe <subgraph> --view <view> --examples
```

Trust `graph catalog` over skill examples for available views, operations,
predicates, truth classes, and review requirements.

## Read Views

Common reads:

```bash
potpie --json graph read --subgraph decisions --view preferences_for_scope --scope repo:<repo>,path:<path> --query "<preferences>"
potpie --json graph read --subgraph debugging --view prior_occurrences --query "<symptom>" --limit 12
potpie --json graph read --subgraph recent_changes --view timeline --time-window 7d --limit 20
potpie --json graph read --subgraph infra_topology --view service_neighborhood --scope service:<service> --depth 2 --direction both
potpie --json graph read --subgraph features --view feature_context --scope anchor_entity_key:<repo-or-service-key>
```

Reads return entities with useful immediate relations inline when supported. Check
coverage, freshness, quality, and source refs before relying on results.

Query expansion is the agent's job. Include symptoms, synonyms, commands, files,
services, frameworks, dependencies, and environment terms a future searcher would
type.

Represent capabilities as `Feature` entities. Link repos or services with
`PROVIDES`; link a feature back to implementation with `IMPLEMENTED_IN` when the
source supports it.

## Resolve Identity

Before linking to an existing entity, find the canonical key:

```bash
potpie --json graph search-entities "<name>" --type Service --limit 10
```

Reuse returned keys. Near-duplicate keys fragment recall.

## Write Flow

Use semantic mutation operations only. Never write raw graph CRUD.

```bash
potpie graph mutation-template --kind <preference-policy|infra-snapshot|timeline-change|bug-fix|repo-baseline|feature|decision>
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id>
potpie --json graph history --plan <plan_id>
```

Inspect the proposed diff, warnings, rejected operations, and review flags before
commit. If the update is plausible but not safe to assert, use the inbox:

```bash
potpie --json graph inbox add --summary "<uncertain learning>" --evidence <source-ref> --subgraph <subgraph>
```

## Bulk Writes

For many agent-authored semantic operations, prefer chunked apply over one huge
mutation file:

```bash
potpie --json graph bulk apply --file mutations.ndjson --chunk-size 100 --manifest graph-bulk-manifest.json --verify
```

Use dry-run mode to validate chunks without committing, `--start-chunk <n>` to
resume after fixing a failed chunk, and stable idempotency keys per source or
profile. `graph bulk apply` only applies mutations the harness already authored;
it does not inspect sources, infer facts, or replace agent judgment.

## Writing Standards

Every durable write needs:

- A compact display summary.
- A retrieval-grade description with search terms, symptoms, scope, aliases, and
  consequences.
- An honest truth class: `authoritative_fact`, `source_observation`,
  `user_decision`, `preference`, `agent_claim`, `timeline_event`, or
  `quality_finding`.
- Evidence for source-backed or high-authority claims.

Do not use the graph as a deterministic code scanner. The harness reads selected
source material, interprets it, resolves identity, then writes semantic facts.
For GitHub, Linear, Jira, and similar systems, hydrate records through the
agent's integration tools/connectors before writing graph updates. Do not use
pot-level connector ingestion commands as the graph write path; write semantic
updates through `graph propose` and `graph commit`.

## Quality And Corrections

Use quality reports as read-only maintenance signals:

```bash
potpie --json graph quality summary
potpie --json graph quality duplicate-candidates --limit 20
potpie --json graph quality stale-facts --subgraph infra_topology --limit 20
potpie --json graph quality conflicting-claims --limit 20
```

Repair meaning through `graph propose` and `graph commit`; do not hard-delete
claims unless the live catalog exposes and permits that operation.

## Nudges

If a Potpie nudge injects context, use it as ranked graph context for the current
task. If it asks you to record a learning, decide whether the learning is durable,
then use the write flow or inbox. A nudge is not an automatic write.
