---
name: potpie-graph
description: "Use for explicit Potpie graph workbench operations or when another Potpie skill needs graph status, catalog, describe, read, search-entities, propose, commit, bulk apply, history, inbox, quality reports, corrections, graph nudges, identity resolution, proposal status handling, or post-ingestion verification."
---

# Potpie Graph Workbench

The graph workbench is the CLI surface for project memory. The harness reads
named views, resolves identity, proposes semantic mutations, commits accepted
plans, and verifies results. Potpie validates and stores; the harness decides
what source material means.

Use text output for routine context reads. Add `--json` when a workflow needs
exact machine parsing, mutation plans, commits, history verification, or full
evidence/debug payloads.

## Preflight

For non-trivial graph work, discover the live contract before relying on skill
examples:

```bash
potpie graph status
potpie graph catalog --task "<task>" --profile read
potpie graph describe <subgraph> --view <view> --examples
```

Trust `graph catalog` over skill examples for available views, operations,
predicates, truth classes, source authorities, and review requirements. If the
CLI is unavailable, gather evidence but stop before committing graph writes.

## Read Views

Common reads:

```bash
potpie graph read --subgraph decisions --view preferences_for_scope --scope repo:<repo>,path:<path> --query "<preferences>" --limit 12
potpie graph read --subgraph debugging --view prior_occurrences --query "<symptom>" --limit 12
potpie graph read --subgraph recent_changes --view timeline --time-window 7d --limit 20 --format table
potpie graph read --subgraph recent_changes --view timeline --source-ref <github-pr-or-issue-ref> --format table
potpie graph read --subgraph infra_topology --view service_neighborhood --scope service:<service> --depth 2 --direction both
potpie graph read --subgraph features --view feature_context --scope anchor_entity_key:<repo-or-service-key>
potpie graph neighborhood --entity service:<service> --predicate USES --detail summary --limit 20
```

Text reads return compact summaries for fast orientation. Use
`--json --detail full --relations full` only when you need full inline relation
payloads for debugging or exact machine processing. Check coverage, freshness,
quality, and source refs before relying on results.
Expand queries with symptoms, aliases, commands, files, services, frameworks,
dependencies, environments, and source IDs a future searcher would type.

## Resolve Identity

Resolve identity before linking to an existing entity:

```bash
potpie graph search-entities "<name>" --limit 10
potpie graph search-entities "<service>" --type Service --environment prod --limit 10
potpie graph search-entities "<feature or repo>" --subgraph features --limit 10
potpie graph search-entities "<ticket-or-pr-id>" --external-id "<external-id>" --limit 10
potpie graph search-entities "<ticket-or-pr-id>" --source-ref <github-pr-or-issue-ref> --limit 10
```

Reuse returned canonical keys. Use type, subgraph, predicate, scope, truth,
environment, external-id, and source-ref filters when known. Prefer
`--source-ref` for exact PR, issue, ticket, doc, and deploy handles.
Near-duplicate keys fragment recall; if resolution is uncertain, use
`graph inbox add` or a reviewed correction flow instead of creating another
entity.

## Write Flow

Use semantic mutation operations only. Never write raw graph CRUD.

```bash
potpie --json graph propose --file mutation.json
```

Inspect the proposal before commit:

- `invalid` or rejected operations: fix the mutation or skip the weak fact.
- `conflict`: resolve identity, narrow scope, or use inbox.
- `review_required`: ask for approval or commit only with the required
  `--approved-by` value when policy allows.
- `validated` / low-risk: commit with `--verify`.

```bash
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

`graph mutation-template --kind <kind>` is an optional skeleton helper. Prefer
`graph describe ... --examples` and the live `graph catalog` when authoring
mutations.

## Writing Standards

Every durable write needs:

- Compact display summary.
- Retrieval-grade description with search terms, symptoms, aliases, scope,
  environment, files, commands, source refs, and consequences.
- Honest truth class: `authoritative_fact`, `source_observation`,
  `user_decision`, `preference`, `agent_claim`, `timeline_event`, or
  `quality_finding`.
- Source authority such as `authoritative_code`, `repository_metadata`,
  `external_system`, `user_statement`, or `agent_observation`.
- Evidence/source refs for source-backed or high-authority claims.

Represent capabilities as `Feature` entities. Link repos or services with
`PROVIDES`; link a feature back to implementation with `IMPLEMENTED_IN` when
the source supports it.

Do not use the graph as a deterministic code scanner. The harness reads selected
source material, interprets it, resolves identity, then writes semantic facts.
For GitHub, Linear, Jira, and similar systems, hydrate records through the
agent's integration tools/connectors before writing graph updates. Do not use
pot-level connector ingestion commands as the graph write path.

## Bulk Writes

For many agent-authored semantic operations, prefer chunked apply over one huge
mutation file:

```bash
potpie --json graph bulk apply --file mutations.ndjson --chunk-size 100 --manifest graph-bulk-manifest.json --verify
```

Use dry-run mode to validate chunks without committing, `--start-chunk <n>` to
resume after fixing a failed chunk, and stable idempotency keys per source or
profile. Bulk apply only applies mutations the harness already authored; it does
not inspect sources, infer facts, or replace agent judgment.

## Inbox

Use inbox when evidence may matter but the canonical update is uncertain:

```bash
potpie --json graph inbox add --summary "<uncertain learning>" --evidence <source-ref> --subgraph <subgraph>
potpie --json graph inbox list --status pending
potpie --json graph inbox show <item_id>
```

Inbox items are pending work, not canonical graph facts.

## Quality Gates

Run quality reports after broad ingestion or corrections:

```bash
potpie --json graph quality summary
potpie --json graph quality duplicate-candidates --limit 20
potpie --json graph quality stale-facts --subgraph infra_topology --limit 20
potpie --json graph quality conflicting-claims --limit 20
potpie --json graph quality orphan-entities --limit 20
potpie --json graph quality low-confidence --limit 20
potpie --json graph quality projection-drift --limit 20
```

Repair meaning through `graph propose` and `graph commit --verify`; do not hard-delete
claims unless the live catalog exposes and permits that operation.

## Nudges

If a Potpie nudge injects context, use it as ranked graph context for the
current task. If it asks you to record a learning, decide whether the learning
is durable, then use the write flow or inbox. A nudge is not an automatic write.
