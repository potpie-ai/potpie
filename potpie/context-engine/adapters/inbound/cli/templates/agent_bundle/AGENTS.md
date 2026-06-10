# Context Engine

This project uses Potpie for project memory. Before non-trivial work, read the
graph to orient yourself. After work, record durable learnings that should help
the next agent.

The harness is the intelligence. Potpie validates, lowers, commits, audits, and
ranks graph memory. It does not scan the repository or decide what prose means
for you.

## Quick Start

```bash
potpie doctor
potpie pot list
potpie graph catalog
```

## Surfaces

Prefer the graph CLI when shell is available:

```bash
potpie --json graph catalog
potpie --json graph read --view <subgraph.view> [--query "..."] [--scope key:value] [--limit N]
potpie --json timeline recent [--time-window 7d] [--limit N]
potpie --json graph search-entities "text" [--type Service] [--predicate DEPENDS_ON] [--environment prod] [--limit N]
potpie --json graph mutate --file mutation.json [--dry-run] [--allow-review-required --approved-by user:you]
```

When only MCP is configured, use the compatibility tools:

- `context_status` - readiness, freshness gaps, and recommended recipe.
- `context_resolve` - primary task context wrap.
- `context_search` - narrow follow-up lookup after a resolve.
- `context_record` - durable preferences, fixes, decisions, workflows, docs, and
  incident notes.

Do not add one tool per context type. Express reads as `graph read --view` or as a
`context_resolve` recipe.

## Views

| View | Use it for |
|---|---|
| `preferences.active_preferences` | project/repo/path preferences for code work |
| `infra_topology.service_neighborhood` | env-qualified dependencies and blast radius |
| `recent_changes.timeline` | project-wide PRs, tickets, docs, incidents, deployments; use `potpie timeline recent` |
| `bugs.prior_occurrences` | prior symptoms, fixes, failed attempts |
| `decisions.active_decisions` | active architectural/product decisions |
| `ownership.owner_context` | owners for a scope |
| `docs.reference_context` | docs and runbooks for a scope |
| `admin.inspection_slice` | raw canonical graph for debugging |

## Recipes

Feature or code work:

```json
{"intent":"feature","include":["coding_preferences","infra_topology","decisions","owners","docs"],"mode":"fast","source_policy":"references_only"}
```

Debugging:

```json
{"intent":"debugging","include":["prior_bugs","infra_topology","timeline"],"mode":"fast","source_policy":"references_only"}
```

Review:

```json
{"intent":"review","include":["coding_preferences","decisions","timeline","owners"],"mode":"balanced","source_policy":"summary"}
```

Operations:

```json
{"intent":"operations","include":["infra_topology","timeline","owners"],"mode":"balanced","source_policy":"summary"}
```

Docs/onboarding:

```json
{"intent":"onboarding","include":["infra_topology","coding_preferences","docs","owners"],"mode":"fast","source_policy":"references_only"}
```

## Writing

Two rules carry most of the value:

1. Resolve identity first with `graph search-entities` before linking to an
   existing service, repo, bug, decision, person, or document.
2. Write retrieval-grade descriptions. Include the symptom text, synonyms, scope,
   environment, service, files, commands, and source refs a future searcher would
   type.

Use semantic operations only: `upsert_entity`, `link_entities`, `assert_claim`,
`append_event`, `end_relation_validity`, and `retract_claim`. Never hard-delete a
claim; end its validity or retract it.

Example infra write:

```json
{
  "graph_contract_version": "v1.5",
  "pot_id": "local/default",
  "idempotency_key": "mutation:infra:payments-ledger-prod",
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
      "description": "payments-api calls ledger-api in prod to post settlements; ledger-api failures surface as refund and settlement timeout incidents.",
      "evidence": [{"source_ref": "github:pr:412", "authority": "external_system"}]
    }
  ]
}
```

`context_record` is the simpler compatibility write for preferences, bug patterns,
fixes, verifications, decisions, doc references, workflows, runbooks, and incident
summaries. It lowers through the same semantic mutation path.

## Ingestion Boundary

There is no local code scan path in the agent instructions. For a repo link, doc,
ticket, PR, issue, or web link, the harness reads the source, decides what durable
facts exist, and writes graph mutations. Do not infer services, dependencies,
features, or preferences from directory names or package files alone.

Existing source queue commands such as `potpie pot linear-team ingest` and
`potpie pot jira-project ingest` queue connector events. They are not working-tree
scans.

## Responding To Nudges

A hook may inject context or instructions from `potpie graph nudge`.

- `inject_context` - use the injected facts for the current task.
- `instruction` - a prompt to decide whether something durable was learned. If it
  was, resolve identity and write a semantic mutation or `context_record`. If not,
  do nothing.

## Skills

Use these repo-local skills under `.agents/skills/`:

- `potpie-project-preferences` - error handling, structure, libraries, frameworks,
  logging, testing, and coding guidelines before code work.
- `potpie-infra-architecture` - environments, adapters, deployment topology,
  service dependencies, datastores, API contracts, and ownership.
- `potpie-change-timeline` - recent or historical PRs, tickets, docs, incidents,
  deployments, and regression correlation.
- `potpie-debug-memory` - prior bugs, fixes, failed attempts, verification, and dev
  setup troubleshooting.
- `potpie-source-ingestion` - harness-led ingestion from repo links, docs, PRs,
  issues, tickets, runbooks, logs, and web links.
- `potpie-graph` - graph CLI contract: catalog/read/search-entities/mutate and
  nudge handling.
- `potpie-agent-context` - MCP `context_*` compatibility recipes.
- `potpie-cli` - CLI setup, pot/source commands, graph commands, and
  troubleshooting.
- `potpie-pot-scope` - resolving active pot and repo-to-pot mapping.
