<!-- potpie-start -->
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
potpie graph status
potpie graph catalog --task "<task>"
```

## Surfaces

Prefer the graph CLI when shell is available:

```bash
potpie graph status
potpie graph catalog --task "<task>" --profile read
potpie graph describe <subgraph> --view <view> --examples
potpie graph read --subgraph <subgraph> --view <view> [--query "..."] [--scope key:value] [--limit N]
potpie graph search-entities "text" [--type Service] [--predicate DEPENDS_ON] [--environment prod] [--limit N]
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

Use text output for routine orientation and context reads. Add `--json` when a
workflow needs exact machine parsing, mutation plans, commits, history
verification, or full evidence/debug payloads.

Express reads through `graph read --subgraph <subgraph> --view <view>`.

## Views

| View | Use it for |
|---|---|
| `decisions.preferences_for_scope` | project/repo/path preferences for code work |
| `infra_topology.service_neighborhood` | env-qualified dependencies and blast radius |
| `recent_changes.timeline` | project-wide PRs, tickets, docs, incidents, deployments |
| `debugging.prior_occurrences` | prior symptoms, fixes, failed attempts |
| `decisions.active_decisions` | active architectural/product decisions |
| `code_topology.ownership_by_path` | owners for a scope |
| `knowledge.document_context` | docs and runbooks for a scope |
| `admin.inspection_slice` | raw canonical graph for debugging |

## Writing

Two rules carry most of the value:

1. Resolve identity first with `graph search-entities` before linking to an
   existing service, repo, bug, decision, person, or document.
2. Write retrieval-grade descriptions. Include the symptom text, synonyms, scope,
   environment, service, files, commands, and source refs a future searcher would
   type.

Use semantic operations only and trust `graph catalog` for the current applicable,
review-required, and deferred operation partitions. Never hard-delete a claim; end
its validity, retract it, supersede it, or merge duplicates according to catalog
policy.

Create and review a plan before committing:

```bash
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

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

preference|policy|bug_pattern|fix|verification|decision|doc_reference|workflow|runbook_note|incident_summary|investigation|diagnostic_signal|service_note|feature_note|integration_note

## Ingestion Boundary

There is no scanner-driven graph write path in the agent instructions. For a
repo link, doc, ticket, PR, issue, or web link, the harness reads the source,
decides what durable facts exist, and writes graph mutations. Local repo
inspection with `rg`, `rg --files`, `git`, manifests, docs, routes, configs,
tests, and CI files is expected for repo understanding; the forbidden part is
blindly turning a tree walk into graph facts. Do not infer services,
dependencies, features, or preferences from directory names or package files
alone.

For explicit repository ingestion, use a todo-driven workflow:

1. Run pot/source/graph preflight (`pot info`, `source list`, `graph status`,
   `graph catalog --task`, and relevant `graph describe ... --examples`).
2. Create discovery todos for docs/product, local repo map, runtime/deploy,
   API/data/integrations, GitHub history, preferences/workflows, synthesis,
   write, and verification.
3. Use read-only subagents for independent discovery slices when available.
   Subagents return candidate facts, evidence, confidence, and uncertainty; they
   do not write graph mutations.
4. Build an evidence matrix, resolve identities, then write through
   `graph propose` / `graph commit --verify` / `graph history`.
5. Treat `graph commit --verify` as the post-write gate. If it warns or fails,
   drill down with affected reads and quality reports such as duplicate,
   low-confidence, and conflicting-claim checks.

For GitHub, Linear, Jira, and other hosted integrations, pull PRs, issues,
tickets, comments, labels/status, and linked docs with the agent's integration
tools/connectors. Do not use Potpie CLI queue ingestion as the ingestion path;
write the graph updates yourself with `graph propose` / `graph commit --verify`
or `graph inbox`.

## Responding To Nudges

A hook may inject context or instructions from `potpie graph nudge`.

- `inject_context` - use the injected facts for the current task.
- `instruction` - a prompt to decide whether something durable was learned. If it
  was, resolve identity, propose a graph plan, commit it with `--verify` when
  policy allows, then inspect history. If nothing durable was learned, do
  nothing.

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
- `potpie-repo-baseline` - repository purpose, services, environments, APIs,
  datastores, integrations, and durable project facts.
- `potpie-source-ingestion` - harness-led ingestion from repo links, docs, PRs,
  issues, tickets, runbooks, logs, and web links.
- `potpie-graph` - graph CLI contract: status/catalog/describe/read/search,
  propose/commit/history, inbox, quality, and nudge handling.
- `potpie-cli` - CLI setup, pot/source commands, graph commands, and
  troubleshooting, including pot scope and setup failures.
<!-- potpie-end -->
