<!-- potpie-start -->
# Context Engine

This project uses Potpie for project memory. Read the graph before non-trivial
work; record durable learnings after. The harness is the intelligence. Potpie
validates, lowers, commits, audits, and ranks graph memory. It does not scan the
repository or decide what prose means for you.

## Quick Start

```bash
potpie doctor
potpie pot list
potpie graph catalog
```

## Graph Surface

Use the CLI when available:

```bash
potpie --json graph catalog
potpie --json graph read --view <subgraph.view> [--query "..."] [--scope key:value] [--limit N]
potpie --json timeline recent [--time-window 7d] [--limit N]
potpie --json graph search-entities "text" [--type Service] [--environment prod]
potpie --json graph mutate --file mutation.json [--dry-run]
```

When only MCP is configured, use `context_status`, `context_resolve`,
`context_search`, and `context_record`. Valid include families:
`coding_preferences`, `infra_topology`, `prior_bugs`, `timeline`, `decisions`,
`owners`, `docs`, `raw_graph`.

Feature/code work:

```json
{"intent":"feature","include":["coding_preferences","infra_topology","decisions","owners","docs"],"mode":"fast","source_policy":"references_only"}
```

Debugging:

```json
{"intent":"debugging","include":["prior_bugs","infra_topology","timeline"],"mode":"fast","source_policy":"references_only"}
```

Operations:

```json
{"intent":"operations","include":["infra_topology","timeline","owners"],"mode":"balanced","source_policy":"summary"}
```

## Use-Case Skills

- `potpie-project-preferences`: coding preferences for error handling, structure,
  libraries, frameworks, logging, tests, and style.
- `potpie-infra-architecture`: environments, adapters, deployments, service
  dependencies, datastores, API contracts, and ownership.
- `potpie-change-timeline`: PRs, tickets, docs, incidents, deployments, and
  regression correlation.
- `potpie-debug-memory`: prior bugs, fixes, failed attempts, verification, and dev
  setup troubleshooting.
- `potpie-source-ingestion`: harness-led ingestion from repo links, docs, PRs,
  issues, tickets, logs, runbooks, and web links.

## Writing

Resolve identity with `graph search-entities` before linking to existing nodes.
Write retrieval-grade descriptions: include symptoms, synonyms, scope,
environment, service, source refs, and the words a future searcher would type.

Use semantic operations only: `upsert_entity`, `link_entities`, `assert_claim`,
`append_event`, `end_relation_validity`, and `retract_claim`. Never hard-delete a
claim.

## Ingestion Boundary

Do not run local code scans or deterministic graph updates from the working tree.
For a repo link, document, ticket, PR, issue, or web link, the harness reads the
source, decides what durable facts exist, resolves identity, and writes graph
mutations or `context_record`.

## Nudges

A Potpie hook may inject context or an instruction. `inject_context` is task
context. `instruction` is a prompt to decide whether a durable learning should be
recorded; if not, do nothing.

## Slash Commands

Use `/potpie-feature` before feature work and `/potpie-record` to capture
learnings.
<!-- potpie-end -->
