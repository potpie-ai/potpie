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
potpie graph status
potpie graph catalog --task "<task>"
```

## Graph Surface

Use the CLI when available:

```bash
potpie graph status
potpie graph catalog --task "<task>" --profile read
potpie graph describe <subgraph> --view <view> --examples
potpie graph read --subgraph <subgraph> --view <view> [--query "..."] [--scope key:value] [--limit N]
potpie graph search-entities "text" [--type Service] [--environment prod]
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

Use text output for routine orientation and context reads. Add `--json` when a
workflow needs exact machine parsing, mutation plans, commits, history
verification, or full evidence/debug payloads.

When only MCP is configured, use `context_status`, `context_resolve`,
`context_search`, and `context_record`. Valid include families:
`coding_preferences`, `infra_topology`, `prior_bugs`, `timeline`, `decisions`,
`owners`, `docs`, `raw_graph`.

Valid `context_record` types:
preference|policy|bug_pattern|fix|verification|decision|doc_reference|workflow|runbook_note|incident_summary|investigation|diagnostic_signal|service_note|feature_note|integration_note

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
`append_event`, `end_relation_validity`, `retract_claim`, and any audited
correction operation currently advertised by `graph catalog`. Never hard-delete a
claim. Create a plan with `graph propose`, commit the returned `plan_id` with
`--verify`, and inspect `graph history`.

## Ingestion Boundary

Do not run scanner-driven graph updates from the working tree. For a repo link,
document, ticket, PR, issue, or web link, the harness reads the source, decides
what durable facts exist, resolves identity, and writes graph plans through
`propose`/`commit`. Local repo inspection with `rg`, `rg --files`, `git`,
manifests, docs, routes, configs, tests, and CI files is expected for repo
understanding; the forbidden part is blindly turning a tree walk into graph
facts. If only MCP is configured, `context_record` is the compatibility fallback.

For explicit repository ingestion, use a todo-driven workflow: preflight
pot/source/graph state, create discovery todos, use read-only subagents for
independent docs/code/runtime/GitHub/preferences slices when available, build an
evidence matrix, resolve identities, propose graph mutations, commit with
`graph commit --verify`, then use affected reads and quality reports only when
the gate warns or fails.

For GitHub, Linear, Jira, and other hosted integrations, pull PRs, issues,
tickets, comments, labels/status, and linked docs with the agent's integration
tools/connectors. Do not use pot-level connector ingestion commands such as
`potpie pot linear-team ingest`, `potpie pot linear-team diff-sync`, or
Jira/GitHub queue commands as the ingestion path; write the graph updates
yourself with `graph propose` / `graph commit --verify` or `graph inbox`.

## Nudges

A Potpie hook may inject context or an instruction. `inject_context` is task
context. `instruction` is a prompt to decide whether a durable learning should be
recorded through `graph propose` and `graph commit --verify`; if not, do nothing.

## Slash Commands

Use `/potpie-feature` before feature work and `/potpie-record` to capture
learnings.
<!-- potpie-end -->
