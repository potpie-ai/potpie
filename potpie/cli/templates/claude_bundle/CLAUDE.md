<!-- potpie-start -->
# Context Engine

This project uses Potpie for project memory. Read the graph before non-trivial
work; record durable learnings after. The harness is the intelligence. Potpie
validates, lowers, commits, audits, and ranks graph memory. It does not scan the
repository or decide what prose means for you.

## Quick Start

Run one discovery pass shared by all relevant skills. Load applicable skills
together, then run independent reads concurrently:

- `potpie resolve "<task>"` for broad context.
- For code work, `potpie graph read --subgraph decisions --view preferences_for_scope --repo current --limit 12`, without `--query`.
- When the task explicitly names an entity whose canonical key is unknown,
  `potpie graph search-entities "<entity name>" --limit 10`, untyped.

Reuse current reads and hook context for the same task, pot, and scope. Skip
branches that do not apply. Use concurrent tool calls for short lookups; local
file discovery can run alongside them. Run `potpie status` concurrently if a
health check is needed. Use a known explicit pot selector on all calls; resolve
ambiguous routing before scoped retrieval and check returned pot IDs before
combining results. Once known, pass `--pot local:<name>` or `managed:<name>`.

Follow returned keys with a needed neighborhood/named view, and returned chunk
IDs with batched `resource get`. Run these follow-ups concurrently once inputs
are known. For a direct entity question, lookup then neighborhood may suffice;
for ordered history with known scope, the timeline can run immediately. Stop
expanding when the task's evidence and applicable constraints are covered.

## Graph Surface

Use the CLI when available:

```bash
potpie resolve "<task>"
potpie search "<known phrase>"
potpie record --type <fix|decision|preference> --summary "<…>" --detail <key>=<value> --scope service:<name>
potpie graph read --subgraph <subgraph> --view <view> [--query "..."] [--scope key:value] [--limit N]
potpie graph search-entities "text" [--limit N]
potpie graph mutation-template --kind <kind>
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
```

`graph mutation-template` is an unscoped, offline schema helper. It accepts
`--pot <origin>:<name>` for uniform command invocation, but ignores the selector
and does not resolve or validate it. Select the actual target on `graph propose`.

`resolve` supplies broad context (intent inferred, triples across families).
Before ingestion, read `potpie graph catalog --profile full` and the ontology
selection guidance in `potpie-graph`. Choose entity types and predicates from
the source meaning before choosing a writer: prescriptions are preferences;
implemented behavior, dependencies, decisions, and events have their own shapes.
Use `record` for a supported structured learning; topology and features use
semantic plans even for one fact. `feature_note` and other free-form notes do
not substitute for those typed relationships. Templates show payload shape,
not the full ontology. Reuse the discovered contract throughout the task.
Text output for reads; `--json` for parsed catalog data, `propose`, `commit`,
and `resource import`. `graph describe --examples` carries read examples only.
`commit --verify` checks persistence and quality; also read the intended graph
family to verify the selected representation.

## Report Back

Show the `potpie` commands behind the answer, verbatim — "I checked the graph"
cannot be re-run, widened, or corrected, and a visible command is how the reader
notices you read `staging` when they meant `prod`. Reads that returned nothing
get one summary line; an empty result is usually why the answer is thin.

Draw a mermaid diagram when the answer *is* a shape: `flowchart LR` for three or
more entities and their edges, `timeline` for an ordered run of deploys, PRs, or
incidents, `flowchart TD` for a symptom moving through attempts to a fix. Skip it
otherwise — one fact, a yes/no, or a list of preferences reads faster as a
sentence, and a diagram that restates one line is noise.

A diagram is a claim: draw only edges your reads support, keep environment
qualifiers on the nodes that carry them (`payments-api (prod)`), dash anything
inferred, and never add an edge just to make the picture connected.

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
- `potpie-resource-pdf` / `potpie-resource-spreadsheet` /
  `potpie-resource-markdown`: ingest document payloads as chunked, searchable
  document memory via `potpie resource import`; payloads never enter the graph.

## Writing

Reuse the keys your reads returned; resolve identity with `graph search-entities`
(untyped) before linking to a node no read has shown you. Write
retrieval-grade descriptions: include symptoms, synonyms, scope,
environment, service, source refs, and the words a future searcher would type.

Use semantic operations only: `upsert_entity`, `link_entities`, `assert_claim`,
`append_event`, `end_relation_validity`, `retract_claim`, and any audited
correction operation currently advertised by `graph catalog`. Never hard-delete a
claim. One fix, decision or preference is `potpie record`; a batch is a plan:
`graph propose`, then commit the returned `plan_id` with `--verify`.

## Ingestion Boundary

Do not run scanner-driven graph updates from the working tree. For a repo link,
document, ticket, PR, issue, or web link, the harness reads the source, decides
what durable facts exist, resolves identity, and writes graph plans through
`propose`/`commit`. Local repo inspection with `rg`, `rg --files`, `git`,
manifests, docs, routes, configs, tests, and CI files is expected for repo
understanding; the forbidden part is blindly turning a tree walk into graph
facts.

For explicit repository ingestion, use a todo-driven workflow: preflight
pot/source/graph state, create discovery todos, use read-only subagents for
independent docs/code/runtime/GitHub/preferences slices when available, build an
evidence matrix, resolve identities, take payload shapes from
`graph mutation-template`, propose graph mutations, commit with
`graph commit --verify`, then use affected reads and quality reports only when
the gate warns or fails.

For GitHub, Linear, Jira, and other hosted integrations, pull PRs, issues,
tickets, comments, labels/status, and linked docs with the agent's integration
tools/connectors. Do not use Potpie CLI queue ingestion as the ingestion path;
write the graph updates yourself with `graph propose` / `graph commit --verify`
or `graph inbox`.

## Nudges

A Potpie hook may inject context or an instruction. `inject_context` is task
context. `instruction` is a prompt to decide whether a durable learning should be
recorded through `potpie record` or `graph propose` and `graph commit --verify`;
if not, do nothing.

## Slash Commands

Use `/potpie-feature` before feature work and `/potpie-record` to capture
learnings.
<!-- potpie-end -->
