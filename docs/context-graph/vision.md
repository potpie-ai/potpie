# Context Engine — Vision

The Context Engine is the project-context layer that every agent in Potpie reads from and writes back into. One pot, one graph, one story.

## The problem

Every agent today rebuilds project context from scratch from raw code, PRs, tickets, threads, and dashboards. That work is duplicated, lossy, and stateless — the next agent doing related work starts from zero. Decisions, prior fixes, ownership, runtime context, and the *intent* behind changes are scattered across systems that none of those agents can query coherently.

The Context Engine exists to hold that context once, keep it fresh, and expose it through a small, stable surface so agents can align to the same understanding of the project.

## What it is

A single project-context graph per **pot**, fed by deterministic source connectors and a reconciliation agent, queried by agents through a four-tool port.

- **Pot** — the tenant boundary. One pot spans the whole project: every repo, service, doc source, ticketing system, ops integration, user, and agent that participates in it. Agents start from the active pot and narrow by repo / feature / service / file / PR / incident / environment / user.
- **Graph** — one canonical graph of the entities that matter at the project level: features, services, decisions, changes, incidents, alerts, runbooks, owners, preferences, agent instructions, and the source references that back them.
- **Agent surface** — four tools, intentionally narrow (see "The four-tool port" below).

What the Context Engine is *not*: a re-implementation of the code graph, a wrapper over GitHub's API, or a place to dump full PR diffs and document bodies.

## The pot model

A pot is the unit of isolation and the unit of context. Every fact, every edge, every ingestion event, every query result lives inside exactly one pot. Cross-pot federation is explicitly out of scope.

Inside a pot, queries can be scoped further (repo, service, environment, feature, file, PR, ticket, user, time window). Outside a pot, nothing leaks.

## Source-reference-first storage

The graph holds compact canonical facts and references back to source systems — not the full payloads.

- **What goes in:** normalized entities (PR-12345, Feature "checkout v2", Decision "rate-limit middleware"), edges between them, compact summaries, provenance, freshness, lifecycle status.
- **What does not go in:** full PR diffs, full document bodies, full conversation transcripts, verbose logs, raw incident payloads, telemetry streams.
- **Why:** source systems are the authoritative copy. Storing payloads in the graph creates a second source of truth that drifts immediately. References + summaries stay correct because the resolver fetches the source on demand when an agent actually needs the body.

The exception is small snippets that materially improve recall or offline use (a PR title, a key paragraph from a doc). Anything larger goes through the source resolver at query time.

## The four-tool port

Agents see four tools and only four:

- `context_resolve` — primary entry point. Pulls together the evidence families an intent needs and returns a structured envelope (answer, evidence, source refs, coverage, freshness, quality, fallbacks, conflicts, recommended next actions).
- `context_search` — narrow, targeted lookup when the agent already knows what shape it wants.
- `context_record` — durable writes (decisions, fixes, preferences, workflows) routed through the same reconciliation pipeline as ingested events.
- `context_status` — health and capability check. Reports pot readiness, registered sources, freshness, and the recipe an agent should reach for given an intent.

The contracts for these tools are in [`agent-contract.md`](./agent-contract.md).

### Why narrow

New use cases become *parameters* on `context_resolve` (`intent`, `include`, `scope`, `mode`, `source_policy`), not new tools. Adding `context_get_feature_context`, `context_get_debugging_context`, `context_get_operational_context` would create exactly the tool sprawl the engine exists to prevent. Skill recipes (defined in agent bundles) compose the four tools into workflows; they do not appear at the tool-port boundary.

## Substrate choice

Graphiti sits underneath as the episodic + temporal + hybrid-search substrate on Neo4j. Above it, Potpie owns the canonical ontology layer: stable entity keys, label and edge validation, deterministic upserts, conflict and supersession handling, exact-read helpers.

The substrate is a deliberate choice, not a permanent commitment. It is right while we're earning the schema and the access patterns. If telemetry later shows it isn't pulling its weight, the boundary above it (the `ContextGraphPort`) is what makes a substrate swap possible.

## Ontology is data, not control flow

The canonical ontology is expected to evolve as we learn what agents actually need to answer. The engine treats it as **declarative data, not branching code.** Every entity and edge spec in [`domain/ontology.py`](../../app/src/context-engine/domain/ontology.py) carries the metadata its downstream consumers need — project-map family, fact family, source-of-truth, freshness TTL, classifier text cues, property signatures, endpoint inference, predicate family. The classifier, structural reader, hybrid graph, graph-quality policy, and query helpers all derive their tables from the spec at import time.

The consequence: adding, renaming, or removing an entity or edge is a single-file edit. If you ever find yourself touching `structural.py`, `hybrid_graph.py`, or `ontology_classifier.py` to teach them about a label, the spec is missing a field — that is the bug to fix.

The agent-facing contract (`include` keys, family names, response shapes) is decoupled from the internal entity labels. A label can be renamed without breaking the agent contract; an include key remaps via its `include_keys` field.

## Anti-goals

These are not under-investments. They are decisions:

- **No second graph store, no parallel graph abstraction.** One substrate.
- **No new public agent tools** beyond the four. Use parameters.
- **No source-specific code in the application layer.** GitHub, Linear, Slack, Notion live behind a connector contract.
- **No full source payloads in the graph.** References plus compact summaries.
- **No "compatibility" code paths** when something is replaced. The old path goes in the same change as the new one.
- **No cross-pot federation.** One pot, one tenant, one graph.
- **No frontend in the Context Engine.** The UI is a consumer of the agent contract, not part of it.

## The long-term direction

Every change to the graph is a validated `ReconciliationPlan`. Every read is a routed call into a registered `ContextReader`. Every source is a `SourceConnector` plugin with the same contract. Every agent answer carries enough provenance and freshness for the agent to decide whether to trust it or ask for verification.

The execution path to that end-state is in [`plan.md`](./plan.md). The current state is in [`architecture.md`](./architecture.md).
