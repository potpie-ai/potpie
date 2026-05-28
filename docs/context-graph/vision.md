# Context Graph — Vision

The Context Engine is the project-context layer that agents read from and
write back into. It must work as both a local open-source service and a
managed Potpie service. One pot, one graph model, one story.

## The problem

Every agent today rebuilds project context from scratch from raw code, PRs, tickets, threads, and dashboards. That work is duplicated, lossy, and stateless — the next agent doing related work starts from zero. Decisions, prior fixes, ownership, runtime context, and the *intent* behind changes are scattered across systems that none of those agents can query coherently.

The Context Engine exists to hold that context once, keep it fresh, and expose it through a small, stable surface so agents can align to the same understanding of the project.

## What it is

A single project-context graph per **pot**, queried by agents through a
four-tool port and backed by the same canonical graph model in every
deployment shape.

- **Pot** — the tenant boundary. One pot spans the whole project: every repo, service, doc source, ticketing system, ops integration, user, and agent that participates in it. Agents start from the active pot and narrow by repo / feature / service / file / PR / incident / environment / user.
- **Graph** — one canonical graph of the entities that matter at the project level: features, services, decisions, changes, incidents, alerts, runbooks, owners, preferences, agent instructions, and the source references that back them.
- **Agent surface** — four tools, intentionally narrow (see "The four-tool port" below).

There are now two supported distribution shapes:

- **Local self-serve / open source** — installed with the Potpie CLI, backed by
  a local daemon and local graph/state store. The agent harness owns source
  access and ingestion reasoning; the daemon stores and serves graph context.
- **Managed Potpie** — hosted API, shared pots, hosted database/graph,
  workers, cloud source integrations, and collaboration controls. This is an
  adapter/deployment of the same core engine, not a separate graph product.

What the Context Engine is *not*: a re-implementation of the code graph, a wrapper over GitHub's API, or a place to dump full PR diffs and document bodies.

The local-first roadmap and managed boundary are in
[`architecture.md`](./architecture.md).

## The pot model

A pot is the unit of isolation and the unit of context. Every fact, every edge, every ingestion event, every query result lives inside exactly one pot. Cross-pot federation is explicitly out of scope.

Inside a pot, queries can be scoped further (repo, service, environment, feature, file, PR, ticket, user, time window). Outside a pot, nothing leaks.

In local mode, a pot is a user-local workspace boundary. In managed mode, a pot
is also the authorization and collaboration boundary. The graph model is the
same in both cases; only the state, auth, and deployment adapters differ.

## Source-reference-first storage

The graph holds compact canonical facts and references back to source systems — not the full payloads.

- **What goes in:** normalized entities (PR-12345, Feature "checkout v2", Decision "rate-limit middleware"), edges between them, compact summaries, provenance, freshness, lifecycle status.
- **What does not go in:** full PR diffs, full document bodies, full conversation transcripts, verbose logs, raw incident payloads, telemetry streams.
- **Why:** source systems are the authoritative copy. Storing payloads in the graph creates a second source of truth that drifts immediately. References + summaries stay correct because the resolver fetches the source on demand when an agent actually needs the body.

The exception is small snippets that materially improve recall or offline use
(a PR title, a key paragraph from a doc). Anything larger should be fetched by
the agent harness, managed source adapter, or cloud source resolver when the
caller explicitly needs the body.

## The four-tool port

Agents see four tools and only four:

- `context_resolve` — primary entry point. Pulls together the evidence families an intent needs and returns a structured envelope (answer, evidence, source refs, coverage, freshness, quality, fallbacks, conflicts, recommended next actions).
- `context_search` — narrow, targeted lookup when the agent already knows what shape it wants.
- `context_record` — durable writes (decisions, fixes, preferences,
  workflows). Local mode can lower structured records directly to claims;
  managed mode can route them through the reconciliation pipeline when needed.
- `context_status` — health and capability check. Reports pot readiness, registered sources, freshness, and the recipe an agent should reach for given an intent.

The contracts for these tools are in [`agent-contract.md`](./agent-contract.md).

### Why narrow

New use cases become *parameters* on `context_resolve` (`intent`, `include`, `scope`, `mode`, `source_policy`), not new tools. Adding `context_get_feature_context`, `context_get_debugging_context`, `context_get_operational_context` would create exactly the tool sprawl the engine exists to prevent. Skill recipes (defined in agent bundles) compose the four tools into workflows; they do not appear at the tool-port boundary.

## Substrate choice

The durable invariant is the Potpie graph model: deterministic entity keys,
canonical claim edges, provenance, source refs, bitemporality, validation, and
one read/write graph port. The physical store is an adapter decision behind
`ContextGraphPort`, `GraphWriterPort`, and `ClaimQueryPort`.

Managed Potpie can use Neo4j for native graph traversal. The local open-source
daemon may use a lighter local store if that is what makes `pip install
potpie` viable. This is an intentional shift from "one physical graph store" to
"one graph model and one graph API."

## Ontology is data, not control flow

The canonical ontology is expected to evolve as we learn what agents actually need to answer. The engine treats it as **declarative data, not branching code.** Every entity, edge, and record spec in [`domain/ontology.py`](../../app/src/context-engine/domain/ontology.py) carries the metadata downstream consumers need: identity, source-of-truth family, freshness TTL, endpoint rules, predicate family, and reader include mapping.

The consequence: adding, renaming, or removing graph vocabulary should start in the ontology. If storage adapters, readers, or validators need one-off label tables, the ontology is probably missing metadata.

The agent-facing contract (`include` keys, family names, response shapes) is decoupled from the internal entity labels. A label can be renamed without breaking the agent contract; an include key remaps via its `include_keys` field.

## Anti-goals

These are not under-investments. They are decisions:

- **No parallel graph model or parallel agent contract.** Storage adapters may
  differ between local and managed deployments, but the graph shape, ports,
  ontology, and envelope must not.
- **No new public agent tools** beyond the four. Use parameters.
- **No source-specific code in the application layer.** Managed integrations
  live behind connector or event-ledger contracts; local source access lives in
  the agent harness and skills.
- **No full source payloads in the graph.** References plus compact summaries.
- **No hidden cloud dependency for local graph use.** Local CLI/MCP must work
  without a Potpie API key. Cloud login is only for optional managed features.
- **No daemon-side LLM dependency by default in local mode.** Agent-mediated
  structured writes are the default; raw-event reconciliation can be optional.
- **No cross-pot federation.** One pot, one tenant, one graph.
- **No frontend in the Context Engine.** The UI is a consumer of the agent contract, not part of it.

## The long-term direction

Every graph write is a validated structured record, scanner result, or
reconciliation mutation. Every read is routed through the read orchestrator into
a registered reader. Every cloud source and webhook path is an adapter around
the same graph model. Every agent answer carries enough provenance, coverage,
and freshness for the agent to decide whether to trust it or verify the source.

The local-first execution path, managed cloud boundary, and current
implementation map are in [`architecture.md`](./architecture.md).
