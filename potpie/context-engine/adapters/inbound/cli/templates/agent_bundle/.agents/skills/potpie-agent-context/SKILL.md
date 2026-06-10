---
name: "potpie-context"
version: "1"
recommended: true
description: "Use when an agent needs project context from the context engine through the MCP context_* tools. Covers context_resolve recipes for feature, debugging, review, operations, docs, and onboarding without adding separate context tools. For the richer graph CLI surface (catalog/read/search-entities/mutate) and writing retrieval-grade descriptions, use potpie-graph."
---

# Potpie Agent Context

Use this skill when the task requires gathering, verifying, or recording project
context through the Potpie **MCP** tools. If you can run the `potpie` CLI, prefer the
`potpie-graph` skill (the V1.5 graph surface) — it is the same engine with richer
reads and direct mutations.

For use-case-specific behavior, load the relevant workflow skill too:
`potpie-project-preferences`, `potpie-infra-architecture`,
`potpie-change-timeline`, `potpie-debug-memory`, or `potpie-source-ingestion`.

## Tool Surface

Use only the minimal context port:

- `context_resolve` — primary task context wrap.
- `context_status` — cheap pot readiness, freshness gaps, and recommended recipe.
- `context_search` — narrow follow-up lookup after `context_resolve`.
- `context_record` — durable learnings: decisions, fixes, preferences, workflows,
  feature notes, doc references, incident summaries.

Do not request or invent separate tools for feature context, debugging context,
operations context, source lookup, or docs context. Express those through
`context_resolve` parameters. These four tools are compatibility wrappers over the
same graph that `potpie graph` reads and writes.

## Operating Loop

1. Resolve or ask for the active `pot_id`.
2. Build a narrow `scope` from known repo, branch, files, services, features,
   environment, PR, ticket, user, or source refs.
3. Call `context_status` when setup or readiness is uncertain or the task is broad.
4. Call `context_resolve` with the matching recipe.
5. Inspect `coverage`, `freshness`, `quality`, `fallbacks`, `open_conflicts`, and
   `source_refs` before relying on the answer.
6. Escalate only when needed:
   - `source_policy="summary"` for compact source-backed summaries.
   - `source_policy="verify"` and `mode="verify"` before production-impacting facts.
   - `source_policy="snippets"` or `mode="deep"` only for bounded source detail.
7. Use `context_record` when the work discovers reusable project memory.

## Valid Include Families

`coding_preferences`, `infra_topology`, `prior_bugs`, `timeline` (reader-backed),
plus `decisions`, `owners`, `docs` (advertised; surface as `unsupported_include`
until a reader backs them), and `raw_graph` (full canonical subgraph, for explorers).
Requesting any other name returns an honest `unsupported_include`.

## Recipes

Feature work:
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

Docs:
```json
{"intent":"docs","include":["docs","decisions"],"mode":"fast","source_policy":"references_only"}
```

Onboarding:
```json
{"intent":"onboarding","include":["infra_topology","coding_preferences","docs","owners"],"mode":"fast","source_policy":"references_only"}
```

## Query Expansion Is Your Job

Recall depends on the query text, and the local embedder is small. Expand the user's
words into a good retrieval query before you call: turn "add retry to the payments
client" into a query that also carries "timeout, flaky, tenacity, backoff,
external call". This expansion is in-session reasoning — it is not something the
daemon does for you.

## Recording Durable Memory

Use `context_record` with a `record_type` from:

bug_pattern|decision|diagnostic_signal|doc_reference|feature_note|fix|incident_summary|integration_note|investigation|policy|preference|runbook_note|service_note|verification|workflow

Always write the `summary`/`details` as a **retrieval card**: include the symptoms,
synonyms, and scope a future searcher would type, not just a display title. Keep
records compact and source-reference-first. `context_record` lowers through the same
semantic mutation path as `graph mutate`, so the same metadata rules apply.

Do not use `context_record` as a local code scanner. The harness must decide what
source material means and record only durable, retrieval-grade memory.

## Quality And Drift

- `quality.status=good` — graph context acceptable for low-risk orientation.
- `quality.status=watch` — verify stale or unverified facts before high-impact action.
- `quality.status=degraded` — prefer source truth; consider recording a correction.
- Follow `quality.recommended_maintenance` when the response suggests jobs such as
  `verify_entity`, `refresh_scope`, `resync_source_scope`, `repair_code_bridges`, or
  `expire_stale_facts`.
