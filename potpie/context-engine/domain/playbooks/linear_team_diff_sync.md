---
name: linear-team-diff-sync
description: Incremental catch-up ingestion for a Linear team by auditing existing context-graph coverage against source refs.
source_system: linear
event_type: linear_team
action: diff_sync
enables_planner: true
---

# Linear team diff sync

A reusable skill for checking how much of a Linear team's source history is
already represented in the context graph, then hydrating only missing or stale
graph coverage with the same stable keys as the one-shot Linear skill. Use
this after `linear-team-one-shot-ingestion` to repair partial graph coverage,
recover from missed writes, or reconcile an existing graph against source.

## Feasibility

Linear supports the source side needed for this graph audit. Its public API is
GraphQL, list responses are cursor-paginated, and issue lists can be ordered by
`updatedAt`. Linear also provides webhooks for create / update / remove events
on Issues, Comments, Issue attachments, Documents, Projects, Project updates,
and related resource types. The skill still treats the context graph as the
primary state to audit; source timestamps are only comparison evidence.

## Inputs

- `team`: Linear team id or key (required).
- `since`: optional ISO-8601 lower-bound for source enumeration. If omitted,
  read the most recent graph-audit cursor from the history file.
- `count`: soft per-kind list limit. Default `120`.
- `batch_size`: items per todo. Default `10`.
- `parallel_per_batch` (`K`): items to hydrate in parallel. Default `5`.
- `event_id`: required for the internal reconciliation agent. The single
  `(linear, linear_team, diff_sync)` event id for the run.

## History file

Maintain a durable history record for every run. The preferred path is the
agent run-history store if available; otherwise write a source-scoped history
file in the pot workspace named:

`context-sync-history/linear-team-<team-slug>.jsonl`

Append one JSON line per run:

- `source_system`: `linear`
- `event_type`: `linear_team`
- `action`: `diff_sync`
- `team`
- `started_at`, `finished_at`
- `input_since`
- `previous_cursor`
- `new_cursor`
- `graph_checked`: object with counts for expected Activity keys searched
- `graph_missing`: list of source refs whose Activity key was absent
- `graph_stale`: list of source refs whose graph `source_updated_at` lagged
  the source `updated_at`
- `status`: `success | partial | failed`
- `processed`: object with counts for `projects`, `documents`, `issues`
- `warnings`: list of strings
- `event_id`

Only advance `new_cursor` after the context graph has been checked and graph
mutations for all missing/stale items at or before that timestamp have
succeeded. Use a small overlap window when querying (`previous_cursor - 2
minutes`) and dedupe by source id + `updated_at` so equal-timestamp updates
are not skipped.

## Tools

- `read_sync_history(source_system="linear", scope="linear_team", key=team)` —
  returns prior audit records oldest→newest plus `latest_cursor`. Call it
  first to recover `previous_cursor`.
- `write_sync_history(record)` — append-only; `source_system`, scope
  (`event_type`), and key (`team`) are taken from the record. Call once per
  run, after graph writes.
- `context_search(query=<activity-key>, node_labels=["Activity"], limit=1)` —
  current graph lookup by expected Activity key. NOTE: `context_search` is
  intent-routed (semantic), not an exact-key get, so treat its result as
  best-effort evidence — see "Audit precision" below. If a stricter
  `context_get_entity(entity_key)` tool exists, prefer it.
- `context_timeline(query=<linear identifier or title>, limit=5)` — optional
  fallback when key lookup is unavailable or ambiguous.
- `linear_list_projects(team_id=team, updated_since=query_since, limit=count)`
  — compact refs `{id, name, updated_at}`, newest-first or updatedAt order.
- `linear_list_documents(team_id=team, updated_since=query_since, limit=count)`
  — compact refs `{id, title, updated_at}`. If unavailable, warn and
  continue.
- `linear_list_issues(team_id=team, updated_since=query_since, limit=count)`
  — compact refs `{id, identifier, updated_at}`.
- `linear_get_project(project_id)`, `linear_get_document(document_id)`,
  `linear_get_issue(issue_id)` — hydrate one item.
- `apply_graph_mutations(plan, event_id, summary)`.
- Planner / todo tools (`read_todos`, `write_todos`, `update_todo_status`).
- `mark_event_processed(event_id, summary)` + `finish_batch(summary)`.

## Procedure

### Phase 0 - Load graph-audit cursor

1. Read the history file or run-history store for this team.
2. Resolve `previous_cursor` from the most recent successful graph audit
   unless `since` is provided.
3. Set `query_since = previous_cursor - 2 minutes` when a previous cursor
   exists. If there is no cursor, require `since`; otherwise abort and tell
   the caller to run `linear-team-one-shot-ingestion` first so there is a
   baseline graph to audit.
4. Initialize todos for reading history and enumerating projects, documents,
   and issues. Continue an existing todo list on resume.

### Phase 1 - Enumerate candidate source refs

1. Call each list tool once with `updated_since=query_since` and `limit=count`.
2. Drop refs whose `(source id, updated_at)` were already recorded as
   successfully graph-checked in the overlap window.
3. For each remaining ref, compute the expected one-shot Activity key:
   - project: `activity:linear:project:<uuid-lowered>`
   - document: `activity:linear:document:<uuid-lowered>`
   - issue: `activity:linear:issue:<identifier-lowered>`
4. Track the maximum `updated_at` seen across all returned refs as the
   candidate cursor. Do not commit it yet.

### Phase 2 - Audit current context graph

For each expected Activity key, query the current context graph before
hydrating source details:

1. If the Activity key is absent, append
   `Process missing linear <kind> <id-or-identifier>`.
2. If the Activity exists but its properties lack `source_updated_at`, warn
   and treat it as stale so the next write backfills that comparison field.
3. If `source_updated_at < source updated_at`, append
   `Process stale linear <kind> <id-or-identifier>`.
4. If `source_updated_at >= source updated_at`, record the item as graph
   current and do not hydrate it.

The goal is to measure context-graph coverage first. Source updates only
matter when the graph is missing the stable Activity key or carries older
source evidence.

#### Audit precision

`context_search` routes by intent and is not a deterministic key lookup, so a
match is evidence, not proof. Resolve ambiguity in the safe direction: when you
cannot confidently confirm the exact Activity key is present and current, treat
the ref as **missing/stale and hydrate it**. Re-hydration is safe — the
one-shot mutations are idempotent on the stable entity keys, so a needless
re-ingest converges instead of duplicating. The only cost of a wrong "present"
call is an unrepaired gap that the next run still catches; the only cost of a
wrong "missing" call is a cheap idempotent re-write. Never skip a ref just
because a fuzzy search returned something that looked close.

### Phase 3 - Hydrate and write graph gaps

Hydrate changed items exactly like `linear-team-one-shot-ingestion` and reuse
its mutation rules, entity keys, no-Fix-from-issue rule, and ontology edge
guidance. Build one `LlmReconciliationPlan` per batch and call
`apply_graph_mutations(plan, event_id, summary)`.

Every Activity upsert emitted by this skill must include
`source_updated_at=<Linear updated_at>` so future graph audits can compare
source state with graph state without rehydrating unchanged items.

For deleted or inaccessible refs, do not delete graph history by default.
Record a warning and, if the connector explicitly returns a tombstone, emit an
invalidation for volatile fields only.

### Phase 4 - Commit history

1. If every graph audit and missing/stale todo succeeded, append a `success`
   record and set `new_cursor` to the candidate cursor.
2. If some items failed after at least one successful batch, append `partial`
   and keep `new_cursor` at the last fully applied batch timestamp.
3. If no graph mutations succeeded, append `failed` and leave the cursor
   unchanged.
4. Mark the single diff-sync event processed and finish the batch.

## Anti-patterns

- Do not page forever. Respect the bounded list results and ask the caller to
  rerun if the response indicates more changes remain.
- Do not skip the context-graph audit and rely on source `updated_at` alone.
- Do not advance the cursor before graph checks and graph writes succeed.
- Do not emit `Fix` nodes from Linear issues. Fix remains reserved for merged
  PRs / commits that shipped code.
- Do not overwrite or compact the history file. It is append-only audit data.
