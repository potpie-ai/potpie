---
name: jira-project-diff-sync
description: Incremental catch-up ingestion for a Jira project by auditing existing context-graph coverage against source refs.
source_system: jira
event_type: jira_project
action: diff_sync
enables_planner: true
---

# Jira project diff sync

A reusable skill for checking how much of a Jira project's source history is
already represented in the context graph, then hydrating only missing or stale
graph coverage with the same stable keys as the one-shot Jira skill. Use this
after `jira-project-one-shot-ingestion` to repair partial graph coverage,
recover from missed writes, or reconcile an existing graph against source.

## Feasibility

Jira Cloud supports the source side needed for this graph audit. The REST API
exposes JQL issue search, so a connector can query `project = <KEY> AND
updated >= <cursor> ORDER BY updated ASC`. Jira Cloud also supports
`jira:issue_created`, `jira:issue_updated`, and `jira:issue_deleted` webhooks,
including JQL filters for issue events. When field-level deltas matter, Jira
Cloud provides issue changelog APIs, including bulk changelog fetch for
multiple issues. The skill still treats the context graph as the primary state
to audit; source timestamps are only comparison evidence.

## Inputs

- `project_key`: Jira project key (required, e.g. `PROJ`).
- `since`: optional ISO-8601 lower-bound for source enumeration. If omitted,
  read the most recent graph-audit cursor from the history file.
- `count`: soft list limit. Default `120`.
- `batch_size`: items per todo. Default `10`.
- `parallel_per_batch` (`K`): items to hydrate in parallel. Default `5`.
- `event_id`: required for the internal reconciliation agent. The single
  `(jira, jira_project, diff_sync)` event id for the run.

## History file

Maintain a durable history record for every run. The preferred path is the
agent run-history store if available; otherwise write a source-scoped history
file in the pot workspace named:

`context-sync-history/jira-project-<project-key-lowered>.jsonl`

Append one JSON line per run:

- `source_system`: `jira`
- `event_type`: `jira_project`
- `action`: `diff_sync`
- `project_key`
- `started_at`, `finished_at`
- `input_since`
- `previous_cursor`
- `new_cursor`
- `graph_checked`: object with counts for expected Activity keys searched
- `graph_missing`: list of issue keys whose Activity key was absent
- `graph_stale`: list of issue keys whose graph `source_updated_at` lagged
  the source `updated_at`
- `status`: `success | partial | failed`
- `processed`: object with counts for `epics`, `issues`
- `warnings`: list of strings
- `event_id`

Only advance `new_cursor` after the context graph has been checked and graph
mutations for all issues at or before that timestamp have succeeded. Use a
small overlap window when querying (`previous_cursor - 2 minutes`) and dedupe
by issue key + `updated_at` so equal-timestamp updates are not skipped.

## Tools

- `read_sync_history(source_system="jira", scope="jira_project", key=project_key)`
  — returns prior audit records oldest→newest plus `latest_cursor`. Call it
  first to recover `previous_cursor`.
- `write_sync_history(record)` — append-only; `source_system`, scope
  (`event_type`), and key (`project_key`) are taken from the record. Call once
  per run, after graph writes.
- `context_search(query=<activity-key>, node_labels=["Activity"], limit=1)` —
  current graph lookup by expected Activity key. NOTE: `context_search` is
  intent-routed (semantic), not an exact-key get, so treat its result as
  best-effort evidence — see "Audit precision" below. If a stricter
  `context_get_entity(entity_key)` tool exists, prefer it.
- `context_timeline(query=<jira issue key or summary>, limit=5)` — optional
  fallback when key lookup is unavailable or ambiguous.
- `jira_search_issues(jql, limit=count)` — compact issue refs from JQL,
  including `{key, summary, issuetype, updated_at}`.
- `jira_get_issue(issue_key)` — full issue payload. Works for epics and
  non-epic issues.
- `jira_get_issue_changelog(issue_key)` or
  `jira_bulk_fetch_changelogs(issue_keys=[...])` — optional, only when the
  update timestamp changed but the hydrated issue does not explain what
  changed.
- `apply_graph_mutations(plan, event_id, summary)`.
- Planner / todo tools (`read_todos`, `write_todos`, `update_todo_status`).
- `mark_event_processed(event_id, summary)` + `finish_batch(summary)`.

## Procedure

### Phase 0 - Load graph-audit cursor

1. Read the history file or run-history store for this project.
2. Resolve `previous_cursor` from the most recent successful graph audit
   unless `since` is provided.
3. Set `query_since = previous_cursor - 2 minutes` when a previous cursor
   exists. If there is no cursor, require `since`; otherwise abort and tell
   the caller to run `jira-project-one-shot-ingestion` first so there is a
   baseline graph to audit.
4. Initialize todos for reading history and enumerating changed issues.
   Continue an existing todo list on resume.

### Phase 1 - Enumerate candidate source refs

1. Call `jira_search_issues` once with:
   `project = <project_key> AND updated >= <query_since> ORDER BY updated ASC`
2. Drop refs whose `(issue key, updated_at)` were already recorded as
   successfully graph-checked in the overlap window.
3. For each remaining ref, compute the expected one-shot Activity key:
   - epic: `activity:jira:epic:<issue-key-lowered>`
   - issue: `activity:jira:issue:<issue-key-lowered>`
4. Track the maximum `updated_at` seen as the candidate cursor. Do not commit
   it yet.

### Phase 2 - Audit current context graph

For each expected Activity key, query the current context graph before
hydrating source details:

1. If the Activity key is absent, append `Process missing jira <kind> <key>`.
2. If the Activity exists but its properties lack `source_updated_at`, warn
   and treat it as stale so the next write backfills that comparison field.
3. If `source_updated_at < source updated_at`, append
   `Process stale jira <kind> <key>`.
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

Hydrate changed items exactly like `jira-project-one-shot-ingestion` and reuse
its mutation rules, entity keys, no-Fix-from-issue rule, and current ontology
edge guidance. Build one `LlmReconciliationPlan` per batch and call
`apply_graph_mutations(plan, event_id, summary)`.

Every Activity upsert emitted by this skill must include
`source_updated_at=<Jira updated_at>` so future graph audits can compare
source state with graph state without rehydrating unchanged items.

Use changelog tools only when the issue-level payload is insufficient to
explain the update. Comments are context, not standalone facts.

For deleted or inaccessible issues, do not delete graph history by default.
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

- Do not skip the context-graph audit and rely on source `updated_at` alone.
- Do not advance the cursor before graph checks and graph writes succeed.
- Do not emit `Fix` nodes from Jira issues. Fix remains reserved for merged
  PRs / commits that shipped code.
- Do not use changelog noise to invent Decisions; require explicit rationale.
- Do not overwrite or compact the history file. It is append-only audit data.
