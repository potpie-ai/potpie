---
name: linear-team-one-shot-ingestion
description: One-time ingestion of a Linear team's recent projects, documents, and issues into the context graph. Not incremental — live updates go through the live Linear webhook path.
source_system: linear
event_type: linear_team
action: one_shot_ingest
enables_planner: true
---

# Linear team one-shot ingestion

A reusable skill for ingesting a Linear team's recent projects, documents,
and issues into the context graph in a single pass. Sibling to the GitHub
`repo_one_shot_ingestion` playbook: same shape, different source. Designed to
be invoked by either Claude Code (as a checklist with a compatible write
path) or the internal reconciliation agent (loaded as a playbook).

## When to invoke

- A user wants to seed the context graph from a Linear team's recent history
  in one pass.
- The Linear team is already connected to the target pot (so connector tools
  are scoped to the right credentials).
- You will NOT run this skill repeatedly against the same team — incremental
  updates flow through live Linear webhook events, which write the same
  Activity keys so a future webhook converges with what this skill already
  wrote.

## Inputs

- `team`: Linear team id or key (required). Must be connected to the active pot.
- `count`: soft per-kind list limit. Default `120` (chosen so the soft
  tool-call cap below stays under the playbook's `max_tool_calls=400`).
  Read `count` from `event.payload.count` and pass it as `limit` on
  each list tool. Hard ceiling: respect whatever each bounded list
  tool returns — do not page past it.
- `batch_size`: items per todo. Default `10`.
- `parallel_per_batch` (`K`): items to hydrate in parallel per batch. Default
  `5`. Drop lower if hydrated items prove unusually large (long bodies or
  many comments).
- `event_id`: required for the internal reconciliation agent. The single
  `(linear, linear_team, one_shot_ingest)` event id for the run.

## Tools assumed available

- `linear_list_projects(team_id=team, limit=count)` — bounded enumeration of
  project refs `{id, name, updated_at}`, newest-first. ONE call.
- `linear_list_documents(team_id=team, limit=count)` — bounded enumeration of
  document refs
  `{id, title, updated_at}`. ONE call. May be unavailable on workspaces
  without docs API access — surface a warning and continue with the other
  kinds.
- `linear_list_issues(team_id=team, limit=count)` — bounded enumeration of issue refs
  `{id, identifier, updated_at}`. ONE call.
- `linear_get_project(project_id)` — name, description, state, lead,
  dates, teams.
- `linear_get_document(document_id)` — title, content, url, project, creator.
- `linear_get_issue(issue_id)` — full issue payload (title, body, labels,
  state, assignee, creator, project, dates, linked comments / linked PRs
  if present).
- `apply_graph_mutations(plan, event_id, summary)` — context-graph write.
  The `plan` argument MUST be an object with this shape:
  - `summary`: string.
  - `entity_upserts`: list of `{entity_key, labels, properties}`.
  - `edge_upserts`: list of `{edge_type, from_entity_key, to_entity_key, properties}`.
  - `edge_deletes`: usually `[]`.
  - `invalidations`: usually `[]`.
  - `evidence`: list of `{kind, ref, metadata}`.
  - `confidence`: optional number.
  - `warnings`: list of strings.
- Planner / todo tools (`read_todos`, `write_todos`, `update_todo_status`) —
  REQUIRED. The todo list rides in the agent's message history and is
  checkpointed; a resumed run continues the existing list instead of
  re-enumerating.
- `mark_event_processed(event_id, summary)` + `finish_batch(summary)` —
  completion.

## Procedure

### Phase 0 — Setup

1. Trust the event payload that the team is connected; there is no
   `sandbox_list_repos` equivalent for Linear. If a list tool returns an
   auth / not-connected error, abort with a warning. Do NOT attempt to
   connect the team from this skill.
2. Initialize the todo list with three entries:
   - `Enumerate <team> projects`
   - `Enumerate <team> documents`
   - `Enumerate <team> issues`

### Phase 1 — Enumerate (three list calls, one each)

1. Call `linear_list_projects(team_id=team, limit=count)` ONCE. Bounded
   server-side.
2. Call `linear_list_documents(team_id=team, limit=count)` ONCE. If this
   errors (workspace has no documents API), record a warning and continue
   with the other two kinds. Do NOT fabricate documents.
3. Call `linear_list_issues(team_id=team, limit=count)` ONCE. Bounded
   server-side.
4. Drain order: projects first (they frame what the issues are about),
   documents next (they often justify Decisions), issues newest-first
   (the timeline spine).
5. For each returned ref, append a todo: `Process linear <kind> <id-or-identifier>`.
6. Use `update_todo_status` or `write_todos` to mark each enumeration todo done.

### Phase 2 — Drain batches

Drain todos sequentially across kinds (projects, then documents, then issues);
within a batch parallelize up to `K`.

1. Choose `K` (`parallel_per_batch`, default 5). If hydrated payloads are
   large or slow, reduce `K` for the remaining items in this batch.
2. In parallel for `K` items at a time, hydrate via the matching `get_*` tool:

   **PROJECT** — `linear_get_project(id)`:
   - Read signals in PRIORITY ORDER, stopping when intent is clear:
     1. **Project name + state** — name carries intent; state (planned /
        started / paused / completed / canceled) carries lifecycle.
     2. **Project lead** — single Person reference; the natural PERFORMED
        actor for the creation activity.
     3. **Project description** — author-stated goal / scope.
     4. **Linked issues count** (if returned) — scale signal.
   - Classify: name + description summary, state, lead handle.

   **DOCUMENT** — `linear_get_document(id)`:
   - Read signals in PRIORITY ORDER:
     1. **Title** — author-stated headline (Spec / PRD / RFC / Runbook
        often prefixed).
     2. **Creator** — single Person reference; the actor on the document
        Activity.
     3. **Project association** (if returned) — anchor to a project.
     4. **Content** (body) — only enough to derive a 1-2 sentence summary
        and to spot rationale + alternatives_rejected if this doc justifies
        a Decision (specs / PRDs / RFCs typically do; runbooks / how-tos
        typically do not).

   **ISSUE** — `linear_get_issue(issue_id)` (pass the Linear identifier
   such as `ENG-123` or the opaque UUID; the adapter normalizes both):
   - Read signals in PRIORITY ORDER (Linear-adapted; the equivalent of
     commits>branch>title>body for GitHub):
     1. **Labels** — Linear's structured labels (`Bug`, `Feature`,
        `Improvement`, `Chore`, `Spec`) are the highest-signal kind
        classifier — they're author-applied and standardized per team.
     2. **Issue state** — `done` / `canceled` / `in_progress` /
        `backlog` / `todo` drive lifecycle status. A `done` bug issue
        earns a `BugPattern`, but **never** a `Fix` — Fix is reserved
        for the merged PR / commit that shipped the change.
     3. **Issue identifier prefix + title** — e.g. `ENG-123: fix rate
        limiter ...` — fallback kind signal if labels are missing.
     4. **Issue body** — author rationale; check for explicit
        `Why:` / linked decisions / linked PRs / repro steps.
     5. **Linked PRs / commits** — if the payload includes them, they
        narrow the implementation scope.
     6. **Comments** — LAST RESORT. Discussion threads are context, not
        standalone facts. Read only when the higher signals leave the
        kind / outcome ambiguous.
   - Classify:
     - **Author / Creator handle** (the issue creator) and the
       **completer** (if the resolution comment / state-change identifies
       one). For Linear, the creator drives `PERFORMED`; an assignee who
       completed the work may also drive `PERFORMED` on the same Activity.
     - **Kind** — `feat | fix | chore | refactor | docs | spec | other`
       derived first from labels, then from identifier+title.
     - **Summary** — 1-2 sentence functional summary of what was
       proposed / completed (use user-visible language).
     - **Bug evidence** — emit a `BugPattern` only when (a) labels
       contain `Bug` (or equivalent), AND (b) state is `done` (the bug
       has actually been confirmed and worked through), AND (c) the
       body or title carries a clear symptom. Do **NOT** emit a `Fix`
       from a Linear issue — Fix is reserved for the merged PR /
       commit that shipped the change (mirrors the GitHub one-shot
       rule). For `canceled` / `won't-fix` / `duplicate` issues with
       a Bug label, skip BugPattern too.
     - **Decision evidence** — emit Decision only when the body explicitly
       documents rationale + alternatives (Linear specs often have a
       `Why:` / `Decision:` / `Alternatives:` section). Most issues do
       NOT — be conservative.

3. Build one `LlmReconciliationPlan`-shaped object for the batch (see
   Mutations section). Call `apply_graph_mutations(plan, event_id, summary)`
   once per batch.
4. Use `update_todo_status` or `write_todos` to mark each batch todo done.
   Move to the next batch.

### Phase 3 — Finalize

1. When all todos are drained (or you've hit the tool-call budget with a
   coherent subset complete), tally:
   - Projects ingested / skipped
   - Documents ingested / skipped (or "documents API unavailable")
   - Issues ingested by kind (feat / fix / spec / chore / ...)
   - Distinct authors / creators
   - BugPattern nodes emitted, Decision nodes emitted
   - (Fix is NEVER emitted from this skill — see Anti-patterns)
2. `mark_event_processed(event_id, summary)` then `finish_batch(summary)`.

## Mutations (per item)

Use the existing ontology. Stable keys ensure backfill + future webhook
converge. Key formats below follow `domain.identity.mint_entity_key` rules
(see `domain.ontology.ENTITY_TYPES`).

Identity rules to respect (these are NOT free-form strings):

- `Activity` is `EXTERNAL_ID` with `key_prefix=activity`. Use
  `activity:linear:issue:<identifier>` (e.g.
  `activity:linear:issue:eng-123` — identifier lowercased),
  `activity:linear:project:<uuid-lowered>`, and
  `activity:linear:document:<uuid-lowered>`. Segments after `activity:` may
  contain `-`, `/`, `.` per `_EXTERNAL_ID_SAFE_RE`. Write the key directly
  as a string in your JSON `entity_key`; if you ever route through
  `mint_entity_key`, pass the leaf id as `external_id` and the
  source/kind/team as `extra_segments=("linear", "<kind>")` — do NOT pass
  the full colon-joined string as a single `external_id`, because
  `_normalize_external_id` strips colons.
- `Person` is `SLUG_ALIAS` with `key_prefix=person`. Use `person:<handle>`
  (the Linear user's handle / email-prefix lowercased; slugify if it
  contains dots or spaces).
- `Period` uses the production builder `timeline:period:daily:<pot>:<yyyy-mm-dd>`
  (matches `adapters/outbound/reconciliation/timeline_plan.py::_period_key`).
- `Document` is `CONTENT_HASH` with `key_prefix=document`. Use
  `document:<12-hex-sha256>` minted from a stable canonical string such as
  `"linear:document:<id>|<title>"` (so the same Linear document reingested
  later collides on the same key).
- `Fix` and `Decision` are `CONTENT_HASH`. Use
  `fix:<12-hex-sha256>` and `decision:<12-hex-sha256>` (mint via
  `mint_entity_key(spec, content=<stable-string>)` or hash inline).
  Stable seeds: `"linear:issue:<identifier>|fix|<symptom>"` and
  `"linear:issue:<identifier>|decision|<title>"`.
- `BugPattern` is `SLUG_ALIAS` with `key_prefix=bug_pattern`. Use
  `bug_pattern:linear-<team-slug>:<symptom-slug>` (e.g.
  `bug_pattern:linear-eng:rate-limiter-burst-collapse`) — each
  colon-separated segment must be a valid slug.

### Always emit (endpoint entities, at least once per batch)

- **Entity** `Period` — one per distinct activity date in the batch.
  - key: `timeline:period:daily:<pot>:<yyyy-mm-dd>`.
  - labels: `["Entity", "Period"]`.
  - properties: `period_kind="daily"`, `label="<yyyy-mm-dd>"`,
    `opened_at="<yyyy-mm-dd>T00:00:00+00:00"`.

### Per ISSUE — always emit

- **Entity** `Activity`
  - key: `activity:linear:issue:<identifier-lowered>`.
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<created_at or completed_at>`,
    `verb="linear_issue_<state>"` (e.g. `linear_issue_done`),
    `title=<issue_title>`, `summary=<your 1-2 sentence summary>`,
    `identifier=<ENG-123>`, `state=<state>`, `kind` (feat/fix/...),
    `issue_url`.
- **Entity** `Person` per creator (and assignee if distinct and the issue
  is `done`).
  - key: `person:<handle-lowered>`.
  - labels: `["Entity", "Person"]`.
  - properties: `name=<display_name>`, `handle=<handle>`.
- **Edge** `PERFORMED` — `person:<creator>` → activity key, with
  `valid_from=<occurred_at>`.
- **Edge** `PERFORMED` — `person:<assignee>` → activity key, with
  `valid_from=<occurred_at>` and `role="assignee"`, only when the assignee
  differs from the creator and the issue is `done`.
- **Edge** `IN_PERIOD` — activity key → period key (date from
  `occurred_at`).

### Per ISSUE — conditionally emit

- **Bug report** (labels include `Bug` AND state is `done` AND symptom
  is clear in title or body):
  - **Entity** `BugPattern`
    - key: `bug_pattern:linear-<team-slug>:<symptom-slug>` (segments
      slug-valid).
    - labels: `["Entity", "BugPattern"]`.
    - properties: `summary=<short canonical symptom sentence>`,
      `symptom_signature=<short canonical sentence>`,
      `title=<symptom title>`, `source_issue=<activity_key>`.
  - **Edge** `SEEN_IN` — bug_pattern → service/component/environment key
    when the issue identifies exactly one such affected scope; otherwise
    omit (Linear issues are not bound to a repo/service scope by default).
  - **Do NOT emit `Fix`** from a Linear issue. Fix is reserved for the
    merged PR / commit that shipped the change. The GitHub one-shot
    skill (sibling) emits Fix from merged PRs and writes the same
    `BugPattern` key as this skill so the two converge. If you find
    yourself building a `fix:<…>` entity here, stop.
  - **Do NOT emit `RESOLVED`** from this skill — that edge connects a
    Fix to a BugPattern, and since this skill does not emit Fix from
    issues, it must not emit RESOLVED either.
- **Design decision** (body explicitly documents rationale + alternatives,
  typically in a spec / PRD / RFC issue):
  - **Entity** `Decision`
    - key: `decision:<12-hex-sha256>` from
      `"linear:issue:<identifier>|decision|<title>"`.
    - labels: `["Entity", "Decision"]`.
    - properties: `title=<short title>`,
      `summary=<one sentence decision summary>`, `status="accepted"`,
      `rationale=<stated rationale>`,
      `alternatives_rejected=<list or string>`,
      `source_issue=<activity_key>`.
  - **Edge** `AFFECTS` — decision key → feature/component/service/code
    asset when the issue identifies exactly one affected scope; otherwise
    omit (Linear issues are often team-scoped, not code-scoped).

### Per PROJECT — always emit

- **Entity** `Activity`
  - key: `activity:linear:project:<id-lowered>`.
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<project_started_at or created_at>`,
    `verb="linear_project_<state>"`, `title=<project_name>`,
    `summary=<your 1-2 sentence summary>`, `state=<state>`, `project_url`.
- **Entity** `Document` — the project's description as a stable document.
  - key: `document:<12-hex-sha256>` from `"linear:project:<id>|<name>"`.
  - labels: `["Entity", "Document"]`.
  - properties: `title=<project_name>`, `summary=<description>`,
    `source_uri=<project_url>`, `source="linear"`, `linear_project_id=<id>`.
- **Edge** `TOUCHED` — activity → document, with `valid_from=<occurred_at>`.
- **Edge** `IN_PERIOD` — activity → period.
- **Edge** `PERFORMED` — `person:<project_lead>` → activity, with
  `valid_from=<occurred_at>`, only if a lead is identified.

### Per DOCUMENT — always emit

- **Entity** `Document`
  - key: `document:<12-hex-sha256>` from `"linear:document:<id>|<title>"`.
  - labels: `["Entity", "Document"]`.
  - properties: `title=<title>`, `summary=<1-2 sentence content summary>`,
    `source_uri=<doc_url>`, `source="linear"`, `linear_document_id=<id>`.
- **Entity** `Activity` for the document's creation / last update.
  - key: `activity:linear:document:<id-lowered>`.
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<updated_at or created_at>`,
    `verb="linear_document_active"`, `title=<title>`,
    `summary=<your 1-2 sentence summary>`.
- **Entity** `Person` for the creator (if identified).
- **Edge** `TOUCHED` — activity → document, with `valid_from=<occurred_at>`.
- **Edge** `IN_PERIOD` — activity → period.
- **Edge** `PERFORMED` — `person:<creator>` → activity, with
  `valid_from=<occurred_at>`.

### Per DOCUMENT — conditionally emit

- **Design decision** (document content explicitly documents rationale +
  alternatives — common for spec / PRD / RFC docs):
  - **Entity** `Decision` keyed `decision:<12-hex-sha256>` from
    `"linear:document:<id>|decision|<title>"`.
  - labels: `["Entity", "Decision"]`.
  - properties: `title=<short title>`,
    `summary=<one sentence decision summary>`, `status="accepted"`,
    `rationale=<stated rationale>`,
    `alternatives_rejected=<list or string>`,
    `source_document=<document_key>`.
  - **Edge** `MADE_IN` — decision key → document key.
  - **Edge** `AFFECTS` — decision key → feature/component/service/code
    asset, ONLY when a single affected scope is clear. Otherwise omit
    AFFECTS.

## Source-priority rationale (why)

Linear's structured signals — labels, state, assignee — are author-applied
and standardized per team. They beat free-form bodies for kind / outcome
classification. PR titles and bodies are author-stated rationale. Comments
are discussion context. Reading the full document/comment text burns budget
on rediscovering intent the author already encoded in metadata. Stop
climbing the priority ladder as soon as you can answer kind + summary +
bug/decision evidence.

## Bounds and budget

- ONE call each of `linear_list_projects`, `linear_list_documents`,
  `linear_list_issues`. No pagination beyond what the bounded list tools
  return.
- Soft tool-call cap: `30 + 3 × count` (each item averages ~3 calls:
  list + get + classify). Plan accordingly.
- If you approach the cap with a coherent recent subset of items ingested
  cleanly, FINISH — do not partially ingest an item. The tail can be
  re-run later with a smaller `count` (stable keys mean already-ingested
  items will be deduplicated, not duplicated).

## Anti-patterns

- Do NOT re-emit `linear_team.added` from this skill — that re-triggers
  the agent's full team-attach bootstrap.
- Do NOT page past the bounded list calls.
- Do NOT read full document content / issue comments unless the
  higher-priority signals leave intent unclear.
- Do NOT emit `Fix` from a Linear issue — even a `done` bug issue.
  Fix is reserved for the merged PR / commit that shipped the change.
  Closing a bug issue is evidence the team believes it is fixed; it is
  NOT itself a Fix. Sibling: the GitHub one-shot skill emits Fix from
  merged PRs and writes the same `BugPattern` key as this skill so the
  two converge.
- Do NOT emit `RESOLVED` from this skill (RESOLVED connects Fix →
  BugPattern, and this skill never emits Fix).
- Do NOT invent BugPatterns, Decisions, Persons, or Documents not
  actually evidenced in the data you read. Emit a warning record instead.
- Do NOT auto-close / auto-resolve any open issue or incident based on
  a Linear `done` status alone — the status is evidence, not closure.
- Do NOT fabricate a kind if list tools error — record a warning and
  continue with the kinds that did return.
- Do NOT run this skill on a team that already has live webhook
  ingestion going against the SAME date window — let the webhook handle
  live updates.

## Single-event contract

This skill, when invoked by the internal agent, runs as a single
`(linear, linear_team, one_shot_ingest)` event. Pass that ONE `event_id`
to every `apply_graph_mutations` call and to the final
`mark_event_processed` — per-item identity is the entity_key
(`activity:linear:issue:...`, `document:<hash>`, etc.), not the event id,
so multiple Activities / Documents under one event id is correct.

When invoked by Claude Code outside the event pipeline, there is no
internal-agent event state, so the internal `apply_graph_mutations` tool
will reject an empty or invented event id. Use this document as the
extraction procedure only when the host provides a compatible
context-graph write path and a valid event/provenance id. Otherwise stop
after producing the proposed plan; do not pretend to apply it.
