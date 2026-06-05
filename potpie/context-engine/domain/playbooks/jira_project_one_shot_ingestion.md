---
name: jira-project-one-shot-ingestion
description: One-time ingestion of a Jira project's recent epics and issues into the context graph. Not incremental — live updates go through the live Jira webhook path.
source_system: jira
event_type: jira_project
action: one_shot_ingest
enables_planner: true
---

# Jira project one-shot ingestion

A reusable skill for ingesting a Jira project's recent epics and issues
into the context graph in a single pass. Sibling to the Linear
`linear_team_one_shot_ingestion` and GitHub `repo_one_shot_ingestion`
playbooks: same shape, different source. Designed to be invoked by either
Claude Code (as a checklist with a compatible write path) or the internal
reconciliation agent (loaded as a playbook).

Jira has no first-class "Document" entity (those live in Confluence, which
is a separate connector out of scope here). Epics serve the spec/PRD role
in this skill — they anchor Decisions when their description documents
rationale + alternatives.

## When to invoke

- A user wants to seed the context graph from a Jira project's recent
  history in one pass.
- The Jira project is reachable via the connected Atlassian site (so
  connector tools are scoped to the right credentials).
- You will NOT run this skill repeatedly against the same project —
  incremental updates flow through live Jira webhook events, which write
  the same Activity keys so a future webhook converges with what this
  skill already wrote.

## Inputs

- `project_key`: Jira project key (required, e.g. `PROJ`). Must be
  reachable on the connected Atlassian site.
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
  `(jira, jira_project, one_shot_ingest)` event id for the run.

## Tools assumed available

- `jira_get_project(project_key)` — name, description, lead, project type,
  url. ONE call at the start of Phase 1.
- `jira_list_epics(project_key, limit=count)` — bounded enumeration of
  epic refs `{key, summary, updated_at}`, newest-first. JQL
  `project=<project_key> AND issuetype=Epic`. ONE call.
- `jira_list_issues(project_key, limit=count)` — bounded enumeration of
  non-epic issue refs `{key, summary, issuetype, updated_at}`,
  newest-first. JQL `project=<project_key> AND issuetype != Epic`. ONE
  call.
- `jira_get_issue(issue_key)` — full issue payload (summary, description,
  issuetype, status, labels, reporter, assignee, created/updated/resolved
  timestamps, url, linked issues / linked dev-panel PRs if present).
  Works for both epics and standalone issues since Jira treats Epic as an
  issuetype.
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

1. Trust the event payload that the project is reachable. If a list
   tool returns an auth / not-found error, abort with a warning. Do NOT
   attempt to connect a new Atlassian site from this skill.
2. Initialize the todo list with three entries:
   - `Fetch <project_key> metadata`
   - `Enumerate <project_key> epics`
   - `Enumerate <project_key> issues`

### Phase 1 — Enumerate (one project + two list calls)

1. Call `jira_get_project(project_key)` ONCE. Mark the metadata todo done.
2. Call `jira_list_epics(project_key, limit=count)` ONCE. Bounded
   server-side via JQL.
3. Call `jira_list_issues(project_key, limit=count)` ONCE. Bounded
   server-side via JQL (`issuetype != Epic` excludes epics already
   covered by the previous call).
4. Drain order: project metadata first (it frames the scope), epics
   next (they often justify Decisions), issues newest-first (the
   timeline spine).
5. For each returned ref, append a todo:
   `Process jira <kind> <key>` (where `<kind>` is `epic` or `issue`).
6. Use `update_todo_status` or `write_todos` to mark each enumeration todo
   done.

### Phase 2 — Drain batches

Drain todos sequentially across kinds (project, then epics, then issues);
within a batch parallelize up to `K`.

1. Choose `K` (`parallel_per_batch`, default 5). If hydrated payloads are
   large or slow, reduce `K` for the remaining items in this batch.
2. In parallel for `K` items at a time, hydrate via the matching tool:

   **PROJECT** — emitted directly from the `jira_get_project(project_key)`
   payload (no per-item enumeration; there is exactly one project being
   ingested):
   - Read signals in PRIORITY ORDER:
     1. **Project name + key** — name carries intent; key is the stable
        identifier.
     2. **Project lead** — single Person reference; the natural
        `PERFORMED` actor for the project Activity.
     3. **Project description** — author-stated goal / scope.
     4. **Project type / category** (if returned) — e.g. `software`,
        `service_desk`.
   - Classify: project name, description summary, lead handle.

   **EPIC** — `jira_get_issue(issue_key)` (epic key is e.g. `PROJ-12`):
   - Read signals in PRIORITY ORDER:
     1. **Summary** (epic title) — author-stated headline.
     2. **Status** — `To Do` / `In Progress` / `Done` / `Cancelled`
        drive lifecycle; status name is workspace-configurable so
        normalize to lowercase + underscores when used in `verb`.
     3. **Reporter / assignee** — Person references; reporter drives
        `PERFORMED` on the epic Activity.
     4. **Description** — only enough to derive a 1-2 sentence summary
        and to spot rationale + alternatives_rejected if this epic
        justifies a Decision (PRDs / RFCs / spec epics typically do;
        umbrella-tracking epics typically do not).
     5. **Linked stories** (if returned) — scale signal.

   **ISSUE** — `jira_get_issue(issue_key)` (issue key is e.g. `PROJ-123`):
   - Read signals in PRIORITY ORDER (Jira-adapted; the equivalent of
     labels>state>title>body for Linear):
     1. **Issuetype** — Jira's first-class field (`Bug`, `Story`,
        `Task`, `Sub-task`, `Spike`). Author-applied and standardized
        per project — highest-signal kind classifier.
     2. **Status** — `Done` / `Cancelled` / `In Progress` /
        `Backlog` / `To Do`. A `Done` Bug-typed issue earns a
        `BugPattern`, but **never** a `Fix` — Fix is reserved for the
        merged PR / commit that shipped the change.
     3. **Labels** — additional intent signal (severity, area, etc.).
     4. **Summary** (issue title) — fallback kind signal if issuetype is
        ambiguous.
     5. **Description** — author rationale; check for explicit
        `Why:` / linked decisions / linked PRs / repro steps. Note
        Jira descriptions can be ADF (Atlassian Document Format) JSON —
        treat as opaque structured text and read the plain-text
        rendering only.
     6. **Linked issues / linked PRs** (dev panel) — if the payload
        includes them, they narrow the implementation scope.
     7. **Comments** — LAST RESORT. Discussion threads are context, not
        standalone facts. Read only when the higher signals leave the
        kind / outcome ambiguous.
   - Classify:
     - **Reporter / assignee handles** — reporter drives `PERFORMED`;
       an assignee who completed the work (status=Done, assignee !=
       reporter) may also drive `PERFORMED` with `role="assignee"` on
       the same Activity.
     - **Kind** — `bug | feat | chore | refactor | docs | spec | other`
       derived first from issuetype (Bug → bug, Story → feat, Task →
       chore/feat by labels, Spike → spec), then from summary.
     - **Summary** — 1-2 sentence functional summary of what was
       proposed / completed (use user-visible language).
     - **Bug evidence** — emit a `BugPattern` only when (a) issuetype is
       `Bug` (or equivalent), AND (b) status is `Done`, AND (c) the
       description or summary carries a clear symptom.
       Do **NOT** emit a `Fix` from a Jira issue — Fix is reserved for
       the merged PR / commit that shipped the change (mirrors the
       Linear / GitHub one-shot rule). For `Cancelled` / `Won't Fix` /
       `Duplicate` Bug-typed issues, skip BugPattern too.
     - **Decision evidence** — emit Decision only when the description
       explicitly documents rationale + alternatives (spec / RFC issues
       often have a `Why:` / `Decision:` / `Alternatives:` section).
       Most issues do NOT — be conservative.

3. Build one `LlmReconciliationPlan`-shaped object for the batch (see
   Mutations section). Call `apply_graph_mutations(plan, event_id, summary)`
   once per batch.
4. Use `update_todo_status` or `write_todos` to mark each batch todo done.
   Move to the next batch.

### Phase 3 — Finalize

1. When all todos are drained (or you've hit the tool-call budget with a
   coherent subset complete), tally:
   - Project metadata captured (yes / no)
   - Epics ingested / skipped
   - Issues ingested by kind (bug / feat / chore / spec / ...)
   - Distinct reporters / assignees
   - BugPattern nodes emitted, Decision nodes emitted
   - (Fix is NEVER emitted from this skill — see Anti-patterns)
2. `mark_event_processed(event_id, summary)` then `finish_batch(summary)`.

## Mutations (per item)

Use the existing ontology. Stable keys ensure backfill + future webhook
converge. Key formats below follow `domain.identity.mint_entity_key` rules
(see `domain.ontology.ENTITY_TYPES`).

Identity rules to respect (these are NOT free-form strings):

- `Activity` is `EXTERNAL_ID` with `key_prefix=activity`. Use
  `activity:jira:project:<project-key-lowered>`,
  `activity:jira:epic:<issue-key-lowered>` (e.g.
  `activity:jira:epic:proj-12`), and
  `activity:jira:issue:<issue-key-lowered>` (e.g.
  `activity:jira:issue:proj-123`). Segments after `activity:` may
  contain `-`, `/`, `.` per `_EXTERNAL_ID_SAFE_RE`. Write the key
  directly as a string in your JSON `entity_key`; if you ever route
  through `mint_entity_key`, pass the leaf id as `external_id` and the
  source/kind as `extra_segments=("jira", "<kind>")` — do NOT pass the
  full colon-joined string as a single `external_id`, because
  `_normalize_external_id` strips colons.
- `Person` is `SLUG_ALIAS` with `key_prefix=person`. Use `person:<handle>`
  (the Atlassian account display name / email-prefix lowercased; slugify
  if it contains dots or spaces). When the payload exposes only an
  Atlassian `accountId`, use `person:atlassian-<accountId-lowered>`.
- `Period` uses the production builder `timeline:period:daily:<pot>:<yyyy-mm-dd>`
  (matches `adapters/outbound/reconciliation/timeline_plan.py::_period_key`).
- `Document` is `CONTENT_HASH` with `key_prefix=document`. Use
  `document:<12-hex-sha256>` minted from a stable canonical string such as
  `"jira:project:<project_key>|<name>"` or
  `"jira:epic:<issue_key>|<summary>"` (so the same item reingested later
  collides on the same key).
- `Fix` and `Decision` are `CONTENT_HASH`. Use
  `fix:<12-hex-sha256>` and `decision:<12-hex-sha256>` (mint via
  `mint_entity_key(spec, content=<stable-string>)` or hash inline).
  Stable seeds for Decision:
  `"jira:issue:<issue_key>|decision|<title>"` and
  `"jira:epic:<issue_key>|decision|<title>"`.
- `BugPattern` is `SLUG_ALIAS` with `key_prefix=bug_pattern`. Use
  `bug_pattern:jira-<project-key-slug>:<symptom-slug>` (e.g.
  `bug_pattern:jira-proj:rate-limiter-burst-collapse`) — each
  colon-separated segment must be a valid slug.

### Always emit (endpoint entities, at least once per batch)

- **Entity** `Period` — one per distinct activity date in the batch.
  - key: `timeline:period:daily:<pot>:<yyyy-mm-dd>`.
  - labels: `["Entity", "Period"]`.
  - properties: `period_kind="daily"`, `label="<yyyy-mm-dd>"`,
    `opened_at="<yyyy-mm-dd>T00:00:00+00:00"`.

### Per ISSUE — always emit

- **Entity** `Activity`
  - key: `activity:jira:issue:<issue-key-lowered>`.
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<resolved_at or updated_at or created_at>`,
    `verb="jira_issue_<status-normalized>"` (e.g. `jira_issue_done`,
    `jira_issue_in_progress` — lowercased, spaces → underscores),
    `title=<issue_summary>`, `summary=<your 1-2 sentence summary>`,
    `key=<PROJ-123>`, `jira_status=<status>` (raw Jira status like
    `"Done"` / `"In Progress"` — do NOT use the bare property name
    `status` here, the Activity validator treats `status` as a
    lifecycle field with allowlist `{in_progress, completed, unknown}`),
    `issuetype=<issuetype>`, `kind` (bug/feat/...), `issue_url`.
- **Entity** `Person` per reporter (and assignee if distinct and the
  issue is `Done`).
  - key: `person:<handle-lowercased>`.
  - labels: `["Entity", "Person"]`.
  - properties: `name=<display_name>`, `handle=<handle>`.
- **Edge** `PERFORMED` — `person:<reporter>` → activity key, with
  `valid_from=<occurred_at>` and `role="reporter"`.
- **Edge** `PERFORMED` — `person:<assignee>` → activity key, with
  `valid_from=<occurred_at>` and `role="assignee"`, only when the
  assignee differs from the reporter and the issue is `Done`.
- **Edge** `IN_PERIOD` — activity key → period key (date from
  `occurred_at`).

### Per ISSUE — conditionally emit

- **Bug report** (issuetype is `Bug` AND status is `Done` AND symptom
  is clear in summary or description):
  - **Entity** `BugPattern`
    - key: `bug_pattern:jira-<project-key-slug>:<symptom-slug>`
      (segments slug-valid).
    - labels: `["Entity", "BugPattern"]`.
    - properties: `summary=<short canonical symptom sentence>`,
      `symptom_signature=<short canonical sentence>`,
      `title=<symptom title>`, `source_issue=<activity_key>`.
  - **Edge** `SEEN_IN` — bug_pattern → service/component/environment key
    when the issue identifies exactly one such affected scope; otherwise
    omit (Jira issues are not bound to a repo/service scope by default).
  - **Do NOT emit `Fix`** from a Jira issue. Fix is reserved for the
    merged PR / commit that shipped the change. The GitHub one-shot
    skill (sibling) emits Fix from merged PRs and writes the same
    `BugPattern` key as this skill so the two converge. If you find
    yourself building a `fix:<…>` entity here, stop.
  - **Do NOT emit `RESOLVED`** from this skill — that edge connects a
    Fix to a BugPattern, and since this skill does not emit Fix from
    issues, it must not emit RESOLVED either.
- **Design decision** (description explicitly documents rationale +
  alternatives, typically in a Spike / RFC issue):
  - **Entity** `Decision`
    - key: `decision:<12-hex-sha256>` from
      `"jira:issue:<issue_key>|decision|<title>"`.
    - labels: `["Entity", "Decision"]`.
    - properties: `title=<short title>`,
      `summary=<one sentence decision summary>`, `status="accepted"`,
      `rationale=<stated rationale>`,
      `alternatives_rejected=<list or string>`,
      `source_issue=<activity_key>`.
  - **Edge** `AFFECTS` — decision key → feature/component/service/code
    asset when the issue identifies exactly one affected scope; otherwise
    omit (Jira issues are often team-scoped, not code-scoped).

### Per EPIC — always emit

- **Entity** `Activity`
  - key: `activity:jira:epic:<issue-key-lowered>`.
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<resolved_at or updated_at or created_at>`,
    `verb="jira_epic_<status-normalized>"`, `title=<epic_summary>`,
    `summary=<your 1-2 sentence summary>`, `key=<PROJ-12>`,
    `jira_status=<status>` (raw Jira status; same rationale as the
    ISSUE block — do NOT use the bare `status` property name on an
    Activity), `epic_url`.
- **Entity** `Document` — the epic's description as a stable document
  (epics serve the Linear-Document role for Jira since Confluence is out
  of scope).
  - key: `document:<12-hex-sha256>` from `"jira:epic:<issue_key>|<summary>"`.
  - labels: `["Entity", "Document"]`.
  - properties: `title=<epic_summary>`,
    `summary=<1-2 sentence description summary>`, `source_uri=<epic_url>`,
    `source="jira"`, `jira_issue_key=<PROJ-12>`.
- **Entity** `Person` for the reporter (if identified).
- **Edge** `TOUCHED` — activity → document, with `valid_from=<occurred_at>`.
- **Edge** `IN_PERIOD` — activity → period.
- **Edge** `PERFORMED` — `person:<reporter>` → activity, with
  `valid_from=<occurred_at>`.

### Per EPIC — conditionally emit

- **Design decision** (epic description explicitly documents rationale +
  alternatives — common for spec / PRD / RFC epics):
  - **Entity** `Decision` keyed `decision:<12-hex-sha256>` from
    `"jira:epic:<issue_key>|decision|<title>"`.
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

### Per PROJECT — always emit

- **Entity** `Activity`
  - key: `activity:jira:project:<project-key-lowered>`.
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<project_created_at or now>`,
    `verb="jira_project_active"`, `title=<project_name>`,
    `summary=<your 1-2 sentence summary>`, `key=<project_key>`,
    `project_url`.
- **Entity** `Document` — the project's description as a stable document.
  - key: `document:<12-hex-sha256>` from `"jira:project:<project_key>|<name>"`.
  - labels: `["Entity", "Document"]`.
  - properties: `title=<project_name>`, `summary=<description>`,
    `source_uri=<project_url>`, `source="jira"`,
    `jira_project_key=<project_key>`.
- **Entity** `Person` — the project lead, only if identified.
  - key: `person:<project_lead>`.
  - labels: `["Entity", "Person"]`.
  - properties: include `display_name`, `email`, and `id` when available.
- **Edge** `TOUCHED` — activity → document, with `valid_from=<occurred_at>`.
- **Edge** `IN_PERIOD` — activity → period.
- **Edge** `PERFORMED` — `person:<project_lead>` → activity, with
  `valid_from=<occurred_at>`, only if a lead is identified.

## Source-priority rationale (why)

Jira's structured signals — issuetype, status, labels — are author-applied
and standardized per project. They beat free-form description text for
kind / outcome classification. Summaries and descriptions are
author-stated rationale. Comments are discussion context. Reading full
description text (especially ADF JSON payloads) or comment threads burns
budget on rediscovering intent the author already encoded in metadata.
Stop climbing the priority ladder as soon as you can answer kind +
summary + bug/decision evidence.

## Bounds and budget

- ONE call each of `jira_get_project`, `jira_list_epics`,
  `jira_list_issues`. No pagination beyond what the bounded list tools
  return.
- Soft tool-call cap: `30 + 3 × count` (each item averages ~3 calls:
  list + get + classify). Plan accordingly.
- If you approach the cap with a coherent recent subset of items ingested
  cleanly, FINISH — do not partially ingest an item. The tail can be
  re-run later with a smaller `count` (stable keys mean already-ingested
  items will be deduplicated, not duplicated).

## Anti-patterns

- Do NOT re-emit `jira_project.added` from this skill — that re-triggers
  any future project-attach bootstrap.
- Do NOT page past the bounded list calls.
- Do NOT read full descriptions / issue comments unless the
  higher-priority signals leave intent unclear.
- Do NOT emit `Fix` from a Jira issue — even a `Done` Bug-typed issue.
  Fix is reserved for the merged PR / commit that shipped the change.
  Marking a bug `Done` is evidence the team believes it is fixed; it is
  NOT itself a Fix. Sibling: the GitHub one-shot skill emits Fix from
  merged PRs and writes the same `BugPattern` key as this skill so the
  two converge.
- Do NOT emit `RESOLVED` from this skill (RESOLVED connects Fix →
  BugPattern, and this skill never emits Fix).
- Do NOT invent BugPatterns, Decisions, Persons, or Documents not
  actually evidenced in the data you read. Emit a warning record instead.
- Do NOT auto-close / auto-resolve any open issue or incident based on
  a Jira `Done` status alone — the status is evidence, not closure.
- Do NOT fabricate a kind if list tools error — record a warning and
  continue with the kinds that did return.
- Do NOT ingest Confluence pages from this skill — Confluence is a
  separate connector entirely (out of scope here).
- Do NOT run this skill on a project that already has live webhook
  ingestion going against the SAME date window — let the webhook handle
  live updates.

## Single-event contract

This skill, when invoked by the internal agent, runs as a single
`(jira, jira_project, one_shot_ingest)` event. Pass that ONE `event_id`
to every `apply_graph_mutations` call and to the final
`mark_event_processed` — per-item identity is the entity_key
(`activity:jira:issue:...`, `document:<hash>`, etc.), not the event id,
so multiple Activities / Documents under one event id is correct.

When invoked by Claude Code outside the event pipeline, there is no
internal-agent event state, so the internal `apply_graph_mutations` tool
will reject an empty or invented event id. Use this document as the
extraction procedure only when the host provides a compatible
context-graph write path and a valid event/provenance id. Otherwise stop
after producing the proposed plan; do not pretend to apply it.
