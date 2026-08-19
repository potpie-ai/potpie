---
name: repo-one-shot-ingestion-gitlab
description: One-time change-history ingestion of a GitLab project's recent merged merge requests and issues into the context graph. Timeline scope only — baseline repo understanding is a separate harness-led procedure (potpie-repo-baseline). Not incremental — live updates go through the merge_request.merged and issue.opened webhook paths.
source_system: gitlab
event_type: repository
action: one_shot_ingest
enables_planner: true
---

# GitLab project change-history ingestion (one-shot)

A reusable skill for ingesting a GitLab project's recent **merged merge
requests** and **issues** into the context graph in a single pass. It uses an
enumerate → batch → hydrate shape for GitLab source history. Designed to be
invoked by either Claude Code (as a checklist with a compatible write path) or
the internal reconciliation agent (loaded as a playbook).

Scope is **change history only**: timeline activities, clear fixes, explicit
decisions, and evidenced bug patterns. Does not walk a working tree, does not
build a module / feature map, and must not be used to infer the project's
baseline architecture — that is the separate harness-led
`potpie-repo-baseline` procedure. It only reconciles authored GitLab source
history. Does **not** cover epics, wikis, or snippets (no connector tool).

This is the GitLab counterpart of `repo-one-shot-ingestion`. GitLab differs
from GitHub in four ways that matter here, and the procedure below already
accounts for them — do not "translate back" to GitHub habits:

1. Merge requests and issues are **separate collections**. Nothing has to be
   filtered out of the issue list.
2. Identifiers are project-scoped **iids** (`!42`, `#42`), not repo-global
   numbers. Two different projects both have a `!1`.
3. Inline review lives in **discussions** (threads that can be resolved), and
   review verdicts live in **approvals** plus **system notes**.
4. A project path may contain nested groups (`group/subgroup/project`).

## When to invoke

- A user wants to seed the context graph from a project's recent **MR merge
  history + issue history** in one pass.
- The project is already attached to the target pot (so tool calls are scoped).
- You will NOT run this skill repeatedly against the same project —
  incremental updates flow through the live `gitlab / merge_request / merged`
  and `gitlab / issue / opened` webhooks, which write the same Activity keys
  so a future webhook converges with what this skill already wrote.

## Inputs

- `project`: `group/project` (required, subgroups allowed:
  `group/subgroup/project`). Must be attached to the active pot.
- `count`: soft per-kind list limit. Default `50`. When the host submits this
  event, read `count` from `event.payload.count`. Pass it as `limit` on
  **both** `gitlab_list_merge_requests` and `gitlab_list_issues`. The server
  still clamps to `CONTEXT_ENGINE_BACKFILL_MAX_ITEMS` (default 300) and the
  trailing window (`CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS`, default 365 days).
  Hard ceiling: respect whatever each bounded list tool returns — do not
  page past it.
- `batch_size`: how many items to mark as one todo. Default `10`.
- `parallel_per_batch` (`K`): how many items to hydrate at once within a
  batch. Default `5`. Drop lower only after a fetched item proves unusually
  expensive to hydrate (large description / note volume or an optional diff
  fetch).
- `event_id`: required for the internal reconciliation agent. The single
  `(gitlab, repository, one_shot_ingest)` event id for the run.

## Tools assumed available

- `gitlab_list_merge_requests(project, limit=count)` — bounded MR refs,
  newest first. **ONE call.**
- `gitlab_list_issues(project, limit=count)` — bounded issue refs, newest
  first. **ONE call.** GitLab keeps issues and MRs apart, so no filtering is
  needed.
- `gitlab_get_merge_request(project, iid)` — full MR metadata (no diff by
  default): `title`, `body`, `state`, `merged_at`, `head_branch`,
  `base_branch`, `author`, `labels`, `milestone`, `reviewers`, `assignees`,
  `draft`, `merge_status`, `url`. The same tool accepts optional
  `include_diff=true`; when set, the response also includes a `files` array
  of per-file objects with `filename`, `status`, and `patch` (use only as a
  last resort; see Phase 2). GitLab does not return per-file
  `additions`/`deletions` — do not expect them.
- `gitlab_get_merge_request_commits(project, iid)` — commit messages.
- `gitlab_get_merge_request_discussions(project, iid)` — inline review
  comments, each with `path`, `line`, `body`, `user`, `discussion_id`, and
  `resolved`. An unresolved thread is unfinished review, not a decision.
- `gitlab_get_merge_request_notes(project, iid)` — MR conversation notes,
  oldest first. **System notes are included and are load-bearing on GitLab
  CE**: "approved this merge request", "requested review from @x", "marked
  this merge request as draft" are the review-action trail.
- `gitlab_get_merge_request_approvals(project, iid)` — who approved and how
  many approvals remain. Returns `available=false` on instances that do not
  expose the endpoint; fall back to the approval system notes.
- `gitlab_get_merge_request_state_events(project, iid)` — opened / closed /
  merged / reopened transitions with the acting user.
- `gitlab_get_merge_request_closes_issues(project, iid)` — the issues this MR
  closes on merge. Prefer this over parsing `Closes #n` out of the body.
- `gitlab_get_issue(project, iid)` — `title`, `body`, `state`, `author`,
  `labels`, `assignees`, `milestone`, `due_date`, `weight`, `issue_type`,
  `time_stats`, `created_at`, `updated_at`, `url`.
- `gitlab_get_issue_notes(project, iid)` — comments on an issue. Unlike the
  GitHub skill, this surface **does** exist here; use it when the body alone
  leaves intent unclear.
- `gitlab_get_issue_links(project, iid)` — merge requests related to, and
  closing, the issue.
- `apply_graph_mutations(plan, event_id, summary)` — context-graph write. The
  `plan` argument MUST be an object with this shape:
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

1. Confirm the event payload names a project (`group/project`). If not, abort
   with a warning. Do NOT attempt to attach or clone a project from this
   skill.
2. Initialize the todo list with two entries:
   - `Enumerate last <count> merged MRs of <project>`
   - `Enumerate last <count> issues of <project>`

### Phase 1 — Enumerate (two list calls, one each)

1. Call `gitlab_list_merge_requests(project, limit=count)` ONCE. Bounded
   server-side. Filter to merged MRs (`merged=true` or `state="merged"` on
   refs); skip closed-unmerged with a warning. Skip `draft=true` MRs that
   never merged.
2. Call `gitlab_list_issues(project, limit=count)` ONCE.
3. Drain order: **merged MRs first** (completed work timeline), then
   **issues newest-first** (bugs, feature requests, questions).
4. Split returned refs into batches of `batch_size`. For each batch, append
   a todo:
   - MRs: `Process MRs [!a, !b, ...]`
   - issues: `Process issues [#a, #b, ...]`
5. Use `update_todo_status` or `write_todos` to mark each enumeration todo
   done.

### Phase 2 — Drain batches

Drain todos sequentially across kinds (MRs first, then issues); within a
batch parallelize up to `K`.

#### Merge request items

For each batch of MR todos, in parallel for `K` MRs at a time:

1. `gitlab_get_merge_request(project, iid)` — title, body, author,
   `merged_at`, `head_branch`, `base_branch`, labels, milestone, reviewers,
   and URL.
2. Read source signals in PRIORITY ORDER, stopping when intent is clear:
   1. **Commit messages** (`gitlab_get_merge_request_commits`) — concise,
      declarative, often conventional-commit prefixed.
   2. **Branch name** (`head_branch`) — `feat/...`, `fix/...`, `chore/...`
      carry intent.
   3. **MR title** — author-stated headline.
   4. **MR description / body** — author rationale; check for
      `Why:` / `Closes #` / linked issues.
   5. **Review signals** — only if higher signals are ambiguous:
      `gitlab_get_merge_request_discussions` for inline threads (an
      unresolved thread is an open question, not an accepted decision), and
      `gitlab_get_merge_request_notes` for the conversation plus the system
      notes that record who approved and who was asked to review. Use
      `gitlab_get_merge_request_approvals` when you need the approval roster
      as data rather than prose.
   6. **Code diff** — LAST RESORT. If needed, call
      `gitlab_get_merge_request(project, iid, include_diff=true)` (optional
      `include_diff` boolean on the same tool as step 1). The response adds
      a `files` array; each element is an object with `filename`, `status`,
      and `patch` — use `filename` for path-based reasoning, not a separate
      path/id field. Reading patches burns budget rediscovering intent the
      author already wrote.
3. Classify:
   - **Author handle(s)** — primary author + co-authors (from
     `Co-authored-by:` trailers).
   - **Reviewers** — from `reviewers` plus whoever actually approved
     (approvals or approval system notes). An assigned reviewer who never
     approved is not a reviewer of record.
   - **Kind** — `feat | fix | chore | refactor | docs | test | other` from
     conventional commit prefix, branch prefix, or title.
   - **Summary** — 1-2 sentence functional summary.
   - **Bug evidence** — does the MR fix a bug? capture symptom signature;
     emit Fix + BugPattern only when symptom is clear.
   - **Decision evidence** — body explicitly documents rationale +
     alternatives_rejected (most MRs do NOT — be conservative). A resolved
     discussion thread where a reviewer's objection was answered can support
     a Decision only when the resolution is stated in words.
   - **Closed issues** — `gitlab_get_merge_request_closes_issues` when the
     body hints at a linkage; record it on the Activity rather than guessing
     from `#n` mentions.

#### Issue items

For each batch of issue todos, in parallel for `K` issues at a time:

1. `gitlab_get_issue(project, iid)` — title, body, state, author, labels,
   assignees, milestone, `due_date`, `created_at`, `updated_at`, `url`.
2. Read source signals in PRIORITY ORDER, stopping when intent is clear:
   1. **Labels** — `bug` / `feature` / `documentation` / `question` are the
      highest-signal kind classifier — author-applied and standardized per
      project. `issue_type` (`issue` / `incident` / `test_case`) refines it.
   2. **State** — `opened` vs `closed`. Do NOT treat closed as
      auto-resolved; closure can mean fixed, won't-fix, or duplicate.
   3. **Title** — author-stated headline.
   4. **Body** — rationale, repro steps, `Why:` / linked MRs / linked
      issues. If body is empty, summarize from title + labels only.
   5. **Comments** (`gitlab_get_issue_notes`) — real discussion; use it when
      title + labels + body leave the report ambiguous.
   6. **Linked work** (`gitlab_get_issue_links`) — the MRs that reference or
      close the issue, when you need to connect the report to the fix.
3. Classify:
   - **Reporter** — `author` from `gitlab_get_issue`.
   - **Kind** — `bug | feat | chore | question | docs | other` from labels
     first, then title.
   - **Summary** — 1-2 sentence summary of what was reported or requested.
   - **Bug report** (open or closed) — capture symptom signature for
     BugPattern. **Do NOT emit `Fix` from an issue** — Fix is reserved for
     merged MRs that shipped a fix.
   - **Assignment / scheduling** — `assignees`, `milestone`, `due_date` are
     task-tracking facts; record them on the Activity, do not invent them.
   - **Decision** — only when the body or a comment explicitly documents
     rationale + alternatives (rare on issues).

Build one `LlmReconciliationPlan`-shaped object for the batch (see Mutations
section). Call `apply_graph_mutations(plan, event_id, summary)` once per
batch. Use `update_todo_status` or `write_todos` to mark each batch todo
done.

### Phase 3 — Finalize

1. When all todos are drained (or you've hit the tool-call budget with a
   coherent subset complete), tally:
   - MRs ingested / skipped
   - Issues ingested by state (opened / closed) and by kind
   - Distinct authors / reporters / approvers
   - Fix nodes emitted (MR-only), BugPattern nodes emitted, Decision
     nodes emitted
2. `mark_event_processed(event_id, summary)` then `finish_batch(summary)`.

## Mutations (per item)

Use the existing ontology. Stable keys ensure backfill + future webhook
converge. Key formats below follow `potpie_context_engine.domain.identity.mint_entity_key` rules
(see `potpie_context_engine.domain.ontology.ENTITY_TYPES`).

Identity rules to respect (these are NOT free-form strings):

- `Repository` is `SLUG_ALIAS` with `key_prefix=repo`. The body must be a
  lowercase slug — letters/digits/hyphens only — NO dots, NO slashes. So
  `repo:gitlab.example.com/acme/api` is INVALID. Slugify the whole project
  path, subgroups included: `group/sub/project` → `repo:group-sub-project`.
- `Person` is `SLUG_ALIAS` with `key_prefix=person`. Use `person:<handle>`
  (the GitLab username lowercased; usually already slug-clean).
- `Period` uses the key form `timeline:period:daily:<pot>:<yyyy-mm-dd>`
  (the `Period` identity_policy in `domain/ontology.py`).
- `Activity` is `EXTERNAL_ID` with `key_prefix=activity`. Two distinct forms:
  - MR: `activity:gitlab:mr:<group>/<project>:<iid>`
  - Issue: `activity:gitlab:issue:<group>/<project>:<iid>`
  Segments after `activity:` may contain `/` per `_EXTERNAL_ID_SAFE_RE`, so a
  nested group path stays intact. Lowercase the project-path segment. Write
  this key directly as a string in your JSON `entity_key`. If you ever route
  through `mint_entity_key`, pass the iid as `external_id` and the path as
  `extra_segments=("gitlab","mr","<group>/<project>")` (or `"issue"`) — do
  NOT pass the full colon-joined string as a single `external_id`, because
  `_normalize_external_id` strips colons and the key collapses to
  `activity:gitlab-mr-<group>-<project>-<iid>`.
- `Fix` and `Decision` are `CONTENT_HASH`. The body must be a 12-hex
  fingerprint of canonical content — NEVER encode the MR / issue iid into
  the key. Use `fix:<12-hex-sha256>` and `decision:<12-hex-sha256>` (mint
  via `mint_entity_key(spec, content=<stable-string>)` or hash inline).
- `BugPattern` is `SLUG_ALIAS` with `key_prefix=bug_pattern`. Use
  `bug_pattern:<project-slug>:<symptom-slug>` (e.g.
  `bug_pattern:acme-api:db-timeout`) — each colon-separated segment must be
  a valid slug.

### Always emit (endpoint entities, at least once per batch)

- **Entity** `Repository`
  - key: `repo:<project-path-slugified>`. Prefer reusing the existing
    Repository entity_key already in the graph if the read path lets you
    look it up by `name=<group>/<project>`.
  - labels: `["Entity", "Repository"]`.
  - properties: `name="<group>/<project>"`, `provider="gitlab"`,
    `provider_host=<the instance host from the MR/issue url — self-managed
    GitLab CE is NOT gitlab.com>`, `owner=<group path>`,
    `repo=<project name>`.
- **Entity** `Period` — one per distinct activity date in the batch.
  - key: `timeline:period:daily:<pot>:<yyyy-mm-dd>`.
  - labels: `["Entity", "Period"]`.
  - properties: `period_kind="daily"`, `date="<yyyy-mm-dd>"`.

### Per merged MR — always emit

- **Entity** `Activity`
  - key: `activity:gitlab:mr:<group>/<project>:<iid>` (path lowercased).
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<merged_at>`, `verb_class="mr_merged"`,
    `title=<mr_title>`, `summary=<your 1-2 sentence summary>`,
    `head_branch`, `base_branch`, `kind` (feat/fix/...), `mr_url`. Add
    `approved_by` (list of usernames) and `closes_issues` (list of iids)
    when you actually fetched them.
- **Entity** `Person` — one per author / co-author / approver.
  - key: `person:<handle-lowercased>`.
  - labels: `["Entity", "Person"]`.
- **Edge** `PERFORMED` — `person:<primary_author>` → activity key.
- **Edge** `AUTHORED` — `person:<co_author>` → activity key, per co-author.
- **Edge** `TOUCHED` — activity key → repository key.
- **Edge** `IN_PERIOD` — activity key → period key.

### Per merged MR — conditionally emit

- **Bug fix** (kind=`fix` AND MR body / linked issue has a clear symptom):
  - **Entity** `BugPattern`
    - key: `bug_pattern:<project-slug>:<symptom-slug>` (segments slug-valid).
    - labels: `["Entity", "BugPattern"]`.
    - properties: `symptom_signature=<short canonical sentence>`,
      `name=<symptom title>`.
  - **Entity** `Fix`
    - key: `fix:<12-hex-sha256>` minted from a stable canonical string such
      as `"gitlab:mr:<group>/<project>:<iid>|fix|<symptom-signature>"`.
    - labels: `["Entity", "Fix"]`.
    - properties: `fix_steps=<short description>`,
      `verification_status="unverified"`, `source_mr=<activity_key>`.
  - **Edge** `RESOLVED` — fix key → bug_pattern key.
  - **Edge** `REPRODUCES` — bug_pattern key → repository key.
- **Design decision** (MR body explicitly documents rationale + alternatives):
  - **Entity** `Decision`
    - key: `decision:<12-hex-sha256>` from
      `"gitlab:mr:<group>/<project>:<iid>|decision|<title>"`.
    - labels: `["Entity", "Decision"]`.
    - properties: `name=<short title>`, `rationale=<stated rationale>`,
      `alternatives_rejected=<list or string>`, `source_mr=<activity_key>`.
  - **Edge** `DECIDED` — decision key → repository key.
  - **Edge** `AFFECTS` — decision key → repository key.

Touched services (optional, only if obvious): if an optional
`gitlab_get_merge_request(project, iid, include_diff=true)` fetch returns
`files`, inspect each entry's `filename` (repo-relative path). When those
paths clearly map to an existing `Service` entity (e.g. every changed
`filename` under `services/auth/`), emit an extra `TOUCHED` edge activity →
that service. Do NOT invent Services that don't already exist in the graph.

### Per issue — always emit

- **Entity** `Activity`
  - key: `activity:gitlab:issue:<group>/<project>:<iid>` (path lowercased).
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<created_at>`,
    `verb_class="gitlab_issue_<state>"` where `<state>` is `opened` or
    `closed` — these are property VALUES, not tool names; `title`,
    `summary=<your 1-2 sentence summary>`, `state`, `kind`
    (bug/feat/chore/question/...), `issue_url`. Add `assignees`,
    `milestone`, and `due_date` when present — they are the task-tracking
    facts this pass exists to capture.
- **Entity** `Person` — the reporter (issue author), plus assignees when the
  issue is assigned.
  - key: `person:<handle-lowercased>`.
  - labels: `["Entity", "Person"]`.
- **Edge** `PERFORMED` — `person:<reporter>` → activity key.
- **Edge** `TOUCHED` — activity key → repository key.
- **Edge** `IN_PERIOD` — activity key → period key.

### Per issue — conditionally emit

- **Bug report** (labels include `bug` AND body or title carries a clear
  symptom):
  - **Entity** `BugPattern`
    - key: `bug_pattern:<project-slug>:<symptom-slug>`.
    - labels: `["Entity", "BugPattern"]`.
    - properties: `symptom_signature=<short canonical sentence>`,
      `name=<symptom title>`, `source_issue=<activity_key>`.
  - **Edge** `REPRODUCES` — bug_pattern key → repository key.
  - **Do NOT emit `Fix`** — Fix is reserved for merged MRs that shipped a
    fix. Closing an issue is not evidence that a fix exists.
- **Design decision** (issue body or a comment explicitly documents rationale
  + alternatives — rare on issues, common on spec / RFC issues):
  - **Entity** `Decision`
    - key: `decision:<12-hex-sha256>` from
      `"gitlab:issue:<group>/<project>:<iid>|decision|<title>"`.
    - labels: `["Entity", "Decision"]`.
    - properties: `name=<short title>`, `rationale=<stated rationale>`,
      `alternatives_rejected=<list or string>`,
      `source_issue=<activity_key>`.
  - **Edge** `DECIDED` — decision key → repository key.
  - **Edge** `AFFECTS` — decision key → repository key.

When an MR you ingested closes an issue, prefer recording the linkage in the
MR Activity's `properties` (a `closes_issues` list) or the plan's `evidence`
array, rather than duplicating an issue Activity inside the MR pass — the
issue pass will emit its own Activity with the stable
`activity:gitlab:issue:...` key, and the two converge on the entity_keys.

## Source-priority rationale (why)

Commit messages, branch names, and GitLab issue labels are author-applied at
intent time. MR / issue titles are author-stated headlines. Bodies carry
stated rationale. Discussion threads and approval notes are context, not
standalone facts. Code diffs and full note threads are voluminous and require
inference. Reading code burns budget on rediscovering intent the author
already encoded in 1-2 lines elsewhere. Stop climbing the priority ladder as
soon as you can answer kind + summary + bug/decision evidence.

## Bounds and budget

- **Two** bounded list calls only — one `gitlab_list_merge_requests` and one
  `gitlab_list_issues`. No pagination beyond what the bounded tools return.
- Soft tool-call cap: `40 + 8 × count` (two kinds, each item averages
  ~4 calls when discussions / diff are unused). Plan accordingly.
- If you approach the cap with a coherent recent subset of items ingested
  cleanly, FINISH — do not partially ingest an item. The tail can be re-run
  later with a smaller `count` (stable keys mean already-ingested items will
  be deduplicated, not duplicated).

## Anti-patterns

- Do NOT re-emit `repository.added` from this skill.
- Do NOT walk the file tree or scan source files. Read MR code diffs only as
  the last resort described above, when commit + branch + title + body +
  discussions all leave intent unclear.
- Do NOT page past the bounded list calls.
- Do NOT confuse an MR iid with an issue iid — `!7` and `#7` are different
  artifacts in the same project and get different Activity keys.
- Do NOT assume `gitlab.com`. Read the instance host off the item's `url`;
  self-managed CE is the common case.
- Do NOT emit `Fix` for an issue filing. Fix is for merged MRs only.
- Do NOT treat an unresolved discussion thread as an accepted decision, or an
  assigned reviewer as an approver — approval is recorded in the approvals
  payload and the "approved this merge request" system note.
- Do NOT auto-close any open issue or existing BugPattern based
  on an MR merge or issue closure alone — that is evidence, not closure.
- Do NOT invent BugPatterns, Decisions, Services, or Persons not actually
  evidenced in the data you read. Emit a warning record instead.
- Do NOT ingest epics, wikis, or snippets — unsupported (no connector tool).

## Single-event contract

This skill, when invoked by the internal agent, runs as a single
`(gitlab, repository, one_shot_ingest)` event. Pass that ONE `event_id` to
every `apply_graph_mutations` call and to the final `mark_event_processed` —
per-artifact identity is the entity_key (`activity:gitlab:mr:...` /
`activity:gitlab:issue:...`), not the event id, so multiple Activities under
one event id is correct.

When invoked by Claude Code outside the event pipeline, there is no
internal-agent event state, so the internal `apply_graph_mutations` tool will
reject an empty or invented event id. Use this document as the extraction
procedure only when the host provides a compatible context-graph write path
and a valid event/provenance id. Otherwise stop after producing the proposed
plan; do not pretend to apply it.
