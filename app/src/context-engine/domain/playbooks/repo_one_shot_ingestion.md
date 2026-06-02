---
name: repo-one-shot-ingestion
description: One-time ingestion of a repository's recent merged PRs and standalone GitHub issues into the context graph. Not incremental — live updates go through the pull_request.merged and issue.opened webhook paths.
source_system: github
event_type: repository
action: one_shot_ingest
enables_planner: true
---

# Repo one-shot ingestion

A reusable skill for ingesting a repository's recent **merged pull requests**
and **standalone GitHub issues** into the context graph in a single pass.
Sibling to `linear_team_one_shot_ingestion`: same enumerate → batch → hydrate
shape, GitHub source. Designed to be invoked by either Claude Code (as a
checklist with a compatible write path) or the internal reconciliation agent
(loaded as a playbook).

Does **not** replace `repository.added` — no sandbox file-tree walk, no
module / feature map. Does **not** cover GitHub Discussions (no connector
tool yet).

## When to invoke

- A user wants to seed the context graph from a repo's recent **PR merge
  history + issue history** in one pass.
- The repo is already attached to the target pot (so tool calls are scoped).
- You will NOT run this skill repeatedly against the same repo — incremental
  updates flow through the live `github / pull_request / merged` and
  `github / issue / opened` webhooks, which write the same Activity keys so
  a future webhook converges with what this skill already wrote.

## Inputs

- `repo`: `owner/name` (required). Must be attached to the active pot.
- `count`: soft per-kind list limit. Default `50`. When the host submits this
  event, read `count` from `event.payload.count`. Pass it as `limit` on
  **both** `github_list_pull_requests` and `github_list_issues`. The server
  still clamps to `CONTEXT_ENGINE_BACKFILL_MAX_ITEMS` (default 300) and the
  trailing window (`CONTEXT_ENGINE_BACKFILL_WINDOW_DAYS`, default 365 days).
  Hard ceiling: respect whatever each bounded list tool returns — do not
  page past it.
- `batch_size`: how many items to mark as one todo. Default `10`.
- `parallel_per_batch` (`K`): how many items to hydrate at once within a
  batch. Default `5`. Drop lower only after a fetched item proves unusually
  expensive to hydrate (large body / comment volume or an optional diff
  fetch).
- `event_id`: required for the internal reconciliation agent. The single
  `(github, repository, one_shot_ingest)` event id for the run.

## Tools assumed available

- `github_list_pull_requests(repo, limit=count)` — bounded PR refs, newest
  first. **ONE call.** PRs only.
- `github_list_issues(repo, limit=count)` — bounded issue refs, newest first.
  **PRs excluded** by the adapter (the list tool already filters PR-shaped
  rows out). **ONE call.**
- `github_get_pull_request(repo, pr_number)` — full PR metadata (no diff by
  default). Same tool accepts optional `include_diff=true`; when set, the
  response also includes a `files` array of per-file objects with `filename`,
  `status`, `additions`, `deletions`, and `patch` (use only as a last resort;
  see Phase 2).
- `github_get_pull_request_commits(repo, pr_number)` — commit messages.
- `github_get_pull_request_review_comments(repo, pr_number)` — inline review.
- `github_get_pull_request_issue_comments(repo, pr_number)` — PR conversation
  (NOT the same as standalone issue comments — different surface entirely).
- `github_get_issue(repo, issue_number)` — title, body, state, author,
  labels, `created_at`, `updated_at`, `url`. There is **no** separate issue
  comments tool — use body + labels; do not invent discussion you did not
  fetch.
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

1. Confirm the repo is attached (`sandbox_list_repos` if the tool exists).
   If not, abort with a warning. Do NOT attempt to attach from this skill.
2. Initialize the todo list with two entries:
   - `Enumerate last <count> merged PRs of <repo>`
   - `Enumerate last <count> issues of <repo>`

### Phase 1 — Enumerate (two list calls, one each)

1. Call `github_list_pull_requests(repo, limit=count)` ONCE. Bounded
   server-side. Filter to merged PRs (`merged=true` on refs); skip
   closed-unmerged with a warning.
2. Call `github_list_issues(repo, limit=count)` ONCE. Standalone issues
   only (the list tool already excludes PR-shaped rows).
3. Drain order: **merged PRs first** (completed work timeline), then
   **issues newest-first** (bugs, feature requests, questions).
4. Split returned refs into batches of `batch_size`. For each batch, append
   a todo:
   - PRs: `Process PRs [#a, #b, ...]`
   - issues: `Process issues [#a, #b, ...]`
5. Use `update_todo_status` or `write_todos` to mark each enumeration todo
   done.

### Phase 2 — Drain batches

Drain todos sequentially across kinds (PRs first, then issues); within a
batch parallelize up to `K`.

#### PR items

For each batch of PR todos, in parallel for `K` PRs at a time:

1. `github_get_pull_request(repo, n)` — title, body, author, merged_at,
   `head_branch`, `base_branch`, labels, milestone, and URL. Do NOT assume
   `head_ref`, `merge_commit_sha`, `changed_files`, or diff-size fields
   exist in this response.
2. Read source signals in PRIORITY ORDER, stopping when intent is clear:
   1. **Commit messages** (`github_get_pull_request_commits`) — concise,
      declarative, often conventional-commit prefixed.
   2. **Branch name** (`head_branch`) — `feat/...`, `fix/...`, `chore/...`
      carry intent.
   3. **PR title** — author-stated headline.
   4. **PR description / body** — author rationale; check for
      `Why:` / `Fixes #` / `Closes #` / linked issues.
   5. **Review / PR comments** — only if higher signals are ambiguous.
   6. **Code diff** — LAST RESORT. If needed, call
      `github_get_pull_request(repo, n, include_diff=true)` (optional
      `include_diff` boolean on the same tool as step 1). The response adds
      a `files` array; each element is an object with `filename`, `status`,
      `additions`, `deletions`, and `patch` — use `filename` for path-based
      reasoning, not a separate path/id field. Reading patches burns budget
      rediscovering intent the author already wrote.
3. Classify:
   - **Author handle(s)** — primary author + co-authors (from
     `Co-authored-by:` trailers).
   - **Kind** — `feat | fix | chore | refactor | docs | test | other` from
     conventional commit prefix, branch prefix, or title.
   - **Summary** — 1-2 sentence functional summary.
   - **Bug evidence** — does the PR fix a bug? capture symptom signature;
     emit Fix + BugPattern only when symptom is clear.
   - **Decision evidence** — body explicitly documents rationale +
     alternatives_rejected (most PRs do NOT — be conservative).

#### Issue items

For each batch of issue todos, in parallel for `K` issues at a time:

1. `github_get_issue(repo, n)` — title, body, state, author, labels,
   `created_at`, `updated_at`, `url`.
2. Read source signals in PRIORITY ORDER, stopping when intent is clear:
   1. **Labels** — `bug` / `enhancement` / `documentation` / `question` are
      the highest-signal kind classifier — author-applied and standardized
      per repo.
   2. **State** — `open` vs `closed`. Do NOT treat closed as auto-resolved;
      closure can mean fixed, won't-fix, or duplicate.
   3. **Title** — author-stated headline.
   4. **Body** — rationale, repro steps, `Why:` / linked PRs / linked
      issues. If body is empty, summarize from title + labels only.
   5. There is **no** separate issue-comments tool — do not invent
      discussion you did not fetch.
3. Classify:
   - **Reporter** — `author` from `github_get_issue`.
   - **Kind** — `bug | feat | chore | question | docs | other` from labels
     first, then title.
   - **Summary** — 1-2 sentence summary of what was reported or requested.
   - **Bug report** (open or closed) — capture symptom signature for
     BugPattern. **Do NOT emit `Fix` from an issue** — Fix is reserved for
     merged PRs that shipped a fix.
   - **Decision** — only when the body explicitly documents rationale +
     alternatives (rare on issues).

Build one `LlmReconciliationPlan`-shaped object for the batch (see Mutations
section). Call `apply_graph_mutations(plan, event_id, summary)` once per
batch. Use `update_todo_status` or `write_todos` to mark each batch todo
done.

### Phase 3 — Finalize

1. When all todos are drained (or you've hit the tool-call budget with a
   coherent subset complete), tally:
   - PRs ingested / skipped
   - Issues ingested by state (open / closed) and by kind
   - Distinct authors / reporters
   - Fix nodes emitted (PR-only), BugPattern nodes emitted, Decision
     nodes emitted
2. `mark_event_processed(event_id, summary)` then `finish_batch(summary)`.

## Mutations (per item)

Use the existing ontology. Stable keys ensure backfill + future webhook
converge. Key formats below follow `domain.identity.mint_entity_key` rules
(see `domain.ontology.ENTITY_TYPES`).

Identity rules to respect (these are NOT free-form strings):

- `Repository` is `SLUG_ALIAS` with `key_prefix=repo`. The body must be a
  lowercase slug — letters/digits/hyphens only — NO dots, NO slashes. So
  `repo:github.com/acme/api` is INVALID. Use `repo:<owner>-<repo>` (slugified,
  matches what `_repo_key` in scanners produces, e.g. `repo:acme-api`).
- `Person` is `SLUG_ALIAS` with `key_prefix=person`. Use `person:<handle>`
  (the GitHub login lowercased; usually already slug-clean).
- `Period` uses the production builder `timeline:period:daily:<pot>:<yyyy-mm-dd>`
  (matches `adapters/outbound/reconciliation/timeline_plan.py::_period_key`).
- `Activity` is `EXTERNAL_ID` with `key_prefix=activity`. Two distinct forms:
  - PR: `activity:github:pr:<owner>/<repo>:<n>`
  - Issue: `activity:github:issue:<owner>/<repo>:<n>`
  Segments after `activity:` may contain `/` per `_EXTERNAL_ID_SAFE_RE`.
  Lowercase the owner/repo segment. Write this key directly as a string in
  your JSON `entity_key`. If you ever route through `mint_entity_key`, pass
  the issue/PR number as `external_id` and the path as
  `extra_segments=("github","pr","<owner>/<repo>")` (or `"issue"`) — do NOT
  pass the full colon-joined string as a single `external_id`, because
  `_normalize_external_id` strips colons and the key collapses to
  `activity:github-pr-<owner>-<repo>-<n>`.
- `Fix` and `Decision` are `CONTENT_HASH`. The body must be a 12-hex
  fingerprint of canonical content — NEVER encode the PR / issue number into
  the key. Use `fix:<12-hex-sha256>` and `decision:<12-hex-sha256>` (mint
  via `mint_entity_key(spec, content=<stable-string>)` or hash inline).
- `BugPattern` is `SLUG_ALIAS` with `key_prefix=bug_pattern`. Use
  `bug_pattern:<repo-slug>:<symptom-slug>` (e.g.
  `bug_pattern:acme-api:db-timeout`) — each colon-separated segment must be
  a valid slug.

### Always emit (endpoint entities, at least once per batch)

- **Entity** `Repository`
  - key: `repo:<owner>-<repo>` (slugified). Prefer reusing the existing
    Repository entity_key already in the graph if the read path lets you
    look it up by `name=<owner>/<repo>`.
  - labels: `["Entity", "Repository"]`.
  - properties: `name="<owner>/<repo>"`, `provider="github"`,
    `provider_host="github.com"`, `owner=<owner>`, `repo=<repo>`.
- **Entity** `Period` — one per distinct activity date in the batch.
  - key: `timeline:period:daily:<pot>:<yyyy-mm-dd>`.
  - labels: `["Entity", "Period"]`.
  - properties: `period_kind="daily"`, `date="<yyyy-mm-dd>"`.

### Per merged PR — always emit

- **Entity** `Activity`
  - key: `activity:github:pr:<owner>/<repo>:<n>` (owner/repo lowercased).
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<merged_at>`, `verb_class="pr_merged"`,
    `title=<pr_title>`, `summary=<your 1-2 sentence summary>`,
    `head_branch`, `base_branch`, `kind` (feat/fix/...), `pr_url`.
- **Entity** `Person` — one per author / co-author.
  - key: `person:<handle-lowercased>`.
  - labels: `["Entity", "Person"]`.
- **Edge** `PERFORMED` — `person:<primary_author>` → activity key.
- **Edge** `AUTHORED` — `person:<co_author>` → activity key, per co-author.
- **Edge** `TOUCHED` — activity key → repository key.
- **Edge** `IN_PERIOD` — activity key → period key.

### Per merged PR — conditionally emit

- **Bug fix** (kind=`fix` AND PR body / linked issue has a clear symptom):
  - **Entity** `BugPattern`
    - key: `bug_pattern:<repo-slug>:<symptom-slug>` (segments slug-valid).
    - labels: `["Entity", "BugPattern"]`.
    - properties: `symptom_signature=<short canonical sentence>`,
      `name=<symptom title>`.
  - **Entity** `Fix`
    - key: `fix:<12-hex-sha256>` minted from a stable canonical string such
      as `"github:pr:<owner>/<repo>:<n>|fix|<symptom-signature>"`.
    - labels: `["Entity", "Fix"]`.
    - properties: `fix_steps=<short description>`,
      `verification_status="unverified"`, `source_pr=<activity_key>`.
  - **Edge** `RESOLVED` — fix key → bug_pattern key.
  - **Edge** `REPRODUCES` — bug_pattern key → repository key.
- **Design decision** (PR body explicitly documents rationale + alternatives):
  - **Entity** `Decision`
    - key: `decision:<12-hex-sha256>` from
      `"github:pr:<owner>/<repo>:<n>|decision|<title>"`.
    - labels: `["Entity", "Decision"]`.
    - properties: `name=<short title>`, `rationale=<stated rationale>`,
      `alternatives_rejected=<list or string>`, `source_pr=<activity_key>`.
  - **Edge** `DECIDED` — decision key → repository key.
  - **Edge** `AFFECTS` — decision key → repository key.

Touched services (optional, only if obvious): if an optional
`github_get_pull_request(repo, n, include_diff=true)` fetch returns `files`,
inspect each entry's `filename` (repo-relative path). When those paths
clearly map to an existing `Service` entity (e.g. every changed `filename`
under `services/auth/`), emit an extra `TOUCHED` edge activity → that
service. Do NOT invent Services that don't already exist in the graph.

### Per issue — always emit

- **Entity** `Activity`
  - key: `activity:github:issue:<owner>/<repo>:<n>` (owner/repo lowercased).
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<created_at>`,
    `verb_class="github_issue_<state>"` where `<state>` is `open` or
    `closed` — these are property VALUES, not tool names; `title`,
    `summary=<your 1-2 sentence summary>`, `state`, `kind`
    (bug/feat/chore/question/...), `issue_url`.
- **Entity** `Person` — the reporter (issue author).
  - key: `person:<handle-lowercased>`.
  - labels: `["Entity", "Person"]`.
- **Edge** `PERFORMED` — `person:<reporter>` → activity key.
- **Edge** `TOUCHED` — activity key → repository key.
- **Edge** `IN_PERIOD` — activity key → period key.

### Per issue — conditionally emit

- **Bug report** (labels include `bug` AND body or title carries a clear
  symptom):
  - **Entity** `BugPattern`
    - key: `bug_pattern:<repo-slug>:<symptom-slug>`.
    - labels: `["Entity", "BugPattern"]`.
    - properties: `symptom_signature=<short canonical sentence>`,
      `name=<symptom title>`, `source_issue=<activity_key>`.
  - **Edge** `REPRODUCES` — bug_pattern key → repository key.
  - **Do NOT emit `Fix`** — Fix is reserved for merged PRs that shipped a
    fix. Closing an issue is not evidence that a fix exists.
- **Design decision** (issue body explicitly documents rationale +
  alternatives — rare on issues, common on spec / RFC issues):
  - **Entity** `Decision`
    - key: `decision:<12-hex-sha256>` from
      `"github:issue:<owner>/<repo>:<n>|decision|<title>"`.
    - labels: `["Entity", "Decision"]`.
    - properties: `name=<short title>`, `rationale=<stated rationale>`,
      `alternatives_rejected=<list or string>`,
      `source_issue=<activity_key>`.
  - **Edge** `DECIDED` — decision key → repository key.
  - **Edge** `AFFECTS` — decision key → repository key.

When a PR you ingested references `Fixes #n` or `Closes #n` in its body,
prefer recording the linkage in the PR Activity's `properties` (e.g. a
`closes_issues` list) or the plan's `evidence` array, rather than
duplicating an issue Activity inside the PR pass — the issue pass will emit
its own Activity with the stable `activity:github:issue:...` key, and the
two converge on the entity_keys.

## Source-priority rationale (why)

Commit messages, branch names, and GitHub issue labels are author-applied at
intent time. PR / issue titles are author-stated headlines. Bodies carry
stated rationale. Review threads / linked discussion are context, not
standalone facts. Code diffs and full comment threads are voluminous and
require inference. Reading code burns budget on rediscovering intent the
author already encoded in 1-2 lines elsewhere. Stop climbing the priority
ladder as soon as you can answer kind + summary + bug/decision evidence.

## Bounds and budget

- **Two** bounded list calls only — one `github_list_pull_requests` and one
  `github_list_issues`. No pagination beyond what the bounded tools return.
- Soft tool-call cap: `40 + 8 × count` (two kinds, each item averages
  ~4 calls when comments / diff are unused). Plan accordingly.
- If you approach the cap with a coherent recent subset of items ingested
  cleanly, FINISH — do not partially ingest an item. The tail can be re-run
  later with a smaller `count` (stable keys mean already-ingested items will
  be deduplicated, not duplicated).

## Anti-patterns

- Do NOT re-emit `repository.added` from this skill — that re-triggers the
  agent's full sandbox bootstrap.
- Do NOT walk the file tree or read source files unless commit + branch +
  title + body + comments all leave intent unclear.
- Do NOT page past the bounded list calls.
- Do NOT ingest the same PR number twice — the issues list tool already
  excludes PR-shaped rows; trust it.
- Do NOT emit `Fix` for an issue filing. Fix is for merged PRs only.
- Do NOT auto-close any existing Incident or open issue / BugPattern based
  on a PR merge or issue closure alone — that is evidence, not closure.
- Do NOT invent BugPatterns, Decisions, Services, or Persons not actually
  evidenced in the data you read. Emit a warning record instead.
- Do NOT ingest GitHub Discussions — unsupported (no connector tool).
- Do NOT invent issue comments — there is no separate issue-comments tool;
  if you need discussion context for an issue, the body + labels are all
  you get.

## Single-event contract

This skill, when invoked by the internal agent, runs as a single
`(github, repository, one_shot_ingest)` event. Pass that ONE `event_id` to
every `apply_graph_mutations` call and to the final `mark_event_processed` —
per-artifact identity is the entity_key (`activity:github:pr:...` /
`activity:github:issue:...`), not the event id, so multiple Activities under
one event id is correct.

When invoked by Claude Code outside the event pipeline, there is no
internal-agent event state, so the internal `apply_graph_mutations` tool will
reject an empty or invented event id. Use this document as the extraction
procedure only when the host provides a compatible context-graph write path
and a valid event/provenance id. Otherwise stop after producing the proposed
plan; do not pretend to apply it.
