---
name: repo-one-shot-ingestion
description: One-time ingestion of a repository's recent merged PR history into the context graph. Not incremental — live updates go through the pull_request.merged webhook path.
source_system: github
event_type: repository
action: one_shot_ingest
enables_planner: true
---

# Repo one-shot ingestion

A reusable skill for ingesting the last N merged PRs of a repository into the
context graph in a single pass. Designed to be invoked by either Claude Code
(as a checklist with a compatible write path) or the internal reconciliation
agent (loaded as a playbook).

## When to invoke

- A user wants to seed the context graph from a repo's recent PR history in
  one pass.
- The repo is already attached to the target pot (so tool calls are scoped).
- You will NOT run this skill repeatedly against the same repo — incremental
  updates flow through the live `github / pull_request / merged` webhook path,
  which writes the same Activity keys so a future webhook converges with what
  this skill already wrote.

## Inputs

- `repo`: `owner/name` (required). Must be attached to the active pot.
- `count`: total recent PRs to ingest. Default `50`. Hard ceiling: respect
  whatever the bounded list tool returns — do not page past it.
- `batch_size`: how many PRs to mark as one todo. Default `10`.
- `parallel_per_batch` (`K`): how many PRs to hydrate at once within a batch.
  Default `5`. Drop lower only after a fetched PR proves unusually expensive
  to hydrate (large body/comment volume or an optional diff fetch).
- `event_id`: required for the internal reconciliation agent. This is the
  single `(github, repository, one_shot_ingest)` event id for the run.

## Tools assumed available

- `github_list_pull_requests(repo)` — bounded enumeration (window + cap, newest
  first). ONE call.
- `github_get_pull_request(repo, pr_number)` — full PR metadata (no diff by
  default). Same tool accepts optional `include_diff=true`; when set, the
  response also includes a `files` array of per-file objects with `filename`,
  `status`, `additions`, `deletions`, and `patch` (use only as a last resort;
  see Phase 2).
- `github_get_pull_request_commits(repo, pr_number)` — commit messages.
- `github_get_pull_request_review_comments(repo, pr_number)` — inline review.
- `github_get_pull_request_issue_comments(repo, pr_number)` — discussion.
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
- `mark_event_processed(event_id, summary)` + `finish_batch(summary)` — completion.

## Procedure

### Phase 0 — Setup

1. Confirm the repo is attached (`sandbox_list_repos` if the tool exists). If
   not, abort with a warning. Do NOT attempt to attach from this skill.
2. Initialize the todo list with one entry: `Enumerate last <count> PRs of <repo>`.

### Phase 1 — Enumerate (one call)

1. Call `github_list_pull_requests(repo)` ONCE. The tool is bounded server-side
   to a trailing window + item cap — never page beyond what it returns.
2. Filter to merged PRs (skip closed-unmerged).
3. Split the returned refs into batches of `batch_size` (default 10). For each
   batch, append a todo: `Process PRs [#a, #b, ...]`.
4. Use `update_todo_status` or `write_todos` to mark the enumeration todo done.

### Phase 2 — Drain batches

For each batch todo (process them sequentially; within a batch parallelize):

1. Choose `K` (`parallel_per_batch`, default 5). If tool results are large or
   slow, reduce `K` for the remaining PRs in this batch.
2. In parallel for `K` PRs at a time:
   a. `github_get_pull_request(repo, n)` — title, body, author, merged_at,
      `head_branch`, `base_branch`, labels, milestone, and URL. Do NOT assume
      `head_ref`, `merge_commit_sha`, `changed_files`, or diff-size fields
      exist in this response.
   b. Read source signals in PRIORITY ORDER, stopping when intent is clear:
      1. **Commit messages** (`github_get_pull_request_commits`) — concise,
         declarative, often conventional-commit prefixed.
      2. **Branch name** (`head_branch`) — `feat/...`, `fix/...`, `chore/...`
         carry intent.
      3. **PR title** — author-stated headline.
      4. **PR description / body** — author rationale; check for
         `Why:` / `Fixes #` / `Closes #` / linked issues.
      5. **Review / issue comments** — only if higher signals are ambiguous.
      6. **Code diff** — LAST RESORT. If needed, call
         `github_get_pull_request(repo, n, include_diff=true)` (optional
         `include_diff` boolean on the same tool as step 2a). The response adds
         a `files` array; each element is an object with `filename`, `status`,
         `additions`, `deletions`, and `patch` — use `filename` for path-based
         reasoning, not a separate path/id field. Reading patches burns budget
         rediscovering intent the author already wrote.
   c. Classify the PR:
      - **Author handle(s)** — primary author + co-authors (from
        `Co-authored-by:` trailers).
      - **Kind** — `feat | fix | chore | refactor | docs | test | other` from
        conventional commit prefix, branch prefix, or title.
      - **Summary** — 1-2 sentence functional summary (what changed in user-
        visible terms).
      - **Bug evidence** — does the PR fix a bug? If yes, capture the symptom
        signature (from PR body or referenced issue title) and whether a
        BugPattern key can be derived.
      - **Decision evidence** — does the PR body explicitly state a design
        choice with rationale and/or alternatives_rejected? Most PRs do NOT —
        only emit Decision when the body actually documents the decision.

3. Build one `LlmReconciliationPlan`-shaped object for the batch (see Mutations
   section). Call `apply_graph_mutations(plan, event_id, summary)` once per
   batch.
4. Use `update_todo_status` or `write_todos` to mark the batch todo done. Move
   to the next batch.

### Phase 3 — Finalize

1. When all todos are drained (or you've hit the tool-call budget with a
   coherent subset complete), tally:
   - PRs ingested
   - Distinct authors
   - Bug fixes (Fix nodes emitted)
   - Decisions emitted
   - PRs skipped (and why)
2. `mark_event_processed(event_id, summary)` then `finish_batch(summary)`.

## Mutations (per PR)

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
- `Activity` is `EXTERNAL_ID` with `key_prefix=activity`. Use
  `activity:github:pr:<owner>/<repo>:<n>` (segments after `activity:` may
  contain `/` per `_EXTERNAL_ID_SAFE_RE`). Lowercase the owner/repo segment.
  Write this key directly as a string in your JSON `entity_key`. If you ever
  route through `mint_entity_key`, pass the PR number as `external_id` and
  the path as `extra_segments=("github","pr","<owner>/<repo>")` — do NOT
  pass the full colon-joined string as a single `external_id`, because
  `_normalize_external_id` strips colons and the key collapses to
  `activity:github-pr-<owner>-<repo>-<n>`.
- `Fix` and `Decision` are `CONTENT_HASH`. The body must be a 12-hex
  fingerprint of canonical content — NEVER encode the PR number into the key.
  Use `fix:<12-hex-sha256>` and `decision:<12-hex-sha256>` (mint via
  `mint_entity_key(spec, content=<stable-string>)` or hash inline).
- `BugPattern` is `SLUG_ALIAS` with `key_prefix=bug_pattern`. Use
  `bug_pattern:<repo-slug>:<symptom-slug>` (e.g. `bug_pattern:acme-api:db-timeout`)
  — each colon-separated segment must be a valid slug.

### Always emit (endpoint entities, at least once per batch)

- **Entity** `Repository`
  - key: `repo:<owner>-<repo>` (slugified). Prefer reusing the existing
    Repository entity_key already in the graph if the read path lets you
    look it up by `name=<owner>/<repo>`.
  - labels: `["Entity", "Repository"]`.
  - properties: `name="<owner>/<repo>"`, `provider="github"`,
    `provider_host="github.com"`, `owner=<owner>`, `repo=<repo>`.
- **Entity** `Period` — one per distinct PR-merge date in the batch.
  - key: `timeline:period:daily:<pot>:<yyyy-mm-dd>` (pot id is available from
    the agent's run state / event scope).
  - labels: `["Entity", "Period"]`.
  - properties: `period_kind="daily"`, `date="<yyyy-mm-dd>"`.

### Always emit per PR

- **Entity** `Activity`
  - key: `activity:github:pr:<owner>/<repo>:<n>` (owner/repo lowercased).
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<merged_at>`, `verb_class="pr_merged"`,
    `title=<pr_title>`, `summary=<your 1-2 sentence functional summary>`,
    `head_branch`, `base_branch`, `kind` (one of feat/fix/chore/refactor/
    docs/test/other), `pr_url`.
- **Entity** `Person` — one per author / co-author.
  - key: `person:<handle-lowercased>`.
  - labels: `["Entity", "Person"]`.
  - properties: `name=<handle>`, `handle=<handle>`.
- **Edge** `PERFORMED` — `person:<primary_author>` → activity key.
- **Edge** `AUTHORED` — `person:<co_author>` → activity key, per co-author.
- **Edge** `TOUCHED` — activity key → repository key.
- **Edge** `IN_PERIOD` — activity key → period key.

### Conditionally emit (only when evidenced)

- **Bug fix** (kind=`fix` AND PR body / linked issue has a clear symptom):
  - **Entity** `BugPattern`
    - key: `bug_pattern:<repo-slug>:<symptom-slug>` (segments must be slugs).
    - labels: `["Entity", "BugPattern"]`.
    - properties: `symptom_signature=<short canonical sentence>`,
      `name=<symptom title>`.
  - **Entity** `Fix`
    - key: `fix:<12-hex-sha256>` minted from a stable canonical string such
      as `"github:pr:<owner>/<repo>:<n>|fix|<symptom-signature>"`. Do NOT
      put PR identifiers in the key body — they go in `properties`.
    - labels: `["Entity", "Fix"]`.
    - properties: `fix_steps=<short description>`,
      `verification_status="unverified"`, `source_pr=<activity_key>`.
  - **Edge** `RESOLVED` — fix key → bug_pattern key.
  - **Edge** `REPRODUCES` — bug_pattern key → repository key.
- **Design decision** (PR body explicitly documents rationale + alternatives):
  - **Entity** `Decision`
    - key: `decision:<12-hex-sha256>` minted from a stable canonical string
      such as `"github:pr:<owner>/<repo>:<n>|decision|<title>"`.
    - labels: `["Entity", "Decision"]`.
    - properties: `name=<short title>`, `rationale=<stated rationale>`,
      `alternatives_rejected=<list or string>`, `source_pr=<activity_key>`.
  - **Edge** `DECIDED` — decision key → repository key (or a Service key if
    the PR clearly touches a single known Service).
  - **Edge** `AFFECTS` — decision key → repository key.

Touched services (optional, only if obvious): if an optional
`github_get_pull_request(repo, n, include_diff=true)` fetch returns `files`,
inspect each entry's `filename` (repo-relative path). When those paths clearly
map to an existing `Service` entity (e.g. every changed `filename` under
`services/auth/`), emit an extra `TOUCHED` edge activity → that service. Do
NOT invent Services that don't already exist in the graph.

## Source-priority rationale (why)

Commit messages and branch names are concise, structured, and produced by the
author at intent-time. PR title and description are author-stated rationale.
Code diffs are voluminous and require inference. Reading code burns budget on
rediscovering intent that the author already wrote in 1-2 lines elsewhere.
Stop climbing the priority ladder as soon as you can answer kind + summary +
bug/decision evidence.

## Bounds and budget

- ONE `github_list_pull_requests` call. No pagination.
- Soft tool-call cap: `30 + 5 * count`. Plan accordingly.
- If you approach the cap with a coherent recent subset of PRs ingested
  cleanly, FINISH — do not partially ingest a PR. The tail can be re-run later
  by invoking this skill with a smaller `count` (but stable keys mean already-
  ingested PRs will be deduplicated, not duplicated).

## Anti-patterns

- Do NOT re-emit `repository.added` from this skill — that re-triggers the
  agent's full bootstrap.
- Do NOT walk the file tree or read source files unless commit + branch +
  title + body + comments all leave intent unclear.
- Do NOT page past the bounded list call.
- Do NOT invent BugPatterns, Decisions, Services, or Persons not actually
  evidenced in the PR data you read. Emit a warning record instead.
- Do NOT auto-close / auto-resolve any existing Incident or open issue based
  on a PR merge alone — the PR is evidence, not closure.
- Do NOT run this skill on a repo that already has live webhook ingestion
  going against the SAME date window — let the webhook handle live updates.

## Single-event contract

This skill, when invoked by the internal agent, runs as a single
`(github, repository, one_shot_ingest)` event. Pass that ONE `event_id` to
every `apply_graph_mutations` call and to the final `mark_event_processed` —
per-PR identity is the entity_key (`activity:github:pr:...`), not the event id,
so multiple Activities under one event id is correct.

When invoked by Claude Code outside the event pipeline, there is no
internal-agent event state, so the internal `apply_graph_mutations` tool will
reject an empty or invented event id. Use this document as the extraction
procedure only when the host provides a compatible context-graph write path and
a valid event/provenance id. Otherwise stop after producing the proposed plan;
do not pretend to apply it.
