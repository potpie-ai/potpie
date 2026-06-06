---
name: github-repo-one-shot-ingestion
description: One-time ingestion of a GitHub repo's recent commits, pull requests, and issues into the context graph. Not incremental — live updates go through the live GitHub webhook path.
source_system: github
event_type: github_repo
action: one_shot_ingest
enables_planner: true
---

# GitHub repo one-shot ingestion

A reusable skill for ingesting a GitHub repository's recent commits, pull requests, and
issues into the context graph in a single pass. Sibling to the Linear
`linear_team_one_shot_ingestion` playbook: same shape, different source. Designed to
be invoked by either Claude Code (as a checklist with a compatible write path) or the
internal reconciliation agent (loaded as a playbook).

## When to invoke

- A user wants to seed the context graph from a GitHub repo's recent history in one pass.
- The GitHub repository is already attached to the target pot (so connector tools are
  scoped to the right credentials).
- You will NOT run this skill repeatedly against the same repo window — incremental
  updates flow through live GitHub merged-PR webhook events, which write the same
  Activity keys so a future webhook converges with what this skill already wrote.

## Inputs

- `repo`: GitHub repository as `owner/repo` (required). Must be attached to the active
  pot.
- `count`: soft per-kind list limit. Default `120` (chosen so the soft tool-call cap
  below stays under the playbook's `max_tool_calls=400`). Read `count` from
  `event.payload.count` and pass it as `limit` on each list tool. Hard ceiling: respect
  whatever each bounded list tool returns — do not page past it.
- `batch_size`: items per todo. Default `10`.
- `parallel_per_batch` (`K`): items to hydrate in parallel per batch. Default `5`. Drop
  lower if hydrated items prove unusually large (long PR bodies or many review comments).
- `event_id`: required for the internal reconciliation agent. The single
  `(github, github_repo, one_shot_ingest)` event id for the run.

## Tools assumed available

- `github_list_commits(repo=repo, limit=count)` — bounded enumeration of commit refs
  `{sha, message, author_login, committed_at}`, newest-first. ONE call.
- `github_list_pull_requests(repo=repo, state="closed", limit=count)` — bounded
  enumeration of PR refs `{number, title, state, merged, merged_at, user_login}`,
  newest-first. ONE call.
- `github_list_issues(repo=repo, state="closed", limit=count)` — bounded enumeration
  of issue refs `{number, title, state, labels, closed_at}`, newest-first. ONE call.
- `github_get_commit(repo, sha)` — full commit payload: message, author, committer,
  stats, changed file list.
- `github_get_pull_request(repo, pr_number)` — full PR payload: title, body, labels,
  merged_at, user, linked issue references, commit count.
- `github_get_issue(repo, issue_number)` — full issue payload: title, body, labels,
  state, assignee, creator, linked PRs.
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
  REQUIRED. The todo list rides in the agent's message history and is checkpointed;
  a resumed run continues the existing list instead of re-enumerating.
- `mark_event_processed(event_id, summary)` + `finish_batch(summary)` — completion.

## Procedure

### Phase 0 — Setup

1. Trust the event payload that the repo is attached to the pot. If a list tool returns
   an auth / not-connected error, abort with a warning. Do NOT attempt to connect the
   repository from this skill.
2. Initialize the todo list with three entries:
   - `Enumerate <repo> commits`
   - `Enumerate <repo> pull requests`
   - `Enumerate <repo> issues`

### Phase 1 — Enumerate (three list calls, one each)

1. Call `github_list_commits(repo=repo, limit=count)` ONCE. Bounded server-side.
2. Call `github_list_pull_requests(repo=repo, state="closed", limit=count)` ONCE.
3. Call `github_list_issues(repo=repo, state="closed", limit=count)` ONCE.
4. Drain order: commits first (they anchor the timeline spine), then PRs (they frame
   code-delivery units and are the source of `Fix` entities), then issues newest-first
   (they provide intent and bug evidence).
5. For each returned ref, append a todo:
   `Process github <kind> <id>` (e.g. `Process github commit abc123`).
6. Mark each enumeration todo done via `update_todo_status` or `write_todos`.

### Phase 2 — Drain batches

Drain todos sequentially across kinds (commits, then PRs, then issues); within a
batch parallelize up to `K`.

1. Choose `K` (`parallel_per_batch`, default 5). Reduce if hydrated payloads are large
   (long PR bodies with many review comments).
2. In parallel for `K` items at a time, hydrate via the matching `get_*` tool:

   **COMMIT** — `github_get_commit(repo, sha)`:
   - Skip merge commits — messages that start with `"Merge pull request #"` or
     `"Merge branch '"` are housekeeping entries that duplicate the PR Activity
     already emitted from the PR phase. Append a warning and do NOT emit an Activity
     for them.
   - Skip bot authors — logins ending in `[bot]` (e.g. `dependabot[bot]`,
     `renovate[bot]`) produce low-signal Activity nodes. Append a warning and skip.
   - Read signals in PRIORITY ORDER, stopping when intent is clear:
     1. **Commit message prefix** — conventional-commit prefix (`fix:`, `feat:`,
        `chore:`, `refactor:`, `docs:`) is the highest-signal kind classifier; it's
        author-applied and standardized per project.
     2. **Author login / email** — the actor for the `PERFORMED` edge.
     3. **Changed file list** — scope signal (which components or services are touched).
     4. **Stats** (additions / deletions) — scale signal.
   - Classify: kind (fix/feat/chore/refactor/docs/other from prefix, fallback from
     message body), author handle, 1-2 sentence functional summary.

   **PULL REQUEST** — `github_get_pull_request(repo, pr_number)`:
   - Skip non-merged PRs — only `merged=true` warrants a `Fix` entity and activity
     write. Append a warning and skip if `merged=false`.
   - Read signals in PRIORITY ORDER:
     1. **PR title** — conventional-commit prefix or explicit `[fix]` / `[feat]` tags
        are the highest-signal kind classifier.
     2. **Labels** — GitHub labels (`bug`, `enhancement`, `breaking change`, etc.)
        narrow kind and flag bug evidence.
     3. **PR body** — author rationale; check for `Fixes #<number>`,
        `Closes #<number>`, `Resolves #<number>` links to issues.
     4. **Linked issues** — if the PR body links a `bug`-labeled issue, that anchors
        the `Fix` → `BugPattern` relationship.
     5. **Commit count / list** — LAST RESORT; confirms kind when title/body are
        ambiguous.
   - Classify: kind (fix/feat/chore/refactor/other), PR author handle, summary.

   **ISSUE** — `github_get_issue(repo, issue_number)`:
   - Read signals in PRIORITY ORDER (GitHub-adapted; equivalent to Linear's issue
     procedure):
     1. **Labels** — `bug` label is the highest-signal kind classifier; it's
        author-applied and standardized per repo.
     2. **Issue state** — `closed` bug issues may warrant `BugPattern` (if symptom is
        clear). `open` issues never warrant `BugPattern`.
     3. **Title** — fallback kind signal if labels are absent.
     4. **Body** — author rationale; check for explicit `Why:` / `Decision:` /
        `Alternatives:` sections that justify a `Decision` entity.
     5. **Comments** — LAST RESORT. Only when higher signals leave kind / outcome
        ambiguous.
   - Classify: author / creator handle, kind (feat/fix/chore/other), summary.

3. Build one `LlmReconciliationPlan`-shaped object for the batch (see Mutations
   section). Call `apply_graph_mutations(plan, event_id, summary)` once per batch.
4. Mark each batch todo done. Move to the next batch.

### Phase 3 — Finalize

When all todos are drained (or you've hit the tool-call budget with a coherent subset
complete), tally:
- Commits ingested / skipped
- PRs merged / non-merged (skipped) / skipped
- Issues ingested by kind (feat / fix / bug / chore / ...)
- Distinct authors / creators
- Fix nodes emitted, BugPattern nodes emitted, Decision nodes emitted

Then `mark_event_processed(event_id, summary)` followed by `finish_batch(summary)`.

## Mutations (per item)

Use the existing ontology. Stable keys ensure backfill + future webhook converge.

Identity rules (NOT free-form strings):

- `Activity` is `EXTERNAL_ID` with `key_prefix=activity`. Use
  `activity:github:commit:<sha-lowered>`,
  `activity:github:pr:<owner>/<repo>/<pr_number>`, and
  `activity:github:issue:<owner>/<repo>/<issue_number>`. Segments after `activity:`
  may contain `-`, `/`, `.` per `_EXTERNAL_ID_SAFE_RE`.
- `Person` is `SLUG_ALIAS` with `key_prefix=person`. Use `person:<github_login_lowered>`.
- `Period` uses the production builder
  `timeline:period:daily:<pot>:<yyyy-mm-dd>`.
- `Fix` is `CONTENT_HASH`. Use `fix:<12-hex-sha256>` minted from
  `"github:pr:<owner>/<repo>/<pr_number>|fix|<symptom-slug>"` (stable so a re-ingest
  of the same PR collides on the same key and a future merged-PR webhook event
  converges).
- `BugPattern` is `SLUG_ALIAS` with `key_prefix=bug_pattern`. Use
  `bug_pattern:github-<repo-slug>:<symptom-slug>` where `repo-slug` slugifies
  `owner/repo` (e.g. `acme/my-repo` → `acme-my-repo`). Each colon-separated segment
  must be a valid slug. This key MUST match the key emitted by the Linear
  `linear_team_one_shot_ingestion` skill for the same symptom on the same repo
  (if a Linear team tracks that repo), so cross-source `BugPattern` nodes converge.
- `Decision` is `CONTENT_HASH`. Use `decision:<12-hex-sha256>` from
  `"github:issue:<owner>/<repo>/<number>|decision|<title>"`.

### Always emit (endpoint entities, at least once per batch)

- **Entity** `Period` — one per distinct activity date in the batch.
  - key: `timeline:period:daily:<pot>:<yyyy-mm-dd>`.
  - labels: `["Entity", "Period"]`.
  - properties: `period_kind="daily"`, `label="<yyyy-mm-dd>"`,
    `opened_at="<yyyy-mm-dd>T00:00:00+00:00"`.

### Per COMMIT — always emit

- **Entity** `Activity`
  - key: `activity:github:commit:<sha-lowered>`.
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<committed_at>`, `verb="github_commit"`,
    `title=<first line of commit message>`, `summary=<1-2 sentence summary>`,
    `kind` (fix/feat/chore/refactor/docs/other), `sha=<full sha>`,
    `repo_name=<owner/repo>`.
- **Entity** `Person` for the commit author.
  - key: `person:<github_login_lowered>`.
  - labels: `["Entity", "Person"]`.
  - properties: `name=<display_name>`, `handle=<login>`.
- **Edge** `PERFORMED` — `person:<author>` → activity key,
  `valid_from=<committed_at>`.
- **Edge** `IN_PERIOD` — activity key → period key (date from `committed_at`).

### Per PULL REQUEST (merged only) — always emit

- **Entity** `Activity`
  - key: `activity:github:pr:<owner>/<repo>/<pr_number>`.
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<merged_at>`, `verb="github_pr_merged"`,
    `title=<pr_title>`, `summary=<1-2 sentence summary>`,
    `pr_number=<number>`, `pr_url=<html_url>`,
    `kind` (fix/feat/chore/refactor/other), `repo_name=<owner/repo>`.
- **Entity** `Person` for the PR author.
  - key: `person:<github_login_lowered>`.
  - labels: `["Entity", "Person"]`.
  - properties: `name=<display_name>`, `handle=<login>`.
- **Edge** `PERFORMED` — `person:<author>` → activity key,
  `valid_from=<merged_at>`.
- **Edge** `IN_PERIOD` — activity key → period key.

### Per PULL REQUEST (merged, fixing a confirmed bug) — conditionally emit

Emit when (a) the PR title or labels signal `bug` / `fix` AND (b) the PR body links
a bug issue via `Fixes #N`, `Closes #N`, or `Resolves #N`, OR the PR labels include
`bug` / `bugfix`.

- **Entity** `Fix`
  - key: `fix:<12-hex-sha256>` from
    `"github:pr:<owner>/<repo>/<pr_number>|fix|<symptom-slug>"`.
  - labels: `["Entity", "Fix"]`.
  - properties: `title=<pr_title>`, `summary=<short symptom sentence>`,
    `source_pr=<activity_key>`, `pr_url=<html_url>`,
    `repo_name=<owner/repo>`.
- **Entity** `BugPattern`
  - key: `bug_pattern:github-<repo-slug>:<symptom-slug>`.
  - labels: `["Entity", "BugPattern"]`.
  - properties: `summary=<short canonical symptom>`,
    `symptom_signature=<short canonical sentence>`,
    `title=<symptom title>`, `source_pr=<activity_key>`.
- **Edge** `RESOLVED` — fix key → bug_pattern key (only when both are emitted).

### Per ISSUE — always emit

- **Entity** `Activity`
  - key: `activity:github:issue:<owner>/<repo>/<issue_number>`.
  - labels: `["Entity", "Activity"]`.
  - properties: `occurred_at=<closed_at or created_at>`,
    `verb="github_issue_<state>"` (e.g. `github_issue_closed`),
    `title=<issue_title>`, `summary=<1-2 sentence summary>`,
    `state=<state>`, `kind` (feat/fix/chore/other), `issue_url`.
- **Entity** `Person` per creator (and assignee if distinct and the issue is closed).
  - key: `person:<github_login_lowered>`.
  - labels: `["Entity", "Person"]`.
  - properties: `name=<display_name>`, `handle=<login>`.
- **Edge** `PERFORMED` — `person:<creator>` → activity key,
  `valid_from=<occurred_at>`.
- **Edge** `PERFORMED` — `person:<assignee>` → activity key,
  `valid_from=<occurred_at>` and `role="assignee"`, only when the assignee
  differs from the creator and the issue is closed.
- **Edge** `IN_PERIOD` — activity key → period key.

### Per ISSUE — conditionally emit

- **Bug report** (labels include `bug` AND state is `closed` AND symptom is clear in
  title or body):
  - **Entity** `BugPattern`
    - key: `bug_pattern:github-<repo-slug>:<symptom-slug>`.
    - labels: `["Entity", "BugPattern"]`.
    - properties: `summary=<short canonical symptom sentence>`,
      `symptom_signature=<short canonical sentence>`,
      `title=<symptom title>`, `source_issue=<activity_key>`.
    - NOTE: This key converges with the `BugPattern` emitted by a merged fixing PR
      (from this skill's PR phase) and with the Linear sibling skill — all write the
      same stable `bug_pattern:` key so cross-source nodes merge.
  - **Do NOT emit `Fix`** from a GitHub issue — even a closed `bug` issue. Fix is
    reserved for the merged PR that shipped the code change.
  - **Do NOT emit `RESOLVED`** from an issue (that edge requires a `Fix`, and issues
    never emit `Fix`).
- **Design decision** (body explicitly documents rationale + alternatives — common for
  RFCs, specs, ADRs filed as issues):
  - **Entity** `Decision`
    - key: `decision:<12-hex-sha256>` from
      `"github:issue:<owner>/<repo>/<number>|decision|<title>"`.
    - labels: `["Entity", "Decision"]`.
    - properties: `title=<short title>`,
      `summary=<one sentence decision summary>`, `status="accepted"`,
      `rationale=<stated rationale>`,
      `alternatives_rejected=<list or string>`,
      `source_issue=<activity_key>`.
  - **Edge** `AFFECTS` — decision key → feature/component/service/code asset, ONLY
    when a single affected scope is clear. Otherwise omit.

## Source-priority rationale (why)

GitHub's structured signals — conventional-commit prefixes, PR merge status, issue
labels — are author-applied and standardized per project. They beat free-form bodies
for kind / outcome classification. A commit message prefix of `fix:` or a PR label of
`bug` encodes intent that would take multiple comment reads to recover. PR merge status
gates `Fix` emission: an un-merged PR is a proposal, not a delivered change. Reading
full diffs, review threads, or comment history burns budget on rediscovering intent the
author already encoded in metadata. Stop climbing the priority ladder as soon as you
can answer kind + summary + bug/decision evidence.

## Bounds and budget

- ONE call each of `github_list_commits`, `github_list_pull_requests`,
  `github_list_issues`. No pagination beyond what the bounded list tools return.
- Soft tool-call cap: `30 + 3 × count` (each item averages ~3 calls:
  list + get + classify). Plan accordingly.
- If you approach the cap with a coherent recent subset ingested cleanly, FINISH —
  do not partially ingest an item. The tail can be re-run later with a smaller `count`
  (stable keys mean already-ingested items will be deduplicated, not duplicated).

## Anti-patterns

- Do NOT emit Activity for merge commits (messages starting with
  `"Merge pull request #"` or `"Merge branch '"`). These duplicate the PR Activity
  from the PR phase and produce noise in the timeline.
- Do NOT emit Person or Activity for bot authors (logins ending in `[bot]`).
  Dependabot, Renovate, and similar bots are not human contributors.
- Do NOT emit `Fix` from a GitHub issue — even a closed `bug` issue. Fix is reserved
  for the merged PR that shipped the code change. A closed issue is evidence the team
  believes it is fixed; it is NOT itself a Fix.
- Do NOT emit `RESOLVED` from a GitHub issue (RESOLVED connects Fix → BugPattern, and
  issues never emit Fix).
- Do NOT process non-merged PRs as `Fix` candidates — skip them with a warning.
- Do NOT page past the bounded list calls.
- Do NOT read full commit diffs or PR review-comment threads unless kind / summary
  cannot be derived from the commit message, PR title, and labels.
- Do NOT invent BugPatterns, Fixes, Decisions, or Persons not actually evidenced in the
  data you read. Emit a warning record instead.
- Do NOT auto-close / auto-resolve any open issue or incident based on a GitHub
  `closed` status alone — the status is evidence, not closure.
- Do NOT run this skill on a repo that already has live merged-PR webhook ingestion
  going against the SAME date window — let the webhook handle live updates.

## Single-event contract

This skill, when invoked by the internal agent, runs as a single
`(github, github_repo, one_shot_ingest)` event. Pass that ONE `event_id` to every
`apply_graph_mutations` call and to the final `mark_event_processed` — per-item
identity is the entity_key (`activity:github:commit:...`, `fix:<hash>`, etc.), not
the event id, so multiple Activities / Fixes / BugPatterns under one event id is
correct.

When invoked by Claude Code outside the event pipeline, there is no internal-agent
event state, so the internal `apply_graph_mutations` tool will reject an empty or
invented event id. Use this document as the extraction procedure only when the host
provides a compatible context-graph write path and a valid event/provenance id.
Otherwise stop after producing the proposed plan; do not pretend to apply it.
