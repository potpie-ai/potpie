# GitHub Intelligence Layer — Technical Overview

**Audience:** Engineers implementing or reviewing the system. This doc is **code-free** (no snippets); behavior, boundaries, and data flow are spelled out in prose. Implementation detail lives in [github-intelligence-layer-architecture.md](./implementation-plans/github-intelligence-layer-architecture.md).

---

## Objective

Potpie’s **code graph** already answers structural questions (symbols, files, call graph). The intelligence layer adds **version-control and collaboration context**: merged PRs, commits, issues, review threads, and inferred relationships (why, feature area, decisions).

Target queries for agents and APIs:

- Why was this function or file last materially changed?
- Which PR / issue / milestone narrative applies?
- What was argued in review and what was decided?
- Who has been driving changes in this path (recency-weighted)?

---

## Single Neo4j graph, two layers + bridges

**Storage:** One Neo4j instance (same URI as today). No separate “intelligence” database.

| Layer | Responsibility | Scoping |
|--------|----------------|---------|
| **Code graph** (existing) | `NODE` labels with types FILE / CLASS / FUNCTION / INTERFACE; structural rels (CALLS, IMPORTS, etc.) | `repoId` = Potpie `project_id` |
| **Graphiti subgraph** | Episodic ingestion + extracted **Entity** nodes (PullRequest, Issue, Decision, Feature, Developer, Commit, …) | `group_id` = Potpie `project_id` |
| **Bridges** (our writes) | Edges from code nodes to Graphiti entities, e.g. function or file → pull request | Same `project_id`; rels carry PR metadata where useful |

**Principle:** Not multiple disconnected graphs—one graph, explicit bridges so traversals can cross from “what the code is” to “why and how it changed.”

---

## Graphiti: responsibilities and limits

**Graphiti** ingests **episodes** (rich text per event), runs **LLM-based entity and relationship extraction**, maintains **bi-temporal** edge semantics, and exposes **hybrid search** (lexical + semantic) with **native `group_id` isolation** per project.

**We still own:**

- **Bridge edges** from parsed code nodes to PR (and optionally Decision) entities—Graphiti does not know our `node_id` / `file_path` / line span model.
- **Deterministic enrichment** before or alongside LLM: issue refs from PR body, ticket IDs from branch names, file lists and hunks from the GitHub API.
- **Episode shape**: section ordering, inclusion of linked issue bodies when PR description is empty, **review threads** grouped by reply chain (not a flat comment list).

**Known product caveat:** Custom entity properties on Neo4j nodes have had reliability issues in some Graphiti versions; the architecture plan calls out mirroring critical “why” text inside the episode body so search and extraction still see it.

---

## Entity model (conceptual)

Custom **Pydantic-defined** types are passed into Graphiti so extraction is constrained. Examples:

- **PullRequest** — number, title, author, merged time, why summary, change class (bugfix / feature / refactor / …), feature area, files touched (as text).
- **Commit** — SHA fragment, message line, author, branch (for standalone commit episodes).
- **Issue** — number, title, distilled problem statement.
- **Feature** — name / description (from labels, milestone, narrative).
- **Decision** — decision, rejected alternatives, rationale (from review discussion).
- **Developer** — GitHub identity; optional inferred expertise from history.

**Edge kinds** (between entities) are also typed where Graphiti supports it—e.g. PR **fixes** issue, PR **authored by** developer, decision **made in** context of PR. Exact maps live in the architecture doc.

---

## Data sources and extraction split

**From GitHub API (structured, no LLM):** PR metadata, file list, patches, commits in PR, issue comments, review comments (with path and line), labels, milestone.

**Deterministic parsing:** “Fixes/closes/resolves #N” from PR body and commit messages; ticket-shaped tokens from branch names; mapping review comment (path, line) to overlapping code-graph nodes by line range.

**LLM (via Graphiti on episode ingest):** Short why summary, change type, feature area, decision objects, issue problem condensation—anything that needs reading prose and thread context.

**Bridge writer (our service):** Parse unified-diff hunks into **new-file line ranges**; overlap with `FUNCTION`/`CLASS` nodes where line geometry matches the **current** parse tree; always establish **file-level** touch edges when path is known. **Backfill** favors file-level bridges when historical hunks no longer align with today’s line numbers; **live merge** path can be tighter.

---

## Ingestion pipeline (ordered steps)

1. **Fetch** merged PR payload plus commits, files (with patches), grouped review threads, linked issues (from refs + API).
2. **Dedup** using Postgres `context_ingestion_log` on (project_id, source_type, source_id)—e.g. `pr_<n>_merged`.
3. **Build episode** — one primary episode per merged PR; commits inside the PR are embedded in that episode, not separate stories.
4. **Persist raw (optional but recommended)** — append-only `raw_events` (or equivalent) so episode format upgrades can replay without re-calling GitHub.
5. **Graphiti `add_episode`** with `group_id = project_id`, custom entity and edge types.
6. **Log** episode UUID to `context_ingestion_log`; mark bridge completion when done.
7. **Bridge** — write `MODIFIED_IN` / `TOUCHED_BY` (and decision links where modeled) from code graph to Entity nodes.

**Ordering for backfill:** Oldest merged PR first so Graphiti’s temporal ordering matches real time.

**Throughput:** Batch backfills (e.g. capped PRs per job), delays for REST rate limits, Celery (or existing queue) for async work.

**Idempotency:** Webhooks can repeat; dedup key prevents double episodes and double bridges.

---

## Operational surfaces

- **Existing:** `get_project_context`–style Graphiti search over `group_id`, extendable with **entity label filters** once custom types are wired.
- **Planned:** Dedicated tools or routes for change history (function/file → PRs + issues + decisions), file ownership from PR authorship graph, decision listing per path.

Exact route names in the architecture sketch are illustrative; implementation may attach these as LangChain tools on the existing agent stack.

---

## Extensibility (Linear, Jira, docs)

New sources add **new entity types**, **new episode builders**, and **new bridge rules**—they do not fork the code graph.

- **Linear / Jira:** Ticket entity; link PR ↔ ticket via branch name and PR body patterns; optional link ticket ↔ feature epic.
- **Confluence / ADR:** Document entity; bridge from paths or sections mentioned in text to FILE nodes where feasible.

Same Postgres dedup + Graphiti namespace pattern applies.

---

## Risks (engineering)

| Risk | Mitigation |
|------|------------|
| LLM mis-extracts or drops entities | Rich episode text; deterministic side paths for IDs and paths; tests on real repos. |
| Line drift on old PRs | File-level bridges for history; function-level mainly for fresh merges. |
| Cost / latency at scale | Small extraction model where acceptable; dedup; batching; observability on tokens and queue depth. |
| Isolation bugs | Always pass `group_id` / `repoId`; never query without project scope. |
| Graphiti + provider quirks | e.g. known Anthropic + `group_id` issues—track upstream; OpenAI path default. |

---

## Phased delivery

| Phase | Deliverables |
|-------|----------------|
| **1** | Typed Graphiti entities, rich PR episodes + thread grouping, bridge writer, backfill + webhook path, indexes for bridge queries, first history-oriented agent tool. |
| **2** | External ticket entities (Linear/Jira), ownership and decision-focused queries, hooks from agent-created PRs/tickets. |
| **3** | Document sources, search tuning, metrics, Neo4j load review. |
| **4** | Slack/Sentry-style episodes, caching if hot paths need it. |

Timelines in the full architecture doc are estimates.

---

## Success criteria (example outputs)

After ingestion, a trace from a **file or function** to **PR → issue → review** should return coherent bundles such as:

- PR identifier, author, merge window, short why, change class, linked issue titles, and a summarized review outcome (e.g. lock choice vs Redis).
- Ranked recent contributors to that path from merged PR authorship.
- Decision bullets tied to review threads on that file.

---

## Terms (quick reference)

| Term | Definition |
|------|------------|
| **VCS** | Version control; here Git as exposed through GitHub APIs. |
| **Episode** | Text blob + metadata sent to Graphiti for one logical event (usually one merged PR). |
| **Entity** | Graphiti-extracted node with custom type (PR, Issue, …). |
| **Bridge** | Our edge from a code-graph `NODE` to a Graphiti `Entity` (or between layers we control). |
| **group_id** | Graphiti namespace key; equals Potpie `project_id`. |
| **repoId** | Code-graph property tying nodes to a project. |

---

*Maintained with [github-intelligence-layer-architecture.md](./implementation-plans/github-intelligence-layer-architecture.md).*
