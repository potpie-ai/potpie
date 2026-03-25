# GitHub Intelligence Layer — Architecture

*Built on our discussion: one graph, Graphiti for entity extraction, custom schemas, bridge to code graph, extensible for Linear/Jira/Confluence.*

**Technical overview (no code samples):** [github-intelligence-layer-overview.md](../github-intelligence-layer-overview.md)

---

## 1. Vision

Answer questions no code search can:

- *"Why was `processPayment` changed?"*
- *"Which feature does this file belong to?"*
- *"What design decisions were rejected during review?"*
- *"Who owns this module and what's their recent focus?"*

This is Potpie's **intelligence layer**: structural code knowledge (what exists, who calls whom) enriched with **temporal intent** (why it changed, who decided, what alternatives were rejected).

---

## 2. Core Principle: One Graph

```
Neo4j (single instance — NEO4J_URI)
│
├── Code layer (existing)
│   (:FILE), (:CLASS), (:FUNCTION), (:INTERFACE)
│   relationships: CONTAINS, CALLS, IMPORTS, HAS_METHOD, REFERENCES
│   scoped by: repoId = project_id
│
├── VCS intelligence layer (new — Graphiti custom entities)
│   (:Entity:PullRequest), (:Entity:Commit), (:Entity:Issue),
│   (:Entity:Feature), (:Entity:Decision), (:Entity:Developer)
│   scoped by: group_id = project_id
│
└── Bridge relationships (new — written by us after Graphiti ingestion)
    (FUNCTION)-[:MODIFIED_IN]->(Entity:PullRequest)
    (FILE)-[:TOUCHED_BY]->(Entity:PullRequest)
```

**Not three graphs. One graph with two layers connected by bridges.** Code nodes and VCS entities live in the same Neo4j instance. Queries traverse freely between them.

---

## 3. Why Graphiti (and where it stops)

### What Graphiti gives us

| Capability | Value |
|---|---|
| Custom entity extraction from episode text | LLM extracts `PullRequest`, `Decision`, `Feature` entities with typed properties |
| Bi-temporal edges with invalidation | PR goes open → merged: old edges auto-expire, facts rewritten to past tense |
| Hybrid search (semantic + BM25 + RRF) | `graphiti.search(query, group_ids=[project_id])` — sub-second |
| `group_id` namespace isolation | Project isolation is native, not a filter hack |
| Entity dedup on re-ingest | Same entity mentioned across episodes → single node, updated summary |

### What Graphiti does NOT do

| Gap | Our solution |
|---|---|
| Bridge VCS entities to code graph nodes (FILE, FUNCTION) | **Bridge Writer**: after episode ingestion, parse diff hunks, match changed line ranges to FUNCTION/CLASS nodes, write `[:MODIFIED_IN]` relationships via Neo4j driver |
| Deterministic structured fields (PR number, SHA, file list) | **Episode text engineering**: include structured data in episode body so LLM extracts it reliably; also write to `context_ingestion_log` for bridge lookups |
| Diff hunk → function mapping | **Hunk Parser**: regex-parse `@@ -n,m +n,m @@` from GitHub patch, compute line-range overlaps with Neo4j NODE entries |

### What Graphiti extracts via LLM (the non-deterministic part)

These fields CANNOT come from GitHub API — they require reading PR body, comments, review threads, and understanding intent:

- `PullRequest.why_summary` — "Why was this change made?"
- `PullRequest.change_type` — bugfix / feature / refactor / performance / infra
- `Feature.name` + `Feature.description` — inferred from labels, milestones, PR clustering
- `Decision.decision_made` + `Decision.alternatives_rejected` — extracted from review comment threads
- `Developer.expertise_areas` — inferred from PR history
- `Issue.problem_statement` — distilled from issue body

### What we extract deterministically (no LLM needed)

These come directly from GitHub API — structured, exact, no guessing:

| Data | Source | Method |
|---|---|---|
| PR number, title, author, dates, branches | `github_provider.get_pull_request()` | Direct |
| Commit SHA, message, author | `pr.get_commits()` | Direct |
| Files changed per PR | `pr.get_files()` | Direct |
| Changed line ranges per file | Diff patch `@@ hunk @@` | Regex |
| PR → Issue link ("Fixes #101") | PR body + commit messages | Regex |
| PR → Ticket (JIRA/Linear) | Branch name `feat/TICKET-421-...` | Regex |
| Review comment → file + line | `pr.get_review_comments()` | Direct (has `path`, `line`) |
| FUNCTION ← changed lines | Neo4j `start_line`/`end_line` overlap | Cypher |

---

## 4. Graphiti Entity Schema

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PullRequest(BaseModel):
    """A GitHub pull request — the unit of intentional change."""
    pr_number: Optional[int] = Field(None, description="PR number e.g. 42")
    title: Optional[str] = Field(None, description="PR title")
    why_summary: Optional[str] = Field(
        None,
        description="1-2 sentence summary of WHY this change was made, "
                    "extracted from PR body, linked issues, and review discussion"
    )
    change_type: Optional[str] = Field(
        None,
        description="One of: bugfix, feature, refactor, performance, infra, docs"
    )
    feature_area: Optional[str] = Field(
        None,
        description="Product/feature area e.g. payments, auth, notifications, onboarding"
    )
    author: Optional[str] = Field(None, description="GitHub username of PR author")
    merged_at: Optional[str] = Field(None, description="ISO datetime when PR was merged")
    files_changed: Optional[str] = Field(
        None, description="Comma-separated list of files modified in this PR"
    )


class Commit(BaseModel):
    """A Git commit."""
    sha: Optional[str] = Field(None, description="Short (12-char) commit SHA")
    message: Optional[str] = Field(None, description="First line of commit message")
    author: Optional[str] = Field(None, description="Commit author GitHub username")
    branch: Optional[str] = Field(None, description="Branch this commit was on")


class Issue(BaseModel):
    """A GitHub issue that motivated a code change."""
    issue_number: Optional[int] = Field(None, description="Issue number")
    title: Optional[str] = Field(None, description="Issue title")
    problem_statement: Optional[str] = Field(
        None, description="What problem this issue describes, distilled from body"
    )


class Feature(BaseModel):
    """A product feature, milestone, or epic that groups related changes."""
    name: Optional[str] = Field(None, description="Feature or milestone name")
    description: Optional[str] = Field(None, description="What this feature does")


class Decision(BaseModel):
    """A design or implementation decision made during code review."""
    decision_made: Optional[str] = Field(None, description="What was decided")
    alternatives_rejected: Optional[str] = Field(
        None, description="What alternatives were considered and why rejected"
    )
    rationale: Optional[str] = Field(None, description="Why this decision was made")


class Developer(BaseModel):
    """A developer who authors code changes."""
    github_login: Optional[str] = Field(None, description="GitHub username")
    display_name: Optional[str] = Field(None, description="Full name if available")
    expertise_areas: Optional[str] = Field(
        None, description="Areas of expertise inferred from PR history"
    )
```

---

## 5. Edge Type Schema

```python
from pydantic import BaseModel, Field
from typing import Optional


class Modified(BaseModel):
    """A PR modified a file."""
    additions: Optional[int] = Field(None, description="Lines added")
    deletions: Optional[int] = Field(None, description="Lines deleted")

class Fixes(BaseModel):
    """A PR fixes/closes a GitHub issue."""
    resolution: Optional[str] = Field(None, description="How the issue was resolved")

class PartOfFeature(BaseModel):
    """A PR is part of a product feature."""
    pass

class MadeIn(BaseModel):
    """A design decision was made in a PR review."""
    pass

class AuthoredBy(BaseModel):
    """A PR was authored by a developer."""
    pass

class Owns(BaseModel):
    """A developer owns a file (recency-weighted)."""
    last_modified: Optional[str] = Field(None, description="When developer last touched this file")
    pr_count: Optional[int] = Field(None, description="Number of PRs touching this file by this dev")


# Edge type map: which edge types can exist between which entity pairs
EDGE_TYPE_MAP = {
    ("PullRequest", "Issue"):     ["Fixes"],
    ("PullRequest", "Feature"):   ["PartOfFeature"],
    ("PullRequest", "Developer"): ["AuthoredBy"],
    ("Decision", "PullRequest"):  ["MadeIn"],
    ("Developer", "PullRequest"): ["AuthoredBy"],
    ("Entity", "Entity"):         ["RELATES_TO"],  # fallback
}
```

---

## 6. Episode Builder — The Critical Component

Episode text quality determines extraction quality. The episode must be **rich enough** for Graphiti's LLM to extract all entities and relationships.

### PR Episode (one per merged PR — the primary unit)

```python
def build_pr_episode(pr_data: dict, commits: list, review_threads: list,
                     linked_issues: list) -> dict:
    """
    Build a Graphiti episode from a merged GitHub PR.

    The episode text is structured so the LLM can extract:
    - PullRequest entity (number, title, why_summary, change_type, feature_area)
    - Developer entity (author)
    - Issue entities (linked issues with problem_statement)
    - Decision entities (from review threads where alternatives were discussed)
    - Feature entity (from labels/milestone)
    """
    n = pr_data["number"]
    title = pr_data.get("title", "")
    body = pr_data.get("body", "") or ""
    author = pr_data.get("author", "unknown")
    head = pr_data.get("head_branch", "?")
    base = pr_data.get("base_branch", "?")
    merged_at = pr_data.get("merged_at", "")
    files = ", ".join(f.get("filename", "") for f in (pr_data.get("files") or []))
    labels = ", ".join(pr_data.get("labels") or [])
    milestone = pr_data.get("milestone", "")

    # Linked issues context
    issues_text = ""
    for issue in linked_issues:
        issues_text += (
            f"Linked Issue #{issue['number']}: {issue['title']}. "
            f"{(issue.get('body') or '')[:300]}\n"
        )

    # Review threads — WHERE DECISIONS LIVE
    # Group review comments by in_reply_to_id to reconstruct threads
    review_text = ""
    for thread in review_threads:
        file_ctx = f"File: {thread['path']}, line {thread.get('line', '?')}"
        diff_hunk = (thread.get("diff_hunk") or "")[-200:]
        comments = "\n".join(
            f"  {c['author']}: {c['body']}" for c in thread["comments"]
        )
        review_text += f"\nReview thread on {file_ctx}:\n{diff_hunk}\nDiscussion:\n{comments}\n"

    # Commit messages
    commit_text = "\n".join(
        f"- {c.get('sha', '?')[:8]}: {(c.get('message') or '').split(chr(10))[0]}"
        for c in commits
    )

    episode_body = f"""
Pull Request #{n}: {title}
Author: {author}
Branch: {head} → {base}
Merged: {merged_at}
Files changed: {files}

WHY THIS CHANGE WAS MADE:
{body or "No description provided"}

RELATED ISSUES:
{issues_text or "None linked"}

COMMITS:
{commit_text or "None"}

REVIEW DISCUSSIONS (design decisions and concerns):
{review_text or "No review comments"}

LABELS: {labels or "None"}
MILESTONE/FEATURE: {milestone or "None"}
""".strip()

    return {
        "name": f"PR #{n}: {title}",
        "episode_body": episode_body,
        "source_description": f"GitHub PR #{n} merged",
        "source_id": f"pr_{n}_merged",
        "reference_time": merged_at,
    }
```

### Review Thread Grouping

GitHub returns review comments as a flat list. Must reconstruct threads:

```python
def group_review_threads(review_comments: list) -> list[dict]:
    """
    Group flat review comments into conversation threads.
    A thread = root comment + all replies (linked by in_reply_to_id).
    Threads with back-and-forth have more signal than isolated comments.
    """
    threads = {}
    for c in review_comments:
        reply_to = c.get("in_reply_to_id")
        root_id = reply_to if reply_to else c["id"]
        if root_id not in threads:
            threads[root_id] = {
                "path": c.get("path", ""),
                "line": c.get("line"),
                "diff_hunk": c.get("diff_hunk", ""),
                "comments": [],
            }
        threads[root_id]["comments"].append({
            "author": (c.get("user") or {}).get("login", "unknown"),
            "body": c.get("body", ""),
            "created_at": c.get("created_at", ""),
        })
    return list(threads.values())
```

### Commit Episode (only for direct pushes — NOT for commits inside a PR)

Commits inside a PR are included in the PR episode. Separate commit episodes are only for direct pushes to default branch that bypass PRs.

```python
def build_commit_episode(commit_data: dict, branch: str) -> dict:
    sha = (commit_data.get("sha") or "")[:12]
    message = (commit_data.get("message") or "").strip().split("\n")[0]
    author = commit_data.get("author", "unknown")
    files = ", ".join(f.get("filename", "") for f in (commit_data.get("files") or []))

    episode_body = f"""
Direct commit {sha} by {author}
Branch: {branch}
Files changed: {files}
Message: {message}
""".strip()

    return {
        "name": f"Commit {sha}: {message[:60]}",
        "episode_body": episode_body,
        "source_description": f"GitHub commit {sha} on {branch}",
        "source_id": f"commit_{sha}",
        "reference_time": commit_data.get("committed_at"),
    }
```

---

## 7. Bridge Writer — Connecting Code Graph to VCS Entities

After Graphiti ingests an episode and creates `Entity:PullRequest` nodes, we write bridge relationships from code graph nodes to those entities.

### Step 1: Parse diff hunks → changed line ranges

```python
import re

def parse_diff_hunks(patch: str) -> list[tuple[int, int]]:
    """
    Parse GitHub unified diff patch → list of (start_line, end_line) in the NEW file.

    @@ -old_start,old_count +new_start,new_count @@
    We care about new_start and new_count (what the file looks like AFTER the PR).
    """
    ranges = []
    for match in re.finditer(r'@@ -\d+,?\d* \+(\d+),?(\d*) @@', patch):
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) else 1
        end = start + count - 1
        if end >= start:
            ranges.append((start, end))
    return ranges
```

### Step 2: Match changed lines to FUNCTION/CLASS nodes

```cypher
// For each (file_path, hunk_start, hunk_end) from the diff:
MATCH (n:NODE {file_path: $file_path, repoId: $project_id})
WHERE n.type IN ['FUNCTION', 'CLASS']
  AND n.start_line <= $hunk_end
  AND n.end_line >= $hunk_start
RETURN n.node_id, n.name, n.type
```

### Step 3: Write bridge relationships

```cypher
// Link code node → PullRequest entity
MATCH (n:NODE {node_id: $node_id, repoId: $project_id})
MATCH (pr:Entity {name: $pr_entity_name, group_id: $project_id})
WHERE 'PullRequest' IN labels(pr)
MERGE (n)-[:MODIFIED_IN {pr_number: $pr_number, merged_at: $merged_at}]->(pr)

// Also create file-level link (always — even when function match fails)
MATCH (f:NODE {file_path: $file_path, repoId: $project_id})
WHERE f.type = 'FILE'
MATCH (pr:Entity {name: $pr_entity_name, group_id: $project_id})
WHERE 'PullRequest' IN labels(pr)
MERGE (f)-[:TOUCHED_BY {pr_number: $pr_number}]->(pr)
```

### Step 4: Link review comments to code nodes

Review comments already have `path` and `line` — direct match:

```cypher
MATCH (n:NODE {file_path: $review_path, repoId: $project_id})
WHERE n.type IN ['FUNCTION', 'CLASS']
  AND n.start_line <= $review_line
  AND n.end_line >= $review_line
MATCH (d:Entity {group_id: $project_id})
WHERE 'Decision' IN labels(d)
  AND d.name CONTAINS $pr_reference
MERGE (n)-[:HAS_DECISION]->(d)
```

### Historical backfill vs live — line match strategy

| PR timing | Strategy | Reason |
|---|---|---|
| **Live (webhook)** | Precise: diff hunk lines → current FUNCTION node overlap | Code graph is current; diff is against current HEAD |
| **Historical backfill** | File-level only: `(FILE)-[:TOUCHED_BY]->(PullRequest)` | Old diff line numbers don't match current FUNCTION positions (code has shifted) |

---

## 8. Deterministic Relationship Extraction

These parsers run BEFORE the episode is sent to Graphiti. Their output is used for bridge writing (§7) and also included in episode text to help LLM extraction.

### "Fixes #N" from PR body and commit messages

```python
import re

def extract_issue_refs(text: str) -> list[int]:
    """Extract issue numbers from 'Fixes #N', 'Closes #N', 'Resolves #N'."""
    pattern = r'(?:fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+#(\d+)'
    return list(set(int(m) for m in re.findall(pattern, text, re.IGNORECASE)))
```

### Ticket ID from branch name

```python
def extract_ticket_from_branch(branch: str) -> str | None:
    """'feat/TICKET-421-payment-fix' → 'TICKET-421'
       'fix/FE-123-bug' → 'FE-123'"""
    match = re.search(r'([A-Z][A-Z0-9]+-\d+)', branch)
    return match.group(1) if match else None
```

### Labels → Feature mapping

```python
def extract_feature_from_labels(labels: list[str], milestone: str | None) -> str | None:
    """Infer feature area from PR labels and milestone."""
    if milestone:
        return milestone
    feature_labels = [l for l in labels if not l.startswith("bug") and l not in ("wontfix", "duplicate", "invalid")]
    return feature_labels[0] if feature_labels else None
```

---

## 9. Ingestion Pipeline

### 9.1 Graphiti Initialisation

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

ENTITY_TYPES = {
    "PullRequest": PullRequest,
    "Commit": Commit,
    "Issue": Issue,
    "Feature": Feature,
    "Decision": Decision,
    "Developer": Developer,
}

async def ingest_episode(graphiti: Graphiti, episode: dict, project_id: str):
    result = await graphiti.add_episode(
        name=episode["name"],
        episode_body=episode["episode_body"],
        source=EpisodeType.text,
        source_description=episode["source_description"],
        reference_time=episode["reference_time"],
        group_id=project_id,
        entity_types=ENTITY_TYPES,
        edge_types=EDGE_TYPES,
        edge_type_map=EDGE_TYPE_MAP,
    )
    return result
```

### 9.2 Full ingestion flow (per merged PR)

```
┌──────────────────────────────────────────────────────┐
│  1. Fetch PR data from GitHub                        │
│     github_provider.get_pull_request(include_diff)   │
│     pr.get_commits()                                 │
│     pr.get_issue_comments()                          │
│     pr.get_review_comments()                         │
│     + fetch linked issues from extract_issue_refs()  │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│  2. Dedup check                                      │
│     context_ingestion_log: (project_id, source_type, │
│     source_id="pr_42_merged") already exists? Skip.  │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│  3. Build episode                                    │
│     build_pr_episode(pr, commits, threads, issues)   │
│     → rich text with WHY, REVIEW DISCUSSIONS, etc.   │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│  4. Graphiti add_episode()                           │
│     entity_types: PullRequest, Commit, Issue,        │
│                   Feature, Decision, Developer       │
│     → LLM extracts entities + edges                  │
│     → Writes to Neo4j as Entity:PullRequest etc.     │
│     → Returns episode UUID                           │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│  5. Log to Postgres                                  │
│     context_ingestion_log: (project_id, source_type, │
│     source_id, episode_uuid)                         │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│  6. Bridge Writer                                    │
│     Parse diff hunks → line ranges per file          │
│     Match FUNCTION/CLASS nodes by line overlap        │
│     Write (NODE)-[:MODIFIED_IN]->(Entity:PullRequest)│
│     Write (FILE)-[:TOUCHED_BY]->(Entity:PullRequest) │
│     Write (NODE)-[:HAS_DECISION]->(Entity:Decision)  │
└──────────────────────────────────────────────────────┘
```

### 9.3 Backfill job (Celery — run once per project)

- **Order:** Oldest merged PR first → newest. Graphiti's temporal edges depend on `reference_time` ordering.
- **Batch:** 100 PRs per run. Resumable via `context_sync_state.last_synced_at`.
- **Rate limit:** GitHub allows 5000 req/hour authenticated. Add 0.5s delay between PR fetches.
- **Bridge:** For historical PRs, only file-level `[:TOUCHED_BY]` (not function-level — line numbers have shifted).
- **Commits:** Only default-branch direct pushes (commits already inside PRs are part of PR episodes).
- **Raw store:** Write raw PR payload to Postgres `raw_events` before processing. If episode builder improves later, re-run from raw — never re-fetch from GitHub.

### 9.4 Live ingestion (webhook on PR merge)

- Webhook fires → validate signature → enqueue Celery task.
- Same pipeline as backfill but for single PR.
- **Bridge: precise function-level** `[:MODIFIED_IN]` (code graph is current, diff is against HEAD).
- Same dedup via `context_ingestion_log`.

---

## 10. Neo4j Indexes (new — for bridge queries)

```cypher
CREATE INDEX node_filepath_repo_idx IF NOT EXISTS
  FOR (n:NODE) ON (n.file_path, n.repoId);

CREATE INDEX node_lineno_idx IF NOT EXISTS
  FOR (n:NODE) ON (n.file_path, n.start_line, n.end_line, n.repoId);

CREATE INDEX entity_name_group_idx IF NOT EXISTS
  FOR (e:Entity) ON (e.name, e.group_id);

CREATE INDEX entity_pr_label_idx IF NOT EXISTS
  FOR (e:Entity:PullRequest) ON (e.group_id);
```

---

## 11. Query API — Agent Tools

### 11.1 `get_change_history` — "Why was this changed?"

```python
class GetChangeHistoryInput(BaseModel):
    project_id: str
    function_name: Optional[str] = None
    file_path: Optional[str] = None
    limit: int = 10
```

```cypher
// By function name
MATCH (f:NODE {name: $function_name, repoId: $project_id})
WHERE f.type IN ['FUNCTION', 'CLASS']
MATCH (f)-[:MODIFIED_IN]->(pr:Entity)
WHERE 'PullRequest' IN labels(pr)
OPTIONAL MATCH (pr)<-[:MadeIn]-(d:Entity)
WHERE 'Decision' IN labels(d)
OPTIONAL MATCH (pr)-[:Fixes]->(i:Entity)
WHERE 'Issue' IN labels(i)
RETURN pr.name, pr.why_summary, pr.change_type, pr.feature_area,
       collect(DISTINCT d.decision_made) AS decisions,
       collect(DISTINCT i.title) AS fixed_issues
ORDER BY pr.merged_at DESC
LIMIT $limit
```

### 11.2 `get_project_context` — Enhanced (existing tool, upgraded)

Keep the existing semantic search via `graphiti.search()` but with typed filters:

```python
from graphiti_core.search.search_filters import SearchFilters

# Search only PullRequest and Decision entities for a project
results = await graphiti.search(
    query=search_text,
    group_ids=[project_id],
    search_filter=SearchFilters(node_labels=["PullRequest", "Decision"]),
    num_results=limit,
)
```

### 11.3 `get_file_owner` — "Who owns this?"

```cypher
MATCH (f:NODE {file_path: $file_path, repoId: $project_id})
WHERE f.type = 'FILE'
MATCH (f)-[:TOUCHED_BY]->(pr:Entity)-[:AuthoredBy]->(dev:Entity)
WHERE 'PullRequest' IN labels(pr) AND 'Developer' IN labels(dev)
RETURN dev.github_login, count(pr) AS pr_count,
       max(pr.merged_at) AS last_touched
ORDER BY last_touched DESC, pr_count DESC
LIMIT 5
```

### 11.4 `get_decisions` — "What was debated?"

```cypher
MATCH (f:NODE {file_path: $file_path, repoId: $project_id})
MATCH (f)-[:MODIFIED_IN|HAS_DECISION]->(e:Entity)
WHERE 'Decision' IN labels(e)
RETURN e.decision_made, e.alternatives_rejected, e.rationale
```

---

## 12. Extensibility — Adding Linear, Jira, Confluence

The schema is designed so new sources = new node type + new relationships. Nothing existing changes.

### Linear/Jira Tickets (Phase 2)

```python
class Ticket(BaseModel):
    """A project management ticket (Linear, Jira)."""
    ticket_id: Optional[str] = Field(None, description="Ticket ID e.g. FE-456")
    title: Optional[str] = Field(None, description="Ticket title")
    status: Optional[str] = Field(None, description="Current status")
    priority: Optional[str] = Field(None, description="Priority level")
    acceptance_criteria: Optional[str] = Field(
        None, description="Acceptance criteria for the ticket"
    )
    source_system: Optional[str] = Field(None, description="linear or jira")
```

New edge types:
```python
EDGE_TYPE_MAP.update({
    ("PullRequest", "Ticket"): ["Implements"],
    ("Ticket", "Feature"):     ["BelongsTo"],
})
```

Linking: same regex from branch name (`feat/FE-456-...`) and PR body (`Implements FE-456`).

New query: "processPayment kyun change hua?" now also returns:
```
→ PR #42 → Implements → Ticket FE-456: "Payment timeout on high load" (priority: urgent)
→ Ticket FE-456 → BelongsTo → Feature: "Q1 Payment Reliability"
```

### Confluence/Docs (Phase 3)

```python
class Document(BaseModel):
    """An architectural document, ADR, or policy."""
    doc_type: Optional[str] = Field(None, description="ADR, PRD, runbook, policy")
    title: Optional[str] = Field(None, description="Document title")
    key_constraints: Optional[str] = Field(
        None, description="Constraints or rules this document imposes"
    )
```

```python
EDGE_TYPE_MAP.update({
    ("Document", "Feature"):   ["Documents"],
    ("Document", "Entity"):    ["Governs"],
})
```

Bridge: `(FILE)-[:GOVERNED_BY]->(Entity:Document)` written when doc mentions file paths.

### Each new source follows this pattern

```
1. Define Pydantic entity model (custom properties)
2. Define edge types + edge_type_map entries
3. Write episode builder (format rich text for Graphiti LLM)
4. Write bridge writer (connect to code graph nodes)
5. Add to ENTITY_TYPES dict in ingestion
6. Add extractor (API/webhook → episode → Graphiti → bridge)
```

Existing code, existing entities, existing bridges — untouched.

---

## 13. Postgres Schema Additions

### `raw_events` (new — immutable event log)

```sql
CREATE TABLE raw_events (
    id            SERIAL PRIMARY KEY,
    project_id    TEXT NOT NULL,
    source_type   TEXT NOT NULL,   -- 'github_pr', 'github_commit', 'linear_ticket', ...
    source_id     TEXT NOT NULL,   -- 'pr_42', 'commit_abc123', 'FE-456'
    payload       JSONB NOT NULL,  -- raw API response, never mutated
    received_at   TIMESTAMP WITH TIME ZONE DEFAULT now(),
    processed_at  TIMESTAMP WITH TIME ZONE,
    UNIQUE (project_id, source_type, source_id)
);
```

If episode builder logic improves, re-process from `raw_events`. Never re-fetch from GitHub.

### `context_ingestion_log` (existing — add column)

```sql
ALTER TABLE context_ingestion_log
    ADD COLUMN bridge_written BOOLEAN DEFAULT FALSE;
```

Track whether bridge relationships were written for this episode.

---

## 14. Build Phases

### Phase 1 — Foundation + PR Intelligence (4–6 weeks)

1. Graphiti custom entity types schema (`schema.py`)
2. Enhanced PR episode builder (with review threads, linked issues, labels, milestone)
3. Review thread grouping (`in_reply_to_id` → threads)
4. Diff hunk parser (regex → line ranges)
5. Bridge writer (FUNCTION ←→ PullRequest linking)
6. `raw_events` Postgres table + Alembic migration
7. Upgrade `ingest_episode()` to pass `entity_types` + `edge_type_map`
8. Backfill job: oldest-first ordering, file-level bridges for historical PRs
9. Webhook handler: live ingestion with function-level bridges
10. `get_change_history` agent tool
11. Upgrade `get_project_context` with `SearchFilters(node_labels=...)`
12. Neo4j indexes for bridge queries

### Phase 2 — Tickets + Ownership (3–4 weeks)

1. `Ticket` entity type (Linear + Jira)
2. Episode builders for Linear issues and Jira tickets
3. Branch name + PR body → ticket ID regex extraction
4. `(PullRequest)-[:Implements]->(Ticket)` edges
5. `get_file_owner` tool (recency-weighted from PR author graph)
6. `get_decisions` tool
7. Agent tool post-hooks: when agent creates a PR or ticket → ingest as episode

### Phase 3 — Documents + Quality (3–4 weeks)

1. `Document` entity type (Confluence, ADRs)
2. Episode builders for Confluence pages and local `/docs/adr/` files
3. `(FILE)-[:GOVERNED_BY]->(Document)` bridges
4. Search quality tuning: recency bias, result caps
5. Observability: what context is returned, latency, LLM cost per episode
6. Evaluate Neo4j load; FalkorDB option if needed

### Phase 4 — Operational Context (ongoing)

1. Slack thread episodes (emoji-triggered summarization)
2. Sentry exception cluster episodes
3. Redis cache for hot project context
4. Custom cross-source edge types (PR resolves Ticket resolves Issue)

---

## 15. Integration Points with Existing Potpie Code

| Existing file | Change needed |
|---|---|
| `app/modules/context_graph/graphiti_client.py` | Pass `entity_types`, `edge_types`, `edge_type_map` to `add_episode()` |
| `app/modules/context_graph/episode_formatters.py` | Rewrite `format_github_pr_episode()` to include review threads, linked issues, labels (rich format from §6) |
| `app/modules/context_graph/ingestion_service.py` | Call bridge writer after `add_episode()` succeeds |
| `app/modules/context_graph/tasks.py` | Fetch review comments + linked issues in `_pr_to_payload()` (partially done); add bridge writing step |
| `app/modules/context_graph/models.py` | Add `bridge_written` column to `ContextIngestionLog`; add `RawEvent` model |
| `app/modules/intelligence/tools/context_tools/get_project_context_tool.py` | Add `SearchFilters` for typed entity search |
| `app/modules/code_provider/github/github_provider.py` | Already has `get_pull_request(include_diff=True)` — no change needed |
| `app/modules/parsing/graph_construction/code_graph_service.py` | No change — bridge writer uses same Neo4j driver independently |
| `start.sh` | Ensure `external-event` queue is consumed (existing gap) |

### New files to create

| New file | Purpose |
|---|---|
| `app/modules/context_graph/entity_schema.py` | All Pydantic entity + edge models from §4 and §5 |
| `app/modules/context_graph/bridge_writer.py` | Diff hunk parser + Neo4j bridge relationship writer from §7 |
| `app/modules/context_graph/review_thread_grouper.py` | `group_review_threads()` from §6 |
| `app/modules/context_graph/deterministic_extractors.py` | `extract_issue_refs()`, `extract_ticket_from_branch()` from §8 |
| `app/modules/intelligence/tools/context_tools/get_change_history_tool.py` | New agent tool from §11.1 |
| `app/modules/intelligence/tools/context_tools/get_file_owner_tool.py` | New agent tool from §11.3 (Phase 2) |
| `app/modules/intelligence/tools/context_tools/get_decisions_tool.py` | New agent tool from §11.4 (Phase 2) |

---

## 16. Success Criteria

After ingesting 6 months of history:

```
GET /why → "processPayment was changed in PR #42 by yashkrishan to fix a
race condition causing double charges on retry. Part of Q3 Payment
Reliability epic. Original issue: #101 (users charged twice). Review
decision: used DB advisory locks instead of Redis locks (simpler, less
infra)."

GET /owner → "Primary: yashkrishan (4 PRs, last 3 months).
Secondary: alice (2 PRs, 6 months ago)."

GET /decisions → "PR #42: bob suggested Redis for distributed lock.
Decision: kept DB advisory locks. Rationale: avoids Redis dependency,
acceptable for current transaction volume. Tracked in Issue #901."
```

**One graph. Code + VCS + decisions. All connected. Extensible for any source.**
