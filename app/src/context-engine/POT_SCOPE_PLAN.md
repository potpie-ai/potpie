# Pot-Scoped Context Engine Plan

## Goal

Move `context-engine` from a single-repo, `project_id`-scoped model to a `pot`-scoped model:

- A `pot` is the top-level context boundary.
- A pot can contain multiple repositories.
- Context graphs are scoped to a pot, not to one repo/project.
- Ingestion writes context into a pot.
- Agent queries read context from a pot.
- Pot membership for multiple users is a future feature; design for it now, but do not block on it.

## Current State

Today the package is built around `project_id`:

- Domain ports accept `project_id` everywhere.
- HTTP APIs accept `project_id` in request bodies.
- CLI resolves the current git repo to a single `project_id`.
- MCP access control is keyed by `project_id`.
- Postgres ledger tables are keyed by `project_id`.
- Structural and episodic graph adapters store/query by `project_id`.

That means the current architecture assumes:

- one repo maps to one project scope
- ingestion scope and query scope are the same thing
- repo identity is enough to find the query boundary

Those assumptions break once one scope needs to hold multiple repos.

## Target Model

Use these concepts explicitly:

### Pot

The durable context boundary.

Suggested fields:

- `id`
- `name`
- `slug` or human-readable handle
- `created_by_user_id`
- `created_at`
- `updated_at`
- `visibility` or sharing mode later

### Pot Repository

Association between a pot and a repo.

Suggested fields:

- `id`
- `pot_id`
- `provider`
- `provider_host`
- `owner`
- `repo`
- `default_branch`
- `remote_url`
- `external_repo_id` if available
- `added_by_user_id`
- `added_at`
- unique constraint on `(pot_id, provider, owner, repo)`

Notes:

- `provider` should be a first-class field, not inferred from URL shape.
- `provider_host` matters for self-hosted GitLab, GitHub Enterprise, Bitbucket Server, etc.
- `external_repo_id` should be provider-native when available because names can change.

### Pot Member

Future feature. Do not implement full auth now, but reserve the model.

Suggested fields:

- `pot_id`
- `user_id`
- `role` such as owner/editor/viewer
- `invited_by_user_id`
- `created_at`

### Repository Context Source

Treat repo as a source of events inside a pot, not as the scope itself.

This distinction matters because:

- one pot can ingest from many repos
- search needs to span the pot by default
- some workflows will still need repo-level filtering inside a pot
- different repos may come from different providers with different APIs and identifiers

## Core Architectural Shift

The important move is:

- `project_id` currently means both "what repo is this?" and "what graph scope should I search?"
- in the new model those become separate concerns

Recommended separation:

- `pot_id`: query and storage boundary
- `repo_ref`: source repository identity for ingestion

Suggested `repo_ref` shape:

- `provider`
- `provider_host`
- `owner`
- `repo`
- `external_repo_id` if known

That gives you this rule:

- graph partition key = `pot_id`
- source identity = `repo_ref`

## Recommended Direction

Do not try to rename everything from `project` to `pot` in one pass.

Instead:

1. Introduce pot concepts alongside existing project-based code.
2. Make `project_id` an implementation detail or compatibility alias where needed.
3. Convert ingress and query surfaces to use `pot_id`.
4. Keep repo-specific ingestion logic using explicit repo metadata.
5. Remove old project terminology after the new flow is stable.

## Changes Needed By Layer

## 1. Domain Layer

Files impacted:

- `domain/ports/pot_resolution.py` (canonical; old `project_resolution` shim removed)
- `domain/ports/episodic_graph.py`
- `domain/ports/structural_graph.py`
- `domain/ports/ingestion_ledger.py`
- `domain/ports/jobs.py`

### Replace project resolution with pot resolution

Current `ResolvedProject` is too narrow:

- it resolves one `project_id`
- it gives one `repo_name`

Replace it with something closer to:

- `ResolvedPot`
- `ResolvedPotRepo`

Suggested shape:

```python
@dataclass(slots=True)
class ResolvedPotRepo:
    pot_id: str
    repo_id: str
    provider: str
    repo_name: str  # owner/repo
    remote_url: str | None = None
    default_branch: str | None = None
    ready: bool = True


@dataclass(slots=True)
class ResolvedPot:
    pot_id: str
    name: str | None
    repos: list[ResolvedPotRepo]
    ready: bool = True
```

Then split port responsibilities:

- resolve pot by `pot_id`
- resolve repo membership by git remote
- list repos inside a pot

Suggested port methods:

- `resolve_pot(pot_id)`
- `find_pots_for_repo(provider, provider_host, owner_repo)`
- `get_repo_in_pot(pot_id, provider, provider_host, owner_repo)`
- `list_pot_repos(pot_id)`
- `known_pot_ids()`

### Update graph ports

Every graph port currently takes `project_id`.

Change those to `pot_id`.

Also add optional repo filters where useful on query methods:

- `repo_name: str | None = None`

This allows:

- query whole pot by default
- narrow to one repo when needed

### Update ingestion ledger semantics

Today ledger rows are keyed by `project_id`.

That is no longer enough if a pot has multiple repos.

Use a compound identity:

- `pot_id`
- `source_type`
- `provider`
- `provider_host`
- `repo_name` or `repo_id`
- `source_id`

Without repo identity in the ledger, PR `#123` from two repos inside the same pot will collide.
Without provider identity, the same `owner/repo` path on GitHub and GitLab can also collide.

## 2. Application Layer

Files impacted:

- `application/use_cases/backfill_project.py`
- `application/use_cases/ingest_single_pr.py`
- `application/use_cases/ingest_merged_pr.py`
- `application/use_cases/ingest_episode.py`
- `application/use_cases/query_context.py`
- `application/services/pr_bundle.py`

### Rename use cases around scope, not project

Examples:

- `backfill_project_context` -> `backfill_pot_context`
- `search_project_context` -> `search_pot_context`

### Make ingestion repo-aware inside a pot

Current ingest flow assumes:

- resolve one project
- use its repo
- write to that same scope

New flow should be:

1. resolve pot
2. resolve repo membership inside that pot
3. route to the correct source-control adapter for that repo provider
4. fetch PR/MR/events from that repo
5. write graph entities into the pot partition
6. stamp repo provenance on entities/edges

### Preserve provenance in written graph data

Every ingested episode/entity should carry enough metadata to answer:

- which pot does this belong to
- which repo did this come from
- which provider object did it come from

Minimum metadata to stamp:

- `pot_id`
- `provider`
- `provider_host`
- `repo_name`
- `provider_object_type` such as PR/MR/issue/comment/event
- `provider_object_id`
- `pr_number` or MR number where applicable

## 3. Storage Layer

Files impacted:

- `adapters/outbound/postgres/models.py`
- `adapters/outbound/postgres/ledger.py`
- graph adapters in `adapters/outbound/graphiti/` and `adapters/outbound/neo4j/`

### Postgres schema changes

Add pot-oriented tables in the Potpie app schema:

- `pots`
- `pot_repositories`
- later `pot_members`

Then migrate context ledger tables away from pure `project_id` keys.

Recommended ledger shape:

- `pot_id` instead of `project_id`
- add `pot_repository_id` if possible
- otherwise add `provider`, `provider_host`, and `repo_name`

Suggested constraint changes:

- sync state unique on `(pot_id, pot_repository_id, source_type)` for repo-level sync jobs
- ingestion log unique on `(pot_id, pot_repository_id, source_type, source_id)`
- raw events unique on `(pot_id, pot_repository_id, source_type, source_id)`

If `pot_repository_id` is not available in the context-engine package, then use:

- `(pot_id, provider, provider_host, repo_name, source_type, source_id)`

### Graph partitioning

Graphiti and Neo4j should partition/query by `pot_id`.

Also stamp `repo_name` on relevant nodes and relationships.

Reason:

- search should span the pot
- structural queries may later support filtering by repo
- provenance and dedupe require repo identity
- provider-aware ingestion and debugging require provider identity too

### Graph migration strategy

Avoid destructive in-place graph rewrites first.

Safer rollout:

1. add support for writing/reading by `pot_id`
2. dual-read if necessary during transition
3. re-backfill pots from source systems
4. retire old project partitions later

If re-backfill is cheap enough, prefer re-backfill over graph data migration scripts.

## 4. HTTP API

Files impacted:

- `adapters/inbound/http/api/v1/context/router.py`
- Potpie-side wiring for auth and project access

### Change request bodies from project to pot scope

Primary API contract should become:

- `pot_id` for search and ingest

For repo-bound operations, also pass repo info when needed:

- `pot_repository_id`
- or `provider` + `provider_host` + `repo_name`

Examples:

```json
{ "pot_id": "...", "query": "how does auth caching work?" }
```

```json
{ "pot_id": "...", "provider": "github", "provider_host": "github.com", "repo_name": "owner/repo", "pr_number": 123 }
```

### Add explicit pot management endpoints later

Not required for the first pass inside `context-engine`, but the full architecture will need:

- create pot
- list pots visible to user
- add repo to pot
- list repos in pot

These likely belong in Potpie app APIs rather than this package.

### Compatibility strategy

For a transition period, support both:

- old `project_id`
- new `pot_id`

Only if needed to keep current callers alive. If there are few callers, cut directly to `pot_id`.

## 5. CLI

Files impacted:

- `adapters/inbound/cli/main.py`
- `adapters/inbound/cli/git_project.py`
- `adapters/inbound/cli/README.md`

### New CLI shape

Your proposed direction is correct:

```bash
context-engine add .
```

Recommended initial commands:

- `context-engine pot create <name>`
- `context-engine pot list`
- `context-engine add <path>`
- `context-engine search <pot-or-infer> "query"`
- `context-engine ingest ... --pot <pot_id>`

### Behavior of `context-engine add .`

For the current repo:

1. verify cwd is inside git repo
2. read `origin` remote
3. normalize to `owner/repo`
4. parse provider and host from remote URL
5. determine target pot
6. call Potpie API to attach repo to that pot

Supported examples to plan for:

- `git@github.com:owner/repo.git`
- `https://github.com/owner/repo.git`
- `git@gitlab.com:group/repo.git`
- `https://gitlab.company.com/group/repo.git`

This means `git_project.py` should really become a generic git remote parser rather than a GitHub-specific resolver.

Important product decision:

- if the user has one active pot, use it
- otherwise require `--pot` or interactive selection

Since you want to skip user-sharing for now, the simplest first version is:

- each user has a default active pot in CLI config
- `context-engine pot use <pot_id>` sets active pot
- `context-engine add .` adds current repo to active pot

That avoids ambiguity and keeps the command ergonomic.

### CLI local config

Add a small local state file alongside stored credentials:

- active `pot_id`
- maybe last used pots

Do not infer pot solely from repo unless a repo can belong to only one pot. That assumption will become false.

Also do not infer provider semantics from `owner/repo` alone. `github.com/org/app` and `gitlab.company.com/org/app` are different sources.

## 6. MCP

Files impacted:

- `adapters/inbound/mcp/project_access.py`
- MCP tool request models

Change MCP access from project allowlist to pot allowlist:

- `CONTEXT_ENGINE_MCP_ALLOWED_POTS`
- `CONTEXT_ENGINE_MCP_TRUST_ALL_POTS`

Tool inputs should accept `pot_id`.

If tools support repo narrowing, make it optional rather than changing the main scope back to repo.

## 7. Potpie Integration

This package alone should not own all pot lifecycle logic.

Recommended responsibility split:

### Potpie app owns

- pot CRUD
- pot membership and auth
- repo attachment to pot
- user-visible list/search/select APIs
- background job orchestration policy

### context-engine owns

- ingest/query execution using `pot_id`
- graph persistence
- graph query APIs
- repo-aware provenance inside the pot

This keeps tenancy and product state in Potpie, and keeps `context-engine` focused on context operations.

## Suggested Incremental Rollout

## Phase 0: Naming and data model decision

Decide these before coding:

- Is `pot` the permanent user-facing term everywhere?
- Can one repo belong to multiple pots?
- Should search always span all repos in a pot by default?
- Do you want repo-filtered search in v1?
- Is there always an active pot in CLI state?

Recommended answers:

- yes, `pot` is the primary term
- yes, one repo can belong to multiple pots
- yes, search spans the whole pot by default
- yes, repo filter should exist but be optional
- yes, CLI should support an active pot

## Phase 1: Introduce pot models and resolution ports

Implement:

- pot and pot-repo DB models in Potpie
- new resolution port interfaces
- compatibility adapters that can still derive a pot-like scope from current project data if needed

Deliverable:

- code can resolve `pot_id` and list repos in a pot

## Phase 2: Change write path to pot scope

Implement:

- ledger schema updates
- graph adapters accept `pot_id`
- ingestion writes `pot_id`, `provider`, `provider_host`, and `repo_name`
- source-control adapter selection becomes provider-aware

Deliverable:

- new ingests land in pot-partitioned graph data

## Phase 3: Change read path to pot scope

Implement:

- query use cases accept `pot_id`
- HTTP/MCP surfaces accept `pot_id`
- optional repo filter

Deliverable:

- agents query pots instead of projects

## Phase 4: Add CLI pot workflow

Implement:

- `pot create`
- `pot list`
- `pot use`
- `add <path>`

Deliverable:

- current repo can be attached to active pot with `context-engine add .`

## Phase 5: Remove project terminology

After callers are migrated:

- ~~rename `project_resolution` module~~ (removed; use `pot_resolution` only)
- remove `project_id` fields from public APIs
- update docs/tests/examples

## Important Design Decisions

## A. Keep repo identity first-class

Do not flatten all repo information away once data enters a pot.

If you do that, you lose:

- dedupe safety
- provenance
- repo-specific filtering
- correct PR identity
- provider-specific identity
- the ability to support mixed GitHub/GitLab pots cleanly

## B. Do not make pot resolution depend only on current git remote

That only works when one repo maps to one scope.

Once one repo can live in multiple pots, CLI needs:

- an active pot
- or explicit `--pot`

## C. Do not model sync purely at pot level

Backfill and PR ingest are operationally repo-scoped, even if the graph is pot-scoped.

That means:

- query partition is pot-level
- operational sync state is pot+repo-level

## D. Expect repo collisions in PR numbering

PR numbers or merge request numbers are only unique inside a repo on a provider.

Any ledger key, graph entity key, or cache key that currently uses:

- `project_id + pr_number`

should become something closer to:

- `pot_id + provider + provider_host + repo_name + provider_object_number`

## E. Keep auth and tenancy outside this package

The future multi-user model should be enforced in Potpie APIs and wiring, not duplicated inside graph adapters.

`context-engine` should receive already-authorized `pot_id` access.

## Concrete Refactor Map

This is the rough file-by-file direction.

### Rename / replace

- `domain/ports/project_resolution.py` -> removed; `domain/ports/pot_resolution.py` is canonical
- `ResolvedProject` -> `ResolvedPot`, `ResolvedPotRepo`
- `search_project_context` -> `search_pot_context`
- `backfill_project_context` -> `backfill_pot_context`

### Signature changes

- all `project_id: str` graph methods -> `pot_id: str`
- ingestion methods also take `repo_name` or resolved repo object
- ledger methods key by `pot_id` + repo

### New supporting modules

- `domain/ports/pot_catalog.py` or equivalent
- CLI active pot config helper
- Potpie adapter for pot/repo resolution

## Open Questions

You should settle these before implementation:

1. Is a pot allowed to contain non-GitHub repos later, or do you want GitHub-only assumptions in the first schema?
2. Can the same repo be attached to multiple pots?
3. Should a user have exactly one default pot, or many named pots with explicit selection?
4. Does `context-engine add .` create a pot automatically if no active pot exists, or fail with guidance?
5. Are structural queries like file ownership intended to search across repos, or should they require a repo filter when file paths collide?

Recommended defaults:

1. schema should be provider-agnostic, implementation can start GitHub-only and GitLab-next
2. yes
3. many pots, one active pot in CLI
4. fail with guidance or suggest `pot create`; do not auto-create silently
5. search across pot by default, but allow repo filter for ambiguous file-path queries

## Recommended First Implementation Slice

If you want the safest path, start here:

1. Add Potpie DB models for `pots` and `pot_repositories`.
2. Add a pot resolution adapter and keep the existing project resolver untouched for now.
3. Introduce `pot_id` plus provider-aware repo identity into graph adapters and ledger tables.
4. Update search and ingest use cases to operate on `pot_id`.
5. Add source-control adapter routing by provider.
6. Add CLI active-pot state and `context-engine add .`.

That slice gets you the core architecture change without taking on user-sharing yet.

## Bottom Line

The architecture should move from:

- repo/project as the context scope

to:

- pot as the context scope
- repo as a source within that scope
- provider identity as part of the repo source model

If you preserve that separation consistently across ports, ledger keys, graph partitioning, and CLI UX, the later multi-user sharing model will fit cleanly instead of forcing another rewrite.
