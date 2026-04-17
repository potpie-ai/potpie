# Pot Tenancy Implementation Plan

## Goal

Implement a pot-first tenancy model where:

- a pot is the top-level user-facing context boundary
- one user can create many pots
- one pot can contain many repositories
- one pot can contain many integrations
- many users can belong to one pot with a role
- a pot is not linked to a Potpie parsing project
- a user can use their API token to query and mutate context graph data for pots they can access

This plan focuses on changes needed around `app/src/context-engine`, while making the Potpie-owned tenancy and auth boundaries explicit.

## Target Model

### Core entities

#### Pot

Top-level context scope.

Suggested fields:

- `id`
- `display_name`
- `slug`
- `created_by_user_id`
- `created_at`
- `updated_at`
- `archived_at` nullable

#### Pot Member

User-to-pot mapping with an authorization role.

Suggested fields:

- `pot_id`
- `user_id`
- `role`
- `invited_by_user_id` nullable
- `created_at`
- `updated_at`

Suggested enum:

- `owner`
- `admin`
- `read_only`

#### Pot Repository

Repository attached to a pot. A repo is a source inside a pot, not the scope itself.

Suggested fields:

- `id`
- `pot_id`
- `provider`
- `provider_host`
- `owner`
- `repo`
- `external_repo_id` nullable
- `remote_url` nullable
- `default_branch` nullable
- `added_by_user_id`
- `created_at`

Suggested uniqueness:

- unique on `(pot_id, provider, provider_host, owner, repo)`

#### Pot Integration

Integration attachment for a pot.

Suggested fields:

- `id`
- `pot_id`
- `integration_type`
- `provider`
- `provider_host` nullable
- `external_account_id` nullable
- `config_json` or reference to secured config
- `created_by_user_id`
- `created_at`
- `updated_at`

## Ownership Boundary

### Potpie owns

Potpie should own all tenancy and auth concerns:

- pot CRUD
- pot membership
- role enforcement
- repo attach/detach
- integration attach/detach
- user API token validation
- resolution of "which pots can this user access"

### context-engine owns

`potpie` should own execution inside an already-authorized pot:

- ingest into a `pot_id`
- query from a `pot_id`
- preserve repo-aware provenance within a pot
- optionally enforce capability checks passed in from Potpie for mutation routes

`potpie` should not become the system of record for user membership or token auth.

## Current Gaps

The package has partial pot support already, but it does not fully match the target model.

### Gap 1: multi-repo pots are modeled but not executed

Several use cases still resolve a pot and immediately use `primary_repo()`.

That preserves the old one-project/one-repo assumption in the write path.

Impacted areas include:

- `application/use_cases/backfill_pot.py`
- `application/use_cases/ingest_single_pr.py`
- `application/services/ingestion_submission_service.py`
- `application/use_cases/context_graph_jobs.py`

### Gap 2: public APIs still leak `project_id`

The package still has request models and intelligence interfaces that use `project_id`.

Impacted areas include:

- `adapters/inbound/http/api/v1/context/router.py`
- `domain/intelligence_models.py`
- `application/services/context_resolution.py`
- `domain/ports/intelligence_provider.py`
- `adapters/outbound/intelligence/*`

### Gap 3: access control is not role-based

Current access is effectively:

- HTTP: "can the pot resolve?"
- MCP: "is the pot in an env allowlist?"

That is not enough for `owner`, `admin`, and `read_only`.

### Gap 4: CLI pot selection still assumes one repo maps to one pot

Current CLI resolution is based on:

1. active pot
2. repo-to-pot env map
3. failure

That is not sufficient if the same repo can exist in multiple pots.

### Gap 5: standalone auth is not user-token auth

The CLI client already sends `X-API-Key` to Potpie, but standalone `potpie` HTTP still uses a single shared env secret.

For your target model, user-scoped API-token auth should happen in Potpie and Potpie should call into `potpie` with already-authorized pot access.

## Recommended Architecture

## 1. Pot as the only public scope

All user-facing and API-facing context operations should take `pot_id`.

Repo identity should be a secondary selector when needed:

- `repo_name`
- or a stronger provider-aware `repo_ref`

Rule:

- graph partition key = `pot_id`
- source identity = `provider + provider_host + repo identity`

## 2. Separate authorization from execution

Potpie should authenticate the API token, load the user's pot memberships, and determine whether the caller can:

- read pot context
- ingest into the pot
- reset the pot
- attach repos/integrations
- manage members

`potpie` should receive only authorized requests.

For defense in depth, `potpie` may optionally accept a small capability object from Potpie, such as:

```python
@dataclass(slots=True)
class PotAccess:
    pot_id: str
    can_read: bool
    can_ingest: bool
    can_reset: bool
    role: str
```

But `potpie` should not query membership tables directly.

## 3. Repo-aware write paths

Any write flow that talks to source control must be explicit about which repo inside the pot is being targeted.

Examples:

- backfill all repos in a pot
- backfill one repo inside a pot
- ingest PR `123` from `owner/repo` inside a pot

Do not use `primary_repo()` as the default operational model.

## 4. Search spans the pot by default

Read paths should:

- search all repos in the pot by default
- allow optional repo filtering
- preserve repo metadata in results where useful

This is already largely aligned in the query API and storage model.

## Proposed API Shape

## Potpie API

These endpoints belong in Potpie, not in `potpie`.

### Pot lifecycle

- `POST /api/v2/context/pots`
- `GET /api/v2/context/pots`
- `GET /api/v2/context/pots/{pot_id}`
- `PATCH /api/v2/context/pots/{pot_id}`

### Pot membership

- `GET /api/v2/context/pots/{pot_id}/members`
- `POST /api/v2/context/pots/{pot_id}/members`
- `PATCH /api/v2/context/pots/{pot_id}/members/{user_id}`
- `DELETE /api/v2/context/pots/{pot_id}/members/{user_id}`

### Pot repositories

- `GET /api/v2/context/pots/{pot_id}/repositories`
- `POST /api/v2/context/pots/{pot_id}/repositories`
- `DELETE /api/v2/context/pots/{pot_id}/repositories/{repository_id}`

### Pot integrations

- `GET /api/v2/context/pots/{pot_id}/integrations`
- `POST /api/v2/context/pots/{pot_id}/integrations`
- `DELETE /api/v2/context/pots/{pot_id}/integrations/{integration_id}`

### Context execution

These continue to call `potpie`, but only with already-authorized `pot_id` access:

- `POST /api/v2/context/query/search`
- `POST /api/v2/context/query/change-history`
- `POST /api/v2/context/query/file-owners`
- `POST /api/v2/context/query/decisions`
- `POST /api/v2/context/ingest`
- `POST /api/v2/context/reset`

## Role semantics

Suggested capability matrix:

- `owner`
  - full access
  - can manage members
  - can manage repos/integrations
  - can ingest/reset/query

- `admin`
  - can manage repos/integrations
  - can ingest/reset/query
  - cannot transfer ownership unless explicitly allowed later

- `read_only`
  - can query
  - cannot ingest
  - cannot reset
  - cannot manage repos, integrations, or members

## Phased Implementation

## Phase 1: lock the public model

Goal: stop mixing project scope and pot scope.

Changes:

1. Declare `pot_id` as the only public scope for new APIs.
2. Mark all remaining `project_id` request fields and helpers as compatibility-only.
3. Add Potpie schema for:
   - `pots`
   - `pot_members`
   - `pot_repositories`
   - `pot_integrations`
4. Add role enum and membership constraints.

Deliverables:

- migration spec in Potpie
- API contract doc for pot membership and repo/integration attachment
- deprecation list for `project_id` in `potpie`

## Phase 2: remove project terminology from context-engine public surfaces

Goal: make `potpie` externally pot-only.

Changes:

1. Rename request models from `project_id` to `pot_id` in:
   - `domain/intelligence_models.py`
   - `adapters/inbound/http/api/v1/context/router.py`
   - `domain/ports/intelligence_provider.py`
   - `adapters/outbound/intelligence/*`
2. Rename methods and docs that still refer to project scope.
3. Keep a short-lived compatibility adapter only where Potpie still needs it.

Deliverables:

- no new public `project_id` fields in `app/src/context-engine`
- docs updated in CLI and package README

## Phase 3: make write paths repo-aware inside a pot

Goal: support many repositories per pot operationally.

Changes:

1. Replace `primary_repo()`-driven workflows with explicit repo targeting.
2. Update use cases to accept either:
   - `pot_id + repo_name`
   - or `pot_id + RepoRef`
3. For pot-wide backfill, iterate all repos in `ResolvedPot.repos`.
4. For single PR ingest, require repo identity when the pot contains more than one repo.
5. Ensure all ledger writes remain keyed by:
   - `pot_id`
   - `provider`
   - `provider_host`
   - `repo_name`
   - `source_type`
   - `source_id`

Impacted files:

- `application/use_cases/backfill_pot.py`
- `application/use_cases/ingest_single_pr.py`
- `application/services/ingestion_submission_service.py`
- `application/use_cases/context_graph_jobs.py`
- any Potpie handler that enqueues context jobs

Deliverables:

- multi-repo backfill works
- repo-specific ingest works within a pot
- no implicit primary-repo dependency in write paths

## Phase 4: integrate Potpie membership and role enforcement

Goal: user-to-pot mapping controls context behavior.

Changes in Potpie:

1. Authenticate user via API token.
2. Resolve user's memberships for requested `pot_id`.
3. Enforce role-based authorization before calling `potpie`.
4. Restrict pot listing to visible pots for that user.

Optional changes in `potpie`:

1. Add capability-aware mutation guards for embedded use.
2. Reject mutation attempts when Potpie passes `can_ingest=False` or `can_reset=False`.

Deliverables:

- user token can query only allowed pots
- `read_only` users cannot mutate context
- `admin` and `owner` can ingest/reset

## Phase 5: rework CLI around explicit active pot selection

Goal: match the new many-repos-per-pot model.

Changes:

1. Keep `potpie pot use <pot_id>` as the primary UX.
2. Downgrade repo-to-pot env mapping to compatibility mode.
3. Stop presenting Potpie parsing projects as the normal pot selection path.
4. Add repo attachment commands through Potpie APIs, for example:
   - `potpie pot repo add .`
   - `potpie pot repo list`
5. For ambiguous repo membership, require explicit `--pot`.

CLI rules:

- active pot is the default selection mechanism
- git remote may help propose a repo attachment
- git remote must not be the canonical pot selector in a many-to-many model

Deliverables:

- CLI language is pot-first
- `pot projects` is deprecated or removed
- repo attachment is explicit

## Phase 6: clean up MCP and standalone behavior

Goal: make non-Potpie entrypoints consistent with tenancy.

Changes:

1. MCP should eventually use Potpie-authorized pot access instead of raw env allowlists where possible.
2. Standalone `potpie` should be treated as a dev/integration mode, not the primary tenancy model.
3. If standalone mode remains, document clearly that its auth is not the Potpie user-membership model.

Deliverables:

- no confusion between dev-mode allowlists and production authorization

## Code Changes by Module

## `domain/ports/pot_resolution.py`

Keep:

- `ResolvedPot`
- `ResolvedPotRepo`
- `RepoRef`

Change:

- stop encouraging `primary_repo()` as a general operational shortcut
- add helpers that make repo selection explicit

Suggested additions:

- `require_repo_in_pot(pot_id, ref)`
- `list_pot_repos(pot_id)`
- `find_pots_for_repo(ref)`

## `application/use_cases/*`

Change all write flows to use explicit repo context.

Rules:

- pot-wide workflows iterate repos
- repo-scoped workflows require repo identity
- search remains pot-wide by default

## `domain/intelligence_models.py` and intelligence provider ports

Rename:

- `project_id` -> `pot_id`

Optional enhancement:

- add optional `repo_name` to deterministic context resolution requests

## `adapters/inbound/http/api/v1/context/router.py`

Change:

- remove remaining public `project_id` request bodies
- use `pot_id` consistently
- do not describe pots as project ids in error text

## `adapters/inbound/cli/*`

Change:

- deprecate project-centric help text
- center commands on:
  - `pot create`
  - `pot list`
  - `pot use`
  - `pot repo add`
  - `pot repo list`

## Data Migration Notes

If old project-scoped context data already exists, migration should be staged.

Recommended approach:

1. Add new pot and pot-membership tables first.
2. Add compatibility mapping from legacy project-backed pots where needed.
3. Switch new ingest/query traffic to `pot_id`.
4. Re-backfill multi-repo pots where required.
5. Retire legacy project-only surfaces later.

Do not attempt a destructive one-shot migration if project-backed traffic is still live.

## Testing Plan

## Unit tests in `app/src/context-engine/tests/unit`

Add or update tests for:

- multi-repo pot resolution
- backfill across many repos in one pot
- single-PR ingest requiring repo identity when pot has multiple repos
- search across a full pot with optional repo filtering
- CLI active-pot behavior when repo-to-pot mapping is ambiguous
- mutation authorization behavior when capabilities are denied

## Potpie integration tests

Add tests for:

- user creates a pot
- user adds a repo to a pot
- user adds another user to a pot as `read_only`
- `read_only` user can query but cannot ingest/reset
- `admin` user can ingest/reset but not remove owner
- one repo attached to multiple pots
- one user belonging to many pots

## Recommended Delivery Order

1. Potpie schema and role model
2. Potpie APIs for pot membership and repo attachment
3. `potpie` removal of public `project_id`
4. `potpie` repo-aware write path refactor
5. Potpie auth enforcement using user API token
6. CLI cleanup and repo management commands
7. MCP and standalone cleanup

## Explicit Non-Goals

These should not be bundled into the first rollout:

- full cross-provider integration abstraction for every source system
- interactive CLI pot selection UI
- automatic pot creation from repo discovery
- moving all tenancy state into `potpie`

## Final Direction

The correct long-term model is:

- Potpie owns tenants, memberships, roles, and API-token auth
- `potpie` executes within an authorized `pot_id`
- a pot is independent of parsing projects
- repositories and integrations are attached resources inside a pot
- users interact with context graph via their own API token and their authorized pots

That direction matches your stated concept and requires finishing the pot migration already underway, while explicitly removing the remaining project-era assumptions from the package.
