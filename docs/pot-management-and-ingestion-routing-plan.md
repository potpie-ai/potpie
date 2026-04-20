# Pot Management And Ingestion Routing Plan

## Purpose

Pots are the product boundary for context. A pot groups the users, repositories, integrations, and provider scopes whose data should land in one context graph partition. The UI at `/pots` should manage that boundary, while ingestion should use the same database records to decide which pot an incoming GitHub, Linear, or future provider event belongs to.

This plan covers the Potpie API, embedded `context-engine`, event routing, and the UI in `/Users/nandan/Desktop/Dev/potpie-ui`.

## Current State

### Backend persistence

The repository already has the first version of pot tenancy in Postgres:

- `context_graph_pots`
- `context_graph_pot_members`
- `context_graph_pot_repositories`
- `context_graph_pot_integrations`

The models live in:

- `app/modules/context_graph/context_graph_pot_model.py`
- `app/modules/context_graph/context_graph_pot_member_model.py`
- `app/modules/context_graph/context_graph_pot_repository_model.py`
- `app/modules/context_graph/context_graph_pot_integration_model.py`

The migrations are:

- `app/alembic/versions/20260406_160000_context_graph_pots.py`
- `app/alembic/versions/20260407_120000_pot_tenancy_members_repos.py`

The API routes are in `app/modules/context_graph/context_pot_routes.py` and are mounted in both:

- `/api/v1/context/pots` with Firebase auth
- `/api/v2/context/pots` with API key auth

Existing route coverage:

- create, list, get, patch/archive pots
- list, add, patch, delete members
- list, add, delete repositories
- list, add, delete integrations

### Role model

The current code has `owner`, `admin`, and `read_only` in `app/modules/context_graph/pot_member_roles.py`. That should be simplified now. The product model has exactly two roles:

- `owner`
- `user`

Owners can manage users, repositories, integrations, and pot settings. Users can access the pot context and manually submit raw notes, links, and content for ingestion, but cannot manage the pot. Do not keep `admin` or `read_only` as active product roles.

### context-engine

`context-engine` already treats `pot_id` as the graph partition key. The important boundary is `PotResolutionPort` in `app/src/context-engine/domain/ports/pot_resolution.py`.

Potpie wires this port through `app/modules/context_graph/wiring.py`:

- `SqlalchemyPotResolution` for worker/session-wide access
- `UserScopedContextGraphPotResolution` for user-scoped access

This means context-engine can already resolve a pot into attached repositories from Postgres.

### Ingestion

The main ingestion entry point is `DefaultIngestionSubmissionService` in `app/src/context-engine/application/services/ingestion_submission_service.py`. It persists canonical events into `context_events`, then either runs inline or enqueues context-graph jobs.

Supported ingestion families today:

- `raw_episode`
- `github_merged_pr`
- `agent_reconciliation`

The event store and step tables already exist in `app/src/context-engine/adapters/outbound/postgres/models.py`:

- `context_events`
- `context_episode_steps`
- `context_reconciliation_runs`

The major gap is upstream routing. Potpie's generic webhook bus handler in `app/modules/event_bus/handlers/webhook_handler.py` still maps GitHub merged PR events by `Project.repo_name` and queues `context_graph_ingest_pr(project.id, ...)`. That is the old project-first path. It needs to route by pot repository attachments.

The standalone context-engine GitHub webhook in `app/src/context-engine/adapters/inbound/http/webhooks/integrations/github.py` also still expects `CONTEXT_ENGINE_REPO_TO_POT`. That is acceptable for standalone mode, but Potpie-hosted mode should use database-backed pot source records.

### UI

The UI route exists at:

- `/Users/nandan/Desktop/Dev/potpie-ui/app/(main)/pots/page.tsx`

Current UI support:

- list pots
- create pot with optional primary repository
- archive pot
- expand pot and list repositories
- manually add `owner/repo`
- remove repositories

Current service support is in:

- `/Users/nandan/Desktop/Dev/potpie-ui/services/BranchAndRepositoryService.ts`

The UI does not yet expose:

- members/users
- owner-only add/remove/update user access
- integrations attached to a pot
- GitHub repository picker flow for adding repos
- Linear team picker flow for adding a team/source to a pot
- ingestion event status per pot/source

## Target Model

### Pot

`context_graph_pots` remains the top-level scope.

Keep:

- `id`
- `display_name`
- `slug`
- `created_by_user_id`
- `primary_repo_name` as a compatibility/display field only
- `created_at`
- `updated_at`
- `archived_at`

Direction:

- Do not use `primary_repo_name` as a source of truth.
- Use `context_graph_pot_repositories` and pot source/integration attachments for routing.

### Pot members

Use `context_graph_pot_members` as the user-to-pot mapping table.

Near-term role values:

- `owner`
- `user`

Permission matrix:

| Capability | owner | user |
| --- | --- | --- |
| See pot | yes | yes |
| Query/search pot context | yes | yes |
| View repos/integrations | yes | yes |
| Add/remove repos | yes | no |
| Add/remove integrations | yes | no |
| Add/remove users | yes | no |
| Accept invite | yes | yes |
| Manually ingest raw notes/content/links | yes | yes |
| Archive pot | yes | no |
| Manual destructive reset | yes | no |

Implementation detail:

- Add `POT_ROLE_USER = "user"`.
- Remove `admin` and `read_only` from the active role set.
- Migrate existing `admin` and `read_only` member rows to `user`.
- Keep a short-term defensive read compatibility path only if needed for zero-downtime deployment, but do not expose legacy roles in API responses or UI.
- Change UI labels and API documentation to owner/user only.

### Repositories

`context_graph_pot_repositories` remains the source of truth for code-host repository membership.

Keep:

- `id`
- `pot_id`
- `provider`
- `provider_host`
- `owner`
- `repo`
- `external_repo_id`
- `remote_url`
- `default_branch`
- `added_by_user_id`
- `created_at`

Add or backfill where possible:

- provider installation/account identifier for GitHub App routing, either in a new nullable column or in a generalized source table
- webhook status / sync health if repositories become first-class sources in the same way Linear teams are

Routing rule:

- Incoming GitHub repo event resolves by `(provider, provider_host, owner, repo)` and optionally `external_repo_id`.
- If one pot matches, submit ingestion for that pot.
- If multiple pots match, submit one ingestion event per pot unless a webhook/integration attachment narrows the target pot.
- If no pot matches, acknowledge the webhook as ignored and record enough logs to debug setup.

### Integrations and sources

The current `context_graph_pot_integrations` table is too thin for real provider scopes. It stores `integration_type`, `provider`, `external_account_id`, and `config_json`, but it does not link to the existing `integrations` table and does not represent source scopes like "Linear team".

Recommended evolution:

1. Keep `context_graph_pot_integrations` as the pot-to-connection attachment.
2. Add `integration_id` as a nullable FK to `integrations.integration_id`.
3. Add a normalized pot source table:

```text
context_graph_pot_sources
- id
- pot_id
- integration_id nullable
- provider                # github, linear, jira, slack, ...
- source_kind             # repository, issue_tracker_team, channel, project, ...
- scope_json              # provider-specific stable scope
- scope_hash              # deterministic hash for dedupe
- sync_enabled
- sync_mode
- webhook_status
- last_sync_at
- last_error
- health_score
- added_by_user_id
- created_at
- updated_at
```

Why add `context_graph_pot_sources` instead of overloading `context_graph_pot_integrations`:

- A connection is account-level credentials.
- A source is what data enters a pot.
- One Linear connection can expose many teams.
- One GitHub connection can expose many repos.
- Detaching a source from a pot should not revoke the connection.

This mirrors the existing project-oriented `project_sources` model in `app/src/integrations/integrations/adapters/outbound/postgres/project_source_model.py`, but scopes it to pots instead of legacy projects.

Repository handling should immediately mirror into `context_graph_pot_sources` with `provider=github`, `source_kind=repository`, and `scope_json={"owner": "...", "repo": "...", "repo_name": "owner/repo"}`. Keep `context_graph_pot_repositories` because context-engine already uses it for repo resolution, but create/update/delete the matching pot source in the same transaction.

This immediate mirror is important for the next repo-processing slice:

- adding a repo to a pot should create a source record
- the source can drive sandbox/index/parse jobs
- agents can use the indexed codebase as evidence during ingestion and context resolution
- source health can track continuous reindexing and parse status independently from pot membership

## Backend API Plan

### Phase 1: tighten existing pot APIs

Files:

- `app/modules/context_graph/pot_member_roles.py`
- `app/modules/context_graph/pot_access.py`
- `app/modules/context_graph/context_pot_routes.py`
- `app/alembic/versions/*`

Tasks:

1. Add `user` role support and remove `admin/read_only` from the active role model.
2. Migrate existing `admin` and `read_only` rows to `user`.
3. Change `AddMemberBody` and `PatchMemberBody` validation to accept `owner/user` semantics, with owner assignment blocked until explicit ownership transfer exists.
4. Enrich `GET /pots/{pot_id}/members` with user display fields needed by UI, such as email/name/avatar if available from `users`.
5. Keep owner as the only role that can add/remove members.
6. Add tests for owner/user access behavior.

### Phase 1b: email invitations

Users are added by email, not by raw user id.

Add a pending invitation model:

```text
context_graph_pot_invitations
- id
- pot_id
- email
- role                    # user for now; owner only after ownership transfer exists
- invited_by_user_id
- accepted_by_user_id nullable
- token_hash
- status                  # pending, accepted, revoked, expired
- expires_at
- created_at
- accepted_at
```

API shape:

```text
GET    /api/v1/context/pots/{pot_id}/invitations
POST   /api/v1/context/pots/{pot_id}/invitations
DELETE /api/v1/context/pots/{pot_id}/invitations/{invitation_id}
POST   /api/v1/context/pot-invitations/{token}/accept
```

Behavior:

- Owner invites by email.
- If the email already belongs to a user, accepting adds that user to `context_graph_pot_members`.
- If the email does not belong to a user yet, the invite remains pending; after signup/login with that email, accepting adds the new user to the pot.
- Store only a token hash, never the raw token.
- Do not add a member row until acceptance.
- The only assignable invite role for now is `user`.

### Phase 2: formalize pot source APIs

Add routes under the existing context router:

```text
GET    /api/v1/context/pots/{pot_id}/sources
POST   /api/v1/context/pots/{pot_id}/sources/github/repository
POST   /api/v1/context/pots/{pot_id}/sources/linear/team
PATCH  /api/v1/context/pots/{pot_id}/sources/{source_id}
DELETE /api/v1/context/pots/{pot_id}/sources/{source_id}
POST   /api/v1/context/pots/{pot_id}/sources/{source_id}/sync
```

The v2 API-key router should expose the same read and mutation shape where it makes sense for CLI/API users.

For GitHub:

- Use the existing repository listing APIs/user GitHub auth where possible.
- Persist into `context_graph_pot_repositories`.
- Always create or update the mirrored `context_graph_pot_sources` row in the same transaction.
- Enqueue the repo sandbox/index/parse preparation job once that pipeline is available.

For Linear:

- Reuse existing connected integrations from `integrations`.
- Reuse the Linear teams fetch pattern in `app/src/integrations/integrations/adapters/inbound/http/sources_router.py`.
- Persist team attachment into `context_graph_pot_sources` with `provider=linear`, `source_kind=issue_tracker_team`, and stable `scope_json`.

### Phase 3: expose pot ingestion status

The context-engine router already supports:

- `GET /api/v1/context/pots/{pot_id}/events`
- `GET /api/v1/context/events/{event_id}`

Make the UI consume these rather than inventing new status tables.

Add source filters if needed:

```text
GET /api/v1/context/pots/{pot_id}/events?provider=github&source_kind=repository
```

If the current `EventListFilters` does not include provider/source filters, add them in the context-engine event query service.

## Ingestion Routing Plan

### Principle

Every inbound event must be normalized before ingestion. The normalizer is responsible for identifying the source and resolving the pot. The ingestion agent should receive a canonical event with a resolved `pot_id`; it should not search tenancy tables itself.

Manual raw notes, links, and pasted content are first-class ingestion inputs. Both owners and users may submit them through the CLI and the UI. These submissions still resolve to a pot, persist a canonical `raw_episode` or agent-backed event, and then flow through the same event store/status APIs as webhook-driven ingestion.

### New service boundary

Add a Potpie-side service, for example:

```text
app/modules/context_graph/ingestion_routing.py
```

Responsibilities:

- normalize provider payloads into source identity
- resolve source identity to one or more pot IDs
- validate source is attached and enabled
- create `IngestionSubmissionRequest`
- call `container.ingestion_submission(db).submit(...)`
- return ignored/queued/done/error result objects

Suggested interface:

```python
class PotIngestionRouter:
    def route_github_pull_request(self, payload: dict, event_type: str) -> list[RouteResult]: ...
    def route_linear_issue_event(self, payload: dict, integration_id: str | None) -> list[RouteResult]: ...
```

### GitHub merged PR path

Replace project lookup in `WebhookEventHandler._process_github_webhook`.

Current behavior:

- find `Project` by `Project.repo_name == repo_name`
- enqueue `context_graph_ingest_pr(project.id, pr_number)`

Target behavior:

1. Extract `repository.full_name`, `repository.id`, PR number, action, merged flag.
2. Ignore non-merged PR close events.
3. Resolve matching pot repository rows:
   - primary: `external_repo_id == repository.id` when present
   - fallback: `(provider="github", provider_host="github.com", owner, repo)`
4. For each matching pot:
   - create `IngestionSubmissionRequest` with `ingestion_kind=github_merged_pr`
   - set `pot_id`
   - set `repo_name`
   - set `source_channel="webhook"`
   - set `source_system="github"`
   - set `source_event_id` from GitHub delivery id when available
   - payload includes `pr_number`, `repo_name`, `repository_id`, and `is_live_bridge=True`
5. Submit async by default.
6. Return a result with all routed pot IDs and event IDs.

This keeps the ingestion details in `context-engine` and moves only routing into Potpie.

### Linear team path

Initial target:

- ingest Linear issue create/update/comment events for teams attached to a pot.

Routing:

1. Resolve the `Integration` row by `integration_id` when available.
2. Extract Linear `team.id` from payload.
3. Find enabled `context_graph_pot_sources` where:
   - `provider="linear"`
   - `source_kind="issue_tracker_team"`
   - `integration_id` matches when available
   - `scope_json.team_id` matches the event team id
4. Submit an `agent_reconciliation` ingestion event to the resolved pot.
5. Include enough payload for the ingestion agent to fetch details through Linear read tools.

The first implementation can persist events and use the existing generic agent planner. A later provider-specific deterministic planner can produce better Linear episodes.

### Ambiguity policy

Same source attached to multiple pots is allowed. In that case, route one event to each pot. The graph partition key is the pot, so duplicate ingestion across pots is correct.

Do not silently choose a "primary" pot.

This fan-out behavior is decided product behavior for GitHub repositories and future shared sources.

### Standalone context-engine

Keep env-map based standalone routing:

- `CONTEXT_ENGINE_REPO_TO_POT`
- `CONTEXT_ENGINE_POTS`

This is useful outside Potpie. Potpie-hosted ingestion should use database-backed routing.

## context-engine Changes

### Keep context-engine pot-scoped

Do not make context-engine own user membership. Potpie authorizes and resolves tenancy. Context-engine executes within a resolved `pot_id`.

### Make multi-repo behavior explicit everywhere

Already improved:

- `resolve_write_repo` refuses to choose a repo when a pot has multiple repos and no `repo_name` is supplied.

Audit and tighten:

- `application/use_cases/backfill_pot.py`
- `application/use_cases/ingest_single_pr.py`
- `application/services/ingestion_submission_service.py`
- `application/use_cases/context_graph_jobs.py`

Rules:

- Backfill with no repo means all repos in pot.
- Single PR ingestion must include `repo_name` when pot has multiple repos.
- Raw episode ingest can work without a repo using the standalone synthetic scope.
- Event status must always include pot and provider scope.

### Raw note/link/content ingestion

Both CLI and UI submissions should use the same context-engine ingestion service.

CLI:

- keep `potpie ingest` as the primary command
- allow pot inference from active pot/env/git as today
- submit as `raw_episode` for direct notes/content
- submit as `agent_reconciliation` when the input is a link or structured external artifact that needs an agent/tool fetch

UI:

- add a pot-scoped "Add context" flow
- allow paste text, markdown, or a URL
- owners and users can submit
- response should show the ingestion event id and status
- event status should appear in the pot Ingestion tab

For URL ingestion, normalize the payload with fields such as `url`, `title`, `submitted_text`, and `content_type_guess`; let the ingestion agent fetch/process it through approved tools rather than writing graph data directly from the UI.

### Provider tool access for ingestion agent

The ingestion agent should receive a canonical event and provider evidence tools. It should not read arbitrary integration credentials.

Plan:

1. Add read-only provider adapter ports for Linear/Jira/etc. as needed.
2. Resolve provider credentials from `integration_id` attached to the pot source.
3. Make tool descriptors available to the planner from the event source metadata.
4. Keep graph mutation in episode step executors, not provider adapters.

## UI Plan

### Information architecture

Replace the current card-only `/pots` screen with a master/detail management surface:

- left or top list of pots
- selected pot detail
- tabs: Overview, Users, Sources, Integrations, Ingestion

The existing page can evolve in place at:

- `/Users/nandan/Desktop/Dev/potpie-ui/app/(main)/pots/page.tsx`

Recommended component split:

```text
app/(main)/pots/components/PotList.tsx
app/(main)/pots/components/PotOverview.tsx
app/(main)/pots/components/PotMembersPanel.tsx
app/(main)/pots/components/PotSourcesPanel.tsx
app/(main)/pots/components/PotIntegrationsPanel.tsx
app/(main)/pots/components/PotIngestionPanel.tsx
services/PotService.ts
```

Move pot-specific API calls out of `BranchAndRepositoryService.ts` into `PotService.ts`.

### Users tab

Owner UI:

- list members
- invite user by email
- change access between `owner/user` only if ownership transfer is implemented; otherwise only add/remove `user`
- remove user
- revoke pending invite

User UI:

- view members
- no mutation controls

Backend support needed:

- member response should include display metadata from `users`
- invitation response should include email, status, inviter, created time, and expiry

### Sources tab

GitHub repositories:

- show attached repos
- add through the existing GitHub repository picker/search, not a raw text box
- preserve manual `owner/repo` fallback for local/dev use
- remove repo
- create and display the mirrored source record
- show sandbox/index/parse status once those jobs exist
- show sync/last event status when available

Linear teams:

- list connected Linear integrations
- choose a Linear workspace/account
- fetch teams
- attach a team to the pot
- remove team
- show sync/last event status

Future providers should fit the same source card pattern.

### Integrations tab

Show account-level connected integrations that can be attached to the pot:

- GitHub connection / installation
- Linear connection
- Jira, Confluence, Slack later

Important distinction in UI copy:

- Connection: credentials/account is connected.
- Source: specific repo/team/channel is included in this pot.

Avoid storing provider secrets in pot-specific UI state.

### Ingestion tab

Use context-engine event APIs:

- list recent events for pot
- filter by source/provider/status
- show queued/processing/done/error
- open event detail with payload summary and error
- provide owner-only retry/sync actions when backend supports them

Add a pot-scoped submit form:

- paste text/markdown
- paste URL
- choose ingestion mode when needed: direct note or agent process
- available to both `owner` and `user`
- submit through the same pot-scoped ingestion endpoint used by CLI-compatible raw ingest

## Migration And Rollout

### Step 1: backend role alignment

- Add `user` role.
- Migrate `admin/read_only` to `user`.
- Remove legacy roles from API responses and UI.
- Update tests and docs.
- Keep API responses stable enough for current UI.

### Step 2: invitations and UI service split

- Create `PotService.ts`.
- Move existing pot/repo calls there.
- Keep current `/pots` behavior working.
- Add member APIs to service even before the full UI is built.
- Add invitation APIs to service.

### Step 3: members and invitations UI

- Add Users tab.
- Support owner invite-by-email and revoke pending invite.
- Support accepted members list and remove user.
- Hide mutation controls for non-owner.

### Step 4: pot source model

- Add `context_graph_pot_sources`.
- Add service helpers similar to `project_sources_service.py`, but pot-scoped and role-aware.
- Add source APIs for GitHub repo and Linear team attachments.
- Mirror every GitHub repository add/delete into pot sources immediately.
- Add placeholders/status fields for repo sandbox/index/parse preparation.

### Step 5: raw note/link/content ingestion

- Add pot-scoped UI submission endpoint if current `/ingest` shape is not ergonomic enough.
- Allow `owner` and `user`.
- Store submissions in `context_events`.
- Surface event status in the UI.

### Step 6: GitHub webhook routing to pots

- Replace `Project` lookup in `WebhookEventHandler._process_github_webhook`.
- Route by `context_graph_pot_repositories` and/or mirrored source rows.
- Submit through `IngestionSubmissionService`.
- Keep old project fallback behind a temporary feature flag only if needed.

### Step 7: Linear source attachment and routing

- Add Linear teams picker in UI.
- Persist team source attachment.
- Route Linear webhooks by pot source.
- Submit `agent_reconciliation` events.

### Step 8: ingestion visibility

- Add Ingestion tab.
- Add provider/source filters to event listing if needed.
- Show last sync/error on source cards.

### Step 9: repo processing and indexing

- Start sandbox/index/parse job when a repo source is added.
- Store job/source health on the pot source row or a child status table.
- Reindex continuously or on webhook/schedule as needed.
- Expose status in the Sources tab.

### Step 10: harden and converge

- Add ownership transfer.
- Add audit logs for member/source changes.
- Add RBAC migration path beyond owner/user.

## Testing Plan

Backend unit tests:

- owner can add/remove users; user cannot
- owner can add/remove repos/integrations/sources; user cannot
- non-member cannot read pot
- legacy `admin/read_only` rows migrate to `user`
- owner can invite by email
- invite accept creates `context_graph_pot_members` row
- GitHub webhook routes to all matching pots
- GitHub webhook ignores unbound repos
- GitHub repo add mirrors to `context_graph_pot_sources`
- multi-repo pot PR event requires or supplies `repo_name`
- Linear webhook routes by team id and integration id
- owner and user can submit raw notes/links/content for ingestion

context-engine unit tests under `app/src/context-engine/tests/unit/`:

- ingestion submission with multi-repo pot and explicit repo
- ambiguous multi-repo event without repo fails clearly
- raw episode works for pot with no repositories
- event list filters by pot and provider/source metadata

UI tests/manual checks:

- owner sees management actions
- user sees non-management views without repo/member/integration controls
- user can submit raw notes/links/content
- owner can invite by email
- invited user can accept and see the pot
- repository picker attaches a repo and refreshes list
- repository attachment creates a source row and shows source status
- Linear team picker attaches a team
- ingestion tab reflects event status from backend

Suggested verification commands:

```bash
uv run pytest app/src/context-engine/tests/unit/
uv run pytest app/tests/ -k context_graph
```

Use the UI dev server at `http://localhost:3000/pots` for manual validation after frontend changes.

## Decisions

1. There are only two active roles now: `owner` and `user`. No `admin`; no `read_only`.
2. Users can manually ingest raw notes, links, and content through both CLI and UI.
3. If the same GitHub repo/source is attached to multiple pots, webhook ingestion fans out to every matching pot.
4. GitHub repository attachments must immediately be mirrored into `context_graph_pot_sources`.
5. Repo source rows are the foundation for the next sandbox/index/parse pipeline so agents can process codebases after repo attachment and continuously reindex when needed.
6. Users are invited by email. They are added to `context_graph_pot_members` only after accepting the invite.
7. Pot integration attachment semantics need clearer naming in implementation: use `context_graph_pot_integrations` for account/connection attachment and `context_graph_pot_sources` for concrete data scopes.

## Recommended Next Implementation Slice

Start with a narrow vertical slice:

1. Add the strict `owner/user` role model and migrate existing non-owner rows to `user`.
2. Add email invitation persistence/API and richer member list responses.
3. Add owner invite-by-email UI and user/member display in `/pots`.
4. Add `context_graph_pot_sources` and mirror GitHub repository attachments into it.
5. Add pot-scoped raw note/link/content ingestion from UI for both owner and user.
6. Refactor the GitHub webhook handler to route merged PR events by pot repository/source rows and submit through `IngestionSubmissionService`.
7. Add tests for role migration, invitations, source mirroring, raw ingestion permissions, and GitHub event-to-pot fan-out.

That slice proves the core model: database-managed pot membership, invite-based access, database-managed source attachment, UI-submitted context ingestion, and webhook ingestion routed to every correct pot without relying on legacy `Project` lookup or graph-only state.

## Self Review

The plan deliberately does not replace the existing pot tables because they already match most of the requested persistence model. The largest new schema proposal is `context_graph_pot_sources`, which is needed because integrations and source scopes are different concepts. Without that split, Linear teams, Slack channels, Jira projects, and GitHub repos would all end up as ad hoc JSON inside pot integrations, making routing and UI management fragile.

The highest-risk area is webhook routing. The code currently has two GitHub webhook paths: a standalone context-engine env-map path and a Potpie event-bus path that still uses `Project`. The implementation should change the Potpie event-bus path first and leave standalone env routing intact. That keeps the product path moving without breaking context-engine standalone usage.

The role model is now intentionally strict: only `owner` and `user` should remain. Because the current code already has `admin/read_only`, implementation should include a migration that rewrites those values to `user` and tests that no API response or UI branch continues exposing the old roles.

The previous open question about pot integrations was mostly a naming/boundary issue. In concrete terms: an integration is the connected account or credential, such as "Nandan's Linear workspace connection" or "GitHub installation 123"; a source is the actual data scope included in a pot, such as "Linear team ENG" or "github.com/openai/potpie". The reviewed plan now treats `context_graph_pot_integrations` as pot-to-connection attachment and `context_graph_pot_sources` as the concrete data scopes that route ingestion. That makes GitHub repositories, Linear teams, Slack channels, and future provider scopes manageable without mixing credentials and ingestable source definitions.
