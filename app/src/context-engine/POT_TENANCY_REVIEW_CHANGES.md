# Pot Tenancy Review Changes

This file captures the changes still needed after the recent pot-tenancy implementation review.

Assumptions for this cleanup:

- legacy `Project` fallback is not needed
- a pot is not a project and must not resolve through project tables
- each pot has exactly one owner

## 1. Remove legacy project fallback entirely

The current implementation still treats `Project` rows as fallback pots.

That needs to be removed from both Potpie and `context-engine` wiring.

### Files to change

- `app/modules/context_graph/pot_access.py`
- `app/modules/context_graph/context_pot_routes.py`
- `app/modules/context_graph/wiring.py`
- `app/modules/context_graph/context_engine_http.py`
- `app/src/context-engine/adapters/inbound/http/api/v1/context/router.py`
- `app/src/context-engine/adapters/inbound/cli/main.py`
- `app/src/context-engine/adapters/inbound/cli/README.md`
- `app/src/context-engine/adapters/outbound/http/potpie_context_api_client.py`
- any tests that still use project-backed pot behavior

### Required changes

#### `app/modules/context_graph/pot_access.py`

- delete `user_role_on_pot()`
- rename `require_pot_member()` / related checks to operate only on context-graph pots
- remove all `Project` imports and queries
- `require_pot_member()` should check membership only through:
  - `context_graph_pot_members`
  - or direct owner relationship on `context_graph_pots`

Suggested behavior:

- unknown pot -> `404`
- known pot but caller not a member -> `404`
- member but insufficient role -> `403`

#### `app/modules/context_graph/context_pot_routes.py`

- remove all `Project` imports and branches
- `GET /pots/{pot_id}` should only return real `context_graph_pots`
- `GET /pots/{pot_id}/repositories` should only return repository rows from `context_graph_pot_repositories`
- remove `"kind": "project"` output branches
- `_context_graph_pot_row_or_404()` should become the single source of truth for pot existence

#### `app/modules/context_graph/wiring.py`

- remove `Project` imports
- delete:
  - `UserScopedSqlalchemyPotResolution`
  - `UserScopedCompositePotResolution`
- stop resolving through `projects`
- `SqlalchemyPotResolution` and the user-scoped resolver should read only:
  - `context_graph_pots`
  - `context_graph_pot_members`
  - `context_graph_pot_repositories`

Recommended end state:

- one resolver for worker/session-wide access to real pots
- one resolver for user-scoped access to real pots
- no `single_github_repo_pot()` fallback from Potpie `Project`

#### `app/modules/context_graph/context_engine_http.py`

- update all error text to remove “or use a project id”
- container comments/docstrings should say “user-scoped pots”, not “user-scoped projects”

#### `app/src/context-engine/adapters/inbound/http/api/v1/context/router.py`

- remove compatibility aliases that still accept `project_id`
- change:
  - `ProjectGraphQuery`
  - `ResolveContextRequest`
- use `pot_id` only
- update `UNKNOWN_POT_DETAIL` to remove project wording

#### CLI cleanup

Remove project-centric commands and messages:

- remove `pot projects` command from `adapters/inbound/cli/main.py`
- remove `list_projects()` from `adapters/outbound/http/potpie_context_api_client.py`
- update help text and errors that mention projects
- update `adapters/inbound/cli/README.md`

Examples of strings to remove:

- “Potpie project UUID”
- “use a project id”
- “pot projects”
- “context-pot or project UUID”

## 2. Fix multi-repo pot-wide backfill

The new backfill implementation still rejects multi-repo pots unless `target_repo_name` is provided.

### File to change

- `app/src/context-engine/application/use_cases/backfill_pot.py`

### Bug

`resolve_write_repo()` is called before deciding whether this is:

- a pot-wide run
- or a repo-specific run

Current behavior:

- `target_repo_name is None`
- pot has multiple repos
- `resolve_write_repo()` returns `None`
- function returns `ambiguous_repo`

That prevents pot-wide backfill from running at all.

### Required change

Change the branching to:

1. resolve pot
2. verify it has repos
3. if `target_repo_name` is provided:
   - resolve exactly one repo
   - fail if not found
4. else:
   - use `resolved.repos` directly

Pseudo-shape:

```python
if target_repo_name:
    selected = resolve_write_repo(resolved, repo_name=target_repo_name)
    if selected is None:
        ...
    repos = [selected]
else:
    repos = list(resolved.repos)
```

Do not call `resolve_write_repo()` on pot-wide backfill.

## 3. Fix repository detach semantics

Deleting the last repository currently leaves the pot operationally attached through `primary_repo_name`.

### Files to change

- `app/modules/context_graph/context_pot_routes.py`
- `app/modules/context_graph/wiring.py`
- optionally migration/model cleanup if `primary_repo_name` is now only a display hint

### Current problem

- deleting a row from `context_graph_pot_repositories` does not clear `context_graph_pots.primary_repo_name`
- resolver falls back to `_resolved_pot_legacy_primary()` when no repo rows remain
- result: detach appears to succeed but the pot still resolves to the old repo

### Required changes

#### Resolver

In `app/modules/context_graph/wiring.py`:

- remove `_resolved_pot_legacy_primary()`
- `_resolved_pot_from_context_graph_row()` should return:
  - `ResolvedPot(..., repos=[], ...)` when no repo rows exist
- never synthesize repo membership from `primary_repo_name`

#### Delete repository route

In `app/modules/context_graph/context_pot_routes.py`:

- after deleting a repo row, recompute `primary_repo_name`
- if repos remain:
  - set `primary_repo_name` to the first remaining repo name, or keep it only as a display cache
- if no repos remain:
  - set `primary_repo_name = None`

If you keep `primary_repo_name`, it must remain consistent with actual repository rows.

## 4. Enforce exactly one owner in the API shape

The current code partly assumes many owners, but the desired model is exactly one owner.

### Files to change

- `app/modules/context_graph/context_pot_routes.py`
- `app/modules/context_graph/pot_member_roles.py`
- related tests

### Required changes

#### Membership create/update

- `POST /pots/{pot_id}/members` must reject `owner`
- `PATCH /pots/{pot_id}/members/{member_user_id}` must reject any change to `owner`
- remove messages that imply “add another owner first”
- remove multi-owner counting logic

Recommended behavior:

- owner cannot be added through member APIs
- owner cannot be demoted through member APIs until an explicit transfer API exists
- owner cannot be removed through member APIs

Suggested messages:

- `"owner role cannot be assigned via this endpoint"`
- `"owner role cannot be changed via this endpoint"`
- `"owner cannot be removed; ownership transfer is not implemented"`

#### Future-proofing

If ownership transfer is needed later, add a dedicated endpoint:

- `POST /pots/{pot_id}/transfer-ownership`

But do not keep partial multi-owner logic in the current member API.

## 5. Validate member targets before insert

Adding a member currently depends on DB FK errors for nonexistent users.

### File to change

- `app/modules/context_graph/context_pot_routes.py`

### Required change

Before inserting a `ContextGraphPotMember` row:

- verify that the target user exists in `users`
- return a clear `404` or `400` if not found

Suggested behavior:

- missing/blank `user_id` -> `400`
- nonexistent user -> `404`

This should also be applied to any owner-transfer API if added later.

## 6. Remove remaining public `project_id` compatibility in context-engine

If legacy fallback is not needed, public compatibility aliases should go too.

### Files to change

- `app/src/context-engine/adapters/inbound/http/api/v1/context/router.py`
- `app/src/context-engine/application/use_cases/graph_query.py`
- `app/src/context-engine/adapters/inbound/mcp/project_access.py`
- any remaining tests/docs using `project_id`

### Required changes

#### HTTP models

- remove `AliasChoices("pot_id", "project_id")`
- require only `pot_id`

#### Graph query models

- remove `project_id` fallback fields and validation
- update error messages to say `pot_id` only

#### MCP

- remove `assert_mcp_project_allowed()`
- remove any API accepting `project_id` as a separate input

## 7. Clean up terminology and docs

Even where behavior is correct, several messages still leak the old model.

### Files to change

- `app/src/context-engine/adapters/inbound/cli/README.md`
- `app/src/context-engine/README.md`
- `app/modules/context_graph/context_engine_http.py`
- `app/src/context-engine/adapters/inbound/http/api/v1/context/router.py`
- `app/src/context-engine/adapters/inbound/cli/credentials_store.py`
- `app/src/context-engine/adapters/inbound/cli/main.py`
- `app/src/context-engine/domain/ports/intelligence_provider.py`
- `app/src/context-engine/application/use_cases/resolve_context.py`

### Strings to remove or rewrite

- “project-scoped knowledge”
- “project id”
- “Potpie project UUID”
- “use a project id”
- “project-backed pots”
- “pot projects”

## 8. Suggested test updates

### Add or update tests for

- pot-wide backfill on a multi-repo pot runs all repos
- repo-specific backfill still works with `target_repo_name`
- deleting the last repo leaves the pot with zero repos and no synthetic fallback
- `GET /pots/{pot_id}` fails for nonexistent or unauthorized pots without consulting `Project`
- `GET /pots/{pot_id}/repositories` returns only actual repo rows
- owner cannot be added, removed, or demoted through member endpoints
- adding a nonexistent user as member returns a clean error
- public HTTP request models reject `project_id`
- CLI no longer exposes `pot projects`

## Recommended implementation order

1. Remove resolver fallback to `Project`
2. Fix repository detach semantics
3. Fix pot-wide backfill control flow
4. Simplify ownership rules to exactly one owner
5. Add explicit user existence validation for member APIs
6. Remove `project_id` compatibility from public `context-engine` surfaces
7. Clean docs and CLI wording

## End state

After these changes:

- a pot is only a `context_graph_pots` record plus related member/repo/integration rows
- no runtime path treats a `Project` as a pot
- multi-repo pot-wide backfill actually works
- removing a repo really removes it from execution scope
- there is exactly one owner per pot
- public APIs and CLI speak only in terms of `pot_id`
