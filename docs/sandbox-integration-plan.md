# Sandbox Integration Plan

This document plans the next phase of `app/src/sandbox`: turning it into the
single library that everything else in potpie uses to materialize repos and run
code, and retiring the `app/modules/repo_manager` + `app/modules/intelligence/
tools/code_changes_manager` stack.

It is a plan, not a spec. Where there are real tradeoffs the doc names them
and points at the surfaces involved, but leaves the call to the implementer.
Read `docs/sandbox-core-setup.md` first — this builds on the model defined
there (`RepoCache` / `Workspace` / `Runtime`).

---

## 1. Where we are

The sandbox module today (`app/src/sandbox/sandbox/`) is hexagonal and runnable:

- Domain: `Workspace`, `Runtime`, `RepoIdentity`, `WorkspaceRequest`,
  `ExecRequest`, etc. (`domain/models.py`).
- Ports: `WorkspaceProvider`, `RuntimeProvider`, `SandboxStore`, `LockManager`
  (`domain/ports/*`).
- Adapters: local-git workspace, local-subprocess runtime, docker runtime,
  daytona workspace + runtime, file/json store, in-memory locks
  (`adapters/outbound/*`).
- Application service: `SandboxService` (`application/services/sandbox_service.py`)
  exposes `get_or_create_workspace`, `get_or_create_runtime`, `exec`,
  `hibernate_runtime`, `destroy_runtime`, `destroy_workspace`. It is
  idempotent, locks per-workspace, and tracks `dirty` after writes.
- E2E tests cover all three runtime backends end-to-end (subprocess, docker,
  daytona). Daytona dev stack is bootstrapped via `scripts/setup-daytona-local.sh`
  and the override compose file.

The rest of potpie still drives a parallel stack:

- `app/modules/repo_manager/` — bare repo + worktree management on disk under
  `.repos/<owner>/<repo>/`. Has its own auth chain, eviction, metadata.
- `app/modules/intelligence/tools/code_changes_manager/` — Redis-backed
  per-conversation file-edit staging area, with ~20 LangChain
  `StructuredTool`s wrapped around it (`add_file_to_changes`,
  `update_file_lines`, `replace_in_file`, …, plus `git_commit`, `git_push`,
  `bash_command`).
- Parsing (`app/modules/parsing/`) calls `RepoManager.prepare_for_parsing()` to
  get a worktree path and walks it directly with `os.walk`/Tree-sitter.
- Agents (PydanticAI in `app/modules/intelligence/agents/`) get tools via
  `ToolService` + `ToolResolver`, with `ChatContext` (`project_id`, `user_id`,
  `conversation_id`, `branch`, `local_mode`) carrying scope.

The two stacks duplicate concepts: both clone, both manage worktrees, both
think about branches. The plan unifies them under the sandbox module.

---

## 2. Goals

- A small, library-shaped public API on `app/src/sandbox` that the rest of
  potpie can import and use to: get a sandbox, get a working tree on a branch,
  read/write files, run commands, search, commit, push.
- **Repo lifecycle owned by sandbox.** The caller passes
  `(repo, branch, user, project)` — sandbox does the cloning, the worktree
  creation, the branch switching, and (eventually) the eviction. Parsing and
  agents both go through it.
- **Pre-provisioned tooling.** Common dev tools (`rg`, `git`, `fd`, `jq`,
  `python`, `node`) are present in the runtime out of the box, so agent tools
  don't need to detect/install at runtime.
- **One backend sandbox, many branches.** A single backend container per
  `(user, project)` hosts multiple worktrees so multiple agent runs on
  different branches share the same expensive thing (the Daytona sandbox /
  Docker container) without serializing.
- **Sandbox-backed agent tools** ship from the sandbox library itself,
  registered into `ToolService` like any other tool. Each tool resolves its
  workspace from the agent's run context.
- **`CodeChangesManager` is gone.** Edits go straight to the worktree (which
  is durable git state). The Redis-staged-changes model is replaced by
  "edit the file, commit when done." If we want stage-and-review semantics
  later they live above the sandbox, not inside it.
- **`.repos` becomes one adapter** under the sandbox `WorkspaceProvider`
  port, used for local development and tests. Daytona is the production path.

## Non-goals

- Replacing the agent framework, the LLM provider abstraction, or the search
  service. The sandbox owns "where the bits live" — not "how the agent
  thinks."
- Full distributed locking. The `InMemoryLockManager` is fine for the single-
  worker case; a Redis-backed lock manager is a separable later concern.
- Snapshot-based hibernation as a primary persistence mechanism. Git worktrees
  + the backend's volume are the persistence.
- Backwards compatibility with the `code_changes_manager` Redis schema. We are
  removing it; there's no migration.

---

## 3. Conceptual additions

The current sandbox model (one `Workspace` = one branch) is right for what it
solves, but the integration goals push two extensions:

### 3.1 RepoCache becomes load-bearing

`RepoCache` exists in `domain/models.py` (lines 116–127) but isn't wired in.
The plan promotes it to a real concept owned by the workspace provider:

- `RepoCache` = "the bare clone, somewhere a backend can reach it."
- `Workspace` = "a working tree on top of a `RepoCache`, scoped to a branch."
- One `RepoCache` per `(user, repo)` — or per `(repo)` if we trust shared
  cache semantics; defer that call until we actually share across users.
- Many `Workspace` per `RepoCache`, one per active branch.

In adapter terms:

| Backend          | RepoCache materialization                   | Workspace materialization              |
| ---------------- | ------------------------------------------- | -------------------------------------- |
| Local git        | `.repos/<owner>/<repo>/.bare/`              | `.repos/.../worktrees/<key>/`          |
| Daytona          | Bare clone *inside* a long-lived sandbox    | Worktree in the same sandbox           |
| Docker           | Bare clone in a named docker volume         | Worktree mounted into the runtime      |

The Daytona row is the interesting one: today each `WorkspaceRequest` creates
its own Daytona sandbox. The new model creates one Daytona sandbox per
`(user, project)` and uses `git worktree` *inside* it for each branch. That's
how we get multi-branch concurrency without paying for N sandboxes.

The implementer should decide whether to:
- (a) Add a `RepoCacheProvider` port alongside `WorkspaceProvider`, or
- (b) Fold the cache semantics into `WorkspaceProvider` (one provider, two
  conceptual layers), or
- (c) Keep `RepoCache` purely as a value object that providers populate
  internally.

(b) or (c) is probably right — adding a port costs more than it pays for
unless we expect to swap cache backends independently.

### 3.2 Workspace as a branch handle

A `Workspace` already has `metadata["branch"]` and a worktree path. The plan
keeps that, but tightens the lifecycle:

- Creation: clone or fetch into the cache, then `git worktree add` for the
  branch.
- Switch-branch: not actually a switch — it's "give me a different
  `Workspace` on the same `RepoCache`." The caller asks for a workspace by
  `(user, project, branch)` and gets the right worktree.
- Reuse: the existing key-based idempotency in `SandboxService.get_or_create_
  workspace` already does this (line 50). Verify the key formula handles the
  `(repo, branch)` case cleanly without `conversation_id` or `task_id`
  required.
- Cleanup: `destroy_workspace` removes the worktree but keeps the cache.
  A separate path destroys the cache (or the whole backend container).

This is the biggest behavioral change. Today the daytona adapter
(`adapters/outbound/daytona/provider.py:131-152`) calls
`daytona.create(...)` per workspace. After the change, sandbox creation is
keyed on `(user, project)` and worktree creation runs *inside* an existing
sandbox.

---

## 4. The public client API

What the rest of potpie should be able to import. Treat names as suggestions.

### 4.1 The package surface

```
app/src/sandbox/sandbox/__init__.py
    # The library export
    SandboxClient
    SandboxClientConfig
    # Re-exports of the small set of types callers actually need:
    WorkspaceHandle, ExecResult, NetworkMode, CommandKind,
    SandboxError, WorkspaceNotFound, ...
```

`SandboxClient` is a thin façade over `SandboxService` plus the bootstrap
container. The intent: callers don't construct providers, stores, locks, or
build runtime specs by hand. They get a client and ask for what they need.

Sketch — flesh out in code:

```python
class SandboxClient:
    @classmethod
    def from_env(cls, *, settings: SandboxSettings | None = None) -> Self: ...
    @classmethod
    def from_container(cls, container: SandboxContainer) -> Self: ...

    # Repo lifecycle
    async def get_workspace(
        self,
        *,
        user_id: str,
        project_id: str,
        repo: str,                  # "owner/name"
        branch: str,                # the branch the agent / parser will work on
        base_ref: str | None = None,  # what to branch from (default: branch itself)
        create_branch: bool = False,
        auth_token: str | None = None,
        mode: WorkspaceMode = WorkspaceMode.EDIT,
    ) -> WorkspaceHandle: ...

    async def release_workspace(self, handle: WorkspaceHandle) -> None: ...

    # The runtime is implicit — caller never constructs a RuntimeRequest.
    async def exec(
        self, handle: WorkspaceHandle, cmd: list[str], *,
        cwd: str | None = None, env: Mapping[str, str] | None = None,
        timeout_s: int | None = None, command_kind: CommandKind = CommandKind.READ,
    ) -> ExecResult: ...

    # Convenience file ops — built on exec, but cheaper to use than
    # crafting `cat` / `tee` invocations everywhere.
    async def read_file(self, handle, path, *, max_bytes: int | None = None) -> bytes: ...
    async def write_file(self, handle, path, content: bytes | str) -> None: ...
    async def list_dir(self, handle, path) -> list[FileEntry]: ...
    async def search(self, handle, pattern, *, glob=None, case=False) -> list[Hit]: ...

    # Git ops — typed wrappers, not free-text bash.
    async def commit(self, handle, message: str, *, paths: list[str] | None = None) -> str: ...
    async def push(self, handle, *, remote="origin", set_upstream=True) -> None: ...
    async def diff(self, handle, *, base_ref: str | None = None) -> str: ...
    async def status(self, handle) -> GitStatus: ...
```

### 4.2 What `WorkspaceHandle` is

A small, opaque object containing the IDs the client needs to talk to the
service: `workspace_id`, the resolved `branch`, the runtime working dir.
*Not* the `Workspace` domain object — the handle is the stable thing returned
to callers. Internally the client looks up the live `Workspace` each call.

Why opaque: it lets the implementer change what's stored (a single ID vs a
small struct vs a `Workspace` snapshot) without churning every caller.

### 4.3 Read/write helpers vs `exec`

The existing `exec()` is enough to do everything, but agent tools end up
shelling out for trivial operations and re-implementing argument quoting.
The helpers (`read_file`, `write_file`, `list_dir`, `search`) are first-class
on the client because:

- They have a fixed surface that's easy to mock / fake in tests.
- They can dispatch to backend-native APIs when available (Daytona toolbox
  has `fs.upload_file`, `fs.download_file`, `git.*` — currently only used by
  the daytona workspace provider).
- They give the agent prompt a small, named tool surface (see §7).

The implementer should look at Daytona's toolbox API
(`adapters/outbound/daytona/provider.py`, lines 154–228 are precedent) and
wire the helpers through `RuntimeProvider` so each backend has one chance
to do it natively before falling back to `exec`.

### 4.4 What we deliberately don't expose (yet)

- No `Runtime` lifecycle on the client. Runtime is implicit; `exec` brings
  it up. `hibernate` / `destroy` show up as `release_workspace`.
- No streaming exec. `exec_stream` exists in the port but the inbound
  surface stays sync-result for v1; revisit when an agent flow actually
  benefits.
- No "preview URL" / port-forwarding. `RuntimeCapabilities` already models
  this; expose it when there's a caller.

---

## 5. Preinstalled tooling

Today the runtime images are minimal: `python:3.12-slim` for docker default,
`daytonaio/sandbox:0.5.0-slim` for daytona, host PATH for subprocess.
Nothing has `rg` (ripgrep), and the Daytona slim image doesn't even ship git
CLI — the daytona adapter explicitly works around that with the toolbox API
(see `provider.py:182-185`).

The plan: ship our own image for the docker and daytona backends with a
known-good toolset.

### 5.1 Image contents (proposal)

A single Dockerfile in `app/src/sandbox/images/agent-sandbox/` produces
`potpie/agent-sandbox:<version>`:

- Base: `python:3.12-slim` (or `debian:stable-slim` if startup time is an
  issue and we want a cheaper base).
- Tools: `git`, `git-lfs`, `ripgrep`, `fd-find`, `jq`, `curl`, `ca-certificates`,
  `tini`, `tree`, `less`, `coreutils`, `procps`, `openssh-client`.
- Runtimes: Python (with `pip`, `uv` optional), Node LTS, GitHub CLI (`gh`)
  if we want it for PR creation.
- A non-root user `agent` with $HOME persistent — the worktree will live
  under `/home/agent/work/`.
- An entrypoint that pre-creates the worktrees dir and exec's the requested
  command (mirrors what the override compose entrypoint does today).

The implementer should check the actual size — going from slim+toolbox to
full-fat is a real tradeoff for Daytona cold-start. If that's painful, split:
a "lean" image without Node and a "full" image with everything; pick per
workload.

### 5.2 Daytona snapshot

Daytona consumes images via "snapshots." The override compose stack already
has a flow for this (`scripts/setup-daytona-local.sh`). Process:

1. Build the image.
2. Push to a registry Daytona can pull from. (For dev: the local docker daemon.)
3. Run `daytona snapshot create potpie/agent-sandbox:<version>`
   (or however the Daytona SDK phrases it — check existing scripts).
4. Default `DAYTONA_SNAPSHOT` env var to the new snapshot in `.env.daytona.local`
   and in `bootstrap/settings.py`.

Tests: extend `tests/e2e/test_daytona_e2e.py` with a sanity check that
`rg --version` and `git --version` succeed inside a fresh sandbox.

### 5.3 Subprocess backend

The local subprocess backend is purely for tests and dev-machine debugging.
It uses host PATH. We don't ship anything; we document that `rg` should be
on PATH. Tests already skip when not available.

---

## 6. Repo lifecycle in the new model

### 6.1 Parsing path

Today `ParseHelper.clone_or_copy_repository` (`app/modules/parsing/`)
calls `RepoManager.prepare_for_parsing` and walks the resulting path with
`os.walk`. Tomorrow:

```
ParsingService.parse_directory(repo_details, ...)
  → SandboxClient.get_workspace(
        user_id, project_id, repo, branch=base_ref,
        mode=WorkspaceMode.ANALYSIS, create_branch=False,
    )
  → handle = ...
  → run parsing against handle (see below)
  → SandboxClient.release_workspace(handle)   # or keep for re-parse
```

How parsing actually reads files matters:

- **Option A (recommended, easier):** parsing keeps running on the host;
  the sandbox client just produces a host-readable path. For the local-fs
  backend this is the existing `.repos/.../worktrees/...`. For Daytona,
  this means: don't run parsing in Daytona — run the cheap clone in
  Daytona only when the agent actually needs a sandbox, and use the
  local-fs backend for parsing. Two backends, one client API, picked per
  workload.

- **Option B (harder):** parsing runs *inside* the sandbox (rust extractor
  baked into the image, results streamed back). The agent (`Map parsing
  flow & repo dependency` writeup, §10) flagged GitPython fork-safety,
  Rust backend distribution, and embedding model size as the costs. Worth
  doing if we want a uniform model for managed-cloud customers, but not
  the v1.

Pick A. Make sure the client's helpers don't accidentally tie callers to
"the path is on the host." The `WorkspaceHandle` should expose a
`local_path: str | None` field that is populated only for local-fs backends
and `None` for daytona — parsing checks for it and skips Daytona for now,
or refuses with a clear error.

### 6.2 Agent path

Today: agent tool fires, calls `RepoManager` to get a worktree, writes
through `CodeChangesManager` to Redis, later commits.
Tomorrow:

```
ChatContext { project_id, user_id, conversation_id, branch, ... }
  → SandboxClient.get_workspace(
        user_id, project_id, repo, branch=branch,
        mode=WorkspaceMode.EDIT, create_branch=True,
    )
  → handle is cached on the agent run (see §7.3 for where)
  → tools call client.{read_file, write_file, search, exec, commit, ...}
  → run ends → release_workspace (don't destroy the worktree yet — keep it
    until conversation cleanup).
```

The branch model: `WorkspaceMode.EDIT` already names branches
`agent/edits-{conversation_id}` (`adapters/outbound/local/git_workspace.py`,
lines 204-211). Keep that. For the daytona backend the branch lives as a
worktree on a shared sandbox per `(user, project)`.

### 6.3 Switching branches

User asked for "checkout somehow when plugging to the agent." Two
interpretations:

- (a) Same conversation, switch branch — operator wants to abandon the
  current branch and resume on another. Express as: release the old
  workspace, get a new workspace with a different branch.

- (b) Same backend container, multiple branches in flight — multiple
  conversations / agents, one Daytona sandbox. Express as: each
  `get_workspace` returns a new worktree inside the existing sandbox.

(b) is the real concurrency story; (a) is a special case of "release +
get_or_create." The implementer shouldn't add a `switch_branch` operation
on the client — it's just two existing calls.

### 6.4 Cleanup

Three layers:

1. Worktree (cheap): destroy on conversation close, or after N days idle.
2. Backend sandbox (expensive in $$$): destroy on user inactivity, or per
   organization quota. Hibernate (Daytona auto-stops after 30s; auto-archives
   after 12h — already configured in
   `adapters/outbound/daytona/provider.py:142`).
3. Repo cache (expensive in disk): LRU evict, or per-repo TTL.

Today's `RepoManager._evict_if_needed` (the writeup §3 in the
code-changes-manager map) is a reasonable starting point. Port it as a
*background task* that the local-fs adapter runs, not as inline-during-
clone behavior — that surprised people. Daytona handles its own GC for
sandboxes; we only need to run cache cleanup if we end up storing caches
in named volumes.

---

## 7. Agent tool exports

The user wants a set of tools they can plug into agents that gives the
agent access to a specific `(repo, branch)` sandbox.

### 7.1 Tool catalog (proposal)

Ship a small, opinionated set. Names are starting points — match the
existing tool naming style (snake_case, verbs). Each tool's input schema
includes `project_id` (the agent already passes this) and the input gets
resolved to a `WorkspaceHandle` at call time.

| Tool                       | Wraps                                       | Agent uses for                      |
| -------------------------- | ------------------------------------------- | ----------------------------------- |
| `sandbox_read_file`        | `client.read_file`                          | view a file (line numbers optional) |
| `sandbox_write_file`       | `client.write_file`                         | full-file replace                   |
| `sandbox_str_replace`      | `client.read_file` + write                  | targeted in-file edit               |
| `sandbox_list_dir`         | `client.list_dir`                           | navigate                            |
| `sandbox_search`           | `client.search` (ripgrep)                   | grep across the tree                |
| `sandbox_run`              | `client.exec`                               | run any command (whitelisted?)      |
| `sandbox_run_tests`        | `client.exec` + framework detection         | language-aware test runner          |
| `sandbox_git_status`       | `client.status`                             | what changed                        |
| `sandbox_git_diff`         | `client.diff`                               | review changes                      |
| `sandbox_git_commit`       | `client.commit`                             | commit                              |
| `sandbox_git_push`         | `client.push`                               | push (auth-token aware)             |
| `sandbox_open_pr`          | provider-specific PR creation               | end of feature flow                 |

The implementer should compare to today's `code_changes_manager` tools
(20-ish tools across staging operations) and consciously drop the ones
that exist only because edits were staged in Redis — `clear_file`,
`get_changes_summary`, `serialize`, `revert_file`, etc. The agent operates
directly on the worktree; Git is the source of truth and the audit log.

### 7.2 Where the tool code lives

Two reasonable placements:

- **Inside the sandbox library**, exported as factory functions. Tools
  import only from the public client API. This is cleanest — the sandbox
  library is self-contained.
- **In `app/modules/intelligence/tools/sandbox/`**, depending only on
  the sandbox client. This matches the rest of the tool layout.

Pick the second. Sandbox stays a pure library; potpie-specific glue
(StructuredTool wrapping, ToolService registration, ChatContext binding)
stays where the rest of that glue lives. The library only exports
data-shaped helpers, not LangChain `StructuredTool`s.

### 7.3 How tools resolve the workspace

The tool needs to know `(user_id, project_id, repo, branch)` to call
`client.get_workspace`. Today `ChatContext` already carries all four
(`app/modules/intelligence/agents/chat_agent.py:71-162`). Two viable
approaches:

- **Closure capture**: `ToolService.__init__` (or a new
  `SandboxToolFactory`) constructs each tool with the live `ChatContext`
  for the run. Tools don't take `project_id` in their args. Pros: agent
  can't "lie" about which project. Cons: tool factories need to be
  re-instantiated per agent run, which doesn't match the current
  `ToolService` lifecycle (per-user singleton).

- **Explicit args**: tool input schema keeps `project_id` (matching all
  current tools — see `code_query_tools/bash_command_tool.py:455-525`).
  The tool resolves it on each call. The branch is discovered server-side
  from `ChatContext` rather than the LLM picking. Pros: fits the
  existing pattern. Cons: agent could in theory pass a different
  project_id, but `ProjectService.get_project_from_db_by_id_sync` already
  validates user-project ownership.

Pick **explicit args** for parity. Branch comes from `ChatContext` via a
mechanism the implementer can mirror from `bash_command_tool` (which uses
`get_or_create_edits_worktree_path` and relies on the conversation context).
For sandbox tools, that translates to: a small `SandboxRunContext`
contextvar set at the top of every agent run with the resolved
`WorkspaceHandle` (or the data needed to fetch it), and tools read it.

### 7.4 Result shape

Match the existing tools' shape: `Dict[str, Any]` with `success`, the
result data, and explicit truncation flags. `bash_command_tool`'s 80k-char
output limit is a precedent — keep similar for `sandbox_run`. For
`sandbox_search` return a list of hits with `file:line:snippet`; that's
what the LLM actually wants.

### 7.5 Allow-list and registry

Add a `sandbox` tool group in
`app/modules/intelligence/tools/registry/definitions.py` and update the
agent allow-lists (`code_gen`, `execute`, `qna` etc. — see
`agents/chat_agents/system_agents/`). The registry-driven path
(`ToolResolver.get_tools_for_agent`) is the preferred entry point;
hardcoded allow-lists should be migrated as part of this same change
since they refer to soon-to-be-removed tool names.

---

## 8. Concurrency model

The combined goal — multiple agents on different branches — needs three
things to be true:

1. **One backend container per `(user, project)`**, shared across
   conversations. Lock key for "create or attach": `repo-cache:{user_id}:
   {project_id}` (or whatever `RepoCache.key()` ends up being). Held only
   for the create critical section.

2. **Per-worktree write isolation.** Today's lock key
   `workspace-command:{workspace_id}` is right (
   `application/services/sandbox_service.py:91-96`) — keep it. Two agents
   on two branches don't share a lock; two agents *in the same
   conversation* serialize.

3. **No process-wide singletons.** `SandboxClient` is per-(potpie process)
   and stateless except for the underlying provider/store/locks. Multiple
   Celery workers each hold their own client; they coordinate through the
   store (which becomes interesting — see below).

### 8.1 Store: in-memory vs durable

`InMemorySandboxStore` and `JsonSandboxStore` work fine for one process.
Multiple Celery workers need something durable + concurrent: a Postgres
table is the path of least resistance (we already have the DB session).
Add a `PostgresSandboxStore` adapter; use the same `SandboxStore` port
unchanged. The implementer should look at the existing context-graph pot
migrations (the recent `20260407_*` migration) for precedent on how
sandbox-related schema gets shipped.

### 8.2 Locks: in-memory vs Redis

Same shape. `InMemoryLockManager` is fine for tests and single-worker.
Production wants `RedisLockManager` (we have Redis; the
`code_changes_manager`'s storage layer already uses it). Same `LockManager`
port — drop in.

The implementer should not block the v1 plan on these; ship the Postgres
store + Redis lock as a follow-up if a single-worker rollout is enough
for the first migration step.

### 8.3 Cancellation

`ChatContext.check_cancelled` is the current escape hatch. Sandbox tools
that call long-running `exec` should poll it (or pass a timeout that
respects it). `SandboxClient.exec` should accept an optional
`cancellation_token` callable.

---

## 9. Migration: removing `CodeChangesManager` and `.repos` glue

The two systems are intertwined. Order:

1. **Stand up the new client surface.** Build `SandboxClient` over the
   existing service, helpers and all. Local-fs backend is enough.
2. **Build the new sandbox tools** (§7) and register them under their own
   names. Don't delete the old tools yet.
3. **Move parsing.** `ParseHelper.clone_or_copy_repository` is the only
   place outside agents that touches `RepoManager`. Replace its
   `RepoManager` calls with `SandboxClient.get_workspace` (using the
   local-fs backend, ANALYSIS mode). Tests in
   `app/modules/parsing/tests/` should still pass.
4. **Migrate one agent** end-to-end — start with `code_gen_agent` since
   it's the most-touched. Replace its hardcoded tool list with the new
   sandbox tool names; update its system prompt. Run the existing
   end-to-end smoke tests.
5. **Delete `CodeChangesManager`.** Once `code_gen_agent` is on the new
   tools, the staging tools (`add_file_to_changes`, `update_file_lines`,
   `replace_in_file`, etc.) have no callers. Delete the package, the
   Redis schema, the `_init_code_changes_manager` plumbing in
   `execution_flows.py`, and the imports that scatter from there.
6. **Make `.repos` an adapter.** At this point `RepoManager` still exists
   as an implementation detail of the local-fs `WorkspaceProvider`.
   Refactor: move the bare-repo + worktree code from
   `app/modules/repo_manager/` into
   `app/src/sandbox/sandbox/adapters/outbound/local/repo_cache.py`
   (or similar), keeping the auth chain (`sync_helper.py`'s GitHub App →
   OAuth → env-token logic) since it's hard-won. The shape of
   `LocalGitWorkspaceProvider` doesn't change much — it just absorbs
   `RepoManager`'s functionality.
7. **Delete `app/modules/repo_manager/`.** The legacy module's only job
   was to be that adapter; now the sandbox owns it.

Each step ships independently. Steps 1–4 are additive; 5–7 are deletions.
Don't intermix.

### 9.1 What to do with the staging idea

`CodeChangesManager` had a useful property: edits were reviewable before
commit. Without it, agent edits land directly in a worktree branch. That's
*also* reviewable (it's a git branch you can diff against base), and
arguably better — diffing a branch is a tool the LLM and humans both
already understand.

If we miss "show the user a pending-changes summary," that's a UI concern
on top of `git diff`, not a sandbox concern. Don't put it back in.

### 9.2 Apply-changes flow

Current flow: `apply_changes` tool reads from `CodeChangesManager` Redis
and writes files into the worktree. New flow: there is no `apply_changes`
— each `sandbox_write_file` / `sandbox_str_replace` writes directly. The
`git_commit` tool (which is now `sandbox_git_commit`) commits whatever is
staged. There's nothing to "apply."

### 9.3 Local mode (VS Code extension)

`local_mode` in `ChatContext` switches several tools to talk to a local
tunnel instead of the worktree. The plan: keep the local-mode path in
the *agent* layer, not the *sandbox* layer. When `local_mode` is on,
the agent uses a different tool set (the existing tunnel-backed tools)
and bypasses `SandboxClient` entirely. The sandbox library doesn't need
to know about VS Code.

### 9.4 Auth token handling

`WorkspaceRequest.auth_token` is already plumbed (the daytona adapter
embeds it via the `#__potpie_token__=...` marker; the local adapter
embeds it as `x-access-token:...@host`). Tokens are never persisted
(`adapters/outbound/file/json_store.py:125`). Keep that property — the
`SandboxClient` API takes the token at workspace creation, and the rest
of the system never sees it again. The implementer should keep the
auth-chain code (`sync_helper.py`) intact when it moves into the
adapter, since GitHub App → OAuth → env-token is more than the sandbox
should re-derive.

---

## 10. `.repos` as an adapter

After §9.6 the local-fs adapter lives at
`adapters/outbound/local/`. Concretely:

- `repo_cache.py`: bare-repo lifecycle (clone, fetch, GC, eviction).
  Absorbs the meaningful parts of `repo_manager.py`.
- `git_workspace.py` (existing, expanded): worktree lifecycle on top of
  the cache.
- `subprocess_runtime.py` (existing): host execution.
- `auth.py`: GitHub App → OAuth → env-token chain. Absorbs
  `sync_helper.py`'s logic.

Public behavior the local adapter must preserve:

- `.repos/<owner>/<repo>/.bare/`, `.../worktrees/<name>/` paths — for
  developer ergonomics and for the existing parsing flow which knows
  these paths.
- `REPOS_BASE_PATH` env var — keep it as the config knob.
- `GH_TOKEN`, `GH_TOKEN_LIST`, `GITHUB_BASE_URL` env vars — keep them.
- The existing eviction policy — port it; don't re-design as part of this.

Things the local adapter may stop doing (because the sandbox layer
above it now does them):

- The `.meta/<owner>/<repo>/branch__commit.json` files. The
  `SandboxStore` is now the source of truth for branch/commit/age
  metadata. Migrate the data into the store on first read; delete the
  files on write.

---

## 11. Prompt updates

System prompts in `app/modules/intelligence/agents/chat_agents/system_
agents/code_gen_agent.py` and the generic `pydantic_agent.py` reference
the old tool names extensively (lines 30–122 and 413–650+ in
`code_gen_agent.py`; lines 172–216 in `pydantic_agent.py`).

Update plan:

- Replace tool-name references with the new ones.
- Replace any "use `CodeChangesManager` to stage edits" framing with
  "edit the file directly; the worktree is yours." Make explicit that
  edits are durable from the moment the tool returns and that
  `sandbox_git_commit` formalizes them.
- Add a short "Sandbox tools" section enumerating the catalog, mirroring
  how `bash_command_tool`'s description is structured.
- Verify the `local_mode` branch of the prompt still hangs together
  given §9.3 — the local-mode tools stay, just without the sandbox
  ones.

The prompts are long. The implementer should diff carefully and run the
agent against a small smoke test (the existing E2E in
`agents/integration/`) before declaring victory.

---

## 12. Phasing

Rough order; each phase is independently shippable.

1. **Image + ripgrep.** Ship `potpie/agent-sandbox` Docker image,
   register as the daytona snapshot default. Verify `rg`, `git`, `jq`
   present in all backends.
2. **Client surface.** `SandboxClient` over existing service. Helpers
   (`read_file`, `write_file`, `list_dir`, `search`, `commit`, `push`).
   Unit tests. No callers yet.
3. **One sandbox per `(user, project)`** for the daytona backend.
   Worktree-per-branch inside it. New e2e test that two workspace
   requests with different branches share the same Daytona sandbox.
4. **Sandbox-backed agent tools** registered in `ToolService`. Old
   tools still present.
5. **Migrate `code_gen_agent`** to the new tools. Update its prompt.
   Smoke test.
6. **Migrate parsing.** `ParseHelper` uses `SandboxClient` via local-fs
   backend. Re-run parsing tests.
7. **Migrate remaining agents.** One per PR; same pattern as `code_gen`.
8. **Delete `CodeChangesManager`.** Including its Redis schema,
   lifecycle plumbing, and the staging tool family.
9. **Move `.repos` glue into the local adapter.** Delete
   `app/modules/repo_manager/`. Migrate `.meta` JSON state on first
   read.
10. **Durable store + Redis locks.** When we go multi-worker.
11. **Background eviction.** Separate from the request path.

Phases 1–3 are foundational and can land concurrently. 4–7 are the
visible migration. 8–9 are the cleanup. 10–11 are scaling work, not
correctness work.

---

## 13. Open questions

These are real choices the implementer should make explicitly, not pretend
the plan answered:

- **One image or two.** Slim vs full. Cold start vs feature coverage.
  Daytona snapshot pull time matters here — measure it.
- **Per-user vs shared `RepoCache`.** Sharing saves disk and clone time;
  separating is simpler for auth and quota. Default to per-user; design
  the key so we can flip later.
- **Where the `SandboxRunContext` contextvar lives.** Closest analogue
  is `_code_changes_manager_ctx` in
  `code_changes_manager/context.py`. Same pattern, different module.
- **Whether to keep `WorkspaceMode`.** ANALYSIS / EDIT / TASK currently
  drive branch naming. Once parsing goes through us with mode=ANALYSIS
  and agents with mode=EDIT, we have one more user (TASK?) — or we
  collapse to two (read vs write). Simpler is better; collapse if the
  test suite still passes.
- **Streaming exec.** `exec_stream` exists in the port but isn't used.
  Promote it once an agent flow needs progress (long test runs, builds).
- **PR creation.** `code_provider_create_pr` is a tool today and lives
  outside the sandbox. Either fold it into `sandbox_open_pr` (clean,
  but ties auth to the sandbox lib), or keep it separate (clean
  layering, slightly worse ergonomics). Lean toward separate.
- **Multi-tenancy at the Daytona level.** One Daytona org per potpie
  deployment? Per user? The dev stack has one `dev@daytona.io` user;
  production needs a story. Out of scope here, but the sandbox client's
  config shape should not preclude per-user Daytona credentials.

---

## 14. What this replaces, in one table

| Concept                                  | Today                                                                | After                                                       |
| ---------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------- |
| Bare repo + worktree on disk             | `app/modules/repo_manager/`                                          | `app/src/sandbox/.../adapters/outbound/local/`              |
| In-flight file edits                     | `CodeChangesManager` (Redis, 24h TTL)                                | The worktree + git                                          |
| Edit-staging tools (~20)                 | `code_changes_manager/tools.py`                                      | A dozen `sandbox_*` tools, mostly just file/git ops         |
| Repo materialization for parsing         | `ParseHelper.clone_or_copy_repository` → `RepoManager`               | `SandboxClient.get_workspace(mode=ANALYSIS)`                |
| Repo materialization for agents          | Tool internals call `get_or_create_edits_worktree_path` per call     | One `SandboxClient.get_workspace` per run, cached on ctx    |
| Branch concurrency                       | One worktree per `(user, repo, conversation)` on disk                | Many worktrees per Daytona sandbox; one cache per repo      |
| Clone auth                               | `sync_helper.py` (GitHub App → OAuth → env)                          | Same code, lives in the local adapter                       |
| `.repos` directory                       | The blessed location                                                 | One adapter's storage; equivalent in Daytona is a sandbox   |
| Eviction                                 | Inline during `prepare_for_parsing`                                  | Background task per adapter                                 |
| Locks                                    | None at app layer; OS file locks for git                             | `LockManager` per workspace; pluggable Redis impl           |
| Durable store                            | DB tables for `Project`; JSON in `.meta/`                            | `SandboxStore` (Postgres adapter when multi-worker)         |
| Tool wiring                              | Hardcoded allow-lists + emerging registry                            | Registry-only; sandbox tools as a group                     |

---

If something in here looks underspecified, that's by design — the plan
should make the call points obvious without freezing the implementation.
The right next step is phase 1: build the image, snapshot it, and prove
`rg` is available in a Daytona sandbox end-to-end. Everything else
follows.
