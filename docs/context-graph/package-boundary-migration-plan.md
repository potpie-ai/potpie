# Potpie Package-Boundary Migration Plan

> Status: Phase 1 planned. This is an execution and review plan, not an
> accepted package-boundary specification or a claim that the migration is
> implemented.
>
> Baseline: `origin/main` at `d9bd5644`.
> Reference implementation: `feat/cli_daemon`.
> Preserved follow-up work: `agent/archive-cli-daemon-2026-07-15`.

## Goal

Move Potpie's user-facing CLI, daemon, and MCP processes into the root `potpie`
distribution without redesigning their behavior. A subsequent PR will make
`potpie-context-engine` an independent library. Daemon, CLI, runtime, and
quality-of-life changes follow only after those two ownership gates land.

The migration is deliberately split into these stages:

1. **Executable ownership:** relocate the existing CLI, daemon, and MCP code.
2. **Library independence:** extract and verify the standalone Context Engine
   package.
3. **Behavioral changes:** improve daemon RPC, runtime composition, CLI
   workflows, and other product behavior in focused PRs.

## Phase 1 Scope

### In scope

- Move `adapters/inbound/cli` to `potpie/cli`.
- Move the daemon lifecycle, process, transport, managed-service, and UI code
  to `potpie/daemon`.
- Move `adapters/inbound/mcp` to `potpie/mcp`.
- Update imports required by those moves.
- Update the root wheel, console-script targets, packaged templates, and
  assets required by the relocated processes.
- Move or update tests whose ownership follows the relocated process code.
- Update documentation paths only where the old locations become false.

### Explicitly out of scope

- Introducing `PotpieRuntime` or a new product composition model.
- Replacing `HostShell`, changing local-versus-daemon routing, or removing
  compatibility layers.
- Replacing dynamic daemon RPC with a typed or allowlisted protocol.
- Removing `/attr`, changing `/rpc`, changing discovery files, or changing
  daemon identity and lifecycle behavior.
- Changing backend defaults, host-mode defaults, storage formats, or legacy
  environment-variable behavior.
- Renaming, adding, removing, or regrouping CLI commands.
- Changing JSON envelopes, exit codes, prompting, or output formatting.
- Renaming `graph backend` to `graph store`.
- Changing MCP tool names, schemas, authorization, or its in-process
  `HostShell` execution model.
- Changing package versions, dependency policy, or release metadata beyond
  what is mechanically required to package the moved files.
- Namespacing or redesigning `potpie-context-engine`; that is Phase 2.

## Phase 1 Behavior Locks

"Move as-is" means externally observable behavior remains stable even though
module paths and package ownership change.

### Distribution entrypoints

The root distribution continues to expose exactly:

- `potpie`
- `potpie-daemon`
- `potpie-mcp`

The implementation targets may change from their current generic module paths
to `potpie.*`, but all three targets must remain importable callables.

### CLI

The top-level command set remains exactly:

```text
auth backend cloud config confluence daemon doctor git github graph jira ledger
linear login logout pot record resolve search service setup skills source status
telemetry timeline ui use whoami
```

The current `HostShell` routing, command help, JSON behavior, output behavior,
and exit semantics remain unchanged.

### Daemon

The daemon continues to expose these process routes:

| Route | Method |
|---|---|
| `/health` | `GET` |
| `/rpc` | `POST` |
| `/attr` | `POST` |

The existing dynamic RPC surfaces, discovery files, authorization, lifecycle,
UI mounting, backend selection, and error payloads remain unchanged.

### MCP

The MCP server continues to expose exactly:

- `context_resolve`
- `context_search`
- `context_record`
- `context_status`

It continues to build and cache the existing in-process `HostShell`. Phase 1
does not route MCP through a new runtime or daemon client.

## Dependency-Direction Guardrail

The root `potpie` distribution may continue to depend on
`potpie-context-engine`. Phase 1 must not add a declared dependency from
`potpie-context-engine` back to root `potpie`.

If engine-owned code still imports a module being moved, the change must make
that dependency explicit during review. Prefer moving product-owned callers or
using a narrow, temporary compatibility seam over creating a hidden package
cycle. Full removal of product residue from the engine belongs to Phase 2.

Phase 1 must remain test-green and installable, but it is not evidence that the
Context Engine is independently packaged. No documentation or release notes may
claim that boundary until Phase 2 is verified.

## Commit Plan

Every commit stages explicit paths or hunks, inspects the full staged diff, and
runs its focused gate. Do not use an unreviewed broad stage.

1. `docs/test: define relocation scope and characterize process surfaces`
2. `refactor(cli): relocate the existing CLI`
3. `refactor(daemon): relocate the existing daemon`
4. `refactor(mcp): relocate the existing MCP server`
5. `build: verify root process packaging and artifacts`

Where practical, each relocation commit should remain independently test-green.
Import-only changes must not be combined with command, protocol, runtime, or
domain behavior changes.

## Verification Gates

The characterization gate added before relocation is:

```bash
uv run pytest tests/characterization/test_product_process_surfaces.py -q
```

The focused existing gates include:

```bash
cd potpie/context-engine
uv run --project . pytest \
  tests/unit/test_agent_surface_contract.py \
  tests/unit/test_cli_usage_errors.py \
  tests/unit/test_daemon_rpc.py \
  tests/unit/test_daemon_seam.py \
  tests/unit/test_mcp_project_access.py \
  -q
```

Before the PR is ready for review:

- Run the complete root and Context Engine non-journey test suites.
- Run Ruff, mypy, pre-commit, and `git diff --check`.
- Build root and Context Engine wheels and sdists.
- Install the built root wheel in isolation and import all three console-script
  targets.
- Confirm moved templates, UI assets, and MCP resources are present in the
  root artifact.
- Review the final diff by commit and confirm that behavior changes are absent.

## Phase 2 Boundary

Phase 2 begins only after Phase 1 lands. It will namespace the engine under
`potpie_context_engine`, remove product/process residue, establish the supported
library API and contracts, define capability extras, and prove that the engine
wheel installs and operates without root Potpie.

Daemon protocol changes, CLI runtime cutover, `HostShell` removal, atomic source
registration, async-loop ownership, and CLI naming changes remain later work.
