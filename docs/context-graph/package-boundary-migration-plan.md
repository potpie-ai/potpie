# Potpie / Context Engine Package-Boundary Migration

> Status: Commit 1 complete; Commit 2 not started. This document implements the
> sequencing contract for [SPEC-PACKAGE-BOUNDARY](../../spec/modules/package-boundary.md) and
> [ADR-0002](../../spec/decisions/ADR-0002-potpie-context-engine-boundary.md).
> The target architecture is proposed; current-state docs remain authoritative
> until the corresponding commit lands.

## Outcome

The migration ends with root `potpie` as the sole product/runtime distribution
and `potpie-context-engine` as a root-independent library. Product interaction
goes through `PotpieRuntime`; engine interaction is visible as
`runtime.engine.*`. The migration provides no compatibility aliases for old
imports, commands, `HostShell`, dynamic RPC, or entrypoints.

The normative behavior is in `SPEC-PACKAGE-BOUNDARY`. This document owns only
execution order, commit scope, verification commands/evidence, and progress.

## Commit Discipline

Every implementation turn performs exactly one planned commit:

1. Re-read this plan and the behavior IDs in scope.
2. Capture `git status --short` and the pre-existing diff.
3. Preserve all pre-existing user changes byte-for-byte.
4. Change only the current commit's scope.
5. Stage explicit paths or hunks; never use an unreviewed broad stage.
6. Inspect the complete staged diff.
7. Run the commit-specific gate and `git diff --check`.
8. Commit only when the gate passes.
9. Report the commit and evidence, then pause for review.

If a planned hunk overlaps a user-owned hunk, execution stops at that boundary
for user direction rather than combining changes.

## Protected Dirty Baseline

The implementation began with these user-owned changes:

```text
 M pyproject.toml
 D telemetry_build_config_values.py
 D telemetry_build_hook.py
 M tests/unit/test_root_telemetry_build_config.py
?? potpie/runtime/telemetry/build_config_values.py
?? potpie/runtime/telemetry/build_hook.py
```

They move telemetry build support under `potpie/runtime/telemetry`. Migration
commits do not stage, revert, rewrite, or absorb these changes. A later packaging
commit may stage only non-overlapping `pyproject.toml` hunks after separately
verifying the baseline remains intact.

## Progress

| # | Planned commit | Behavior scope | Status | Commit | Evidence |
|---:|---|---|---|---|---|
| 1 | `docs(spec): define the potpie package boundary` | All IDs as proposed contract | Complete | This commit | 11 spec files, 18 behavior IDs, metadata/backlink/link validation, diff checks |
| 2 | `test(boundary): characterize engine and product behavior` | `PKG-VERIFY-001` | Not started | — | Characterization fixtures and current behavior tests |
| 3 | `refactor(engine): namespace the context engine package` | `PKG-OWN-002`, `PKG-API-001`, `PKG-DIST-001` | Not started | — | Engine tests, root tests, isolated wheel import |
| 4 | `refactor(engine): add the standalone ContextEngine API` | `PKG-API-001`, `PKG-CONFIG-001`, `PKG-SETUP-001`, `PKG-STATUS-001` | Not started | — | Public API and no-home-write tests |
| 5 | `refactor(runtime): introduce PotpieRuntime and typed daemon RPC` | `PKG-RUNTIME-001`, `PKG-MODE-001`, `PKG-RPC-001` | Not started | — | DTO registry, mode, protocol, parity tests |
| 6 | `refactor(cli): route engine workflows through runtime.engine` | `PKG-RUNTIME-001` | Not started | — | Context/graph/pot/source/ledger/timeline parity |
| 7 | `refactor(product): extract auth integrations and configuration` | `PKG-AUTH-001`, `PKG-CONFIG-001`, `PKG-OWN-001` | Not started | — | Auth, credential, provider, config tests |
| 8 | `refactor(product): extract skills and installation` | `PKG-SKILL-001`, `PKG-OWN-001` | Not started | — | Installed-wheel skill and manifest tests |
| 9 | `refactor(product): split setup doctor and status` | `PKG-SETUP-001`, `PKG-STATUS-001` | Not started | — | Setup/status scenario matrix |
| 10 | `refactor(engine): remove product residue and legacy queue wiring` | `PKG-OWN-002`, `PKG-QUEUE-001`, `PKG-OBS-001` | Not started | — | Forbidden import/name scans and engine suite |
| 11 | `refactor(cli): apply the workflow-first command contract` | `PKG-CLI-001`, `PKG-CLI-002` | Not started | — | Command snapshots, removed paths, JSON tests |
| 12 | `refactor(mcp): move the public MCP server into potpie` | `PKG-MCP-001`, `PKG-STATUS-001` | Not started | — | Four-tool discovery and parity tests |
| 13 | `build(packaging): finalize distribution boundaries` | `PKG-DIST-001`, `PKG-API-001` | Not started | — | Wheels, sdists, metadata, entrypoints, isolated installs |
| 14 | `docs(architecture): publish the completed package split` | `PKG-VERIFY-001` and all IDs | Not started | — | Full suite, docs, final verification record |

The actual SHA for a completed commit is recorded during the following commit or
in the final verification record; a commit cannot contain its own stable SHA.

## Commit 1 — Documentation Contract

Commit message: `docs(spec): define the potpie package boundary`

Changes:

- Initialize `spec/index.md`, `process.md`, `glossary.md`, `product.md`, and
  `system.md`.
- Initialize module, decision, question, and verification indexes.
- Accept `ADR-0001` for the spec process.
- Write `SPEC-PACKAGE-BOUNDARY` version `0.1.0`, status `proposed`.
- Accept `ADR-0002` for the product/engine split.
- Write this migration plan.
- Add target-spec banners to the current architecture, CLI, vision, skills,
  telemetry, and CLI package docs without rewriting their current-state bodies.

Gate:

- Every spec file has valid required metadata.
- Every normative behavior has a stable ID.
- `SPEC-PACKAGE-BOUNDARY` and `ADR-0002` link in both directions.
- Index, glossary, product, system, question, and verification relationships are
  consistent.
- There are no unresolved boundary questions.
- Local Markdown links resolve.
- `git diff --check` passes.
- The staged diff contains only documentation/spec paths and excludes the dirty
  telemetry baseline.

## Commit 2 — Characterization

Commit message: `test(boundary): characterize engine and product behavior`

Changes:

- Capture resolve/search/record behavior.
- Capture pot/source, graph read/write, ledger, timeline, and persistent-data
  behavior.
- Capture existing MCP schemas for resolve/search/record.
- Add result normalization for local/daemon comparisons.
- Add isolated-import test harnesses used by later commits.
- Do not snapshot obsolete command placement.

Gate: new characterization tests and existing relevant root and engine suites
pass separately.

## Commit 3 — Engine Namespace

Commit message: `refactor(engine): namespace the context engine package`

Changes:

- Move reusable code under `src/potpie_context_engine`.
- Rewrite imports mechanically without behavior changes.
- Move package resources into the namespace.
- Keep benchmarks outside the runtime wheel and make them import the namespaced
  package.
- Update workspace callers sufficiently to keep tests green.

Gate: engine unit/conformance tests, root unit tests, and isolated engine-wheel
import pass.

## Commit 4 — Standalone Engine API

Commit message: `refactor(engine): add the standalone ContextEngine API`

Changes:

- Add `EngineConfig`, `EngineDependencies`, `EngineClient`, `ContextEngine`,
  `create_engine`, and public DTOs.
- Add context, pots, sources, graph, ledger, timeline, provision, and lifecycle
  capabilities.
- Add persistent and in-memory construction.
- Add optional injected HTTP application construction.
- Stop exporting `HostShell`; retain it internally only until root migration.

Gate: a clean environment installs the engine wheel and runs in-memory
resolve/status without root Potpie or home-directory writes.

## Commit 5 — Product Runtime and RPC

Commit message: `refactor(runtime): introduce PotpieRuntime and typed daemon RPC`

Changes:

- Add `ProductSettings`, `PotpieRuntime`, `LocalEngineClient`, and
  `DaemonEngineClient`.
- Add protocol-v1 DTOs, explicit method registry, and `/healthz`.
- Make daemon mode the product default and remove silent fallback.
- Add stable unavailable and protocol-mismatch failures.
- Host `ContextEngine` inside the daemon.

Gate: every engine RPC has request/result schema coverage; local and daemon
results match after transport metadata is removed.

## Commit 6 — Engine Workflow Routing

Commit message: `refactor(cli): route engine workflows through runtime.engine`

Changes:

- Route context, pot, source, graph, ledger, and timeline commands through
  `runtime.engine.*`.
- Remove direct CLI imports of engine types and internals.
- Limit handlers to parse, call, render.
- Retain current command placement until Commit 11.

Gate: migrated commands pass in explicit daemon and in-process modes.

## Commit 7 — Auth, Integrations, and Config

Commit message: `refactor(product): extract auth integrations and configuration`

Changes:

- Move account auth, provider OAuth clients, keyring, and credential persistence
  out of `potpie/cli`.
- Add account, integration, and product-config services.
- Map product identity to an engine actor only at the runtime boundary.
- Make backend selection a product-setting operation.

Gate: account/provider/credential/config tests pass without engine-private
imports or engine credential access.

## Commit 8 — Skills and Installation

Commit message: `refactor(product): extract skills and installation`

Changes:

- Move bundled resources to root skills.
- Move/rewrite catalog, manager, targets, drift, and installer services.
- Add a static product command manifest.
- Compare Typer registration to the manifest in tests.
- Validate every packaged command snippet at build/test time.

Gate: install/update/remove/status work from an installed root wheel; invalid
template commands fail before release.

## Commit 9 — Setup, Doctor, and Status

Commit message: `refactor(product): split setup doctor and status`

Changes:

- Implement the root product setup workflow.
- Limit engine provisioning to typed inspect/apply operations.
- Implement pure engine status and flat root status.
- Start/reconcile daemon before daemon-mode provisioning calls.

Gate: fresh, already configured, daemon unavailable, provision failed, missing
skills, and degraded backend scenarios pass in human and JSON modes.

## Commit 10 — Engine Residue Removal

Commit message: `refactor(engine): remove product residue and legacy queue wiring`

Changes:

- Inject `ContextGraphJobQueuePort`.
- Remove path mutation and legacy queue import.
- Separate generic engine observability from product telemetry.
- Remove engine-owned auth/config/skills/install/daemon lifecycle.
- Replace `bootstrap`/`host` with engine-only composition.
- Delete `HostShell`, `RemoteHostShell`, `RemoteSurface`, and dynamic RPC.

Gate: forbidden import/name scans are zero and the complete engine suite passes.

## Commit 11 — CLI Contract

Commit message: `refactor(cli): apply the workflow-first command contract`

Changes:

- Add the shared success/error JSON envelope and exit-code mapping.
- Apply the exact command tree in `SPEC-PACKAGE-BOUNDARY`.
- Nest providers under `integration`, backend under `graph`, and local service
  lifecycle under `daemon`.
- Remove cloud, top-level use, legacy provider/auth groups, and graph aliases.
- Normalize provider resource enumeration to `list`.

Gate: command-tree snapshots, removed-command failures, JSON envelope tests, and
CLI journeys pass.

## Commit 12 — Root MCP

Commit message: `refactor(mcp): move the public MCP server into potpie`

Changes:

- Create root `potpie/mcp`.
- Register exactly the four approved tools.
- Route resolve/search/record through `runtime.engine.context`.
- Route status through root product status.
- Remove engine MCP process ownership.

Gate: MCP discovery reports exactly four tools and local/daemon/schema tests
pass.

## Commit 13 — Distribution Metadata

Commit message: `build(packaging): finalize distribution boundaries`

Changes:

- Root owns all three scripts.
- Engine owns no script and becomes version `0.2.0`.
- Engine uses a minimal core and explicit capability extras.
- Root selects explicit engine extras rather than `[all]`.
- Move product dependencies to root and remove stale package exclusions/hooks.
- Update the lockfile without unrelated dependency churn.

Gate: wheels/sdists build, metadata validates, entrypoints and wheel contents are
exact, and clean-environment smokes pass.

## Commit 14 — Current Documentation and Verification

Commit message: `docs(architecture): publish the completed package split`

Changes:

- Rewrite architecture, CLI flow, skills, telemetry, installation, and package
  docs to describe the landed code.
- Publish the clean-break import/command/RPC migration notes.
- Complete this progress table with evidence.
- Promote `SPEC-PACKAGE-BOUNDARY` to accepted `1.0.0`, then implemented and
  verified only when the corresponding lifecycle evidence exists.
- Add a verification record covering every behavior ID.

Gate: full root and engine suites, static checks, package builds, isolated
installs, docs links, cross-spec checks, and final residue scans pass.

## Final Release Gate

The migration is incomplete until all acceptance criteria in
`SPEC-PACKAGE-BOUNDARY` pass. In particular:

- root imports only the declared engine public API;
- engine imports no root Potpie code;
- CLI and MCP obtain `PotpieRuntime` and do not import engine internals;
- `HostShell`, dynamic surfaces, generic top-level packages, and path hacks are
  absent;
- daemon failure does not change runtime mode;
- CLI JSON and MCP status contracts match the spec;
- root and engine wheels expose their exact intended scripts and namespaces;
- existing persistent-data fixtures remain compatible;
- a commit-scoped verification record proves every stable behavior ID.
