# Potpie / Context Engine Package-Boundary Migration

> Status: Commits 1-5 complete; Commit 6 not started. This document implements the
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
| 1 | `docs(spec): define the potpie package boundary` | All IDs as proposed contract | Complete | `322bccacf4c9b91bf7d8b3cd10ae043c443302e6` | 11 spec files, 18 behavior IDs, metadata/backlink/link validation, diff checks |
| 2 | `test(boundary): characterize engine and product behavior` | `PKG-VERIFY-001` | Complete | `d691ea06fc7e642125c2e106c9807f55331f1d7d` | 10 characterization tests; root/engine suites and premerge journey pass separately |
| 3 | `refactor(engine): namespace the context engine package` | `PKG-OWN-002`, `PKG-API-001`, `PKG-DIST-001` | Complete | `5f22bd0` | 1,151 engine tests, 999 root unit tests, isolated wheel/import and namespace scans pass |
| 4 | `refactor(engine): add the standalone ContextEngine API` | `PKG-API-001`, `PKG-CONFIG-001`, `PKG-SETUP-001`, `PKG-STATUS-001` | Complete | `e00b4cb` | Public API, explicit paths, no-home-write, HTTP factory, full-suite, and isolated-wheel tests pass |
| 5 | `refactor(runtime): introduce PotpieRuntime and typed daemon RPC` | `PKG-RUNTIME-001`, `PKG-MODE-001`, `PKG-RPC-001` | Complete | This commit | 30-method schema registry, mode precedence, no fallback, local/daemon parity, and detached-daemon tests pass |
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

Evidence recorded on 2026-07-12:

- `uv run pytest tests/characterization -q`: 5 root tests passed.
- `uv run --project . pytest tests/characterization -q` from
  `potpie/context-engine`: 5 engine tests passed.
- `uv run --project . pytest tests -m "not premerge_journey" -q` from
  `potpie/context-engine`: 1,151 passed and 32 skipped.
- `uv run pytest tests -m "not premerge_journey" -q`: 1,053 passed, 4 skipped,
  and 1 separately gated journey deselected.
- `PREMERGE_JOURNEY_REQUIRED=1 uv run pytest
  tests/integration/test_premerge_cli_journey.py -q`: 1 passed.
- Root RPC tests were aligned to the current root-owned daemon-client import and
  current structured validation-error payload; no production behavior changed.
- Characterization fixtures cover current pot/source, graph, ledger cursor,
  credential, and daemon discovery formats.
- MCP fixtures capture the field, required-argument, nullable-type, and default
  contracts for `context_resolve`, `context_search`, and `context_record`.

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

Evidence recorded on 2026-07-12:

- `uv run --project . pytest tests -m "not premerge_journey" -q` from
  `potpie/context-engine`: 1,151 passed and 32 skipped.
- `uv run pytest tests/unit -q`: 999 root unit tests passed.
- `uv run pytest tests -m "not premerge_journey" -q`: 1,054 passed, 4 skipped,
  and 1 separately gated journey deselected.
- The engine wheel contains `potpie_context_engine` only; benchmarks and the
  former generic top-level packages are absent.
- A clean interpreter imports `potpie_context_engine` and its domain namespace
  from the built wheel.
- Engine and root Ruff checks, `git diff --check`, and the repository-wide scan
  for legacy generic imports pass.
- A supplemental legacy suite ran 49 of 50 tests successfully. Its remaining
  import-only test fails inside NetworkX under Python 3.14 before application
  code loads (`wrapper_descriptor` has no `__annotate__`); this is outside the
  namespace gate and does not reproduce in the supported root/engine suites.

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

Evidence recorded on 2026-07-12:

- The package exports `ContextEngine`, `EngineClient`, `EngineConfig`,
  `EngineDependencies`, and `create_engine`; request/result DTOs are declared
  under `potpie_context_engine.contracts`.
- Five focused API tests cover configuration validation, in-memory no-home
  construction, caller-owned persistent paths, typed async operations, pure
  status/provision results, and optional HTTP application injection.
- `uv run --project . pytest tests -m "not premerge_journey" -q`: 1,156 passed
  and 32 skipped.
- `uv run pytest tests/unit -q`: 999 root unit tests passed.
- A newly built engine wheel installed into a clean virtual environment and ran
  in-memory resolve/status without root Potpie installed or creating the
  deliberately nonexistent `HOME` path.
- Engine Ruff checks and `git diff --check` pass.

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

Evidence recorded on 2026-07-12:

- `ProductSettings`, `PotpieRuntime`, `LocalEngineClient`,
  `DaemonEngineClient`, `create_runtime`, and `get_runtime` are root-owned.
- Runtime precedence tests cover explicit override, `POTPIE_RUNTIME_MODE`,
  persisted configuration, and daemon default; invalid values fail closed.
- Daemon unavailability raises `RUNTIME_DAEMON_UNAVAILABLE` with
  `potpie daemon start` and never constructs a local engine.
- The protocol-v1 registry declares 30 `engine.*` methods. Every entry has a
  request adapter, result adapter, and allowlisted handler; product methods and
  dynamic attributes are absent.
- Protocol mismatch, unknown method, malformed parameters, authorization, and
  removed `/attr` behavior fail deterministically before dispatch.
- Representative context, pot, source, graph, ledger, timeline, and provision
  results are equal between direct local calls and typed RPC encode/dispatch/
  decode.
- A real detached daemon starts with one `ContextEngine`; typed client calls and
  backend-preserving restart pass.
- Root unit tests: 994 passed. Root non-journey suite: 1,049 passed, 4 skipped,
  and 1 separately gated journey deselected after the isolated-wheel package
  marker fix.
- Focused mypy, Ruff, pre-commit, isolated-wheel, and `git diff --check` gates
  pass.

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
