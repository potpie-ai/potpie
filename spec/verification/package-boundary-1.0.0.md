---
id: VERIFY-PACKAGE-BOUNDARY-1.0.0
title: Potpie Package Boundary 1.0.0 Verification
kind: verification-record
status: accepted
version: 1.0.0
owners:
  - Potpie Engineering
depends_on:
  - SPEC-PROCESS
  - SPEC-PACKAGE-BOUNDARY
related_specs:
  - SPEC-GLOSSARY
  - SPEC-PRODUCT
  - SPEC-SYSTEM
related_decisions:
  - ADR-0002
affects:
  - SPEC-PACKAGE-BOUNDARY
open_questions: []
verification:
  code_status: not_applicable
  verified_commit: f435fb4
  verified_at: 2026-07-13
  verified_by: Codex
  behavior_scope:
    - PKG-OWN-001
    - PKG-OWN-002
    - PKG-API-001
    - PKG-RUNTIME-001
    - PKG-MODE-001
    - PKG-RPC-001
    - PKG-CONFIG-001
    - PKG-AUTH-001
    - PKG-SKILL-001
    - PKG-SETUP-001
    - PKG-STATUS-001
    - PKG-MCP-001
    - PKG-CLI-001
    - PKG-CLI-002
    - PKG-QUEUE-001
    - PKG-OBS-001
    - PKG-DIST-001
    - PKG-VERIFY-001
  evidence:
    - docs/context-graph/package-boundary-migration-plan.md
    - tests/characterization/test_distribution_artifacts.py
    - tests/unit/test_distribution_metadata.py
  cross_spec_status: passed
  cross_spec_checked_against:
    - SPEC-GLOSSARY
    - SPEC-PRODUCT
    - SPEC-SYSTEM
    - ADR-0002
  drift_status: current
---

# Potpie Package Boundary 1.0.0 Verification

## Result

Passed. Commit `f435fb4` implements every behavior in
`SPEC-PACKAGE-BOUNDARY` 1.0.0. No known boundary gap remains.

## Behavior Evidence

| Behavior | Verification evidence |
|---|---|
| `PKG-OWN-001` | Root owns CLI, daemon, MCP, auth, config, setup, skills, install, status, and telemetry modules; root artifact tests enumerate them. |
| `PKG-OWN-002` | Engine-to-root import scan returns zero; standalone engine-wheel smoke installs without root Potpie. |
| `PKG-API-001` | Engine wheel has one runtime namespace, `potpie_context_engine`, and exposes the declared facade/contracts API. |
| `PKG-RUNTIME-001` | Runtime composition and CLI parity tests exercise `PotpieRuntime` with visible `runtime.engine.*` routing. |
| `PKG-MODE-001` | Settings-precedence and unavailable-daemon tests prove daemon default, explicit in-process mode, and no fallback. |
| `PKG-RPC-001` | Registry/schema tests cover every typed `engine.*` method, malformed input, unknown methods, and protocol mismatch. |
| `PKG-CONFIG-001` | ProductSettings and EngineConfig tests cover distinct persistence, mode, path, and in-memory behavior. |
| `PKG-AUTH-001` | Auth/provider/keyring tests are root-owned; engine scans find no product authentication dependency. |
| `PKG-SKILL-001` | Installed-wheel skill lifecycle and static-command-manifest tests pass from packaged root resources. |
| `PKG-SETUP-001` | Setup scenario tests cover fresh, existing, unavailable daemon, failed provision, and missing skills. |
| `PKG-STATUS-001` | Pure engine status tests pass; MCP and CLI compare the same flat product status data. |
| `PKG-MCP-001` | Root MCP discovery returns exactly resolve, search, record, and status with characterized schemas. |
| `PKG-CLI-001` | Exact workflow-first command-tree snapshot passes; every removed command is unknown. |
| `PKG-CLI-002` | JSON envelope, single-stdout-value, prompt suppression, list shape, and exit-code tests pass. |
| `PKG-QUEUE-001` | Injected queue/default tests pass; scans find no path mutation or legacy Celery application import. |
| `PKG-OBS-001` | Engine contains only generic observability ports/adapters; root Sentry/PostHog tests pass. |
| `PKG-DIST-001` | Root wheel has three scripts; engine wheel has none, a Pydantic-only core, and nine explicit extras. |
| `PKG-VERIFY-001` | Full suites, static checks, artifact builds, metadata parsing, isolated installs, docs links, and residue scans pass. |

## Commands And Results

- Root suite: `uv run pytest tests -m "not premerge_journey" -q` — 1,110
  passed, 4 skipped, 1 deselected.
- Engine suite from `potpie/context-engine`: the same command — 1,050 passed,
  32 skipped.
- Artifact characterization: root and engine wheels and sdists build; metadata,
  namespaces, resources, extras, and entrypoints match the accepted contract.
- Engine isolated install: core wheel plus Pydantic runs in-memory
  create/resolve/status, creates no requested home path, and installs no product
  executable.
- Root isolated install: root and engine wheels expose exactly `potpie`,
  `potpie-daemon`, and `potpie-mcp`; all entrypoint callables import.
- Ruff, pre-commit, `git diff --check`, spec metadata, Markdown links, and
  forbidden import/name scans pass.

## Cross-Spec Review

- `SPEC-GLOSSARY` uses the same meanings for product runtime, engine, client,
  runtime mode, and product service.
- `SPEC-PRODUCT` preserves separate end-user and library-embedder install goals.
- `SPEC-SYSTEM` delegates the initial runtime/package guarantees to the boundary
  spec without contradiction.
- `ADR-0002` records the same clean break, ownership split, RPC rules, and
  rejected alternatives.

## Known Gaps

None within the package-boundary scope. Pre-existing user-owned telemetry file
moves remained outside every migration commit and were not used as verification
evidence.
