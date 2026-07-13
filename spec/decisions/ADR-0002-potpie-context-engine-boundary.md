---
id: ADR-0002
title: Separate Product Runtime from Context Engine
kind: decision
status: accepted
version: 1.0.1
owners:
  - Potpie Engineering
depends_on:
  - ADR-0001
related_specs:
  - SPEC-GLOSSARY
  - SPEC-PRODUCT
  - SPEC-SYSTEM
  - SPEC-PACKAGE-BOUNDARY
related_decisions:
  - ADR-0001
affects:
  - SPEC-GLOSSARY
  - SPEC-PRODUCT
  - SPEC-SYSTEM
  - SPEC-PACKAGE-BOUNDARY
open_questions: []
verification:
  code_status: not_applicable
  verified_commit: null
  verified_at: null
  verified_by: null
  behavior_scope: []
  evidence: []
  cross_spec_status: passed
  cross_spec_checked_against:
    - SPEC-GLOSSARY
    - SPEC-PRODUCT
    - SPEC-SYSTEM
    - SPEC-PACKAGE-BOUNDARY
  drift_status: unverified
---

# ADR-0002: Separate Product Runtime from Context Engine

## Context

The branch has already moved the `potpie` CLI and daemon executables into the
root distribution, but their runtime still imports generic engine packages and
binds commands to an engine-owned `HostShell`. That shell includes engine
operations alongside product auth, setup, skills, installation, config, daemon,
and status responsibilities.

Detached execution reproduces the same shape through `RemoteHostShell` and
dynamic `RemoteSurface` objects. The RPC codec carries Python module and class
identities, so transport compatibility depends on matching internal package
layout rather than a stable wire contract.

The engine wheel also owns `potpie-mcp`, ships generic top-level packages, and
contains product-facing capabilities. Consequently it is not yet a clean
standalone library even though root Potpie is becoming the public product
distribution.

## Decision

Adopt the ownership and behavior contract in
[SPEC-PACKAGE-BOUNDARY](../modules/package-boundary.md).

The decisions are:

1. Root `potpie` is the sole product/runtime distribution and owns the `potpie`,
   `potpie-daemon`, and `potpie-mcp` processes.
2. `potpie-context-engine` is a standalone, root-independent library imported
   through `potpie_context_engine`.
3. Root composes a `PotpieRuntime`; engine interaction is visible under
   `runtime.engine.*` and product services remain siblings.
4. `ContextEngine` is the in-process engine facade. A typed `EngineClient`
   protocol is implemented locally and by root's daemon client.
5. Daemon RPC is an explicit versioned DTO protocol limited to `engine.*`
   operations. Python module/class identity is not serialized.
6. Product settings and engine configuration are separate. Persistent library
   use requires an explicit data directory; in-memory use does not touch the
   user's home directory.
7. Root owns product auth, provider auth, credentials, setup, skills,
   installation, status enrichment, daemon lifecycle, MCP, UI, and product
   telemetry.
8. The engine owns context/graph domain behavior, application use cases, ports,
   backends, stores, connector adapters, pure provisioning, generic
   observability, and optional HTTP factories.
9. Daemon mode is the product default. An unavailable daemon fails explicitly;
   there is no silent in-process fallback.
10. The public CLI is redesigned to the workflow-first hierarchy in the spec,
    with one JSON envelope and no deprecated aliases.
11. Public context status is flat and root-composed. MCP remains exactly four
    tools.
12. The migration is a clean break for imports, commands, facades, entrypoints,
    and RPC. Existing persistent data formats and product paths remain
    compatible.

## Consequences

### Positive

- The standalone engine can be installed, imported, configured, and tested
  without root Potpie or user-home side effects.
- Product concerns have a single owner and do not leak into reusable engine
  code.
- CLI and MCP share product composition without making presentation part of the
  engine.
- Local and daemon behavior can be compared through one typed protocol.
- The wire contract is resilient to internal Python file moves.
- Wheel contents and executable ownership become directly testable.
- `runtime.engine.*` makes architecture violations visible during review.

### Costs

- `HostShell`, dynamic RPC, current command paths, engine imports, and package
  metadata require coordinated rewrites.
- Consumers of generic engine imports must migrate directly to the public
  namespace; no shim is provided.
- Existing daemon processes must restart when protocol v1 lands.
- The branch is not release-ready during intermediate commits that retain
  unused legacy implementation for sequencing.
- Documentation must distinguish current state from target state until each
  implementation phase lands.

## Alternatives Considered

### Keep `HostShell` as the public boundary

Rejected because it combines product and engine ownership, obscures which
operations are remote, and forces a standalone library to know product services.

### Move every non-engine concern into `potpie/cli`

Rejected because auth, config, setup, skills, installation, daemon lifecycle,
and telemetry are product services used by more than Typer presentation. They
need first-class root modules, not a larger CLI package.

### Keep MCP in the engine

Rejected because public status includes root runtime, daemon, setup, and skills
state, and because the public process belongs to the user-installed product.

### Keep dynamic RPC with a stricter allowlist

Rejected because even allowlisted dynamic object traversal and class-path
serialization couple transport compatibility to internal implementation shape.

### Silently fall back to an in-process engine

Rejected because it changes persistence and process ownership invisibly, can
hide daemon failures, and makes a first-class daemon operationally ambiguous.

### Make in-process mode the product default

Rejected because the chosen product model treats the daemon as the persistent
runtime. Explicit in-process mode remains available for embedding, testing, and
intentional local use.

### Preserve deprecated commands and import aliases

Rejected because this branch is defining a clean product boundary and the user
has accepted a breaking CLI redesign. Compatibility aliases would prolong the
old ownership model and complicate validation.

### Put the engine inside the root `potpie` Python namespace

Rejected because the engine is independently installable and versioned. Its
`potpie_context_engine` namespace makes that distribution boundary explicit and
prevents accidental shadowing by generic top-level packages.

## Affected Specs

- [SPEC-GLOSSARY](../glossary.md)
- [SPEC-PRODUCT](../product.md)
- [SPEC-SYSTEM](../system.md)
- [SPEC-PACKAGE-BOUNDARY](../modules/package-boundary.md)

## Follow-Up Status

- `SPEC-PACKAGE-BOUNDARY` version 2.0.0 is accepted with the public graph-store
  command vocabulary; version 1.0.0 remains verified at `f435fb4`.
- Implementation evidence is recorded commit by commit in the migration plan.
- The 1.0.0 verification record covers every behavior at `f435fb4` with no known
  gap. Version 2.0.0 awaits commit-scoped verification.
- Future auth, daemon, setup, CLI, MCP, and observability module specs link back
  to this ADR when they are introduced.
