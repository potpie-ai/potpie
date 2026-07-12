---
id: SPEC-INDEX
title: Potpie Spec Index
kind: index
status: draft
version: 0.1.0
owners:
  - Potpie Engineering
depends_on:
  - SPEC-PROCESS
related_specs:
  - SPEC-GLOSSARY
  - SPEC-PRODUCT
  - SPEC-SYSTEM
  - SPEC-PACKAGE-BOUNDARY
related_decisions:
  - ADR-0001
  - ADR-0002
affects: []
open_questions: []
verification:
  code_status: not_applicable
  verified_commit: null
  verified_at: null
  verified_by: null
  behavior_scope: []
  evidence: []
  cross_spec_status: not_checked
  cross_spec_checked_against: []
  drift_status: unverified
---

# Potpie Spec Index

This is the entrypoint for Potpie's living specification. Read this file before
editing a spec or implementing behavior governed by one.

The package-boundary proposal describes a target architecture. Until a behavior
is implemented and verified, current-state documentation and code remain the
source of truth for runtime behavior.

## Read Order

1. [Spec Process](process.md)
2. [Glossary](glossary.md)
3. [Product Specification](product.md)
4. [System Specification](system.md)
5. [Potpie / Context Engine Package Boundary](modules/package-boundary.md)
6. [ADR-0002: Separate Product Runtime from Context Engine](decisions/ADR-0002-potpie-context-engine-boundary.md)
7. [Open Questions](questions/open.md)
8. [Verification Records](verification/README.md)

## Specs

| ID | File | Status | Version | Summary | Code status |
|---|---|---:|---:|---|---|
| `SPEC-PROCESS` | [process.md](process.md) | accepted | 1.0.0 | Spec lifecycle, metadata, behavior IDs, backlinks, acceptance, and verification. | not applicable |
| `SPEC-GLOSSARY` | [glossary.md](glossary.md) | proposed | 0.1.0 | Canonical package-boundary and runtime terms. | not applicable |
| `SPEC-PRODUCT` | [product.md](product.md) | draft | 0.1.0 | Product goals, users, workflows, constraints, and non-goals. | not applicable |
| `SPEC-SYSTEM` | [system.md](system.md) | draft | 0.1.0 | Cross-system guarantees and interfaces; initially scoped to the package-boundary proposal. | not implemented |
| `SPEC-PACKAGE-BOUNDARY` | [modules/package-boundary.md](modules/package-boundary.md) | proposed | 0.1.0 | Ownership and interaction boundary between root Potpie and the standalone context engine. | not implemented |

## Decisions

| ID | File | Status | Version | Summary |
|---|---|---:|---:|---|
| `ADR-0001` | [ADR-0001: Adopt a Living Spec Process](decisions/ADR-0001-spec-process.md) | accepted | 1.0.0 | Establishes this spec tree and maintenance rules. |
| `ADR-0002` | [ADR-0002: Separate Product Runtime from Context Engine](decisions/ADR-0002-potpie-context-engine-boundary.md) | accepted | 1.0.0 | Establishes root product ownership and a standalone engine library. |

## Open Questions

There are no open package-boundary questions. See
[questions/open.md](questions/open.md) for the canonical log.

## Migration Tracking

Implementation order and evidence are tracked in the
[package-boundary migration plan](../docs/context-graph/package-boundary-migration-plan.md).
