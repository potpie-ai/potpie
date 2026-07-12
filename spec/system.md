---
id: SPEC-SYSTEM
title: Potpie System Specification
kind: system-spec
status: draft
version: 0.1.0
owners:
  - Potpie Engineering
depends_on:
  - SPEC-PROCESS
  - SPEC-GLOSSARY
  - SPEC-PRODUCT
related_specs:
  - SPEC-PACKAGE-BOUNDARY
related_decisions:
  - ADR-0002
affects:
  - SPEC-PACKAGE-BOUNDARY
open_questions: []
verification:
  code_status: not_implemented
  verified_commit: null
  verified_at: null
  verified_by: null
  behavior_scope: []
  evidence: []
  cross_spec_status: passed
  cross_spec_checked_against:
    - SPEC-GLOSSARY
    - SPEC-PRODUCT
    - SPEC-PACKAGE-BOUNDARY
  drift_status: unverified
---

# Potpie System Specification

## Purpose

This draft is the cross-system entrypoint. The first fully described system
boundary is the proposed
[Potpie / Context Engine Package Boundary](modules/package-boundary.md).

## Ownership And Boundaries

Root Potpie is the product system; the context engine is a reusable subsystem.
Normative ownership is defined in `SPEC-PACKAGE-BOUNDARY`.

## Scope And Non-Goals

This initial draft covers only cross-system concepts necessary to locate the
package-boundary proposal. It does not yet specify every datastore, hosted
service, integration, or deployment topology.

## Terminology

Use [SPEC-GLOSSARY](glossary.md).

## Actors And Permissions

The package-boundary actors are product users, coding agents, product operators,
and engine embedders. Detailed permissions remain within the owning module spec.

## Normative Requirements

Not applicable: no independent system-wide requirements are accepted in this
initial draft; `SPEC-PACKAGE-BOUNDARY` owns the proposed requirements.

## Global Guarantees

Not applicable: global guarantees will be added as module contracts are accepted.

## Cross-Module Interfaces

The initial proposed interface is `PotpieRuntime` to `EngineClient`, defined in
`SPEC-PACKAGE-BOUNDARY`.

## Invariants

Not applicable: proposed package invariants are owned by
`SPEC-PACKAGE-BOUNDARY`.

## Failure Philosophy

Expected failures should be typed and actionable. Product runtime failures must
not be hidden by silently selecting a different execution mode. This statement
is descriptive until the package-boundary proposal is accepted.

## Security And Privacy

Credentials and product telemetry remain product concerns. Engine requests
receive only the identity and capability data required for the operation.

## Observability And Auditability

Product crash and usage telemetry is distinct from generic engine observability.
Detailed guarantees remain in the package-boundary proposal and future
observability specs.

## Compatibility And Migration

The first migration is tracked in the
[package-boundary migration plan](../docs/context-graph/package-boundary-migration-plan.md).

## Examples

- A product CLI command obtains `PotpieRuntime` and calls `runtime.engine`.
- A library embedder creates an in-memory engine without installing root Potpie.

## Acceptance Criteria

This system spec remains draft until system-wide behavior beyond the initial
package boundary has stable behavior IDs and testable acceptance criteria.

## Cross-Spec Consistency

Reviewed against `SPEC-GLOSSARY`, `SPEC-PRODUCT`, `SPEC-PACKAGE-BOUNDARY`, and
`ADR-0002`; no contradiction is known.

## Open Questions

No package-boundary question is open. Future modules may add their own stable
question IDs.

## Rationale

Keeping this file intentionally small prevents the initialization commit from
claiming broad system behavior that has not been designed.

## Implementation Notes

Not applicable: implementation details belong to module specs and migration
plans.
