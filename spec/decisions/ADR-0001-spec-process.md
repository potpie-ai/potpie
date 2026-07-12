---
id: ADR-0001
title: Adopt a Living Spec Process
kind: decision
status: accepted
version: 1.0.0
owners:
  - Potpie Engineering
depends_on: []
related_specs:
  - SPEC-INDEX
  - SPEC-PROCESS
related_decisions: []
affects:
  - SPEC-INDEX
  - SPEC-PROCESS
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
    - SPEC-PROCESS
  drift_status: unverified
---

# ADR-0001: Adopt a Living Spec Process

## Context

Potpie has detailed architecture and workflow documents, but no single
Git-controlled structure that distinguishes proposed behavior, accepted
behavior, implementation claims, verification evidence, decisions, and open
questions.

The package-boundary migration spans two distributions, multiple executables,
CLI and MCP contracts, RPC, configuration, persistent data, and documentation.
It needs stable requirements that can be referenced across a sequence of small
commits.

## Decision

Adopt the `spec/` tree governed by [SPEC-PROCESS](../process.md).

- Normative behavior lives in module or system specs.
- Every normative requirement uses a stable behavior ID.
- ADRs explain architectural decisions but do not replace behavior specs.
- Open questions and verification evidence have dedicated files.
- Spec metadata is the machine-readable lifecycle and relationship state.

## Consequences

- Implementations and migration plans can reference stable behavior IDs.
- Proposed behavior is clearly separated from current code behavior.
- Accepted specs require explicit failure modes, examples, security posture, and
  testable acceptance criteria.
- Maintaining the spec tree becomes part of completing normative changes.

## Alternatives Considered

### Keep architecture prose only

Rejected because prose does not encode lifecycle, verification, open questions,
or stable traceability.

### Store the migration only in an issue or pull request

Rejected because the architectural contract must remain available in the
repository after the migration is merged.

### Treat tests as the complete specification

Rejected because tests do not adequately describe ownership, non-goals,
security, compatibility, or rationale.

## Affected Specs

- [SPEC-INDEX](../index.md)
- [SPEC-PROCESS](../process.md)
- [SPEC-PACKAGE-BOUNDARY](../modules/package-boundary.md)

## Follow-Up Spec Changes

All future module specs use the metadata, behavior-ID, backlink, acceptance, and
verification rules established here.
