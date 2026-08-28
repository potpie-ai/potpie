---
id: ADR-0001
title: Use Git-Based Living Specifications With Explicit Acceptance
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-20T02:52:20+05:30
---

# ADR-0001: Use Git-Based Living Specifications With Explicit Acceptance

## Context

The Context Engine, Potpie Resource Manager, daemon, and CLI boundary migration
needs a contract that can become binding before the implementation conforms.
Informal architecture documentation cannot distinguish intended behavior from a
current implementation snapshot, and passing tests cannot show who authorized a
behavioral promise.

## Decision

Markdown under `spec/` is the canonical behavioral contract. Contracts use
integer revisions, stable behavior identifiers, typed provenance, explicit
change records, and named acceptance authority.

Contract maturity, behavior lifecycle, implementation claims, verification
results, and derived freshness remain separate. ADRs explain decisions,
implementation contains code, and conformance records preserve pinned claims
and evidence.

`user:dsantra` is the acceptance authority for the initial revision-1
contract set. Ownership by `team:potpie`, authorship by `agent:codex`, code,
tests, or tool output do not independently confer acceptance.

## Authority And Sources

> authority [active]: user:dsantra

## Consequences

- Accepted contract text has an immutable revision identity in Git.
- Later edits create a new revision and change record, including editorial
  edits.
- Target behavior can be accepted while implementation remains unclaimed and
  verification remains unverified.
- Review includes semantic, provenance, consistency, and historical reasoning
  in addition to deterministic validation.
- Revision allocation, matching change records, immutable pinned conformance,
  stable behavior identifiers, and acceptance records remain independently
  reviewable.
- The repository carries more process artifacts in exchange for explicit
  authority and reconstructable lineage.

## Alternatives Considered

### Keep architecture documentation informal

This does not provide stable behavior identifiers, explicit authority, or a
reliable distinction between current and target architecture.

### Treat merged code or passing tests as the contract

Code and tests show implementation and evidence at selected refs, but they do
not establish who authorized the intended behavior.

### Store implementation and verification on contract metadata

This combines independent state axes and causes evidence to become ambiguous
when either the contract or implementation changes.

### Maintain a mutable latest specification

This weakens historical reconstruction because a revision number and immutable
repository ref no longer identify exact accepted text.

## Affected Behavior IDs

The canonical affected-behavior view is derived from active
`> decision:ADR-0001` edges. It is not duplicated here, so later behavior
splits cannot leave a stale reverse list.

## Follow-Up Change Records

- `SPEC-CHANGE-0001`
