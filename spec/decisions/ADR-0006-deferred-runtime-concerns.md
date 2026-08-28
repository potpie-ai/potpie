---
id: ADR-0006
title: Defer Parsing Extensions External Hosting And Exact Protocol Details
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-20T02:52:20+05:30
---

# ADR-0006: Defer Parsing, Extensions, External Hosting, And Exact Protocol Details

## Context

The first migration commit needs a stable ownership boundary. Designing parsing,
an extension system, external-host deployment, exact public methods, and every
wire detail at the same time would make acceptance depend on unrelated choices
and preserve speculative abstractions.

## Decision

This revision deliberately defers:

- Parsing redesign.
- Public plugin, extension-registration, and manifest contracts.
- External-host transport and deployment protocols.
- Exact Context Engine façade method grouping and sync/async exposure.
- Exact authorized-context-lease Python representation.
- Exact daemon-controller API and supervisor integration.
- Daemon transport selection, wire fields, cancellation, timeout propagation,
  idempotency keys, and operation concurrency matrix.
- Exact machine JSON fields and complete numeric nonzero exit-code mapping.
- Compatibility and removal sequencing for Context Core, HostShell, the active
  reflection daemon, and the incomplete candidate runtime.

The ownership, identity, typed-boundary, discovery, readiness, concurrency,
presentation, destructive-intent, and deletion end-state invariants in the
accepted contracts bind without those details. The order of deletion remains
deferred; the required final removal does not.

## Authority And Sources

> authority [active]: user:dsantra

## Consequences

- Commit 1 can bind the durable architecture without speculative APIs.
- Later implementation commits stop before choices required by a deferred
  question and accept the necessary contract revision first.
- Context Engine remains compatible with future hosts without publishing an
  external-host protocol now.
- Existing extension and runtime code creates no public promise merely because
  it exists.
- A later concurrency decision selects typed safety classes and conflict scope
  before replacement coordination is implemented.

## Alternatives Considered

### Design every API and protocol in commit 1

This would mix boundary acceptance with details that need implementation
spikes, compatibility analysis, and separate user decisions.

### Treat existing extension and runtime code as accepted

Existing code is observation rather than authority and includes competing,
incomplete approaches.

### Remove deferred topics without recording them

That would make future implementers unable to distinguish deliberate deferral
from accidental omission.

## Affected Behavior IDs

The canonical affected-behavior view is derived from active
`> decision:ADR-0006` edges. It is not duplicated here, so later behavior
splits cannot leave a stale reverse list.

## Follow-Up Change Records

- `SPEC-CHANGE-0003`
- `SPEC-CHANGE-0004`
- `SPEC-CHANGE-0005`
- `SPEC-CHANGE-0008`
