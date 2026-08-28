---
id: ADR-0003
title: Use Explicit Host Composition And Immutable Context Scoping
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-20T02:52:20+05:30
---

# ADR-0003: Use Explicit Host Composition And Immutable Context Scoping

## Context

Implicit environment discovery, process-global selection, and per-call context
selectors make it hard to know which resources and identity an engine uses.
They also prevent several isolated engines from coexisting safely in one
process.

## Decision

A compatible host supplies Context Engine's context identity, dependencies, and
resource-ownership modes explicitly. A supplied resource remains borrowed
unless a typed construction contract transfers ownership. One engine instance
remains bound to one logical context identity for its complete lifetime. A
different identity uses a different instance.

Engine-owned adapters can remain in the package where useful, but a host
selects and composes them explicitly. Context Engine does not silently discover
or provision host-managed resources.

## Authority And Sources

> authority [active]: user:dsantra
> observation [active]: code:potpie/context-engine/src/potpie_context_engine/bootstrap/host_wiring.py@a341978880b9d4c1b403831931279ccedf6184ae

## Consequences

- Engine behavior is reproducible from construction input.
- Multiple context identities can be hosted safely in one process.
- Potpie selection occurs before engine construction rather than inside an
  operation.
- Resource and engine reuse considers identity and compatible configuration.
- Context Engine never closes borrowed host dependencies.
- The Potpie lease retains host-resource ownership and orders terminal cleanup.
- Exact constructor and lifecycle method shapes remain open.

## Alternatives Considered

### Let Context Engine discover environment and active pot state

This makes independent library use host-specific and hides resource ownership.

### Permit a mutable current context on one engine

This creates cross-request isolation risks and makes cached service state
ambiguous.

### Use one process-global engine singleton

This prevents safe multi-context composition and couples engine lifetime to one
host process.

## Affected Behavior IDs

The canonical affected-behavior view is derived from active
`> decision:ADR-0003` edges. It is not duplicated here, so later behavior
splits cannot leave a stale reverse list.

## Follow-Up Change Records

- `SPEC-CHANGE-0003`
- `SPEC-CHANGE-0004`
- `SPEC-CHANGE-0005`
- `SPEC-CHANGE-0006`
