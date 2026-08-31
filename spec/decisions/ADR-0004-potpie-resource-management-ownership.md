---
id: ADR-0004
title: Assign Product Resource Management To Potpie
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-20T02:52:20+05:30
---

# ADR-0004: Assign Product Resource Management To Potpie

## Context

Current engine-side host wiring and setup orchestration combine pot selection,
credentials, provisioning, installation, daemon lifecycle, and domain services.
Moving all of that behind another universal façade would preserve rather than
remove the ownership problem.

## Decision

Potpie owns a focused logical Resource Manager that resolves selection, applies
authorization, acquires host-managed resources, constructs or reuses a
context-bound engine, and returns an authorized context lease with explicit
ownership and release semantics.

The Resource Manager is defined by responsibility rather than one mandatory
class or package. It does not dispatch Context Engine operations, return domain
or generic product-operation results, expose a service graph, own
context-domain semantics, implement daemon transport, or render presentation.
Typed operation handlers invoke explicit Context Engine operations through the
lease.

## Authority And Sources

> authority [active]: user:dsantra
> observation [active]: code:potpie/context-engine/src/potpie_context_engine/host/shell.py@a341978880b9d4c1b403831931279ccedf6184ae
> observation [active]: code:potpie/context-engine/src/potpie_context_engine/application/services/setup_orchestrator.py@a341978880b9d4c1b403831931279ccedf6184ae

## Consequences

- Potpie has one focused boundary for selection, authorization, and
  machine-resource policy.
- Direct and daemon-hosted Potpie compositions can share resource policy.
- Context Engine receives explicit dependencies rather than product-global
  state.
- Daemon handlers receive a scoped lease rather than a universal runtime.
- Implementation can use several cohesive services instead of one universal
  runtime object.

## Alternatives Considered

### Leave setup and resource management inside Context Engine

This prevents the engine from being independently usable with another host.

### Let the daemon compose Context Engine directly per operation

This couples engine construction to transport and duplicates in-process
composition policy.

### Create a universal Potpie runtime façade

This risks recreating HostShell with a different name and absorbing unrelated
product behavior.

## Affected Behavior IDs

The canonical affected-behavior view is derived from active
`> decision:ADR-0004` edges. It is not duplicated here, so later behavior
splits cannot leave a stale reverse list.

## Follow-Up Change Records

- `SPEC-CHANGE-0003`
- `SPEC-CHANGE-0004`
- `SPEC-CHANGE-0005`
- `SPEC-CHANGE-0006`
- `SPEC-CHANGE-0008`
