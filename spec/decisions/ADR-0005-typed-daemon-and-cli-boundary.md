---
id: ADR-0005
title: Use One Typed Daemon Boundary And CLI-Owned Presentation
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-20T02:52:20+05:30
---

# ADR-0005: Use One Typed Daemon Boundary And CLI-Owned Presentation

## Context

The active daemon mirrors HostShell through reflection, transports Python class
identity, writes overlapping discovery records, and serializes all RPC work
through one lock. A second incomplete runtime stack exists with incompatible
composition and discovery. Human confirmation is also not consistently
validated through the lower boundary.

## Decision

Potpie converges on one daemon subsystem with an external controller and one
foreground runtime. The controller creates and observes the process. The
runtime provides:

- One per-boot instance identity and lifetime ownership lock.
- One atomically published canonical discovery record.
- A live authenticated handshake as readiness truth.
- A finite typed product-operation surface and typed client.
- One explicit typed handler per product operation.
- Handler acquisition of an authorized context lease followed by direct
  invocation of an explicit Context Engine operation.
- Context- and operation-aware coordination without a global request mutex.
- Structured failures and redacted operational observability.

The runtime follows the exact starting, ready, draining, failed, and stopped
transition graph in the daemon contract. Completion removes the non-selected
runtime, reflective endpoints and clients, and duplicate discovery formats.

The CLI owns human and machine presentation. Destructive confirmation becomes
an untrusted typed assertion. After authentication and context resolution, the
Resource Manager validates that assertion and authorization before a handler
invokes the explicit destructive domain command.

## Authority And Sources

> authority [active]: user:dsantra
> observation [active]: code:potpie/daemon/main.py@a341978880b9d4c1b403831931279ccedf6184ae
> observation [active]: code:potpie/daemon/client.py@a341978880b9d4c1b403831931279ccedf6184ae
> observation [active]: code:potpie/daemon/rpc.py@a341978880b9d4c1b403831931279ccedf6184ae
> observation [active]: code:potpie/daemon/runtime/__main__.py@a341978880b9d4c1b403831931279ccedf6184ae

## Consequences

- Adding an internal service no longer automatically exposes remote methods.
- The Resource Manager cannot become a replacement HostShell.
- Process creation and process readiness have different owners.
- PID files and discovery presence no longer masquerade as readiness.
- Independent contexts can progress concurrently.
- The ten canonical error families survive boundary translation.
- Typed client and daemon versions have an explicit compatibility boundary.
- Human strings and terminal behavior remain outside the daemon.
- Exact transport and wire representation remain separately decidable.

## Alternatives Considered

### Retain reflection with a larger allowlist

This remains coupled to internal object graphs and Python deployment identity.

### Preserve both daemon stacks behind a wrapper

This adds a third layer without deleting duplicate lifecycle, discovery, and
composition models.

### Use PID or discovery-file presence as readiness

Both can remain stale after a crash and cannot prove protocol compatibility.

### Keep one global RPC lock

This serializes independent contexts and hides the operation-level safety model.

### Let the daemon render human output

This couples product presentation to the process protocol and weakens machine
output guarantees.

## Affected Behavior IDs

The canonical affected-behavior view is derived from active
`> decision:ADR-0005` edges. It is not duplicated here, so later behavior
splits cannot leave a stale reverse list.

## Follow-Up Change Records

- `SPEC-CHANGE-0003`
- `SPEC-CHANGE-0004`
- `SPEC-CHANGE-0005`
- `SPEC-CHANGE-0006`
- `SPEC-CHANGE-0007`
- `SPEC-CHANGE-0008`
