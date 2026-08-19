---
id: SPEC-CHANGE-0007
title: Initialize the Potpie Daemon Contract
kind: spec-change
change_status: accepted
spec_id: SPEC-DAEMON
from_revision: 0
from_ref: null
to_revision: 1
change_type: normative
initiated_by: user:dsantra
authored_by:
  - agent:codex
accepted_by:
  - user:dsantra
accepted_at: 2026-08-20T02:52:20+05:30
---

# SPEC-CHANGE-0007: Initialize the Potpie Daemon Contract

## Intent

Define one typed daemon process boundary with safe identity, discovery,
readiness, lifecycle, authentication, concurrency, error, and observability
semantics.

## Provenance Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0005

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| add | — | DAEMON-001 | Require one active daemon architecture. |
| add | — | DAEMON-002 | Require one foreground runtime process. |
| add | — | DAEMON-003 | Host one logical Resource Manager. |
| add | — | DAEMON-004 | Expose finite typed operations. |
| add | — | DAEMON-005 | Prohibit reflection and class-coupled wire formats. |
| add | — | DAEMON-006 | Acquire an authorized lease before engine access. |
| add | — | DAEMON-007 | Generate one per-boot identity. |
| add | — | DAEMON-008 | Hold a lifetime ownership lock. |
| add | — | DAEMON-009 | Publish one atomic discovery record. |
| add | — | DAEMON-010 | Separate discovery from readiness. |
| add | — | DAEMON-011 | Require a live authenticated readiness handshake. |
| add | — | DAEMON-012 | Treat PID as diagnostic only. |
| add | — | DAEMON-013 | Fail safely on stale discovery. |
| add | — | DAEMON-014 | Distinguish lifecycle states. |
| add | — | DAEMON-015 | Remove owned discovery during clean shutdown. |
| add | — | DAEMON-016 | Authenticate before handler dispatch. |
| add | — | DAEMON-017 | Prevent discovery data from authorizing requests. |
| add | — | DAEMON-018 | Remove process-global request serialization. |
| add | — | DAEMON-019 | Coordinate by operation, context, and resource. |
| add | — | DAEMON-020 | Permit unrelated safe concurrency. |
| add | — | DAEMON-021 | Prevent unsafe automatic mutation replay. |
| add | — | DAEMON-022 | Preserve structured failure categories. |
| add | — | DAEMON-023 | Exclude terminal presentation. |
| add | — | DAEMON-024 | Redact sensitive logs by default. |
| add | — | DAEMON-025 | Exclude domain semantics. |
| add | — | DAEMON-026 | Validate destructive intent before routing. |
| add | — | DAEMON-027 | Register one explicit handler per operation. |
| add | — | DAEMON-028 | Require direct handler-to-engine invocation. |
| add | — | DAEMON-029 | Exclude ad hoc engine construction from handlers and transport. |
| add | — | DAEMON-030 | Assign external process creation and observation. |
| add | — | DAEMON-031 | Prevent controller observations from claiming readiness. |
| add | — | DAEMON-032 | Define the exact lifecycle transition graph. |
| add | — | DAEMON-033 | Prohibit returning an existing boot to ready. |
| add | — | DAEMON-034 | Require a new identity after restart. |
| add | — | DAEMON-035 | Clean ownership during controlled failure shutdown. |
| add | — | DAEMON-036 | Require removal of the non-selected runtime. |
| add | — | DAEMON-037 | Require removal of reflection and class-coupled protocol paths. |
| add | — | DAEMON-038 | Require removal of duplicate discovery formats. |
| add | — | DAEMON-039 | Give compatibility adapters retirement criteria. |
| add | — | DAEMON-040 | Match readiness to per-boot identity. |
| add | — | DAEMON-041 | Establish protocol compatibility in readiness. |
| add | — | DAEMON-042 | Require ready lifecycle state in readiness. |
| add | — | DAEMON-043 | Release a lease after every handler outcome. |
| add | — | DAEMON-044 | Keep the runtime foreground and non-daemonizing. |
| add | — | DAEMON-045 | Keep readiness and runtime control outside the context lease path. |
| add | — | DAEMON-046 | Separate PID diagnostics from daemon identity. |
| add | — | DAEMON-047 | Prohibit Resource Manager or lease domain dispatch. |
| add | — | DAEMON-048 | Exclude ad hoc engine construction from context handlers. |
| add | — | DAEMON-049 | Prevent compatibility adapters from defining a second architecture. |
| add | — | DAEMON-050 | Keep failed state until discovery and lock ownership are released. |
| add | — | DAEMON-051 | Type controller process failures as ResourceLifecycleError. |

## Semantic Diff

This initial contract defines the converged daemon target. It does not declare
either current daemon stack conforming and does not freeze deferred wire
details.

## Compatibility, Security, And Failure Impact

The target replaces dynamic reflection with explicit handlers, authenticates
live readiness, defines lifecycle transitions, distinguishes failure
categories, prevents unsafe mutation replay, and scopes concurrency. Later
commits define compatibility sequencing while the deletion end state remains
fixed.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| SPEC-CLI | Use only the typed client and preserve presentation outside daemon. | — | agent:codex |
| SPEC-POTPIE-RESOURCE-MANAGER | — | Its authorized-lease and resource-lifecycle contract already supplies the inward boundary. | agent:codex |
| Active reflection daemon | Remove reflection endpoints, clients, generic forwarding, and Python-identity wire codecs during implementation migration. | — | agent:codex |
| Candidate runtime stack | Select reusable implementation pieces, then remove the non-selected daemon runtime as a supported or dormant architecture. | — | agent:codex |
| Existing daemon conformance | — | No conformance record exists. | agent:codex |

## Conformance Invalidation

None. This is the initial daemon contract.

## Validation

```text
Structural: passed
Semantic review: passed
Authority/provenance review: passed
Dependency/consistency review: passed
Historical mutation: not applicable for revision 1
Fresh-agent reconstruction: passed
Thermo-nuclear boundary review: passed
Current implementation characterization: passed (11 tests; not target conformance)
Implementation conformance: unclaimed
Verification conformance: unverified
```

## Acceptance

Accepted by `user:dsantra` at `2026-08-20T02:52:20+05:30`. This acceptance
binds revision 1 as the target contract and does not claim implementation or
verification conformance.
