---
id: SPEC-CHANGE-0004
title: Initialize the Context Runtime System Contract
kind: spec-change
change_status: accepted
spec_id: SPEC-SYSTEM
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

# SPEC-CHANGE-0004: Initialize the Context Runtime System Contract

## Intent

Establish the target dependency path, ownership model, trust boundaries, error
taxonomy, destructive-intent flow, and migration posture for Context Engine and
the Potpie host.

## Provenance Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0001
> decision [active]: decision:ADR-0002
> decision [active]: decision:ADR-0003
> decision [active]: decision:ADR-0004
> decision [active]: decision:ADR-0005
> decision [active]: decision:ADR-0006

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| add | — | SYS-001 | Establish the canonical hosted call path. |
| add | — | SYS-002 | Exclude Potpie host concerns from Context Engine. |
| add | — | SYS-003 | Require explicit host composition. |
| add | — | SYS-004 | Assign Potpie resource-management ownership. |
| add | — | SYS-005 | Assign daemon process-boundary ownership. |
| add | — | SYS-006 | Assign CLI presentation ownership. |
| add | — | SYS-007 | Preserve typed failure categories. |
| add | — | SYS-008 | Place authentication and authorization outside the engine. |
| add | — | SYS-009 | Separate destructive intent and security steps. |
| add | — | SYS-010 | Prevent engine retargeting. |
| add | — | SYS-011 | Keep parsing behavior outside this revision. |
| add | — | SYS-012 | Require an explicit current implementation gap. |
| add | — | SYS-013 | Assign process creation and observation to the controller. |
| add | — | SYS-014 | Separate pre-endpoint controller use from live client use. |
| add | — | SYS-015 | Keep public extensions and manifests outside this revision. |
| add | — | SYS-016 | Keep external-host protocols outside this revision. |
| add | — | SYS-017 | Separate target contracts from conformance claims. |
| add | — | SYS-018 | Define default host ownership of supplied dependencies. |
| add | — | SYS-019 | Require direct handler-to-engine invocation through a lease. |
| add | — | SYS-020 | Keep CLI destructive intent untrusted until validation. |
| add | — | SYS-021 | Use the typed client for post-endpoint live operations. |
| add | — | SYS-022 | Prevent delegation of domain dispatch to Resource Manager. |
| add | — | SYS-023 | Permit controller-selected OS-supervisor delegation. |

## Semantic Diff

This initial system contract replaces no accepted behavior. It establishes the
target architecture against which later implementation commits are reviewed.

## Compatibility, Security, And Failure Impact

The contract permits temporary compatibility mechanisms while requiring one
eventual architecture. It establishes an authenticated and authorized host path,
end-to-end destructive intent, safe discovery semantics, and typed failure
preservation.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| SPEC-CONTEXT-ENGINE | Define domain ownership and explicit construction. | — | agent:codex |
| SPEC-POTPIE-RESOURCE-MANAGER | Define selection, resources, lifecycle, and authorization scope. | — | agent:codex |
| SPEC-DAEMON | Define the typed process boundary and readiness model. | — | agent:codex |
| SPEC-CLI | Define human and machine presentation boundaries. | — | agent:codex |
| Existing implementation | — | Migration occurs in later commits and remains unclaimed. | agent:codex |
| Existing architecture documents | — | They remain implementation snapshots outside this commit. | agent:codex |

## Conformance Invalidation

None. No earlier system-contract conformance record exists.

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
