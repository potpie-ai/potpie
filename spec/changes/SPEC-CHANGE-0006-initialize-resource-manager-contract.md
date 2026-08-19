---
id: SPEC-CHANGE-0006
title: Initialize the Potpie Resource Manager Contract
kind: spec-change
change_status: accepted
spec_id: SPEC-POTPIE-RESOURCE-MANAGER
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

# SPEC-CHANGE-0006: Initialize the Potpie Resource Manager Contract

## Intent

Define Potpie's focused ownership of context resolution, authorization scope,
resource composition, context-bound engine construction, lifecycle, and
coordination.

## Provenance Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0003
> decision [active]: decision:ADR-0004
> decision [active]: decision:ADR-0005

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| add | — | RM-001 | Resolve one logical context identity. |
| add | — | RM-002 | Separate selection from presentation. |
| add | — | RM-003 | Establish authorized context scope. |
| add | — | RM-004 | Preserve distinct selection-failure variants. |
| add | — | RM-005 | Validate destructive intent end to end. |
| add | — | RM-006 | Own Potpie-managed resource lifecycle. |
| add | — | RM-007 | Construct explicit engine dependencies. |
| add | — | RM-008 | Return a lease containing one context-bound engine. |
| add | — | RM-009 | Prevent engine retargeting. |
| add | — | RM-010 | Constrain reuse by identity and configuration. |
| add | — | RM-011 | Own idempotent release of Potpie-managed resources. |
| add | — | RM-012 | Share composition policy across Potpie hosting modes. |
| add | — | RM-013 | Coordinate resource and lease conflicts at narrow scope. |
| add | — | RM-014 | Exclude a process-global operation lock. |
| add | — | RM-015 | Return only a lease or canonical provider failure. |
| add | — | RM-016 | Keep provider failure categories distinct. |
| add | — | RM-017 | Exclude domain semantics. |
| add | — | RM-018 | Exclude terminal presentation. |
| add | — | RM-019 | Type authorization denial. |
| add | — | RM-020 | Bind lease identity and authorization scope. |
| add | — | RM-021 | Make lease resource ownership explicit. |
| add | — | RM-022 | Prohibit engine dispatch and generic results. |
| add | — | RM-023 | Prohibit generic service lookup and mirroring. |
| add | — | RM-024 | Exclude daemon transport. |
| add | — | RM-025 | Attempt cleanup of resources opened by failed acquisition. |
| add | — | RM-026 | Order terminal engine and host-resource cleanup. |
| add | — | RM-027 | Retain cleanup ownership for borrowed resources. |
| add | — | RM-028 | Exclude credentials and payloads from logs and errors. |
| add | — | RM-029 | Drop Resource Manager cleanup ownership after transfer. |
| add | — | RM-030 | Expose one idempotent lease-release capability. |
| add | — | RM-031 | Continue host-resource release after engine cleanup failure. |
| add | — | RM-032 | Preserve operation and lease-release outcomes together. |
| add | — | RM-033 | Type destructive-intent rejection as AuthorizationError. |
| add | — | RM-034 | Preserve acquisition and cleanup failure detail. |
| add | — | RM-035 | Forbid lease issuance after destructive-intent rejection. |

## Semantic Diff

This initial contract introduces the logical Resource Manager boundary. It does
not select a concrete implementation class or claim that current setup and host
wiring conform.

## Compatibility, Security, And Failure Impact

The contract centralizes authorization scope and resource lifecycle while
returning a narrow lease rather than dispatching operations. It requires
destructive-intent validation, explicit cleanup ownership, and distinct
provider failures without freezing implementation representation.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| SPEC-DAEMON | Acquire authorized context leases from this boundary while keeping explicit operation dispatch in typed daemon handlers. | — | agent:codex |
| SPEC-CLI | Supply selectors and untrusted intent without owning final resolution or authorization. | — | agent:codex |
| SPEC-CONTEXT-ENGINE | — | Its explicit host-composition contract already supplies the inward boundary. | agent:codex |
| Existing setup and host wiring | — | Implementation migration occurs later. | agent:codex |
| Conformance records | — | No existing record is invalidated. | agent:codex |

## Conformance Invalidation

None. This is the initial Resource Manager contract.

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
