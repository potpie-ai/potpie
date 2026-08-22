---
id: SPEC-CHANGE-0003
title: Initialize the Context Runtime Product Contract
kind: spec-change
change_status: accepted
spec_id: SPEC-PRODUCT
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

# SPEC-CHANGE-0003: Initialize the Context Runtime Product Contract

## Intent

Define the product outcomes for an independently usable Context Engine, the
Potpie-managed host experience, and distinct human and machine CLI
presentations.

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
| add | — | PROD-001 | Establish independent Context Engine usability. |
| add | — | PROD-002 | Establish Potpie's managed host role. |
| add | — | PROD-003 | Separate human and automation presentation. |
| add | — | PROD-004 | Preserve domain semantics across hosting modes. |
| add | — | PROD-005 | Require accepted authorization before intentional observable behavior change. |
| add | — | PROD-006 | Exclude new domain features from commit 1. |
| add | — | PROD-007 | Separate acceptance from implementation conformance. |

## Semantic Diff

This initial product contract establishes the intended library and product-host
outcomes. It adds no user-facing domain capability and does not claim that the
current implementation satisfies the target.

## Compatibility, Security, And Failure Impact

The product contract prevents intentional observable behavior changes from
bypassing contract review while allowing internal package and hosting
boundaries to migrate. Security and failure details are defined by the system
and module contracts.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| SPEC-SYSTEM | Define the dependency path that realizes these outcomes. | — | agent:codex |
| Module contracts | Assign product responsibilities to bounded modules. | — | agent:codex |
| Existing implementation | — | Implementation migration occurs in later commits. | agent:codex |
| Existing user documentation | — | Existing documents remain outside this commit's scope. | agent:codex |

## Conformance Invalidation

None. No conformance record covers this initial contract.

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
