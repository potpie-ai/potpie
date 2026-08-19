---
id: SPEC-CHANGE-0002
title: Initialize the Context Runtime Glossary
kind: spec-change
change_status: accepted
spec_id: SPEC-GLOSSARY
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

# SPEC-CHANGE-0002: Initialize the Context Runtime Glossary

## Intent

Define canonical terms for the Context Engine, Potpie host boundary, daemon
process boundary, CLI presentation boundary, identity model, failure taxonomy,
and conformance model.

## Provenance Sources

> authority [active]: user:dsantra

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| add | — | GLOSS-001 | Make accepted definitions canonical. |
| add | — | GLOSS-002 | Separate context identity from product and process selectors. |
| add | — | GLOSS-003 | Separate confirmation from security authority. |
| add | — | GLOSS-004 | Separate discovery from readiness. |
| add | — | GLOSS-005 | Define independent library usability without implicit provisioning. |
| add | — | GLOSS-006 | Separate destructive assertion from security and execution. |
| add | — | GLOSS-007 | Define the canonical ten-category error taxonomy. |
| add | — | GLOSS-008 | Separate context identity from product and process identifiers. |
| add | — | GLOSS-009 | Separate confirmation from authentication. |
| add | — | GLOSS-010 | Separate confirmation from authorization. |
| add | — | GLOSS-011 | Prevent discovery from proving readiness. |
| add | — | GLOSS-012 | Separate engine usability from host provisioning. |

## Semantic Diff

This initial revision introduces the shared vocabulary required by all product,
system, and module contracts. There is no earlier glossary behavior to mutate.

## Compatibility, Security, And Failure Impact

The identity and security definitions prevent downstream contracts from
collapsing confirmation, authentication, authorization, context selection, and
process identity into one implicit mechanism.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| SPEC-PRODUCT | Use the accepted host, engine, human-mode, and machine-mode terms. | — | agent:codex |
| SPEC-SYSTEM | Use canonical identity, trust-boundary, and failure terms. | — | agent:codex |
| Module contracts | Use the canonical ownership and lifetime terms. | — | agent:codex |
| Existing implementation | — | Definitions create no implementation claim. | agent:codex |

## Conformance Invalidation

None. This is the initial glossary revision.

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
