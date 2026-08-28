---
id: SPEC-CHANGE-0008
title: Initialize the Potpie CLI Contract
kind: spec-change
change_status: accepted
spec_id: SPEC-CLI
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

# SPEC-CHANGE-0008: Initialize the Potpie CLI Contract

## Intent

Define the typed-client boundary and distinct human and machine presentation
contracts, including end-to-end destructive intent and standard-stream
discipline.

## Provenance Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0004
> decision [active]: decision:ADR-0005
> decision [active]: decision:ADR-0006

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| add | — | CLI-001 | Route hosted domain commands through the typed client. |
| add | — | CLI-002 | Prohibit dynamic remote methods and reflection RPC. |
| add | — | CLI-003 | Capture selectors while delegating resolution. |
| add | — | CLI-004 | Use the external controller for process creation and observation. |
| add | — | CLI-005 | Exclude direct engine construction. |
| add | — | CLI-006 | Exclude domain semantics. |
| add | — | CLI-007 | Separate credential acquisition from authorization. |
| add | — | CLI-008 | Define permitted human presentation. |
| add | — | CLI-009 | Require destructive human confirmation. |
| add | — | CLI-010 | Create an untrusted typed destructive-intent assertion. |
| add | — | CLI-011 | Preserve cancellation as a distinct category. |
| add | — | CLI-012 | Prevent false cancellation claims. |
| add | — | CLI-013 | Derive human presentation from typed errors. |
| add | — | CLI-014 | Prohibit machine prompts. |
| add | — | CLI-015 | Emit one machine JSON value. |
| add | — | CLI-016 | Protect machine standard output. |
| add | — | CLI-017 | Prevent non-interactive blocking. |
| add | — | CLI-018 | Reject missing destructive intent before dispatch. |
| add | — | CLI-019 | Reserve exit zero for success. |
| add | — | CLI-020 | Map typed failure categories. |
| add | — | CLI-021 | Defer exact machine JSON fields. |
| add | — | CLI-022 | Exclude server symbols and HostShell dependencies. |
| add | — | CLI-023 | Use the typed client after an endpoint exists. |
| add | — | CLI-024 | Defer the exact numeric exit-code mapping. |
| add | — | CLI-025 | Exclude credentials from output and diagnostics. |
| add | — | CLI-026 | Prohibit exception-message error classification. |
| add | — | CLI-027 | Prevent replay of unknown-outcome mutations. |
| add | — | CLI-028 | Keep machine diagnostics on stderr or suppress them. |
| add | — | CLI-029 | Redact CLI telemetry by default. |
| add | — | CLI-030 | Keep pre-dispatch cancellation local. |
| add | — | CLI-031 | Reserve nonzero exits for failures. |
| add | — | CLI-032 | Select exactly one presentation mode. |
| add | — | CLI-033 | Type missing pre-dispatch destructive intent. |

## Semantic Diff

This initial contract defines the target CLI boundary without changing command
names or claiming current implementation conformance.

## Compatibility, Security, And Failure Impact

The target preserves a shared operation surface while making machine
non-interactivity, standard streams, cancellation, typed errors, authorization,
and destructive intent explicit. Exact envelope and numeric mapping remain
deferred.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| SPEC-DAEMON | — | Its typed operation and error contract supplies the inward boundary. | agent:codex |
| Existing CLI commands | — | Implementation migration occurs later. | agent:codex |
| Existing JSON consumers | — | Exact compatibility receives a later accepted contract. | agent:codex |
| Existing tests | — | No implementation or verification claim is created. | agent:codex |

## Conformance Invalidation

None. This is the initial CLI contract.

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
