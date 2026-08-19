---
id: SPEC-CHANGE-0005
title: Initialize the Context Engine Contract
kind: spec-change
change_status: accepted
spec_id: SPEC-CONTEXT-ENGINE
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

# SPEC-CHANGE-0005: Initialize the Context Engine Contract

## Intent

Establish one importable Context Engine distribution and one context-bound
public façade with explicit host composition and a transport-neutral domain
boundary.

## Provenance Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0002
> decision [active]: decision:ADR-0003
> decision [active]: decision:ADR-0004
> decision [active]: decision:ADR-0005
> decision [active]: decision:ADR-0006

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| add | — | CE-001 | Establish the public façade. |
| add | — | CE-002 | Preserve independent host usability. |
| add | — | CE-003 | Place required public behavior in one distribution. |
| add | — | CE-004 | Remove Context Core as a permanent boundary. |
| add | — | CE-005 | Require explicit identity and dependencies. |
| add | — | CE-006 | Bind one identity per instance. |
| add | — | CE-007 | Prevent per-operation identity override. |
| add | — | CE-008 | Use a different instance for a different identity. |
| add | — | CE-009 | Permit isolated coexisting instances. |
| add | — | CE-010 | Make engine-owned cleanup safely repeatable. |
| add | — | CE-011 | Assign context-domain ownership. |
| add | — | CE-012 | Require transport-neutral typed engine outcomes. |
| add | — | CE-013 | Exclude terminal and daemon presentation. |
| add | — | CE-014 | Exclude Potpie product concerns. |
| add | — | CE-015 | Exclude implicit resource discovery and provisioning. |
| add | — | CE-016 | Restrict package adapters to explicitly composed engine-owned ports. |
| add | — | CE-017 | Make destructive domain commands explicit and non-default. |
| add | — | CE-018 | Defer public extensions and manifests. |
| add | — | CE-019 | Preserve parsing behavior. |
| add | — | CE-020 | Prohibit service location and dynamic dispatch. |
| add | — | CE-021 | Keep the façade thin over focused domain modules. |
| add | — | CE-022 | Default supplied dependencies to borrowed ownership. |
| add | — | CE-023 | Protect borrowed dependencies from engine cleanup. |
| add | — | CE-024 | Clean up transferred and engine-created dependencies. |
| add | — | CE-025 | Type post-closure use as EngineLifecycleError. |
| add | — | CE-026 | Exclude Potpie concerns from package adapters. |
| add | — | CE-027 | Require removal rather than renaming of HostShell and host wiring. |
| add | — | CE-028 | Keep public operation types engine-owned. |
| add | — | CE-029 | Classify domain semantic failures. |
| add | — | CE-030 | Classify engine-port dependency failures. |
| add | — | CE-031 | Exclude secrets from engine errors and observability. |
| add | — | CE-032 | Prevent failed construction from yielding a usable engine. |
| add | — | CE-033 | Require explicit host selection of package adapters. |
| add | — | CE-034 | Prohibit default or inferred destructive operations. |

## Semantic Diff

This initial contract defines the target engine boundary. It replaces no
accepted contract and makes no claim that current HostShell or package layout
conforms.

## Compatibility, Security, And Failure Impact

The boundary preserves direct library usability while moving product auth,
selection, installation, provisioning, process, and presentation concerns
outward. It requires explicit dependency ownership and removes HostShell rather
than renaming it. Exact compatibility shims remain deferred.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| SPEC-POTPIE-RESOURCE-MANAGER | Own explicit engine composition and host-resource lifecycle. | — | agent:codex |
| SPEC-DAEMON | Make typed context-domain handlers acquire an authorized context lease before invoking explicit engine operations. | — | agent:codex |
| SPEC-CLI | Avoid direct engine construction in the hosted path. | — | agent:codex |
| Current engine package | — | Implementation migration occurs later. | agent:codex |
| Existing tests and conformance | — | No conformance record is created or invalidated. | agent:codex |

## Conformance Invalidation

None. This is the initial Context Engine contract.

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
