---
id: SPEC-CHANGE-0009
title: Correct Resource Manager Authentication Outcomes
kind: spec-change
change_status: accepted
spec_id: SPEC-POTPIE-RESOURCE-MANAGER
from_revision: 1
from_ref: 047cbe067c9c726e7e14f066675453372d8a8406
to_revision: 2
change_type: failure-mode
initiated_by: user:dsantra
authored_by:
  - agent:codex
accepted_by:
  - user:dsantra
accepted_at: 2026-08-21T14:55:35+05:30
---

# SPEC-CHANGE-0009: Correct Resource Manager Authentication Outcomes

## Intent

Correct the exhaustive Resource Manager acquisition-outcome list so it
includes the authentication failure already selected by ADR-0009 and required
to remain structurally distinct by the system error taxonomy.

## Provenance Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0004
> decision [active]: decision:ADR-0005
> decision [active]: decision:ADR-0009
> decision [active]: decision:ADR-0010
> observation [active]: code:potpie/runtime/resource_manager.py@097bae17bfb2696b18293ed33585ea4a50f4337d
> observation [active]: test:tests/unit/test_context_resource_manager.py@097bae17bfb2696b18293ed33585ea4a50f4337d

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| replace | RM-015 | RM-036 | Include AuthenticationError in the exhaustive acquisition outcome. |
| replace | RM-016 | RM-037 | Keep all four Resource Manager failure categories structurally distinct. |
| add | — | RM-038 | Require AuthenticationError when caller authentication fails. |
| add | — | RM-039 | Stop acquisition before protected downstream stages after authentication failure. |

## Semantic Diff

Revision 2 adds `AuthenticationError` to the allowed Resource Manager
acquisition outcomes and makes authentication failure terminate acquisition
before authorization, composition, construction, or lease issuance. All other
Resource Manager behaviors are carried forward unchanged.

This is a failure-mode correction rather than a clarification because revision
1's exhaustive list disallowed an authentication outcome that ADR-0009's
selected acquisition sequence can produce.

## Compatibility, Security, And Failure Impact

The correction preserves the implemented typed category and prevents callers
from treating failed authentication as authorization denial or an internal
resource failure. It adds no credential format, remote authentication scheme,
transport, retry, or presentation behavior.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| ADR-0009 | ADR-0010 supersedes only its incomplete acquisition-outcome enumeration. | The immutable accepted ADR is not edited. | agent:codex |
| SPEC-SYSTEM / SYS-007 | — | The system taxonomy already includes structurally distinct AuthenticationError. | agent:codex |
| SPEC-DAEMON / DAEMON-016, DAEMON-022 | — | The daemon already authenticates before protected dispatch and preserves AuthenticationError. | agent:codex |
| SPEC-CLI / CLI-007, CLI-013, CLI-020 | — | Credential acquisition remains separate and typed categories continue to drive presentation. | agent:codex |
| SPEC-CONTEXT-ENGINE | — | Caller authentication remains outside Context Engine. | agent:codex |
| Current implementation | Add permanent tests for RM-038 and RM-039 during the corrective implementation series. | The implementation already returns AuthenticationError before composition. | agent:codex |
| Conformance records | — | No conformance records exist. | agent:codex |

## Conformance Invalidation

None. No existing conformance record covers `RM-015` or `RM-016`.

## Validation

```text
Structural: passed with 0 warnings
Semantic: reviewed; failure category and short-circuit are explicit
Provenance: reviewed; user authority and accepted decision lineage retained
Historical mutation: reviewed; RM-015 and RM-016 retire without ID reuse
Dependency/consistency: reviewed; downstream accepted contracts already distinguish AuthenticationError
Fresh-agent reconstruction: passed; outcome and short-circuit are independently reconstructable
Independent conformance state: unverified; no conformance record exists
```

## Acceptance

Accepted by `user:dsantra` at `2026-08-21T14:55:35+05:30`. This acceptance
binds Resource Manager revision 2 when committed with ADR-0010 and this change
record. It creates no implementation or verification claim.
