---
id: SPEC-CHANGE-0012
title: Restrict Daemon Signal Fallback
kind: spec-change
change_status: accepted
spec_id: SPEC-DAEMON
from_revision: 1
from_ref: 047cbe067c9c726e7e14f066675453372d8a8406
to_revision: 2
change_type: security
initiated_by: user:dsantra
authored_by:
  - agent:codex
accepted_by:
  - user:dsantra
accepted_at: 2026-08-27T11:09:34+05:30
---

# SPEC-CHANGE-0012: Restrict Daemon Signal Fallback

## Intent

Prevent stale or reused daemon PID records from authorizing operating-system
signals to an unrelated process while preserving authenticated typed shutdown
and bounded termination of a directly owned child.

## Provenance Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0005
> decision [active]: decision:ADR-0009
> decision [active]: decision:ADR-0012
> observation [active]: code:potpie/daemon/lifecycle.py@942b704045469d7b61efdd92a6e58fb3c6ef192f
> observation [active]: code:potpie/runtime/controller.py@942b704045469d7b61efdd92a6e58fb3c6ef192f
> rationale [active]: issue:potpie-ai/potpie#1057-discussion-r3836380222

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| add | — | DAEMON-052 | Permit OS termination signals only for a directly owned child process. |
| add | — | DAEMON-053 | Fail attached shutdown without signalling or claiming stopped. |
| add | — | DAEMON-054 | Serialize canonical PID, discovery, and credential publication with runtime ownership. |
| add | — | DAEMON-055 | Serialize canonical PID, discovery, credential, and owned-UDS removal with runtime ownership. |
| add | — | DAEMON-056 | Require exact expected PID and per-boot instance matching for identity-bound cleanup. |

## Semantic Diff

Revision 2 removes operating-system signal fallback for daemon processes
reattached from runtime records. Such processes retain authenticated typed
shutdown, but authentication failure, shutdown failure, and shutdown timeout
return a typed non-success outcome without `SIGTERM` or `SIGKILL`.

Direct children created by the current controller retain bounded termination
for failed readiness and incomplete shutdown. Cleanup after an attached stop is
limited to records proven to belong to the exact expected PID and per-boot
instance identity. All prior daemon behaviors remain active and unchanged.

## Compatibility, Security, And Failure Impact

The change closes a local same-user process-termination vulnerability caused by
PID reuse. Healthy cross-invocation typed shutdown remains compatible. Forced
termination of an unresponsive daemon by a later CLI invocation is intentionally
removed until Potpie has a stable process-incarnation identity contract.

Unsafe stop refusal becomes `ResourceLifecycleError` with a nonzero CLI exit.
It does not report the daemon as stopped and recommends manual recovery.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| ADR-0009 | ADR-0012 supersedes only its unauthenticated OS-termination fallback. | The immutable accepted ADR is not edited. | agent:codex |
| SPEC-SYSTEM / SYS-005, SYS-007, SYS-013 | — | Authenticated instance identity, typed failures, and controller ownership already support the stricter module rule. | agent:codex |
| SPEC-CLI / CLI-004, CLI-023, CLI-031 | — | The CLI already requires controller/client separation, typed graceful stop, and nonzero failure exits. | agent:codex |
| Daemon lifecycle and controller | Track direct-child ownership, reject signals for attached processes, and preserve typed stop failure. | — | agent:codex |
| Daemon discovery lifecycle | Publish credentials and remove exact expected PID and instance records while holding the runtime ownership lock; an attached controller leaves successful cleanup to the authenticated daemon. | — | agent:codex |
| Daemon tests | Add stale-PID, attached-controller refusal, CLI failure, and healthy typed-stop controls. | — | agent:codex |
| Existing daemon conformance | Publish a successor after implementation and verification. | Revision 1 remains immutable at its pinned ref. | agent:codex |
| Existing CLI conformance | Publish a successor that checks CLI revision 1 against accepted Daemon revision 2 and the new implementation ref. | The CLI contract text remains unchanged. | agent:codex |
| Cross-system conformance | Publish a successor for the changed PR head and implementation identity. | Other module contracts remain unchanged. | agent:codex |

## Conformance Invalidation

On acceptance, the current daemon record derives stale because it pins
`SPEC-DAEMON` revision 1 and an implementation that permits PID-only signal
fallback. The CLI record derives stale because it pins Daemon revision 1 as a
related dependency. The cross-system record also derives stale when the PR head
and implementation identity advance. Existing records remain unchanged at
their pinned Git refs.

## Validation

```text
Structural: passed with 0 warnings
Semantic: reviewed; signal authority, refusal outcome, publication serialization, removal serialization, and identity matching are separate atomic obligations
Provenance: reviewed; user authority and ADR-0005/ADR-0009/ADR-0012 lineage are explicit
Historical mutation: reviewed; revision advances 1 to 2 and DAEMON-052 through DAEMON-056 are unused IDs
Dependency/consistency: reviewed; system and CLI contracts already require compatible identity and failure behavior
Fresh-agent reconstruction: passed after atomicity, direct-dependency, reverse-impact, and derived-index corrections
Independent conformance state: revision 1 currently passed; successor pending implementation and verification
```

## Acceptance

Accepted by `user:dsantra` at `2026-08-27T11:09:34+05:30`. This acceptance
binds Daemon revision 2 when committed with ADR-0012 and this change record. It
creates no implementation or verification claim.
