---
id: ADR-0012
title: Restrict Forceful Daemon Termination To Owned Children
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-27T11:09:34+05:30
---

# ADR-0012: Restrict Forceful Daemon Termination To Owned Children

## Context

ADR-0009 selects authenticated typed shutdown for the canonical daemon but
also permits bounded operating-system termination when no live endpoint can be
authenticated. A later CLI invocation reconstructs a process handle from the
numeric PID in `daemon.pid`. If the daemon crashed and the operating system
reused that PID, missing, invalid, or unauthenticated discovery causes the
controller to send `SIGTERM` and potentially `SIGKILL` to an unrelated process.

This fallback conflicts with the accepted daemon requirements that a PID is
diagnostic metadata rather than daemon instance identity and that stale
discovery fails safely. Owner-only runtime files prove control of those files;
they do not prove the identity of the process currently holding a reused PID.

## Decision

Bounded operating-system termination is permitted only through the process
handle of a foreground child created directly by the current controller. A
controller reconstructed by a later CLI invocation uses authenticated typed
shutdown only. It does not send operating-system termination signals when
typed shutdown cannot authenticate, fails, or times out.

An attached stop that cannot complete safely returns a typed
`ResourceLifecycleError`, does not claim that the daemon stopped, and preserves
runtime records unless cleanup is guarded by the exact expected daemon
instance identity and PID. Credential publication and all cleanup are
serialized through the runtime ownership lock so a concurrent stop cannot
erase an in-progress replacement boot. Manual recovery or a later independently designed
process-incarnation mechanism may handle an unresponsive attached daemon.

This decision supersedes only ADR-0009's permission to use bounded
operating-system termination when no live endpoint can be authenticated. It
does not change direct-child readiness cleanup, authenticated typed shutdown,
the daemon transport, bearer authentication, discovery schema, or normal
restart behavior.

## Authority And Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0005
> decision [active]: decision:ADR-0009
> observation [active]: code:potpie/daemon/lifecycle.py@942b704045469d7b61efdd92a6e58fb3c6ef192f
> observation [active]: code:potpie/runtime/controller.py@942b704045469d7b61efdd92a6e58fb3c6ef192f
> rationale [active]: issue:potpie-ai/potpie#1057-discussion-r3836380222

## Consequences

- A stale or reused recorded PID cannot authorize `SIGTERM` or `SIGKILL`.
- Healthy daemons started by an earlier CLI invocation still stop through the
  authenticated typed daemon-control operation.
- A directly owned child may still receive bounded termination during failed
  readiness or when its typed shutdown does not complete.
- An unresponsive daemon attached by a later invocation now fails closed and
  can require manual recovery.
- Stop refusal becomes a typed non-success outcome instead of a successful
  detail string.
- Runtime-record cleanup cannot delete a replacement boot's records through an
  unverified stale controller or a check-then-unlink race.

## Alternatives Considered

### Keep bounded PID-only termination

Bounding the wait limits duration but does not establish target identity. PID
reuse can therefore terminate an unrelated same-user process.

### Authenticate once and then signal the recorded PID

Authentication establishes the daemon endpoint at one instant but does not
bind later signals to the same operating-system process incarnation. The daemon
can exit and its PID can be reused between authentication and signalling.

### Persist a cross-platform process-incarnation token

Linux pidfds or process start times could support stronger reattachment, but a
portable identity contract and recovery policy require a separate decision.
The P1 fix does not depend on introducing that mechanism.

### Match executable name or command line

Names and command lines are not stable, unspoofable process identity and do not
close the PID-reuse race.

## Affected Behavior IDs

- Add `DAEMON-052` to authorize operating-system signals only for a directly
  owned child process.
- Add `DAEMON-053` to require typed failure without a stopped claim when an
  attached shutdown cannot complete.
- Add `DAEMON-054` to serialize canonical runtime-record publication with
  runtime ownership.
- Add `DAEMON-055` to serialize canonical runtime-record removal with runtime
  ownership.
- Add `DAEMON-056` to require exact expected PID and per-boot instance matching
  for identity-bound cleanup.

## Follow-Up Change Records

- `SPEC-CHANGE-0012`

## Acceptance

Accepted by `user:dsantra` at `2026-08-27T11:09:34+05:30`. This acceptance
restricts forceful termination to directly owned child processes and creates
no implementation or verification claim.
