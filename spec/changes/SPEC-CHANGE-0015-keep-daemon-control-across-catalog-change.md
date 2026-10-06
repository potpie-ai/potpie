---
id: SPEC-CHANGE-0015
title: Keep Daemon Control Available Across An Operation-Catalog Change
kind: spec-change
change_status: proposed
spec_id: SPEC-DAEMON
from_revision: 2
from_ref: 217e51cd8b4ba4905c0f42d6bd6ec0c9aef86717
to_revision: 3
change_type: normative
initiated_by: team:potpie
authored_by:
  - team:potpie
accepted_by: []
accepted_at: null
---

# SPEC-CHANGE-0015: Keep Daemon Control Available Across An Operation-Catalog Change

## Intent

Let a client stop and inspect a daemon that a different Potpie build started,
without an operating-system signal.

Every release that adds a typed operation changes the operation-catalog
fingerprint. A daemon from the previous install then refuses the new client's
handshake. Status and shutdown need a ticket from a compatible handshake, and
`DAEMON-052` and `DAEMON-053` forbid signalling an attached process. So the
only way to stop that daemon is for the user to end the process by hand.

Status and shutdown do not depend on the operation catalog. Their wire
semantics already follow the protocol version. This change lets such a
handshake succeed with a ticket scoped to those two operations, and keeps
every context-domain and resource operation refused.

## Provenance Sources

> decision [active]: decision:ADR-0005
> decision [active]: decision:ADR-0009
> decision [active]: decision:ADR-0012
> observation [active]: code:potpie/runtime/server.py@32b8cbbb15e7dccff72890c4f5f9cbb4ddf6aaff
> observation [active]: code:potpie/runtime/clients.py@32b8cbbb15e7dccff72890c4f5f9cbb4ddf6aaff
> observation [active]: code:potpie/runtime/controller.py@32b8cbbb15e7dccff72890c4f5f9cbb4ddf6aaff

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| clarify | DAEMON-041 | DAEMON-041 | Tie the compatibility a handshake establishes to the operations its ticket authorizes. |
| add | — | DAEMON-057 | Answer a catalog-mismatched handshake with a ticket scoped to daemon control. |
| add | — | DAEMON-058 | Limit a control-scoped ticket to daemon status and shutdown, and refuse everything else before dispatch. |
| add | — | DAEMON-059 | Govern the handshake, status, and shutdown wire semantics by the protocol version. |
| add | — | DAEMON-060 | Keep a catalog-mismatched handshake from counting as readiness for context work in the client. |

## Semantic Diff

Revision 3 would add a second, narrower ticket scope and leave every other
daemon behavior unchanged. Until acceptance, `spec/modules/daemon.md` stays at
revision 2; this record carries the proposed text.

Proposed behavior nodes:

```text
DAEMON-041 [active]: A readiness handshake MUST establish compatible protocol semantics for every operation that its compatibility ticket authorizes.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-011

DAEMON-057 [active]: When an authenticated handshake has a compatible protocol range and expected instance identity but a client operation-catalog fingerprint that differs from the daemon's, the daemon MUST answer with its own fingerprint and a compatibility ticket scoped to daemon control, bound so that its holder cannot change that scope.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0009
  @ DAEMON-016
  @ DAEMON-040
  @ DAEMON-041

DAEMON-058 [active]: A control-scoped compatibility ticket MUST authorize only daemon status and daemon shutdown, and the daemon runtime MUST refuse every other operation presented with it with a typed ProtocolError before dispatching any handler.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0009
  @ DAEMON-045
  @ DAEMON-057

DAEMON-059 [active]: A change to the wire semantics of daemon handshake, status, or shutdown MUST be accompanied by a protocol version change.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0009
  @ DAEMON-041
  @ DAEMON-057

DAEMON-060 [active]: A typed daemon client MUST NOT treat a handshake whose operation-catalog fingerprint differs from its own as readiness for context-domain or resource operations, and MUST use a control-scoped ticket only for daemon status and shutdown.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0009
  @ DAEMON-011
  @ DAEMON-031
  @ DAEMON-058
```

Proposed prose changes in the module:

- Lifecycle Model, `ready` row: append "; a handshake answered with a
  control-scoped ticket confirms only that status and shutdown can be
  accepted".
- Failure Summary: add the row "Operation presented with a control-scoped
  ticket | ProtocolError `operation_catalog_mismatch`; no handler dispatch".
- Failure Summary, "Attached shutdown cannot authenticate, fails, or times
  out": unchanged. It still covers a daemon whose build predates this change,
  which refuses the handshake outright.

## Compatibility, Security, And Failure Impact

The handshake request and result keep their wire shapes, and the protocol
version stays 2. A client that predates this change receives a successful
handshake from a newer daemon, sees the differing fingerprint, and refuses it
with the same `operation_catalog_mismatch` code it reports today. A daemon
that predates this change still refuses a mismatched handshake, so a newer
client cannot stop it. That transition needs one manual stop, as today.
Every pair of builds that both include this change can see, stop, and
restart each other.

The bearer credential and instance identity are still required. A
control-scoped ticket grants strictly less than a full ticket. The daemon
binds the scope into the ticket's authentication code, so a holder cannot
widen it. It authorizes no context lease, no Context Engine call and no
resource operation (`DAEMON-045`). Shutdown remains typed and authenticated;
no signal path is added (`DAEMON-052`, `DAEMON-053`).

`DAEMON-059` makes a catalog change safe for daemon control. Without it,
status or shutdown could change shape under a stable protocol version and fail
to decode across builds.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| ADR-0009 | — | Its handshake fields and envelope are unchanged. Its rule that a client completes a compatible handshake before administrative operations still holds: the control ticket comes from a handshake whose protocol semantics are compatible for those operations. The accepted ADR is not edited. | team:potpie |
| ADR-0012, DAEMON-052, DAEMON-053 | — | No signal fallback is added. Attached shutdown stays typed, and a daemon that refuses the handshake still gets the typed refusal. | team:potpie |
| DAEMON-011, DAEMON-031, DAEMON-042 | — | Readiness still requires a live authenticated handshake reporting `ready`. DAEMON-060 keeps a control-scoped handshake from counting as readiness for context work. | team:potpie |
| DAEMON-045 | — | Status and shutdown handlers already take no context lease. | team:potpie |
| SPEC-CLI / CLI-023 | — | The CLI already uses the typed client for readiness and graceful stop. | team:potpie |
| Daemon runtime | Issue scoped tickets on a fingerprint-only mismatch, and refuse non-control operations with a control-scoped ticket before dispatch. | — | team:potpie |
| Typed daemon client and controller | Keep the control-scoped ticket for status and shutdown, never treat it as readiness, and stop readiness polling on a settled catalog or protocol refusal. | — | team:potpie |
| Daemon lifecycle and CLI | Report a catalog-mismatched daemon as not compatible and stale in `daemon status`, and let `daemon stop` and `restart` replace it. | — | team:potpie |
| Tests | Cover scoped issuance, refusal and non-widening; two real daemon processes with different catalogs; and a pin on the control wire shape. | — | team:potpie |
| Conformance validator | On acceptance, raise the covered active-behavior count from 195 to 199. | Not changed while proposed. | team:potpie |

## Conformance Invalidation

None while proposed. On acceptance, the current Daemon record derives stale
because it pins revision 2. A successor record must verify `DAEMON-041` and
`DAEMON-057` through `DAEMON-060` against the accepted revision. The CLI and
cross-system records, which list the Daemon contract as related, derive stale
on the same transition.

## Validation

```text
Structural: passed; scripts/validate_conformance_history.py with the module unchanged at revision 2
Semantic: proposed; scoped issuance, scope limits, protocol governance of control semantics, and client readiness are separate obligations
Provenance: proposed; authority edges await user:dsantra
Historical mutation: reviewed; revision would advance 2 to 3, and DAEMON-057 through DAEMON-060 are unused IDs
Dependency/consistency: reviewed against SPEC-SYSTEM, SPEC-CLI, and ADR-0009/ADR-0012; see the impact review
Fresh-agent reconstruction: pending acceptance review
Independent conformance state: revision 2 record unchanged; an implementation and tests accompany this proposal without a conformance claim
```

## Acceptance

Proposed. Pending acceptance by `user:dsantra`. Daemon revision 2 remains the
binding contract until then. Merging the accompanying implementation waits
for acceptance. This record creates no implementation or verification claim.
