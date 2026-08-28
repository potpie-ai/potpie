---
id: ADR-0010
title: Correct Resource Manager Authentication Outcomes
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-21T14:55:35+05:30
---

# ADR-0010: Correct Resource Manager Authentication Outcomes

## Context

ADR-0009 requires `ContextResourceManager` to perform selection,
authentication, authorization, resource composition, and engine acquisition in
that order. Its acquisition-outcome sentence omitted `AuthenticationError`,
while the same accepted runtime taxonomy requires authentication failures to
remain structurally distinct. Revision 1 of the Resource Manager contract
copied that incomplete exhaustive outcome list into `RM-015` and `RM-016`.

The implementation already returns a typed `AuthenticationError` and stops
before authorization or composition when authentication fails. A conformance
record cannot treat the omission as editorial because it changes the set of
allowed outcomes at the Resource Manager boundary.

## Decision

Resource Manager acquisition includes `AuthenticationError` alongside
`SelectionError`, `AuthorizationError`, and `ResourceLifecycleError`.
Authentication failure terminates acquisition before authorization, resource
composition, engine construction, or lease issuance.

This decision supersedes only ADR-0009's incomplete acquisition-outcome
enumeration. It does not change the selected authentication mechanism,
selection/authentication/authorization ordering, daemon bearer-token boundary,
authorization policy, engine lifetime, transport, or retry behavior.

## Authority And Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0004
> decision [active]: decision:ADR-0005
> decision [active]: decision:ADR-0009
> observation [active]: code:potpie/runtime/resource_manager.py@097bae17bfb2696b18293ed33585ea4a50f4337d
> observation [active]: test:tests/unit/test_context_resource_manager.py@097bae17bfb2696b18293ed33585ea4a50f4337d

## Consequences

- The Resource Manager contract can describe the already selected acquisition
  order without collapsing authentication into authorization.
- Local and daemon-hosted handlers retain the same Resource Manager policy.
- System, daemon, and CLI error taxonomies need no revision because they
  already distinguish `AuthenticationError`.
- Conformance must prove both the returned category and the pre-composition
  short circuit.

## Alternatives Considered

### Convert authentication failures to AuthorizationError

This would erase the structural distinction required by the accepted system
and daemon error taxonomies.

### Move authentication outside ContextResourceManager

This would contradict ADR-0009's selected execution ordering and create
different acquisition policy between local and daemon-hosted handlers.

### Treat the omission as an editorial correction

Adding an allowed failure category changes observable boundary behavior. The
mutation protocol therefore requires replacement behavior IDs and a new
contract revision.

## Affected Behavior IDs

- Retire `RM-015` and replace it with `RM-036`.
- Retire `RM-016` and replace it with `RM-037`.
- Add `RM-038` for the authentication-failure outcome.
- Add `RM-039` for authentication short-circuit ordering.

## Follow-Up Change Records

- `SPEC-CHANGE-0009`

## Acceptance

Accepted by `user:dsantra` at `2026-08-21T14:55:35+05:30`. This acceptance
corrects only the Resource Manager authentication-outcome enumeration and
creates no implementation or verification claim.
