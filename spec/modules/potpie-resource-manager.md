---
id: SPEC-POTPIE-RESOURCE-MANAGER
title: Potpie Resource Manager Contract
kind: module-spec
revision: 2
maturity: accepted
owners:
  - team:potpie
depends_on:
  - SPEC-GLOSSARY
  - SPEC-SYSTEM
  - SPEC-CONTEXT-ENGINE
change_ref: SPEC-CHANGE-0009
---

# Potpie Resource Manager Contract

## Purpose

The Potpie Resource Manager is the product-owned provider of authorized,
context-bound engine leases. It resolves selection, authenticates the caller,
applies authorization, acquires host-managed resources, composes or reuses a
compatible engine, and returns an authorized context lease. It does not
execute product or domain operations.

The Resource Manager is a logical subsystem. This contract does not require one
class, process, or package with this name.

## Ownership And Boundaries

The Resource Manager owns:

- Resolution of product selection to one context identity.
- Authentication of caller material supplied at the Potpie operation boundary.
- Authorization policy for actor, operation, and resolved context.
- Provisioning, opening, validating, and releasing Potpie-managed resources.
- Construction of explicit Context Engine dependencies with ownership modes.
- Construction or compatible reuse of a context-bound engine.
- Issuance and release of authorized context leases.
- Recovery from failed lease acquisition and resource failure.

It excludes product-operation dispatch, Context Engine method mirroring,
context-domain semantics, arbitrary service lookup, daemon transport, and
terminal presentation.

## Scope And Non-Goals

This revision defines the logical responsibility and lease boundary. It does
not freeze implementation decomposition, cache shape, eviction, idle timeout,
or a public Python symbol. It does not define daemon transport, CLI prompting,
or the exact lease representation.

## Actors And Permissions

| Actor | Interaction |
|---|---|
| Potpie operation handler | Supplies caller authentication material, typed operation, selection request, and intent where needed |
| Resource Manager | Resolves, authenticates, authorizes, acquires, leases, and releases |
| Context Engine | Is held by the lease and invoked directly by the operation handler |
| Resource adapter | Opens or provisions one explicitly selected dependency |

## Normative Requirements

RM-001 [active]: The Potpie Resource Manager MUST resolve a context-selection request to exactly one logical context identity or return SelectionError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004

RM-002 [active]: Context-selection resolution MUST remain separate from CLI prompting, rendering, and remediation text.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-001

RM-003 [active]: The Resource Manager MUST issue an authorized context lease only after receiving an authenticated actor and authorizing that actor's operation against the resolved context identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-001

RM-004 [active]: Unknown, ambiguous, and unavailable context-selection outcomes MUST remain distinct SelectionError variants.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-001
  @ SYS-007

RM-005 [active]: Before issuing a lease for a destructive operation, the Resource Manager MUST validate the destructive-intent assertion against the authenticated actor, requested operation, context-selection request, and resolved context identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ RM-003
  @ SYS-020

RM-006 [active]: The Resource Manager MUST open, provision, validate, and release Potpie-managed resources through explicit resource adapters.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004

RM-007 [active]: The Resource Manager MUST construct the explicit compatible dependency set and ownership declarations supplied to Context Engine.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-006
  @ CE-005

RM-008 [active]: The Resource Manager MUST return an authorized context lease containing one Context Engine permanently bound to the resolved context identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-001
  @ RM-007
  @ CE-006

RM-009 [active]: The Resource Manager MUST NOT retarget an existing Context Engine instance to a different context identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ RM-008
  @ CE-008

RM-010 [active]: If resources or engines are reused, reuse MUST require the same context identity and an explicitly compatible composition configuration.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-008

RM-011 [active]: The Resource Manager MUST own idempotent release of every Potpie-managed resource acquired for an authorized context lease unless and until ownership is explicitly transferred under RM-027.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-006

RM-012 [active]: Potpie in-process and daemon-hosted operation handlers MUST use the same selection, authorization, dependency-composition, and lease-acquisition policies.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-007
  @ RM-008

RM-013 [active]: The Resource Manager MUST coordinate resource acquisition, resource release, and lease-lifecycle conflicts at the narrowest context and resource scope that preserves correctness.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ RM-008

RM-014 [active]: The Resource Manager MUST NOT serialize unrelated contexts through a process-wide operation lock.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ RM-013

RM-015 [retired]: Resource Manager acquisition MUST return either an authorized context lease or a typed SelectionError, AuthorizationError, or ResourceLifecycleError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ SYS-007

RM-016 [retired]: SelectionError, AuthorizationError, and ResourceLifecycleError MUST remain distinct at the Resource Manager boundary.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-004
  @ RM-015

RM-017 [active]: The Resource Manager MUST NOT implement or override Context Engine domain semantics.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ CE-011

RM-018 [active]: The Resource Manager MUST NOT own terminal presentation.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-002

RM-019 [active]: The Resource Manager MUST return AuthorizationError when an authenticated actor lacks the requested operation or resolved-context scope.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-003
  @ SYS-007

RM-020 [active]: An authorized context lease MUST identify its resolved context identity and authenticated actor-operation-context authorization scope.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-003
  @ RM-008

RM-021 [active]: An authorized context lease MUST expose explicit ownership and lifecycle metadata for every resource-bearing dependency in its scope.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ RM-007
  @ RM-020

RM-022 [active]: The Resource Manager and authorized context lease MUST NOT invoke or dispatch Context Engine operations or return domain or generic product-operation results.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-008
  @ SYS-019

RM-023 [active]: The Resource Manager and authorized context lease MUST NOT expose generic operation dispatch, arbitrary service lookup, or a mirrored Context Engine service surface.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-022
  @ CE-020

RM-024 [active]: The Resource Manager MUST NOT own daemon transport.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-018

RM-025 [active]: A failed lease-acquisition attempt MUST attempt release of every Potpie-managed resource opened by that attempt.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-006
  @ RM-011
  @ RM-036

RM-026 [active]: When a lease owns terminal engine cleanup, lease release MUST request cleanup of engine-owned resources before releasing the host-managed resources supplied to that engine.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ RM-011
  @ CE-024

RM-027 [active]: The Resource Manager MUST retain cleanup ownership for a borrowed Potpie-managed dependency.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ RM-007
  @ CE-022

RM-028 [active]: Resource Manager logs and error details MUST exclude credentials and sensitive context payloads by default.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004

RM-029 [active]: The Resource Manager MUST NOT retain duplicate cleanup ownership after explicitly transferring a dependency to Context Engine.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ RM-027
  @ CE-024

RM-030 [active]: Every issued authorized context lease MUST expose an idempotent release capability.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-011
  @ RM-020

RM-031 [active]: Failure of engine-owned cleanup MUST NOT prevent an attempt to release each host-owned resource for which the lease owns terminal cleanup.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-026
  @ RM-030

RM-032 [active]: A lease-release failure MUST be represented as ResourceLifecycleError without erasing an already-produced domain result, DomainError, DependencyError, or EngineLifecycleError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-030
  @ SYS-007

RM-033 [active]: Failed destructive-intent validation MUST return AuthorizationError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ RM-005
  @ RM-019

RM-034 [active]: When cleanup after failed acquisition is incomplete, the resulting ResourceLifecycleError MUST preserve both acquisition-failure and cleanup-failure detail.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ RM-025

RM-035 [active]: Failed destructive-intent validation MUST NOT issue an authorized context lease.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ RM-005
  @ RM-033

RM-036 [active]: Resource Manager acquisition MUST return either an authorized context lease or a typed SelectionError, AuthenticationError, AuthorizationError, or ResourceLifecycleError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  > decision [active]: decision:ADR-0010
  @ RM-003
  @ SYS-007

RM-037 [active]: SelectionError, AuthenticationError, AuthorizationError, and ResourceLifecycleError MUST remain structurally distinct at the Resource Manager boundary.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  > decision [active]: decision:ADR-0010
  @ RM-036
  @ SYS-007

RM-038 [active]: Failed caller authentication during Resource Manager acquisition MUST return AuthenticationError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0010
  @ RM-003
  @ RM-036

RM-039 [active]: Failed caller authentication MUST stop Resource Manager acquisition before authorization, resource composition, engine construction, or lease issuance.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0010
  @ RM-003
  @ RM-038

## Lease Model

The conceptual lease contains:

- One resolved context identity.
- One Context Engine bound to that identity.
- Authenticated actor-operation-context authorization scope.
- Explicit dependency ownership and one release capability.

It contains no `execute` function, method forwarding, arbitrary services,
domain results, transport DTOs, or presentation behavior. These statements
summarize `RM-008` and `RM-020` through `RM-023`.

## Acquisition And Release

```text
selection
  -> resolved identity
  -> authenticated actor
  -> authorized operation scope
  -> resources acquired
  -> compatible engine obtained
  -> authorized context lease issued
  -> handler invokes engine directly
  -> lease released
```

Failed acquisition follows `RM-025`. The exact reuse, eviction, and idle
lifetime remain deferred.

## Failure Summary

| Condition | Outcome |
|---|---|
| No, several, or unavailable matching contexts | SelectionError variant |
| Caller authentication material is rejected | AuthenticationError |
| Authenticated actor lacks operation or context scope | AuthorizationError |
| Invalid destructive intent at the authorized boundary | AuthorizationError |
| Host resource cannot be acquired or released | ResourceLifecycleError |

Domain, dependency, and engine-lifecycle outcomes occur after the typed
context-domain handler invokes Context Engine and therefore do not pass through
the Resource Manager. Lease-release failure is a separate typed outcome that
preserves the already-produced operation outcome.

## Acceptance Criteria

- The subsystem can return only an authorized lease or its own typed failure.
- No lease or Resource Manager surface can dispatch engine operations.
- Dependency ownership and terminal cleanup order are explicit.
- Selection, authentication, and authorization precede protected engine access.
- Reuse cannot retarget an engine.
- Resource and lease coordination cannot serialize unrelated contexts.
- The subsystem owns neither domain semantics, transport, nor presentation.

## Implementation Notes

The exact module layout, caching, lease representation, and service
decomposition remain implementation choices. Current setup and host-wiring
paths are migration evidence only.
