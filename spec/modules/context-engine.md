---
id: SPEC-CONTEXT-ENGINE
title: Context Engine Contract
kind: module-spec
revision: 1
maturity: accepted
owners:
  - team:potpie
depends_on:
  - SPEC-GLOSSARY
  - SPEC-SYSTEM
change_ref: SPEC-CHANGE-0005
---

# Context Engine Contract

## Purpose

Context Engine is the importable context-domain library. It exposes a finite,
thin façade bound to one context identity and operates through dependencies
supplied explicitly by a compatible host.

## Ownership And Boundaries

Context Engine owns:

- Context-domain operations, values, errors, and invariants.
- Focused domain modules behind its public façade.
- Declared engine-owned ports and package adapters for those ports.
- Engine-internal state and dependencies explicitly transferred to it.
- Transport-neutral engine outcomes suitable for direct or hosted use.

It excludes Potpie selection, caller security policy, host-resource
provisioning, daemon behavior, CLI presentation, installation, and product
lifecycle.

## Scope And Non-Goals

This revision defines ownership, construction, identity isolation, dependency
ownership, outcomes, lifecycle, and migration end state. It does not freeze the
exact façade method catalog, sync or async exposure, compatibility shim, or
adapter catalog. Parsing and public extensions remain outside scope.

## Actors And Permissions

| Actor | Interaction |
|---|---|
| Compatible host | Supplies context identity, dependencies, ownership modes, and permitted calls |
| Potpie Resource Manager | Acts as the Potpie host that composes or obtains the engine placed into an authorized context lease |
| Domain caller | Invokes a finite operation on the façade after host authorization |
| Context Engine | Enforces domain invariants for the bound identity |

Context Engine trusts the host to establish caller identity and permission. It
still validates domain input and preserves its immutable context identity.

## Normative Requirements

CE-001 [active]: Context Engine MUST expose `ContextEngine` as a finite, thin public façade whose explicitly named methods represent context-domain operations.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  ~ potpie/context-engine/src/potpie_context_engine/api.py

CE-002 [active]: A compatible host MUST be able to use the public Context Engine façade without importing Potpie daemon or CLI internals.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  @ CE-001

CE-003 [active]: Types and behavior required by the public Context Engine contract MUST belong to the single `potpie-context-engine` distribution.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  @ CE-001
  ~ potpie/context-engine/pyproject.toml

CE-004 [active]: The migration end state MUST NOT retain `potpie-context-core` as an independent architectural or public distribution boundary.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  @ CE-003
  ~ potpie/context-core/pyproject.toml

CE-005 [active]: A host MUST explicitly supply Context Engine's context identity, every required dependency, and the ownership mode of every resource-bearing dependency during construction.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ SYS-018

CE-006 [active]: One Context Engine instance MUST remain permanently bound to exactly one logical context identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-005

CE-007 [active]: A Context Engine operation MUST NOT accept a second selector that overrides the instance's bound context identity.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-006

CE-008 [active]: A host that needs a different context identity MUST construct or obtain a different Context Engine instance.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-006

CE-009 [active]: Context Engine instances for different context identities MUST be able to coexist without process-global context-identity state.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-006

CE-010 [active]: Cleanup of engine-owned resources MUST be safe to request more than once without duplicating destructive effects.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-005

CE-011 [active]: Context Engine MUST be the sole owner of context-domain semantics and invariants in this system.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002

CE-012 [active]: Every Context Engine operation MUST return a typed transport-neutral domain value or a typed DomainError, DependencyError, or EngineLifecycleError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CE-011
  @ SYS-007

CE-013 [active]: Context Engine MUST NOT prompt, print, render terminal presentation, or select process exit codes.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CE-012

CE-014 [active]: Context Engine MUST NOT authenticate callers or enforce Potpie product authorization policy.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ CE-011

CE-015 [active]: Context Engine MUST NOT silently discover or provision host-managed resources.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-005

CE-016 [active]: A package-resident Context Engine adapter MAY implement only a declared engine-owned port.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-015

CE-017 [active]: A destructive Context Engine operation MUST be represented by an explicit destructive domain command.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CE-006

CE-018 [active]: This revision MUST NOT expose a public plugin, extension-registration, or manifest contract.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006
  ~ potpie/context-engine/src/potpie_context_engine/__init__.py

CE-019 [active]: Work performed under this boundary contract MUST NOT redesign parsing behavior.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006

CE-020 [active]: The `ContextEngine` façade MUST NOT expose dynamic dispatch, arbitrary service lookup, a service-container graph, or pass-through mirrors of internal services.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  @ CE-001

CE-021 [active]: The `ContextEngine` façade MUST delegate implementation to focused context-domain modules rather than accumulate a universal host or service implementation.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  @ CE-001
  @ CE-020

CE-022 [active]: Context Engine MUST treat a host-supplied resource-bearing dependency as borrowed unless a typed construction contract explicitly transfers ownership.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-005

CE-023 [active]: Context Engine MUST NOT close or release a borrowed dependency.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-022

CE-024 [active]: Context Engine MUST release a transferred or engine-created dependency according to its declared lifecycle during engine closure.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-010
  @ CE-022

CE-025 [active]: An operation attempted after terminal engine closure MUST return EngineLifecycleError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-010
  @ CE-012

CE-026 [active]: A package-resident Context Engine adapter MUST NOT implement or depend on Potpie selection, product authorization, resource provisioning, installation, daemon, CLI, or product-lifecycle modules.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  @ CE-016

CE-027 [active]: Completion of the Context Engine boundary migration MUST remove HostShell and Potpie-owned host wiring from the Context Engine distribution rather than retain or rename them behind `ContextEngine`.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  @ CE-020
  ~ potpie/context-engine/src/potpie_context_engine/host/shell.py
  ~ potpie/context-engine/src/potpie_context_engine/bootstrap/host_wiring.py

CE-028 [active]: Public Context Engine operation requests, results, and errors MUST use engine-owned types rather than Potpie selection records, daemon protocol DTOs, or CLI presentation types.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  @ CE-012

CE-029 [active]: A context-domain semantic failure MUST return DomainError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CE-012

CE-030 [active]: A dependency failure while fulfilling a declared engine-owned port MUST return DependencyError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CE-012

CE-031 [active]: Context Engine error and observability output MUST exclude secrets by default.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002

CE-032 [active]: Failed Context Engine construction MUST NOT yield a usable engine instance.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-005

CE-033 [active]: A package-resident Context Engine adapter MUST be selected and composed explicitly by the host.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ CE-016

CE-034 [active]: A destructive Context Engine operation MUST NOT be reachable through a default or inferred operation.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CE-017

## Data And State Model

An engine instance contains one immutable context identity, one explicit
dependency set with declared ownership, focused domain modules, and lifecycle
state. A pot identifier or CLI selector can be input to the host but is not
mutable engine state.

## Lifecycle Summary

This state sketch summarizes `CE-010`, `CE-023`, `CE-024`, `CE-025`,
and `CE-032`:

```text
construction -> usable -> closing -> closed
       |
       -> construction failure, with no usable instance
```

Closure releases engine-owned resources and detaches from borrowed resources.
Changing context identity is not a lifecycle transition.

## Outcome Summary

| Condition | Typed outcome |
|---|---|
| Domain input, state, capability, or invariant rejects the operation | DomainError |
| Engine-owned port fails during an otherwise valid operation | DependencyError |
| Operation is attempted after terminal closure | EngineLifecycleError |
| Potpie selection, authentication, authorization, resource, protocol, or presentation failure | Not originated by Context Engine |

This table summarizes `CE-012`, `CE-025`, `CE-029`, and `CE-030`.

## Compatibility, Migration, And Rollout

Later commits merge required Context Core types into Context Engine and remove
Potpie-owned composition from the distribution. Temporary import shims can
preserve callers only as migration mechanisms; they do not satisfy the final
boundary.

## Acceptance Criteria

- The public façade cannot become a service locator or renamed HostShell.
- One instance cannot be retargeted.
- Borrowed and transferred ownership cannot be confused.
- Package adapters implement only engine-owned ports.
- Engine outcomes remain typed and transport neutral.
- Product authentication, provisioning, daemon, and presentation remain
  outside the distribution.

## Implementation Notes

The current HostShell, host wiring, separate Context Core package, CLI auth,
installer, setup, and skill-related ports are migration evidence rather than the
target architecture. These notes create no conformance claim.
