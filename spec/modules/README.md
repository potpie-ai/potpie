---
id: SPEC-MODULES-INDEX
title: Context Runtime Module Index
kind: index
revision: 1
maturity: draft
owners:
  - team:potpie
depends_on: []
change_ref: null
---

# Context Runtime Module Index

The module contracts divide domain behavior, Potpie lease and resource
management, process hosting, and presentation into separate boundaries.

## Read Order

1. [Context Engine](context-engine.md)
2. [Potpie Resource Manager](potpie-resource-manager.md)
3. [Daemon](daemon.md)
4. [CLI](cli.md)

Read [the system contract](../system.md) first for the controller path, hosted
domain path, trust boundary, error taxonomy, and destructive-intent flow.

## Ownership Summary

| Module | Owns | Excludes |
|---|---|---|
| Context Engine | Thin domain façade, one bound identity, focused domain modules, typed engine outcomes | Product selection, authorization policy, provisioning, daemon, presentation |
| Resource Manager | Selection, authorization, host resources, explicit dependency composition, authorized context leases | Engine-operation dispatch, generic results, service lookup, daemon transport, presentation |
| Daemon subsystem | External controller, one foreground runtime, discovery, authenticated readiness, typed handlers | Domain semantics, terminal behavior |
| CLI | User intent, controller/client selection, human and machine presentation, streams and exits | Resource composition, daemon internals, domain semantics |

## Execution Direction

```text
CLI -> typed daemon handler -> handler requests lease from Resource Manager
                            -> Resource Manager issues lease
                            -> handler invokes Context Engine
```

The Resource Manager is in the acquisition path but not the domain-operation
dispatch path. Daemon context-domain handlers depend directly on the Context
Engine façade for invocation and on the Resource Manager for lease acquisition.
Context Engine does not import or depend on any Potpie product boundary.

## Lifecycle Direction

```text
CLI -> external daemon controller -> foreground runtime
CLI -> typed daemon client -> live readiness and runtime operations
```

Controller process observation never substitutes for the authenticated
readiness handshake.

## Deferred Detail

The exact public engine API, lease representation, Resource Manager cache
policy, controller API, daemon implementation source and protocol, concurrency
matrix, destructive-intent representation, machine JSON envelope, and
compatibility sequence are recorded in
[the question log](../questions/open.md).

## Conformance

These are accepted target contracts. No module implementation or verification
claim exists until a conformance record pins an accepted contract and
implementation ref.
