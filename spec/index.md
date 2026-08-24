---
id: SPEC-INDEX
title: Context Runtime Specification Index
kind: index
revision: 1
maturity: draft
owners:
  - team:potpie
depends_on: []
change_ref: null
---

# Context Runtime Specification Index

This is the navigation entrypoint for the Context Engine and Potpie runtime
boundary. Canonical target behavior lives in contract envelopes and behavior
nodes. Decisions explain why, change records describe revision transitions,
questions preserve deferred choices, and conformance records separately
describe implementation and evidence.

Eight revision-1 contracts and corrective Resource Manager revision 2 are
accepted binding targets. Resource Manager revision 1 remains historically
addressable at `047cbe067c9c726e7e14f066675453372d8a8406`.
Existing architecture documents and code remain implementation snapshots and
can disagree with the target while migration is incomplete.

## Read Order

1. [Specification process](process.md)
2. [Glossary](glossary.md)
3. [Product contract](product.md)
4. [System contract](system.md)
5. [Potpie Capabilities contract](modules/potpie-capabilities.md)
6. [Context Engine contract](modules/context-engine.md)
7. [Potpie Resource Manager contract](modules/potpie-resource-manager.md)
8. [Daemon contract](modules/daemon.md)
9. [CLI contract](modules/cli.md)
10. Relevant [decisions](#decision-registry), [questions](questions/open.md), and
   [change records](#change-record-registry)

## Target Paths

```text
daemon lifecycle control

human or automation
        |
        v
Potpie CLI -> external daemon controller -> foreground daemon runtime
                                             |
                                             v
                              authenticated readiness through typed client

hosted domain execution

Potpie CLI
  -> typed daemon client
  -> explicit typed daemon operation handler
      -> handler requests authorized context lease
      -> Resource Manager issues lease
      -> handler invokes explicit ContextEngine operation
      -> handler releases lease
```

Direct compatible hosts enter at the Context Engine boundary with an explicit
identity, dependency set, and ownership declarations.

## Contract Registry

This table is a derived navigation view. Contract envelopes and behavior nodes
remain canonical.

| ID | File | Revision | Maturity | Behaviors | Summary |
|---|---|---:|---|---:|---|
| SPEC-PROCESS | [process.md](process.md) | 1 | accepted | 21 | Authority, acceptance, mutation, and conformance policy |
| SPEC-GLOSSARY | [glossary.md](glossary.md) | 1 | accepted | 12 | Canonical runtime, lease, ownership, and error terminology |
| SPEC-PRODUCT | [product.md](product.md) | 1 | accepted | 7 | Product outcomes and non-goals |
| SPEC-SYSTEM | [system.md](system.md) | 1 | accepted | 23 | Cross-module ownership, trust, error, and call-path contract |
| SPEC-POTPIE-CAPABILITIES | [modules/potpie-capabilities.md](modules/potpie-capabilities.md) | 1 | accepted | 12 | Capability-oriented root source ownership and behavior-preserving migration |
| SPEC-CONTEXT-ENGINE | [modules/context-engine.md](modules/context-engine.md) | 1 | accepted | 34 | Thin importable context-domain library boundary |
| SPEC-POTPIE-RESOURCE-MANAGER | [modules/potpie-resource-manager.md](modules/potpie-resource-manager.md) | 2 | accepted | 37 | Authorized context lease and resource-lifecycle boundary, including distinct authentication failure |
| SPEC-DAEMON | [modules/daemon.md](modules/daemon.md) | 1 | accepted | 51 | Controller, typed runtime, discovery, readiness, lifecycle, and handlers |
| SPEC-CLI | [modules/cli.md](modules/cli.md) | 1 | accepted | 33 | Human and machine presentation plus controller/client selection |

The catalog contains 230 accepted active behavior nodes.

## Exact Contract Dependencies

| Contract | Direct dependencies |
|---|---|
| SPEC-PROCESS | none |
| SPEC-GLOSSARY | none |
| SPEC-PRODUCT | SPEC-GLOSSARY |
| SPEC-SYSTEM | SPEC-GLOSSARY, SPEC-PRODUCT |
| SPEC-POTPIE-CAPABILITIES | SPEC-PRODUCT, SPEC-SYSTEM |
| SPEC-CONTEXT-ENGINE | SPEC-GLOSSARY, SPEC-SYSTEM |
| SPEC-POTPIE-RESOURCE-MANAGER | SPEC-GLOSSARY, SPEC-SYSTEM, SPEC-CONTEXT-ENGINE |
| SPEC-DAEMON | SPEC-GLOSSARY, SPEC-SYSTEM, SPEC-POTPIE-RESOURCE-MANAGER, SPEC-CONTEXT-ENGINE |
| SPEC-CLI | SPEC-GLOSSARY, SPEC-SYSTEM, SPEC-DAEMON |

The primary runtime layering spine is:

```text
CLI -> daemon -> Resource Manager -> Context Engine
```

Shared glossary and system dependencies are explicit in the table rather than
hidden by that simplified spine.

## Module Registry

See [modules/README.md](modules/README.md) for the ownership summary and module
read order.

## Decision Registry

| ID | Decision | Status |
|---|---|---|
| ADR-0001 | [Git-based specification governance](decisions/ADR-0001-spec-governance.md) | accepted |
| ADR-0002 | [One importable Context Engine distribution](decisions/ADR-0002-single-importable-context-engine.md) | accepted |
| ADR-0003 | [Explicit composition and immutable context scoping](decisions/ADR-0003-explicit-composition-and-context-scoping.md) | accepted |
| ADR-0004 | [Potpie resource-management ownership](decisions/ADR-0004-potpie-resource-management-ownership.md) | accepted |
| ADR-0005 | [Typed daemon and CLI boundary](decisions/ADR-0005-typed-daemon-and-cli-boundary.md) | accepted |
| ADR-0006 | [Deferred runtime concerns](decisions/ADR-0006-deferred-runtime-concerns.md) | accepted |
| ADR-0007 | [Context runtime migration path](decisions/ADR-0007-context-runtime-migration-path.md) | accepted |
| ADR-0008 | [Async Context Engine public contract](decisions/ADR-0008-async-context-engine-public-contract.md) | accepted |
| ADR-0009 | [Typed local runtime execution contract](decisions/ADR-0009-typed-local-runtime-execution-contract.md) | accepted |
| ADR-0010 | [Correct Resource Manager authentication outcomes](decisions/ADR-0010-resource-manager-authentication-outcomes.md) | accepted |
| ADR-0011 | [Capability-oriented Potpie layout](decisions/ADR-0011-capability-oriented-potpie-layout.md) | accepted |

## Open Questions

Three intentionally deferred questions and nine resolved implementation-readiness
questions are recorded in [questions/open.md](questions/open.md). No accepted
active behavior depends on a deferred question.

## Change Record Registry

| ID | Contract | Transition | Status |
|---|---|---|---|
| [SPEC-CHANGE-0001](changes/SPEC-CHANGE-0001-initialize-spec-process.md) | SPEC-PROCESS | 0 → 1 | accepted |
| [SPEC-CHANGE-0002](changes/SPEC-CHANGE-0002-initialize-glossary.md) | SPEC-GLOSSARY | 0 → 1 | accepted |
| [SPEC-CHANGE-0003](changes/SPEC-CHANGE-0003-initialize-product-contract.md) | SPEC-PRODUCT | 0 → 1 | accepted |
| [SPEC-CHANGE-0004](changes/SPEC-CHANGE-0004-initialize-system-contract.md) | SPEC-SYSTEM | 0 → 1 | accepted |
| [SPEC-CHANGE-0005](changes/SPEC-CHANGE-0005-initialize-context-engine-contract.md) | SPEC-CONTEXT-ENGINE | 0 → 1 | accepted |
| [SPEC-CHANGE-0006](changes/SPEC-CHANGE-0006-initialize-resource-manager-contract.md) | SPEC-POTPIE-RESOURCE-MANAGER | 0 → 1 | accepted |
| [SPEC-CHANGE-0007](changes/SPEC-CHANGE-0007-initialize-daemon-contract.md) | SPEC-DAEMON | 0 → 1 | accepted |
| [SPEC-CHANGE-0008](changes/SPEC-CHANGE-0008-initialize-cli-contract.md) | SPEC-CLI | 0 → 1 | accepted |
| [SPEC-CHANGE-0009](changes/SPEC-CHANGE-0009-correct-resource-manager-authentication-outcomes.md) | SPEC-POTPIE-RESOURCE-MANAGER | 1 → 2 | accepted |
| [SPEC-CHANGE-0010](changes/SPEC-CHANGE-0010-initialize-potpie-capability-contract.md) | SPEC-POTPIE-CAPABILITIES | 0 → 1 | accepted |

## Conformance Summary

Five final records verify the Context Runtime implementation at
`5101871348ceae9f59830dd82af06d890a6d0f48`:

| Record | Contract | Revision | Result |
|---|---|---:|---|
| [CONF-CONTEXT-ENGINE-2026-08-21-01](conformance/context-engine-2026-08-21.md) | SPEC-CONTEXT-ENGINE | 1 | passed |
| [CONF-POTPIE-RESOURCE-MANAGER-2026-08-21-01](conformance/potpie-resource-manager-2026-08-21.md) | SPEC-POTPIE-RESOURCE-MANAGER | 2 | passed |
| [CONF-DAEMON-2026-08-21-01](conformance/daemon-2026-08-21.md) | SPEC-DAEMON | 1 | passed |
| [CONF-CLI-2026-08-21-01](conformance/cli-2026-08-21.md) | SPEC-CLI | 1 | passed |
| [CONF-SYSTEM-2026-08-21-01](conformance/cross-system-2026-08-21.md) | SPEC-SYSTEM | 1 | passed |

Together they record passed verification for all 178 active behaviors in the
four implementation modules and their cross-system contract. Freshness is
derived from the pinned specification, implementation, dependency, and
evidence identities; it is not stored in these records.

## Current Snapshot

Initial implementation observations remain pinned to base
`a341978880b9d4c1b403831931279ccedf6184ae` and explain the migration need. The
final conformance records above separately establish implementation and
verification claims at `5101871348ceae9f59830dd82af06d890a6d0f48`.
