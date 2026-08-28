---
id: ADR-0008
title: Define The Async Context Engine Public Contract
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-20T14:07:59+05:30
---

# ADR-0008: Define The Async Context Engine Public Contract

## Context

The accepted Context Engine contract binds one finite, thin, context-bound
façade, explicit construction dependencies, transport-neutral typed outcomes,
and removal of HostShell. It deliberately defers the exact operation catalog
and synchronous or asynchronous exposure. The current implementation mixes
synchronous and asynchronous bridges and exposes engine-owned and Potpie-owned
services through one HostShell-shaped object graph.

Implementation now needs one public construction and execution model that is
usable by direct Python hosts and can be mirrored exactly by local and daemon
clients without retaining paired method families or service lookup.

## Decision

The public `potpie_context_engine` package will expose this canonical
construction model:

```python
result = await create_engine(
    context=context_identity,
    config=engine_config,
    dependencies=engine_dependencies,
)
```

`create_engine` is asynchronous and returns
`Success[ContextEngine] | Failure[EngineError]`. Failed construction never
returns a usable engine. A successfully constructed engine supports
`async with` and an idempotent asynchronous `close()` operation.

`EngineError` is the public union of the already accepted engine-origin error
families: `DomainError`, `DependencyError`, and `EngineLifecycleError`. It does
not create a fourth canonical error category. Each variant carries a stable
category, code, message, optional structured details, recommended next action,
and retry posture. Expected operational failures return typed `Failure` values
rather than requiring callers to match exception text.

Every public Context Engine operation is asynchronous, accepts one engine-owned
typed request, and returns `Success[Result] | Failure[EngineError]`. The façade
does not publish paired synchronous methods. A host that requires a synchronous
adapter owns that adapter outside the Context Engine public façade.

The façade has this explicitly named flat method catalog for the currently
supported engine-owned operation families:

```text
resolve                 search                  record
data_plane_status       catalog                 describe
read                    search_entities         mutate
neighborhood            inspect                 export_snapshot
import_snapshot         repair                  propose
commit                  history                 quality
inbox_add               inbox_list              inbox_show
inbox_claim             inbox_mark_applied      inbox_mark_rejected
inbox_close             submit_event            submit_artifact
processing_status       nudge
```

Each method accepts its corresponding engine-owned request type, including
operations that currently use only scalar arguments. Adding a method outside
this catalog requires a later public-API decision; internal processing helpers
such as batch execution are not public methods merely because the façade uses
them.

Each method delegates to a focused domain component. The façade does not expose
dynamic dispatch, service lookup, a public service-container graph, or generic
`execute` and `call` methods.

The instance context identity is supplied once during construction. Public
operation requests contain neither `pot_id` nor any active-pot, repository,
daemon, or product selector capable of retargeting the instance.

Pot administration, source acquisition and credentials, authentication,
authorization, setup, configuration, skills, ledger presentation, UI,
installation, and daemon lifecycle are not Context Engine operations.

On acceptance, this decision resolves `OQ-CE-API-001`.

## Public Types And Ownership

- `ContextIdentity` is immutable and opaque to Context Engine domain behavior.
- `EngineConfig` contains only engine-owned configuration and never discovers
  Potpie home or active-pot state implicitly.
- `EngineDependencies` declares every required port and the ownership mode of
  every resource-bearing dependency.
- Borrowed dependencies remain host-owned; transferred or engine-created
  dependencies are closed in declared reverse ownership order.
- Calling an operation after terminal closure returns `EngineLifecycleError`.
- Public requests, results, outcomes, and errors belong to the Context Engine
  distribution and contain no CLI or daemon DTOs.

## Consequences

- Direct hosts have one canonical async API rather than sync/async pairs.
- Local and daemon clients can mirror one method catalog and one outcome model.
- Context selection cannot leak back into domain request types.
- Existing synchronous implementations may be adapted internally during the
  migration, but those adapters are not public façade methods and are removed
  when no longer needed.
- Product services move to Potpie instead of being renamed behind the new
  façade.
- Adding a new daemon operation does not automatically add a Context Engine
  method; it must represent an accepted engine-owned domain capability.

## Alternatives Considered

### Paired synchronous and asynchronous façade methods

This preserves current calling styles but doubles the public surface and keeps
the sync/async bridge as a permanent compatibility obligation.

### Synchronous construction with deferred readiness

This makes construction appear complete before asynchronous dependency and
capability checks have established a usable engine.

### Operation objects with one generic execute method

This provides a finite type union but recreates generic dispatch on the public
façade and obscures the named domain boundary required by `CE-001` and
`CE-020`.

### Typed exceptions as the public failure contract

This is idiomatic for some Python APIs but makes exact local/daemon outcome
parity depend on exception translation and encourages exception-string
matching in compatibility code.

## Contract And Impact Review

No accepted behavior changes. This decision selects details explicitly
deferred by the accepted contracts and requires no contract revision or
spec-change record.

| Artifact or area | Required change after acceptance | No-change reason |
|---|---|---|
| SPEC-CONTEXT-ENGINE | — | `CE-001`, `CE-005` through `CE-012`, `CE-020` through `CE-025`, and `CE-028` through `CE-032` already permit and constrain this API. |
| SPEC-SYSTEM | — | Direct-host composition, error origins, and ownership direction remain unchanged. |
| SPEC-POTPIE-RESOURCE-MANAGER | — | Potpie selection and lease responsibilities remain outside Context Engine. |
| SPEC-DAEMON | — | The daemon mirrors typed operations without changing engine semantics. |
| SPEC-CLI | — | Presentation and selection remain CLI and Potpie responsibilities. |
| ADR-0001 through ADR-0007 | — | Existing accepted decisions remain immutable. |
| SPEC-CHANGE-0001 through SPEC-CHANGE-0008 | — | No behavior operation or contract revision occurs. |
| Existing conformance | — | No conformance records exist and this decision makes no implementation claim. |

## Authority And Sources

> decision [active]: decision:ADR-0002
> decision [active]: decision:ADR-0003
> decision [active]: decision:ADR-0006
> decision [active]: decision:ADR-0007
> observation [active]: code:potpie/context-engine/src/potpie_context_engine/host/shell.py@5a54cbb2060b67f43718bf47e6e453bebc598325
> observation [active]: code:potpie/context-core/src/potpie_context_core/ports/graph_service.py@5a54cbb2060b67f43718bf47e6e453bebc598325

## Acceptance

Accepted by `user:dsantra` at `2026-08-20T14:07:59+05:30`. This acceptance
resolves `OQ-CE-API-001`. It does not claim implementation or verification
conformance.
