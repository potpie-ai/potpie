---
id: ADR-0013
title: Support A Graph-Runtime Builder For Embedding Hosts
kind: decision
decision_status: proposed
owners:
  - team:potpie
initiated_by: team:potpie
decision_makers:
  - user:dsantra
decided_at: null
---

# ADR-0013: Support A Graph-Runtime Builder For Embedding Hosts

## Context

ADR-0002 and ADR-0008 make `ContextEngine` the finite, context-bound public
façade, and ADR-0006 records that existing runtime code creates no public
promise merely because it exists. Revision 1 accordingly removed
`build_graph_runtime`, `GraphRuntime`, and `GraphExtension` from the public
exports.

A second kind of compatible host needs a supported entry point. An embedding
host is a long-running service that brings its own graph backend and its own
plan and inbox stores, serves many contexts from one process, and owns caller
authentication, authorization, tenancy, and audit. The `ContextEngine` façade
does not fit that host:

- One instance is bound to one context identity (`CE-006`), so a multi-context
  service would construct and cache an engine per context.
- Its construction takes `EngineDependencies`, whose operation groups are
  implemented only by root `potpie` (`potpie/runtime/local_engine.py`), so a
  host cannot compose an engine from the engine distribution alone.

Such a host therefore imports `potpie_context_engine.core.runtime`, which has
no compatibility promise, and patches the composed runtime after construction:
the commit listing index, the rollback preview store, the host label, and the
actor and authorization callbacks. It also sets the runtime's graph service,
claim query, snapshot port, and journal on its resource facade by hand.
Patching is fragile, and a missed authorization patch left a permissive
single-user default in place.

## Decision

`potpie_context_engine.api` exports `build_graph_runtime`, `GraphRuntime`,
and `ResourceFacade` as the supported composition surface for embedding hosts.
The package root continues to export only the context-bound `ContextEngine`
surface.

The builder takes every dependency and wiring choice at construction:

```python
runtime = build_graph_runtime(
    backend,
    plan_store,
    inbox_store,
    definition,
    policy=...,
    observability=...,
    reconciliation_config=...,
    resource_index=...,
    resource_store=...,
    commit_mirror=...,
    preview_store=...,
    commit_host=...,
    commit_actor=...,
    commit_authorize=...,
)
```

- `backend`, `plan_store`, `inbox_store`, and `definition` may be passed by
  position. Every other argument is keyword-only, so later wiring can be added
  without breaking callers.
- Ports may implement synchronous methods, asynchronous `*_async` methods, or
  both; the runtime bridges them.
- Supplied dependencies are borrowed. The runtime never closes them.
- `commit_authorize` is an asynchronous `(context, access)` callback that
  raises to deny; `commit_actor` returns the acting principal. Both belong to
  the host. Without `commit_authorize` the runtime refuses commit history and
  rollback; without `commit_actor` it records an anonymous `unnamed` actor,
  never the account that owns the process.
- Invalid composition raises `RuntimeCompositionError` during construction.

`GraphRuntime` is not bound to a context. Every operation that addresses
context data takes the context identity per call. The host authenticates the
caller and authorizes the context before calling. The runtime invokes the
host's commit-history authorization callback for access levels known only
during the operation, such as the access a rollback requires; it defines no
product policy of its own.

The supported runtime catalog is the following operations, each with an
`*_async` twin:

```text
status               catalog              resolve              describe
read                 search               record               search_entities
mutate               propose              commit               commit_status
verify_commit        history              quality              inbox_add
inbox_list           inbox_show           inbox_claim          inbox_mark_applied
inbox_mark_rejected  inbox_close          journal_status       commits
commit_show          revert_preview       rollback_preview     apply_preview
disable_rollback     rebuild_commits
```

It also includes the read-only attributes `backend`, `plan_store`,
`inbox_store`, `definition`, `policy`, `reconciliation_config`,
`observability`, and `graph`, each typed by an exported port or value. Other
members, including `workbench`, `commit_service`, and `commit_mirror`, are
composition details without a stability promise. Runtime operations return
their existing result values, such as `GraphMutationCommitResult`, rather than
`ContextEngine` outcomes.

A host that serves documents composes `ResourceFacade` over the runtime with
`ResourceFacade.from_runtime(runtime, *, store, index=None, drain=None)`. The
facade takes the graph service, claim query, snapshot port, and journal from
the runtime, so nothing is set on it afterwards. `store` and `index` are the
instances passed to the builder as `resource_store` and `resource_index`;
`drain` is the host's index drain, which the facade only signals and never
starts or stops. The supported facade operations are:

```text
import_dir           get                  list                 current_manifest
delete               purge_pot            status               index_status
index_build          index_rebuild        export_snapshot      import_snapshot
```

Every facade operation that addresses one context takes it per call.
`status`, `index_status`, and `index_build` called without a context address
the whole store, so a multi-tenant host authorizes those calls separately.
The facade's fields are composition details; `from_runtime` is the supported
constructor.

The builder is exported from the engine-level `potpie_context_engine.api`, not
from `potpie_context_engine.core.api`, because it composes the engine's
default graph-service implementation, which the core contract surface does not
reach.

`GraphExtension` stays internal. The builder accepts a `GraphDefinition`
value, and public extension registration remains deferred under `CE-018`,
`SYS-015`, and ADR-0006.

This decision supersedes only ADR-0006's consequence that existing runtime
code creates no public promise, and only for `build_graph_runtime`,
`GraphRuntime`, and `ResourceFacade` with the catalogs above. ADR-0006's other
deferrals, including extensions and external-host transport, remain in force.

This decision adds no `ContextEngine` methods and decides none. The façade
already carries methods outside the ADR-0008 catalog: `reset_context`; the
resource methods `resource_import`, `resource_get`, `resource_list`,
`resource_rm`, `resource_status`, `resource_index_status`,
`resource_index_build`, and `resource_index_rebuild`; and the commit-history
methods `commit_status`, `verify_commit`, `journal_status`, `commits`,
`commit_show`, `revert_preview`, `rollback_preview`, `apply_preview`,
`disable_rollback`, and `rebuild_commits`. ADR-0008 requires a later public-API
decision before any of them joins the catalog. That decision is recorded as
`OQ-CE-API-002` and is left to its own decision record.

## Authority And Sources

> decision [active]: decision:ADR-0002
> decision [active]: decision:ADR-0003
> decision [active]: decision:ADR-0006
> decision [active]: decision:ADR-0008
> observation [active]: code:potpie/context-engine/src/potpie_context_engine/core/runtime.py@32b8cbbb15e7dccff72890c4f5f9cbb4ddf6aaff
> observation [active]: test:potpie/context-engine/tests/unit/test_public_api.py@32b8cbbb15e7dccff72890c4f5f9cbb4ddf6aaff

## Consequences

- An embedding host composes from `potpie_context_engine.api` without daemon,
  CLI, or root product modules.
- Accepting this decision neither accepts nor rejects the façade methods
  implemented outside the ADR-0008 catalog; `OQ-CE-API-002` stays open until
  its own decision.
- Definition, resource, and commit-history wiring happens at construction.
  Mutating a composed runtime is unnecessary and unsupported.
- Context Engine carries a second compatibility promise beside `ContextEngine`.
  Changing the builder parameters or the runtime catalog requires a later
  decision.
- The runtime and façade catalogs overlap. Both delegate to the same focused
  domain modules and must not diverge in domain semantics.
- Embedding hosts import the pot and auth values `PotInfo`, `SourceInfo`, and
  `PotAggregateStatus` from `potpie/pots/contracts.py` and `AuthIdentity` from
  `potpie/auth/ports/identity.py` in the root `potpie` distribution. Those
  modules load without the daemon, CLI, or local runtime composition. This
  makes such hosts depend on root `potpie`; Context Engine itself still does
  not. Moving these values into Context Engine is not proposed and is revisited
  only if a host must not depend on root `potpie`.
- Omitting the commit callbacks does not grant a default policy: commit
  history and rollback are refused until the host supplies its own.

## Alternatives Considered

### Grow the `ContextEngine` façade instead

Per-context binding forces a multi-context host to construct and cache one
engine per context. The façade's operation groups are implemented only by root
`potpie`, so the host would either depend on Potpie's local runtime or
re-implement every group.

### Decide the façade catalog additions here as well

The façade methods added outside the ADR-0008 catalog serve the
Potpie-hosted path through one bound context, not embedding hosts. Choosing
them has its own options and consequences and, like ADR-0008, calls for a
decision record rather than a contract revision. Folding it in here would make
acceptance of the builder depend on an unrelated choice, which ADR-0006
avoids.

### Add a separate engine-level composer with a narrower result type

This duplicates `GraphRuntime` behind a translation layer without removing the
need for a stable multi-context surface. A narrower composer can still be
introduced later on top of the same builder.

### Export the builder from the core API

The core contract surface would then reach the engine's default graph-service
implementation, which it deliberately does not import.

### Publish `GraphExtension` with the builder

This would create the public extension-registration contract that `CE-018`
and `SYS-015` defer.

## Affected Behavior IDs

- Clarify `CE-001` to separate the context-bound façade from the
  embedding-host composition surface.
- Clarify `CE-012` to scope typed outcomes to `ContextEngine` façade
  operations.
- Add `CE-035` for the supported builder, runtime, and resource-facade
  exports.
- Add `CE-036` for construction-time wiring without post-construction
  mutation.
- Add `CE-037` for an explicit per-call context identity.
- Add `CE-038` for composition without daemon, CLI, or root product modules.

## Follow-Up Change Records

- `SPEC-CHANGE-0013`

## Acceptance

Proposed. Pending a decision by `user:dsantra`. Until accepted, this record
binds nothing and ADR-0006 applies unchanged.
