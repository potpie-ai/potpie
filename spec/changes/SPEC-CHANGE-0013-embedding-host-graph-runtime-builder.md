---
id: SPEC-CHANGE-0013
title: Support A Graph-Runtime Builder For Embedding Hosts
kind: spec-change
change_status: proposed
spec_id: SPEC-CONTEXT-ENGINE
from_revision: 1
from_ref: 047cbe067c9c726e7e14f066675453372d8a8406
to_revision: 2
change_type: normative
initiated_by: team:potpie
authored_by:
  - team:potpie
accepted_by: []
accepted_at: null
---

# SPEC-CHANGE-0013: Support A Graph-Runtime Builder For Embedding Hosts

## Intent

Give embedding hosts, which bring their own graph backend and plan and inbox
stores and serve many contexts from one process, one supported composition
surface. They should no longer import internal module paths or mutate a
composed runtime, and the context-bound `ContextEngine` façade should not grow
to serve them.

## Provenance Sources

> decision [active]: decision:ADR-0002
> decision [active]: decision:ADR-0003
> decision [active]: decision:ADR-0006
> decision [active]: decision:ADR-0008
> decision [active]: decision:ADR-0013
> observation [active]: code:potpie/context-engine/src/potpie_context_engine/core/runtime.py@32b8cbbb15e7dccff72890c4f5f9cbb4ddf6aaff
> observation [active]: test:potpie/context-engine/tests/unit/test_public_api.py@32b8cbbb15e7dccff72890c4f5f9cbb4ddf6aaff

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| clarify | CE-001 | CE-001 | Keep the façade finite and thin while stating that the embedding-host composition surface is separate from it. |
| clarify | CE-012 | CE-012 | Scope typed outcomes to `ContextEngine` façade operations; graph-runtime operations keep their existing result values. |
| add | — | CE-035 | Export the supported graph-runtime builder and runtime type from the engine API. |
| add | — | CE-036 | Complete composition through construction arguments without post-construction mutation. |
| add | — | CE-037 | Require an explicit context identity on every context-addressing runtime operation. |
| add | — | CE-038 | Keep embedding-host composition independent of daemon, CLI, and root product modules. |

## Semantic Diff

Revision 2 would add one supported composition surface beside the
`ContextEngine` façade and leave the façade, its ADR-0008 catalog, and every
other behavior unchanged. Until acceptance, `spec/modules/context-engine.md`
stays at revision 1; this record carries the proposed text.

Proposed behavior nodes:

```text
CE-001 [active]: Context Engine MUST expose `ContextEngine` as a finite, thin public façade whose explicitly named methods represent context-domain operations; the embedding-host composition surface defined by CE-035 is separate from that façade and adds no façade methods.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002
  > decision [active]: decision:ADR-0013
  ~ potpie/context-engine/src/potpie_context_engine/api.py

CE-012 [active]: Every `ContextEngine` façade operation MUST return a typed transport-neutral domain value or a typed DomainError, DependencyError, or EngineLifecycleError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  > decision [active]: decision:ADR-0013
  @ CE-011
  @ SYS-007

CE-035 [active]: Context Engine MUST export `build_graph_runtime` and `GraphRuntime` from `potpie_context_engine.api` as the supported composition surface for an embedding host that supplies its own graph backend and plan and inbox stores.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0013
  @ CE-001
  @ CE-003
  ~ potpie/context-engine/src/potpie_context_engine/api.py

CE-036 [active]: The graph-runtime builder MUST accept every host-supplied dependency and wiring choice, including graph definition, resource, and commit-history wiring, as an explicit construction argument, and composition MUST NOT require the host to mutate the returned runtime or its collaborators.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0013
  @ CE-022
  @ CE-035
  ~ potpie/context-engine/src/potpie_context_engine/core/runtime.py

CE-037 [active]: Every supported graph-runtime operation that addresses context data MUST take its context identity as an explicit per-call argument and MUST NOT derive it from process-global or active-selection state.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  > decision [active]: decision:ADR-0013
  @ CE-009
  @ CE-035

CE-038 [active]: An embedding host MUST be able to compose and use the supported graph runtime without importing Potpie daemon, CLI, or root product modules.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0013
  @ CE-002
  @ CE-035
```

Proposed prose changes in the module:

- Purpose: after the façade sentence, add "It also exposes one supported
  composition surface for embedding hosts that bring their own graph backend
  and stores."
- Actors And Permissions: add the row "Embedding host | Supplies its own graph
  backend and stores, composes a multi-context graph runtime through the
  supported builder, and authenticates callers and authorizes each context
  before invoking it".
- Data And State Model: add "A graph runtime holds no context identity. It
  shares one definition, policy, and set of borrowed dependencies across every
  context it serves."
- Scope And Non-Goals: replace "Parsing and public extensions remain outside
  scope" with "Parsing and public extensions remain outside scope; the
  embedding-host builder accepts a `GraphDefinition` value and publishes no
  extension registration."

The exact builder parameters and the supported runtime catalog are recorded in
ADR-0013, as ADR-0008 records the façade catalog.

## Compatibility, Security, And Failure Impact

The export is additive. In the builder, every argument after `definition`
becomes keyword-only; existing callers already pass those arguments by
keyword. The package root keeps exporting only the `ContextEngine` surface.

Commit-history authorization fails closed. An embedding host that omits
`commit_authorize` gets a runtime that refuses commit history, rollback
previews and their application; one that omits `commit_actor` records an
anonymous `unnamed` actor rather than the account that owns the process.
ADR-0013 requires a host that serves commit history to pass both callbacks.
The builder rejects a non-callable actor or authorization callback during
construction.

Embedding hosts import the pot and auth values `PotInfo`, `SourceInfo`, and
`PotAggregateStatus` from `potpie/pots/contracts.py` and `AuthIdentity` from
`potpie/auth/ports/identity.py` in the root `potpie` distribution. Those
imports load no daemon, CLI, or runtime composition modules and start no
daemon. The dependency runs from the embedding host to root `potpie`, never
from Context Engine to root `potpie`, so `CE-026`, `SYS-002`, and the engine
import lock are unchanged.

Invalid composition raises `RuntimeCompositionError` during construction and
never yields a usable runtime. An exception raised by a host-supplied
authorization callback propagates to the caller unchanged.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| ADR-0006 | ADR-0013 supersedes only its "no public promise" consequence, and only for `build_graph_runtime` and `GraphRuntime`. | The immutable accepted ADR is not edited; its other deferrals remain. | team:potpie |
| ADR-0002, ADR-0008 | — | The façade, its construction model, and its method catalog are unchanged. | team:potpie |
| CE-005 through CE-010 | — | They govern `ContextEngine` instances. A graph runtime is not one; it holds no context identity, and CE-037 requires the identity per call. | team:potpie |
| CE-018, SYS-015 | — | `GraphExtension` stays internal and no extension registration is published. | team:potpie |
| CE-020, CE-021 | — | They govern the façade. The runtime catalog is enumerated in ADR-0013, and its attributes are typed by exported ports. | team:potpie |
| CE-022, CE-023, SYS-018 | — | The builder treats every supplied dependency as borrowed and never closes it. | team:potpie |
| CE-025, CE-029, CE-030 | — | They classify `ContextEngine` outcomes through their dependency on CE-012. | team:potpie |
| SYS-001, SYS-016 | — | Embedding hosts are direct compatible hosts, not the Potpie-hosted path, and the builder is an in-process library API, not a transport. | team:potpie |
| SYS-008 | — | The host authenticates and authorizes before calling. The commit-history callback is host code that the runtime invokes for access known only during the operation. | team:potpie |
| SPEC-GLOSSARY | — | "Host" already covers software that constructs Context Engine and supplies its dependencies. | team:potpie |
| Engine API and builder | Export `build_graph_runtime` and `GraphRuntime`, make wiring after `definition` keyword-only, document the stable members, and validate callback arguments. | — | team:potpie |
| Engine tests | Restore the public runtime conformance suite, test composition from the supported API with host-owned async stores and commit policy, and lock the export layer. | — | team:potpie |
| Root tests | Lock that the pot and auth values import without daemon, CLI, or runtime modules. | — | team:potpie |
| Conformance validator | On acceptance, raise the covered active-behavior count from 195 to 199. | Not changed while proposed. | team:potpie |

## Conformance Invalidation

None while proposed. On acceptance, the current Context Engine record derives
stale because it pins revision 1. A successor record must verify `CE-001`,
`CE-012`, and `CE-035` through `CE-038` against the accepted revision. Records
that list Context Engine as a related contract derive stale on the same
transition.

## Validation

```text
Structural: passed; scripts/validate_conformance_history.py with the module unchanged at revision 1
Semantic: proposed; each added node is one obligation, and the two clarifications keep their façade obligations
Provenance: proposed; authority edges await user:dsantra
Historical mutation: reviewed; revision would advance 1 to 2, and CE-035 through CE-038 are unused IDs
Dependency/consistency: reviewed against SPEC-SYSTEM and SPEC-GLOSSARY; see the impact review
Fresh-agent reconstruction: pending acceptance review
Independent conformance state: revision 1 record unchanged; an implementation and tests accompany this proposal without a conformance claim
```

## Acceptance

Proposed. Pending acceptance by `user:dsantra`. Context Engine revision 1
remains the binding contract until then. Merging the accompanying
implementation waits for acceptance. This record creates no implementation or
verification claim.
