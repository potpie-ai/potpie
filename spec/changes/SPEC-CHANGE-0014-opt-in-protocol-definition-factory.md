---
id: SPEC-CHANGE-0014
title: Support An Opt-In Protocol Definition Factory
kind: spec-change
change_status: proposed
spec_id: SPEC-CONTEXT-ENGINE
from_revision: 2
from_ref: null
to_revision: 3
change_type: normative
initiated_by: team:potpie
authored_by:
  - team:potpie
accepted_by: []
accepted_at: null
---

# SPEC-CHANGE-0014: Support An Opt-In Protocol Definition Factory

## Intent

Let a host opt into the protocol ontology, which records versioned protocol,
message and field definitions and reads them back through a bounded
`protocols.message_context` view, by passing one supported graph definition to
the graph-runtime builder. Nothing turns it on by default, and no
extension-registration contract is published.

This record builds on the revision 2 that SPEC-CHANGE-0013 proposes: the value
this factory returns is a `definition` argument to the builder that CE-035 and
CE-036 make supported. `from_ref` stays empty until that revision is accepted;
acceptance pins it.

## Provenance Sources

> decision [active]: decision:ADR-0006
> decision [active]: decision:ADR-0013

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| clarify | CE-018 | CE-018 | State that a supported factory returning a complete first-party definition is not an extension-registration contract. |
| add | — | CE-039 | Export an opt-in protocol definition factory from the engine API. |
| add | — | CE-040 | Export the identity helpers that writers need to address protocol entities. |
| add | — | CE-041 | Keep the engine from selecting the protocol ontology on its own. |
| add | — | CE-042 | Keep a runtime composed without the ontology free of protocol behavior. |

## Semantic Diff

Revision 3 would add one optional, first-party graph definition beside the
default one, and leave the `ContextEngine` façade, its ADR-0008 catalog, the
builder parameters, and `DEFAULT_GRAPH_DEFINITION` unchanged. Until acceptance,
`spec/modules/context-engine.md` stays at revision 1; this record carries the
proposed text.

Proposed behavior nodes:

```text
CE-018 [active]: This revision MUST NOT expose a public plugin, extension-registration, or manifest contract; a supported factory that returns a complete first-party `GraphDefinition` value, such as the one CE-039 defines, is not an extension-registration contract.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006
  ~ potpie/context-engine/src/potpie_context_engine/__init__.py

CE-039 [active]: Context Engine MUST export `protocols_definition` from `potpie_context_engine.api` as a supported factory that returns a new `GraphDefinition` extending a supplied base definition, `DEFAULT_GRAPH_DEFINITION` when none is supplied, with the versioned protocol ontology: the `Protocol`, `ProtocolMessage`, and `ProtocolField` entity types, their six relation types, and the `protocols.message_context` view with its reader.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0013
  @ CE-018
  @ CE-036
  ~ potpie/context-engine/src/potpie_context_engine/api.py

CE-040 [active]: Context Engine MUST export `protocol_entity` and `protocol_entity_key` from `potpie_context_engine.api` as the supported helpers that derive a protocol entity's key deterministically from its versioned, typed, case-preserving identity properties.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0013
  @ CE-039
  ~ potpie/context-engine/src/potpie_context_engine/api.py

CE-041 [active]: Context Engine MUST NOT select the protocol ontology on its own: `DEFAULT_GRAPH_DEFINITION`, the builder's default definition, and engine environment or configuration MUST leave it out, so that it is present only in a runtime whose host passed a definition that carries it.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006
  > decision [active]: decision:ADR-0013
  @ CE-036
  @ CE-039

CE-042 [active]: A graph runtime composed with a definition that lacks the protocol ontology MUST NOT advertise protocol entity types, relation types, views, or include families, and MUST NOT run protocol-specific mutation validation, journal restoration checks, or resource source protection.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006
  @ CE-041
```

Proposed prose changes in the module:

- Scope And Non-Goals: after the sentence SPEC-CHANGE-0013 proposes ("the
  embedding-host builder accepts a `GraphDefinition` value and publishes no
  extension registration"), add "One first-party optional definition, the
  protocol ontology, is available through a supported factory and is absent
  unless a host passes it to the builder."
- Data And State Model: add "Protocol data written by a runtime that carried
  the protocol ontology stays stored when a later composition omits it. That
  runtime neither reads, validates, nor protects it, and refuses new protocol
  writes as unknown types."

## Compatibility, Security, And Failure Impact

The exports are additive, and the default is unchanged: a host that does not
pass the definition sees exactly the revision 2 contract. Default resolve
recipes never include the `protocols` family; a caller asks for it explicitly.

Extending a base that already carries the protocol ontology raises
`GraphDefinitionError`, so a host cannot compose it twice. A runtime composed
without the ontology refuses protocol writes as unknown entity types and
predicates.

Turning the ontology off keeps its data but also lifts its source protection:
the resource layer no longer refuses refreshing or removing a source that
stored protocol claims cite. If the host later composes with the ontology
again, claims whose evidence is gone or changed report unverified, never
complete coverage.

The Potpie local runtime exposes the opt-in as the persisted product
configuration key `graph.protocols` (`on` or `off`, off when absent), read once
when the runtime is composed. A running daemon therefore keeps the definition
it started with until it restarts. Embedding hosts opt in by passing the
factory's value to the builder themselves.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| SPEC-CHANGE-0013, CE-035, CE-036 | — | The factory's value is a `definition` argument to the builder they define. This change depends on their acceptance and changes neither. | team:potpie |
| ADR-0006 | — | Extension registration stays deferred: the factory returns one closed first-party definition and accepts only a base definition, not an extension. | team:potpie |
| CE-018 | Clarify that a supported first-party definition factory is not extension registration. | — | team:potpie |
| SYS-015 | — | No extension, plugin-registration, or manifest contract is published. | team:potpie |
| CE-001, ADR-0008 | — | No `ContextEngine` façade method is added. | team:potpie |
| CE-026, SYS-002 | — | The factory lives in Context Engine and imports nothing from root `potpie`. | team:potpie |
| SPEC-POTPIE-CAPABILITIES | — | `graph.protocols` is persisted local configuration, which PCAP-002 already assigns to the configuration capability; composition passes the definition explicitly. | team:potpie |
| SPEC-CLI, SPEC-DAEMON | — | No contract node names configuration keys. `config set` validates the key's value and reports that a restart applies it; a daemon applies composition inputs when it starts, as before. | team:potpie |
| SPEC-GLOSSARY | — | "Host" and "definition" need no new terms. | team:potpie |
| Engine API | Export `protocol_entity` and `protocol_entity_key` beside `protocols_definition`; point the view's identity guidance at the API path. | — | team:potpie |
| Engine tests | Lock the three exports and the base-only default; keep protocol acceptance and backend conformance on the explicit definition. | — | team:potpie |
| Root tests | Composition is off by default, on with the key, and an explicit definition wins; a CLI journey covers off, on, and off again. | — | team:potpie |
| Conformance validator | On acceptance, raise the covered active-behavior count by four on top of the four SPEC-CHANGE-0013 adds: from 195 to 203 with only those two accepted. SPEC-CHANGE-0015 adds its own four. | Not changed while proposed. | team:potpie |

## Conformance Invalidation

None while proposed. On acceptance, the current Context Engine record derives
stale because it pins an earlier revision. A successor record must verify
`CE-018` and `CE-039` through `CE-042` against the accepted revision. Records
that list Context Engine as a related contract derive stale on the same
transition.

## Validation

```text
Structural: passed; scripts/validate_conformance_history.py with the module unchanged at revision 1
Semantic: proposed; each added node is one obligation, and the CE-018 clarification keeps its prohibition
Provenance: proposed; authority edges await user:dsantra
Historical mutation: reviewed; revision would advance 2 to 3 after SPEC-CHANGE-0013, and CE-039 through CE-042 are unused IDs
Dependency/consistency: reviewed against SPEC-SYSTEM, SPEC-POTPIE-CAPABILITIES, and SPEC-GLOSSARY; see the impact review
Fresh-agent reconstruction: pending acceptance review
Independent conformance state: revision 1 record unchanged; an implementation and tests accompany this proposal without a conformance claim
```

## Alternatives Considered

### Publish a `GraphExtension` registration hook

A host could then register its own extensions. That is the public
extension-registration contract CE-018 and SYS-015 defer, and the only
extension a known host needs is this first-party one.

### Turn the protocol ontology on by default

Every host would advertise protocol types and views whether or not it holds
protocol sources, and the default graph contract would grow without a
decision. An explicit definition keeps the default unchanged.

### Keep the factory internal

Hosts already import it, and its identity helpers, from internal module paths.
Supporting them gives those imports a compatibility promise instead.

## Acceptance

Proposed. Pending acceptance by `user:dsantra`, after SPEC-CHANGE-0013.
Context Engine revision 1 remains the binding contract until then. Merging the
accompanying implementation waits for acceptance. This record creates no
implementation or verification claim.
