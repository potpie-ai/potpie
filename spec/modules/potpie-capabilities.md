---
id: SPEC-POTPIE-CAPABILITIES
title: Potpie Capability Ownership Contract
kind: module-spec
revision: 1
maturity: accepted
owners:
  - team:potpie
depends_on:
  - SPEC-PRODUCT
  - SPEC-SYSTEM
change_ref: SPEC-CHANGE-0010
---

# Potpie Capability Ownership Contract

## Purpose

This contract defines how root Potpie application concerns are owned and
composed around the independently usable Context Engine. It makes the source
boundary discoverable by capability without creating a generic product-layer
namespace or changing observable Potpie behavior.

## Ownership And Boundaries

| Capability | Owns | Excludes |
|---|---|---|
| Configuration | Potpie home resolution, persisted local configuration, and public-value redaction | Pot/source state, skill installation, Context Engine settings ownership |
| Pots | Pot and source contracts, lifecycle policy, repository defaults, and local persistence | Graph-domain semantics, setup sequencing, agent presentation |
| Skills | Built-in catalog, validation, install/update/remove policy, and agent-specific targets | Setup sequencing, graph-domain behavior, runtime assembly |
| Setup | First-run orchestration, installation seam, state provisioning, and migrations | Capability implementation selection outside setup, CLI presentation |
| Authentication | Local identity lifecycle alongside existing credential contracts and adapters | Context Engine authorization policy, duplicate authentication ownership |
| Agent context composition | Product-facing composition of graph, pot, and skill status | Ownership of the underlying capabilities or graph-domain semantics |
| Runtime composition | Selection and wiring of concrete root capability implementations | Capability policy, dynamic service lookup, Context Engine domain semantics |

Context Engine remains an independently importable distribution and owns no
root Potpie capability.

## Scope And Non-Goals

This revision governs root Potpie capability ownership, composition direction,
the removal of the branch-local `potpie.product` namespace, and behavior
preservation during that relocation.

It does not define new CLI commands, daemon operations, persistence formats,
setup steps, authentication behavior, skill catalog entries, pot/source
semantics, Context Engine behavior, or a general plugin model. It does not make
the replacement implementation modules a supported public Python API.

## Terminology

- **Capability**: one named root Potpie concern whose contract, policy, and
  concrete local mechanisms are discoverable through its source namespace.
- **Root Potpie**: the `potpie` distribution surrounding, composing, and
  delivering the independently usable Context Engine.
- **Cross-capability composition**: a root service or composition root that
  combines two or more capability contracts without transferring their
  ownership.
- **Generic architectural bucket**: a source namespace named for a broad layer,
  such as product or control plane, rather than for an owned capability.

## Actors And Permissions

| Actor | Interaction |
|---|---|
| Potpie maintainer | Changes capability internals while preserving accepted ownership and behavior boundaries |
| Runtime composition | Selects concrete implementations and wires cross-capability dependencies |
| Context Engine integrator | Imports and operates Context Engine without importing root Potpie |
| CLI or daemon caller | Observes the same supported behavior before and after internal relocation |

## Normative Requirements

PCAP-001 [active]: Root Potpie application source MUST be organized by explicitly owned capabilities rather than by a generic product-layer or control-plane namespace.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/product@930632fb222beba87908389a2f326eb4825d77e6

PCAP-002 [active]: The configuration capability MUST own root Potpie home resolution and persisted local configuration behavior.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/product/services/config.py@930632fb222beba87908389a2f326eb4825d77e6
  @ PCAP-001

PCAP-003 [active]: The pots capability MUST own pot and source contracts, lifecycle policy, repository defaults, and local persistence.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/product/services/pot_management.py@930632fb222beba87908389a2f326eb4825d77e6
  @ PCAP-001

PCAP-004 [active]: The skills capability MUST own its catalog, command-snippet validation, install/update/remove policy, and agent-specific targets.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/product/services/skills.py@930632fb222beba87908389a2f326eb4825d77e6
  @ PCAP-001

PCAP-005 [active]: The setup capability MUST own first-run orchestration together with its installation, state-provisioning, and migration seams.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/product/services/setup.py@930632fb222beba87908389a2f326eb4825d77e6
  @ PCAP-001

PCAP-006 [active]: Root Potpie local identity lifecycle MUST extend the existing authentication capability rather than create a second authentication owner.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/auth@930632fb222beba87908389a2f326eb4825d77e6
  @ PCAP-001

PCAP-007 [active]: A service that composes graph behavior with multiple root Potpie capabilities MUST remain outside every individual capability namespace.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/product/services/agent_context.py@930632fb222beba87908389a2f326eb4825d77e6
  @ PCAP-003
  @ PCAP-004

PCAP-008 [active]: Root runtime composition MUST be the production assembly point that selects concrete implementations across Potpie capabilities.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/runtime/composition.py@930632fb222beba87908389a2f326eb4825d77e6
  @ PCAP-002
  @ PCAP-003
  @ PCAP-004
  @ PCAP-005
  @ PCAP-006
  @ SYS-004

PCAP-009 [active]: Context Engine production code MUST NOT import root Potpie capability code.
  > authority [active]: user:dsantra
  @ SYS-002
  @ CE-026

PCAP-010 [active]: The final root Potpie source tree MUST NOT contain the `potpie.product` namespace, a forwarding alias for it, or a compatibility re-export from it.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/product@930632fb222beba87908389a2f326eb4825d77e6
  @ PCAP-001

PCAP-011 [active]: Relocating root Potpie capabilities MUST preserve supported CLI, daemon, setup, configuration, pot, source, skill, persistence, and Context Engine behavior.
  > authority [active]: user:dsantra
  @ PROD-004
  @ SYS-011

PCAP-012 [active]: The branch-local `potpie.product.*` implementation imports MUST NOT be treated as a supported public Python compatibility contract.
  > authority [active]: user:dsantra
  > observation [active]: code:potpie/product@930632fb222beba87908389a2f326eb4825d77e6
  @ PCAP-010

## Data And State Model

This contract introduces no data or persistent-state model. Existing
`config.json`, `pots.json`, repository-default, skill-installation, daemon, and
Context Engine state remain governed by their current behavior and adapters.

## State Machines

Not applicable: source ownership has no runtime lifecycle state machine. Setup,
daemon, authentication, and Context Engine state transitions remain governed by
their existing contracts and behavior.

## Interfaces

Capability contracts and implementations are imported by explicit module path.
Capability package initializers do not establish aggregate service locators or
broad re-export APIs. Runtime composition supplies concrete dependencies to the
existing root and engine service groups.

## Invariants

The dependency direction is root Potpie to Context Engine. Capability
relocation does not permit the reverse dependency, combine root and engine
service groups, or transfer graph-domain behavior into a root capability.

## Consistency, Ordering, And Idempotency

The relocation changes source ownership only. Existing setup ordering,
configuration write replacement, pot/source persistence, skill operation
idempotency, and daemon/runtime coordination remain unchanged under
`PCAP-011`.

## Failure Modes And Recovery

Missing package data, stale imports, reverse Context Engine imports, or
partially relocated capability ownership are conformance failures. Recovery is
to complete or revert the affected capability slice; compatibility forwarding
through `potpie.product` is prohibited by `PCAP-010`.

## Security And Privacy

Secret redaction, credential persistence, local identity behavior, daemon
authentication, authorization, and Context Engine trust boundaries remain
unchanged. Moving identity lifecycle into the existing authentication
capability does not combine credential stores or weaken their ownership seams.

## Observability And Auditability

No new runtime telemetry or audit event is introduced. Git history, accepted
contract identity, architecture tests, package inventories, and conformance
records provide change evidence.

## Performance And Limits

No performance or capacity behavior is added. The relocation must not add a
runtime compatibility router, dynamic service lookup, or alternate composition
path.

## Compatibility, Migration, And Rollout

Repository callers migrate capability by capability in green commits. A moved
old path disappears in the same commit as its callers move; no forwarding shim
is introduced. The final implementation deletes the entire branch-local
`potpie.product` namespace and updates root distribution manifests.

## Examples

- Configuration and pot implementations can depend on Context Engine contracts,
  but Context Engine cannot import either root capability.
- Agent context composition can join graph, pot, and skill status without
  becoming part of pots, skills, or Context Engine.
- A repository import of a moved `potpie.product.*` path is migrated to the
  owning capability rather than preserved through an alias.

## Acceptance Criteria

- `PCAP-001` through `PCAP-008`: a fresh reader can assign every in-scope root
  concern to one capability or the explicit cross-capability composition seam.
- `PCAP-009`: dependency rules can reject any Context Engine import of root
  Potpie.
- `PCAP-010` and `PCAP-012`: migration and compatibility scope are explicit and
  admit no forwarding namespace.
- `PCAP-011`: behavior-preservation evidence is defined independently from
  contract acceptance and can later be pinned in conformance records.
- Structural, semantic, provenance, dependency-consistency, historical, and
  fresh-reader validation have no blocking findings.

## Rationale

Capability ownership makes source placement answer what code does and who owns
it. It preserves the accepted separation between root Potpie and Context Engine
without replacing one generic bucket with another.

## Implementation Notes

The concrete file layout and ordered migration sequence require a separately
accepted architectural decision. This contract makes no implementation or
verification claim.
