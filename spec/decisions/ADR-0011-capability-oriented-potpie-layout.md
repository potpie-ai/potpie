---
id: ADR-0011
title: Choose A Capability-Oriented Potpie Layout
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-23T14:53:44+05:30
---

# ADR-0011: Choose A Capability-Oriented Potpie Layout

## Context

`SPEC-POTPIE-CAPABILITIES` revision 1 at
`b23b6cb9158e6b929f7d5f01c4dc0ca62a69f1df` requires root Potpie source to be
owned by explicit capabilities, removes the branch-local `potpie.product`
namespace, keeps cross-capability composition explicit, and preserves
supported behavior. The contract intentionally does not choose exact
implementation paths.

At planning ref `930632fb222beba87908389a2f326eb4825d77e6`, configuration,
pot/source lifecycle, skills, setup, local identity, and agent-context
composition are grouped beneath `potpie.product` by architectural layer. Root
configuration and skills also obtain Potpie home resolution from a Context
Engine adapter, even though Context Engine must remain independently packaged
and must not import root Potpie. Local identity is separate from the existing
`potpie.auth` owner, and runtime composition already provides the production
assembly point for root and engine services.

The relocation needs one exact destination for every in-scope source category
before imports move. It must not replace `product` with another generic bucket,
create a second authentication owner, turn package initializers into service
locators, or imply that branch-local implementation imports are public APIs.

## Decision

### Selected source layout

Root Potpie capability source will use this layout:

```text
potpie/
  agent_context.py

  auth/
    ports/
      identity.py
    adapters/
      local_identity.py

  config/
    contracts.py
    local.py
    local_paths.py

  pots/
    contracts.py
    local_service.py
    local_store.py

  setup/
    contracts.py
    orchestrator.py
    installation.py
    local_installer.py
    state.py
    flat_file_state.py

  skills/
    contracts.py
    manager.py
    catalog.py
    targets.py
    installer.py
```

Each new capability package has an `__init__.py` containing package
documentation only. Package initializers do not aggregate contracts, expose a
broad re-export API, select implementations, or recreate `potpie.product`
through aliases.

### Configuration and Potpie home ownership

`potpie.config.contracts` owns the configuration contract and DTOs.
`potpie.config.local` owns the local persisted implementation and redaction
behavior. `potpie.config.local_paths` owns root Potpie resolution of
`$CONTEXT_ENGINE_HOME` and the `~/.potpie` default.

Context Engine retains its independent equivalent local-path helper. The two
distributions may implement the same environment precedence and filesystem
default, but Context Engine production code does not import the root helper and
root capability code does not use a Context Engine adapter to obtain its home.

### Pots ownership

`potpie.pots.contracts` owns pot and source DTOs and protocols.
`potpie.pots.local_service` owns lifecycle policy and repository-default
behavior. `potpie.pots.local_store` owns local persistence. The existing
`config.json`, `pots.json`, temporary-write, environment-precedence, and
repository-default behaviors remain unchanged.

### Skills ownership

`potpie.skills.contracts` owns skill DTOs and protocols.
`potpie.skills.manager` owns install, update, remove, and validation policy.
`potpie.skills.catalog`, `potpie.skills.targets`, and
`potpie.skills.installer` own the built-in catalog, agent-specific target
selection, and filesystem installation mechanism respectively. Catalog
identities, catalog versions, target outcomes, support-file installation, and
manager dependency injection remain unchanged.

### Setup ownership

`potpie.setup.contracts` and `potpie.setup.orchestrator` own setup requests,
results, step policy, ordering, preview, and execution. Installation and setup
state are setup lifecycle seams rather than independent root capabilities, so
their contracts and local mechanisms live in `potpie.setup.installation`,
`potpie.setup.local_installer`, `potpie.setup.state`, and
`potpie.setup.flat_file_state`.

The relocation preserves the existing seam plan, hard and soft failure policy,
skip behavior, idempotency, migration behavior, observer isolation, and error
folding. It does not redesign setup or generalize its mechanisms for unrelated
capabilities.

### Authentication ownership

`AuthIdentity` and `AuthService` move into
`potpie.auth.ports.identity`. `LocalAuthService` moves into
`potpie.auth.adapters.local_identity`. This extends the existing authentication
capability instead of creating another root owner.

Local identity lifecycle remains distinct from integration credential storage
and provider authentication. Sharing the existing `potpie.auth` capability
does not merge their persistence, lifecycle, or policy.

### Cross-capability agent context

`AgentContextService` moves to `potpie.agent_context`. It remains outside an
individual capability package because it combines Context Engine graph
behavior with pot and skill services. The relocation preserves its delegation,
status aggregation, guidance, and public Context Engine DTO usage.

### Runtime composition and dependency direction

`potpie.runtime.composition` remains the only production assembly point that
selects concrete implementations across root capabilities and constructs the
separate root and Context Engine service groups. Capability modules use
explicit constructor dependencies and module imports; they do not introduce a
dynamic registry, global service locator, or alternate production composition
path.

Dependency direction remains root Potpie to Context Engine. Context Engine
production code imports no root `potpie` namespace, and the independently built
Context Engine distribution remains importable without root Potpie installed.

### Migration discipline

The relocation proceeds only after path-independent behavior characterization,
then through four production slices:

1. Configuration and pots, including root-owned home resolution.
2. Skills.
3. Setup and local identity.
4. Agent-context composition followed by complete `potpie.product` deletion.

Every moved old path disappears in the same slice as its direct callers move.
No forwarding module, compatibility re-export, `sys.modules` registration, or
dynamic alias is introduced. If a slice needs expansion to stay green, it may
include only direct import dependents.

Permanent architecture and packaging tests follow the relocation and reject
the deleted namespace, reverse imports, alternate composition, service-group
mixing, stale distribution metadata, and entry-point drift.

No `app`, `application`, `product`, `control_plane`, or `capabilities` source
umbrella is introduced. Conceptual references to the Potpie product remain
valid in accepted specifications and user-facing documentation; this decision
removes only the generic source namespace.

## Authority And Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0001
> decision [active]: decision:ADR-0002
> decision [active]: decision:ADR-0003
> observation [active]: code:potpie/product@930632fb222beba87908389a2f326eb4825d77e6
> observation [active]: code:potpie/auth@930632fb222beba87908389a2f326eb4825d77e6
> observation [active]: code:potpie/runtime/composition.py@930632fb222beba87908389a2f326eb4825d77e6
> observation [active]: code:pyproject.toml@930632fb222beba87908389a2f326eb4825d77e6

## Consequences

- Source paths state capability ownership directly and avoid a replacement
  umbrella with the same ambiguity as `product`.
- Root Potpie and Context Engine retain equivalent local-path behavior without
  introducing a reverse package dependency.
- Setup mechanisms remain colocated with the lifecycle that currently owns
  them; extracting a reusable installer or state capability would require a
  later decision backed by another consumer.
- Cross-capability composition stays visible at `potpie.agent_context` and
  `potpie.runtime.composition` instead of being hidden inside a capability.
- Repository imports and package manifests must change together, making the
  migration larger than a namespace rename but mechanically reviewable.
- Deleting old paths without forwarding means uncontracted branch-local Python
  imports break immediately when their owning slice moves.
- Architecture tests become responsible for keeping ownership, distribution
  independence, and composition direction from drifting after relocation.

## Alternatives Considered

### Retain `potpie.product`

This keeps current imports stable but preserves a generic bucket that does not
answer which capability owns a contract, policy, or persistence mechanism.

### Rename the umbrella to `app`, `application`, `control_plane`, or `capabilities`

This changes vocabulary without improving ownership. A new umbrella would
remain a default location for unrelated root concerns and would recreate the
same layer-first organization.

### Keep ports, services, and adapters as root architectural layers

This makes a single capability span multiple distant directories and turns
technical role into the primary navigation axis. The selected layout keeps
technical roles explicit within the capability that owns them.

### Share root home resolution from the Context Engine package

This avoids a small equivalent helper but gives a domain distribution
ownership of root product configuration and encourages root-to-adapter
coupling. The selected duplicate is deliberately small and keeps each wheel
independent.

### Put `AgentContextService` under pots or skills

Either choice would make one capability appear to own orchestration that also
depends on another capability and Context Engine graph behavior.

### Preserve `potpie.product` through forwarding modules

This would make a branch-local implementation path look supported, leave two
names for every owner, and weaken the permanent deletion invariant.

## Affected Behavior IDs

| Behavior | Decision effect |
|---|---|
| `PCAP-001` | Selects capability packages with no replacement umbrella. |
| `PCAP-002` | Selects `potpie.config`, including root-owned local paths. |
| `PCAP-003` | Selects `potpie.pots` for pot/source contracts, policy, and persistence. |
| `PCAP-004` | Selects `potpie.skills` for catalog, policy, targets, and installation. |
| `PCAP-005` | Selects `potpie.setup` for orchestration, installation, state, and migration seams. |
| `PCAP-006` | Selects existing `potpie.auth` ports and adapters for local identity. |
| `PCAP-007` | Selects top-level `potpie.agent_context` for cross-capability composition. |
| `PCAP-008` | Retains `potpie.runtime.composition` as the sole concrete production assembly point. |
| `PCAP-009` | Selects independent root and Context Engine local-path helpers with no reverse import. |
| `PCAP-010` | Requires deletion without aliases, forwarding, or compatibility re-exports. |
| `PCAP-011` | Constrains every relocation slice to preserve supported behavior. |
| `PCAP-012` | Permits deletion without a public-Python compatibility shim. |

This section is a review aid. The accepted contract remains canonical and is
not modified by this decision.

## Accepted Contract Impact Review

| Accepted artifact | Required revision | No-change reason |
|---|---|---|
| SPEC-PROCESS revision 1 | — | The decision follows the existing authorship and acceptance workflow. |
| SPEC-GLOSSARY revision 1 | — | No accepted runtime or ownership term changes meaning. |
| SPEC-PRODUCT revision 1 | — | Conceptual product outcomes and terminology remain unchanged. |
| SPEC-SYSTEM revision 1 | — | Runtime composition and root-to-engine dependency direction remain unchanged. |
| SPEC-POTPIE-CAPABILITIES revision 1 | — | This decision selects the concrete layout required by the accepted ownership behaviors without changing them. |
| SPEC-CONTEXT-ENGINE revision 1 | — | Context Engine remains independently importable and gains no root dependency. |
| SPEC-POTPIE-RESOURCE-MANAGER revision 2 | — | Lease, authorization, and resource ownership are unaffected. |
| SPEC-DAEMON revision 1 | — | Daemon protocol, lifecycle, discovery, and operations are unaffected. |
| SPEC-CLI revision 1 | — | Command inventory, presentation, streams, and exits are unaffected. |
| ADR-0001 through ADR-0010 | — | No prior accepted decision is superseded or contradicted. |
| Existing conformance records | — | This accepted decision makes no implementation or verification claim. |

## Follow-Up Change Records

None. This decision does not mutate an accepted contract.

## Acceptance

Accepted by `user:dsantra` at `2026-08-23T14:53:44+05:30` after authorized
review of the exact layout, alternatives, contract impact, and migration
discipline. Acceptance creates no implementation or verification claim.
