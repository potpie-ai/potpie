---
id: SPEC-CHANGE-0010
title: Initialize Potpie Capability Ownership Contract
kind: spec-change
change_status: accepted
spec_id: SPEC-POTPIE-CAPABILITIES
from_revision: 0
from_ref: null
to_revision: 1
change_type: normative
initiated_by: user:dsantra
authored_by:
  - agent:codex
accepted_by:
  - user:dsantra
accepted_at: 2026-08-23T13:21:44+05:30
---

# SPEC-CHANGE-0010: Initialize Potpie Capability Ownership Contract

## Intent

Establish a binding capability-ownership boundary for root Potpie before
relocating the branch-local `potpie.product` implementation namespace. Keep the
existing product, system, Context Engine, Resource Manager, daemon, and CLI
contracts unchanged because the proposal changes source ownership rather than
observable behavior.

## Provenance Sources

> authority [active]: user:dsantra
> observation [active]: code:potpie/product@930632fb222beba87908389a2f326eb4825d77e6
> observation [active]: code:potpie/runtime/composition.py@930632fb222beba87908389a2f326eb4825d77e6
> observation [active]: test:tests/characterization/test_cli_package_boundary.py@930632fb222beba87908389a2f326eb4825d77e6
> observation [active]: test:tests/characterization/test_product_process_surfaces.py@930632fb222beba87908389a2f326eb4825d77e6

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| add | — | PCAP-001 | Require capability-oriented root source ownership. |
| add | — | PCAP-002 | Assign configuration and Potpie home ownership. |
| add | — | PCAP-003 | Assign pot/source lifecycle and persistence ownership. |
| add | — | PCAP-004 | Assign skill catalog, policy, and target ownership. |
| add | — | PCAP-005 | Assign setup orchestration and lifecycle-seam ownership. |
| add | — | PCAP-006 | Place local identity lifecycle with existing authentication. |
| add | — | PCAP-007 | Keep cross-capability agent context composition neutral. |
| add | — | PCAP-008 | Make root runtime composition the production assembly point. |
| add | — | PCAP-009 | Preserve the Context Engine reverse-import prohibition. |
| add | — | PCAP-010 | Require final deletion without compatibility forwarding. |
| add | — | PCAP-011 | Preserve supported observable and persistence behavior. |
| add | — | PCAP-012 | Bound compatibility to supported surfaces, not branch-local imports. |

## Semantic Diff

Revision 1 adds twelve root-source ownership and migration obligations where no
Potpie capability contract previously existed. It does not change the allowed
or required runtime outcomes in an accepted contract. The proposal makes the
final absence of the generic source namespace and the preservation of supported
behavior separately testable.

## Compatibility, Security, And Failure Impact

The proposal does not change a supported CLI, daemon, wire, setup, persistence,
authentication, authorization, or Context Engine interface. Repository-internal
imports under `potpie.product.*` must migrate and receive no compatibility
alias. Existing security and privacy guarantees remain binding; capability
relocation cannot weaken them.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| SPEC-PRODUCT revision 1 | — | Product outcomes and compatibility guard remain unchanged. | agent:codex |
| SPEC-SYSTEM revision 1 | — | Ownership direction and hosted call paths remain unchanged. | agent:codex |
| SPEC-CONTEXT-ENGINE revision 1 | — | Independent engine behavior and reverse-import prohibition remain unchanged. | agent:codex |
| SPEC-POTPIE-RESOURCE-MANAGER revision 2 | — | Lease, authorization, and resource ownership remain unchanged. | agent:codex |
| SPEC-DAEMON revision 1 | — | Runtime, protocol, discovery, and lifecycle remain unchanged. | agent:codex |
| SPEC-CLI revision 1 | — | Commands, presentation, streams, and exits remain unchanged. | agent:codex |
| Existing decisions ADR-0001 through ADR-0010 | — | The proposal is additive and does not rewrite accepted decisions. | agent:codex |
| Existing conformance records | — | Acceptance adds a contract but makes no implementation claim and does not rewrite pinned evidence. | agent:codex |
| Specification indexes | Add the contract and change transition. | — | agent:codex |

## Conformance Invalidation

Acceptance alone invalidates no existing conformance record because no accepted
contract revision changes and the new contract has no prior evidence. Any later
implementation ref is evaluated through new append-only conformance records;
freshness of old evidence is derived rather than edited.

## Validation

```text
Structural: passed; validate_spec.py reported 0 warnings
Semantic: passed; twelve atomic ownership and migration obligations reviewed
Provenance: passed; active authority and pinned observations resolve
Historical mutation: not applicable; initial revision
Fresh-agent reconstruction: passed; authorized reviewer accepted after reading from spec/index.md
```

## Acceptance

Accepted by `user:dsantra` at `2026-08-23T13:21:44+05:30` after authorized
review and completed validation. Acceptance establishes the contract target
but makes no implementation or verification claim.
