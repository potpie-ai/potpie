---
id: SPEC-CHANGE-0011
title: Stabilize Conformance Record Paths
kind: spec-change
change_status: accepted
spec_id: SPEC-PROCESS
from_revision: 1
from_ref: 047cbe067c9c726e7e14f066675453372d8a8406
to_revision: 2
change_type: normative
initiated_by: user:dsantra
authored_by:
  - agent:codex
accepted_by:
  - user:dsantra
accepted_at: 2026-08-25T14:33:00+05:30
---

# SPEC-CHANGE-0011: Stabilize Conformance Record Paths

## Intent

Retain one current conformance document per module or system scope, use Git
history as the version store, and place pull-request/base integration evidence
in the existing cross-system record. Avoid dated successor files and separate
PR-specific records in the current tree.

## Provenance Sources

> authority [active]: user:dsantra
> decision [active]: decision:ADR-0001
> observation [active]: code:spec/conformance@3e5edfd584aea53682720c3684e6fd78646fa1b3

## Behavior Operations

| Operation | From behavior | To behavior | Reason |
|---|---|---|---|
| clarify | PROC-011 | PROC-011 | Define immutability at the record's Git ref while allowing a successor at the same stable path. |
| add | — | PROC-022 | Require at most one current stable path for each conformance scope. |
| add | — | PROC-023 | Require flat, resolvable ID, Git-ref, and path fields for the immediately previous record. |
| add | — | PROC-024 | Keep cross-module integration evidence in the applicable system record. |
| add | — | PROC-025 | Define the durable PR-head and base-commit identity for pre-merge verification. |
| add | — | PROC-026 | Define the non-self-referential publication boundary without exempting implementation or evidence changes. |

## Semantic Diff

Revision 2 makes the logical append-only model explicit: a final record version
is immutable at its Git ref, while the stable scope path advances to a new Git
object when verification changes. Historical versions do not remain as dated
files in the current tree. The revision also defines the system record as the
owner of cross-module PR/base integration evidence and removes any need to
predict GitHub's eventual merge commit.

`PROC-011` retains its existing immutability obligation. The clarification
identifies the record version precisely as stable ID plus repository path plus
Git ref; it does not permit rewriting an existing commit or blob.

Revision 2 also resolves the self-reference boundary: the commit that publishes
a record may identify the verified predecessor when the intervening diff is
limited to conformance, specification governance, derived indexes, or the
conformance validator. Runtime, in-scope contract, test, and cited-evidence
changes remain re-verification triggers.

## Compatibility, Security, And Failure Impact

This change affects specification storage and verification reconstruction only.
It changes no product, Context Engine, Resource Manager, daemon, CLI, protocol,
authorization, persistence, or failure behavior. Verification conclusions and
their pinned implementation/specification identities retain their original
strength.

## Computed Impact Review

| Artifact or behavior | Required change | No-change reason | Reviewed by |
|---|---|---|---|
| Current conformance records | Move the latest record for each of six scopes to its stable path and add historical Git pointers. | Earlier record blobs remain unchanged at their existing refs. | agent:codex |
| Cross-system conformance | Fold the approved PR `#1057` head/base and synthetic merge evidence into `cross-system.md`. | Module-specific behavior evidence remains in the five module records. | agent:codex |
| Conformance index | List only the six stable current records and explain Git-history retrieval. | The index remains derived navigation rather than verification authority. | agent:codex |
| SPEC-INDEX | Register SPEC-PROCESS revision 2, this change, and the stable conformance index. | Other contract identities and dependencies remain unchanged. | agent:codex |
| SPEC-GLOSSARY and product/system/module contracts | — | Storage and evidence lineage do not change runtime obligations. | agent:codex |
| ADR-0001 | — | Its Git-governance rationale already supports immutable refs and derived navigation. | agent:codex |
| Existing final conformance versions | — | They remain addressable at commits `012d3638f2eae62685ea2f711c9c7a7b0dfeae84` and `3e5edfd584aea53682720c3684e6fd78646fa1b3`. | agent:codex |

## Conformance Invalidation

None. This process revision neither changes an accepted runtime behavior nor
weakens an existing implementation or verification conclusion. The stable
records preserve the latest six conclusions and link to their prior committed
versions. Freshness remains derived from pinned identities.

## Validation

```text
Structural: passed; validate_spec.py reported 0 warnings
Semantic: passed; stable-path, lineage, integration-scope, and PR/base obligations are atomic and do not change runtime behavior
Provenance: passed; all changed or added process behaviors carry active user authority
Historical mutation: passed; revision advances 1 to 2, PROC-011 retains immutability, and PROC-022 through PROC-026 are unused new IDs
Dependency/consistency: passed; six current scopes, indexes, historical refs, and accepted runtime contracts agree
Fresh-agent reconstruction: passed; spec/index.md leads to the process, six stable records, module contracts, PR/base identity, and Git-history retrieval
Independent conformance state: unchanged; six scopes cover 190 applicable active behaviors
```

## Acceptance

Accepted by `user:dsantra` at `2026-08-25T14:33:00+05:30` after review of the
final format and completion of the applicable validation gates. Acceptance
binds Specification Process revision 2 and makes no new runtime implementation
claim.
