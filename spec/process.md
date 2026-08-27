---
id: SPEC-PROCESS
title: Specification Process
kind: process
revision: 2
maturity: accepted
owners:
  - team:potpie
depends_on: []
change_ref: SPEC-CHANGE-0011
---

# Specification Process

## Purpose

This contract defines how Potpie proposes, accepts, changes, and verifies its
behavioral contracts. Markdown under `spec/` is the canonical contract.
Architecture decisions explain why a contract exists, implementation shows how
software behaves at a selected ref, and conformance records preserve claims and
evidence. Those artifacts have separate roles.

## Acceptance Authority

For the initial revision-1 contract set, the acceptance authority is
`user:dsantra`. The `team:potpie` owner maintains the documents but does not
gain acceptance authority from ownership alone. `agent:codex` may author
proposals and execute validation without gaining acceptance authority.

A later change to acceptance authority is represented by a new accepted
revision of this contract.

## State Model

Contract maturity describes whether contract text is draft, proposed, accepted,
or retired. Behavior lifecycle describes whether an individual behavior is
active, deprecated, or retired. Implementation claims and verification results
are recorded per behavior in conformance records. Freshness is derived by
comparing pinned identities and evidence with selected current refs.

These axes answer different questions:

| Axis | Question |
|---|---|
| Contract maturity | Does this exact contract revision bind? |
| Behavior lifecycle | Does this behavior currently impose an obligation? |
| Implementation claim | What does an implementation claim to cover? |
| Verification result | What did the recorded evidence establish? |
| Derived freshness | Can that recorded evidence establish current state? |

## Normative Requirements

PROC-001 [active]: Contract maturity, behavior lifecycle, implementation claim, verification result, and derived freshness MUST remain independent state axes.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001

PROC-002 [active]: Every edit to an accepted contract MUST allocate the next positive-integer revision.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-001

PROC-003 [active]: Only an actor named by the accepted specification process as an acceptance authority MUST be permitted to bind a contract revision.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-001

PROC-004 [active]: Authorship, implementation work, test results, tool output, and agent assertions MUST NOT be treated as contract acceptance.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-003

PROC-005 [active]: Implementation claims and verification results MUST be recorded only in conformance records.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-001

PROC-006 [active]: A contract revision MUST pass applicable structural, semantic, provenance, dependency-consistency, historical-mutation, and fresh-agent-reconstruction review before acceptance.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-002
  @ PROC-003

PROC-007 [active]: Every accepted contract revision MUST remain addressable through Git history.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-002

PROC-008 [active]: An accepted active or deprecated behavior MUST NOT contain an unresolved question edge.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-006

PROC-009 [active]: Code, test, incident, documentation, and runtime observations MUST NOT override accepted behavior without an accepted contract revision.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-003

PROC-010 [active]: Every edit to an accepted contract MUST have a matching accepted change record.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-002

PROC-011 [active]: A final conformance record version identified by its stable record ID, repository path, and Git ref MUST be immutable at that ref; publishing a successor at the same stable path MUST create a new Git object without altering the prior version.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-005

PROC-012 [active]: A conformance record MUST pin the exact specification revision and ref and the exact implementation ref it evaluates.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-005

PROC-013 [active]: Every accepted change record MUST remain addressable through Git history.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-010

PROC-014 [active]: Provenance lineage for an accepted revision MUST remain addressable through Git history.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-007

PROC-015 [active]: Every retired behavior identifier MUST remain addressable through Git history.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-007

PROC-016 [active]: A retired behavior identifier MUST NOT be reused for different semantics.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-015

PROC-017 [active]: An accepted active or deprecated behavior MUST NOT contain an explicit assumption edge.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-006

PROC-018 [active]: An initial contract MUST begin at revision 1 with a proposed change from revision 0.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001

PROC-019 [active]: Every semantic contract change MUST declare an operation for each affected behavior identifier.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-010

PROC-020 [active]: A contract revision MUST NOT silently repurpose an existing behavior identifier.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-016
  @ PROC-019

PROC-021 [active]: An acceptance record MUST identify the accepted revision, matching change record, authorized acceptors, and acceptance time.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001
  @ PROC-003
  @ PROC-010

PROC-022 [active]: Each defined conformance scope MUST have at most one current record at a stable, scope-named path; dates, sequence numbers, implementation versions, and pull-request numbers MUST NOT be encoded in current conformance filenames.
  > authority [active]: user:dsantra
  @ PROC-011

PROC-023 [active]: A successor conformance record MUST identify the immediately preceding record version through flat `previous_record_id`, `previous_record_ref`, and `previous_record_path` fields, and that historical target MUST resolve without requiring the prior file to remain in the current tree.
  > authority [active]: user:dsantra
  @ PROC-011
  @ PROC-022

PROC-024 [active]: Verification evidence spanning multiple module scopes MUST be recorded in the applicable system-scope conformance record rather than in an additional pull-request-specific or release-specific conformance file.
  > authority [active]: user:dsantra
  @ PROC-022

PROC-025 [active]: Pre-merge integration verification MUST pin the repository, pull-request number, pull-request head commit, target base ref and commit, accepted specification identities, and implementation ref; a predicted eventual merge commit MUST NOT be required as the verification identity.
  > authority [active]: user:dsantra
  @ PROC-012
  @ PROC-024

PROC-026 [active]: A successor commit that only publishes conformance, specification-governance, derived-index, or conformance-validation artifacts MAY point to its verified predecessor because a record cannot contain its own commit hash; any intervening change to an in-scope contract, runtime implementation, test, or cited verification evidence MUST be reverified instead of classified as publication-only.
  > authority [active]: user:dsantra
  @ PROC-011
  @ PROC-012
  @ PROC-025

## Proposals And Mutations

An initial contract begins at revision 1 with a proposed change record from
revision 0. A mutation of accepted behavior starts by pinning the previous
accepted revision and Git ref, allocating the next integer revision, and
declaring a behavior operation for every semantic change.

Behavior identifiers are stable. Changed semantics use replacement, split, or
merge operations with retained tombstones rather than silently repurposing an
existing identifier. Reverse impact is derived from canonical forward
dependencies, provenance, questions, and repository references.

## Review And Acceptance

Review distinguishes the author, acceptance authority, and any conformance
performer. The acceptance record identifies the exact revision, matching change
record, authorized acceptors, and acceptance time. The resulting Git commit or
immutable blob completes the accepted revision identity.

The section-level collaboration procedure used while authoring commit 1 is an
operational control for that commit. It is not a permanent acceptance
requirement for later repository work.

## Conformance And Freshness

Conformance records pin an accepted specification revision and ref, an
implementation ref, the behavior scope, claims, verification results, evidence,
performers, and time. Each scope has one current stable path. A new result
publishes a successor at that path and points to the previous record's ID, full
Git ref, and historical path. The previous Git object remains the immutable record;
the stable filename is navigation to the current record, not its version
identity.

The system-scope record owns evidence that crosses module boundaries. For a
GitHub pull request, it pins the repository, pull-request number and head,
target base ref and commit, accepted specification refs, and implementation
ref. A synthetic merge candidate and tree may be recorded as supporting
evidence, but the eventual merge commit is not predicted or required.

The commit that publishes a conformance record cannot self-pin. A later commit
containing only conformance, specification-governance, derived-index, or
conformance-validation artifacts may therefore point to its verified
predecessor. This exception does not cover an in-scope contract, runtime,
test, or cited-evidence change; any such change requires a new verification
result at the stable path.

Freshness is calculated from pinned identities, dependencies, and available
evidence. A failed result and a stale result answer different questions:
failure records a contradiction at selected refs, while staleness means prior
evidence no longer establishes current state.

## Retention

Git history preserves accepted revisions and change records. Retired behavior
identifiers remain as lineage tombstones and are not reused. Source and
conformance record versions remain addressable at their immutable refs even
when their stable current paths advance or an older dated path is absent from
the current tree.

## Acceptance Criteria

This revision is ready for acceptance when:

- `PROC-001` through `PROC-026` are structurally valid and carry active
  authority.
- The contract and `SPEC-CHANGE-0011` identify the same transition.
- ADR-0001 explains the decision without introducing behavior absent from these
  nodes.
- Applicable structural, semantic, provenance, consistency, and reconstruction
  reviews have no blocking findings.
- Implementation and verification remain independently unclaimed and
  unverified until conformance records exist.
- The current conformance tree contains one stable file per defined scope, each
  successor resolves its historical pointer, and cross-system integration
  evidence does not require a separate PR-specific record.

## Rationale

Separating authority, desired behavior, implementation, and evidence makes it
possible to accept a target architecture before its migration is complete
without presenting unfinished code as compliant. Stable scope paths keep the
current verification state discoverable while Git refs preserve every final
record version without accumulating dated copies in the current tree.
