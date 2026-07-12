---
id: SPEC-PROCESS
title: Potpie Spec Process
kind: process
status: accepted
version: 1.0.0
owners:
  - Potpie Engineering
depends_on: []
related_specs:
  - SPEC-INDEX
related_decisions:
  - ADR-0001
affects:
  - SPEC-INDEX
  - SPEC-GLOSSARY
  - SPEC-PRODUCT
  - SPEC-SYSTEM
  - SPEC-PACKAGE-BOUNDARY
open_questions: []
verification:
  code_status: not_applicable
  verified_commit: null
  verified_at: null
  verified_by: null
  behavior_scope: []
  evidence: []
  cross_spec_status: passed
  cross_spec_checked_against:
    - SPEC-INDEX
  drift_status: unverified
---

# Potpie Spec Process

This file defines how Potpie specs are written, reviewed, accepted, linked,
versioned, implemented, and verified. The rationale is recorded in
[ADR-0001](decisions/ADR-0001-spec-process.md).

## Spec Contract

PROCESS-001: The spec tree MUST define expected behavior, guarantees, failure
modes, interfaces, security rules, acceptance criteria, open questions,
architectural decisions, and verification state.

PROCESS-002: Accepted specs MUST NOT hide unresolved behavior in vague prose.

PROCESS-003: Every normative behavior MUST have a stable behavior ID that does
not contain the spec version.

PROCESS-004: Every spec file MUST carry complete YAML metadata describing its
identity, lifecycle, relationships, open questions, and verification state.

PROCESS-005: A relationship that affects normative behavior MUST be linked in
both directions between specs and decisions.

PROCESS-006: A spec MUST NOT be marked `verified` without a commit, explicit
behavior scope, evidence, and a cross-spec consistency result.

## Lifecycle

```text
sketch -> draft -> proposed -> accepted -> implemented -> verified
                                    \-> stale
```

- `draft` is structured but may remain incomplete.
- `proposed` is reviewable end to end but not binding.
- `accepted` is the normative source of truth.
- `implemented` means code claims to satisfy the accepted behavior.
- `verified` means the accepted behavior and related specs were checked against
  a named commit.
- `stale` means prior acceptance or verification needs review.

## Versioning

Specs use semantic versions independently of package versions:

- `0.x.x` for draft and proposed work.
- `1.0.0` for the first accepted contract.
- Patch for non-normative clarification.
- Minor for compatible normative expansion.
- Major for removed behavior, changed failure semantics, or other breaking
  normative changes.

## Acceptance

A spec can become accepted only when its ownership, scope, interfaces,
invariants, failure behavior, security posture, examples, acceptance criteria,
relationships, and open-question state are complete and internally consistent.

## Verification

PROCESS-006 requires verification to name:

- the code commit;
- the behavior IDs checked;
- commands and evidence;
- related specs and ADRs reviewed;
- known gaps;
- the resulting drift state.

Detailed evidence belongs in `spec/verification/`; compact state belongs in the
owning spec's metadata.

## Updates

PROCESS-007: Normative updates MUST revise affected behavior, acceptance
criteria, examples, relationships, version, and verification metadata together.
Verification is reset or narrowed whenever prior evidence no longer covers the
changed contract.
