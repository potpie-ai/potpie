---
id: SPEC-VERIFICATION-INDEX
title: Verification Record Index
kind: index
status: accepted
version: 1.0.0
owners:
  - Potpie Engineering
depends_on:
  - SPEC-PROCESS
related_specs:
  - SPEC-PACKAGE-BOUNDARY
related_decisions: []
affects: []
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
    - SPEC-PROCESS
  drift_status: unverified
---

# Verification Record Index

Verification records connect an accepted spec version to code and cross-spec
evidence at a named commit.

`SPEC-PACKAGE-BOUNDARY` is proposed and not implemented, so it has no code
verification record yet. Commit 14 of the migration will add a record containing:

- the accepted spec version and verified commit;
- every behavior ID in scope;
- tests, package builds, isolated-install checks, and import-boundary evidence;
- cross-spec consistency results;
- known gaps and the final result.
