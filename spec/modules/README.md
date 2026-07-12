---
id: SPEC-MODULE-INDEX
title: Module Spec Index
kind: index
status: draft
version: 0.1.0
owners:
  - Potpie Engineering
depends_on:
  - SPEC-INDEX
  - SPEC-PROCESS
related_specs:
  - SPEC-PACKAGE-BOUNDARY
related_decisions:
  - ADR-0002
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
    - SPEC-INDEX
  drift_status: unverified
---

# Module Spec Index

Module specs define bounded behavior, ownership, interfaces, failure modes, and
acceptance criteria.

| ID | File | Status | Version | Summary |
|---|---|---:|---:|---|
| `SPEC-PACKAGE-BOUNDARY` | [package-boundary.md](package-boundary.md) | proposed | 0.1.0 | Root Potpie and standalone context-engine ownership and interaction contract. |
