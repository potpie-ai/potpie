---
id: CONF-POTPIE-RESOURCE-MANAGER
title: Potpie Resource Manager Conformance
kind: conformance-record
record_status: final
spec_id: SPEC-POTPIE-RESOURCE-MANAGER
spec_revision: 2
spec_ref: a8c03337f3568232b35851dc2c86d128f7d23c0e
implementation_ref: ecf37757561166f94a66a7375483cb48b6b5ef58
performed_by: agent:codex
performed_at: "2026-08-27T12:47:45+05:30"
result: passed
previous_record: null
previous_record_id: CONF-POTPIE-RESOURCE-MANAGER
previous_record_ref: e05a4f1adb9d440552e576616c39d1df14990c2d
previous_record_path: spec/conformance/potpie-resource-manager.md
---

# Potpie Resource Manager Conformance Record

## Scope

This final record version verifies all 37 active revision-2 behaviors:
`RM-001` through `RM-014` and `RM-017` through `RM-039`. Retired `RM-015` and
`RM-016` remain lineage rather than current obligations. This successor also
verifies construction-failure cleanup, shared runtime coordination, keyed
snapshot destinations, inbox serialization, cancelled-writer wakeups,
retryable shutdown cleanup, and cancellation-safe lease accounting.

## Behavior Trace

| Behavior ID | Implementation claim | Verification result | Evidence | Notes |
|---|---|---|---|---|
| RM-001 | complete | passed | RM2-E1, RM2-E2 | Selection remains typed. |
| RM-002 | complete | passed | RM2-E1, RM2-E3 | Resolution remains presentation-independent. |
| RM-003 | complete | passed | RM2-E1, RM2-E2 | Authentication and authorization precede leases. |
| RM-004 | complete | passed | RM2-E2 | Selection variants remain distinct. |
| RM-005 | complete | passed | RM2-E2 | Destructive intent remains request-bound. |
| RM-006 | complete | passed | RM2-E1, RM2-E2 | Resource acquisition and release remain explicit. |
| RM-007 | complete | passed | RM2-E1, RM2-E2 | Ownership declarations remain explicit. |
| RM-008 | complete | passed | RM2-E1, RM2-E2 | Leases carry context-bound engines. |
| RM-009 | complete | passed | RM2-E2 | Cached engines are not retargeted. |
| RM-010 | complete | passed | RM2-E2 | Reuse keys retain identity and composition. |
| RM-011 | complete | passed | RM2-E2 | Owned release remains idempotent and cancellation-safe. |
| RM-012 | complete | passed | RM2-E2 | Local and daemon paths share acquisition policy. |
| RM-013 | complete | passed | RM2-E1, RM2-E2 | Coordination keys match context and normalized resource conflict scopes. |
| RM-014 | complete | passed | RM2-E2 | Unrelated contexts remain concurrent. |
| RM-017 | complete | passed | RM2-E1, RM2-E3 | Manager owns no domain semantics. |
| RM-018 | complete | passed | RM2-E1, RM2-E3 | Manager owns no presentation. |
| RM-019 | complete | passed | RM2-E2 | Scope denial remains typed authorization failure. |
| RM-020 | complete | passed | RM2-E1, RM2-E2 | Lease identity and scope remain exposed. |
| RM-021 | complete | passed | RM2-E1, RM2-E2 | Lease ownership metadata remains exposed. |
| RM-022 | complete | passed | RM2-E1, RM2-E3 | Manager and lease never dispatch domain operations. |
| RM-023 | complete | passed | RM2-E1, RM2-E3 | Generic dispatch remains absent. |
| RM-024 | complete | passed | RM2-E1, RM2-E3 | Daemon transport remains external. |
| RM-025 | complete | passed | RM2-E2 | Failed composition or engine construction releases opened resources. |
| RM-026 | complete | passed | RM2-E2 | Engine cleanup precedes host cleanup and unresolved failures remain retryable. |
| RM-027 | complete | passed | RM2-E2 | Borrowed cleanup remains host-owned. |
| RM-028 | complete | passed | RM2-E2, RM2-E4 | Sensitive values remain redacted. |
| RM-029 | complete | passed | RM2-E2 | Transferred resources have one cleanup owner. |
| RM-030 | complete | passed | RM2-E2 | Lease release remains idempotent. |
| RM-031 | complete | passed | RM2-E2 | Engine cleanup failure does not skip host cleanup. |
| RM-032 | complete | passed | RM2-E2 | Release failure preserves the primary outcome. |
| RM-033 | complete | passed | RM2-E2 | Invalid destructive intent remains authorization failure. |
| RM-034 | complete | passed | RM2-E2 | Acquisition and cleanup failures are retained. |
| RM-035 | complete | passed | RM2-E2 | Invalid intent issues no lease. |
| RM-036 | complete | passed | RM2-E1, RM2-E2 | Acquisition retains its typed outcome union. |
| RM-037 | complete | passed | RM2-E1, RM2-E2 | Failure categories remain distinct. |
| RM-038 | complete | passed | RM2-E1, RM2-E2 | Authentication failure remains distinct. |
| RM-039 | complete | passed | RM2-E2 | Authentication failure stops later stages. |

## Reproducible Evidence

- **RM2-E1 — pinned source review:** Resource Manager, ownership, local engine,
  typed clients, coordinator, and runtime composition at the implementation ref.
- **RM2-E2 — complete root lane:**
  `uv run pytest tests -m "not premerge_journey" -q`; result:
  `1447 passed, 4 skipped, 1 deselected`.
- **RM2-E3 — permanent architecture gates:** the full characterization lane
  reported `31 passed`, including service-group separation and sole concrete
  runtime assembly.
- **RM2-E4 — lint and format:** authoritative root Ruff policy passed; root
  format check reported `774 files already formatted`.

## Related Contracts Checked

| Spec ID | Revision | Spec ref | Result |
|---|---:|---|---|
| SPEC-GLOSSARY | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-SYSTEM | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-CONTEXT-ENGINE | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-POTPIE-CAPABILITIES | 1 | b23b6cb9158e6b929f7d5f01c4dc0ca62a69f1df | passed |

## Known Gaps

None for active revision-2 behavior. Retired `RM-015` and `RM-016` are not
counted as current obligations.

## Aggregate Result

`passed`: all 37 active Resource Manager behaviors have complete
implementation claims and passed verification.

## Freshness

Freshness is derived by comparing the pinned specification, implementation,
dependency, and evidence identities with the selected current targets.
