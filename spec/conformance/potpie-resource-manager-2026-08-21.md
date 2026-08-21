---
id: CONF-POTPIE-RESOURCE-MANAGER-2026-08-21-01
title: Potpie Resource Manager Conformance at 51018713
kind: conformance-record
record_status: final
spec_id: SPEC-POTPIE-RESOURCE-MANAGER
spec_revision: 2
spec_ref: a8c03337f3568232b35851dc2c86d128f7d23c0e
implementation_ref: 5101871348ceae9f59830dd82af06d890a6d0f48
performed_by: agent:codex
performed_at: "2026-08-21T13:54:37Z"
result: passed
previous_record: null
---

# Potpie Resource Manager Conformance Record

## Scope

This record verifies all 37 active behaviors in
`SPEC-POTPIE-RESOURCE-MANAGER` revision 2: `RM-001` through `RM-014` and
`RM-017` through `RM-039`. Retired `RM-015` and `RM-016` are preserved as
lineage and are not implementation obligations for revision 2.

## Behavior Trace

| Behavior ID | Implementation claim | Verification result | Evidence | Notes |
|---|---|---|---|---|
| RM-001 | complete | passed | RM-E1, RM-E2 | Selection resolves one identity or a typed selection failure. |
| RM-002 | complete | passed | RM-E1, RM-E5 | Resolution is independent of CLI presentation. |
| RM-003 | complete | passed | RM-E1, RM-E2 | Authentication and authorization precede lease issuance. |
| RM-004 | complete | passed | RM-E1, RM-E2 | Selection variants remain distinct. |
| RM-005 | complete | passed | RM-E1, RM-E2 | Destructive intent binds actor, operation, selector, identity, and request. |
| RM-006 | complete | passed | RM-E1, RM-E2 | Explicit adapters acquire and release resources. |
| RM-007 | complete | passed | RM-E1, RM-E2 | Dependency composition carries ownership declarations. |
| RM-008 | complete | passed | RM-E1, RM-E2 | Lease contains the context-bound engine. |
| RM-009 | complete | passed | RM-E1, RM-E2 | Cached engines are never retargeted. |
| RM-010 | complete | passed | RM-E1, RM-E2 | Reuse keys include identity and composition fingerprint. |
| RM-011 | complete | passed | RM-E1, RM-E2 | Host-resource release is owned and idempotent. |
| RM-012 | complete | passed | RM-E2, RM-E3 | Local and daemon handlers share acquisition policy. |
| RM-013 | complete | passed | RM-E1, RM-E2 | Coordination uses narrow context/resource keys. |
| RM-014 | complete | passed | RM-E2 | Unrelated contexts compose concurrently. |
| RM-017 | complete | passed | RM-E1, RM-E5 | Manager owns no domain semantics. |
| RM-018 | complete | passed | RM-E1, RM-E5 | Manager owns no terminal presentation. |
| RM-019 | complete | passed | RM-E1, RM-E2 | Scope denial returns `AuthorizationError`. |
| RM-020 | complete | passed | RM-E1, RM-E2 | Lease exposes resolved identity and authorization scope. |
| RM-021 | complete | passed | RM-E1, RM-E2 | Lease exposes resource ownership metadata. |
| RM-022 | complete | passed | RM-E1, RM-E2 | Manager and lease never dispatch engine operations. |
| RM-023 | complete | passed | RM-E1, RM-E5 | No generic dispatch, service lookup, or mirrored façade exists. |
| RM-024 | complete | passed | RM-E1, RM-E5 | Daemon transport remains outside the manager. |
| RM-025 | complete | passed | RM-E2 | Failed acquisition releases all opened resources. |
| RM-026 | complete | passed | RM-E2 | Engine cleanup precedes host-resource cleanup. |
| RM-027 | complete | passed | RM-E1, RM-E2 | Borrowed resource cleanup remains host-owned. |
| RM-028 | complete | passed | RM-E1, RM-E4 | Logs and typed details exclude credentials and payloads. |
| RM-029 | complete | passed | RM-E2 | Transferred resources have one cleanup owner. |
| RM-030 | complete | passed | RM-E1, RM-E2 | Lease release is idempotent. |
| RM-031 | complete | passed | RM-E2 | Engine cleanup failure does not skip host cleanup. |
| RM-032 | complete | passed | RM-E1, RM-E2 | Release failure preserves the primary operation outcome. |
| RM-033 | complete | passed | RM-E1, RM-E2 | Invalid destructive intent returns `AuthorizationError`. |
| RM-034 | complete | passed | RM-E2 | Acquisition and cleanup failures are both retained. |
| RM-035 | complete | passed | RM-E2 | Invalid destructive intent issues no lease. |
| RM-036 | complete | passed | RM-E1, RM-E2 | Acquisition has the complete typed outcome union. |
| RM-037 | complete | passed | RM-E1, RM-E2 | Selection, authentication, authorization, and lifecycle errors are distinct. |
| RM-038 | complete | passed | RM-E1, RM-E2 | Authentication failure returns `AuthenticationError`. |
| RM-039 | complete | passed | RM-E2 | Authentication failure stops before later acquisition stages. |

## Reproducible Evidence

- **RM-E1 — pinned source review:**
  `potpie/runtime/{resource_manager.py,ownership.py,local_engine.py,clients.py,coordinator.py}` at the implementation ref.
- **RM-E2 — focused manager and execution boundary:**
  `uv run pytest tests/unit/test_context_resource_manager.py tests/unit/test_runtime_clients.py tests/unit/test_operation_coordinator.py tests/unit/test_runtime_ownership.py tests/unit/test_runtime_protocol_codec.py tests/unit/test_runtime_transport.py -q`;
  result: `88 passed in 0.24s`.
- **RM-E3 — real local/daemon path evidence:**
  `uv run pytest tests/characterization/test_context_runtime_architecture.py tests/integration/test_context_runtime_contract.py tests/integration/test_canonical_daemon_runtime.py tests/integration/test_daemon_controller.py -q`;
  result: `27 passed in 7.21s`.
- **RM-E4 — complete root lane:**
  `uv run pytest tests -m "not premerge_journey" -q`; result:
  `1401 passed, 4 skipped, 1 deselected in 32.98s`.
- **RM-E5 — permanent architecture inventory:** the AST/TOML checks in
  `tests/characterization/test_context_runtime_architecture.py` passed and found
  no HostShell, generic compatibility dispatch, Context Core import, engine-to-root
  import, reflective route, legacy discovery, or competing runtime definition.

## Related Contracts Checked

| Spec ID | Revision | Spec ref | Result |
|---|---:|---|---|
| SPEC-GLOSSARY | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-SYSTEM | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-CONTEXT-ENGINE | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |

## Known Gaps

None for active revision-2 behavior. Retired `RM-015` and `RM-016` are not
counted as current obligations.

## Aggregate Result

`passed`: all 37 active Resource Manager behaviors have complete
implementation claims and passed verification.

## Freshness

Freshness is derived by comparing the pinned specification, implementation,
dependency, and evidence identities with the selected current targets.
