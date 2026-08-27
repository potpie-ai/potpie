---
id: CONF-CONTEXT-ENGINE
title: Context Engine Conformance
kind: conformance-record
record_status: final
spec_id: SPEC-CONTEXT-ENGINE
spec_revision: 1
spec_ref: 047cbe067c9c726e7e14f066675453372d8a8406
implementation_ref: ecf37757561166f94a66a7375483cb48b6b5ef58
performed_by: agent:codex
performed_at: "2026-08-27T12:47:45+05:30"
result: passed
previous_record: null
previous_record_id: CONF-CONTEXT-ENGINE
previous_record_ref: e05a4f1adb9d440552e576616c39d1df14990c2d
previous_record_path: spec/conformance/context-engine.md
---

# Context Engine Conformance Record

## Scope

This final record version verifies every active behavior, `CE-001` through
`CE-034` after the PR-review remediation. It pins the unchanged accepted
Context Engine contract and verifies typed reset execution, retryable unresolved
cleanup, active-operation draining before resource release, and validation
inside the domain boundary.

## Behavior Trace

| Behavior ID | Implementation claim | Verification result | Evidence | Notes |
|---|---|---|---|---|
| CE-001 | complete | passed | CE2-E1, CE2-E2 | Finite public façade remains intact. |
| CE-002 | complete | passed | CE2-E2, CE2-E3 | Public use remains independent of root CLI and daemon packages. |
| CE-003 | complete | passed | CE2-E3 | Public types and implementation ship together. |
| CE-004 | complete | passed | CE2-E3 | Context Core remains absent. |
| CE-005 | complete | passed | CE2-E1, CE2-E2 | Construction remains explicit. |
| CE-006 | complete | passed | CE2-E2 | Engine identity remains immutable. |
| CE-007 | complete | passed | CE2-E2 | Request models reject context selectors. |
| CE-008 | complete | passed | CE2-E2 | Different identities require different instances. |
| CE-009 | complete | passed | CE2-E2 | Context-bound instances remain isolated. |
| CE-010 | complete | passed | CE2-E2 | Close rejects new work, drains admitted operations, and retries only unresolved cleanup. |
| CE-011 | complete | passed | CE2-E1, CE2-E2 | Domain semantics remain engine-owned. |
| CE-012 | complete | passed | CE2-E1, CE2-E2 | Outcomes remain typed and transport-neutral. |
| CE-013 | complete | passed | CE2-E1, CE2-E3 | No terminal presentation boundary exists. |
| CE-014 | complete | passed | CE2-E1, CE2-E3 | Caller authentication and root authorization remain absent. |
| CE-015 | complete | passed | CE2-E1, CE2-E2 | Host provisioning remains explicit. |
| CE-016 | complete | passed | CE2-E1, CE2-E2 | Engine adapters implement engine ports. |
| CE-017 | complete | passed | CE2-E1, CE2-E2 | Typed context reset remains explicitly destructive. |
| CE-018 | complete | passed | CE2-E1, CE2-E3 | No public extension-registration contract exists. |
| CE-019 | complete | passed | CE2-E2 | Full engine regressions preserve parsing behavior. |
| CE-020 | complete | passed | CE2-E1, CE2-E2 | No dynamic dispatch or service locator is exposed. |
| CE-021 | complete | passed | CE2-E1, CE2-E2 | The façade delegates to focused services. |
| CE-022 | complete | passed | CE2-E2 | Host dependencies default to borrowed ownership. |
| CE-023 | complete | passed | CE2-E2 | Borrowed dependencies are not engine-closed. |
| CE-024 | complete | passed | CE2-E2 | Transferred dependencies close in order. |
| CE-025 | complete | passed | CE2-E2 | Post-close operations return typed lifecycle failure. |
| CE-026 | complete | passed | CE2-E3, CE2-E4 | Engine production imports no root `potpie` namespace. |
| CE-027 | complete | passed | CE2-E3 | HostShell and engine-owned root wiring remain absent. |
| CE-028 | complete | passed | CE2-E1, CE2-E2 | Boundary types remain engine-owned. |
| CE-029 | complete | passed | CE2-E2 | Semantic rejection remains a domain error. |
| CE-030 | complete | passed | CE2-E2 | Port failures remain dependency errors. |
| CE-031 | complete | passed | CE2-E1, CE2-E2 | Boundary failures remain redacted. |
| CE-032 | complete | passed | CE2-E2 | Failed construction yields no usable engine. |
| CE-033 | complete | passed | CE2-E1, CE2-E2 | Hosts explicitly select adapters. |
| CE-034 | complete | passed | CE2-E1, CE2-E2 | Destructive behavior is never inferred. |

## Reproducible Evidence

- **CE2-E1 — pinned source review:** the public façade, composition, requests,
  results, outcomes, and public exports under
  `potpie/context-engine/src/potpie_context_engine` at the implementation ref.
- **CE2-E2 — complete independent engine lane:** from `potpie/context-engine`,
  `uv run --project . pytest tests -m "not premerge_journey"`; result:
  `1153 passed, 32 skipped, 6 warnings in 79.62s`.
- **CE2-E3 — isolated distribution:** root and Context Engine wheels and sdists
  built; the engine wheel installed into a fresh environment; its public
  package and `ContextEngine` imported while `find_spec("potpie")` returned
  `None`.
- **CE2-E4 — permanent reverse-import gate:**
  `tests/characterization/test_cli_package_boundary.py` and
  `tests/characterization/test_potpie_capability_ownership.py`; the full
  characterization lane reported `31 passed`.

## Related Contracts Checked

| Spec ID | Revision | Spec ref | Result |
|---|---:|---|---|
| SPEC-GLOSSARY | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-SYSTEM | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-POTPIE-CAPABILITIES | 1 | b23b6cb9158e6b929f7d5f01c4dc0ca62a69f1df | passed |

## Known Gaps

None for the 34 active behaviors. The 32 skipped tests require external
integration services and do not cover obligations introduced by this change.

## Aggregate Result

`passed`: all 34 active Context Engine behaviors have complete implementation
claims and passed verification at the selected implementation ref.

## Freshness

Freshness is derived by comparing the pinned specification, implementation,
dependency, and evidence identities with the selected current targets.
