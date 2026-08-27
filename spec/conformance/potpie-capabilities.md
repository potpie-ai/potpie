---
id: CONF-POTPIE-CAPABILITIES
title: Potpie Capabilities Conformance
kind: conformance-record
record_status: final
spec_id: SPEC-POTPIE-CAPABILITIES
spec_revision: 1
spec_ref: b23b6cb9158e6b929f7d5f01c4dc0ca62a69f1df
implementation_ref: a530fcc05de8080fd982ea2c3bf796c25cfd400f
performed_by: agent:codex
performed_at: "2026-08-27T12:35:42+05:30"
result: passed
previous_record: null
previous_record_id: CONF-POTPIE-CAPABILITIES
previous_record_ref: 604c3eb5c9a561eec959ab688c279d04e9e6ff5b
previous_record_path: spec/conformance/potpie-capabilities.md
---

# Potpie Capabilities Conformance Record

## Scope

This final record version verifies every active capability behavior,
`PCAP-001` through `PCAP-012`, against the completed capability-first source
layout and behavior-preservation evidence, including process-safe configuration
and pot persistence and metadata-only pot services.

## Behavior Trace

| Behavior ID | Implementation claim | Verification result | Evidence | Notes |
|---|---|---|---|---|
| PCAP-001 | complete | passed | PCAP-E1, PCAP-E2 | Root source is capability-owned with no generic umbrella. |
| PCAP-002 | complete | passed | PCAP-E1, PCAP-E3 | Configuration owns root home resolution and process-safe persistence. |
| PCAP-003 | complete | passed | PCAP-E1, PCAP-E3 | Pots own metadata, policy, defaults, and process-safe persistence; graph reset remains engine-owned. |
| PCAP-004 | complete | passed | PCAP-E1, PCAP-E3 | Skills own catalog, validation, policy, and targets. |
| PCAP-005 | complete | passed | PCAP-E1, PCAP-E3 | Setup owns orchestration and lifecycle seams. |
| PCAP-006 | complete | passed | PCAP-E1, PCAP-E3 | Local identity extends the existing auth capability. |
| PCAP-007 | complete | passed | PCAP-E1, PCAP-E3 | Agent context composition is a top-level façade. |
| PCAP-008 | complete | passed | PCAP-E1, PCAP-E2 | Runtime composition is the sole concrete assembly point. |
| PCAP-009 | complete | passed | PCAP-E2, PCAP-E4 | Engine production imports no root Potpie namespace. |
| PCAP-010 | complete | passed | PCAP-E2, PCAP-E4 | `potpie.product` and aliases are absent. |
| PCAP-011 | complete | passed | PCAP-E3, PCAP-E4 | Supported runtime and persistence behavior is preserved. |
| PCAP-012 | complete | passed | PCAP-E2, PCAP-E4 | No compatibility API recreates old internal imports. |

## Reproducible Evidence

- **PCAP-E1 — pinned ownership review:**
  `potpie/{agent_context.py,config,pots,skills,setup,auth,runtime}` and root
  package metadata at the implementation ref.
- **PCAP-E2 — permanent ownership gates:**
  `tests/characterization/test_potpie_capability_ownership.py` verifies exact
  owners, documentation-only initializers, no replacement umbrella, no old
  namespace/import/alias/re-export, sole concrete runtime assembly, separated
  service groups, package metadata, and independent package lanes; combined
  boundary result: `18 passed`.
- **PCAP-E3 — behavior preservation:** the complete root suite reported
  `1446 passed, 4 skipped, 1 deselected`; the independent Context Engine suite
  reported `1152 passed, 32 skipped, 6 warnings`.
- **PCAP-E4 — packaging and isolated imports:** both wheels and sdists built;
  root artifacts contain capability packages and no `potpie/product`; fresh
  engine-only and root environments passed isolation verification; installed
  CLI and daemon entrypoints remained unchanged.

## Related Contracts Checked

| Spec ID | Revision | Spec ref | Result |
|---|---:|---|---|
| SPEC-PRODUCT | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-SYSTEM | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-CONTEXT-ENGINE | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-POTPIE-RESOURCE-MANAGER | 2 | a8c03337f3568232b35851dc2c86d128f7d23c0e | passed |
| SPEC-DAEMON | 2 | e73ebdbd6f0960e063344468051f84e37174697c | passed |
| SPEC-CLI | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |

## Known Gaps

None for `PCAP-001` through `PCAP-012`. This record does not establish the new
internal capability paths as a supported public Python API.

## Aggregate Result

`passed`: all 12 active Potpie capability behaviors have complete
implementation claims and passed verification.

## Freshness

Freshness is derived by comparing the pinned specification, implementation,
dependency, and evidence identities with the selected current targets.
