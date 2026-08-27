---
id: CONF-CLI
title: Potpie CLI Conformance
kind: conformance-record
record_status: final
spec_id: SPEC-CLI
spec_revision: 1
spec_ref: 047cbe067c9c726e7e14f066675453372d8a8406
implementation_ref: a0b52654f6fed50ec790cc2a72ccd581611ed3be
performed_by: agent:codex
performed_at: "2026-08-24T05:20:34Z"
result: passed
previous_record: null
previous_record_id: CONF-CLI-2026-08-24-01
previous_record_ref: 3e5edfd584aea53682720c3684e6fd78646fa1b3
previous_record_path: spec/conformance/cli-2026-08-24.md
---

# Potpie CLI Conformance Record

## Scope

This final record version verifies every active CLI behavior, `CLI-001`
through `CLI-033`, after the internal capability relocation.

## Behavior Trace

| Behavior ID | Implementation claim | Verification result | Evidence | Notes |
|---|---|---|---|---|
| CLI-001 | complete | passed | CLI2-E1, CLI2-E2 | Hosted commands remain typed. |
| CLI-002 | complete | passed | CLI2-E2, CLI2-E3 | Dynamic remote dispatch remains absent. |
| CLI-003 | complete | passed | CLI2-E1, CLI2-E2 | Selectors remain Resource Manager inputs. |
| CLI-004 | complete | passed | CLI2-E1, CLI2-E2 | Lifecycle commands use the controller. |
| CLI-005 | complete | passed | CLI2-E2, CLI2-E3 | Hosted CLI does not construct engines. |
| CLI-006 | complete | passed | CLI2-E1, CLI2-E3 | CLI owns no domain semantics. |
| CLI-007 | complete | passed | CLI2-E2 | Lower boundaries authenticate and authorize. |
| CLI-008 | complete | passed | CLI2-E2 | Presentation and prompting remain explicit. |
| CLI-009 | complete | passed | CLI2-E2 | Destructive actions require confirmation. |
| CLI-010 | complete | passed | CLI2-E2 | Confirmation remains request-bound intent. |
| CLI-011 | complete | passed | CLI2-E2 | Human decline remains typed cancellation. |
| CLI-012 | complete | passed | CLI2-E2 | Unknown lower outcomes are not cancellation. |
| CLI-013 | complete | passed | CLI2-E2 | Errors render from typed categories. |
| CLI-014 | complete | passed | CLI2-E2 | Machine mode never prompts. |
| CLI-015 | complete | passed | CLI2-E2 | Machine output remains one JSON document. |
| CLI-016 | complete | passed | CLI2-E2 | Machine stdout remains uncontaminated. |
| CLI-017 | complete | passed | CLI2-E2 | Noninteractive execution does not block. |
| CLI-018 | complete | passed | CLI2-E2 | Missing machine intent fails pre-dispatch. |
| CLI-019 | complete | passed | CLI2-E2 | Successful commands exit zero. |
| CLI-020 | complete | passed | CLI2-E2 | Typed codes drive exit mapping. |
| CLI-021 | complete | passed | CLI2-E1 | Current JSON fields remain outside the contract. |
| CLI-022 | complete | passed | CLI2-E3 | Hosted CLI has no legacy host calls. |
| CLI-023 | complete | passed | CLI2-E1, CLI2-E2 | Readiness and stop use the typed client. |
| CLI-024 | complete | passed | CLI2-E1 | Full numeric mapping remains outside the contract. |
| CLI-025 | complete | passed | CLI2-E2 | Output excludes credentials. |
| CLI-026 | complete | passed | CLI2-E2 | Presentation does not parse exception messages. |
| CLI-027 | complete | passed | CLI2-E2 | Unknown mutation outcomes are not replayed. |
| CLI-028 | complete | passed | CLI2-E2 | Machine diagnostics avoid stdout. |
| CLI-029 | complete | passed | CLI2-E2 | Telemetry excludes sensitive payloads. |
| CLI-030 | complete | passed | CLI2-E2 | Human decline dispatches nothing. |
| CLI-031 | complete | passed | CLI2-E2 | Failure and cancellation remain nonzero. |
| CLI-032 | complete | passed | CLI2-E2 | One presentation mode is selected. |
| CLI-033 | complete | passed | CLI2-E2 | Missing machine intent is presentation failure. |

## Reproducible Evidence

- **CLI2-E1 — pinned source review:** root CLI command, formatting, bootstrap,
  setup, and typed client paths at the implementation ref.
- **CLI2-E2 — complete root lane:**
  `uv run pytest tests -m "not premerge_journey" -q`; result:
  `1417 passed, 4 skipped, 1 deselected in 32.46s`.
- **CLI2-E3 — permanent architecture and process gates:** the characterization
  lane reported `31 passed`, including unchanged command inventory, installed
  metadata entrypoints, and legacy-boundary absence.
- **CLI2-E4 — isolated entrypoint smoke:** the fresh root installation resolved
  `potpie` to `potpie.cli.main:main` and imported `potpie.cli.main`.

## Related Contracts Checked

| Spec ID | Revision | Spec ref | Result |
|---|---:|---|---|
| SPEC-GLOSSARY | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-SYSTEM | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-DAEMON | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-POTPIE-CAPABILITIES | 1 | b23b6cb9158e6b929f7d5f01c4dc0ca62a69f1df | passed |

## Known Gaps

None for the 33 active behaviors. The exact JSON field set and complete
nonzero exit-code mapping remain deliberately deferred. The Rust-dependent
`premerge_journey` lane was not run locally and is not claimed here.

## Aggregate Result

`passed`: all 33 active CLI behaviors have complete implementation claims and
passed verification.

## Freshness

Freshness is derived by comparing the pinned specification, implementation,
dependency, and evidence identities with the selected current targets.
