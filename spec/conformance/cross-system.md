---
id: CONF-SYSTEM
title: Context Runtime Cross-System Conformance
kind: conformance-record
record_status: final
spec_id: SPEC-SYSTEM
spec_revision: 1
spec_ref: 047cbe067c9c726e7e14f066675453372d8a8406
implementation_ref: a530fcc05de8080fd982ea2c3bf796c25cfd400f
target_repository: potpie-ai/potpie
target_pull_request: 1057
target_base_ref: main
target_base_commit: 20a8389cabec6e5924b1e3d4ef12d1dcfe900a3c
target_pr_head_ref: refactor/context-runtime-boundary
target_pr_head_commit: a530fcc05de8080fd982ea2c3bf796c25cfd400f
target_merge_candidate: 6f1ba68e957b082635b3c2c1a381396cedb03a0d
target_merge_tree: e74aaa84b885dc1751b60be855db11915981abf5
performed_by: agent:codex
performed_at: "2026-08-27T12:35:42+05:30"
result: passed
previous_record: null
previous_record_id: CONF-SYSTEM
previous_record_ref: 604c3eb5c9a561eec959ab688c279d04e9e6ff5b
previous_record_path: spec/conformance/cross-system.md
---

# Context Runtime Cross-System Conformance

## Scope

This record verifies `SYS-001` through `SYS-023`, checks the five current module
records against one implementation ref, and records the exact pre-merge
integration target for pull request `#1057`.

The implementation under verification and pinned PR head are
`a530fcc05de8080fd982ea2c3bf796c25cfd400f`. This identity contains the complete
review remediation and its regression tests before publication of this stable
conformance successor.

## Integration Target

| Field | Pinned identity |
|---|---|
| Repository | `potpie-ai/potpie` |
| Pull request | `https://github.com/potpie-ai/potpie/pull/1057` |
| Base ref | `main` |
| Base commit | `20a8389cabec6e5924b1e3d4ef12d1dcfe900a3c` |
| PR head ref | `refactor/context-runtime-boundary` |
| PR head commit | `a530fcc05de8080fd982ea2c3bf796c25cfd400f` |
| Implementation commit | `a530fcc05de8080fd982ea2c3bf796c25cfd400f` |
| Synthetic merge candidate | `6f1ba68e957b082635b3c2c1a381396cedb03a0d` |
| Merge-candidate parents | `20a8389cabec6e5924b1e3d4ef12d1dcfe900a3c`, `a530fcc05de8080fd982ea2c3bf796c25cfd400f` |
| Merge-candidate tree | `e74aaa84b885dc1751b60be855db11915981abf5` |

The locally reconstructed synthetic merge tree combines the pinned PR head
with the exact current `main` base. This record does not predict the eventual
merge commit; the pinned PR head and base commit are the durable pre-merge
identity.

The commit that publishes this stable record necessarily follows the pinned PR
head and cannot contain its own hash. Under `PROC-026`, that successor is a
publication boundary only while its intervening diff is limited to conformance,
specification governance, derived indexes, and conformance validation. Any
in-scope contract, runtime, test, or cited-evidence change requires a new
verification result.

## Behavior Trace

| Behavior ID | Implementation claim | Verification result | Evidence | Notes |
|---|---|---|---|---|
| SYS-001 | complete | passed | SYS-E1, SYS-E2 | Typed hosted call path remains intact. |
| SYS-002 | complete | passed | SYS-E1, SYS-E2 | Engine imports no root Potpie. |
| SYS-003 | complete | passed | SYS-E1, SYS-E2 | Hosts provide explicit identity and dependencies. |
| SYS-004 | complete | passed | SYS-E1, SYS-E2 | Potpie retains resource and lease ownership. |
| SYS-005 | complete | passed | SYS-E1, SYS-E2 | Daemon retains live runtime state. |
| SYS-006 | complete | passed | SYS-E1, SYS-E2 | CLI retains presentation ownership. |
| SYS-007 | complete | passed | SYS-E1, SYS-E2 | Failure categories remain distinct. |
| SYS-008 | complete | passed | SYS-E1, SYS-E2 | Authentication and authorization precede execution. |
| SYS-009 | complete | passed | SYS-E1, SYS-E2 | Confirmation and authorization remain distinct. |
| SYS-010 | complete | passed | SYS-E1, SYS-E2 | Hosting never retargets an engine. |
| SYS-011 | complete | passed | SYS-E1, SYS-E2 | Complete integration regressions preserve behavior. |
| SYS-012 | complete | passed | SYS-E1, SYS-E3 | Accepted contracts and prior records remain addressable. |
| SYS-013 | complete | passed | SYS-E1, SYS-E2 | Controller owns process creation. |
| SYS-014 | complete | passed | SYS-E1, SYS-E2 | Pre-endpoint creation uses the controller. |
| SYS-015 | complete | passed | SYS-E1, SYS-E3 | No public extension contract is introduced. |
| SYS-016 | complete | passed | SYS-E1, SYS-E3 | No external-host transport is published. |
| SYS-017 | complete | passed | SYS-E1, SYS-E3 | Claims remain in pinned conformance records. |
| SYS-018 | complete | passed | SYS-E1, SYS-E2 | Host ownership remains explicit. |
| SYS-019 | complete | passed | SYS-E1, SYS-E2 | Handler invokes engine after lease acquisition. |
| SYS-020 | complete | passed | SYS-E1, SYS-E2 | CLI intent remains untrusted until validated. |
| SYS-021 | complete | passed | SYS-E1, SYS-E2 | Readiness and operations use the typed client. |
| SYS-022 | complete | passed | SYS-E1, SYS-E2 | Resource Manager does not dispatch domain work. |
| SYS-023 | complete | passed | SYS-E1, SYS-E2 | Direct process creation remains canonical. |

## Reproducible Evidence

- **SYS-E1 — pinned full behavior conformance:** this cross-system record and
  its five linked module records passed all `195` current behaviors at
  implementation `a530fcc05de8080fd982ea2c3bf796c25cfd400f`.
  Evidence includes root result `1446 passed, 4 skipped, 1 deselected`, the
  independent Context Engine result `1152 passed, 32 skipped`, the
  Rust-dependent premerge journey result `1 passed, 1450 deselected`, all-file
  pre-commit, package builds, fresh-environment installs, isolated imports, and
  structural spec validation with zero warnings.
- **SYS-E2 — pinned PR-head checks:** live GitHub state rechecked on
  `2026-08-27` reported PR `#1057` open and mergeable at the pinned head, with
  all `19` reported checks successful. Regression workflow run
  `33047993868` includes pre-commit; Python 3.12, 3.13, and 3.14 root, Context
  Engine, Context package, parsing, and sandbox lanes; and the Rust-dependent
  `premerge-cli-journey`. Docs-check run `33047993939` and CodeRabbit also
  passed. GitHub still requires human review, which is a merge-governance gate
  rather than a conformance failure.
- **SYS-E3 — exact current PR/base reconstruction:** after fetching current
  `origin/main`, locally reconstructed merge candidate
  `6f1ba68e957b082635b3c2c1a381396cedb03a0d` resolves to the pinned base and PR
  head as its two parents and tree
  `e74aaa84b885dc1751b60be855db11915981abf5`. `git diff --check` passed for the
  exact reconstructed merge.
- **SYS-E4 — advanced-base impact and exact merge-tree check:** `main` advanced
  from `b45323127f81be40f07c44cab7f7581fda4a0ae7` to the pinned base through four
  documentation-dispatch files only: one workflow, `.pre-commit-config.yaml`,
  and two docs-check test files. There is no delta under `potpie/`, `spec/`,
  `pyproject.toml`, or `uv.lock`. The exact synthetic merge tree passed
  `git diff --check` and all `91` Node docs-check tests.

## Related Contracts Checked

| Spec ID | Revision | Spec ref | Result |
|---|---:|---|---|
| SPEC-GLOSSARY | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-PRODUCT | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-SYSTEM | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-CONTEXT-ENGINE | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-POTPIE-RESOURCE-MANAGER | 2 | a8c03337f3568232b35851dc2c86d128f7d23c0e | passed |
| SPEC-DAEMON | 2 | e73ebdbd6f0960e063344468051f84e37174697c | passed |
| SPEC-CLI | 1 | 047cbe067c9c726e7e14f066675453372d8a8406 | passed |
| SPEC-POTPIE-CAPABILITIES | 1 | b23b6cb9158e6b929f7d5f01c4dc0ca62a69f1df | passed |

## Conformance Records Checked

| Scope | Stable record | Implementation ref | Result |
|---|---|---|---|
| Context Engine | [CONF-CONTEXT-ENGINE](context-engine.md) | `a530fcc05de8080fd982ea2c3bf796c25cfd400f` | passed |
| Potpie Resource Manager | [CONF-POTPIE-RESOURCE-MANAGER](potpie-resource-manager.md) | `a530fcc05de8080fd982ea2c3bf796c25cfd400f` | passed |
| Daemon | [CONF-DAEMON](daemon.md) | `a530fcc05de8080fd982ea2c3bf796c25cfd400f` | passed |
| CLI | [CONF-CLI](cli.md) | `a530fcc05de8080fd982ea2c3bf796c25cfd400f` | passed |
| Potpie Capabilities | [CONF-POTPIE-CAPABILITIES](potpie-capabilities.md) | `a530fcc05de8080fd982ea2c3bf796c25cfd400f` | passed |

All five linked records are resolved from the same repository tree as this
cross-system record. Their immediately previous versions remain pinned through
their `previous_record` Git references.

## Known Gaps

None for the `23` active system behaviors or the pinned PR/base integration
target. PR `#1057` still requires human review and has not merged; this record
does not claim approval or merge completion.

## Aggregate Result

`passed`: all `23` active system behaviors have complete implementation claims
and passed verification, all five module records pin the same implementation
and accepted contract set, and the exact PR head passed integration checks
against the exact `main` base.

## Freshness

Freshness is derived by comparing the pinned specification, implementation,
PR head, `main` base, dependency, merge-tree, and evidence identities with the
selected current targets. A later in-scope contract, runtime, test, cited
evidence, or base change requires a successor at this same stable path. A
publication-only PR-head advance is handled by `PROC-026` and does not claim a
new implementation identity.
