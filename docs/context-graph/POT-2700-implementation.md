# POT-2700 implementation tracker

Baseline: `c6b306ba`. This document is the alignment and evidence ledger for
reliable graph mutations, corrections, and claim history. Implementation is
local to `graph_db_atomicity`; deployment and data migration require a separate
operational rollout after acceptance.

## Fixed invariants

| ID | Contract | Owning steps |
| --- | --- | --- |
| I1 | A pot-scoped request identity permanently binds semantic content. Replays reuse frozen preparation; changed content conflicts. | 3, 6, 7 |
| I2 | Graph changes, revision, and authoritative execution receipt commit together. No partial publication to readers. | 4, 5 |
| I3 | New nonempty batches advance once; empty batches receive a receipt without advancing; replay never advances; supported reset always advances and retains receipts. | 4, 5, 6 |
| I4 | Recovery checks authoritative execution first. Unknown/unsupported is not absent. Attempt ownership is compare-and-set fenced. | 3, 7 |
| I5 | Logical claim keys and incarnation record IDs are distinct. Closed intervals survive reassertion; ordinary and journal modes agree. | 8 |
| I6 | Singleton replacement is confined to pot/subgraph/subject/predicate/environment/source-system/source-reference. Competing sources remain visible. | 8 |
| I7 | Corrections freeze exact keys/incarnations and all dependent effects. Retry never reselects current targets. | 3, 9 |
| I8 | UTC half-open validity, separate occurrence/observation, filter before limit, frozen generated timestamps. | 10, 12 |
| I9 | Historical execution and current verification are separate; verification failure never hides or repeats a commit. | 11 |
| I10 | Corrupt stores fail closed. Migration preserves receipt fingerprints and does not invent history. | 7, 12 |
| I11 | Public sync/async contracts and backend capabilities are explicit; native guarantees need native evidence. | 3, 13 |

The user selected preserving competing sources. A missing environment is an
explicit scope, not a wildcard. Automatic supersession needs stable source
identity. Backdated changes to closed intervals require explicit correction.
Request reservation must be shared by all writers coordinating a pot.

## Commit sequence and gates

| Step | Commit subject | Gate | Status/evidence |
| --- | --- | --- | --- |
| 1 | test(graph): establish reliability contract and conformance harness | Existing baseline; isolated backend/process fixtures | 92 existing targeted tests + 4 new memory/JSON conformance tests pass; existing multiprocessing restart helper retained |
| 2 | refactor(graph): isolate preparation execution and verification | Existing behavior and approval/authorization tests | 116 workbench/quality/inbox/runtime/record/correction tests pass; preparation, execution, state and verification each have a canonical module |
| 3 | feat(graph): version prepared mutations and execution receipts | Stable preparation, fingerprint compatibility, explicit capabilities | 101 targeted tests pass; prepared v1, receipt v2, historical batch fingerprint v1 preserved; GPT-6.1 xhigh contract review findings addressed before adapter integration |
| 4 | fix(graph): atomically publish memory and embedded mutations | Rollback, observer failure, coherent/fresh reads | Pending |
| 5 | fix(graph): enforce native atomic execution and receipt metadata | Native concurrency, internal failure, restart replay | Pending |
| 6 | refactor(graph): unify supported graph write entry points | Entry-point inventory covered; repair/reset revision semantics | Pending |
| 7 | fix(graph): make plan persistence and public recovery crash-safe | Corruption fail-closed, crash recovery, stale owner fencing | Pending |
| 8 | fix(graph): preserve claim incarnations and scoped supersession | Historical intervals and source/environment isolation | Pending |
| 9 | fix(graph): freeze correction targets and dependent effects | Preview/commit/verify targets identical; stale conflicts | Pending |
| 10 | fix(graph): normalize temporal writes and query semantics | Offset/boundary/occurrence/query-limit parity | Pending |
| 11 | fix(graph): separate execution success from current verification | Current lifecycle results independent of execution receipt | Pending |
| 12 | feat(graph): add resumable legacy graph repair | Dry-run, resumability, ambiguity rejection, receipt preservation | Pending |
| 13 | test(graph): require full backend and public API acceptance | Live native matrix, supported Windows, independent review/QA | Pending |

Checkpoint A follows step 7, B follows step 11, release acceptance follows 13.
Each behavior change includes regression tests. No permanent expected failures.

## Write entry points and ownership

| Surface | Current owner | Required invariant |
| --- | --- | --- |
| Direct semantic sync/async mutation | application graph service | Same preparation and execution as plans |
| Propose/commit/retry | workbench service and plan store | Frozen preparation; native authoritative receipt |
| Reconciled events/raw batches | mutation backend and apply-plan writer | Atomic complete batch and stable supplied mutation identity |
| Invalidation | mutation backend | Canonical revisioned batch |
| Entity/label repair | analytics backend | Revisioned administrative mutation |
| Reset | mutation backend | Revision barrier; retained receipts; journal restrictions explicit |
| Journal restore/resource operations | journal port and commit service | Existing authorization, native receipt/revision atomicity |
| Readback/verification | claim query/workbench verification | Coherent revision; no implicit writes |

## Backend acceptance matrix

| Profile | Atomicity | Receipt durability | Required evidence |
| --- | --- | --- | --- |
| In-memory | Process-local staged publication | Lifetime of instance | Concurrent readers/writers; injected failure |
| Embedded JSON | Process lock and durable single-state replacement | Restart | Separate instances/processes; interrupted persistence; Windows |
| FalkorDB server | Native guarded command | Backend persistence configuration | Live concurrency, internal failure, restart |
| FalkorDB Lite | Native guarded command | Local persisted database | Live isolated embedded server, reopen |
| Neo4j | Native transaction | Backend persistence configuration | Live concurrency, transaction rollback, reconnect/restart |
| Custom adapters | Explicit advertised capability | Explicit advertised capability | Unsupported guarantees rejected, never silently upgraded |

Native tests are selected through `POTPIE_RELIABILITY_PROFILES`. Selecting a
server profile without its isolated test endpoint fails setup rather than
silently skipping a required backend. CI must select all supported profiles.

## Delegation and alignment

The primary owns shared ports/models, orchestration, shared Cypher, public APIs,
conformance harness, integration commits, and this ledger. Backend workers own
only their named adapters/readers/journal modules and dedicated tests. Maximum:
primary plus three agents. Contracts freeze before parallel backend edits.

Every assignment names invariant and step IDs, owned files, required scenarios,
and interface version. Workers must not revert other agents' edits or add private
compatibility branches. Shared changes return to the primary. Handoffs include
exact validation commands/results and unresolved risks. Only the primary stages
and commits intended files. Shell commands use `rtk`.

New findings are classified as invariant blockers, missing acceptance coverage,
or follow-up work. Blockers update dependent assignments before implementation.
Thermo-nuclear reviews at A/B/release reject duplicated ownership, silent
fallbacks, and complexity moved rather than removed. A passing suite or worker
completion claim cannot override a known invariant violation.

## Rollout and completion

Complete all code/test gates before operational migration. Back up, dry-run,
pause writes, upgrade all writers, apply checkpointed repair, validate identity
and receipt replay, then resume writes. Older writers must not return to the
migrated store. Rollback requires a coordinated validated backup restore or a
forward fix. Previously erased intervals remain explicitly unavailable.

Record local evidence separately from CI/native/platform evidence. No step is
complete merely because its code exists; limitations remain visible here.

## Shared contract freeze after step 3

`GraphMutationPlanRecord.prepared` is the canonical persisted intent descriptor;
`lowered_batch` owns its resolved content. The descriptor binds pot/mutation ID,
request and batch digests, expected revision and preparation timestamp. Validate
it before execution; provenance carries the descriptor only across the port.
Legacy plans/receipts keep `None` for unavailable metadata. New backend results
must carry `MutationResult.execution` with the actual execution timestamp,
before/after revision, and affected incarnation IDs.

Backend assignments must call `validate_prepared_execution` before touching data,
validate completed replays with `validate_execution_replay`, and build the receipt
with `complete_execution` inside their transaction/staging boundary. New execution
must enforce both the port's expected revision and the frozen preparation's
revision. Receipt lookup precedes revision rejection. `complete_execution` rejects
failed results and frozen-revision drift. Legacy batch digest bytes do not change.
No adapter may silently turn unsupported/unknown execution state into absence.

Contract review caught request identity replay checks, frozen revision checks,
strict metadata decoding, missing-vs-null legacy fields, and sync/async capability
parity. Each has a shared contract fix; adapter integration in 4/5 is still pending.
Subagents use GPT-6.1 with xhigh reasoning as requested by the user.
