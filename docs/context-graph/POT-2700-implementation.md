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
| 4 | fix(graph): atomically publish memory and embedded mutations | Rollback, observer failure, coherent/fresh reads | 93 focused integration tests pass; worker broader gate 176 passed/31 unselected-native skips; 35 new rollback/concurrency/restart cases. Observer publication race required canonical CAS-loss result handling. Windows evidence remains step13. |
| 5 | fix(graph): enforce native atomic execution and receipt metadata | Native concurrency, internal failure, restart replay | FalkorDB v4.16.0/Lite: 121 passed, 64 deselected; server AOF restart and Lite reopen pass. Neo4j 5.26: 31 passed plus actual restart test, 18 unit tests. Fresh-pot Falkor revision race fixed under native WATCH. Forced-kill durability and full platform matrix remain step13. |
| 6 | refactor(graph): unify supported graph write entry points | Entry-point inventory covered; repair/reset revision semantics | 1736 unit/core passed (3 unrelated skips), plus 10 isolated Falkor wiring tests. Neo4j 80 passed; Falkor server/Lite 154 passed. Native restart guards unselected here; actual restarts proven in step5. Direct semantics share prepared workbench; empty commits get receipts; reset/import/repair share native execution; reset returns retry identity. |
| 7 | fix(graph): make plan persistence and public recovery crash-safe | Corruption fail-closed, crash recovery, stale owner fencing | Checkpoint A approved by independent shared and backend reviewers. Engine unit/core + public runtime/semantic gate: 1929 passed, 3 unrelated skips. Root CLI/runtime gate: 146 passed; final reset gate: 16 passed. Native integrity: 25 passed across server/Lite/Neo4j. Detailed evidence below. |
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
| Reset | mutation backend | Revision barrier; retained receipts; returned execution identity; journal restrictions explicit |
| Snapshot import | snapshot port prepares; mutation backend executes | Canonical identity-preserving payload; atomic exact merge, receipt and revision |
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
parity. Adapter integration in 4/5 and checkpoint A validation are complete.
Subagents use GPT-6.1 with xhigh reasoning as requested by the user.


## Findings tracked during implementation

- Step6 invariant blocker for step12: a repair delta above the existing 5000-entity batch cap must use resumable revisioned batches. Current behavior fails before writing; a native 5001-entity regression proves no partial changes. Do not truncate silently or add a private cap bypass.
- Step7 blockers: fail-closed plan parsing, durable replacement, defensive copies in the process-local plan store, receipt lookup before workflow status/takeover checks, and original execution timestamp/revision recovery. Read-only audit independently confirmed each issue.
- Cross-store resource purge remains the existing ordered workflow. The graph transaction does not claim to atomically commit external resource bytes or ingestion-ledger deletion.

## Checkpoint A acceptance (2026-10-19)

Both independent GPT-6.1 xhigh reviews approved their scopes after the concrete
findings were fixed and reproduced against the fixes. The thermo-nuclear review
required one execution coordinator, one receipt-binding validator, and source
effects resolved by the existing preparation owner.

| Invariant | Implementation and regression evidence | Review |
| --- | --- | --- |
| I1/I4 | `mutation_preparation`, `mutation_execution_service`, `mutation_state`; permanent reservation, receipt-first recovery, frozen request/batch metadata, attempt CAS. `test_mutation_recovery_contract`, `test_prepared_mutation_contract`, `test_plan_storage_reliability`. | checkpoint_a_shared approved; focused final gate 75 passed |
| I2/I3/I10 | Strict native receipt/revision readers reject missing or regressed revisions, including modern empty receipts at zero; duplicate nested JSON fails before interpretation. `test_falkordb_revision_integrity` and `test_neo4j_revision_integrity`. | checkpoint_a_backends independently ran 25 native cases, all passed |
| I10 | Shared strict JSON decoder and fsync/replacement helper; plan and embedded stores reject corruption, invalid schema, duplicate keys, and disappearance after observation. `test_embedded_metadata_integrity` plus plan-storage subprocess crash cases. | memory/embedded gate 236 passed, 31 unselected native skips; independent focused gate 135 passed |
| I1/I4/I11 | Resource revisions bind stable desired intent. A host-only resolver discovers removed live sections/dependencies inside the preparation revision bracket and is not called on replay. Delete uses one manifest snapshot, requires applied graph status, and fences byte deletion by revision. `test_resource_mutation_recovery`: 40 cases across graph/store combinations. | checkpoint_a_shared approved; eight concurrent imports converge on one plan and revision |
| I4/I11 | Typed `recover_stale=True` default with explicit opt-out; capability declarations survive runtime bridging. Typed reset returns identity/execution; recovery never repurges newer resources and reports unknown external purge outcome. `test_public_graph_runtime`, root `test_graph_commit_recovery` and `test_local_engine`. | checkpoint_a_shared approved; reset contradiction cases pass |

Native backend evidence uses FalkorDB v4.16.0 with AOF persistence, Lite on macOS,
and Neo4j 5.26 community. The revision-floor change passed actual FalkorDB server
restart/Lite reopen and Neo4j container restart. Final duplicate-decoder native
regressions passed without another restart; decoder changes do not alter storage.
Receipt-floor validation currently scans a pot's receipts. An indexed floor is
follow-up performance work requiring a migration design, not a silent bypass.

The acceptance reviewer generated genuine baseline `c6b306ba` graph/plan files
in an isolated child and reopened them with this implementation: the original
mutation identity/fingerprint survived, unavailable metadata remained `None`,
and graph bytes were unchanged. This does not claim restoration of erased history.

Resource bytes/index/ledger remain separate ordered stores. A graph receipt proves
graph execution only. Resource purge failures retain the reset receipt; a reset
retry explicitly reports `unknown_on_recovery` for purge rather than deleting
resources written after the original reset. Conflicted source preparations remain
bound to their original identity and require an explicit fresh operation.

Remaining release evidence: forced-kill durability, full supported Windows/native
CI, and lifecycle/temporal/correction/verification/migration work in steps 8–13.
