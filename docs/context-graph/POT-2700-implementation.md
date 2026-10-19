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
| 8 | fix(graph): preserve claim incarnations and scoped supersession | Historical intervals and source/environment isolation | Approved by independent shared/adapter reviewer. Shared history and source policy, physical identity and endpoint guards; detailed local/native evidence below. |
| 9 | fix(graph): freeze correction targets and dependent effects | Preview/commit/verify targets identical; stale conflicts | Independent shared review approved. Engine core/unit 2265 passed 3 unrelated skips; Falkor server/Lite 472 passed 1 restart-opt-in skip; Neo 111 passed 1 restart-opt-in skip. Details below. |
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


## Step 8 contract freeze (2026-10-19)

Design reviewer: independent GPT-6.1 xhigh shared-contract reviewer, approved after resolving future interval overlap, explicit endpoint changes, and timestamp ownership.

- `ClaimWriteResolution` is the sole authority for selected incarnation ID, preserved interval, assertion effective time, and exact singleton predecessors. It is serialized only when present, preserving old fingerprint-v1 bytes. Prepared schema 2 distinguishes resolved assertions; schema 1 remains readable for completed receipt recovery and requires fresh preparation before unexecuted work.
- One pure policy operates over canonical `ClaimRow`s in preparation and raw backend atomic boundaries. Adapters validate frozen physical selections; they cannot select replacement targets on retry. Preparation maintains a virtual candidate set for earlier writes in the same batch.
- A historical endpoint at/before the frozen preparation/execution anchor prevents ordinary overlapping edits. Scheduled endpoints after the anchor survive refresh; at-end assertions create another incarnation. Explicit changed finite endpoints and truth-bearing changes require correction. Backdated assertions before an existing start and simultaneous/future singleton overlaps are rejected; no silent interval clipping.
- Singleton scope is pot, subgraph, subject, predicate, exact environment (missing is its own scope), source system and source reference. Both claims need an explicit stable-source marker; legacy missing markers and generated request/event references never authorize automatic replacement.
- Exact metadata review patches only the review marker allowlist and preserves content, observation time and lifecycle. Its guarded snapshot carries the physical record ID and full scope.
- Snapshot identity is physical `record_id`, with explicit legacy logical-key fallback until identity normalization. Multiple incarnations of one logical claim survive export/import.
- Shared closure preserves completed intervals and prevents empty/negative intervals. Journal and ordinary writers use this same policy.

Implementation uses one core lifecycle/codec/preparation owner. Native Cypher and
Falkor publication consume exact physical selections; memory staging uses the
same policy. Snapshot imports and journal inverse validation reject overlapping
logical incarnations, including legacy archives whose missing IDs would otherwise
invent a duplicate. Missing/retired endpoints and ambiguous physical IDs reject
before publication. Reactivation is explicit; omitted retirement metadata preserves
existing status. Different sources remain visible in quality reporting.

Step 8 validation evidence:

- Engine unit/core plus memory/JSON journal conformance: 2089 passed, 3 unrelated
  skips, 129 Falkor-named tests deselected; the native/Falkor gates below cover
  those adapters. Existing naive-datetime test emits one deprecation warning.
- Memory/JSON final owned incarnation/atomic/admin gate: 253 passed, covering
  raw/frozen identity collisions, retirement, restart, failure rollback and history.
- FalkorDB v4.16.0 server/Lite full gate: 275 passed, 64 deselected, no skips,
  including actual server restart. Final endpoint/journal/snapshot gate after
  shared history validation: 104 passed, 144 deselected. Writer/native vector shim:
  40 passed. Final staged entity/claim identity guard: 75 passed, one optional
  restart test unselected; 32 added native collision cases preserve graph/revision/receipts.
  Dedicated native tests exercise ordinary and journal modes.
- Neo4j 5.26 lifecycle/atomic/recovery gate after global identity guards: 50 passed,
  one restart opt-in test unselected; actual restart had already passed the earlier
  86-case native gate. Snapshot/entrypoint follow-up: 50 passed after the shared history checks.
- Root public CLI/runtime unit contracts: 257 passed. Canonical daemon integration:
  8 passed with loopback service access. The resource-without-query fixture now
  asserts the fail-closed reconciliation capability error introduced in step 7.
- Independent adapter reviewer: 122 native lifecycle cases and 73 focused identity,
  endpoint, annotation and occurrence-time cases passed. Shared snapshot/policy
  regressions: 64 passed. Ruff and diff whitespace checks pass.

Review findings remaining with their planned owners:

- I7/step9: freeze exact correction incarnations, intersect selector axes, and
  lower dependent/canonical SUPERSEDES effects centrally. Explicit empty target
  sets must never become wildcard selectors.
- I8/step10: native raw writers must preserve `event_received_at` observation time
  as memory does; occurrence time and frozen assertion time remain separate.
- I10/step12: ordinary entity updates must not overwrite immutable record IDs/UUIDs.
  Repair/import are explicit identity boundaries; migrate legacy duplicate fixtures
  through out-of-band setup instead of keeping an identity overwrite write door.

Final step 8 review approved after independently rerunning both Falkor staged-ID
collision reproductions twice. Each rejected with unchanged canonical data and
receipts, revision zero. Final legacy UUID promotion guard: 44 native cases passed.
General entity-ID overwrite and raw observation-time issues remain explicitly
assigned to steps 12 and 10; step 8 approval does not waive those release gates.

## Step 9 contract freeze (2026-10-19)


Status: APPROVED by independent thermo-nuclear design review; shared interfaces are being implemented. I7 primary; depends on I2/I5/I6/I8.

## One execution representation

- Add immutable `ClaimTarget(claim_key: str | None, record_id: str)` and `ClaimInterval(valid_at, invalid_at, valid_until)` plus `ClaimClosureResolution(target, before, after)`. Optional logical key preserves raw structural claims that lack semantic keys; semantic correction selection continues to require logical keys.
- Add `closures: tuple[ClaimClosureResolution, ...] | None` to existing `InvalidationOp` and `EdgeDelete`. None means unresolved raw input, empty tuple means frozen no claims. Existing target-key/record-ID fields are validated against closures, never competing authorities.
- Frozen closure execution validates exact identity and prior interval, applies the intended after interval, and never uses execution time to decide whether to close. Start remains immutable; positive intervals and nonoverlapping logical history are required. Explicit historical correction can change endpoints; ordinary raw deletion preserves closed intervals.
- Prepared schema 3 requires frozen closures and assertions. Versions 1/2 remain readable for unconditional completed-receipt recovery; unexecuted older preparation requires fresh identity. Receipt schema 2 and fingerprint version 1 remain unchanged. Serializers omit absent new fields to retain old batch digests.

## Preparation and selection

- Extend the existing preparation owner with one pure correction/effect planner over a consistent claim/entity read set. Semantic validation resolves exact target pairs; raw adapters use the same selector within their atomic boundary. No adapter-local correction policy.
- Semantic selectors add optional physical IDs and explicit target-source filters. Missing environment selects missing environment only. Evidence and correcting actor describe replacement provenance, never implicitly filter the old source.
- Endpoint/subgraph/source/environment/logical-key/physical-ID axes intersect. Reject missing requested IDs, duplicate physical IDs, key-only ambiguity between incarnations, and legacy semantic claims without identity. Explicit physical IDs allow selecting closed history. Resolved empty selections remain empty.
- Stage closures/deletes before preparing assertions against the resulting virtual state. Execute the same phase order: entity writes, closures/deletes, assertions. This prevents replacement assertion policy from reselecting the old active incarnation or closing it twice.
- Freeze one cutover time for supersede: old closes at that time, replacement starts there, and never inherits the old endpoint. If both supplied valid_from and valid_until imply different cutovers, reject and explain; separate operations can express an intentional gap. One generated preparation anchor supplies missing times.
- Lower SUPERSEDES bookkeeping to ordinary internal EdgeUpsert with frozen timestamps, source scope and references to exact old targets. Remove backend-generated SUPERSEDES paths. Shared structural validation permits system-generated effects; public semantic assertion still cannot author arbitrary system predicates.

## Projections and dependencies

- Semantic retract_claim targets claims. Native-only implicit entity stamping is removed. Any intentionally supported raw structural entity invalidation must become an explicit, reviewed EntityUpsert projection patch through the shared effect planner, with identity/revision fencing. Explicit empty claim selection produces no derived entity stamp or SUPERSEDES edge.
- Resource dependent selection carries record ID, logical key, environment, subgraph and source into each semantic retraction, deduplicating by physical ID rather than endpoint triple.
- Evidence-only conclusions stay active and receive exact metadata review. Compose graph review annotations with graph structure cleanup in the same prepared graph mutation where both result from one resource revision. Resource bytes/index/ledger retain their existing ordered workflow; no cross-store atomicity claim.
- Preview and minimal correction verification report exact target pairs/intervals; step 11 still owns full current verification states. Prepared descriptor target IDs and receipt affected IDs are populated from actual frozen effects.

## Acceptance and delegation

Primary owns DTOs/serialization, shared pure policy/preparation, semantic/resource lowering, preview/verification, shared Cypher and conformance cases. Backend workers consume the frozen contract and remove their duplicated selection/generation branches. No backend edits before review/freeze.

Required cases: later incarnation after preview conflicts; exact one among same-key physical rows; absent/prod/stage/source isolation; empty entity selection has no hidden effects; explicit closed-history endpoint correction; delayed execution across an endpoint uses frozen intent; supersede starts at cutover and remains live; resource dependents preserve each scope; failure after closure/system/projection work rolls back graph/revision/receipt. Include live native and ordinary/journal profiles.

Review questions: verify historical endpoint-extension semantics, raw entity projection compatibility, and whether a smaller immutable closure shape provides the same guarantee. Reject private execution branches, repeated timestamp ownership, and new wire fingerprints for old receipts.

## Accepted review clarifications

- Raw closure injection is rejected even for `()`. Prepared 3 requires every closure; old plans are read only for completed-receipt recovery.
- Canonical phases are entity writes, edge deletions, invalidations, assertions. Duplicate physical closure targets across operations are rejected.
- Explicit end T updates invalid_at=T and existing valid_until=T (otherwise None), retaining the start. Validate final touched history including deterministic stable-source singleton conflicts across objects; joint corrections may resolve overlap.
- SUPERSEDES identity includes the old physical ID and carries old/replacement references. Remove adapter-private generation.
- Combined resource graph review and cleanup are resolved lazily once, and pending review is cleared only after the combined authoritative receipt.

Step 9 projection clarification (independent design approval): EntityUpsert gains
`preserve_labels=False`, serialized only when true. A true projection carries no
authored labels and patches one existing entity, preserving registered/base/custom
labels. Canonical inference must not rewrite it. Generic label upsert/repair
semantics remain unchanged. This avoids weakening repair while making the derived
entity effect explicit in the frozen batch.


## Step 9 acceptance

Prepared schema 3 freezes closure target pairs and before/after intervals. Explicit
empty selections stay empty. Missing, duplicate or contradictory nonempty selectors
fail; derived selectors retain keyless IDs and unique logical keys across distinct
incarnations. Preparation-only typed supersession bindings attach exact replacement
IDs before fingerprinting, including raw keyless replacements. Snapshot/imported
legacy receipt bytes and fingerprint 1 remain unchanged.

Resource review markers, exact removed-section relationships and dependent claim
closures share the one graph mutation. Property-only entity projections explicitly
preserve labels and unmentioned display properties. External resource stores retain
the separately documented ordered workflow.

Independent thermo-nuclear review approved shared ownership after fixing source
association, duplicate/missing selectors, keyless repeated validation, native
cross-kind closure identity guards, and raw assertion final-history parity. The
resource facade shrank, and backend-specific correction selection was removed.

Validation: engine unit/core 2265 passed, 3 unrelated skips, one existing utcnow warning;
root public graph/resource/CLI recovery 201 passed; FalkorDB 4.16 server/Lite 472 passed,
1 optional restart skip, 64 non-Falkor cases deselected; Neo4j 5.26 native 111 passed,
1 optional restart skip. Actual restarts already passed steps 5/8. Independent reviewer
111 shared/resource/prepared cases plus 4 new native reproductions passed. No known
step 9 invariant blocker remains. Syntax/lint/whitespace gates pass.
