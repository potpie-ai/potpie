---
title: Reliable mutations, recovery and legacy repair
description: "Execution guarantees, retry and verification semantics, supported backends, and the coordinated legacy repair rollout."
---

# Reliable mutations, recovery and legacy repair

Graph execution is complete only when the backend has an authoritative receipt.
Graph changes, the pot revision and that receipt share one atomic boundary. Plan
status is recoverable bookkeeping. A lost response or cancelled caller can leave
a completed execution; retry the same returned identity to recover its receipt.

## Identity and lifecycle

A request key is scoped to a pot and permanently binds its semantic body. Reusing
it with changed content fails. Preparation persists resolved operations, exact
claim incarnations, expected revision, generated times and versioned fingerprints.
A conflict or expired unexecuted preparation requires a fresh proposal and identity.
All writers coordinating requests for a pot must share the reservation store.
Separate local plan files do not provide distributed request idempotency.

Each newly accepted nonempty mutation advances the pot revision once. An empty
prepared mutation receives a receipt without advancing it. Replays retain their
original revisions and timestamp. Supported reset always advances the revision,
keeps receipts and invalidates outstanding preparations; replay cannot restore
reset data. Existing restrictions in journal mode still apply.

Snapshot imports default to a content-derived identity, so repeating an import
recovers its original receipt even after reset. To intentionally restore the same
snapshot again, pass a fresh identity:
`potpie graph import <snapshot> --pot <pot> --mutation-id <fresh-id> --yes`.
Typed callers use `ImportSnapshotRequest(mutation_id=..., ...)`; both path and
inline imports, including resource archives, forward that identity. Reuse it only
to retry that specific import.

A newly executed reset purges configured document resources, including when its
identity was supplied by the caller. Recovering a completed reset receipt never
purges again, because newer documents may now exist. The public result preserves
the graph receipt and reports resource purge status separately. `pot reset` and
`pot archive` fail on incomplete teardown; archive marks the pot archived only
after teardown completes.

`claim_key` identifies a logical claim and `record_id` its physical incarnation.
Retract then reassert creates another incarnation and preserves the earlier
interval. Automatic singleton replacement requires the same pot, subgraph,
subject, predicate, environment, source system and stable source reference.
Missing environment is a separate scope. Competing sources coexist and are
reported as conflicts until explicitly corrected. Corrections freeze exact
incarnations and dependent effects during preparation.

Times use UTC and half-open validity intervals. Observation time records when the
system learned a fact; occurrence/validity time can be earlier or later. Backdated
changes to already closed intervals require explicit correction.

## Execution and verification

Typed commit requests default `recover_stale=True`. Completed receipt lookup is
always performed. Active attempts remain pending; stale takeover first checks
authoritative state. Setting `False` disables takeover, not receipt recovery.
Unknown or unsupported state never authorizes blind replay. Older workers cannot
replace a recovered workflow attempt through a late finalization.

Top-level commit success describes execution. Current-state verification has a
separate `not_requested`, `deferred`, `completed` or `unsupported` state. Detailed
results distinguish matching, superseded/retracted, mismatched and inconclusive
content at a checked revision. Later corrections never rewrite the execution
receipt. Verification does not execute the mutation again. Current-state readback
of raw legacy storage repair is explicitly unsupported/inconclusive; its execution
receipts remain available.

Use async public operations inside an active event loop. The runtime retains its
existing bridge and driver ownership. Sync legacy repair inside a running event
loop rejects the call before work starts; use `repair_legacy_async` or the typed
async `repair(RepairRequest(...))` operation.

## Review and apply legacy repair

```sh
potpie graph repair --legacy --dry-run --pot <pot>
potpie --json graph repair --legacy --dry-run --repair-id <returned-id> --pot <pot>
potpie graph repair --legacy --apply --repair-id <returned-id> --pot <pot> --yes
```

The first command reserves an immutable reviewed repair. Repeating dry-run with
its ID returns that frozen review. Human output shows at most 20 changes and 20
findings; use JSON to inspect the complete run before apply. Preview requires read
access; apply requires admin access and destructive confirmation. These flags
cannot be combined with other repair targets. Typed callers use
`RepairRequest(legacy_mode="dry_run")`, then
`RepairRequest(legacy_mode="apply", repair_id=...)` with confirmation through the
operation client.

The run owns bounded ordinary child mutations and a 24-hour preparation lifetime.
Each child advances the revision only when it does new work. Resume with the same
repair ID after a response loss, cancellation, process crash or backend restart:
completed children recover their original receipts and remaining children retain
their frozen targets and revision expectations. An unrelated intervening write
conflicts. Expiry leaves unexecuted children unapplied; create a new dry-run and
review it. A completed run remains replayable after expiry or reset. A second
fresh scan of repaired data performs no duplicate work.

Repair can normalize unambiguous offset timestamps and backfill missing physical
record IDs. It preserves valid existing IDs, UUIDs, receipt identities and stored
fingerprints. It does not assign logical claim keys or reconstruct erased history.
Naive/invalid times, unsupported precision, contradictory aliases or scope,
nonpositive intervals and duplicate identities require explicit resolution and a
fresh review. Corrupt or unreadable plan storage fails closed; do not delete it to
force execution. Restore a validated backup or reconcile authoritative receipts.

Raw storage repair is not reversible through the typed journal restore schema.
Journal activation validates physical identities and requires repair when they
are missing; activation does not silently migrate graph data.

## Backend and custom-adapter guarantees

| Backend | Atomic publication | Receipt lifetime | Validation environment |
| --- | --- | --- | --- |
| In-memory | Staged state under the shared process lock | Running store instance | Unit/conformance, Windows/Linux/macOS selections |
| Embedded JSON | Process lock plus flushed durable file replacement | Persisted store across processes/restarts | Separate instances, interrupted persistence, Windows CI |
| FalkorDB server | Native WATCH/guarded command | Depends on configured server persistence | Live pinned 4.16.0; AOF restart/forced-kill fixtures |
| FalkorDB Lite | Native guarded command | Local persisted database | Supported Linux/macOS; real reopen/process restart |
| Neo4j | Native managed transaction | Persistent database | Live pinned 5.26 Community; rollback/restart |

The test configuration is not evidence that a particular deployment enables
persistence. Configure and validate server durability before relying on receipts
across restart. Memory makes no cross-process durability claim. Neo4j does not
provide the optional journal API; existing unsupported journal operations remain
explicit. FalkorDB Lite is not selected on Windows.

Custom mutation adapters must advertise `MutationCapabilities` accurately and
implement atomic compare-and-apply, revision and authoritative lookup contracts
(or their async equivalents). A process-local lock cannot substitute for a native
multi-writer transaction. Unsupported atomic execution fails before writes;
unsupported/ambiguous recovery does not become absence. Repair additionally needs
a coherent lossless `LegacyInventoryPort`/async equivalent with transaction limits,
plus atomic `RepairRunStorePort` manifest-and-child reservation. Ordinary plan
stores still work without repair, but cannot silently claim repair support.

Injected raw writers used by the backend adapters must implement scoped
`claim_rows`: reviewed record identities, touched subject/predicate histories and
unresolved correction selectors. Return complete history within those scopes,
including closed incarnations; an empty scope returns no rows. The mutation
planner fetches any additional histories needed by corrections before writing.

## Operational rollout

1. Complete required backend/platform CI and independent acceptance reviews.
2. Back up graph state, plan storage and journal data; test that the backup is
   restorable. Run dry-run and resolve blocking findings.
3. Pause writes and stop or disable every old daemon, ingestion job and binary
   that can access the store. Revoke old writer credentials/access where needed.
   Upgrade all writers together. The schema marker alone is not a downgrade fence:
   some older writers ignored it, and native stores have no writer-version fence.
4. Apply the reviewed repair with writers excluded. If preparation expired,
   generate and review a fresh run while writes remain stopped.
5. Validate counts, record/UUID ownership, representative temporal queries and
   original receipt replay, then enable only upgraded writers.
6. Use existing status/logging surfaces for conflicts, unresolved recovery,
   corruption, migration-required errors and deferred verification.

Do not resume older writers against migrated data. Recover with a forward fix or
validated full-backup restoration while all writers are stopped. An uncoordinated
binary downgrade is unsupported. Historical intervals erased before this change
remain unavailable.
