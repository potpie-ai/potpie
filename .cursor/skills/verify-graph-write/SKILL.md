---
name: verify-graph-write
description: "Check the pot graph write door after a change to compare_and_apply, mutation.apply, reset_pot, mutation receipts, or correction claim keys. Covers FalkorDB, FalkorDB Lite, Neo4j, embedded JSON, and in-memory, behind async ContextEngine. Use when those files change, when a counterexample needs reading, or when the model may no longer match the code."
---

# Verify the pot graph write door

The model in `spec/graph-write/GraphWrite.tla` checks the atomic write shared by the FalkorDB, FalkorDB Lite, Neo4j, embedded JSON, and in-memory backends. A passing contract run means every interleaving inside the stated bounds keeps the properties below. It does not mean the rest of the engine is proved.

Read `spec/graph-write/README.md` before editing the model. If covered code changed and nobody re-ran this skill for that change, stop and use `/maintain-formal-verification-skill`.

## When a change touches this model

It touches the model when it edits any file in the coverage record, or adds a new writer of the same pot revision, receipts, or `target_claim_keys`. A comment, a log line, or a read path that does not change those steps does not.

## Run

From the repo root:

```bash
make check-graph-write
```

Java only. The script downloads pinned `tla2tools.jar` (TLA+ release `v1.8.0`, sha256 `15d103b7e5617780bcc904c0af12f1ad242e2b345c0301136c5be4588f17edac`, TLC banner `2026.10.15.145915`).

- Exit 0: the contract properties passed, and `MutateHonorsPlannedRevision` failed with the known finding. A green run does not mean `mutation.apply` honors the planned revision.
- Non-zero on the first TLC invocation: a contract property failed. Treat it as a bug until you have replayed the trace and found a step the code cannot take.
- Non-zero after the promise check, with text that the promise passed: the mutate path now matches the promise, or the model no longer describes it. Update the model from the code, then decide whether the property belongs in `GraphWrite.cfg`.

## Properties

Contract, in `GraphWrite.cfg`. These pass on the current code:

- `StaleCompareRejected`. A new `compare_and_apply` commits only at the revision it planned on.
- `ReplayIsStable`. The same mutation id and body returns the stored receipt and does not write.
- `DifferentBodyRejected`. The same mutation id and a different body is rejected and does not write.
- `ReceiptsStick`. A stored fingerprint is not replaced or deleted.
- `ResetShape`. Reset bumps the revision, clears live claims, and keeps receipts.
- `CorrectionPrecision`. A correction that lands removes only the claim keys fixed at preview.

Promise, in `GraphWritePromise.cfg`. This fails on the current code. Leave the failure in place:

- `MutateHonorsPlannedRevision`. A fresh `mutation.apply` commits only at the revision observed when it was planned.

The known failure, as calls:

1. `GraphService.mutate` validates and lowers at revision 0. Claims `c1` and `c2` are live. Mutation id `m1` has no receipt.
2. `reset_pot` moves the revision to 1 and deletes live claims. Receipts are unchanged.
3. `mutate` calls `mutation.apply` with no planned revision. FalkorDB retries at the new revision. Neo4j, embedded JSON, and in-memory skip the comparison. Body `A` lands, `c1` is live again, and the revision is 2.

Harm: a mutate, and `ContextEngine.record` without an idempotency key, writes a batch that was built against a graph the reset deleted. The same gap writes over a revision a concurrent `compare_and_apply` already moved. Do not weaken `MutateApply` or this property to turn the check green.

## Tell a bug from a modeling error

Replay the printed states in order against the functions in the coverage record.

- A step that the code cannot take is a modeling error. Fix the action so one indivisible code section stays one action. A lock, a file lock, a FalkorDB `WATCH`/`EXEC`, or a Neo4j transaction is one action. Do not split it.
- A trace whose every step can happen is a bug in the code. Report it. Do not change product code unless the user asks. Do not delete the property and do not constrain the action so the trace disappears.
- `MutateHonorsPlannedRevision` is the second kind. It is a finding, not drift.

## Update the model when the code change is intended

1. Read the diff of every covered file. Map each changed branch to an action in `GraphWrite.tla`.
2. Change the action so it matches the new code. Change a property only when the promise changed on purpose, and cite the doc comment, test, or commit message that says so. Write down the old wording and the new wording.
3. Run `make check-graph-write`.
4. If you added or moved a check, show TLC can still fail. Set `SkipRevisionCheck` or `SkipReceiptLookup` to `TRUE`, or delete the new branch in a scratch copy of the action, and confirm the matching property fails. The last scratch that was reverted: the revision comparison was removed from `_Mutation.compare_and_apply`, `EmbeddedGraphBackend._apply_once`, `falkordb_atomic.apply_atomic`, and `neo4j_atomic.apply_atomic`, `SkipRevisionCheck` was set `TRUE`, and TLC violated `StaleCompareRejected` (commit planned at revision 0, reset to 1, `compare_and_apply` still wrote). Restore the code and set the constant back to `FALSE`. Re-run until the script is back to a passing contract and the known mutate finding.
5. Move the sync commit below to the commit you just checked. Do that only after this run. Do not move it for a check you could not run.

## Coverage

Sync commit: `c6b306ba77c0ca4a6b06eaa0fa3ccaa226bf109f`

The model was confirmed against these files at that commit. Product code was not changed.

| File | Functions |
| --- | --- |
| `potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/falkordb_atomic.py` | `apply_atomic`, `reset_pot`, `lookup_execution` |
| `potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/backends/falkordb_backend.py` | `_compare_and_apply_sync`, `compare_and_apply`, `apply_async`, `reset_pot`. `FalkorDBLiteGraphBackend` uses this class |
| `potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/neo4j_atomic.py` | `apply_atomic`, `reset_atomic`, `lookup_execution` |
| `potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/backends/neo4j_backend.py` | `compare_and_apply`, `apply_async`, `reset_pot` |
| `potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/backends/in_memory_backend.py` | `_Mutation.compare_and_apply`, `_Mutation.apply`, `_Mutation.reset_pot`, `_Mutation._apply_invalidations` |
| `potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/backends/embedded_backend.py` | `EmbeddedGraphBackend._apply_once`, `_EmbeddedMutation.compare_and_apply`, `_EmbeddedMutation.apply`, `_EmbeddedMutation.reset_pot` |
| `potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/_mutation_execution.py` | `MutationExecutionRegistry.lookup`, `MutationExecutionRegistry.execute` |
| `potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/cypher.py` | `apply_invalidations_async` |
| `potpie/context-engine/src/potpie_context_engine/core/correction_targets.py` | `resolve_correction_targets` |
| `potpie/context-engine/src/potpie_context_engine/core/semantic_mutation_lowering.py` | `lower_semantic_request` (copies `correction_targets` onto `target_claim_keys`) |
| `potpie/context-engine/src/potpie_context_engine/core/workbench_service.py` | `GraphWorkbenchService.commit`, `_apply_if_current` |
| `potpie/context-engine/src/potpie_context_engine/application/services/graph_service.py` | `GraphService.mutate`, `GraphService.record` |
| `potpie/context-engine/src/potpie_context_engine/application/services/record_capture.py` | `record_once` |
| `potpie/context-engine/src/potpie_context_engine/context_engine.py` | `ContextEngine.commit`, `ContextEngine.mutate`, `ContextEngine.record`, `ContextEngine.reset_context` |

## Bounds and what is out of the model

Revision `0` through `2`. Two writers, two mutation ids, two bodies, two claim keys. 43177 distinct contract states. One action is one lock or transaction, as named above.

Not covered, so a pass says nothing about them: an active graph journal and its restore path, the plan-store update after a successful backend commit, approval and expiry, legacy non-atomic writers, and a mid-batch `JournalError` inside in-memory `_apply_uncached`. Details are in `spec/graph-write/README.md`.
