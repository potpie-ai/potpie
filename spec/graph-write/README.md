# Pot graph write model

This is a small [TLA+](https://lamport.azurewebsites.net/tla/tla.html) model of one promise: a pot's graph write either lands as a whole, or it does not land. [TLC](https://lamport.azurewebsites.net/tla/tools.html) checks every interleaving inside the bounds below. You can read a failure as a sequence of calls. You do not need to write TLA+ to act on one.

The model is `GraphWrite.tla`. It describes the code at commit `c6b306ba77c0ca4a6b06eaa0fa3ccaa226bf109f`. It is not a second design.

## What the model represents

One pot. Two writers. Each writer plans a write, then runs it. A plan is one of:

- `commit` or `correct`, the `compare_and_apply` door used by `GraphWorkbenchService.commit`
- `mutate`, the `mutation.apply` door used by `GraphService.mutate`

The pot has a revision, the set of live claim keys, a slot naming the body that last landed, and a receipt for each mutation id. A receipt stores the body fingerprint. `c1` is the claim key a correction fixed at preview. `c2` is another live claim on the same edge that preview did not select.

A step is one indivisible critical section in the code. The in-memory backend holds `state_lock` across the check and the write. The embedded JSON backend holds the file lock across load, check, write, and replace. FalkorDB and FalkorDB Lite run the batch, the revision bump, and the receipt insert in one `GRAPH.QUERY` under `WATCH`. Neo4j does the same inside one transaction that locks the `PotpieRevision` node. TLC does not interleave inside those sections.

## How each action maps to code

| Action | Code |
| --- | --- |
| `Plan` | The writer has read the pot at `snap`. For a commit, that read is `current_version` stored as `expected_subgraph_versions`. For a mutate, it is the claim read inside `GraphService.mutate` before `apply`. The code does not store that revision on the mutate path. |
| `CompareAndApply` | `compare_and_apply` on all four atomic backends. Receipt lookup comes first. The same mutation id and the same body returns the stored result and does not write. The same id and a different body raises `MutationExecutionReuseError`. A new id commits only when the pot revision equals the planned revision, then bumps the revision and stores the receipt. `SkipRevisionCheck` deletes that comparison. `SkipReceiptLookup` deletes the lookup. Both constants are `FALSE` in the checked model. |
| `MutateApply` | `GraphService.mutate` calls `backend.mutation.apply`. FalkorDB `apply_async` passes `expected_version=None` and retries until the write lands at the revision it reads then. Neo4j `apply_atomic` skips the version comparison when `expected_version` is `None`. Embedded `_apply_once` compares only when `expected_version` is set. In-memory `apply` never compares. |
| `Reset` | `reset_pot` / `reset_atomic`. Deletes live pot data, adds one to the revision, leaves receipts in place. |

Backend functions behind `CompareAndApply` and `Reset`:

- FalkorDB and FalkorDB Lite: `falkordb_atomic.apply_atomic`, `falkordb_atomic.reset_pot`, reached from `FalkorDBGraphBackend._compare_and_apply_sync`
- Neo4j: `neo4j_atomic.apply_atomic`, `neo4j_atomic.reset_atomic`
- In memory: `_Mutation.compare_and_apply`, `_Mutation.apply`, `_Mutation.reset_pot`
- Embedded JSON: `EmbeddedGraphBackend._apply_once`, then the in-memory apply, persisted by `_write_state`

A correction's effect is `_Mutation._apply_invalidations`, `cypher.apply_invalidations_async`, and the invalidation clause in `falkordb_atomic._compile`. Each of those retires `target_claim_keys` and does not also walk the edge. `resolve_correction_targets` binds those keys once, and `lower_semantic_request` copies them onto the invalidation before commit.

## How to run it

From the repo root:

```bash
make check-graph-write
```

That is `scripts/check-graph-write.sh`. It needs Java. On first run it downloads `tla2tools.jar` from the TLA+ `v1.8.0` release into `spec/graph-write/.tools/` and checks sha256 `15d103b7e5617780bcc904c0af12f1ad242e2b345c0301136c5be4588f17edac`. The jar reports itself as TLC `2026.10.15.145915`. Nothing else is installed.

The script runs two checks. The contract check must pass. The promise check must fail `MutateHonorsPlannedRevision`. That failure is a finding about `mutation.apply`, recorded below. A contract failure, or a promise check that suddenly passes, means the model and the code have diverged and the script exits non-zero.

Bounds: revision `0..2`, writers `{w1, w2}`, mutation ids `{m1, m2}`, bodies `{A, B}`, claims `{c1, c2}`. The contract search visits 43177 distinct states and finishes in about a second. `-deadlock` is set because a writer left planned at the revision bound is the bound, not a stuck protocol.

## How to read a counterexample

TLC prints `State 1`, `State 2`, and so on. The angle brackets name the action. Only the variables that matter change:

- `revision` moved, or a writer committed against a `snap` that is not `revision`
- `receipts[m1]` changed from `null` to a body, or a second use of `m1` did not
- `live` lost `c2` on a correction, or gained claims after `Reset` cleared it
- `slot` is the body that landed. `none` means reset wiped it

Read the trace as calls, in order. If a step cannot happen in the code, the model is wrong. If every step can happen, the code breaks the property named at the top of the error.

The promise check fails with this trace:

1. `Plan(w1, mutate, m1, A)` at revision 0. `GraphService.mutate` validates and lowers while claims `c1` and `c2` are live. There is no receipt for `m1`.
2. `Reset`. `reset_pot` moves the revision to 1 and deletes live claims. Receipts stay empty.
3. `MutateApply(w1)`. `mutate` calls `mutation.apply`, which does not receive revision 0. The write lands at the new revision, stores receipt `m1 -> A`, and `c1` is live again. The revision is 2.

Harm: a mutate, including `ContextEngine.record` without an idempotency key, was built against the graph that reset deleted, and the backend writes it anyway. The same missing check lets that write land on top of a commit that moved the revision after planning.

`ContextEngine.record` with an idempotency key does not take this door. `record_once` goes through `propose` and `commit`, which call `compare_and_apply`.

## How to add a property

Add a temporal formula next to the others in `GraphWrite.tla`. `[][P]_vars` means every real step satisfies `P`. Primed variables (`revision'`) are the values after the step. Quantify over `Writers` when the step belongs to one writer, and make the antecedent name that writer (`op[w]` before, `op'[w]` after). A property that freezes the whole pot whenever any writer is stale will fail because the other writer is still allowed to run.

List the formula under `PROPERTY` in `GraphWrite.cfg` if the code keeps it, or in `GraphWritePromise.cfg` if you are recording a promise the code breaks. Then run `make check-graph-write`.

Show that the new property can fail before you trust it. Set `SkipRevisionCheck` or `SkipReceiptLookup` to `TRUE` in `GraphWrite.cfg`, or delete the corresponding branch in the action, and confirm TLC prints a violation. Put the constant back. The revision-check scratch, mirrored in the four `compare_and_apply` implementations and then reverted, violates `StaleCompareRejected`: a commit planned at revision 0 still lands after reset has moved the pot to revision 1.

## Finding

`MutateHonorsPlannedRevision` fails on the code as it is. Do not edit the action or the property to make that check pass. The contract properties pass.

## What this does not cover

- An active graph journal: `FalkorJournal`, `MemoryJournal`, and `EmbeddedJournal`, including restore and the resource guard. `reset_pot` refuses while a journal is active.
- The plan-store compare-and-set around `GraphWorkbenchService.commit`. A crash after the backend commits and before the plan row updates is recovered by replaying the receipt. That second state machine is not in the model.
- Approval, expiry, and post-commit verification.
- The contents of a batch beyond "the fresh body lands" and "a correction retires only the frozen keys".
- A legacy FalkorDB or Neo4j writer with `atomic_mutations_supported` false. That path keeps receipts in a process-local `MutationExecutionRegistry`.
- In-memory `_apply_uncached` if a later verb raises `JournalError` after an earlier verb has already edited the store. The write door does not create the duplicate live rows that raise it. FalkorDB, Neo4j, and the embedded file replace roll the batch back when the write does not finish.
- Anything past revision 2, or more than two writers, two mutation ids, or two claim keys. A pass covers this model inside those bounds.
