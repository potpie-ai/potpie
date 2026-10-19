------------------------------ MODULE GraphWrite ------------------------------
(***************************************************************************)
(* Pot graph write door behind async ContextEngine.                        *)
(*                                                                         *)
(* One step here is one indivisible critical section in the code: the      *)
(* in-memory state_lock, the embedded JSON file lock, the FalkorDB         *)
(* WATCH/MULTI/EXEC, or the Neo4j revision-node transaction. FalkorDB and  *)
(* FalkorDB Lite share falkordb_atomic.apply_atomic.                       *)
(*                                                                         *)
(* Journal capture (an active graph journal, restore, resource guards) is  *)
(* not this model. See spec/graph-write/README.md.                         *)
(***************************************************************************)

EXTENDS Integers

CONSTANTS MaxRevision, SkipRevisionCheck, SkipReceiptLookup

Mids == {"m1", "m2"}
Bodies == {"A", "B"}
Writers == {"w1", "w2"}
Claims == {"c1", "c2"}
\* c1 is the claim key fixed when the correction was previewed.
\* c2 is another live claim on the same edge that preview did not select.
Frozen == {"c1"}
NULL == "null"

VARIABLES
  revision, \* pot revision. compare_and_apply calls this expected_version.
  live,     \* claim keys currently live
  slot,     \* last fresh body that landed. "none" after reset.
  receipts, \* mutation id -> body fingerprint, or NULL
  op,       \* per writer: none | commit | mutate | correct
  snap,     \* revision observed when that writer planned
  mid,
  body

vars == <<revision, live, slot, receipts, op, snap, mid, body>>

Init ==
  /\ revision = 0
  /\ live = Claims
  /\ slot = "none"
  /\ receipts = [m \in Mids |-> NULL]
  /\ op = [w \in Writers |-> "none"]
  /\ snap = [w \in Writers |-> 0]
  /\ mid = [w \in Writers |-> "m1"]
  /\ body = [w \in Writers |-> "A"]

\* Receipt lookup shared by compare_and_apply and mutation.apply.
\* SkipReceiptLookup mirrors deleting that lookup.
Gate(w) ==
  IF SkipReceiptLookup \/ receipts[mid[w]] = NULL
  THEN "absent"
  ELSE IF receipts[mid[w]] = body[w]
  THEN "replay"
  ELSE "reject"

Release(w) == op' = [op EXCEPT ![w] = "none"]

Plan(w, chosenOp, chosenMid, chosenBody) ==
  /\ op[w] = "none"
  /\ op' = [op EXCEPT ![w] = chosenOp]
  /\ snap' = [snap EXCEPT ![w] = revision]
  /\ mid' = [mid EXCEPT ![w] = chosenMid]
  /\ body' = [body EXCEPT ![w] = chosenBody]
  /\ UNCHANGED <<revision, live, slot, receipts>>

\* compare_and_apply: receipt first, then the planned revision, then one write.
CompareAndApply(w) ==
  /\ op[w] \in {"commit", "correct"}
  /\ LET gate == Gate(w)
     IN \/ /\ gate \in {"replay", "reject"}
           /\ Release(w)
           /\ UNCHANGED <<revision, live, slot, receipts, snap, mid, body>>
        \/ /\ gate = "absent"
           /\ revision # snap[w]
           /\ ~SkipRevisionCheck
           /\ Release(w)
           /\ UNCHANGED <<revision, live, slot, receipts, snap, mid, body>>
        \/ /\ gate = "absent"
           /\ \/ revision = snap[w]
              \/ SkipRevisionCheck
           /\ revision < MaxRevision
           /\ revision' = revision + 1
           /\ IF op[w] = "correct"
              THEN /\ live' = live \ Frozen
                   /\ slot' = slot
              ELSE /\ live' = live \cup (IF body[w] = "A" THEN {"c1"} ELSE {"c2"})
                   /\ slot' = body[w]
           /\ receipts' = [receipts EXCEPT ![mid[w]] = body[w]]
           /\ Release(w)
           /\ UNCHANGED <<snap, mid, body>>

\* mutation.apply / apply_async. No planned revision is passed in.
\* FalkorDB retries until the write lands at whatever revision is current.
\* Neo4j, embedded JSON, and in-memory skip the version check.
MutateApply(w) ==
  /\ op[w] = "mutate"
  /\ LET gate == Gate(w)
     IN \/ /\ gate \in {"replay", "reject"}
           /\ Release(w)
           /\ UNCHANGED <<revision, live, slot, receipts, snap, mid, body>>
        \/ /\ gate = "absent"
           /\ revision < MaxRevision
           /\ revision' = revision + 1
           /\ live' = live \cup (IF body[w] = "A" THEN {"c1"} ELSE {"c2"})
           /\ slot' = body[w]
           /\ receipts' = [receipts EXCEPT ![mid[w]] = body[w]]
           /\ Release(w)
           /\ UNCHANGED <<snap, mid, body>>

\* reset_pot. Bumps the revision, drops live claims, keeps receipts.
\* Refuses while a journal is active; that refusal is outside this model.
Reset ==
  /\ revision < MaxRevision
  /\ revision' = revision + 1
  /\ live' = {}
  /\ slot' = "none"
  /\ UNCHANGED <<receipts, op, snap, mid, body>>

Next ==
  \/ \E w \in Writers, chosenOp \in {"commit", "mutate", "correct"},
          chosenMid \in Mids, chosenBody \in Bodies:
       Plan(w, chosenOp, chosenMid, chosenBody)
  \/ \E w \in Writers: CompareAndApply(w)
  \/ \E w \in Writers: MutateApply(w)
  \/ Reset

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* Properties the atomic compare_and_apply / reset / correction door keeps.

\* A compare or correction whose receipt is absent commits only at its snap.
\* A removed revision check makes a stale writer bump the revision here.
StaleCompareRejected ==
  [][
    \A w \in Writers:
      /\ op[w] \in {"commit", "correct"}
      /\ op'[w] = "none"
      /\ revision # snap[w]
      /\ receipts[mid[w]] = NULL
      => revision' = revision /\ live' = live /\ slot' = slot
         /\ receipts' = receipts
  ]_vars

\* Same mutation id and same body returns the stored receipt and writes nothing.
ReplayIsStable ==
  [][
    \A w \in Writers:
      /\ op[w] # "none"
      /\ op'[w] = "none"
      /\ receipts[mid[w]] = body[w]
      /\ receipts[mid[w]] # NULL
      => UNCHANGED <<revision, live, slot, receipts>>
  ]_vars

\* Same mutation id and a different body is rejected and writes nothing.
DifferentBodyRejected ==
  [][
    \A w \in Writers:
      /\ op[w] # "none"
      /\ op'[w] = "none"
      /\ receipts[mid[w]] # NULL
      /\ receipts[mid[w]] # body[w]
      => UNCHANGED <<revision, live, slot, receipts>>
  ]_vars

\* A stored fingerprint is never replaced or dropped.
ReceiptsStick ==
  [][
    \A m \in Mids: receipts[m] # NULL => receipts'[m] = receipts[m]
  ]_vars

\* Reset is the only step that bumps the revision without retiring a writer.
ResetShape ==
  [][
    revision' = revision + 1 /\ op' = op =>
      /\ live' = {}
      /\ slot' = "none"
      /\ receipts' = receipts
  ]_vars

\* A correction that lands removes only the keys fixed at preview.
CorrectionPrecision ==
  [][
    \A w \in Writers:
      op[w] = "correct" /\ op'[w] = "none" /\ revision' = revision + 1 =>
        /\ (live \ live') \subseteq Frozen
        /\ (live' \ live) = {}
  ]_vars

-----------------------------------------------------------------------------
\* Promise the mutate door does not keep. Checked by GraphWritePromise.cfg.
\* A fresh mutate built at snap still lands after the revision has moved.

MutateHonorsPlannedRevision ==
  [][
    \A w \in Writers:
      /\ op[w] = "mutate"
      /\ op'[w] = "none"
      /\ receipts[mid[w]] = NULL
      => revision' = revision \/ revision = snap[w]
  ]_vars

=============================================================================
