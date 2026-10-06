---
title: Graph write guarantees
description: "Optimistic concurrency per graph backend, mutation receipts and retries, exact correction targeting, and reset behaviour."
---

## Overview

> Status: reflects the typed-operation CLI, last reviewed 2026-10-06.

The graph workbench uses optimistic concurrency. A proposal records the pot's
current monotonic revision, and commit compares that revision while applying the
mutation. If another canonical write landed first, commit returns `conflict`
without applying the stale plan. The write flow itself is described in
[graph-workbench.md](./graph-workbench.md) and [writing.md](./writing.md).

## Built-in backends

| Backend | Compare-and-apply boundary | Restart behaviour |
|---|---|---|
| InMemory | One shared per-backend lock covers revision check, mutation, receipt, and revision increment. Definition-bound facades share that lock and revision map. | State is intentionally process-local. |
| Embedded | One process and interprocess file lock covers reload, receipt lookup, revision check, mutation, receipt persistence, revision increment, and atomic JSON replacement. | Graph revisions and mutation receipts persist in `graph.json`. |
| Native Neo4j | One datastore transaction owns the pot revision, mutation batch, and durable receipt. | Revision and receipt survive process restart. |
| Native FalkorDB and FalkorDB Lite | Redis `WATCH` detects concurrent graph writes; one `GRAPH.QUERY` within `MULTI`/`EXEC` applies the complete batch, revision, and receipt. A query error rolls back that query. | Revision and receipt live in the graph and follow the configured datastore persistence. Lite uses its append-only file. |

Every successful canonical mutation advances the pot revision, including
entity-only updates, invalidation, reset, and snapshot import. A completed
mutation receipt is checked before the expected revision, so retrying a known
successful mutation returns its original result even after later writes.

Custom or legacy injected writer adapters expose only the guarantees their
mutation adapter declares. An adapter with `atomic_mutations_supported = false`
uses the legacy process-local workbench guard and content token. That protects
one running workbench instance but is not a cross-process or crash-atomic
promise. Native compare-and-apply errors are not silently retried through this
fallback.

The boundary is the canonical mutation port. Direct database edits and operator
maintenance routines that bypass it require exclusive access; they are not
covered by the optimistic-concurrency contract. A datastore's own disk and
replication settings still determine durability after machine failure.

## Corrections and verification

Correction validation resolves an exact frozen claim set before commit. The
plan preview includes `retracted_claim_keys`, and the lowered invalidation keeps
the same `target_claim_keys`; environment and other claim identity fields are
therefore not rediscovered during apply. Verification reads those exact keys
back, checks the requested invalidation time, and compares asserted claim and
entity content with the lowered mutation. Runtime-generated provenance fields
and unrelated pre-existing properties do not cause false mismatches.

An idempotency key is bound atomically to its request fingerprint in persistent
plan storage. Reusing it with the same active or committed request returns the
existing plan. Reusing it with different content is rejected. An identical
request whose prior plan ended in conflict, error, or expiry may reserve a new
plan against the current revision.

A historical successful receipt proves that mutation was applied once. It does
not prove that its values are still current: later valid writes may supersede or
invalidate them. Run commit verification to compare the stored content with the
original mutation when current truth matters.

Reset advances the pot revision and preserves completed mutation receipts.
Retrying an old mutation after reset returns its historical receipt without
recreating the deleted data. Submit a new mutation ID when intentionally
restoring data after reset.

Time, validity and document-citation rules are in
[graph time and evidence](./graph-time-and-evidence.md).
