# 06 — Conflict surfacing

## Problem

The six-episode test set contains an obvious direct contradiction:

- **2025-01-15** `ledger collection is stored in MongoDB` (source: `design-doc`)
- **2025-08-15** `Ledger migrated from MongoDB to Postgres` + `MongoDB cluster decommissioned` (source: `release-notes`)

`context_status` is supposed to return an `open_conflicts` list (Phase 7 `graph_quality.py`), but today it does not flag this pair. `search` returns both rows side-by-side with no tag. An agent that reads the first three rows could mis-report the current datastore.

## Proposal

Three-part: detection, persistence, surfacing.

### Detection

Run after each reconciliation batch. Inputs: set of newly-applied edges in the batch. For each new edge:

1. Look up the `predicate_family` (see #01 family table).
2. Query the graph for all other live edges on the same `(subject, family)` with a different object.
3. Compare `valid_at`:
   - Newer edge supersedes older → if #01 is live, auto-invalidate and no conflict. Otherwise file conflict.
   - Overlapping validity windows → conflict of type `overlap`.
   - Same `valid_at` + different object → conflict of type `contradiction` (human resolution required).

### Persistence

Use the canonical `QualityIssue` node already introduced in Phase 7. Properties:

```
kind: "conflict"
severity: "warning" | "blocking"
family: "datastore_binding"
subject_uuid: <node>
edge_a_uuid: <older edge>
edge_b_uuid: <newer edge>
detected_at: <timestamp>
auto_resolvable: bool
suggested_action: "supersede_older" | "human_review"
```

Link via `FLAGS` edge from `QualityIssue` to the involved edges (pattern already in the ontology).

### Surfacing

1. **`context_status`** — `open_conflicts` list now actually populated; each item includes both edge UUIDs, the subject, and `suggested_action`.
2. **`search`** — rows involved in an open conflict get a visual tag in CLI render: `[!] conflict with row N`. In `--json` mode, rows gain `conflict_ids: [<quality_issue_uuid>, …]`.
3. **`context_resolve`** — `quality.conflicts` array, referenced from `recommended_next_actions` when `conflict.auto_resolvable=false`.

## Files touched

- `app/src/context-engine/domain/graph_quality.py` — `detect_family_conflicts()` implementation.
- `app/src/context-engine/application/use_cases/reconciliation_validation.py` — invoke detector per batch.
- `app/src/context-engine/domain/graph_mutations.py` — `QualityIssueCreate` + `FlagsEdge` mutation types.
- `app/src/context-engine/application/services/context_resolution.py` — populate `quality.conflicts` in responses.
- `app/src/context-engine/adapters/inbound/http/api/v1/context/router.py` — `context_status` returns `open_conflicts` from graph, not placeholder.
- `app/src/context-engine/adapters/inbound/cli/output.py` — conflict tag rendering.
- `app/src/context-engine/adapters/inbound/cli/main.py` — new subcommand `potpie conflict list` / `potpie conflict resolve <id> --action supersede_older`.
- Tests: integration test with the 6-episode fixture asserting the datastore contradiction is detected.

## Interaction with #01 and #02

- If #01 auto-supersedes on ingest, direct temporal contradictions never become conflicts — they become resolved supersessions. Only **overlap** or **same-timestamp** cases surface as conflicts. That is the right split: conflicts should represent actual human-resolution-needed cases, not normal state transitions.
- If #02 ships first, the predicate families are better populated because edge types are specific rather than `MODIFIED`.

## Risks

- Noisy conflicts when the family table is too coarse. Mitigation: start with three families (datastore_binding, owner_binding, deployment_target), measure false-positive rate, expand.
- Performance: one family-query per new edge. For batches of N edges this is N Cypher queries; acceptable at current volume, revisit when batch size grows.

## Rollout

1. Land detector and `QualityIssue` writes behind `CONTEXT_ENGINE_CONFLICT_DETECT=1`, detection only (no CLI surface yet).
2. Backfill on one pot, inspect the `QualityIssue` nodes manually, tune family table.
3. Expose in `context_status` and `search`; add the `conflict` CLI subcommand.
4. Move flag default to on.

## Done when

- After ingesting the 6-episode fixture, `potpie context-status` returns ≥ 1 open conflict *unless* #01 auto-supersedes, in which case `quality.resolved_conflicts` shows the supersession.
- `potpie search "ledger"` visually tags the MongoDB-storage row and the Postgres-migration row as being in an open conflict (or as resolved/superseded, per #01).
- `potpie conflict resolve <uuid> --action supersede_older` closes the issue and writes an `invalid_at` onto the older edge.
