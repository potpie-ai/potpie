# Fix 02 — Cross-family auto-supersede

Completes **plan #01 Layer C** ([temporal resolution in search](../context-graph-improvements/01-temporal-resolution-in-search.md)): edges that belong to the same **predicate family** must share one temporal bucket per logical subject even when Graphiti’s `RELATES_TO.name` strings differ—e.g. `CHOSE` (ledger → MongoDB) and `MIGRATED_TO` (same ledger → Postgres). Unambiguous names such as `STORED_IN` and `MIGRATED_TO` were already mapped together via `PREDICATE_FAMILY_EDGE_NAMES`; this fix adds label-aware resolution so overloaded names like `CHOSE` can join that family without broad over-invalidation.

## Symptom

In the fresh test pot, both of these edges were `temporal_flag: current` with `invalid_at: —`:

```
CHOSE        2025-01-15  The team chose MongoDB as the ledger datastore…
MIGRATED_TO  2025-08-12  Ledger was migrated from MongoDB to Postgres on 2025-08-12.
```

They contradict: same subject entity, same “where does this ledger persist?” semantics. Graphiti’s built-in supersession aligns rows that share the **same** normalized edge name (for example two `STORED_IN` rows with different targets). It does **not** merge edges whose LLM relation names differ unless post-ingest predicate-family supersede groups them under a shared **family id**.

## Root cause

1. **`CHOSE` was not in `datastore_binding`** for lookup via `predicate_family_for_edge_name("CHOSE")` (returns `None`)—by design: `CHOSE` is overloaded (vendor, tool, datastore, …) and must not always bucket with datastore edges.

2. **Edge-name-only bucketing is insufficient.** Even if `CHOSE` were naively added to the static family table, grouping **only** by relation name would risk over-invalidation. Resolution must consider **target node labels** when the name is ambiguous.

## Fix (implemented)

### 1. Predicate family with label-aware `CHOSE`

In `app/src/context-engine/domain/ontology.py`:

- **`predicate_family_for_episodic_supersede(edge_name, target_labels)`** — returns a family id for temporal bucketing, conflict detection, and auto-supersede. Non-`CHOSE` edges delegate to `predicate_family_for_edge_name` (backed by `PREDICATE_FAMILY_EDGE_NAMES`).
- For **`CHOSE`**: join **`datastore_binding`** only when the target’s labels include a canonical **`DataStore`** label after `canonical_entity_labels(...)` (intersection with ontology `ENTITY_TYPES`). If there is no qualifying canonical label on the target, the function returns `None` (no cross-type supersede for that edge—same spirit as name-only behavior until labels exist).
- **`object_counterparty_uuid_for_edge`** and **`temporal_subject_key_for_edge`** accept optional **`predicate_family=...`** so callers pass the resolved family when the edge name alone would not map (e.g. `CHOSE` with hints).

### 2. Post-ingest supersede (Graphiti / Neo4j)

In `app/src/context-engine/adapters/outbound/graphiti/temporal_supersede.py`:

- The live-edge Cypher returns **`labels(b) AS target_labels`** (target endpoint of each `RELATES_TO`).
- Rows are filtered with **`predicate_family_for_episodic_supersede(name, target_labels)`** instead of membership in a precomputed edge-name-only union.
- Bucketing uses the resolved **`family`** on each row consistently for subject/object keys and `EpisodicSupersessionRecord` audit.

**Toggle:** `CONTEXT_ENGINE_AUTO_SUPERSEDE` (default on; set to `0` / `false` / `off` to disable). Optional filter: **`CONTEXT_ENGINE_AUTO_SUPERSEDE_FAMILIES`** — comma-separated family ids; when set, only those families are processed.

**Call site:** invoked from `app/src/context-engine/adapters/outbound/graphiti/episodic.py` immediately after `add_episode(...)` succeeds (same `group_id` as the pot). Failures are logged and do not fail the ingest.

### 3. Conflict detection alignment

In `app/src/context-engine/domain/graph_quality.py` and `app/src/context-engine/adapters/outbound/graphiti/family_conflict_detection.py` (conflict feature flag: `CONTEXT_ENGINE_CONFLICT_DETECT`, default on):

- **`EpisodicEdgeConflictInput`** includes optional **`target_labels`**.
- **`detect_family_conflicts`** uses **`predicate_family_for_episodic_supersede`** and the same **`predicate_family=`** wiring on `temporal_subject_key_for_edge` / `object_counterparty_uuid_for_edge` so detection and auto-supersede stay consistent.

### 4. Out of scope for this fix

Structural reconciliation (`app/src/context-engine/application/use_cases/reconciliation_validation.py`) does **not** drive episodic `RELATES_TO` supersede; the module docstring points to Graphiti + ontology instead. No new `SupersededBy` edge between edge-uuid nodes—existing pattern stamps **`invalid_at`** / **`superseded_by_uuid`** on the episodic relationship, with **`EpisodicSupersessionRecord`** for audit.

### 5. Related code references

| Concern | Location |
|--------|-----------|
| Family table + resolver | `app/src/context-engine/domain/ontology.py` (`PREDICATE_FAMILY_EDGE_NAMES`, `predicate_family_for_episodic_supersede`) |
| Auto-supersede | `app/src/context-engine/adapters/outbound/graphiti/temporal_supersede.py` |
| Conflict inputs + detection | `app/src/context-engine/domain/graph_quality.py`, `app/src/context-engine/adapters/outbound/graphiti/family_conflict_detection.py` |
| Post-ingest orchestration | `app/src/context-engine/adapters/outbound/graphiti/episodic.py` |

## Tests

| Test file | What it covers |
|-----------|------------------|
| `app/src/context-engine/tests/unit/test_ontology.py` | `CHOSE` + `DataStore` hint, `temporal_subject_key_for_edge(..., predicate_family=...)` |
| `app/src/context-engine/tests/unit/test_graph_quality.py` | `detect_family_conflicts` for `CHOSE` vs `MIGRATED_TO` |
| `app/src/context-engine/tests/integration/test_temporal_supersede.py` | Mocked driver: env toggle, same-type supersede, cross-type `CHOSE` vs `MIGRATED_TO` |

## Dependencies

- **Labels at supersede time:** `apply_predicate_family_auto_supersede` runs **before** `apply_episodic_canonical_labels` in `app/src/context-engine/adapters/outbound/graphiti/episodic.py`. For `CHOSE`, **`DataStore` must already appear on the target node** when `labels(b)` is read—usually from Graphiti extraction with `ENTITY_TYPES`, or from `canonical_type` / maintenance backfill. `EDGE_ENDPOINT_INFERRED_LABELS` in `app/src/context-engine/domain/ontology.py` does not currently infer `DataStore` from `CHOSE` alone; [fix 04](04-canonical-node-labels-write-path.md) widens reliable labeling end-to-end.
- **`PREDICATE_FAMILY_EDGE_NAMES`** remains shared with conflict detection ([fix 05](05-conflict-detector-wiring.md)).

## Risks

- **Over-grouping** if `CHOSE` were always treated as datastore — mitigated by requiring **`DataStore`** (or future expanded hints in `_DATASTORE_CHOOSE_TARGET_LABEL_HINTS`) on the target for `CHOSE` only.
- **Under-grouping** when multiple labels are present: any qualifying canonical match on the target puts `CHOSE` in **`datastore_binding`** (intentional default).

## Rollout

1. Deploy ontology + supersede + conflict updates (behavior changes only when targets carry **`DataStore`** where required for `CHOSE`, and **`CONTEXT_ENGINE_AUTO_SUPERSEDE`** is on).
2. Backfill labels / re-ingest where needed; older `CHOSE` edges should gain **`invalid_at`** when a later **`MIGRATED_TO`** (or other datastore-family edge) wins on the same subject.
3. Confirm search JSON: the older row shows **`temporal_flag: superseded`** and not both rows **`current`**.

## Done when

- Ingesting `CHOSE` → MongoDB at `t1` with target labeled **`DataStore`**, followed by `MIGRATED_TO` → Postgres at `t2 > t1` on the same subject, results in the older edge carrying **`invalid_at`** aligned with the winner (newer `valid_at`, then `created_at`, then stable uuid tie-break per `_winner_sort_key` in `temporal_supersede.py`).
- `potpie --json search "ledger" -n 10` shows the **`CHOSE`** row with **`temporal_flag: superseded`** when applicable.
- Search does not show both rows as **`current`** for that contradiction once supersede has run.
