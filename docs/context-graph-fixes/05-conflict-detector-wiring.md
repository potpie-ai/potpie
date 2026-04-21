# Fix 05 — Conflict detector never surfaced same-time contradictions

## Symptom (historical)

`potpie conflict list` and `potpie conflict resolve` responded cleanly, but forced **same-`valid_at`** predicate-family contradictions (e.g. two different `STORED_IN` targets at one instant) produced zero open `QualityIssue` rows.

| Case | Expectation | Failure mode |
|------|-------------|--------------|
| `STORED_IN Bar→Postgres` and `STORED_IN Bar→MySQL` at the **same** `valid_at` | `QualityIssue` with `conflict_type: contradiction` | Often empty: see root cause below |
| `CHOSE` at `t1` + `MIGRATED_TO` at `t2` (cross-family) | `supersession_pending` or auto-supersede | Usually OK (distinct times) |
| `STORED_IN Foo→Postgres` at `t1` + `STORED_IN Foo→Redis` at `t2 > t1` | Temporal supersession (not an open conflict) | OK |

## Root cause

Conflict detection **was implemented** (`domain.graph_quality.detect_family_conflicts`, `adapters/outbound/graphiti/family_conflict_detection.py`, wired from `GraphitiEpisodicAdapter.add_episode_async`), and the HTTP/CLI paths read real Neo4j state.

The bug was **ordering in `apply_predicate_family_auto_supersede`** (`adapters/outbound/graphiti/temporal_supersede.py`), which runs **before** `apply_family_conflict_detection` on each ingest.

For a bucket with the same predicate-family subject and **different object endpoints**, the old logic picked a single “winner” (including UUID tie-break when `valid_at` matched) and **invalidated every other edge**. That left at most **one** live edge, so `detect_family_conflicts` never saw a pairwise contradiction.

**Fix:** auto-supersede now:

- Invalidates edges **strictly older** than the latest `valid_at` / `created_at` in the bucket (temporal supersession).
- If multiple edges share the **same** max timestamp but point at **different** objects, it **does not** invalidate any of them — they stay live for `detect_family_conflicts` to emit a `contradiction` record.
- Buckets with **no** timestamps at all are skipped (left for conflict detection / human review).

## Actual architecture (not reconciliation batch wiring)

The fix note in older drafts mentioned `reconciliation_validation.py` and structural mutations. In this codebase, episodic conflicts are **Graphiti / Neo4j**:

| Layer | Role |
|-------|------|
| `adapters/outbound/graphiti/temporal_supersede.py` | Predicate-family auto-supersede (must not eat same-time contradictions) |
| `adapters/outbound/graphiti/family_conflict_detection.py` | Loads live `RELATES_TO` rows, runs `detect_family_conflicts`, creates `QualityIssue` nodes |
| `adapters/outbound/graphiti/episodic.py` | After `add_episode`: supersede → canonical labels → **family conflict detection** |
| `adapters/inbound/http/api/v1/context/router.py` | `POST .../conflicts/list`, `POST .../conflicts/resolve` |

**CLI:** `potpie conflict list` → `POST /api/v2/context/conflicts/list` with `{ "pot_id": "<uuid>" }` (same auth as other v2 context routes).

## Persistence and queries

- Nodes use **`group_id`** (pot scope), not `pot_id` on the label.
- Example inspection:

```cypher
MATCH (qi:Entity:QualityIssue {group_id: $pot_id, kind: 'conflict', status: 'open'})
RETURN qi ORDER BY qi.detected_at DESC LIMIT 10
```

## Feature flags

- `CONTEXT_ENGINE_CONFLICT_DETECT` — conflict detection + persistence (default **on**).
- `CONTEXT_ENGINE_AUTO_SUPERSEDE` — temporal auto-supersede (default **on**).

## Resolve path

`potpie conflict resolve <issue_uuid> --action supersede_older` calls the server resolver, which closes the `QualityIssue` and stamps `invalid_at` on the **older** episodic edge by effective time (same `valid_at` falls back to stable ordering).

Optional future UX: require `--edge-to-keep` when `conflict_type == contradiction` so the operator explicitly chooses which edge survives.

## Tests

- `tests/integration/test_temporal_supersede.py` — same `valid_at`, two objects → **0** invalidations; distinct times → older edge invalidated.
- `tests/integration/test_family_conflict_detection.py` — mocked Neo4j driver + `detect_family_conflicts` pipeline.
- `tests/unit/test_graph_quality.py` — `detect_family_conflicts` semantics.

## Done when

- After ingesting two conflicting same-time bindings in one predicate family, `potpie conflict list` shows an item with both edge UUIDs and `conflict_type: contradiction` (or the API returns it under `items`).
- After ingesting a **strictly later** fact that supersedes an older layer, older edges are invalidated by auto-supersede and do not spuriously appear as open conflicts.

## Dependencies

- Cross-family / datastore hint behavior (fix 02) remains shared via `predicate_family_for_episodic_supersede` and related helpers in `domain/ontology.py`.
