# 01 — Temporal resolution in search

## Problem

The `search` CLI and the underlying `POST /api/v2/context/query/search` already carry `valid_at`, `invalid_at`, `created_at` on each row, but:

1. They are **hidden behind `--with-temporal`** in human output; the default is a flat list with no temporal hint.
2. Graphiti does not automatically set `invalid_at` on a prior fact when a newer, contradicting fact is ingested. Concretely, after ingesting "migrated from MongoDB to Postgres on 2025-08-12" the older fact `The ledger collection is stored in MongoDB.` still returns `invalid_at: —`.
3. The ranking mixes invalidated and current facts together. An agent reading the top-5 can't tell "current" from "historical" without opting into flags.

### Evidence

CLI run in `temporal-test-1776692390`:

```
1. MODIFIED   MongoDB cluster … decommissioned   valid_at: 2025-08-12  invalid_at: —
2. STORED_IN  ledger collection stored in MongoDB valid_at: 2025-03-01  invalid_at: —   ← should be invalidated by (4)
4. MODIFIED   Ledger migrated MongoDB → Postgres  valid_at: 2025-08-12  invalid_at: —
```

Rows 2 and 4 contradict on the same subject (`ledger collection`/`Ledger service` datastore), yet row 2 is neither marked `invalid_at` nor downranked.

## Proposal

Three layers, independently shippable.

### Layer A — Surface temporal fields by default (small, do first)

- Flip `print_search_results` in `adapters/inbound/cli/output.py` so `valid_at` and `invalid_at` render in the default Panel when present (compact form: `valid 2025-08-12 • expired —`).
- Keep `--with-temporal` as the opt-in for the full triple including `created_at`.
- JSON output already includes the fields; no schema change.

### Layer B — Temporal ranking adjustments

In `application/use_cases/query_context.py` add a post-retrieval re-ranker:

- Rows with `invalid_at <= now` (or `<= as_of` when set) get demoted below all non-invalidated rows, not removed, unless `--include-invalidated` is set, in which case the tag `[superseded]` is added.
- Ties broken by `valid_at` descending (newer first).
- Add a response field `temporal_flag: "current" | "superseded" | "planned"` computed from `valid_at` / `invalid_at` vs. `as_of`.

### Layer C — Auto-invalidation on contradiction (the real fix)

Graphiti's fact extractor already produces new edges for "migrated from X to Y"; the gap is we don't close the old edges. Add a reconciliation step:

1. After each episode's extraction pass, compute a contradiction check for each new edge: same `(subject, predicate_family)` pair with a different object, where `predicate_family` groups edges like `STORED_IN`, `MIGRATED_TO`, `HOSTED_ON` into one "datastore-binding" family.
2. For every older edge matching the family, set `invalid_at = new_edge.valid_at`.
3. Record a `SupersededBy` relationship between old and new edge UUIDs so audit/history remains intact.

Ontology already has `open_conflicts` hooks — wire into `domain/graph_quality.py`.

## Files touched

- `app/src/context-engine/adapters/inbound/cli/output.py` — default temporal render.
- `app/src/context-engine/adapters/inbound/cli/main.py` — drop the `with_temporal` branch gate for `valid_at`/`invalid_at`.
- `app/src/context-engine/application/use_cases/query_context.py` — post-retrieval rerank + `temporal_flag`.
- `app/src/context-engine/domain/graph_mutations.py` — new `SupersededBy` mutation + predicate-family table.
- `app/src/context-engine/domain/graph_quality.py` — emit a `QualityIssue` when auto-invalidation fires.
- `app/src/context-engine/application/use_cases/reconciliation_validation.py` — call the contradiction detector on each ingestion batch.
- Tests: `app/src/context-engine/tests/integration/test_temporal_supersede.py` (new).

## Predicate-family table (first pass)

Start hand-curated, not learned:

| Family | Member edges |
|--------|--------------|
| `datastore_binding` | `STORED_IN`, `PERSISTS_TO`, `MIGRATED_TO` (target side) |
| `owner_binding` | `OWNS`, `OWNED_BY`, `MAINTAINED_BY` |
| `deployment_target` | `DEPLOYED_TO`, `RUNS_ON`, `HOSTED_ON` |
| `lifecycle_status` | `PROPOSED`, `IN_PROGRESS`, `COMPLETED`, `DEPRECATED`, `DECOMMISSIONED` |

Families live in `domain/ontology.py` so they evolve with the ontology version.

## Risks

- False invalidations when the old and new edge are scoped differently (e.g. different `service` but both touching `ledger`). Mitigation: include *all* edge properties in the subject key, not just the head node.
- Users who rely on the current ordering see behaviour change. Mitigation: this is new-default, not new-capability; `--include-invalidated` already documented.

## Rollout

1. Ship Layer A under a single PR — zero behaviour change beyond render.
2. Ship Layer B behind `CONTEXT_ENGINE_TEMPORAL_RERANK=1` for one week on staging, then flip on.
3. Ship Layer C with auto-invalidation **disabled by default** (`CONTEXT_ENGINE_AUTO_SUPERSEDE=0`), migrate one pot, verify the `open_conflicts` stream looks sane, then flip global default.

## Done when

- Default `potpie search "what database does the ledger use?"` returns Postgres facts above MongoDB facts in the same pot without flags.
- `invalid_at` is set on the MongoDB-storage edge within the same ingestion batch that introduces the Postgres-migration edge.
- `context_status` reports the auto-invalidation as a resolved `open_conflict`, not a pending one.
