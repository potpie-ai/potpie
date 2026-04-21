# Context graph fixes (post-implementation test pass, 2026-04-21)

After the first implementation pass of plans 01–06 in [`../context-graph-improvements/`](../context-graph-improvements/README.md), we re-ran the six-query harness against two pots and captured what works, what regressed, and what's missing. Each bounded fix lives in its own file here.

## Scorecard at time of testing

| # | Plan | Landed | Remaining |
|---|------|--------|-----------|
| 01 | Temporal resolution | default render, `temporal_flag` + rerank, same-edge auto-supersede | cross-family auto-supersede missing |
| 02 | Edge-type collapse | specific edges on new ingests (`CHOSE`, `MIGRATED_TO`, `CAUSED`, `DECOMMISSIONED`) | graceful downgrade missing — unknown labels/edges reject the whole batch |
| 03 | Canonical node labels | — | `--node-labels Decision/Service/…` returns 0; only base `Entity` matches |
| 04 | Causal / multi-hop | `CAUSED` extraction + surfacing in search | — |
| 05 | Provenance in CLI | default render, `--episode`, `--no-provenance`, server-side `--source` filter | — |
| 06 | Conflict surfacing | CLI scaffolding (`conflict list` / `resolve`) | detector never fires; 0 `QualityIssue` rows on forced contradictions |

## Fix files

| # | File | Blocks |
|---|------|--------|
| 01 | [`01-surface-ingest-validation-errors.md`](01-surface-ingest-validation-errors.md) | Meta-fix. Unblocks debugging of 02–05. |
| 02 | [`02-cross-family-auto-supersede.md`](02-cross-family-auto-supersede.md) | Completes plan #01 Layer C. |
| 03 | [`03-graceful-ontology-downgrade.md`](03-graceful-ontology-downgrade.md) | Recovers episodes currently being silently dropped. |
| 04 | [`04-canonical-node-labels-write-path.md`](04-canonical-node-labels-write-path.md) | Delivers plan #03 end-to-end. |
| 05 | [`05-conflict-detector-wiring.md`](05-conflict-detector-wiring.md) | Makes plan #06 actually surface contradictions. |

## Ordering and risk

1. **01 (surface the error)** — ship alone first. Small, safe, makes every other fix debuggable.
2. **03 (downgrade vs reject)** — restores data currently being silently dropped.
3. **02 + 04** — share the `PREDICATE_FAMILIES` table and the label-inference test fixture; natural to ship together.
4. **05** — depends on 02's family table.

## Reusable test fixture

Every fix in this folder is validated against the same six-episode fixture from the original plans:

```
2025-01-15  design-doc       "chose MongoDB because familiar"
2025-03-20  incident-review  "heavy write contention, 40+ min aggregations (scaling pain caused datastore rethink)"
2025-04-10  adr-0042         "Alice + Ravi decided to migrate ledger to Postgres"
2025-08-15  release-notes    "migration complete 2025-08-12, Mongo decommissioned"
2025-11-02  pr-1287          "auth middleware session token leak fixed in v2.14.1 by Priya"
2026-04-01  planning-doc     "Q2 2026 priorities: context-engine v1, deprecate legacy recon API, OTel on ingest"
```

All six episodes must land (0 rejected) and the canonical queries from plans 01–06 must pass their "done when" assertions before this folder is closed.

## Evidence captured during the 2026-04-21 run

Keep raw runs out of git but link bug-report style findings here as they arise:

- Silent ingest failure example (see fix 01): server-side `event show` returned `ontology validation failed: adr:0042: unknown canonical labels: ADR; DECIDED_BY: unknown canonical edge type; …` while the CLI only showed `Server returned no episode UUID`.
- Cross-family contradiction miss (see fix 02): `CHOSE MongoDB` at `2025-01-15` and `MIGRATED_TO Postgres` at `2025-08-12` on the same ledger subject — both rows returned with `temporal_flag: current`, no `invalid_at`.
- Label query miss (see fix 04): across 8 candidate labels, only `Entity` (Graphiti base) returned results. `Decision`, `Service`, `Incident`, `Release`, `Component`, `Datastore`, `Person`, `Fix` all `0`.
- Conflict detector silence (see fix 05): forced same-`valid_at` contradiction (`Bar→Postgres` vs `Bar→MySQL`) produced zero `QualityIssue` rows.
