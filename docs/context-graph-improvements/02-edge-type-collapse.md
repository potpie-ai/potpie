# 02 — Edge-type collapse onto `MODIFIED`

## Problem

The Graphiti extractor is emitting `MODIFIED` for a wide and semantically distinct set of actions:

- "Ledger service was *migrated* from MongoDB to Postgres" → `MODIFIED`
- "OpenTelemetry spans *will be added* to the ingest path" → `MODIFIED`
- "MongoDB cluster … *decommissioned*" → `MODIFIED`
- "Q2 2026 priorities *include* deprecating the legacy API" → `MODIFIED`

A reader looking at the edge label alone cannot tell *completed work* from *planned work* from *removed resource*. Downstream reasoners (causal tracing, staleness, lifecycle filters) degrade to "read the summary string" which is fragile.

## Proposal

Two complementary moves.

### Move 1 — Tighten extractor outputs (not just prompts)

Graphiti exposes an edge-type schema per pot. Rather than prompting in free text, register an explicit allow-list for this pot-family:

```
MIGRATED_TO, DEPRECATED, DECOMMISSIONED, PLANNED, COMPLETED,
OWNS, FIXES, CAUSED, DEPENDS_ON, REPLACES, ADDED_TO, REMOVED_FROM,
DECIDES_FOR, STORED_IN, DEPLOYED_TO, …
```

When the extractor cannot confidently place an action into one of these, fall through to a `lifecycle_status`-tagged generic edge rather than the current `MODIFIED` catch-all (see Move 2).

Allow-list lives in `domain/ontology.py` alongside the existing canonical edge catalog — Phase 1 already introduced that structure.

### Move 2 — `lifecycle_status` property on every edge

Every action edge carries one of: `proposed | planned | in_progress | completed | deprecated | decommissioned | unknown`.

- Derivable from tense + modality ("will be added" → `planned`; "was decommissioned" → `decommissioned`; "is being migrated" → `in_progress`).
- Stored on the edge, not the node, because the same subject can be simultaneously `completed` for one action and `planned` for another.
- Exposed in the CLI search output as a short tag: `[planned]`, `[done]`, `[deprecated]`. Current facts (`completed`) get no tag to keep output clean.

## Files touched

- `app/src/context-engine/domain/ontology.py` — expand canonical edge catalog; add `LifecycleStatus` enum.
- `app/src/context-engine/adapters/outbound/graphiti/episodic.py` — pass edge-type schema to Graphiti's extractor; inject post-extraction validator that downgrades unknown labels.
- `app/src/context-engine/application/use_cases/reconciliation_validation.py` — reject extraction batches where > X% of edges are generic `MODIFIED` (signal of extractor regression).
- `app/src/context-engine/adapters/inbound/cli/output.py` — render `lifecycle_status` tag.
- `app/src/context-engine/domain/graph_mutations.py` — edge mutation schema gains `lifecycle_status`.
- Tests: `app/src/context-engine/tests/unit/test_ontology_lifecycle.py`, golden-file extractor tests.

## Edge migration

Existing pots contain many `MODIFIED` edges. A one-shot `maintenance_job: classify_modified_edges` (already a canonical maintenance-job type per Phase 7) re-runs extraction in classify-only mode over historical episodes and writes back the specific edge type. Idempotent; runs behind feature flag.

## Risks

- Extractor precision drops short-term because the schema is now constrained. Mitigation: log-compare old vs. new on a fixed fixture set before each ontology rev.
- "One-of" enforcement will sometimes be wrong; the fall-through to `lifecycle_status`-tagged generic edge is the safety valve.

## Rollout

1. Land ontology extension + `lifecycle_status` column (no behaviour change; defaults to `unknown`).
2. Turn on edge-type allow-list in extractor for new ingests only.
3. Run `classify_modified_edges` maintenance job on historical data per pot, behind `--dry-run` first.
4. Update CLI render once data is populated.

## Done when

- Ingesting the six-episode test set produces zero `MODIFIED` edges for action verbs — all mapped to specific types or `lifecycle_status`-tagged.
- `context_resolve intent=feature` filters by `lifecycle_status in ("planned", "in_progress")` without needing substring checks on the summary.
