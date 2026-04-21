# Fix 04 — Canonical node labels on the write and search path

Delivers [plan/improvement #03](../context-graph-improvements/03-canonical-node-labels.md) end-to-end: `--node-labels <OntologyLabel>` must match **Neo4j native labels** on episodic `Entity` nodes, not only the Graphiti base label `Entity`.

## Symptom (observed in harness runs)

Across test pots, counting search hits when filtering by candidate ontology labels often showed only the base label populated:

| Label | Typical result |
|-------|----------------|
| `Entity` (Graphiti base) | non-zero |
| `Incident` | sometimes (depends on edges / rules) |
| `Decision`, `Service`, `Release`, `Component`, `DataStore`, `Person`, `Fix`, … | often **0** |

When ingestion produces a `DECIDES_FOR` episodic edge but endpoints stay labeled only `:Entity`, Graphiti search filtered by `Decision` returns nothing. The CLI flag is only useful if nodes carry the same strings as [public entity types](../../app/src/context-engine/domain/ontology.py) (`ENTITY_TYPES` keys), e.g. `DataStore` not `Datastore`.

## Actual architecture (two complementary paths)

Canonical labels reach storage in **two** places; both rely on the same rule table and the same env flag for inference.

### 1 — Rule table (single source of truth)

**Module:** [`app/src/context-engine/domain/ontology.py`](../../app/src/context-engine/domain/ontology.py)

- **`EDGE_ENDPOINT_INFERRED_LABELS`**: `dict[tuple[str, str], tuple[str, ...]]`  
  Key: `(normalize_graphiti_edge_name(name), "source" | "target")` → tuple of ontology labels to add on that endpoint.
- **`inferred_labels_for_episodic_edge_endpoint(edge_name, role)`**: safe lookup helper.

Only **unambiguous** rows are included. Deliberately omitted patterns (e.g. `OWNS` / `FIXES` on an endpoint without extra context) return empty tuples — see [`test_label_inference.py`](../../app/src/context-engine/tests/unit/test_label_inference.py) (`test_ambiguous_roles_return_empty`).

Current rows (representative; see ontology for the full set):

```text
("DECIDES_FOR", "target")   → ("Decision",)
("CAUSED", "target")        → ("Incident",)
("DEPLOYED_TO", "target")   → ("Deployment",)
("DEPRECATED", "target")    → ("LegacyArtifact",)
("DECOMMISSIONED", "target")→ ("LegacyArtifact",)
```

Expanding the table for more symptom-row labels (e.g. `FIXES` → `Fix`, `STORED_IN` → `DataStore`) is follow-on work: each new row must be **safe** under multi-label graphs (no systematic wrong-type tagging).

### 2a — Reconciliation plans → structural Neo4j upserts

**Merge step:** [`app/src/context-engine/domain/canonical_label_inference.py`](../../app/src/context-engine/domain/canonical_label_inference.py) — `enrich_reconciliation_plan_entity_labels(plan)` mutates **`EntityUpsert.labels`** (and fills required properties when a new canonical label appears).

**When it runs:** [`validate_reconciliation_plan`](../../app/src/context-engine/application/use_cases/reconciliation_validation.py) calls `enrich_reconciliation_plan_entity_labels` **after** hard structural limits checks and **only if** `infer_canonical_labels_enabled()` is true (see flag below).

**Neo4j apply:** [`StructuralGraphPort` implementation](../../app/src/context-engine/adapters/outbound/neo4j/structural.py) `upsert_entities`: `MERGE (e:Entity {…})` then, for each canonical label on the upsert, `MATCH … SET e:Decision` (etc.). There is **no** separate `NodeLabelsAdd` mutation type — labels are part of `EntityUpsert`.

### 2b — Graphiti episodic graph → post-ingest label pass

Episode ingestion creates `Entity` + `RELATES_TO` in Graphiti’s store. After each successful `add_episode_async`, [`apply_episodic_canonical_labels`](../../app/src/context-engine/adapters/outbound/graphiti/apply_canonical_labels.py) runs Cypher that, for each `EDGE_ENDPOINT_INFERRED_LABELS` entry, matches living `RELATES_TO` edges (`invalid_at IS NULL`) and **`SET`** the corresponding endpoint node with the ontology label (e.g. `SET b:Decision`). A second pass adds labels from **`canonical_type`** when present and valid.

**Maintenance / backfill:** `relabel_nodes_from_edges(driver, group_id)` in the same module calls `apply_episodic_canonical_labels(…, force=True)` so maintenance can refresh labels even when inference is disabled for normal traffic.

**Order in ingest:** episodic adapter runs auto-supersede, then canonical labels, then family conflict detection — see [`episodic.py`](../../app/src/context-engine/adapters/outbound/graphiti/episodic.py) `add_episode_async`.

### 3 — Search / `--node-labels`

**Module:** [`app/src/context-engine/adapters/outbound/graphiti/episodic.py`](../../app/src/context-engine/adapters/outbound/graphiti/episodic.py) — `_build_search_filters` passes `node_labels` into Graphiti’s search filter object when provided. Filtering is **not** implemented in our adapter as raw Cypher `WHERE n.entity_type = …`; it depends on Graphiti matching nodes that actually carry those labels in Neo4j. If labels were only stored as properties, this filter would not align with native-label search — the write path above must keep **labels** consistent with ontology names.

Hybrid search entrypoints forward `node_labels` into the same episodic search stack — see [`hybrid_graph.py`](../../app/src/context-engine/adapters/outbound/intelligence/hybrid_graph.py) `search_context`.

## Feature flag

| Variable | Default | Meaning |
|----------|---------|---------|
| `CONTEXT_ENGINE_INFER_LABELS` | on (`true`) | Enables reconciliation enrichment **and** gates the episodic post-pass (`apply_episodic_canonical_labels` skips when off unless `force=True`). Implemented in [`reconciliation_flags.py`](../../app/src/context-engine/domain/reconciliation_flags.py) (`infer_canonical_labels_enabled`). |

This is the knob referenced in rollout; it is **not** named `CONTEXT_ENGINE_INFER_LABELS=1` as a separate symbol — any truthy env value works per `_truthy` in that module.

## Diagnostics (in order)

### D1 — Rule table and helper behave

Unit tests: [`app/src/context-engine/tests/unit/test_label_inference.py`](../../app/src/context-engine/tests/unit/test_label_inference.py)

- `test_edge_endpoint_rules_cover_decides_for_target` — `DECIDES_FOR` / `target` → `Decision`
- `test_ambiguous_roles_return_empty` — documents intentional omission for noisy edges
- `test_enrich_plan_adds_decision_label_and_defaults` — `enrich_reconciliation_plan_entity_labels` adds `Decision` on the target of `DECIDES_FOR` and backfills required Decision fields

If D1 fails: fix `EDGE_ENDPOINT_INFERRED_LABELS` / `inferred_labels_for_episodic_edge_endpoint`, not downstream callers.

### D2 — Reconciliation path merges labels before validation

Same file: `test_validate_reconciliation_runs_enrich_when_flag` — full `validate_reconciliation_plan` runs enrichment when the flag is on.

If D2 fails: check `validate_reconciliation_plan` order and `infer_canonical_labels_enabled()`.

### D3 — Neo4j nodes carry native labels (Graphiti pot)

After ingest, against the tenant’s Neo4j (group id = pot id):

```cypher
MATCH (n:Entity {group_id: $gid})
WHERE $uuid IN [n.uuid, n.entity_key] // adapt to how the node is keyed in your graph
RETURN labels(n) AS labels, n.entity_key AS key LIMIT 25;
```

For episodic `RELATES_TO`:

```cypher
MATCH (a:Entity {group_id: $gid})-[e:RELATES_TO]->(b:Entity {group_id: $gid})
WHERE toUpper(trim(e.name)) = 'DECIDES_FOR' AND e.invalid_at IS NULL
RETURN labels(a) AS src, labels(b) AS tgt LIMIT 10;
```

Expect `Decision` (and other inferred labels) on endpoints **after** `apply_episodic_canonical_labels` has run.

### D4 — Structural upsert applies labels

For deterministic reconciliation writes, `upsert_entities` applies each canonical label with a separate `SET e:\{lbl\}` — see [`structural.py`](../../app/src/context-engine/adapters/outbound/neo4j/structural.py) (~1313–1321). If labels are present on `EntityUpsert` but missing in the graph, debug the reconciliation apply pipeline (not Graphiti).

## Remaining gaps / backlog

- **Broader rule table:** add only rows that pass ambiguity review (see improvement doc risk section on over-labelling).
- **Naming in CLI/docs:** use exact ontology spellings (`DataStore`, `PullRequest`, …) in examples so search filters match.
- **Scorecard:** closing this fix requires re-running the harness “six-query” checks with non-zero rows for the labels you care about; update [`README.md`](README.md) when verified.

## Files (authoritative)

| Area | Path |
|------|------|
| Rule table + helper | `domain/ontology.py` (`EDGE_ENDPOINT_INFERRED_LABELS`, `inferred_labels_for_episodic_edge_endpoint`) |
| Plan merge | `domain/canonical_label_inference.py` |
| Validation hook | `application/use_cases/reconciliation_validation.py` |
| Flag | `domain/reconciliation_flags.py` |
| Structural write | `adapters/outbound/neo4j/structural.py` (`upsert_entities`) |
| Episodic post-pass + backfill | `adapters/outbound/graphiti/apply_canonical_labels.py` |
| Ingest ordering | `adapters/outbound/graphiti/episodic.py` |
| Search filter wiring | `adapters/outbound/graphiti/episodic.py` (`_build_search_filters`) |
| Tests | `tests/unit/test_label_inference.py` |

## Risks (unchanged in spirit)

- **Over-labelling:** broad rules (e.g. `OWNS` → single type) mis-tag nodes. Mitigation: keep the table small, add tests per row, prefer `canonical_type` hints where extractors are reliable.
- **Label injection:** episodic pass uses a fixed allowlist via `_safe_label` / `is_canonical_entity_label` + `ENTITY_TYPES`; structural upserts guard before `SET e:Label`.

## Rollout checklist

1. Confirm `EDGE_ENDPOINT_INFERRED_LABELS` covers the edge/role pairs you need for the fixture episodes.
2. Ship reconciliation + Graphiti paths together so CLI ingests and server-side reconciliation both see the same semantics.
3. With `CONTEXT_ENGINE_INFER_LABELS` on (default), ingest a fresh pot and verify `potpie search "…" --node-labels Decision` returns the expected `Entity` rows.
4. Run `relabel_nodes_from_edges` (or API equivalent) for existing pots if historical data predates the pass.
5. Update project documentation only when behavior is verified (avoid stale `EDGE_TO_CANONICAL_LABEL` / `NodeLabelsAdd` references).

## Done when

- `potpie search "<query>" --node-labels Decision -n 5` in a fresh pot returns nodes that participate in a `DECIDES_FOR` edge once that episode is ingested.
- Search by other enabled inference labels (e.g. `Incident` where `CAUSED` exists) returns non-zero rows where the rule table applies.
- Unit tests in `test_label_inference.py` remain green; optional integration tests cover ingest + search if the team adds them.
