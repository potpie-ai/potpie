# Fix 03 — Graceful ontology downgrade

Closes the missing Move 1 from plan #02. Stops silently dropping episodes.

## Symptom

The reconciliation validator currently rejects an entire batch when **any** extracted entity has an unknown canonical label, unknown edge type, missing required property, or invalid lifecycle/status. The ADR-0042 episode in the 2026-04-21 test run was lost end-to-end because the extractor produced:

- labels: `Database`, `Technology`, `ADR` (none canonical)
- edge: `DECIDED_BY` (not in the canonical edge catalog)
- lifecycle status: `recorded` (not in `{accepted, proposed, rejected, superseded, unknown}`)

Net effect: zero `Decision` nodes, zero `DECIDES_FOR` edges, no provenance for Alice / Ravi / ADR-0042 in the graph. Agents asking "who decided to migrate the ledger?" get nothing.

Hard reject is strictly worse than a lossy extraction — a downgraded `RELATED_TO` edge with the original type preserved as a property is still useful context; no edge at all is useless.

## Fix — split validation into hard + soft

### Hard checks (still fatal → HTTP 422 via fix 01)

- Duplicate UUIDs within the batch.
- Missing `pot_id`.
- Malformed property types (e.g. `valid_at` not a valid ISO 8601).
- Batch size / payload size exceeded.

### Soft checks (downgrade + log)

| Issue | Downgrade action |
|-------|------------------|
| Unknown canonical label on node | Drop the unknown label; if the node has at least one canonical label remaining, keep it. If none, apply base `Entity`. |
| Unknown canonical edge type | Rewrite to `RELATED_TO` (or the agreed catch-all — confirm with ontology owner). Preserve `original_edge_type: "DECIDED_BY"` as an edge property. Set `confidence: 0.3`. |
| Missing required property | Fill with `null` if the property is nullable in the schema; otherwise drop the entity and emit a `QualityIssue`. |
| Invalid lifecycle/status | Coerce to `unknown`. |

Every downgrade writes a `QualityIssue` of kind `ontology_downgrade` with:

```json
{
  "kind": "ontology_downgrade",
  "entity_uuid": "…",
  "original": {"labels": ["ADR"], "edge_type": "DECIDED_BY"},
  "applied": {"labels": ["Entity"], "edge_type": "RELATED_TO"},
  "severity": "info",
  "episode_uuid": "…"
}
```

This gives the ontology team a clean feed of what needs to be added to the canonical catalog.

### Response surface

Ingest response extends fix 01's shape:

```json
{
  "status": "applied",
  "episode_uuid": "…",
  "errors": [],
  "downgrades": [
    {"entity_uuid": "…", "kind": "edge_type", "from": "DECIDED_BY", "to": "RELATED_TO"},
    {"entity_uuid": "…", "kind": "lifecycle_status", "from": "recorded", "to": "unknown"}
  ]
}
```

CLI prints a compact `N downgrades applied (potpie quality downgrades)` line under the normal success message.

## Files touched

- `app/src/context-engine/application/use_cases/reconciliation_validation.py` — split `validate_or_raise` into `validate_hard` + `apply_soft_downgrades`, wire both.
- `app/src/context-engine/domain/ontology.py` — expose `CANONICAL_LABELS`, `CANONICAL_EDGE_TYPES`, `ALLOWED_LIFECYCLE_STATUSES` as importable sets for the downgrade step.
- `app/src/context-engine/domain/graph_quality.py` — `QualityIssue.kind = "ontology_downgrade"` support.
- `app/src/context-engine/domain/graph_mutations.py` — if `RELATED_TO` fallback is new, register it as a canonical edge type.
- `app/src/context-engine/adapters/inbound/http/api/v1/context/router.py` — include `downgrades` in ingest response.
- `app/src/context-engine/adapters/inbound/cli/output.py::print_ingest_result` — render the compact downgrade line.
- Tests: `tests/unit/test_soft_downgrade.py` (golden-file set of known-bad extractor outputs → expected downgrades).

## Feature flag

Gate behind `CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL=1` for the first release. Default **off** until downgrade test fixtures pass in CI, then flip on. Keep `CONTEXT_ENGINE_ONTOLOGY_STRICT=1` as an always-strict override for CI extraction-quality tests.

## Risks

- **Downgrade hides real extractor bugs.** Mitigation: `QualityIssue(kind="ontology_downgrade")` feed lets the ontology team see the drift weekly. Require an alert threshold (e.g. > 10% of recent ingests include a downgrade).
- **`RELATED_TO` becomes a junk drawer.** Mitigation: preserve `original_edge_type` on the edge; weekly review of top-N `original_edge_type` values feeds back into the canonical catalog.
- **Silent data quality regression.** Mitigation: `context_status` includes a `recent_downgrades` counter alongside `open_conflicts`.

## Rollout

1. Ship downgrade logic behind the flag, default off.
2. Wire `QualityIssue` writes.
3. Dev pot flip on, monitor downgrade feed for a week.
4. Add top-N downgrade types to the canonical catalog (bumps ontology version).
5. Staging flip on; prod flip on.

## Done when

- The ADR-0042 fixture episode lands end-to-end: a `Decision` node (or the best canonical approximation) exists, `Alice` and `Ravi` are on canonical nodes, the causal edge between them and the migration is captured as `RELATED_TO` with `original_edge_type: "DECIDED_BY"`.
- `potpie ingest` on that episode exits `0`, with a downgrade summary line shown to the user.
- `potpie quality downgrades` (or the chosen surface) lists the drops so the ontology team can see them.
