# Fix: infra_topology reader ignores semantic similarity in ranking

**Date:** 2026-07-07
**Area:** `potpie/context-engine/application/readers/infra_topology.py`, `_common.py`, sibling readers

## Problem

`InfraTopologyReader` produced ranking candidates without `semantic_similarity`:

- It never passed `fact_query=req.query` to `ClaimQueryPort.find_claims`, so backends
  never stamped `row.properties["semantic_similarity"]` on infra claim rows.
- It never copied that property into `Candidate.semantic_similarity`.

The shared `RankingService` substitutes a neutral default of `0.5` for a missing
similarity — and similarity is the highest-weighted factor (1.3). Consequences:

1. All infra claims clustered at a flat semantic score regardless of the query.
2. Flat-scored infra claims (often `deterministic` strength = 1.0) could outrank
   genuinely query-relevant results from other families (timeline, prior_bugs, ...)
   when the `EnvelopeBuilder` merges and sorts all families by score.

Every other reader (timeline, docs, owners, features, decisions, prior_bugs,
coding_preferences) already wires the query through and reads the stamped score.

## Fix

1. **Unanchored path** (no scope anchors → single `find_claims`): pass
   `fact_query=req.query`, identical to the sibling-reader pattern. The backend
   orders by similarity and stamps the score.
2. **Anchored path** (BFS traversal): the traversal stays query-free on purpose.
   On vector backends (FalkorDB + embedder) a `fact_query` turns `find_claims`
   into an ANN top-k search; frontier edges outside the top-k would vanish and
   silently prune the walk. Instead, after the traversal completes, one follow-up
   `find_claims(claim_key_in=<collected keys>, fact_query=req.query)` stamps
   similarity onto the already-discovered rows. Topology discovery is unchanged;
   only scoring becomes query-sensitive.
3. **Candidate wiring**: `Candidate.semantic_similarity` is now populated from the
   stamped property.
4. **Refactor**: the `sim = row.properties.get("semantic_similarity")` +
   isinstance-check snippet was copy-pasted across seven readers; extracted into
   `claim_semantic_similarity(row)` in `application/readers/_common.py` and all
   readers now use it.

## Edge cases considered

- **No query** (`req.query is None`): no follow-up query, no stamping; the ranker
  keeps the neutral 0.5 default. Behavior identical to before the fix.
- **Rows without `claim_key`**: excluded from the follow-up lookup; they keep a
  neutral similarity rather than crashing or mis-mapping.
- **Hard filters on the follow-up lookup**: `predicate_in`, `source_ref_in`,
  `include_invalidated`, and `as_of` are forwarded so semantic stamping stays
  aligned with the traversal's structural filters.
- **Vector backend returns partial stamps** (ANN top-k misses some claims): the
  missing rows simply stay neutral (0.5) — graceful degradation, no ordering
  distortion of the traversal itself.
- **Traversal behavior**: hop queries are untouched, so which edges are
  discovered (depth, direction, environment filtering, dedupe) is byte-for-byte
  the same as before.
- **`bool` similarity values**: rejected in the shared helper even though Python
  treats `bool` as an `int`; this prevents fabricated 0.0/1.0 scores if a bad
  backend/test row ever stamps a boolean.

## Tests

Added to `tests/unit/test_p9_readers.py` (`TestInfraTopologyReader`):

- query present → infra candidates carry differentiated `semantic_similarity` in
  the score breakdown, and the query-relevant claim ranks first (anchored path).
- query present, unanchored path → same, via the direct `fact_query` route.
- no query → breakdown keeps the neutral 0.5 (regression guard).
- query present → traversal discovers the same edge set as without a query
  (topology discovery must stay query-insensitive).

Added to `tests/conformance/test_graph_surface_lite_e2e.py`:

- `test_infra_resolve_ranks_by_task_query_end_to_end` — mutate two infra edges
  through `DefaultGraphService`, then `context_resolve` with a task query; asserts
  the query-relevant dependency ranks first with a non-flat semantic score. This
  exercises the real agent read trunk (orchestrator → envelope), not just the
  reader in isolation.

Follow-up (same change set): `graph read --subgraph infra_topology --view
service_neighborhood` used to declare `query` as an unsupported filter, so the
CLI/workbench read path rejected the query before it ever reached the reader —
while `DefaultGraphService.read()` already forwards `request.query` into the
orchestrator. Added `query` to the view's `supported_filters` and
`optional_scope` in `graph_workbench_ontology.py` (matching the timeline,
debugging, decisions, features, and knowledge views, which all declare it).
The e2e test asserts both paths: `context_resolve` with a task query and
`graph read` with `--query` produce query-differentiated similarity scores.

## Notes

- `coding_rules.md` was requested as the style reference but does not exist in
  this workspace; existing P9 reader conventions were followed instead.
