# 05 — Provenance in CLI output

## Problem

`potpie search` defaults to printing `uuid / name / summary / fact`. It does **not** show:

- `source` — the `--source` label attached at ingest (e.g. `adr-0042`, `pr-1287`, `incident-review`).
- `reference_time` — when the fact was observed (distinct from Graphiti's edge `valid_at`).
- `episode_uuid` — the ingested episode that produced the edge.

Provenance is exactly the Phase 2 investment, and the MCP `context_search` envelope includes `source_refs`, but the CLI hides all of it. An operator cannot tell by eye whether a row came from a design doc, a PR, or a Slack thread.

## Proposal

### CLI default render

Add a single line per result:

```
source: adr-0042 • ref: 2025-04-10 • episode: df605b8d
```

Shortened UUIDs (first 8 chars) for episode, ISO date only for `reference_time`, raw `source` string as-is.

If multiple source refs exist on the edge, show `source: adr-0042, pr-1287 • …`.

### Backing data

The `/api/v2/context/query/search` response already carries enough to compute this — `source_refs` are attached during context resolution. Verify and if missing, plumb from `ingestion_event_store` through `hybrid_graph` into the search result row.

Namely, the outbound adapter at `adapters/outbound/http/potpie_context_api_client.py` returns rows as-is from server; the server path in `adapters/inbound/http/api/v1/context/router.py` must ensure `source_refs`, `reference_time`, and `episode_uuid` are included when `source_policy in ("references_only", "summary", "verify", …)`. Default for `search` endpoint today is unclear — confirm.

### `--source` / `--episode` filters (free rider)

Once the fields flow through, `potpie search --source adr-0042 "ledger"` can filter server-side. One new query parameter, one CLI flag.

## Files touched

- `app/src/context-engine/adapters/inbound/http/api/v1/context/router.py` — include `source_refs`, `reference_time`, `episode_uuid` on search response rows; accept `source` and `episode_uuid` filters.
- `app/src/context-engine/application/use_cases/query_context.py` — pass filters to the episodic adapter.
- `app/src/context-engine/adapters/outbound/graphiti/episodic.py` — propagate source metadata from edges.
- `app/src/context-engine/adapters/inbound/cli/output.py::print_search_results` — new default render.
- `app/src/context-engine/adapters/inbound/cli/main.py` — `--source-filter` / `--episode` CLI options on `search`.
- `app/src/context-engine/adapters/inbound/cli/README.md` — document new flags.
- Tests: CLI golden-output tests; server response contract test.

## Risks

- Output width blow-up on narrow terminals. Mitigation: render the provenance line at `dim` style and allow `--no-provenance` to hide.
- Backwards compatibility of the JSON response: adding fields is safe; consumers not already consuming are unaffected.

## Rollout

One PR end-to-end. Small, local, high-visibility win.

## Done when

- Default `potpie search` prints source/ref/episode for every row without any flag.
- `potpie search --source adr-0042 "ledger"` returns only rows whose edge cites `adr-0042`.
- `potpie --json search …` includes `source_refs`, `reference_time`, `episode_uuid` in every row.
