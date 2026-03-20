# Context Graph (Phase 1)

Graphiti-based context store for PR and commit context. Used by the spec-gen agent via `get_project_context`.

## Env vars

| Variable | Description |
|----------|-------------|
| `CONTEXT_GRAPH_ENABLED` | Set to `true` to enable ETL, webhook ingestion, and the tool. Default: `false`. |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | **Same** as the code graph. Context graph uses the same Neo4j instance (no separate instance). |

## Running locally

1. Use your existing Neo4j (same `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` as the code graph).
2. Set in `.env`: `CONTEXT_GRAPH_ENABLED=true`.
3. Run migrations: `alembic upgrade head`.
4. (Optional) Backfill one project: from a Celery shell, `context_graph_backfill_project.delay("<project_id>")`. Or run `python scripts/context_graph_backfill_all.py` to enqueue all eligible projects (ensure a worker is consuming the `context-graph-etl` queue).

## Module layout

- `graphiti_client.py` — async wrapper (add_episode, search) with `group_id` / `group_ids`.
- `episode_formatters.py` — GitHub PR and commit → episode text and source_id.
- `ingestion_service.py` — dedup via `context_ingestion_log`, format, add to Graphiti, log.
- `models.py` — `ContextSyncState`, `ContextIngestionLog` (Postgres).
- `tasks.py` — Celery: `context_graph_backfill_project`, `ingest_pr_from_webhook`.
