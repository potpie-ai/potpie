# Context Graph (Phase 1)

Graphiti-based context store for PR and commit context. Used by the spec-gen agent via `get_project_context`.

## Env vars

| Variable | Description |
|----------|-------------|
| `CONTEXT_GRAPH_ENABLED` | Set to `true` to enable ETL, webhook ingestion, and the tool. Default: `false`. |
| `GRAPHITI_NEO4J_URI` | Neo4j URI for the **dedicated** Graphiti instance (not the code-graph Neo4j). |
| `GRAPHITI_NEO4J_USER` | Neo4j user for Graphiti. |
| `GRAPHITI_NEO4J_PASSWORD` | Neo4j password for Graphiti. |

## Running locally

1. Start a second Neo4j (e.g. Docker: `docker run -p 7688:7687 -e NEO4J_AUTH=neo4j/pass neo4j:latest`).
2. Set in `.env`: `CONTEXT_GRAPH_ENABLED=true`, `GRAPHITI_NEO4J_URI=bolt://localhost:7688`, `GRAPHITI_NEO4J_USER=neo4j`, `GRAPHITI_NEO4J_PASSWORD=pass`.
3. Run migrations: `alembic upgrade head`.
4. (Optional) Backfill one project: from a Celery shell, `context_graph_backfill_project.delay("<project_id>")`. Or run `python scripts/context_graph_backfill_all.py` to enqueue all eligible projects (ensure a worker is consuming the `context-graph-etl` queue).

## Module layout

- `graphiti_client.py` — async wrapper (add_episode, search) with `group_id` / `group_ids`.
- `episode_formatters.py` — GitHub PR and commit → episode text and source_id.
- `ingestion_service.py` — dedup via `context_ingestion_log`, format, add to Graphiti, log.
- `models.py` — `ContextSyncState`, `ContextIngestionLog` (Postgres).
- `tasks.py` — Celery: `context_graph_backfill_project`, `ingest_pr_from_webhook`.
