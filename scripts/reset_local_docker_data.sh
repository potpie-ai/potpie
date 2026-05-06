#!/usr/bin/env bash
# Wipe local Docker volumes for Potpie: Postgres, Neo4j, Redis.
# Use when you want a clean slate to re-parse repos from scratch.
#
# Usage (from repo root):
#   ./scripts/reset_local_docker_data.sh           # prompts for confirmation
#   ./scripts/reset_local_docker_data.sh --yes     # no prompt
#
# After reset:
#   ./start.sh
#   or: docker compose up -d && uv run alembic upgrade heads
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

YES=false
for arg in "$@"; do
  if [[ "$arg" == "--yes" || "$arg" == "-y" ]]; then
    YES=true
  fi
done

if [[ "$YES" != true ]]; then
  echo "This will REMOVE all local data in Docker volumes:"
  echo "  - postgres_data (Postgres DB: projects, users, context graph log, …)"
  echo "  - neo4j_data    (code graph + Graphiti context graph)"
  echo "  - redis_data    (Celery broker / cache)"
  echo "  - neo4j_logs"
  read -r -p "Type YES to continue: " reply
  if [[ "$reply" != "YES" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

echo "Stopping local app processes (gunicorn / celery) if any..."
pkill -f "gunicorn.*app.main:app" 2>/dev/null || true
pkill -f "celery -A app.celery.celery_app" 2>/dev/null || true

echo "Removing containers and named volumes..."
docker compose down -v

echo "Done. Local Neo4j and Postgres (and Redis) data are gone."
echo "Next: ./start.sh   (or docker compose up -d then uv run alembic upgrade heads)"
