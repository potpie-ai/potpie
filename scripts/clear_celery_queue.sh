#!/bin/bash
# Clear Celery task queues so workers don't pick up old/stale tasks.
# Run from repo root. Prefers Python script (uses project Redis); falls back to celery purge.
set -e

cd "$(dirname "$0")/.."
if uv run python scripts/clear_celery_queue.py 2>/dev/null; then
  exit 0
fi
echo "Python clear failed; trying celery purge..."
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
QUEUE_NAME="${CELERY_QUEUE_NAME:-staging}"
Q1="${QUEUE_NAME}_process_repository"
Q2="${QUEUE_NAME}_agent_tasks"
celery -A app.celery.celery_app purge -f -Q "$Q1,$Q2" 2>/dev/null || true
echo "Done."
