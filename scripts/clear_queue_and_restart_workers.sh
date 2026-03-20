#!/bin/bash
# 1) Clear Celery queues in Redis (so workers don't pick old tasks).
# 2) Kill existing Gunicorn and Celery workers.
# Run from repo root. Then start the app with ./scripts/start.sh (or ./start.sh).
# Requires: Docker (and thus Redis) running first — e.g. docker compose up -d.
set -e

cd "$(dirname "$0")/.."
echo "Clearing Celery queues..."
if uv run python scripts/clear_celery_queue.py 2>/dev/null; then
  echo "Queues cleared."
else
  echo "Could not clear queues (is Redis running? Start with: docker compose up -d)"
fi
echo "Stopping Gunicorn and Celery workers..."
pkill -f "gunicorn.*app.main:app" || true
pkill -f "celery.*worker.*celery_app" || true
sleep 2
echo "Workers stopped. Start the server with: ./scripts/start.sh  (or ./start.sh)"
echo "Then run stress tests with: CONVERSATION_ID=<id> AUTH_HEADER='Bearer <token>' uv run python scripts/stress_harness.py --profile conversation"
