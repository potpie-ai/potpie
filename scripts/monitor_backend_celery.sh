#!/bin/bash
# Monitor backend (Gunicorn) and Celery while stress test runs.
# Usage: ./scripts/monitor_backend_celery.sh [interval_seconds]
# Run from repo root. Requires: uv run (or activate venv) for celery inspect.
set -e
cd "$(dirname "$0")/.."
INTERVAL="${1:-5}"
echo "Monitoring backend + Celery every ${INTERVAL}s (Ctrl+C to stop)"
while true; do
  echo ""
  echo "=== $(date '+%H:%M:%S') ==="
  echo "--- Gunicorn ---"
  pgrep -fl "gunicorn.*app.main:app" 2>/dev/null | wc -l | xargs echo "  processes:"
  echo "--- Celery ---"
  OUT=$(source .venv/bin/activate 2>/dev/null && celery -A app.celery.celery_app inspect active 2>/dev/null)
  ACTIVE=$(echo "$OUT" | grep "execute_agent_background" | wc -l | tr -d ' ')
  OUT2=$(source .venv/bin/activate 2>/dev/null && celery -A app.celery.celery_app inspect reserved 2>/dev/null)
  RESERVED=$(echo "$OUT2" | grep "execute_agent_background" | wc -l | tr -d ' ')
  echo "  active tasks: $ACTIVE"
  echo "  reserved (prefetched): $RESERVED"
  pgrep -fl "celery.*worker" 2>/dev/null | wc -l | xargs echo "  worker processes:"
  sleep "$INTERVAL"
done
