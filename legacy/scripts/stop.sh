#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Stopping Potpie services..."

# Kill the FastAPI (gunicorn) and Celery processes
echo "Stopping FastAPI and Celery processes..."
pkill -f "gunicorn" || true
pkill -f "celery" || true

# Stop Docker Compose services
echo "Stopping Docker Compose services..."
docker compose -f "$REPO_ROOT/compose.yaml" down

echo "All Potpie services have been stopped successfully!"
