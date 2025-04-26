#!/bin/bash
set -e

echo "Stopping Potpie services..."

# Kill the FastAPI (gunicorn) and Celery processes
echo "Stopping FastAPI and Celery processes..."
pkill -f "gunicorn" || true
pkill -f "celery" || true

# Stop Docker Compose services
echo "Stopping Docker Compose services..."
docker compose down

echo "All Potpie services have been stopped successfully!"
