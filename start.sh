#!/bin/bash
source .env

docker-compose up -d

# Wait for postgres to be ready
echo "Waiting for postgres to be ready..."
until docker inspect -f {{.State.Health.Status}} postgres | grep -q "healthy"; do
  sleep 5
done

# Run momentum application with migrations
echo "Starting momentum application..."
alembic upgrade head && gunicorn --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 1800 --bind 0.0.0.0:8001 --log-level debug app.main:app