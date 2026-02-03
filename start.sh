#!/bin/bash
set -e

source .env

# Set up Service Account Credentials
export GOOGLE_APPLICATION_CREDENTIALS="./service-account.json"

uv sync

source .venv/bin/activate


alembic upgrade heads

echo "Starting momentum application..."
gunicorn --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 1800 --bind 0.0.0.0:8001 --log-level debug app.main:app &

echo "Starting Celery worker..."
# Start Celery worker with the uv-managed environment
celery -A app.celery.celery_app worker --loglevel=debug -Q "${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks" -E --concurrency=1 --pool=solo &
