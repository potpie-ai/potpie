#!/bin/bash
# Run the Celery worker that processes agent and parsing tasks.
# Agent tasks (conversations) are routed to ${CELERY_QUEUE_NAME}_agent_tasks;
# without -Q you would only consume the default "celery" queue and see no task logs.
set -e
source "${BASH_SOURCE%/*}/../.env" 2>/dev/null || true
export CELERY_QUEUE_NAME="${CELERY_QUEUE_NAME:-staging}"
celery -A app.celery.celery_app worker --loglevel=info \
  -Q "${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks" \
  -E --concurrency=1 --pool=solo
