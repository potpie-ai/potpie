# Celery worker entrypoint.
# Run with the queues that agent/parsing tasks use, e.g.:
#   celery -A app.celery.celery_app worker -l info -Q staging_agent_tasks,staging_process_repository
# Or: ./scripts/run_celery_worker.sh
# (Without -Q the worker only consumes the default "celery" queue and agent tasks never run here.)
from app.celery.celery_app import celery_app, logger
from integrations.application.bootstrap import load_providers

load_providers()
from app.celery.tasks.parsing_tasks import (
    process_parsing,  # Ensure the task is imported
)
from app.celery.tasks.agent_tasks import (
    execute_agent_background,  # Import agent task
    execute_regenerate_background,  # Import regenerate task
)
from app.modules.event_bus.tasks.event_tasks import (
    process_webhook_event,
    process_custom_event,
)
from app.modules.context_graph.tasks import (
    context_graph_flush_windowed_batches,
    context_graph_process_batch,
    context_graph_reap_stale_batches,
)
from app.modules.intelligence.tools.sandbox.tasks import (
    sandbox_storage_sweep,
)


# Register tasks
def register_tasks():
    logger.info("Registering tasks")

    # Register parsing tasks
    celery_app.tasks.register(process_parsing)

    # Register agent tasks
    celery_app.tasks.register(execute_agent_background)
    celery_app.tasks.register(execute_regenerate_background)

    # Register event bus tasks
    celery_app.tasks.register(process_webhook_event)
    celery_app.tasks.register(process_custom_event)

    # Register context graph tasks. Backfill is no longer a task — a source
    # attach emits one agent_reconciliation event that flows through
    # context_graph_process_batch like any other event.
    celery_app.tasks.register(context_graph_process_batch)
    celery_app.tasks.register(context_graph_flush_windowed_batches)
    celery_app.tasks.register(context_graph_reap_stale_batches)

    # Register sandbox maintenance tasks (beat-scheduled storage sweep)
    celery_app.tasks.register(sandbox_storage_sweep)
    logger.info("Tasks registered successfully")


# Call register_tasks() immediately
register_tasks()

logger.info("Celery worker initialization completed")

if __name__ == "__main__":
    logger.info("Starting Celery worker")
    celery_app.start()
