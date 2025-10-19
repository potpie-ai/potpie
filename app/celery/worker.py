# Import the module containing the task
from app.celery.celery_app import celery_app, logger
from app.celery.tasks.parsing_tasks import (
    process_parsing,  # Ensure the task is imported
)
from app.celery.tasks.agent_tasks import (
    execute_agent_background,  # Import agent task
    execute_regenerate_background,  # Import regenerate task
from app.modules.event_bus.tasks.event_tasks import (
    process_webhook_event,
    process_custom_event,
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
    logger.info("Tasks registered successfully")


# Call register_tasks() immediately
register_tasks()

logger.info("Celery worker initialization completed")

if __name__ == "__main__":
    logger.info("Starting Celery worker")
    celery_app.start()
