# Import the module containing the task
from app.celery.celery_app import celery_app, logger
from app.celery.tasks.parsing_tasks import (
    process_parsing,  # Ensure the task is imported
)


# Register tasks
def register_tasks():
    logger.info("Registering tasks")

    # Register parsing tasks
    celery_app.tasks.register(process_parsing)
    # If there are more tasks in other modules, register them here
    # For example:
    # from app.celery.tasks import other_tasks
    # celery_app.tasks.register(other_tasks.some_other_task)
    logger.info("Tasks registered successfully")


# Call register_tasks() immediately
register_tasks()

logger.info("Celery worker initialization completed")

if __name__ == "__main__":
    logger.info("Starting Celery worker")
    celery_app.start()
