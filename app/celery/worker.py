from app.celery.celery_app import celery_app, logger


# Register tasks
def register_tasks():
    logger.info("Registering tasks")
    # Ensure the task is decorated with @celery_app.task in the parsing_tasks module
    # No need to manually register the task here
    logger.info("Tasks registered successfully")


# Call register_tasks() after all modules have been imported
celery_app.on_after_configure.connect(lambda sender, **kwargs: register_tasks())

logger.info("Celery worker initialization completed")

if __name__ == "__main__":
    logger.info("Starting Celery worker")
    celery_app.start()
