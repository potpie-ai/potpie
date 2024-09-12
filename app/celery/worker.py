from app.celery.celery_app import celery_app, logger

# Register tasks
def register_tasks():
    logger.info("Registering tasks")
    from app.celery.tasks import parsing_tasks
    celery_app.tasks.register(parsing_tasks.process_parsing)
    logger.info("Tasks registered successfully")

# Call register_tasks() after all modules have been imported
celery_app.on_after_configure.connect(lambda sender, **kwargs: register_tasks())

logger.info("Celery worker initialization completed")

if __name__ == "__main__":
    logger.info("Starting Celery worker")
    celery_app.start()