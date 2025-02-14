import logging
import os

from celery import Celery
from dotenv import load_dotenv

from app.core.models import *  # noqa #This will import and initialize all models

# Load environment variables from a .env file if present
load_dotenv()
# Redis configuration
redishost = os.getenv("REDISHOST", "localhost")
redisport = int(os.getenv("REDISPORT", 6379))
redisuser = os.getenv("REDISUSER", "")
redispassword = os.getenv("REDISPASSWORD", "")
queue_name = os.getenv("CELERY_QUEUE_NAME", "staging")

# Construct the Redis URL
if redisuser and redispassword:
    redis_url = f"redis://{redisuser}:{redispassword}@{redishost}:{redisport}/0"
else:
    redis_url = f"redis://{redishost}:{redisport}/0"

# Initialize the Celery app
celery_app = Celery("KnowledgeGraph", broker=redis_url, backend=redis_url)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add logging for Redis connection
logger.info(f"Connecting to Redis at: {redis_url}")
try:
    celery_app.backend.client.ping()
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")


def configure_celery(queue_prefix: str):
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_routes={
            "app.celery.tasks.parsing_tasks.process_parsing": {
                "queue": f"{queue_prefix}_process_repository"
            },
        },
        # Optimize task distribution
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_track_started=True,
        task_time_limit=5400,  # 90 minutes in seconds
        # Add fair task distribution settings
        worker_max_tasks_per_child=200,  # Restart worker after 200 tasks to prevent memory leaks
        worker_max_memory_per_child=2000000,  # Restart worker if using more than 2GB
        task_default_rate_limit="10/m",  # Limit tasks to 10 per minute per worker
        task_reject_on_worker_lost=True,  # Requeue tasks if worker dies
        broker_transport_options={
            "visibility_timeout": 5400
        },  # 45 minutes visibility timeout
    )


configure_celery(queue_name)

# Import the lock decorator

# Import the lock decorator
from celery.contrib.abortable import AbortableTask  # noqa

# Import tasks to ensure they are registered
import app.celery.tasks.parsing_tasks  # noqa # Ensure the task module is imported
