"""
Celery configuration for Potpie Workflows.
This module provides a centralized Celery app instance for both task queuing and worker execution.
"""

import os
from celery import Celery


# Create the Celery app
def create_celery_app(worker_name: str = "workflow_worker") -> Celery:
    """Create a shared Celery app instance for both task queue and worker."""
    broker_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    result_backend = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    app = Celery(
        worker_name,
        broker=broker_url,
        backend=result_backend,
        include=[
            "src.adapters.celery_worker",
            "src.adapters.event_bus_subscriber_single_queue",
        ],  # Include the module with task definitions
    )

    # Configure Celery
    app.conf.update(
        # Serialization settings
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_compression="gzip",
        # Timezone settings
        timezone="UTC",
        enable_utc=True,
        # Task tracking and result settings
        task_track_started=True,
        task_ignore_result=False,
        task_store_errors_even_if_ignored=True,
        result_extended=True,  # Store args and kwargs in the result
        # Task persistence and recovery settings
        task_acks_late=True,  # Acknowledge tasks after completion, not before
        task_reject_on_worker_lost=True,  # Reject tasks when worker is lost
        task_always_eager=False,  # Don't execute tasks synchronously
        task_eager_propagates=True,  # Propagate exceptions in eager mode
        task_persistent=True,  # Ensure tasks are persistent
        # Visibility timeout - how long a task is invisible after being claimed
        task_visibility_timeout=3600,  # 1 hour visibility timeout
        # Broker settings for persistence
        broker_transport_options={
            "visibility_timeout": 3600,  # 1 hour visibility timeout
            "fanout_prefix": True,
            "fanout_patterns": True,
        },
        # Result backend settings
        result_backend_transport_options={
            "visibility_timeout": 3600,
        },
        result_expires=3600,  # Results expire after 1 hour
        result_persistent=True,  # Store results persistently
        # Worker settings for task recovery
        worker_prefetch_multiplier=1,
        worker_disable_rate_limits=False,
        worker_enable_remote_control=True,
        worker_state_db=None,  # Disable worker state DB to avoid conflicts
        worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
        worker_max_memory_per_child=200000,  # Restart worker after 200MB memory usage
        # Use 'solo' pool instead of 'prefork' to avoid SIGSEGV with async code and database connections
        # 'solo' is single-threaded and better for async/await code
        worker_pool="solo",
        worker_concurrency=1,  # Solo pool only supports concurrency=1
        # Queue and routing settings
        task_default_queue="workflow_execution",
        task_default_exchange="workflow_execution",
        task_default_routing_key="workflow_execution",
        task_routes={
            "execute_node_task": {"queue": "workflow_execution"},
            # Event bus task routes - single queue
            "app.modules.event_bus.tasks.event_tasks.process_external_event": {
                "queue": "external-event"
            },
            # Legacy task route for external services
            "app.modules.event_bus.tasks.event_tasks.process_webhook_event": {
                "queue": "external-event"
            },
        },
        # Task aliases for backward compatibility with external services
        # Celery Insights recommended configuration
        worker_send_task_events=True,  # Enables task events
        task_send_sent_event=True,  # Enables sent events
        # Task execution settings for better fault tolerance
        task_time_limit=900,  # 15 minute hard time limit
        task_soft_time_limit=840,  # 14 minute soft time limit
        task_annotations={
            "execute_node_task": {
                "rate_limit": "10/m",  # Max 10 tasks per minute
                "time_limit": 900,
                "soft_time_limit": 840,
            },
            # Event bus task annotations - single queue
            "app.modules.event_bus.tasks.event_tasks.process_external_event": {
                "rate_limit": "100/m",  # Max 100 external events per minute
                "time_limit": 300,  # 5 minute time limit
                "soft_time_limit": 270,  # 4.5 minute soft limit
            },
        },
        # Worker pool settings for better async support
        worker_pool_restarts=True,
        worker_direct=True,  # Enable direct task execution
    )

    return app


# Create the shared app instance
app = create_celery_app()

# Make the app available for Celery CLI
if __name__ == "__main__":
    app.start()
