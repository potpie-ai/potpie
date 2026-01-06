import os
import asyncio

# Set TOKENIZERS_PARALLELISM before any tokenizer imports to prevent fork warnings
# This must be set before sentence-transformers or any HuggingFace tokenizers are used
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from urllib.parse import urlparse, urlunparse

from celery import Celery
from celery.signals import worker_process_shutdown, worker_process_init
from dotenv import load_dotenv

from app.core.models import *  # noqa #This will import and initialize all models
from app.modules.utils.logger import configure_logging, setup_logger

# Load environment variables from a .env file if present
load_dotenv()

# Configure logging
configure_logging()
logger = setup_logger(__name__)

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


def sanitize_redis_url(url: str) -> str:
    """
    Sanitize Redis URL by masking credentials for safe logging.
    Returns URL with masked credentials (e.g., redis://***:***@host:port/0)
    """
    try:
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            # Mask username and password
            masked_netloc = f"***:***@{parsed.hostname}"
            if parsed.port:
                masked_netloc += f":{parsed.port}"
            sanitized = urlunparse(
                (
                    parsed.scheme,
                    masked_netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
            return sanitized
        return url
    except Exception:
        # If parsing fails, return a safe fallback
        return "redis://***:***@***:***/0"


# Initialize the Celery app
celery_app = Celery("KnowledgeGraph", broker=redis_url, backend=redis_url)

# Add logging for Redis connection
logger.info("Connecting to Redis", redis_url=sanitize_redis_url(redis_url))
try:
    celery_app.backend.client.ping()
    logger.info("Successfully connected to Redis")
except Exception:
    logger.exception(
        "Failed to connect to Redis", redis_url=sanitize_redis_url(redis_url)
    )


def configure_celery(queue_prefix: str):
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Disable Celery's default logging hijacking so our intercept handler works
        worker_hijack_root_logger=False,
        task_routes={
            "app.celery.tasks.parsing_tasks.process_parsing": {
                "queue": f"{queue_prefix}_process_repository"
            },
            "app.celery.tasks.agent_tasks.execute_agent_background": {
                "queue": f"{queue_prefix}_agent_tasks"
            },
            "app.celery.tasks.agent_tasks.execute_regenerate_background": {
                "queue": f"{queue_prefix}_agent_tasks"  # Same queue as other agent tasks
            },
            # Event bus task routes - both go to the same queue
            "app.modules.event_bus.tasks.event_tasks.process_webhook_event": {
                "queue": "external-event"
            },
            "app.modules.event_bus.tasks.event_tasks.process_custom_event": {
                "queue": "external-event"
            },
        },
        # Optimize task distribution
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_track_started=True,
        task_time_limit=5400,  # 90 minutes in seconds
        # Add fair task distribution settings
        worker_max_tasks_per_child=200,  # Restart worker after 200 tasks to prevent memory leaks
        # Memory limit: Restart worker if using more than configured limit (in KB)
        # Note: SIGKILL (signal 9) from OS can kill workers before this limit is reached
        # File size limits in CodeChangesManager help prevent memory spikes
        worker_max_memory_per_child=int(
            os.getenv("CELERY_WORKER_MAX_MEMORY_KB", "2000000")
        ),  # Default: 2GB (2000000 KB)
        # Removed task_default_rate_limit - was limiting to 10 tasks/min per worker, severely restricting concurrency
        # If rate limiting is needed, apply it per-task using @task(rate_limit='...') decorator
        task_reject_on_worker_lost=False,  # Don't requeue tasks if worker dies (prevents infinite retry loops)
        broker_transport_options={
            "visibility_timeout": 5400
        },  # 45 minutes visibility timeout
    )


configure_celery(queue_name)


def setup_phoenix_tracing():
    """Initialize Phoenix tracing for LLM monitoring in Celery workers."""
    try:
        from app.modules.intelligence.tracing.phoenix_tracer import (
            initialize_phoenix_tracing,
        )

        initialize_phoenix_tracing()
    except Exception as e:
        logger.warning(
            "Phoenix tracing initialization failed in Celery worker (non-fatal)",
            error=str(e),
        )


setup_phoenix_tracing()


def configure_litellm_for_celery():
    """
    Configure LiteLLM to use synchronous logging in Celery workers.
    This prevents async logging handlers from creating unawaited coroutines
    that cause SIGTRAP errors in forked worker processes.

    Set LITELLM_DEBUG=true to enable verbose debug logging.
    """
    try:
        import litellm

        logger.info("Configuring LiteLLM for Celery workers...")

        # Enable debug logging if LITELLM_DEBUG environment variable is set
        litellm_debug = os.getenv("LITELLM_DEBUG", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        if litellm_debug:
            litellm.set_verbose = True
            litellm._turn_on_debug()
            logger.info("LiteLLM debug logging ENABLED (LITELLM_DEBUG=true)")
        else:
            logger.debug(
                "LiteLLM debug logging disabled (set LITELLM_DEBUG=true to enable)"
            )

        # Disable verbose logging to reduce async handler usage
        # Use getattr/setattr to avoid type checking issues
        if hasattr(litellm, "verbose"):
            old_verbose = getattr(litellm, "verbose", None)
            setattr(litellm, "verbose", False)
            logger.debug(f"LiteLLM verbose logging: {old_verbose} -> False")

        # Disable async success handlers that create coroutines
        # This prevents "Task was destroyed but it is pending" errors
        for attr_name in ["success_callback", "async_success_callback"]:
            if hasattr(litellm, attr_name):
                old_value = getattr(litellm, attr_name, None)
                setattr(litellm, attr_name, [])
                logger.debug(
                    f"LiteLLM {attr_name}: {old_value} -> [] (disabled async handlers)"
                )

        # Try to disable litellm's internal async logging handlers more aggressively
        # Check for Logging class and set async_success_handler to a no-op function
        # Setting it to None causes TypeError when LiteLLM tries to call it
        try:
            from litellm.litellm_core_utils.litellm_logging import Logging

            # Create a no-op async function to replace async_success_handler
            # This prevents "TypeError: 'NoneType' object is not callable" errors
            async def noop_async_success_handler(*args, **kwargs):
                """No-op async handler that does nothing."""
                pass

            if hasattr(Logging, "async_success_handler"):
                # Set to no-op function instead of None to prevent call errors
                try:
                    setattr(
                        Logging, "async_success_handler", noop_async_success_handler
                    )
                    logger.debug(
                        "Set LiteLLM Logging.async_success_handler to no-op function"
                    )
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Could not set Logging.async_success_handler: {e}")

            # Also try to patch instance-level handlers if Logging instances exist
            # This handles cases where logging_obj is an instance, not the class
            try:
                # Check if there's a default instance or singleton
                if hasattr(Logging, "_instance") or hasattr(Logging, "default"):
                    # Try to get the default instance
                    default_instance = getattr(Logging, "_instance", None) or getattr(
                        Logging, "default", None
                    )
                    if default_instance and hasattr(
                        default_instance, "async_success_handler"
                    ):
                        setattr(
                            default_instance,
                            "async_success_handler",
                            noop_async_success_handler,
                        )
                        logger.debug(
                            "Set LiteLLM Logging instance async_success_handler to no-op function"
                        )
            except (AttributeError, TypeError):
                pass  # Instance patching is optional

            # Monkey-patch the Logging class __init__ to set async_success_handler on new instances
            # This ensures all future instances have the no-op handler
            try:
                original_init = Logging.__init__

                def patched_init(self, *args, **kwargs):
                    original_init(self, *args, **kwargs)
                    # Set async_success_handler to no-op after initialization
                    if hasattr(self, "async_success_handler"):
                        setattr(
                            self, "async_success_handler", noop_async_success_handler
                        )

                Logging.__init__ = patched_init
                logger.debug(
                    "Monkey-patched LiteLLM Logging.__init__ to set no-op async_success_handler"
                )
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not monkey-patch Logging.__init__: {e}")

            # Also patch litellm.utils._client_async_logging_helper to handle None handlers gracefully
            # This is a safety net in case logging_obj instances are created elsewhere
            try:
                import litellm.utils as litellm_utils

                if hasattr(litellm_utils, "_client_async_logging_helper"):
                    original_helper = litellm_utils._client_async_logging_helper

                    async def patched_helper(*args, **kwargs):
                        """
                        Patched helper that safely handles async_success_handler calls.
                        Accepts all arguments that the original function accepts, including
                        is_completion_with_fallbacks and any other keyword arguments.

                        Since we've set all async_success_handler instances to no-op functions,
                        we can safely skip calling them to avoid creating unawaited coroutines.
                        """
                        # Extract logging_obj from args or kwargs
                        logging_obj = None
                        if args:
                            logging_obj = args[0]
                        elif "logging_obj" in kwargs:
                            logging_obj = kwargs.get("logging_obj")

                        # Check if we have a logging_obj with async_success_handler
                        if logging_obj and hasattr(
                            logging_obj, "async_success_handler"
                        ):
                            handler = getattr(
                                logging_obj, "async_success_handler", None
                            )
                            # If handler is our no-op function, None, or not callable, do nothing
                            # This prevents creating unawaited coroutines in Celery workers
                            if (
                                handler is noop_async_success_handler
                                or handler is None
                                or not callable(handler)
                            ):
                                return None

                            # If we somehow have a real handler (shouldn't happen after our patching),
                            # we could call it, but to be safe in Celery workers, we'll skip it
                            # and log a warning
                            logger.warning(
                                "Found non-no-op async_success_handler in Celery worker, skipping to avoid unawaited coroutines"
                            )
                            return None

                        # If no logging_obj or handler, do nothing
                        return None

                    litellm_utils._client_async_logging_helper = patched_helper
                    logger.debug(
                        "Monkey-patched litellm.utils._client_async_logging_helper to accept all arguments"
                    )
            except (AttributeError, ImportError, TypeError) as e:
                logger.debug(
                    f"Could not monkey-patch _client_async_logging_helper: {e}"
                )

        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not access LiteLLM Logging class: {e}")

        # Ensure logging is synchronous by removing async handlers
        import logging

        litellm_logger = logging.getLogger("LiteLLM")
        initial_handler_count = len(litellm_logger.handlers)
        logger.debug(
            f"LiteLLM logger has {initial_handler_count} handlers before cleanup"
        )

        # Remove any async handlers that might cause issues
        handlers_to_remove = []
        for handler in litellm_logger.handlers:
            handler_name = type(handler).__name__
            # Check if handler has async emit method
            if hasattr(handler, "emit"):
                try:
                    if asyncio.iscoroutinefunction(handler.emit):
                        handlers_to_remove.append((handler, handler_name))
                        logger.debug(f"Found async handler to remove: {handler_name}")
                except (TypeError, AttributeError):
                    pass

        for handler, handler_name in handlers_to_remove:
            litellm_logger.removeHandler(handler)
            logger.debug(f"Removed async handler: {handler_name}")

        final_handler_count = len(litellm_logger.handlers)
        logger.info(
            f"LiteLLM configured for Celery workers: {initial_handler_count} -> {final_handler_count} handlers "
            f"(removed {len(handlers_to_remove)} async handlers)"
        )
    except ImportError:
        logger.debug("LiteLLM not available, skipping configuration")
    except Exception as e:
        logger.warning(f"Failed to configure LiteLLM for Celery: {e}", exc_info=True)


@worker_process_init.connect
def log_worker_memory_config(sender, **kwargs):
    """
    Log worker memory configuration on initialization.
    This helps debug memory-related issues like SIGKILL.
    """
    try:
        import psutil
        import os as os_module

        process = psutil.Process()
        memory_info = process.memory_info()
        max_memory_kb = int(os_module.getenv("CELERY_WORKER_MAX_MEMORY_KB", "2000000"))
        max_memory_mb = max_memory_kb / 1024
        baseline_mb = memory_info.rss / 1024 / 1024
        logger.info(
            f"Worker process {process.pid} initialized. "
            f"Baseline memory (RSS): {baseline_mb:.2f} MB "
            f"(includes Python runtime + imported modules). "
            f"Max memory limit: {max_memory_mb:.2f} MB. "
            f"Memory pressure threshold: {max_memory_mb * 0.80:.2f} MB (80%). "
            f"File size limit: {8} MB (configured in CodeChangesManager, reduced from 10MB)"
        )

        # Log system memory info if available
        try:
            system_memory = psutil.virtual_memory()
            logger.info(
                f"System memory: {system_memory.total / 1024 / 1024 / 1024:.2f} GB total, "
                f"{system_memory.available / 1024 / 1024 / 1024:.2f} GB available, "
                f"{system_memory.percent:.1f}% used"
            )
        except Exception:
            pass
    except ImportError:
        # psutil not available, skip detailed logging
        logger.info(
            "Worker process initialized. Install psutil for detailed memory logging."
        )
    except Exception as e:
        logger.debug(f"Could not log worker memory config: {e}")


@worker_process_shutdown.connect
def cleanup_async_tasks_on_shutdown(sender, **kwargs):
    """
    Clean up any pending async tasks before worker process shutdown.
    This prevents "Task was destroyed but it is pending" warnings.
    """
    logger.info("Worker process shutting down, cleaning up async tasks...")

    try:
        # Try to get any running event loop and clean up pending tasks
        try:
            loop = asyncio.get_running_loop()
            pending_tasks = [
                task for task in asyncio.all_tasks(loop) if not task.done()
            ]
            if pending_tasks:
                logger.warning(
                    f"Found {len(pending_tasks)} pending async tasks during worker shutdown"
                )
                # Cancel all pending tasks
                for task in pending_tasks:
                    task_name = getattr(task, "__name__", str(task))
                    logger.debug(f"Cancelling pending task: {task_name}")
                    task.cancel()
        except RuntimeError:
            # No running event loop, nothing to clean up
            pass

        # Re-check litellm configuration one more time
        try:
            import logging

            litellm_logger = logging.getLogger("LiteLLM")
            async_handlers = [
                h
                for h in litellm_logger.handlers
                if hasattr(h, "emit") and asyncio.iscoroutinefunction(h.emit)
            ]
            if async_handlers:
                logger.warning(
                    f"Found {len(async_handlers)} async handlers in LiteLLM logger during shutdown"
                )
        except Exception:
            pass

        logger.info("Async task cleanup completed")
    except Exception as e:
        logger.warning(f"Error during async task cleanup: {e}", exc_info=True)


# Configure LiteLLM for Celery workers before tasks are imported
configure_litellm_for_celery()

# Install SIGSEGV handler for git operations in multiprocessing contexts
try:
    from app.modules.code_provider.git_safe import install_sigsegv_handler

    install_sigsegv_handler()
except Exception as e:
    logger.warning(f"Could not install SIGSEGV handler: {e}")

# Import the lock decorator
from celery.contrib.abortable import AbortableTask  # noqa

# Import tasks to ensure they are registered
import app.celery.tasks.parsing_tasks  # noqa # Ensure the task module is imported
import app.celery.tasks.agent_tasks  # noqa # Ensure the agent task module is imported
import app.modules.event_bus.tasks.event_tasks  # noqa # Ensure event bus tasks are registered
