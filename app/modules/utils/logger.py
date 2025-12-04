import json
import logging
import os
import sys
from contextlib import contextmanager
from loguru import logger as _loguru_logger
from typing import Optional

_LOGGING_CONFIGURED = False
_logger = _loguru_logger


def production_log_sink(message):
    """Custom sink for production that outputs flat JSON format for better machine readability.

    When serialize=True, loguru outputs JSON string. We parse it and reformat as flat JSON
    for easier parsing by log aggregation tools (ELK, Datadog, Splunk, CloudWatch, etc.).
    """
    try:
        # Parse the serialized JSON from loguru
        full_record = json.loads(message)
        record = full_record.get("record", full_record)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: if message is not JSON, output as-is (shouldn't happen with serialize=True)
        sys.stdout.write(message)
        sys.stdout.flush()
        return

    # Extract exception info if present
    exception = None
    exc = record.get("exception")
    if exc:
        exception = {
            "type": exc.get("type", {}).get("name", "Exception")
            if isinstance(exc.get("type"), dict)
            else str(exc.get("type", "Exception")),
            "value": exc.get("value", ""),
            "traceback": exc.get("traceback", ""),
        }

    # Build flat JSON structure - easier for log parsers
    log_data = {
        "timestamp": record.get("time", {}).get("repr", ""),
        "level": record.get("level", {}).get("name", "INFO"),
        "logger": record.get("extra", {}).get("name", record.get("name", "unknown")),
        "function": record.get("function", ""),
        "line": record.get("line", 0),
        "message": record.get("message", ""),
    }

    # Add all extra fields (conversation_id, user_id, etc.) at top level
    extras = record.get("extra", {})
    for key, value in extras.items():
        if key != "name":  # Already included as "logger"
            log_data[key] = value

    # Add exception if present
    if exception:
        log_data["exception"] = exception

    # Write flat JSON to stdout (one JSON object per line - JSONL format)
    sys.stdout.write(json.dumps(log_data, default=str) + "\n")
    sys.stdout.flush()


class InterceptHandler(logging.Handler):
    """Intercept standard library logging and route through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _logger.bind(name=record.name).opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(level: Optional[str] = None):
    """
    Configure unified logging with loguru.

    Standard practice:
    1. Your app logs: Use loguru directly
    2. Third-party logs: Intercept selectively with appropriate levels
    3. Filter at sink level for production vs development
    """
    global _LOGGING_CONFIGURED, _logger

    if _LOGGING_CONFIGURED:
        return

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    env = os.getenv("ENV", "development")

    _logger.remove()

    def patcher(record):
        if "name" not in record["extra"]:
            record["extra"]["name"] = record.get(
                "name", record.get("module", "unknown")
            )

    _logger = _logger.patch(patcher)

    if env == "production":
        # Production: Flat JSON format for better machine readability
        # This format is easier for log aggregation tools (ELK, Datadog, Splunk, etc.)
        # Use serialize=True to get structured data, then format as flat JSON
        _logger.add(
            production_log_sink,
            format="{message}",
            level=level,
            serialize=True,  # Get structured record, then format in sink
        )
    else:
        _logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level> - {extra}",
            level=level,
            colorize=True,
        )

    intercept_handler = InterceptHandler()

    # Define logging levels for different libraries
    library_levels = {
        # Your application - let through at configured level
        "app": level,
        "api": level,
        "services": level,
        "models": level,
        # Infrastructure - reduce verbosity
        "uvicorn": "INFO",
        "uvicorn.access": "WARNING",
        "uvicorn.error": "INFO",
        "fastapi": "INFO",
        # Database - CRITICAL: Set appropriate levels
        "sqlalchemy.engine": "WARNING",
        "sqlalchemy.pool": "WARNING",
        "sqlalchemy.orm": "WARNING",
        "alembic": "INFO",
        # Task queue
        "celery": "INFO",
        "kombu": "WARNING",
        # HTTP clients
        "httpx": "WARNING",
        "urllib3": "WARNING",
        # AWS/Cloud
        "boto3": "WARNING",
        "botocore": "WARNING",
        # Add your other libraries here
    }

    # Configure root logger with environment-based level
    # Respect LOG_LEVEL environment variable for root logger
    log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
    if log_level_env == "DEBUG":
        root_level = logging.DEBUG
    else:
        root_level = logging.INFO

    logging.basicConfig(
        handlers=[intercept_handler],
        level=root_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Apply levels to specific loggers
    for logger_name, log_level in library_levels.items():
        lib_logger = logging.getLogger(logger_name)
        lib_logger.handlers = [intercept_handler]
        lib_logger.setLevel(log_level)
        lib_logger.propagate = False

    # Catch-all for any other loggers: set to WARNING by default
    # This prevents unknown libraries from spamming logs
    logging.getLogger().setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


@contextmanager
def log_context(**kwargs):
    """
    Context manager to add domain IDs to all logs within the context.

    Uses Loguru's native contextualize() which is thread-safe and async-safe.

    Usage:
        with log_context(conversation_id=conv_id, user_id=user_id):
            logger.info("This log will include conversation_id and user_id")
            logger.exception("Error occurred")  # Also includes context
    """
    with _logger.contextualize(**kwargs):
        yield


def setup_logger(name: str):
    """
    Setup a logger with the given name.

    Standard practice: Use this for YOUR application code.
    For third-party libraries, let the interception handle it.

    For context, use log_context() context manager or add context as kwargs:
        logger.info("Message", user_id=user_id, conversation_id=conv_id)
        logger.exception("Error", user_id=user_id)  # Includes stack trace + context
    """
    if not _LOGGING_CONFIGURED:
        configure_logging()

    return _logger.bind(name=name)


# Convenience function for dynamic level adjustment
def set_library_log_level(library_name: str, level: str):
    """
    Dynamically adjust log level for a specific library.

    Usage:
        set_library_log_level("sqlalchemy.engine", "DEBUG")  # Temporarily debug SQL
        set_library_log_level("sqlalchemy.engine", "WARNING")  # Back to normal
    """
    lib_logger = logging.getLogger(library_name)
    lib_logger.setLevel(level)
    _logger.info(f"Set log level for '{library_name}' to {level}")


def log_error_with_context(logger, message: str, error: Exception, **context):
    """
    DEPRECATED: Use logger.exception() with context kwargs instead.

    This helper is kept for backward compatibility during migration.
    Prefer the native Loguru pattern:

    ✅ RECOMMENDED:
        try:
            # code
        except Exception as e:
            logger.exception("Error message", user_id=user_id, project_id=project_id)

    ❌ OLD WAY (still works but not recommended):
        log_error_with_context(logger, "Error message", e, user_id=user_id)

    The native pattern is better because:
    - Uses Loguru's built-in exception handling
    - Context is added as kwargs (more Pythonic)
    - Works seamlessly with log_context() context manager
    """
    # Use logger.exception() with context - Loguru's native way
    logger.bind(**context).exception(f"{message}: {str(error)}")
