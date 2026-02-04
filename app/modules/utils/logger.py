import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from typing import Optional

from loguru import logger as _loguru_logger

_LOGGING_CONFIGURED = False
_logger = _loguru_logger

# Control whether to include stack traces in error logs
# Set LOG_STACK_TRACES=false to disable stack traces
SHOW_STACK_TRACES = os.getenv("LOG_STACK_TRACES", "true").lower() in (
    "true",
    "1",
    "yes",
)

# Sensitive data patterns to redact in logs
SENSITIVE_PATTERNS = [
    # Credentials in key=value format
    (
        re.compile(r'(password|passwd|pwd)=["\']?([^"\'\s&]+)', re.IGNORECASE),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(
            r'(token|access_token|refresh_token|id_token)=["\']?([^"\'\s&]+)',
            re.IGNORECASE,
        ),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(
            r'(secret|client_secret|api_secret)=["\']?([^"\'\s&]+)', re.IGNORECASE
        ),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(r'(api[_-]?key|apikey)=["\']?([^"\'\s&]+)', re.IGNORECASE),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(r'(auth|authorization)=["\']?([^"\'\s&]+)', re.IGNORECASE),
        r"\1=***REDACTED***",
    ),
    # Bearer tokens
    (
        re.compile(r"Bearer\s+([A-Za-z0-9\-._~+/]+=*)", re.IGNORECASE),
        r"Bearer ***REDACTED***",
    ),
    # Basic auth
    (re.compile(r"Basic\s+([A-Za-z0-9+/]+=*)", re.IGNORECASE), r"Basic ***REDACTED***"),
    # Redis/Database URLs with passwords
    (
        re.compile(
            r"(redis|postgresql|mysql|mongodb)://([^:]+):([^@]+)@", re.IGNORECASE
        ),
        r"\1://\2:***REDACTED***@",
    ),
    # OAuth authorization codes (typically 20-100 chars alphanumeric)
    (
        re.compile(r"([?&]code=)([A-Za-z0-9\-._~]{20,100})([&\s]|$)", re.IGNORECASE),
        r"\1***REDACTED***\3",
    ),
    # Generic secrets in quotes
    (
        re.compile(
            r'("(?:password|token|secret|api_key)"\s*:\s*)"([^"]+)"', re.IGNORECASE
        ),
        r'\1"***REDACTED***"',
    ),
    (
        re.compile(
            r"('(?:password|token|secret|api_key)'\s*:\s*)'([^']+)'", re.IGNORECASE
        ),
        r"\1'***REDACTED***'",
    ),
]


def filter_sensitive_data(text: str) -> str:
    """
    Filter sensitive data from log messages.

    Args:
        text: Log message text to filter

    Returns:
        Filtered text with sensitive data redacted
    """
    if not isinstance(text, str):
        return text

    filtered = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        filtered = pattern.sub(replacement, filtered)

    return filtered


def production_log_sink(message):
    """Custom sink for production that outputs flat JSON format for better machine readability.

    When serialize=True, loguru outputs JSON string. We parse it and reformat as flat JSON
    for easier parsing by log aggregation tools (ELK, Datadog, Splunk, CloudWatch, etc.).

    Also filters sensitive data patterns to prevent credential leakage.
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
            "type": (
                exc.get("type", {}).get("name", "Exception")
                if isinstance(exc.get("type"), dict)
                else str(exc.get("type", "Exception"))
            ),
            "value": filter_sensitive_data(str(exc.get("value", ""))),
            "traceback": filter_sensitive_data(str(exc.get("traceback", ""))),
        }

    # Build flat JSON structure - easier for log parsers
    # Filter sensitive data from message
    log_data = {
        "timestamp": record.get("time", {}).get("repr", ""),
        "level": record.get("level", {}).get("name", "INFO"),
        "logger": record.get("extra", {}).get("name", record.get("name", "unknown")),
        "function": record.get("function", ""),
        "line": record.get("line", 0),
        "message": filter_sensitive_data(str(record.get("message", ""))),
    }

    # Add all extra fields (conversation_id, user_id, etc.) at top level
    # Filter sensitive data from extra fields too
    extras = record.get("extra", {})
    for key, value in extras.items():
        if key != "name":  # Already included as "logger"
            # Convert value to string and filter if it's a string-like type
            if isinstance(value, (str, bytes)):
                log_data[key] = filter_sensitive_data(str(value))
            else:
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

    env = (os.getenv("ENV") or "development").lower().strip()

    _logger.remove()

    def patcher(record):
        if "name" not in record["extra"]:
            record["extra"]["name"] = record.get(
                "name", record.get("module", "unknown")
            )

        # Filter sensitive data from the message
        if "message" in record:
            record["message"] = filter_sensitive_data(str(record["message"]))

    _logger = _logger.patch(patcher)

    if env != "development":
        _logger.add(
            production_log_sink,
            format="{message}",
            level=level,
            serialize=True,  # Get structured record, then format in sink
            backtrace=SHOW_STACK_TRACES,
            diagnose=SHOW_STACK_TRACES,
        )
    else:

        def _filter(record):
            """Filter sensitive data in development logs"""
            record["message"] = filter_sensitive_data(str(record["message"]))
            # Filter extra fields
            extra_value = ""
            for key, value in record.get("extra", {}).items():
                if isinstance(value, (str, bytes)):
                    record["extra"][key] = filter_sensitive_data(str(value))
                if key != "name":
                    extra_value += f" {key}: {record['extra'][key]},"
            if extra_value:
                record["message"] = record["message"] + " |" + extra_value[:-1]
            return True

        _logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
            level=level,
            colorize=True,
            filter=_filter,
            backtrace=SHOW_STACK_TRACES,
            diagnose=SHOW_STACK_TRACES,
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


def should_show_stack_trace() -> bool:
    """Check if stack traces should be shown in logs.

    Controlled by LOG_STACK_TRACES environment variable.
    Set LOG_STACK_TRACES=false to disable stack traces.

    Usage in logging calls:
        logger.warning("Error occurred", exc_info=should_show_stack_trace())
    """
    return SHOW_STACK_TRACES


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
