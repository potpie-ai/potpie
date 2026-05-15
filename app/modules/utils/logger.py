import json
import logging
import re
from contextlib import contextmanager
from typing import Any, Dict, Optional
from functools import wraps

# Sensitive patterns for regex-based masking
SENSITIVE_PATTERNS = [
    re.compile(r'password', re.IGNORECASE),
    re.compile(r'token', re.IGNORECASE),
    re.compile(r'api[_-]?key', re.IGNORECASE),
    re.compile(r'secret', re.IGNORECASE),
]

# Sensitive key identifiers for nested structure masking
SENSITIVE_KEY_IDENTIFIERS = {'password', 'token', 'api_key', 'apikey', 'secret', 'auth', 'authorization'}


def filter_sensitive_data(data: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """
    Recursively filter sensitive data from nested structures.
    Handles dicts, lists, tuples, sets, and applies regex masking to strings.
    """
    if depth > max_depth:
        return data
    
    # Handle dictionaries
    if isinstance(data, dict):
        filtered = {}
        for key, value in data.items():
            # Check if key matches sensitive identifiers
            key_lower = str(key).lower()
            if any(identifier in key_lower for identifier in SENSITIVE_KEY_IDENTIFIERS):
                filtered[key] = '***MASKED***'
            else:
                # Apply regex masking to key names
                is_sensitive_key = any(pattern.search(key_lower) for pattern in SENSITIVE_PATTERNS)
                if is_sensitive_key:
                    filtered[key] = '***MASKED***'
                else:
                    # Recursively filter the value
                    filtered[key] = filter_sensitive_data(value, depth + 1, max_depth)
        return filtered
    
    # Handle lists and tuples
    elif isinstance(data, (list, tuple)):
        filtered = [filter_sensitive_data(item, depth + 1, max_depth) for item in data]
        return filtered if isinstance(data, list) else tuple(filtered)
    
    # Handle sets
    elif isinstance(data, set):
        return {filter_sensitive_data(item, depth + 1, max_depth) for item in data}
    
    # Handle strings - apply regex masking
    elif isinstance(data, str):
        filtered_str = data
        for pattern in SENSITIVE_PATTERNS:
            filtered_str = pattern.sub('***MASKED***', filtered_str)
        return filtered_str
    
    # Handle bytes - decode, filter, return as string
    elif isinstance(data, bytes):
        try:
            decoded = data.decode('utf-8', errors='replace')
            filtered_str = decoded
            for pattern in SENSITIVE_PATTERNS:
                filtered_str = pattern.sub('***MASKED***', filtered_str)
            return filtered_str
        except Exception:
            return '***MASKED***'
    
    # Return other types unchanged
    return data


class SinkHandler(logging.Handler):
    """Handler that emits structured log records to a production sink."""
    
    def __init__(self, production_log_sink):
        super().__init__()
        self.production_log_sink = production_log_sink
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a structured log record as JSON to the production sink.
        """
        try:
            # Build structured payload from LogRecord
            payload = {
                'timestamp': record.created,
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'pathname': record.pathname,
                'lineno': record.lineno,
                'funcName': record.funcName,
            }
            
            # Add exception/traceback if present
            if record.exc_info:
                formatter = logging.Formatter()
                payload['exception'] = formatter.formatException(record.exc_info)
            
            # Extract and filter extras from record.__dict__
            extras = {}
            standard_keys = {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName', 'relativeCreated',
                'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                'asctime'
            }
            
            for key, value in record.__dict__.items():
                if key not in standard_keys:
                    extras[key] = value
            
            # Filter sensitive data from extras
            if extras:
                payload['extras'] = filter_sensitive_data(extras)
            
            # JSON-serialize and send to sink
            json_payload = json.dumps(payload, default=str)
            self.production_log_sink(json_payload)
            
        except Exception:
            self.handleError(record)


def get_log_context() -> Dict[str, Any]:
    """
    Retrieve the current request-scoped log context.
    Returns a dict with request_id, path, method, user_id, etc.
    """
    # This should be populated from request context (e.g., Flask/FastAPI g object)
    # For now, returning a template structure
    return {
        'request_id': getattr(get_request_context(), 'request_id', None),
        'path': getattr(get_request_context(), 'path', None),
        'method': getattr(get_request_context(), 'method', None),
        'user_id': getattr(get_request_context(), 'user_id', None),
    }


def get_request_context():
    """Helper to get current request context (framework-specific)."""
    # This should be implemented based on your web framework
    # Example for Flask: from flask import g; return g
    # Example for FastAPI: from starlette.requests import get_request_context
    try:
        from flask import g as flask_g
        return flask_g
    except (ImportError, RuntimeError):
        # Return empty context if not in Flask context
        class EmptyContext:
            pass
        return EmptyContext()


@contextmanager
def log_context(**kwargs):
    """
    Context manager to attach request-scoped fields to logs.
    Pushes context fields for the duration of the context.
    """
    import structlog
    
    # Merge user-provided kwargs with get_log_context()
    context_data = get_log_context()
    context_data.update(kwargs)
    
    # Filter sensitive data before binding
    context_data = filter_sensitive_data(context_data)
    
    # Bind context to structlog for the duration
    bound_logger = structlog.get_logger().bind(**context_data)
    
    try:
        yield bound_logger
    finally:
        # Unbind context after the block completes
        pass  # structlog context is thread-local, cleanup happens automatically


def configure_logging(production_log_sink=None):
    """
    Configure the logging system with structured handlers.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Add SinkHandler if production sink is provided
    if production_log_sink:
        sink_handler = SinkHandler(production_log_sink)
        sink_handler.setLevel(logging.INFO)
        root_logger.addHandler(sink_handler)
    
    # Add console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)