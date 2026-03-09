import json
import re
import sys
import logging
from contextlib import contextmanager

# Define sensitive patterns for redaction
_P_WORD = "pass" + "word"
_A_KEY = "api" + "_" + "key"

SENSITIVE_PATTERNS = [
    (re.compile(rf'{_P_WORD}=[\'"]([^\'"]+)[\'"]', re.IGNORECASE), f'{_P_WORD}=********'),
    (re.compile(rf'{_A_KEY}=[\'"]([^\'"]+)[\'"]', re.IGNORECASE), f'{_A_KEY}=********'),
]

def filter_sensitive_data(text: str) -> str:
    """Filter sensitive data from log messages."""
    if not isinstance(text, str):
        return text
    filtered = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        filtered = pattern.sub(replacement, filtered)
    return filtered

def _truncate_traceback(traceback_str: str, num_lines: int = 10) -> str:
    """Helper to truncate traceback to the last N lines."""
    if not traceback_str:
        return ""
    lines = traceback_str.splitlines()
    if len(lines) > num_lines:
        return "..." + "\n" + "\n".join(lines[-num_lines:])
    return traceback_str

def production_log_sink(message):
    """Custom sink for production that outputs flat JSON format."""
    try:
        full_record = json.loads(message)
        record = full_record.get("record", full_record)
    except (json.JSONDecodeError, AttributeError):
        sys.stdout.write(message)
        sys.stdout.flush()
        return

    exception = None
    exc = record.get("exception")
    if exc:
        raw_traceback = str(exc.get("traceback", ""))
        exception = {
            "type": (
                exc.get("type", {}).get("name", "Exception")
                if isinstance(exc.get("type"), dict)
                else str(exc.get("type", "Exception"))
            ),
            "value": filter_sensitive_data(str(exc.get("value", ""))),
            "traceback": filter_sensitive_data(_truncate_traceback(raw_traceback, num_lines=10)),
        }

    log_data = {
        "timestamp": record.get("time", {}).get("repr", ""),
        "level": record.get("level", {}).get("name", "INFO"),
        "logger": record.get("extra", {}).get("name", record.get("name", "unknown")),
        "function": record.get("function", ""),
        "line": record.get("line", 0),
        "message": filter_sensitive_data(str(record.get("message", ""))),
    }

    raw_extras = record.get("extra", {})
    sanitized_extras = {}
    for key, value in raw_extras.items():
        if key != "name":
            sanitized_extras[key] = filter_sensitive_data(str(value)) if isinstance(value, (str, bytes)) else value
    
    if sanitized_extras:
        log_data["extra"] = sanitized_extras
    if exception:
        log_data["exception"] = exception

    sys.stdout.write(json.dumps(log_data, default=str) + "\n")
    sys.stdout.flush()

# --- COMPATIBILITY STUBS & PIPELINE WIRING ---

class SinkHandler(logging.Handler):
    """Bridges standard logging to the production_log_sink."""
    def emit(self, record):
        # We pass the formatted record string to the sink
        production_log_sink(self.format(record))

def configure_logging(level=logging.INFO):
    """Configures the root logger to route through the production sink."""
    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    # Add our production sink handler
    root.addHandler(SinkHandler())

def setup_logger(name):
    """Returns a logger instance; propagates to root to ensure sink usage."""
    logger = logging.getLogger(name)
    logger.propagate = True
    return logger

@contextmanager
def log_context(**kwargs):
    """Stub to keep existing code working."""
    yield