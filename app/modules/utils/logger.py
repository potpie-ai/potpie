def _truncate_traceback(traceback_str: str, num_lines: int = 10) -> str:
    """Helper to truncate traceback to the last N lines."""
    if not traceback_str:
        return ""
    lines = traceback_str.splitlines()
    if len(lines) > num_lines:
        # Keeping the last N lines is usually best for identifying the root cause
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
            # FIX: Apply truncation before sensitive data filtering
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

    extras = record.get("extra", {})
    for key, value in extras.items():
        if key != "name":
            if isinstance(value, (str, bytes)):
                log_data[key] = filter_sensitive_data(str(value))
            else:
                log_data[key] = value

    if exception:
        log_data["exception"] = exception

    sys.stdout.write(json.dumps(log_data, default=str) + "\n")
    sys.stdout.flush()