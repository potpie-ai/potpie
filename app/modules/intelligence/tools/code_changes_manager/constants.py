"""Constants for code changes management."""

# Redis key prefix and expiry for code changes
CODE_CHANGES_KEY_PREFIX = "code_changes"
CODE_CHANGES_TTL_SECONDS = 24 * 60 * 60  # 24 hours expiry

# Maximum file size to read into memory (8MB - reduced from 10MB for safety margin)
# This prevents OOM kills when processing very large files
# Reduced to 8MB to leave headroom for Python object overhead and memory spikes
MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024  # 8MB

# Database query timeout in seconds - prevents deadlocks in forked workers
DB_QUERY_TIMEOUT = 15.0  # 15 seconds max for any database query
DB_SESSION_CREATE_TIMEOUT = 10.0  # 10 seconds max for creating a new session

# Memory pressure threshold - skip non-critical operations if memory usage exceeds this
MEMORY_PRESSURE_THRESHOLD = 0.80  # 80% of worker memory limit
