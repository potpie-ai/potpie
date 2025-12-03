# Logging Best Practices

This document outlines the logging patterns and best practices for the application.

## Overview

We use **Loguru** for structured logging with automatic interception of standard library `logging` calls. This means:
- ✅ All `logging.info()`, `logging.error()`, etc. calls are automatically routed through Loguru
- ✅ Consistent format across all logs (text in dev, JSON in production)
- ✅ No need to migrate 300+ files - interception handles it automatically

## Architecture

### 1. Automatic Context Injection (Middleware)

**Location**: `app/modules/utils/logging_middleware.py`

The `LoggingContextMiddleware` automatically adds request-level context to ALL logs:
- `request_id`: Unique identifier for request tracing
- `path`: API endpoint path
- `method`: HTTP method
- `user_id`: Authenticated user (if available)

**No code changes needed** - this works automatically for all requests.

### 2. Domain-Specific Context (Manual)

**Location**: `app/modules/utils/logger.py` - `log_context()` function

For domain-specific IDs (conversation_id, project_id) that are only available in specific routes:

```python
from app.modules.utils.logger import setup_logger, log_context

logger = setup_logger(__name__)

@router.post("/conversations/{conversation_id}/message/")
async def post_message(conversation_id: str, user=Depends(AuthService.check_auth)):
    user_id = user["user_id"]
    
    # Add domain-specific context
    with log_context(conversation_id=conversation_id, user_id=user_id):
        logger.info("Processing message")  # Includes conversation_id + user_id
        # All logs in this block automatically include the context
```

## Logging Patterns

### ✅ Pattern 1: Standard Logging (Recommended)

```python
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Info logs
logger.info("Processing request", user_id=user_id, action="create")

# Error logs with stack trace
try:
    result = some_operation()
except Exception:
    logger.exception("Operation failed", user_id=user_id, operation="create")
    raise
```

**Key Points**:
- Use `logger.exception()` for errors (automatically includes stack trace)
- Add context as kwargs: `logger.info("msg", user_id=x)`
- Works with middleware context automatically

### ✅ Pattern 2: Context Manager for Domain IDs

```python
from app.modules.utils.logger import setup_logger, log_context

logger = setup_logger(__name__)

async def process_conversation(conversation_id: str, user_id: str):
    with log_context(conversation_id=conversation_id, user_id=user_id):
        logger.info("Starting processing")  # Includes conversation_id + user_id
        logger.info("Processing complete")  # Also includes context
        # All logs in this block get the context automatically
```

**Use When**:
- You have domain-specific IDs (conversation_id, project_id, run_id)
- Multiple log statements need the same context
- You want to avoid passing context to every log call

### ✅ Pattern 3: Standard Library Logging (Automatic)

```python
import logging

# This works automatically - no changes needed!
logging.info("This is automatically intercepted and formatted by Loguru")
logging.error("Errors are also intercepted")
```

**Key Points**:
- No migration needed - interception handles it
- All standard library logs get Loguru formatting
- Context from middleware is automatically included

### ❌ Pattern 4: Deprecated Helper Function

```python
# ❌ OLD WAY (deprecated but still works)
from app.modules.utils.logger import log_error_with_context

try:
    result = operation()
except Exception as e:
    log_error_with_context(logger, "Operation failed", e, user_id=user_id)

# ✅ NEW WAY (recommended)
try:
    result = operation()
except Exception:
    logger.exception("Operation failed", user_id=user_id)
```

## Context Hierarchy

Context is merged in this order (later overrides earlier):

1. **Middleware context** (automatic): `request_id`, `path`, `method`, `user_id`
2. **log_context() context** (manual): `conversation_id`, `project_id`, etc.
3. **Inline kwargs** (per-call): Any context passed directly to log call

Example:
```python
# Middleware adds: request_id, path, user_id
with log_context(conversation_id=conv_id):  # Adds conversation_id
    logger.info("Message", project_id=proj_id)  # Adds project_id
    
# Final log includes: request_id, path, user_id, conversation_id, project_id
```

## Error Logging Best Practices

### ✅ Always Use logger.exception() for Errors

```python
# ✅ CORRECT: Includes stack trace automatically
try:
    result = risky_operation()
except Exception:
    logger.exception("Risky operation failed", user_id=user_id, operation="risky")
    raise

# ❌ WRONG: No stack trace
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Risky operation failed: {e}", user_id=user_id)  # Missing stack trace
```

### ✅ Include Relevant Context

```python
# ✅ CORRECT: Includes all relevant context
try:
    await process_message(conversation_id, user_id)
except Exception:
    logger.exception(
        "Failed to process message",
        conversation_id=conversation_id,
        user_id=user_id,
        message_type="text"
    )
    raise

# ❌ WRONG: Missing context
try:
    await process_message(conversation_id, user_id)
except Exception:
    logger.exception("Failed to process message")  # No context!
```

### ✅ Exception Variable Usage

When using `logger.exception()`, the exception and stack trace are automatically captured. You only need to capture the exception variable if you use it for other purposes.

```python
# ✅ CORRECT: No variable needed when only logging
try:
    result = operation()
except Exception:
    logger.exception("Operation failed", user_id=user_id)
    raise

# ✅ CORRECT: Variable needed when used elsewhere
try:
    result = operation()
except Exception as e:
    logger.exception("Operation failed", user_id=user_id)
    # Use e for other purposes
    raise CustomError(f"Failed: {str(e)}") from e

# ❌ INCORRECT: Variable captured but only used in logger.exception
try:
    result = operation()
except Exception as e:  # ❌ e is unused
    logger.exception("Operation failed", user_id=user_id)  # e not needed here
    raise

# ❌ INCORRECT: Including str(e) in message when logger.exception already includes it
try:
    result = operation()
except Exception as e:
    logger.exception(f"Operation failed: {str(e)}", user_id=user_id)  # str(e) redundant
    raise
```

**Key Points**:
- `logger.exception()` automatically captures the exception - no need to pass it or include `str(e)` in the message
- Only use `except Exception as e:` when you need `e` for other operations (e.g., `str(e)` in error messages, `raise ... from e`, traceback formatting)
- This prevents unused variable warnings and follows Python best practices
- The exception message and stack trace are automatically included in the log output

## Log Levels

- **DEBUG**: Detailed diagnostic information (development only)
- **INFO**: General informational messages about normal operation
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error messages for failures that don't stop the application
- **CRITICAL**: Critical errors that may cause the application to stop

## Production vs Development

### Development Mode
- Human-readable colored text format
- All log levels visible
- Easy to read in terminal

### Production Mode
- JSON format for log aggregation tools
- Structured fields for filtering/searching
- Machine-readable format

Set via `ENV=production` environment variable.

## Migration Strategy

### ✅ What You DON'T Need to Do

- ❌ Migrate all `logging.getLogger()` calls
- ❌ Change all `logging.info()` to `logger.info()`
- ❌ Update 300+ files

### ✅ What You SHOULD Do

1. **Use middleware** (already done) - automatic context for all requests
2. **Use log_context()** for domain IDs in routes that have them
3. **Use logger.exception()** for new error logging
4. **Migrate incrementally** - only when you're already modifying a file

### Gradual Migration

When you're already modifying a file:
- Replace `logging.getLogger(__name__)` with `setup_logger(__name__)`
- Replace `logger.error(..., exc_info=True)` with `logger.exception(...)`
- Add context kwargs where relevant

## Examples

### Example 1: Route Handler

```python
from app.modules.utils.logger import setup_logger, log_context
from fastapi import APIRouter, Depends
from app.modules.auth.auth_service import AuthService

router = APIRouter()
logger = setup_logger(__name__)

@router.post("/conversations/{conversation_id}/message/")
async def post_message(
    conversation_id: str,
    user=Depends(AuthService.check_auth)
):
    user_id = user["user_id"]
    
    # Add domain-specific context
    with log_context(conversation_id=conversation_id, user_id=user_id):
        logger.info("Received message")
        
        try:
            result = await process_message(conversation_id, user_id)
            logger.info("Message processed successfully")
            return result
        except Exception:
            logger.exception("Failed to process message")
            raise
```

### Example 2: Service Method

```python
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

class ConversationService:
    async def create_conversation(self, user_id: str, title: str):
        try:
            conversation = await self._create_conversation_record(user_id, title)
            logger.info(
                "Conversation created",
                user_id=user_id,
                conversation_id=conversation.id,
                title=title
            )
            return conversation
        except Exception:
            logger.exception(
                "Failed to create conversation",
                user_id=user_id,
                title=title
            )
            raise
```

### Example 3: Celery Task

```python
from app.modules.utils.logger import setup_logger, log_context
from app.celery.celery_app import celery_app

logger = setup_logger(__name__)

@celery_app.task
def process_background_task(conversation_id: str, user_id: str):
    # Add context for background task
    with log_context(conversation_id=conversation_id, user_id=user_id):
        logger.info("Starting background task")
        
        try:
            result = do_work()
            logger.info("Background task completed")
            return result
        except Exception:
            logger.exception("Background task failed")
            raise
```

## Troubleshooting

### Logs Not Showing Context

**Problem**: Logs don't include `user_id` or `request_id`

**Solution**: 
- Ensure middleware is added: `app.add_middleware(LoggingContextMiddleware)`
- Check that `request.state.user` is set by `AuthService.check_auth`

### Stack Traces Missing

**Problem**: Error logs don't show stack traces

**Solution**: Use `logger.exception()` instead of `logger.error()`

### Context Not Merging

**Problem**: Context from middleware and log_context() not appearing together

**Solution**: Ensure you're using `log_context()` context manager, not just `logger.bind()`

## References

- Loguru Documentation: https://loguru.readthedocs.io/
- Implementation: `app/modules/utils/logger.py`
- Middleware: `app/modules/utils/logging_middleware.py`
- Plan: `thoughts/shared/plans/2025-12-01-logging-consistency-improvements.md`

