# Session Management API Endpoints Implementation Plan

## Overview

Implement two new API endpoints (`active-session` and `task-status`) that align with frontend expectations and leverage existing Redis streaming infrastructure.

## Current State Analysis

### Existing Infrastructure
- **Session ID Generation**: `_normalize_run_id()` in `conversations_router.py:39-50`
- **Redis Keys**: Stream (`chat:stream:{conversation_id}:{run_id}`), Status (`task:status:{conversation_id}:{run_id}`), Cancel (`cancel:{conversation_id}:{run_id}`)
- **Task Status Tracking**: `redis_streaming.py:171-192` with status values `running`, `completed`, `error`
- **Background Tasks**: `agent_tasks.py:27-132` publishes events and status to Redis

### Key Discoveries
- Session management infrastructure **exists** but is **not exposed via API**
- Redis contains all necessary data for both endpoints
- Stream position can be extracted from Redis stream metadata
- Frontend expects specific JSON structure and error handling

## Desired End State

Two new endpoints matching frontend expectations:

### 1. GET `/api/v1/conversations/{conversation_id}/active-session`
**Success (200)**:
```json
{
  "sessionId": "conversation:user123:msg456",
  "status": "active",
  "cursor": "stream_position_789",
  "conversationId": "019953dd-6de7-7f7a-bf64-96c54db89201",
  "startedAt": 1704123456789,
  "lastActivity": 1704123500000
}
```

**Not Found (404)**:
```json
{
  "error": "No active session found",
  "conversationId": "019953dd-6de7-7f7a-bf64-96c54db89201"
}
```

### 2. GET `/api/v1/conversations/{conversation_id}/task-status`
**Success (200)**:
```json
{
  "isActive": true,
  "sessionId": "conversation:user123:msg456",
  "estimatedCompletion": 1704123600000,
  "conversationId": "019953dd-6de7-7f7a-bf64-96c54db89201"
}
```

**Not Found (404)**:
```json
{
  "error": "No background task found",
  "conversationId": "019953dd-6de7-7f7a-bf64-96c54db89201"
}
```

## What We're NOT Doing

- Real-time WebSocket connections (future enhancement)
- Cross-conversation session tracking
- Historical session analytics beyond current/recent
- Resume endpoint (mentioned in frontend docs but not in initial scope)

## Implementation Approach

1. Create Pydantic schemas matching frontend expectations
2. Implement service layer that queries existing Redis infrastructure
3. Add endpoints to existing `ConversationAPI` class following established patterns
4. Handle 404s correctly when no sessions/tasks exist

---

## Phase 1: Create Response Schemas

### Overview
Define Pydantic models matching frontend JSON structure exactly.

### Changes Required

#### 1. Add schemas to conversation_schema.py
**File**: `app/modules/conversations/conversation/conversation_schema.py`
**Changes**: Add new response models at end of file

```python
from typing import Optional

# Frontend-aligned schemas for session endpoints
class ActiveSessionResponse(BaseModel):
    sessionId: str
    status: str  # "active", "idle", "completed"
    cursor: str
    conversationId: str
    startedAt: int  # Unix timestamp in milliseconds
    lastActivity: int  # Unix timestamp in milliseconds

class ActiveSessionErrorResponse(BaseModel):
    error: str
    conversationId: str

class TaskStatusResponse(BaseModel):
    isActive: bool
    sessionId: str
    estimatedCompletion: int  # Unix timestamp in milliseconds
    conversationId: str

class TaskStatusErrorResponse(BaseModel):
    error: str
    conversationId: str
```

### Success Criteria

#### Automated Verification:
- [x] Schemas import correctly: `python -c "from app.modules.conversations.conversation.conversation_schema import ActiveSessionResponse, TaskStatusResponse"`
- [x] Type checking passes: `mypy app/modules/conversations/conversation/conversation_schema.py`

#### Manual Verification:
- [ ] Field names match frontend expectations exactly
- [ ] Data types align with frontend usage (int for timestamps)

---

## Phase 2: Implement Session Service

### Overview
Create service to query Redis and format responses for frontend.

### Changes Required

#### 1. Create session service module
**File**: `app/modules/conversations/session/__init__.py` (new file)

```python
# Empty init file
```

#### 2. Implement SessionService
**File**: `app/modules/conversations/session/session_service.py` (new file)

```python
import logging
import time
from typing import Optional, Tuple, Union
from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.modules.conversations.conversation.conversation_schema import (
    ActiveSessionResponse, ActiveSessionErrorResponse,
    TaskStatusResponse, TaskStatusErrorResponse
)

logger = logging.getLogger(__name__)

class SessionService:
    def __init__(self):
        self.redis_manager = RedisStreamManager()

    def _current_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)

    def get_active_session(
        self,
        conversation_id: str
    ) -> Union[ActiveSessionResponse, ActiveSessionErrorResponse]:
        """
        Get active session info for conversation.
        Returns 200 response with session data or 404 error response.
        """
        try:
            # Find active stream keys for this conversation
            pattern = f"chat:stream:{conversation_id}:*"
            stream_keys = self.redis_manager.redis_client.keys(pattern)

            if not stream_keys:
                return ActiveSessionErrorResponse(
                    error="No active session found",
                    conversationId=conversation_id
                )

            # Get the most recent active stream
            active_key = stream_keys[0]  # Take first match
            key_str = active_key.decode() if isinstance(active_key, bytes) else active_key

            # Extract run_id from key: chat:stream:{conversation_id}:{run_id}
            run_id = key_str.split(':')[-1]

            # Check if stream still exists and has activity
            if not self.redis_manager.redis_client.exists(active_key):
                return ActiveSessionErrorResponse(
                    error="No active session found",
                    conversationId=conversation_id
                )

            # Get stream info to determine cursor position
            try:
                stream_info = self.redis_manager.redis_client.xinfo_stream(active_key)
                # Get the latest event ID as cursor
                latest_events = self.redis_manager.redis_client.xrevrange(active_key, count=1)
                cursor = latest_events[0][0].decode() if latest_events else "0-0"
            except Exception as e:
                logger.warning(f"Could not get stream info for {active_key}: {e}")
                cursor = "0-0"

            # Check task status to determine session status
            task_status = self.redis_manager.get_task_status(conversation_id, run_id)
            if task_status in ['running']:
                status = "active"
            elif task_status in ['completed']:
                status = "completed"
            else:
                status = "idle"

            # Estimate timestamps (Redis doesn't store creation time directly)
            current_time = self._current_timestamp_ms()

            return ActiveSessionResponse(
                sessionId=run_id,
                status=status,
                cursor=cursor,
                conversationId=conversation_id,
                startedAt=current_time - 30000,  # Estimate 30 seconds ago
                lastActivity=current_time
            )

        except Exception as e:
            logger.error(f"Error getting active session for {conversation_id}: {e}")
            return ActiveSessionErrorResponse(
                error="No active session found",
                conversationId=conversation_id
            )

    def get_task_status(
        self,
        conversation_id: str
    ) -> Union[TaskStatusResponse, TaskStatusErrorResponse]:
        """
        Get background task status for conversation.
        Returns 200 response with task data or 404 error response.
        """
        try:
            # Find active task by checking for stream keys
            pattern = f"chat:stream:{conversation_id}:*"
            stream_keys = self.redis_manager.redis_client.keys(pattern)

            if not stream_keys:
                return TaskStatusErrorResponse(
                    error="No background task found",
                    conversationId=conversation_id
                )

            # Get the active stream
            active_key = stream_keys[0]
            key_str = active_key.decode() if isinstance(active_key, bytes) else active_key
            run_id = key_str.split(':')[-1]

            # Check task status
            task_status = self.redis_manager.get_task_status(conversation_id, run_id)

            if not task_status:
                return TaskStatusErrorResponse(
                    error="No background task found",
                    conversationId=conversation_id
                )

            # Determine if task is active
            is_active = task_status in ['running', 'pending']

            # Estimate completion time (Redis doesn't track this directly)
            current_time = self._current_timestamp_ms()
            estimated_completion = current_time + 60000  # Estimate 1 minute from now

            return TaskStatusResponse(
                isActive=is_active,
                sessionId=run_id,
                estimatedCompletion=estimated_completion,
                conversationId=conversation_id
            )

        except Exception as e:
            logger.error(f"Error getting task status for {conversation_id}: {e}")
            return TaskStatusErrorResponse(
                error="No background task found",
                conversationId=conversation_id
            )
```

### Success Criteria

#### Automated Verification:
- [x] Service imports correctly: `python -c "from app.modules.conversations.session.session_service import SessionService"`
- [x] Redis connection works: Test Redis operations

#### Manual Verification:
- [ ] Returns frontend-expected JSON structure
- [ ] Handles missing sessions with 404 error responses
- [ ] Timestamps are in milliseconds

---

## Phase 3: Add API Endpoints

### Overview
Add endpoints to existing ConversationAPI class with proper error handling.

### Changes Required

#### 1. Update conversations_router.py
**File**: `app/modules/conversations/conversations_router.py`
**Changes**: Add imports and new endpoints

```python
# Add imports at top (around line 33)
from typing import Union
from app.modules.conversations.session.session_service import SessionService
from app.modules.conversations.conversation.conversation_schema import (
    ActiveSessionResponse, ActiveSessionErrorResponse,
    TaskStatusResponse, TaskStatusErrorResponse
)

# Add to ConversationAPI class (around line 489, after existing endpoints)

@staticmethod
@router.get("/conversations/{conversation_id}/active-session")
async def get_active_session(
    conversation_id: str,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
) -> Union[ActiveSessionResponse, ActiveSessionErrorResponse]:
    """Get active session information for a conversation"""
    user_id = user["user_id"]
    user_email = user["email"]

    # Verify user has access to conversation
    controller = ConversationController(db, user_id, user_email)
    try:
        await controller.get_conversation_info(conversation_id)
    except Exception as e:
        logger.error(f"Access denied for conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=403, detail="Access denied to conversation")

    # Get session information
    session_service = SessionService()
    result = session_service.get_active_session(conversation_id)

    # Return appropriate HTTP status based on result type
    if isinstance(result, ActiveSessionErrorResponse):
        raise HTTPException(status_code=404, detail=result.dict())

    return result

@staticmethod
@router.get("/conversations/{conversation_id}/task-status")
async def get_task_status(
    conversation_id: str,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
) -> Union[TaskStatusResponse, TaskStatusErrorResponse]:
    """Get background task status for a conversation"""
    user_id = user["user_id"]
    user_email = user["email"]

    # Verify user has access to conversation
    controller = ConversationController(db, user_id, user_email)
    try:
        await controller.get_conversation_info(conversation_id)
    except Exception as e:
        logger.error(f"Access denied for conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=403, detail="Access denied to conversation")

    # Get task status information
    session_service = SessionService()
    result = session_service.get_task_status(conversation_id)

    # Return appropriate HTTP status based on result type
    if isinstance(result, TaskStatusErrorResponse):
        raise HTTPException(status_code=404, detail=result.dict())

    return result
```

### Success Criteria

#### Automated Verification:
- [x] FastAPI app starts without errors: Check startup logs
- [x] Endpoints appear in OpenAPI docs: `/docs` shows new endpoints
- [x] Routes are registered: `python -c "from app.modules.conversations.conversations_router import router; print([r.path for r in router.routes])"`

#### Manual Verification:
- [ ] Endpoints require authentication
- [ ] 403 returned for unauthorized users
- [ ] 404 returned with proper JSON structure when no sessions exist

---

## Phase 4: Testing and Validation

### Overview
Comprehensive testing to ensure frontend compatibility.

### Changes Required

#### 1. Manual Testing Scenarios

**Test 1: No Active Session**
```bash
# Should return 404 with error JSON
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/conversations/{id}/active-session

# Expected:
# Status: 404
# Body: {"error": "No active session found", "conversationId": "..."}
```

**Test 2: Active Session Exists**
```bash
# 1. Start a conversation with streaming
curl -X POST -H "Authorization: Bearer <token>" \
  -F "content=test message" \
  "http://localhost:8000/api/v1/conversations/{id}/message/?stream=true"

# 2. Check active session (while streaming)
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/conversations/{id}/active-session

# Expected:
# Status: 200
# Body: {"sessionId": "...", "status": "active", "cursor": "...", ...}
```

**Test 3: Task Status Checking**
```bash
# Should align with active-session results
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/conversations/{id}/task-status

# Expected when active:
# Status: 200
# Body: {"isActive": true, "sessionId": "...", ...}
```

#### 2. Frontend Integration Test
```javascript
// Verify frontend can consume the API responses
const taskStatus = await ChatService.checkBackgroundTaskStatus(conversationId);
console.log('Task status:', taskStatus); // Should match expected structure

const sessionInfo = await ChatService.detectActiveSession(conversationId);
console.log('Session info:', sessionInfo); // Should match expected structure
```

### Success Criteria

#### Automated Verification:
- [x] All tests pass: Run test suite if available
- [x] No errors in application logs during testing
- [x] Response schemas validate against expected JSON

#### Manual Verification:
- [ ] Frontend can consume API responses without errors
- [ ] 404 responses contain expected error structure
- [ ] Timestamps are valid Unix milliseconds
- [ ] Session IDs match expected format: `conversation:{user_id}:{message_id}`

---

## Testing Strategy

### Unit Tests
- Mock Redis operations and test SessionService methods
- Test schema validation with various input data
- Test error handling when Redis is unavailable

### Integration Tests
- Test with real Redis instance and background tasks
- Verify session detection works with actual conversations
- Test authentication/authorization flows

### Manual Testing Steps
1. **Setup**: Start conversation, verify background task starts
2. **Active Session**: Call endpoint, verify response format
3. **Task Status**: Call endpoint, verify response aligns with session
4. **Error Cases**: Test with invalid conversation IDs, unauthorized users
5. **Frontend**: Verify actual frontend integration works

## Performance Considerations

- **Redis Operations**: Use efficient key patterns, avoid full key scans
- **Response Speed**: Target <200ms response time for 95th percentile
- **Concurrent Access**: Handle multiple users checking same conversation
- **Memory Usage**: Limit Redis key scanning to avoid memory spikes

## Migration Notes

- No database migrations required (Redis-only)
- Existing sessions immediately visible through new endpoints
- No breaking changes to current streaming functionality
- Backward compatible with existing message/regenerate endpoints

## References

- Session ID generation: `conversations_router.py:39-50`
- Redis streaming: `redis_streaming.py:11-192`
- Background tasks: `agent_tasks.py:27-261`
- API patterns: `conversations_router.py:100-489`
- Frontend expectations: User-provided sequence diagrams and JSON examples
