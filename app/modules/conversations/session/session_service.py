import logging
import time
from typing import Union
from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.modules.conversations.conversation.conversation_schema import (
    ActiveSessionResponse,
    ActiveSessionErrorResponse,
    TaskStatusResponse,
    TaskStatusErrorResponse,
)

logger = logging.getLogger(__name__)


class SessionService:
    def __init__(self):
        self.redis_manager = RedisStreamManager()

    def _current_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)

    def get_active_session(
        self, conversation_id: str
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
                    error="No active session found", conversationId=conversation_id
                )

            # Sort stream keys to get the most recent session
            stream_keys_decoded = [
                key.decode() if isinstance(key, bytes) else key for key in stream_keys
            ]
            stream_keys_sorted = sorted(stream_keys_decoded, reverse=True)

            # Get the most recent active stream
            active_key = stream_keys_sorted[0]
            key_str = active_key

            # Extract run_id from key: chat:stream:{conversation_id}:{run_id}
            # The run_id is everything after 'chat:stream:{conversation_id}:'
            prefix = f"chat:stream:{conversation_id}:"
            run_id = key_str[len(prefix) :]

            # Check if stream still exists and has activity
            if not self.redis_manager.redis_client.exists(active_key):
                return ActiveSessionErrorResponse(
                    error="No active session found", conversationId=conversation_id
                )

            # Get stream info to determine cursor position
            try:
                stream_info = self.redis_manager.redis_client.xinfo_stream(active_key)
                # Get the latest event ID as cursor
                latest_events = self.redis_manager.redis_client.xrevrange(
                    active_key, count=1
                )
                cursor = latest_events[0][0].decode() if latest_events else "0-0"
            except Exception as e:
                logger.warning(f"Could not get stream info for {active_key}: {e}")
                cursor = "0-0"

            # Check task status to determine session status
            task_status = self.redis_manager.get_task_status(conversation_id, run_id)
            if task_status in ["running"]:
                status = "active"
            elif task_status in ["completed"]:
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
                lastActivity=current_time,
            )

        except Exception as e:
            logger.error(f"Error getting active session for {conversation_id}: {e}")
            return ActiveSessionErrorResponse(
                error="No active session found", conversationId=conversation_id
            )

    def get_task_status(
        self, conversation_id: str
    ) -> Union[TaskStatusResponse, TaskStatusErrorResponse]:
        """
        Get background task status for conversation.
        Returns 200 response with task data or 404 error response.
        """
        try:
            # Find task by checking for stream keys (most recent first)
            pattern = f"chat:stream:{conversation_id}:*"
            stream_keys = self.redis_manager.redis_client.keys(pattern)

            if not stream_keys:
                return TaskStatusErrorResponse(
                    error="No background task found", conversationId=conversation_id
                )

            # Sort stream keys to get the most recent session
            # Stream keys have format: chat:stream:{conv_id}:{run_id}
            # The run_id often ends with numbers (e.g., -1, -2), so we sort to get the highest
            stream_keys_decoded = [
                key.decode() if isinstance(key, bytes) else key for key in stream_keys
            ]
            stream_keys_sorted = sorted(stream_keys_decoded, reverse=True)

            # Try each stream key until we find one with a valid task status
            for key_str in stream_keys_sorted:
                # Extract full run_id from key: chat:stream:{conversation_id}:{run_id}
                prefix = f"chat:stream:{conversation_id}:"
                run_id = key_str[len(prefix) :]
                task_status = self.redis_manager.get_task_status(
                    conversation_id, run_id
                )

                if task_status:
                    # Determine if task is active
                    is_active = task_status in ["running", "pending"]

                    # Set appropriate completion time
                    current_time = self._current_timestamp_ms()
                    if is_active:
                        # For active tasks, estimate completion in the future
                        estimated_completion = current_time + 60000  # 1 minute from now
                    else:
                        # For completed tasks, set completion time in the past
                        estimated_completion = current_time - 5000  # 5 seconds ago

                    return TaskStatusResponse(
                        isActive=is_active,
                        sessionId=run_id,
                        estimatedCompletion=estimated_completion,
                        conversationId=conversation_id,
                    )

            # If no valid task status found for any stream key
            return TaskStatusErrorResponse(
                error="No background task found", conversationId=conversation_id
            )

        except Exception as e:
            logger.error(f"Error getting task status for {conversation_id}: {e}")
            return TaskStatusErrorResponse(
                error="No background task found", conversationId=conversation_id
            )
