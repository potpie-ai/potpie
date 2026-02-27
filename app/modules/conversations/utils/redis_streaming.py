import asyncio
import json
import redis
import time
from datetime import datetime
from typing import Generator, Optional

from app.core.config_provider import ConfigProvider
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class RedisStreamManager:
    def __init__(self):
        config = ConfigProvider()
        redis_url = config.get_redis_url()
        # Prevent indefinite blocking: Redis can hang without timeouts (e.g. under load)
        self.redis_client = redis.from_url(
            redis_url,
            socket_connect_timeout=10,
            socket_timeout=30,
            decode_responses=False,
        )
        self.stream_ttl = ConfigProvider.get_stream_ttl_secs()
        self.max_len = ConfigProvider.get_stream_maxlen()

    def stream_key(self, conversation_id: str, run_id: str) -> str:
        return f"chat:stream:{conversation_id}:{run_id}"

    def publish_event(
        self, conversation_id: str, run_id: str, event_type: str, payload: dict
    ):
        """Synchronous Redis stream publishing for Celery tasks"""
        key = self.stream_key(conversation_id, run_id)

        def serialize_value(v):
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            elif isinstance(v, (dict, list)):
                return json.dumps(
                    v,
                    default=lambda x: (
                        x.decode("utf-8", errors="replace")
                        if isinstance(x, bytes)
                        else str(x)
                    ),
                )
            else:
                return str(v)

        event_data = {
            "type": event_type,
            "conversation_id": conversation_id,
            "run_id": run_id,
            "created_at": datetime.utcnow().isoformat(),
            **{k: serialize_value(v) for k, v in payload.items()},
        }

        try:
            # Publish to stream with max length limit
            self.redis_client.xadd(
                key, event_data, maxlen=self.max_len, approximate=True
            )

            # Refresh TTL
            self.redis_client.expire(key, self.stream_ttl)

            logger.debug(f"Published {event_type} event to stream {key}")
        except Exception as e:
            logger.error(f"Failed to publish event to Redis stream {key}: {str(e)}")
            raise

    def consume_stream(
        self, conversation_id: str, run_id: str, cursor: Optional[str] = None
    ) -> Generator[dict, None, None]:
        """Synchronous Redis stream consumption for HTTP streaming"""
        key = self.stream_key(conversation_id, run_id)

        try:
            # Only replay existing events if cursor is explicitly provided (for reconnection)
            events = []
            if cursor:
                events = self.redis_client.xrange(key, min=cursor, max="+")

                for event_id, event_data in events:
                    formatted_event = self._format_event(event_id, event_data)
                    yield formatted_event

            # Set starting point for live events
            if cursor and events:
                last_id = events[-1][0]
            elif self.redis_client.exists(key):
                # For fresh requests, start from the latest event in the stream
                # to avoid replaying old messages
                latest_events = self.redis_client.xrevrange(key, count=1)
                last_id = latest_events[0][0] if latest_events else "0-0"
            else:
                last_id = "0-0"

            # If no cursor provided (fresh request), wait for stream to be created
            if not cursor and not self.redis_client.exists(key):
                # Wait for the stream to be created (with timeout)
                # Increased timeout to 120 seconds to handle queued Celery tasks
                wait_timeout = 120  # 2 minutes
                wait_start = datetime.now()

                while not self.redis_client.exists(key):
                    if (datetime.now() - wait_start).total_seconds() > wait_timeout:
                        yield {
                            "type": "end",
                            "status": "timeout",
                            "message": "Stream creation timeout - task may be queued",
                            "stream_id": "0-0",
                        }
                        return

                    # Check every 500ms
                    time.sleep(0.5)

            while True:
                # Check if key still exists (TTL expiry detection)
                if not self.redis_client.exists(key):
                    yield {
                        "type": "end",
                        "status": "expired",
                        "message": "Stream expired",
                        "stream_id": last_id,
                    }
                    return

                events = self.redis_client.xread({key: last_id}, block=5000, count=1)
                if not events:
                    continue

                for stream_key, stream_events in events:
                    for event_id, event_data in stream_events:
                        last_id = event_id
                        event = self._format_event(event_id, event_data)

                        # Check for end events
                        if event.get("type") == "end":
                            logger.info(
                                f"Stream {key} ended with status: {event.get('status')}"
                            )
                            yield event
                            return

                        yield event

        except Exception as e:
            logger.error(f"Error consuming Redis stream {key}: {str(e)}")
            yield {
                "type": "end",
                "status": "error",
                "message": f"Stream error: {str(e)}",
                "stream_id": cursor or "0-0",
            }

    def _format_event(self, event_id, event_data: dict) -> dict:
        """Format Redis stream event for client consumption"""
        # Ensure event_id is string
        stream_id_str = event_id.decode() if isinstance(event_id, bytes) else event_id
        formatted = {"stream_id": stream_id_str}

        for k, v in event_data.items():
            # Ensure key is string for comparison
            key_str = k.decode() if isinstance(k, bytes) else k
            value_str = v.decode() if isinstance(v, bytes) else v

            if key_str.endswith("_json"):
                try:
                    parsed_value = json.loads(value_str)
                    formatted_key = key_str.replace("_json", "")
                    if formatted_key == "tool_calls":
                        pass  # No special handling needed for tool_calls
                    formatted[formatted_key] = parsed_value
                except Exception as e:
                    logger.error(f"Failed to parse {key_str}: {value_str}, error: {e}")
                    formatted[key_str.replace("_json", "")] = []
            else:
                formatted[key_str] = value_str
        return formatted

    def check_cancellation(self, conversation_id: str, run_id: str) -> bool:
        """Check if cancellation signal exists for this conversation/run"""
        cancel_key = f"cancel:{conversation_id}:{run_id}"
        return bool(self.redis_client.get(cancel_key))

    def set_cancellation(self, conversation_id: str, run_id: str) -> None:
        """Set cancellation signal for this conversation/run"""
        cancel_key = f"cancel:{conversation_id}:{run_id}"
        self.redis_client.set(cancel_key, "true", ex=300)  # 5 minute expiry
        logger.info(f"Set cancellation signal for {conversation_id}:{run_id}")

    def set_task_status(self, conversation_id: str, run_id: str, status: str) -> None:
        """Set task status for health checking"""
        status_key = f"task:status:{conversation_id}:{run_id}"
        self.redis_client.set(status_key, status, ex=600)  # 10 minute expiry
        logger.debug(f"Set task status {status} for {conversation_id}:{run_id}")

    def get_task_status(self, conversation_id: str, run_id: str) -> Optional[str]:
        """Get task status for health checking"""
        status_key = f"task:status:{conversation_id}:{run_id}"
        status = self.redis_client.get(status_key)
        return status.decode() if status else None

    def set_task_id(self, conversation_id: str, run_id: str, task_id: str) -> None:
        """Store Celery task ID for this conversation/run"""
        task_id_key = f"task:id:{conversation_id}:{run_id}"
        self.redis_client.set(task_id_key, task_id, ex=600)  # 10 minute expiry
        logger.debug(f"Stored task ID {task_id} for {conversation_id}:{run_id}")

    def get_task_id(self, conversation_id: str, run_id: str) -> Optional[str]:
        """Get Celery task ID for this conversation/run"""
        task_id_key = f"task:id:{conversation_id}:{run_id}"
        task_id = self.redis_client.get(task_id_key)
        return task_id.decode() if task_id else None

    def get_stream_snapshot(self, conversation_id: str, run_id: str) -> dict:
        """
        Read all events from the stream and return accumulated chunk content.
        Used when stopping generation so we can persist partial response before clearing.
        Returns dict with keys: content (str), citations (list), tool_calls (list), chunk_count (int).
        """
        key = self.stream_key(conversation_id, run_id)
        content = ""
        citations = []
        tool_calls = []
        chunk_count = 0
        try:
            if not self.redis_client.exists(key):
                return {
                    "content": content,
                    "citations": citations,
                    "tool_calls": tool_calls,
                    "chunk_count": chunk_count,
                }
            events = self.redis_client.xrange(key, min="-", max="+")
            for event_id, event_data in events:
                formatted = self._format_event(event_id, event_data)
                if formatted.get("type") != "chunk":
                    continue
                chunk_count += 1
                content += formatted.get("content", "") or ""
                for c in formatted.get("citations") or []:
                    if c not in citations:
                        citations.append(c)
                for tc in formatted.get("tool_calls") or []:
                    tool_calls.append(tc)
            return {
                "content": content,
                "citations": citations,
                "tool_calls": tool_calls,
                "chunk_count": chunk_count,
            }
        except Exception as e:
            logger.error(
                f"Failed to get stream snapshot for {conversation_id}:{run_id}: {str(e)}"
            )
            return {
                "content": content,
                "citations": citations,
                "tool_calls": tool_calls,
                "chunk_count": chunk_count,
            }

    def clear_session(self, conversation_id: str, run_id: str) -> None:
        """Clear session data when stopping - publishes end event, then removes all keys from Redis."""
        try:
            # Publish an end event with cancelled status so clients know to stop (before deleting stream)
            self.publish_event(
                conversation_id,
                run_id,
                "end",
                {
                    "status": "cancelled",
                    "message": "Generation stopped by user",
                },
            )

            # Set task status to cancelled so any in-flight consumers see it
            self.set_task_status(conversation_id, run_id, "cancelled")

            # Brief delay so any client blocking on xread can receive the end event before we delete the stream
            time.sleep(0.2)

            # Remove this session's data from Redis so the stream and metadata are gone
            stream_key = self.stream_key(conversation_id, run_id)
            cancel_key = f"cancel:{conversation_id}:{run_id}"
            task_id_key = f"task:id:{conversation_id}:{run_id}"
            status_key = f"task:status:{conversation_id}:{run_id}"
            self.redis_client.delete(stream_key, cancel_key, task_id_key, status_key)

            logger.info(
                f"Cleared session for {conversation_id}:{run_id} (stream and keys removed from Redis)"
            )
        except Exception as e:
            logger.error(
                f"Failed to clear session for {conversation_id}:{run_id}: {str(e)}"
            )

    def wait_for_task_start(
        self,
        conversation_id: str,
        run_id: str,
        timeout: int = 10,
        require_running: bool = True,
    ) -> bool:
        """Wait for background task to signal it has started.
        When require_running=True (default), only returns True when status is 'running'.
        """
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = self.get_task_status(conversation_id, run_id)
            if require_running:
                if status == "running":
                    return True
            else:
                if status in ["queued", "running", "completed", "error"]:
                    return True
            time.sleep(0.5)
        return False


# ---------------------------------------------------------------------------
# Async Redis stream manager for FastAPI (native async, no event-loop blocking)
# ---------------------------------------------------------------------------

try:
    from redis.asyncio import Redis as AsyncRedis
except ImportError:
    AsyncRedis = None  # type: ignore[misc, assignment]

# Exceptions that are worth retrying on transient Redis/network issues
_RETRYABLE_REDIS_EXCEPTIONS: tuple = (
    redis.exceptions.ConnectionError,
    redis.exceptions.TimeoutError,
    redis.exceptions.BusyLoadingError,
    OSError,
)


async def _retry_redis_async(
    coro_factory,
    max_attempts: int = 3,
    base_delay: float = 0.1,
):
    """
    Retry an async Redis operation on transient errors.
    coro_factory: callable that returns a coroutine (e.g. lambda: client.get(key)).
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except _RETRYABLE_REDIS_EXCEPTIONS as e:
            last_exc = e
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "Redis transient error (attempt %s/%s), retrying in %.2fs: %s",
                attempt + 1,
                max_attempts,
                delay,
                e,
            )
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc


def _format_stream_event(event_id, event_data: dict) -> dict:
    """Format Redis stream event for client (shared helper)."""
    stream_id_str = (
        event_id.decode() if isinstance(event_id, bytes) else event_id
    )
    formatted = {"stream_id": stream_id_str}
    for k, v in event_data.items():
        key_str = k.decode() if isinstance(k, bytes) else k
        value_str = v.decode() if isinstance(v, bytes) else v
        if key_str.endswith("_json"):
            try:
                parsed_value = json.loads(value_str)
                formatted[key_str.replace("_json", "")] = parsed_value
            except Exception as e:
                logger.error(f"Failed to parse {key_str}: {value_str}, error: {e}")
                formatted[key_str.replace("_json", "")] = []
        else:
            formatted[key_str] = value_str
    return formatted


class AsyncRedisStreamManager:
    """Async Redis stream manager for FastAPI routes. Uses redis.asyncio."""

    def __init__(self, max_connections: int = 50):
        if AsyncRedis is None:
            raise RuntimeError("redis.asyncio not available; install redis>=4.2")
        config = ConfigProvider()
        self.redis_client: AsyncRedis = AsyncRedis.from_url(
            config.get_redis_url(),
            max_connections=max_connections,
        )
        self.stream_ttl = ConfigProvider.get_stream_ttl_secs()
        self.max_len = ConfigProvider.get_stream_maxlen()

    def stream_key(self, conversation_id: str, run_id: str) -> str:
        return f"chat:stream:{conversation_id}:{run_id}"

    async def set_task_status(
        self, conversation_id: str, run_id: str, status: str
    ) -> None:
        status_key = f"task:status:{conversation_id}:{run_id}"
        await _retry_redis_async(
            lambda: self.redis_client.set(status_key, status, ex=600)
        )
        logger.debug(f"Set task status {status} for {conversation_id}:{run_id}")

    async def get_task_status(
        self, conversation_id: str, run_id: str
    ) -> Optional[str]:
        status_key = f"task:status:{conversation_id}:{run_id}"
        status = await _retry_redis_async(
            lambda: self.redis_client.get(status_key)
        )
        return status.decode() if isinstance(status, bytes) else (status or None)

    async def set_task_id(
        self, conversation_id: str, run_id: str, task_id: str
    ) -> None:
        task_id_key = f"task:id:{conversation_id}:{run_id}"
        await _retry_redis_async(
            lambda: self.redis_client.set(task_id_key, task_id, ex=600)
        )
        logger.debug(f"Stored task ID {task_id} for {conversation_id}:{run_id}")

    async def get_task_id(
        self, conversation_id: str, run_id: str
    ) -> Optional[str]:
        task_id_key = f"task:id:{conversation_id}:{run_id}"
        task_id = await _retry_redis_async(
            lambda: self.redis_client.get(task_id_key)
        )
        return (
            task_id.decode() if isinstance(task_id, bytes) else (task_id or None)
        )

    async def publish_event(
        self,
        conversation_id: str,
        run_id: str,
        event_type: str,
        payload: dict,
    ) -> None:
        key = self.stream_key(conversation_id, run_id)

        def serialize_value(v):
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            elif isinstance(v, (dict, list)):
                return json.dumps(
                    v,
                    default=lambda x: (
                        x.decode("utf-8", errors="replace")
                        if isinstance(x, bytes)
                        else str(x)
                    ),
                )
            else:
                return str(v)

        event_data = {
            "type": event_type,
            "conversation_id": conversation_id,
            "run_id": run_id,
            "created_at": datetime.utcnow().isoformat(),
            **{k: serialize_value(v) for k, v in payload.items()},
        }
        try:
            await _retry_redis_async(
                lambda: self.redis_client.xadd(
                    key, event_data, maxlen=self.max_len, approximate=True
                )
            )
            await _retry_redis_async(
                lambda: self.redis_client.expire(key, self.stream_ttl)
            )
            logger.debug(f"Published {event_type} event to stream {key}")
        except Exception as e:
            logger.error(f"Failed to publish event to Redis stream {key}: {str(e)}")
            raise

    async def set_cancellation(self, conversation_id: str, run_id: str) -> None:
        cancel_key = f"cancel:{conversation_id}:{run_id}"
        await _retry_redis_async(
            lambda: self.redis_client.set(cancel_key, "true", ex=300)
        )
        logger.info(f"Set cancellation signal for {conversation_id}:{run_id}")

    async def get_stream_snapshot(
        self, conversation_id: str, run_id: str
    ) -> dict:
        key = self.stream_key(conversation_id, run_id)
        content = ""
        citations = []
        tool_calls = []
        chunk_count = 0
        try:
            exists = await _retry_redis_async(
                lambda: self.redis_client.exists(key)
            )
            if not exists:
                return {
                    "content": content,
                    "citations": citations,
                    "tool_calls": tool_calls,
                    "chunk_count": chunk_count,
                }
            events = await _retry_redis_async(
                lambda: self.redis_client.xrange(key, min="-", max="+")
            )
            for event_id, event_data in events:
                formatted = _format_stream_event(event_id, event_data)
                if formatted.get("type") != "chunk":
                    continue
                chunk_count += 1
                content += formatted.get("content", "") or ""
                for c in formatted.get("citations") or []:
                    if c not in citations:
                        citations.append(c)
                for tc in formatted.get("tool_calls") or []:
                    tool_calls.append(tc)
            return {
                "content": content,
                "citations": citations,
                "tool_calls": tool_calls,
                "chunk_count": chunk_count,
            }
        except Exception as e:
            logger.error(
                f"Failed to get stream snapshot for {conversation_id}:{run_id}: {e}"
            )
            return {
                "content": content,
                "citations": citations,
                "tool_calls": tool_calls,
                "chunk_count": chunk_count,
            }

    async def clear_session(self, conversation_id: str, run_id: str) -> None:
        try:
            await self.publish_event(
                conversation_id,
                run_id,
                "end",
                {"status": "cancelled", "message": "Generation stopped by user"},
            )
            await self.set_task_status(conversation_id, run_id, "cancelled")
            await asyncio.sleep(0.2)
            stream_key = self.stream_key(conversation_id, run_id)
            cancel_key = f"cancel:{conversation_id}:{run_id}"
            task_id_key = f"task:id:{conversation_id}:{run_id}"
            status_key = f"task:status:{conversation_id}:{run_id}"
            await _retry_redis_async(
                lambda: self.redis_client.delete(
                    stream_key, cancel_key, task_id_key, status_key
                )
            )
            logger.info(
                f"Cleared session for {conversation_id}:{run_id} (stream and keys removed)"
            )
        except Exception as e:
            logger.error(
                f"Failed to clear session for {conversation_id}:{run_id}: {e}"
            )

    async def wait_for_task_start(
        self,
        conversation_id: str,
        run_id: str,
        timeout: int = 10,
        require_running: bool = True,
    ) -> bool:
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = await self.get_task_status(conversation_id, run_id)
            if require_running:
                if status == "running":
                    return True
            else:
                if status in ["queued", "running", "completed", "error"]:
                    return True
            await asyncio.sleep(0.5)
        return False

    async def aclose(self) -> None:
        await self.redis_client.aclose()
        logger.debug("AsyncRedisStreamManager connection closed")
