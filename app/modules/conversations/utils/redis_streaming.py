"""
Redis stream manager for chat streaming. Uses sync redis client (no decode_responses)
so stream keys and values match Celery task usage. Tunnel/socket services use
decode_responses=True; keep that in mind when sharing Redis URLs or adding new clients.
"""
import redis
import threading
import time
from typing import Generator, Optional
import json
from datetime import datetime
from app.core.config_provider import ConfigProvider
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class RedisStreamManager:
    def __init__(self):
        config = ConfigProvider()
        self.redis_client = redis.from_url(config.get_redis_url())
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
        self,
        conversation_id: str,
        run_id: str,
        cursor: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[dict, None, None]:
        """Synchronous Redis stream consumption for HTTP streaming.
        If stop_event is set, the generator exits promptly so the caller can shut down.
        """
        key = self.stream_key(conversation_id, run_id)

        try:
            # Only replay existing events if cursor is explicitly provided (for reconnection)
            events = []
            if cursor:
                events = self.redis_client.xrange(key, min=cursor, max="+")

                for event_id, event_data in events:
                    if stop_event and stop_event.is_set():
                        return
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
                wait_timeout = 120  # 2 minutes
                wait_start = datetime.now()

                while not self.redis_client.exists(key):
                    if stop_event and stop_event.is_set():
                        return
                    if (datetime.now() - wait_start).total_seconds() > wait_timeout:
                        yield {
                            "type": "end",
                            "status": "timeout",
                            "message": "Stream creation timeout - task may be queued",
                            "stream_id": "0-0",
                        }
                        return
                    time.sleep(0.5)

            while True:
                if stop_event and stop_event.is_set():
                    return
                # Check if key still exists (TTL expiry detection)
                if not self.redis_client.exists(key):
                    yield {
                        "type": "end",
                        "status": "expired",
                        "message": "Stream expired",
                        "stream_id": last_id,
                    }
                    return

                # Use 1s block so we can check stop_event at least every second
                events = self.redis_client.xread({key: last_id}, block=1000, count=1)
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
        require_running: bool = False,
    ) -> bool:
        """
        Wait for background task to signal it has started.
        If require_running is True, only returns True when status is running/completed/error
        (worker has picked up the task); otherwise also accepts 'queued' (can false-positive
        since we set 'queued' before the worker runs).
        """
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = self.get_task_status(conversation_id, run_id)
            if require_running:
                if status in ("running", "completed", "error"):
                    return True
            else:
                if status in ("queued", "running", "completed", "error"):
                    return True
            time.sleep(0.5)
        return False
