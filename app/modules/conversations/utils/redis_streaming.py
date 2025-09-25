import redis
from typing import Generator, Optional
import json
from datetime import datetime
from app.core.config_provider import ConfigProvider
import logging

logger = logging.getLogger(__name__)


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
                wait_timeout = 30  # 30 seconds
                wait_start = datetime.now()

                while not self.redis_client.exists(key):
                    if (datetime.now() - wait_start).total_seconds() > wait_timeout:
                        yield {
                            "type": "end",
                            "status": "timeout",
                            "message": "Stream creation timeout",
                            "stream_id": "0-0",
                        }
                        return

                    # Check every 500ms
                    import time

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

    def wait_for_task_start(
        self, conversation_id: str, run_id: str, timeout: int = 10
    ) -> bool:
        """Wait for background task to signal it has started"""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = self.get_task_status(conversation_id, run_id)
            if status in ["running", "completed", "error"]:
                return True
            import time

            time.sleep(0.5)
        return False
