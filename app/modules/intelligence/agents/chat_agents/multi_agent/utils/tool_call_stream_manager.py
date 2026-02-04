"""Redis stream manager for tool call streaming using call_id as key"""

import asyncio
import redis
import json
from typing import Optional, AsyncGenerator
from datetime import datetime
from functools import partial
from app.core.config_provider import ConfigProvider
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ToolCallStreamManager:
    """Manages Redis streams for tool call responses using call_id as the stream key"""

    def __init__(self):
        config = ConfigProvider()
        self.redis_client = redis.from_url(config.get_redis_url())
        # Use same TTL and max_len as conversation streams
        self.stream_ttl = ConfigProvider.get_stream_ttl_secs()
        self.max_len = ConfigProvider.get_stream_maxlen()

    def stream_key(self, call_id: str) -> str:
        """Generate Redis stream key for a tool call_id"""
        return f"tool_call:stream:{call_id}"

    def _sync_publish_stream_part(
        self,
        key: str,
        event_data: dict,
    ):
        """Synchronous Redis publish - called from thread pool"""
        # Publish to stream with max length limit
        self.redis_client.xadd(key, event_data, maxlen=self.max_len, approximate=True)
        # Refresh TTL
        self.redis_client.expire(key, self.stream_ttl)

    def publish_stream_part(
        self,
        call_id: str,
        stream_part: str,
        is_complete: bool = False,
        tool_response: Optional[str] = None,
        tool_call_details: Optional[dict] = None,
    ):
        """Publish a streaming part to Redis stream for a tool call.

        Note: This method is synchronous but uses non-blocking approach when possible.
        For async contexts, use publish_stream_part_async instead.

        Args:
            call_id: The tool call ID (used as stream key)
            stream_part: The partial content for this stream update
            is_complete: Whether this is the final part (False for partial updates)
            tool_response: Optional full tool response text (for final part)
            tool_call_details: Optional tool call details dict
        """
        key = self.stream_key(call_id)

        event_data = {
            "type": "tool_call_stream_part",
            "call_id": call_id,
            "stream_part": stream_part,
            "is_complete": "true" if is_complete else "false",
            "created_at": datetime.utcnow().isoformat(),
        }

        if tool_response is not None:
            event_data["tool_response"] = tool_response

        if tool_call_details:
            event_data["tool_call_details_json"] = json.dumps(
                tool_call_details,
                default=lambda x: (
                    x.decode("utf-8", errors="replace")
                    if isinstance(x, bytes)
                    else str(x)
                ),
            )

        try:
            self._sync_publish_stream_part(key, event_data)
            logger.debug(f"Published stream part to tool call stream {key}")
        except Exception as e:
            logger.error(
                f"Failed to publish stream part to Redis stream {key}: {str(e)}"
            )
            raise

    async def publish_stream_part_async(
        self,
        call_id: str,
        stream_part: str,
        is_complete: bool = False,
        tool_response: Optional[str] = None,
        tool_call_details: Optional[dict] = None,
    ):
        """Async version of publish_stream_part - runs Redis operations in thread pool.

        This should be used in async contexts to avoid blocking the event loop.

        Args:
            call_id: The tool call ID (used as stream key)
            stream_part: The partial content for this stream update
            is_complete: Whether this is the final part (False for partial updates)
            tool_response: Optional full tool response text (for final part)
            tool_call_details: Optional tool call details dict
        """
        key = self.stream_key(call_id)

        event_data = {
            "type": "tool_call_stream_part",
            "call_id": call_id,
            "stream_part": stream_part,
            "is_complete": "true" if is_complete else "false",
            "created_at": datetime.utcnow().isoformat(),
        }

        if tool_response is not None:
            event_data["tool_response"] = tool_response

        if tool_call_details:
            event_data["tool_call_details_json"] = json.dumps(
                tool_call_details,
                default=lambda x: (
                    x.decode("utf-8", errors="replace")
                    if isinstance(x, bytes)
                    else str(x)
                ),
            )

        try:
            # Run synchronous Redis operations in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,  # Use default thread pool
                partial(self._sync_publish_stream_part, key, event_data),
            )
            logger.debug(f"Published stream part to tool call stream {key} (async)")
        except Exception as e:
            logger.error(
                f"Failed to publish stream part to Redis stream {key}: {str(e)}"
            )
            # Don't raise - just log the error to prevent blocking the main flow

    def _sync_publish_end_event(self, key: str, end_event_data: dict):
        """Synchronous Redis publish for end event - called from thread pool"""
        self.redis_client.xadd(
            key, end_event_data, maxlen=self.max_len, approximate=True
        )
        self.redis_client.expire(key, self.stream_ttl)

    def publish_complete(
        self,
        call_id: str,
        tool_response: str,
        tool_call_details: Optional[dict] = None,
    ):
        """Publish the final complete response for a tool call

        Args:
            call_id: The tool call ID
            tool_response: The complete tool response
            tool_call_details: Optional tool call details dict
        """
        self.publish_stream_part(
            call_id=call_id,
            stream_part=tool_response,  # For complete, stream_part equals full response
            is_complete=True,
            tool_response=tool_response,
            tool_call_details=tool_call_details,
        )
        # Publish end event
        key = self.stream_key(call_id)
        end_event_data = {
            "type": "tool_call_stream_end",
            "call_id": call_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        try:
            self._sync_publish_end_event(key, end_event_data)
        except Exception as e:
            logger.error(f"Failed to publish end event to Redis stream {key}: {str(e)}")

    async def publish_complete_async(
        self,
        call_id: str,
        tool_response: str,
        tool_call_details: Optional[dict] = None,
    ):
        """Async version of publish_complete - runs Redis operations in thread pool.

        Args:
            call_id: The tool call ID
            tool_response: The complete tool response
            tool_call_details: Optional tool call details dict
        """
        await self.publish_stream_part_async(
            call_id=call_id,
            stream_part=tool_response,
            is_complete=True,
            tool_response=tool_response,
            tool_call_details=tool_call_details,
        )
        # Publish end event
        key = self.stream_key(call_id)
        end_event_data = {
            "type": "tool_call_stream_end",
            "call_id": call_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, partial(self._sync_publish_end_event, key, end_event_data)
            )
        except Exception as e:
            logger.error(f"Failed to publish end event to Redis stream {key}: {str(e)}")

    async def consume_stream(
        self, call_id: str, cursor: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """Consume stream events for a tool call asynchronously

        Args:
            call_id: The tool call ID to consume streams for
            cursor: Optional cursor position (for reconnection)

        Yields:
            Dictionary with stream event data
        """
        key = self.stream_key(call_id)

        try:
            loop = asyncio.get_event_loop()

            # Only replay existing events if cursor is explicitly provided
            events = []
            if cursor:
                events = await loop.run_in_executor(
                    None, partial(self.redis_client.xrange, key, min=cursor, max="+")
                )
                for event_id, event_data in events:
                    formatted_event = self._format_event(event_id, event_data)
                    yield formatted_event

            # Set starting point for live events
            if cursor and events:
                last_id = events[-1][0]
            else:
                key_exists = await loop.run_in_executor(
                    None, self.redis_client.exists, key
                )
                if key_exists:
                    # For fresh requests, start from the latest event in the stream
                    latest_events = await loop.run_in_executor(
                        None, partial(self.redis_client.xrevrange, key, count=1)
                    )
                    last_id = latest_events[0][0] if latest_events else "0-0"
                else:
                    last_id = "0-0"

            # If no cursor provided (fresh request), wait for stream to be created
            key_exists_check = await loop.run_in_executor(
                None, self.redis_client.exists, key
            )
            if not cursor and not key_exists_check:
                # Wait for the stream to be created (with timeout)
                # Generous timeout to allow for complex tool operations
                wait_timeout = 660  # 11 minutes for tool call streams (aligns with delegation timeout)
                wait_start = datetime.now()

                while True:
                    key_exists_now = await loop.run_in_executor(
                        None, self.redis_client.exists, key
                    )
                    if key_exists_now:
                        break

                    if (datetime.now() - wait_start).total_seconds() > wait_timeout:
                        yield {
                            "type": "tool_call_stream_end",
                            "call_id": call_id,
                            "status": "timeout",
                            "message": "Stream creation timeout",
                            "stream_id": "0-0",
                        }
                        return

                    # Check every 500ms
                    await asyncio.sleep(0.5)

            # Stream live events
            while True:
                # Check if key still exists (TTL expiry detection)
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                key_exists = await loop.run_in_executor(
                    None, self.redis_client.exists, key
                )
                if not key_exists:
                    yield {
                        "type": "tool_call_stream_end",
                        "call_id": call_id,
                        "status": "expired",
                        "message": "Stream expired",
                        "stream_id": last_id,
                    }
                    return

                # Use blocking read with timeout - run in thread pool to avoid blocking event loop
                # block=1000 means 1 second timeout in Redis
                events = await loop.run_in_executor(
                    None,
                    partial(
                        self.redis_client.xread, {key: last_id}, block=1000, count=10
                    ),
                )
                if not events:
                    continue

                for stream_key, stream_events in events:
                    for event_id, event_data in stream_events:
                        last_id = event_id
                        event = self._format_event(event_id, event_data)

                        # Check for end events
                        if event.get("type") == "tool_call_stream_end":
                            logger.debug(f"Tool call stream {key} ended")
                            yield event
                            return

                        yield event

        except Exception as e:
            logger.error(f"Error consuming tool call stream {key}: {str(e)}")
            yield {
                "type": "tool_call_stream_end",
                "call_id": call_id,
                "status": "error",
                "message": f"Stream error: {str(e)}",
                "stream_id": cursor or "0-0",
            }

    def _format_event(self, event_id, event_data: dict) -> dict:
        """Format Redis stream event for consumption"""
        stream_id_str = event_id.decode() if isinstance(event_id, bytes) else event_id
        formatted = {"stream_id": stream_id_str}

        for k, v in event_data.items():
            key_str = k.decode() if isinstance(k, bytes) else k
            value_str = v.decode() if isinstance(v, bytes) else v

            if key_str.endswith("_json"):
                try:
                    parsed_value = json.loads(value_str)
                    formatted_key = key_str.replace("_json", "")
                    formatted[formatted_key] = parsed_value
                except Exception as e:
                    logger.error(f"Failed to parse {key_str}: {value_str}, error: {e}")
                    formatted[key_str.replace("_json", "")] = {}
            else:
                formatted[key_str] = value_str

        return formatted

    def cleanup_stream(self, call_id: str):
        """Clean up a tool call stream (delete the key)"""
        key = self.stream_key(call_id)
        try:
            self.redis_client.delete(key)
            logger.debug(f"Cleaned up tool call stream {key}")
        except Exception as e:
            logger.warning(f"Failed to cleanup tool call stream {key}: {str(e)}")
