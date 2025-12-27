"""In-memory async queue for Potpie (replaces Redis streams for local mode)."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional
from uuid import uuid4


@dataclass
class QueueMessage:
    """A message in the queue."""

    id: str
    data: Any
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class AsyncQueue:
    """In-memory async queue for local mode.

    Replaces Redis streams for message passing and event handling.
    """

    def __init__(self, maxlen: int = 1000):
        """Initialize queue.

        Args:
            maxlen: Maximum number of messages to retain.
        """
        self._queues: dict[str, asyncio.Queue] = {}
        self._maxlen = maxlen

    def _get_queue(self, channel: str) -> asyncio.Queue:
        """Get or create a queue for a channel."""
        if channel not in self._queues:
            self._queues[channel] = asyncio.Queue(maxsize=self._maxlen)
        return self._queues[channel]

    async def publish(self, channel: str, data: Any) -> str:
        """Publish a message to a channel.

        Args:
            channel: Channel name.
            data: Message data.

        Returns:
            Message ID.
        """
        queue = self._get_queue(channel)
        msg_id = str(uuid4())
        message = QueueMessage(id=msg_id, data=data)

        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            # Remove oldest message if full
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(message)

        return msg_id

    async def subscribe(
        self, channel: str, timeout: Optional[float] = None
    ) -> AsyncIterator[QueueMessage]:
        """Subscribe to a channel and receive messages.

        Args:
            channel: Channel name.
            timeout: Optional timeout per message.

        Yields:
            QueueMessage objects.
        """
        queue = self._get_queue(channel)

        while True:
            try:
                if timeout:
                    message = await asyncio.wait_for(queue.get(), timeout=timeout)
                else:
                    message = await queue.get()
                yield message
            except asyncio.TimeoutError:
                break

    async def get(
        self, channel: str, timeout: Optional[float] = None
    ) -> Optional[QueueMessage]:
        """Get a single message from a channel.

        Args:
            channel: Channel name.
            timeout: Optional timeout.

        Returns:
            QueueMessage or None if timeout.
        """
        queue = self._get_queue(channel)

        try:
            if timeout:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                return await queue.get()
        except asyncio.TimeoutError:
            return None

    def channel_size(self, channel: str) -> int:
        """Get number of pending messages in a channel."""
        if channel not in self._queues:
            return 0
        return self._queues[channel].qsize()

    def clear_channel(self, channel: str) -> int:
        """Clear all messages from a channel.

        Returns:
            Number of messages cleared.
        """
        if channel not in self._queues:
            return 0

        count = 0
        queue = self._queues[channel]
        while not queue.empty():
            try:
                queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count

    def clear_all(self) -> None:
        """Clear all channels."""
        self._queues.clear()
