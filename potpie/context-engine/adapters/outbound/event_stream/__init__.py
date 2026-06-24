"""Event stream publisher adapters.

The deployable adapter is ``RedisEventStreamPublisher``. The in-memory
publisher (``inmemory_publisher.py``) is a test double — import it from its
module directly; it is deliberately not re-exported here.
"""

from adapters.outbound.event_stream.redis_publisher import (
    RedisEventStreamPublisher,
)

__all__ = [
    "RedisEventStreamPublisher",
]
