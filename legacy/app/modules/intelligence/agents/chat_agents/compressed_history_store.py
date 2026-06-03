"""
Phase 3: Persist compressed or summarized history per conversation.

Provides conversation-scoped storage for the final message list after each agent run,
so the next request in the same conversation can start from compressed history
instead of raw "last N text blobs" from ctx.history.

Storage is keyed by conversation_id (and optionally user_id for per-user isolation).
"""

import logging
import threading
import time
from typing import List, Optional, Protocol

from pydantic_ai.messages import ModelMessage

logger = logging.getLogger(__name__)


class CompressedHistoryStore(Protocol):
    """Protocol for conversation-scoped compressed history storage.

    Implementations may be in-memory (with TTL/eviction) or Redis for
    multi-instance/long-lived persistence.
    """

    def get(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[List[ModelMessage]]:
        """Return stored compressed history for the conversation, or None."""
        ...

    def set(
        self,
        conversation_id: str,
        messages: List[ModelMessage],
        user_id: Optional[str] = None,
    ) -> None:
        """Store the final message list for the conversation (overwrites existing)."""
        ...

    def delete(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Remove stored history for the conversation."""
        ...


def _storage_key(conversation_id: str, user_id: Optional[str] = None) -> str:
    """Build cache key. Phase 3 uses conversation_id only; user_id reserved for future."""
    if user_id:
        return f"{user_id}:{conversation_id}"
    return conversation_id


class InMemoryCompressedHistoryStore:
    """In-memory store with TTL and max-size eviction.

    Key: conversation_id (or composite user_id:conversation_id if user_id provided).
    Value: List[ModelMessage] (no serialization).
    Entries expire after TTL seconds; oldest by last-write evicted when at capacity.
    """

    def __init__(
        self,
        ttl_seconds: int = 86400,
        max_conversations: int = 500,
    ):
        self._ttl_seconds = max(1, ttl_seconds)
        self._max_conversations = max(1, max_conversations)
        self._data: dict[str, tuple[List[ModelMessage], float]] = {}
        self._lock = threading.Lock()

    def get(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[List[ModelMessage]]:
        key = _storage_key(conversation_id, user_id)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            messages, written_at = entry
            if time.monotonic() - written_at > self._ttl_seconds:
                del self._data[key]
                return None
            return messages

    def set(
        self,
        conversation_id: str,
        messages: List[ModelMessage],
        user_id: Optional[str] = None,
    ) -> None:
        key = _storage_key(conversation_id, user_id)
        now = time.monotonic()
        with self._lock:
            if key in self._data:
                self._data[key] = (messages, now)
                return
            while len(self._data) >= self._max_conversations:
                oldest_key = min(
                    self._data.keys(),
                    key=lambda k: self._data[k][1],
                )
                del self._data[oldest_key]
                logger.debug(
                    "Compressed history store evicted conversation (max size): %s",
                    oldest_key,
                )
            self._data[key] = (messages, now)

    def delete(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
    ) -> None:
        key = _storage_key(conversation_id, user_id)
        with self._lock:
            self._data.pop(key, None)


_store: Optional[CompressedHistoryStore] = None
_store_lock = threading.Lock()


def get_compressed_history_store() -> Optional[CompressedHistoryStore]:
    """Return the global compressed history store, or None if disabled.

    The store is created lazily using config from context_config.
    """
    global _store
    with _store_lock:
        if _store is not None:
            return _store
        from app.modules.intelligence.agents.context_config import (
            COMPRESSED_HISTORY_STORE_BACKEND,
            use_persisted_compressed_history,
        )

        if not use_persisted_compressed_history():
            return None
        if COMPRESSED_HISTORY_STORE_BACKEND != "memory":
            # Phase 3b: Redis backend would be created here
            return None
        from app.modules.intelligence.agents.context_config import (
            COMPRESSED_HISTORY_MAX_CONVERSATIONS,
            COMPRESSED_HISTORY_TTL_SECONDS,
        )

        _store = InMemoryCompressedHistoryStore(
            ttl_seconds=COMPRESSED_HISTORY_TTL_SECONDS,
            max_conversations=COMPRESSED_HISTORY_MAX_CONVERSATIONS,
        )
        return _store


def reset_compressed_history_store() -> None:
    """Reset the global store (for tests)."""
    global _store
    with _store_lock:
        _store = None
