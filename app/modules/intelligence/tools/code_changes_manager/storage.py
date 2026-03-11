"""Redis persistence for code changes."""

import json
from datetime import datetime
from typing import Dict, Any

import redis

from app.modules.utils.logger import setup_logger

from .constants import CODE_CHANGES_TTL_SECONDS
from .models import ChangeType, FileChange

logger = setup_logger(__name__)


def load_changes_from_redis(
    redis_client: redis.Redis, redis_key: str
) -> Dict[str, FileChange]:
    """Load changes from Redis into a dict.

    Args:
        redis_client: Redis client instance
        redis_key: Redis key to load from

    Returns:
        Dict mapping file_path to FileChange
    """
    try:
        data = redis_client.get(redis_key)
        if data:
            json_str = data.decode("utf-8") if isinstance(data, bytes) else data
            parsed = json.loads(json_str)
            changes = {}
            for change_data in parsed.get("changes", []):
                change = FileChange(
                    file_path=change_data["file_path"],
                    change_type=ChangeType(change_data["change_type"]),
                    content=change_data.get("content"),
                    previous_content=change_data.get("previous_content"),
                    created_at=change_data.get("created_at", datetime.now().isoformat()),
                    updated_at=change_data.get("updated_at", datetime.now().isoformat()),
                    description=change_data.get("description"),
                )
                changes[change.file_path] = change
            logger.debug(
                f"storage.load_changes_from_redis: Loaded {len(changes)} changes from Redis"
            )
            return changes
        else:
            logger.debug("storage.load_changes_from_redis: No existing data in Redis")
            return {}
    except Exception as e:
        logger.warning(f"storage.load_changes_from_redis: Error loading: {e}")
        return {}


def save_changes_to_redis(
    redis_client: redis.Redis,
    redis_key: str,
    changes: Dict[str, FileChange],
    conversation_id: str | None,
    ttl: int = CODE_CHANGES_TTL_SECONDS,
) -> None:
    """Save changes to Redis with expiry.

    Args:
        redis_client: Redis client instance
        redis_key: Redis key to save to
        changes: Dict of file_path -> FileChange
        conversation_id: Conversation ID for metadata
        ttl: TTL in seconds
    """
    try:
        data: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "changes": [
                {
                    "file_path": change.file_path,
                    "change_type": change.change_type.value,
                    "content": change.content,
                    "previous_content": change.previous_content,
                    "created_at": change.created_at,
                    "updated_at": change.updated_at,
                    "description": change.description,
                }
                for change in changes.values()
            ],
        }
        json_str = json.dumps(data)
        redis_client.setex(redis_key, ttl, json_str)
        logger.debug(
            f"storage.save_changes_to_redis: Saved {len(changes)} changes (key={redis_key}, ttl={ttl}s)"
        )
    except Exception as e:
        logger.error(f"storage.save_changes_to_redis: Error saving: {e}")
