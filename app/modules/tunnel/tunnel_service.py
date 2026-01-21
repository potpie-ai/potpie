"""
Tunnel Service for managing connections to local VS Code extension servers.

This service tracks active tunnel connections, stores tunnel URLs,
and routes requests to the correct local server via tunnel.
"""

import json
import time
import redis
import httpx
from typing import Optional, Dict
from app.modules.utils.logger import setup_logger
from app.core.config_provider import ConfigProvider

logger = setup_logger(__name__)

# Redis key prefix for tunnel connections
TUNNEL_KEY_PREFIX = "tunnel_connection"
TUNNEL_TTL_SECONDS = 24 * 60 * 60  # 24 hours expiry


class TunnelService:
    """Service for managing tunnel connections to local servers"""

    def __init__(self):
        """Initialize TunnelService with Redis connection"""
        config = ConfigProvider()
        redis_url = config.get_redis_url()
        if redis_url:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        else:
            logger.warning("Redis URL not configured, tunnel service will use in-memory storage")
            self.redis_client = None
            self._in_memory_tunnels: Dict[str, str] = {}

    def _get_tunnel_key(self, user_id: str, conversation_id: Optional[str] = None) -> str:
        """Generate Redis key for tunnel connection"""
        if conversation_id:
            return f"{TUNNEL_KEY_PREFIX}:user:{user_id}:conversation:{conversation_id}"
        return f"{TUNNEL_KEY_PREFIX}:user:{user_id}"

    def register_tunnel(
        self,
        user_id: str,
        tunnel_url: str,
        conversation_id: Optional[str] = None,
    ) -> bool:
        """
        Register a tunnel connection for a user/conversation.

        Args:
            user_id: User ID
            tunnel_url: URL of the tunnel (e.g., https://xyz.trycloudflare.com)
            conversation_id: Optional conversation ID for conversation-specific tunnels

        Returns:
            True if registration succeeded
        """
        try:
            key = self._get_tunnel_key(user_id, conversation_id)
            tunnel_data = {
                "tunnel_url": tunnel_url,
                "user_id": user_id,
                "conversation_id": conversation_id or "",
                "registered_at": str(int(time.time())),
            }

            if self.redis_client:
                self.redis_client.setex(
                    key,
                    TUNNEL_TTL_SECONDS,
                    json.dumps(tunnel_data),
                )
            else:
                self._in_memory_tunnels[key] = json.dumps(tunnel_data)

            logger.info(
                f"Registered tunnel for user {user_id}, conversation {conversation_id}: {tunnel_url}"
            )
            return True
        except Exception as e:
            logger.error(f"Error registering tunnel: {e}")
            return False

    def get_tunnel_url(
        self, user_id: str, conversation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get tunnel URL for a user/conversation.

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID

        Returns:
            Tunnel URL if found, None otherwise
        """
        try:
            # Try conversation-specific first, then user-level
            if conversation_id:
                key = self._get_tunnel_key(user_id, conversation_id)
                tunnel_data = self._get_tunnel_data(key)
                if tunnel_data:
                    return tunnel_data.get("tunnel_url")

            # Fall back to user-level tunnel
            key = self._get_tunnel_key(user_id)
            tunnel_data = self._get_tunnel_data(key)
            if tunnel_data:
                return tunnel_data.get("tunnel_url")

            return None
        except Exception as e:
            logger.error(f"Error getting tunnel URL: {e}")
            return None

    def _get_tunnel_data(self, key: str) -> Optional[Dict]:
        """Get tunnel data from Redis or in-memory storage"""
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                data = self._in_memory_tunnels.get(key)
                if data:
                    return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting tunnel data: {e}")
            return None

    def unregister_tunnel(
        self, user_id: str, conversation_id: Optional[str] = None
    ) -> bool:
        """
        Unregister a tunnel connection.

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID

        Returns:
            True if unregistration succeeded
        """
        try:
            key = self._get_tunnel_key(user_id, conversation_id)
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                self._in_memory_tunnels.pop(key, None)

            logger.info(
                f"Unregistered tunnel for user {user_id}, conversation {conversation_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Error unregistering tunnel: {e}")
            return False

    def is_tunnel_available(
        self, user_id: str, conversation_id: Optional[str] = None
    ) -> bool:
        """
        Check if a tunnel is available for a user/conversation.

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID

        Returns:
            True if tunnel is available
        """
        tunnel_url = self.get_tunnel_url(user_id, conversation_id)
        return tunnel_url is not None


# Global instance
_tunnel_service: Optional[TunnelService] = None


def get_tunnel_service() -> TunnelService:
    """Get or create the global tunnel service instance"""
    global _tunnel_service
    if _tunnel_service is None:
        _tunnel_service = TunnelService()
    return _tunnel_service
