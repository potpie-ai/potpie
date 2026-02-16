"""
Tunnel Service for managing connections to local VS Code extension servers.

This service tracks active tunnel connections, stores tunnel URLs,
and routes requests to the correct local server via tunnel.
"""

import json
import time
import redis
import httpx
import os
from typing import Optional, Dict, List, cast
from app.modules.utils.logger import setup_logger
from app.core.config_provider import ConfigProvider

logger = setup_logger(__name__)

# Redis key prefix for tunnel connections
TUNNEL_KEY_PREFIX = "tunnel_connection"
TUNNEL_TTL_SECONDS = 60 * 60  # 1 hour expiry (refreshed on each registration)

# In-process lookup cache: avoid hitting Redis on every get_tunnel_url when the same
# lookup is repeated (e.g. multiple tools in one message). Entries expire after this many seconds.
TUNNEL_LOOKUP_CACHE_TTL = 60  # seconds

# Health check settings
TUNNEL_HEALTH_TIMEOUT = 5.0  # seconds

# When VSCODE_LOCAL_TUNNEL_SERVER is set, use that URL and skip cloudflared (optional override).
# In dev mode we still use the registered cloudflared tunnel unless this env is set.


def _get_local_tunnel_server_url() -> Optional[str]:
    """Return LocalServer URL for local use when set via env (e.g. VSCODE_LOCAL_TUNNEL_SERVER=http://localhost:3001)."""
    url = (os.getenv("VSCODE_LOCAL_TUNNEL_SERVER") or "").strip()
    if url and (url.startswith("http://") or url.startswith("https://")):
        return url.rstrip("/")
    return None


class TunnelConnectionError(Exception):
    """Raised when tunnel connection fails after retries"""

    def __init__(self, message: str, last_error: Optional[str] = None):
        self.message = message
        self.last_error = last_error
        super().__init__(self.message)


class TunnelService:
    """Service for managing tunnel connections to local servers"""

    def __init__(self):
        """Initialize TunnelService with Redis connection"""
        config = ConfigProvider()
        redis_url = config.get_redis_url()
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test Redis connection
                self.redis_client.ping()
                logger.info(f"[TunnelService] ✅ Connected to Redis at {redis_url}")
            except Exception as e:
                logger.error(
                    f"[TunnelService] ❌ Failed to connect to Redis: {e}", exc_info=True
                )
                logger.warning("[TunnelService] Falling back to in-memory storage")
                self.redis_client = None
                self._in_memory_tunnels: Dict[str, str] = {}
        else:
            logger.warning(
                "[TunnelService] Redis URL not configured, tunnel service will use in-memory storage"
            )
            self.redis_client = None
            self._in_memory_tunnels: Dict[str, str] = {}

        # In-process cache for get_tunnel_url lookups: (cache_key -> (tunnel_url, expiry_ts))
        # Avoids blocking Redis on every tool call when the same user/conv is resolved repeatedly.
        self._lookup_cache: Dict[str, tuple[str, float]] = {}

    def _lookup_cache_key(
        self,
        user_id: str,
        conversation_id: Optional[str],
        repository: Optional[str],
        branch: Optional[str],
    ) -> str:
        """Stable key for the in-process lookup cache."""
        c = conversation_id or ""
        r = (repository or "").strip()
        b = (branch or "").strip()
        return f"{user_id}:{c}:{r}:{b}"

    def _get_conversation_key(self, conversation_id: str) -> str:
        """Generate Redis key for conversation-level tunnel"""
        return f"{TUNNEL_KEY_PREFIX}:conversation:{conversation_id}"

    def _get_user_key(self, user_id: str) -> str:
        """Generate Redis key for user-level tunnel"""
        return f"{TUNNEL_KEY_PREFIX}:user:{user_id}"

    def _get_workspace_key(
        self, user_id: str, repository: str, branch: str
    ) -> str:
        """Generate Redis key for workspace-level tunnel (one per user/repo/branch)."""
        # Normalize for key: repo may be "owner/repo", branch is usually safe
        repo_safe = (repository or "").strip().replace(" ", "_")
        branch_safe = (branch or "").strip().replace(" ", "_")
        return f"{TUNNEL_KEY_PREFIX}:workspace:{user_id}:{repo_safe}:{branch_safe}"

    def _get_tunnel_key(
        self, user_id: str, conversation_id: Optional[str] = None
    ) -> str:
        """
        Generate Redis key for tunnel connection (legacy method for backward compatibility).
        Prefers conversation-level key if conversation_id is provided.
        """
        if conversation_id:
            return self._get_conversation_key(conversation_id)
        return self._get_user_key(user_id)

    def register_tunnel(
        self,
        user_id: str,
        tunnel_url: str,
        conversation_id: Optional[str] = None,
        local_port: Optional[int] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> bool:
        """
        Register a tunnel connection for a user/conversation/workspace.

        Stores tunnel at:
        1. Workspace-level: If repository and branch provided, stored at
           tunnel_connection:workspace:{user_id}:{repository}:{branch} (one active tunnel per workspace).
        2. User-level: Always stored at tunnel_connection:user:{user_id}
        3. Conversation-level: If conversation_id provided, also stored at
           tunnel_connection:conversation:{conversation_id}

        Args:
            user_id: User ID
            tunnel_url: URL of the tunnel (e.g., https://xyz.trycloudflare.com)
            conversation_id: Optional conversation ID for conversation-specific tunnels
            local_port: Optional local port (for direct connection when backend is local)
            repository: Optional repository (e.g. owner/repo) for workspace-scoped tunnel
            branch: Optional branch for workspace-scoped tunnel

        Returns:
            True if registration succeeded
        """
        try:
            current_time = str(int(time.time()))
            # Build tunnel data with last_seen for tracking activity.
            # local_port is stored every time so provision can use it for ingress when not sent in the request.
            tunnel_data = {
                "tunnel_url": tunnel_url,
                "user_id": user_id,
                "conversation_id": conversation_id or "",
                "registered_at": current_time,
                "last_seen": current_time,
            }
            if local_port is not None:
                tunnel_data["local_port"] = local_port
            if repository:
                tunnel_data["repository"] = repository
            if branch:
                tunnel_data["branch"] = branch

            tunnel_data_json = json.dumps(tunnel_data)

            logger.info(
                f"[TunnelService] 📝 Registering tunnel: user_id={user_id}, "
                f"conversation_id={conversation_id}, tunnel_url={tunnel_url}, "
                f"local_port={local_port}, repository={repository}, branch={branch}, "
                f"redis_available={self.redis_client is not None}"
            )

            # Invalidate in-process lookup cache for this user so next get_tunnel_url sees new URL from Redis
            to_drop = [k for k in self._lookup_cache if k.startswith(f"{user_id}:")]
            for k in to_drop:
                del self._lookup_cache[k]

            # Workspace-level: one active tunnel per (user, repository, branch)
            if repository and branch:
                workspace_key = self._get_workspace_key(user_id, repository, branch)
                if not self._store_tunnel_data(workspace_key, tunnel_data_json):
                    return False
                logger.info(
                    f"[TunnelService] ✅ Registered workspace-level tunnel: {workspace_key}"
                )

            # Always register at user-level
            user_key = self._get_user_key(user_id)
            if not self._store_tunnel_data(user_key, tunnel_data_json):
                return False
            logger.info(f"[TunnelService] ✅ Registered user-level tunnel: {user_key}")

            # Also register at conversation-level if conversation_id is provided
            if conversation_id:
                conversation_key = self._get_conversation_key(conversation_id)
                if not self._store_tunnel_data(conversation_key, tunnel_data_json):
                    logger.warning(
                        f"[TunnelService] ⚠️ Failed to register conversation-level tunnel, "
                        f"but user-level succeeded"
                    )
                    # Don't fail the whole registration if conversation-level fails
                else:
                    logger.info(
                        f"[TunnelService] ✅ Registered conversation-level tunnel: {conversation_key}"
                    )

            # type=quick (trycloudflare.com) vs named (e.g. potpie.ai) is chosen by the client; we only store the URL
            tunnel_type = "quick" if "trycloudflare.com" in tunnel_url else "named"
            logger.info(
                f"[TunnelService] Stored tunnel type={tunnel_type} for user={user_id}, conversation={conversation_id}, "
                f"repo={repository}, branch={branch}: {tunnel_url}"
            )
            return True
        except Exception as e:
            logger.error(
                f"[TunnelService] Error registering tunnel: {e}", exc_info=True
            )
            return False

    def _store_tunnel_data(self, key: str, tunnel_data_json: str) -> bool:
        """Store tunnel data in Redis or in-memory storage"""
        try:
            if self.redis_client:
                self.redis_client.setex(
                    key,
                    TUNNEL_TTL_SECONDS,
                    tunnel_data_json,
                )
                # Verify the registration was successful
                verify_data = self.redis_client.get(key)
                if not verify_data:
                    logger.error(
                        f"[TunnelService] ❌ Failed to verify tunnel registration in Redis: {key}"
                    )
                    return False
            else:
                self._in_memory_tunnels[key] = tunnel_data_json
            return True
        except Exception as e:
            logger.error(
                f"[TunnelService] Error storing tunnel data for key {key}: {e}",
                exc_info=True,
            )
            return False

    def get_tunnel_url(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        tunnel_url: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get tunnel URL for a user/conversation/workspace.

        Lookup priority:
        1. tunnel_url parameter (if provided, takes highest priority)
        2. Workspace-level: tunnel_connection:workspace:{user_id}:{repository}:{branch} when repository and branch provided
        3. Conversation-level: tunnel_connection:conversation:{conversation_id}
        4. User-level: tunnel_connection:user:{user_id}

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID
            tunnel_url: Optional tunnel URL from request (takes priority over stored state)
            repository: Optional repository (e.g. owner/repo) to resolve tunnel for current workspace
            branch: Optional branch to resolve tunnel for current workspace

        Returns:
            Tunnel URL if found, None otherwise
        """
        try:
            # When VSCODE_LOCAL_TUNNEL_SERVER is set, use it (skip cloudflared; e.g. local testing)
            env_local_url = _get_local_tunnel_server_url()
            if env_local_url:
                logger.debug(
                    f"[TunnelService] VSCODE_LOCAL_TUNNEL_SERVER set: using {env_local_url}"
                )
                return env_local_url

            # If tunnel_url is provided directly, use it (highest priority)
            if tunnel_url:
                logger.debug(
                    f"[TunnelService] Using tunnel_url from request: {tunnel_url}"
                )
                return tunnel_url

            # In-process cache: avoid Redis round-trip when same lookup repeats (e.g. multiple tools per message)
            cache_key = self._lookup_cache_key(user_id, conversation_id, repository, branch)
            now = time.time()
            if cache_key in self._lookup_cache:
                cached_url, expiry = self._lookup_cache[cache_key]
                if now < expiry:
                    logger.debug(
                        f"[TunnelService] Using cached tunnel URL for {user_id}"
                    )
                    return cached_url
                del self._lookup_cache[cache_key]

            logger.debug(
                f"[TunnelService] Looking up tunnel for user={user_id}, conversation={conversation_id}, "
                f"repository={repository}, branch={branch}"
            )

            # Workspace-level: match tunnel for this repo + branch (per tunnel_plan: route to correct tunnel by repo)
            if repository and branch:
                workspace_key = self._get_workspace_key(user_id, repository, branch)
                logger.info(
                    f"[TunnelService] Checking workspace-level key: {workspace_key}"
                )
                tunnel_data = self._get_tunnel_data(workspace_key)
                if tunnel_data:
                    resolved = tunnel_data.get("tunnel_url")
                    if resolved:
                        self._lookup_cache[cache_key] = (
                            resolved,
                            time.time() + TUNNEL_LOOKUP_CACHE_TTL,
                        )
                    logger.debug(
                        f"[TunnelService] Found workspace-level tunnel for {repository}@{branch}"
                    )
                    return resolved
                else:
                    logger.debug(
                        f"[TunnelService] No workspace-level tunnel for {repository}@{branch}"
                    )

            # Try conversation-level (if conversation_id provided)
            if conversation_id:
                conversation_key = self._get_conversation_key(conversation_id)
                logger.info(
                    f"[TunnelService] Checking conversation-level key: {conversation_key}"
                )
                tunnel_data = self._get_tunnel_data(conversation_key)
                if tunnel_data:
                    resolved = tunnel_data.get("tunnel_url")
                    if resolved:
                        self._lookup_cache[cache_key] = (
                            resolved,
                            time.time() + TUNNEL_LOOKUP_CACHE_TTL,
                        )
                    logger.debug(
                        f"[TunnelService] Found conversation-level tunnel"
                    )
                    return resolved
                else:
                    logger.debug(
                        f"[TunnelService] No conversation-level tunnel found"
                    )

            # Fall back to user-level tunnel
            user_key = self._get_user_key(user_id)
            logger.debug(f"[TunnelService] Checking user-level key: {user_key}")
            tunnel_data = self._get_tunnel_data(user_key)
            if tunnel_data:
                resolved = tunnel_data.get("tunnel_url")
                if resolved:
                    self._lookup_cache[cache_key] = (
                        resolved,
                        time.time() + TUNNEL_LOOKUP_CACHE_TTL,
                    )
                logger.debug(f"[TunnelService] Found user-level tunnel")
                return resolved
            else:
                logger.debug(f"[TunnelService] No user-level tunnel found")

            return None
        except Exception as e:
            logger.error(f"Error getting tunnel URL: {e}")
            return None

    def get_tunnel_info(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Get tunnel metadata for a user/conversation/workspace (URL + optional local_port).

        Lookup order matches get_tunnel_url: workspace (when repo+branch) -> conversation -> user.

        Returns:
            Dict with at least {"tunnel_url": str, ...} or None if not found.
        """
        try:
            if repository and branch:
                key = self._get_workspace_key(user_id, repository, branch)
                tunnel_data = self._get_tunnel_data(key)
                if tunnel_data:
                    return tunnel_data
            if conversation_id:
                key = self._get_tunnel_key(user_id, conversation_id)
                tunnel_data = self._get_tunnel_data(key)
                if tunnel_data:
                    return tunnel_data
            key = self._get_tunnel_key(user_id)
            tunnel_data = self._get_tunnel_data(key)
            return tunnel_data
        except Exception as e:
            logger.error(f"Error getting tunnel info: {e}")
            return None

    def _get_tunnel_data(self, key: str) -> Optional[Dict]:
        """Get tunnel data from Redis or in-memory storage"""
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    logger.debug(
                        f"[TunnelService] ✅ Found tunnel data in Redis for key: {key}"
                    )
                    return json.loads(data)
                else:
                    logger.debug(
                        f"[TunnelService] ❌ No tunnel data in Redis for key: {key}"
                    )
            else:
                data = self._in_memory_tunnels.get(key)
                if data:
                    logger.debug(
                        f"[TunnelService] ✅ Found tunnel data in memory for key: {key}"
                    )
                    return json.loads(data)
                else:
                    logger.debug(
                        f"[TunnelService] ❌ No tunnel data in memory for key: {key}"
                    )
            return None
        except Exception as e:
            logger.error(
                f"[TunnelService] Error getting tunnel data for key {key}: {e}",
                exc_info=True,
            )
            return None

    def unregister_tunnel(
        self, user_id: str, conversation_id: Optional[str] = None
    ) -> bool:
        """
        Unregister a tunnel connection.

        If conversation_id is provided, only removes the conversation-level mapping.
        If conversation_id is not provided, removes the user-level mapping.

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID

        Returns:
            True if unregistration succeeded
        """
        try:
            if conversation_id:
                # Unregister conversation-level
                conversation_key = self._get_conversation_key(conversation_id)
                self._delete_tunnel_key(conversation_key)
                logger.info(
                    f"[TunnelService] Unregistered conversation-level tunnel: {conversation_key}"
                )
            else:
                # Unregister user-level
                user_key = self._get_user_key(user_id)
                self._delete_tunnel_key(user_key)
                logger.info(
                    f"[TunnelService] Unregistered user-level tunnel: {user_key}"
                )

            logger.info(
                f"Unregistered tunnel for user {user_id}, conversation {conversation_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Error unregistering tunnel: {e}")
            return False

    def _delete_tunnel_key(self, key: str) -> None:
        """Delete a tunnel key from Redis or in-memory storage"""
        if self.redis_client:
            self.redis_client.delete(key)
        else:
            self._in_memory_tunnels.pop(key, None)

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

    def list_user_tunnels(self, user_id: str) -> Dict[str, Dict]:
        """
        List all tunnels for a user (for debugging).

        Searches both user-level and conversation-level tunnels
        that belong to the user.

        Args:
            user_id: User ID

        Returns:
            Dictionary mapping tunnel keys to tunnel data
        """
        tunnels = {}
        try:
            if self.redis_client:
                # Search for user-level tunnel
                user_key = self._get_user_key(user_id)
                user_data = self.redis_client.get(user_key)
                if user_data:
                    tunnels[user_key] = json.loads(user_data)

                # Search for conversation-level tunnels belonging to this user
                # We need to scan all conversation keys and filter by user_id
                conversation_pattern = f"{TUNNEL_KEY_PREFIX}:conversation:*"
                # Sync redis client: .keys() returns a list; cast for type checker (avoids Awaitable confusion)
                conversation_keys: List[str] = cast(
                    List[str], self.redis_client.keys(conversation_pattern)
                )
                for key in conversation_keys:
                    data = self.redis_client.get(key)
                    if data:
                        tunnel_data = json.loads(data)
                        if tunnel_data.get("user_id") == user_id:
                            tunnels[key] = tunnel_data

                logger.info(
                    f"[TunnelService] Found {len(tunnels)} tunnel(s) for user {user_id}"
                )
            else:
                # Check in-memory storage - user-level
                user_key = self._get_user_key(user_id)
                if user_key in self._in_memory_tunnels:
                    tunnels[user_key] = json.loads(self._in_memory_tunnels[user_key])

                # Check conversation-level tunnels
                conversation_prefix = f"{TUNNEL_KEY_PREFIX}:conversation:"
                for key, value in self._in_memory_tunnels.items():
                    if key.startswith(conversation_prefix):
                        tunnel_data = json.loads(value)
                        if tunnel_data.get("user_id") == user_id:
                            tunnels[key] = tunnel_data
        except Exception as e:
            logger.error(
                f"[TunnelService] Error listing tunnels for user {user_id}: {e}",
                exc_info=True,
            )
        return tunnels

    def verify_tunnel_health(
        self, tunnel_url: str, timeout: float = TUNNEL_HEALTH_TIMEOUT
    ) -> bool:
        """
        Verify tunnel is reachable by calling health endpoint.

        Args:
            tunnel_url: The tunnel URL to check
            timeout: Timeout in seconds (default: 5.0)

        Returns:
            True if tunnel is healthy, False otherwise
        """
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{tunnel_url}/health")
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Check tunnel status from response if available
                        tunnel_info = data.get("tunnel", {})
                        if tunnel_info.get("healthy") is False:
                            logger.warning(
                                f"[TunnelService] Tunnel reports unhealthy: {tunnel_info}"
                            )
                            return False
                        logger.debug(
                            f"[TunnelService] ✅ Tunnel health check passed: {tunnel_url}"
                        )
                        return True
                    except json.JSONDecodeError:
                        # Response is not JSON but status is 200, consider healthy
                        return True
                else:
                    logger.warning(
                        f"[TunnelService] ❌ Health check failed ({response.status_code}): {tunnel_url}"
                    )
                    return False
        except httpx.TimeoutException:
            logger.warning(f"[TunnelService] ❌ Health check timeout: {tunnel_url}")
            return False
        except httpx.ConnectError as e:
            logger.warning(f"[TunnelService] ❌ Health check connection error: {e}")
            return False
        except Exception as e:
            logger.warning(f"[TunnelService] ❌ Health check failed: {e}")
            return False

    def execute_tool_call(
        self,
        tunnel_url: str,
        endpoint: str,
        payload: Dict,
        method: str = "POST",
        max_retries: int = 2,
        timeout: float = 30.0,
        verify_health_on_retry: bool = True,
    ) -> Dict:
        """
        Execute a tool call through the tunnel with retry logic.

        Args:
            tunnel_url: The tunnel URL
            endpoint: API endpoint (e.g., "/api/files/write")
            payload: Request payload (JSON body)
            method: HTTP method (default: POST)
            max_retries: Maximum number of retries (default: 2)
            timeout: Request timeout in seconds (default: 30.0)
            verify_health_on_retry: Whether to verify health before retrying (default: True)

        Returns:
            Response JSON dict

        Raises:
            TunnelConnectionError: If all retries fail
        """
        last_error = None
        url = f"{tunnel_url}{endpoint}"

        for attempt in range(max_retries + 1):
            try:
                # Verify tunnel is healthy before retrying (skip on first attempt for speed)
                if attempt > 0 and verify_health_on_retry:
                    is_healthy = self.verify_tunnel_health(tunnel_url)
                    if not is_healthy:
                        logger.warning(
                            f"[TunnelService] Tunnel unhealthy on attempt {attempt + 1}, "
                            f"waiting before retry..."
                        )
                        time.sleep(2**attempt)  # Exponential backoff
                        continue

                with httpx.Client(timeout=timeout) as client:
                    if method.upper() == "GET":
                        response = client.get(url, params=payload)
                    else:
                        response = client.post(
                            url,
                            json=payload,
                            headers={"Content-Type": "application/json"},
                        )
                    response.raise_for_status()
                    return response.json()

            except httpx.TimeoutException as e:
                last_error = f"Timeout calling {endpoint}: {e}"
                logger.warning(
                    f"[TunnelService] Tool call timeout (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP error {e.response.status_code}: {e}"
                logger.warning(
                    f"[TunnelService] Tool call HTTP error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                # Don't retry client errors (4xx)
                if e.response.status_code < 500:
                    raise TunnelConnectionError(
                        f"Tool call failed with client error: {last_error}",
                        last_error=last_error,
                    )

            except httpx.ConnectError as e:
                last_error = f"Connection failed: {e}"
                logger.warning(
                    f"[TunnelService] Tool call connection failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.warning(
                    f"[TunnelService] Tool call unexpected error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )

            # Wait before retry with exponential backoff
            if attempt < max_retries:
                wait_time = 2**attempt
                logger.info(f"[TunnelService] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        # All retries exhausted
        raise TunnelConnectionError(
            f"Failed to execute tool call after {max_retries + 1} attempts. "
            f"The VS Code extension tunnel may be disconnected. "
            f"Last error: {last_error}",
            last_error=last_error,
        )

    def execute_tool_call_with_fallback(
        self,
        user_id: str,
        conversation_id: Optional[str],
        endpoint: str,
        payload: Dict,
        tunnel_url: Optional[str] = None,
        method: str = "POST",
        max_retries: int = 2,
        timeout: float = 30.0,
    ) -> Dict:
        """
        Execute a tool call with automatic tunnel URL resolution and fallback.

        Args:
            user_id: User ID for tunnel lookup
            conversation_id: Conversation ID for tunnel lookup
            endpoint: API endpoint (e.g., "/api/files/write")
            payload: Request payload (JSON body)
            tunnel_url: Optional tunnel URL from request (takes priority)
            method: HTTP method (default: POST)
            max_retries: Maximum number of retries (default: 2)
            timeout: Request timeout in seconds (default: 30.0)

        Returns:
            Response JSON dict

        Raises:
            TunnelConnectionError: If no tunnel available or all retries fail
        """
        # Get tunnel URL with priority resolution
        resolved_url = self.get_tunnel_url(user_id, conversation_id, tunnel_url)

        if not resolved_url:
            raise TunnelConnectionError(
                f"No tunnel available for user {user_id}, conversation {conversation_id}. "
                f"Please ensure the VS Code extension is running and connected.",
                last_error="No tunnel URL found",
            )

        try:
            return self.execute_tool_call(
                tunnel_url=resolved_url,
                endpoint=endpoint,
                payload=payload,
                method=method,
                max_retries=max_retries,
                timeout=timeout,
            )
        except TunnelConnectionError:
            # If conversation-level failed and we have a user-level fallback, try it
            if conversation_id and tunnel_url != resolved_url:
                user_level_url = self.get_tunnel_url(user_id, None)
                if user_level_url and user_level_url != resolved_url:
                    logger.info(
                        f"[TunnelService] Retrying with user-level tunnel: {user_level_url}"
                    )
                    # Invalidate the failed conversation-level tunnel
                    self.unregister_tunnel(user_id, conversation_id)
                    return self.execute_tool_call(
                        tunnel_url=user_level_url,
                        endpoint=endpoint,
                        payload=payload,
                        method=method,
                        max_retries=max_retries,
                        timeout=timeout,
                    )
            raise

    @staticmethod
    def format_tunnel_error_response(error: "TunnelConnectionError") -> Dict:
        """
        Format a TunnelConnectionError into a response dict for the agent.

        Args:
            error: The TunnelConnectionError

        Returns:
            Dict with error information for the agent
        """
        return {
            "success": False,
            "error": "TUNNEL_DISCONNECTED",
            "message": str(error),
            "last_error": error.last_error,
            "suggestion": (
                "The local VS Code extension tunnel appears to be disconnected. "
                "Please check VS Code and ensure the Potpie extension is running. "
                "You can reload the VS Code window (Cmd/Ctrl+Shift+P → 'Reload Window') "
                "and try again."
            ),
        }


# Global instance
_tunnel_service: Optional[TunnelService] = None


def get_tunnel_service() -> TunnelService:
    """Get or create the global tunnel service instance"""
    global _tunnel_service
    if _tunnel_service is None:
        _tunnel_service = TunnelService()
    return _tunnel_service
