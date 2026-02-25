"""
Tunnel Service for managing connections to local VS Code extension servers.

Socket.IO mode: extension connects via WebSocket, registers workspace_id.
Tool calls are executed over Socket.IO (WorkspaceSocketService). Workspace
isolation by workspace_id = sha256(user_id:repo_url)[:16].
"""

import hashlib
import json
import redis
import os
from typing import Optional, Dict, List
from app.modules.utils.logger import setup_logger
from app.core.config_provider import ConfigProvider

logger = setup_logger(__name__)

# Redis key prefix for tunnel connections
TUNNEL_KEY_PREFIX = "tunnel_connection"
TUNNEL_TTL_SECONDS = 60 * 60  # 1 hour expiry (refreshed on each registration)

# Workspace tunnel record (metadata: user_id, repo_url, status), stored in Redis
WORKSPACE_TUNNEL_RECORD_PREFIX = "workspace_tunnel:"
WORKSPACE_TUNNEL_RECORD_TTL = 24 * 60 * 60  # 24h TTL so stale records expire

# In-process lookup cache: avoid hitting Redis on every get_tunnel_url when the same
# lookup is repeated (e.g. multiple tools in one message). Entries expire after this many seconds.
TUNNEL_LOOKUP_CACHE_TTL = 60  # seconds

# Pseudo-URL prefix for socket-backed "tunnel" (get_tunnel_url returns socket://{workspace_id} when socket is online)
SOCKET_TUNNEL_PREFIX = "socket://"


def _is_local_tunnel_url(url: str) -> bool:
    """True if URL is a direct local address (localhost/127.0.0.1); only used for tunnel bypass in development."""
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    return u.startswith("http://localhost") or u.startswith("http://127.0.0.1")


def _get_local_tunnel_server_url() -> Optional[str]:
    """Return LocalServer URL for local dev when set via env (VSCODE_LOCAL_TUNNEL_SERVER)."""
    env_name = (os.getenv("ENV") or "").strip().lower()
    if env_name not in ("development", "dev", "local"):
        return None
    url = (os.getenv("VSCODE_LOCAL_TUNNEL_SERVER") or "").strip()
    if url and (url.startswith("http://") or url.startswith("https://")):
        return url.rstrip("/")
    return None


def normalise_repo_url(repository: str) -> str:
    """
    Normalise repo identifier to a single form for workspace_id hashing.
    Backend and extension must use the same rule: lowercase, strip protocol, strip .git, rstrip /.
    Accepts owner/repo or full URL (https://github.com/owner/repo or github.com/owner/repo).
    """
    if not repository or not isinstance(repository, str):
        return ""
    s = repository.strip().lower().rstrip("/")
    # Strip protocol
    for prefix in ("https://", "http://", "git@"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    # git@github.com:owner/repo -> github.com/owner/repo
    if ":" in s and "@" in s:
        s = s.replace(":", "/", 1)
    if s.endswith(".git"):
        s = s[: -len(".git")]
    return s


def compute_workspace_id(user_id: str, repo_url: str) -> str:
    """
    workspace_id = sha256(user_id + ':' + normalise(repo_url)).hexdigest()[:16].
    Same formula on backend and extension.
    """
    normalised = normalise_repo_url(repo_url) if repo_url else ""
    payload = f"{user_id}:{normalised}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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
                self._in_memory_tunnels = {}
                self._in_memory_workspace_tunnel_records = {}
            except Exception as e:
                logger.error(
                    f"[TunnelService] ❌ Failed to connect to Redis: {e}", exc_info=True
                )
                logger.warning("[TunnelService] Falling back to in-memory storage")
                self.redis_client = None
                self._in_memory_tunnels = {}
                self._in_memory_workspace_tunnel_records = {}
        else:
            logger.warning(
                "[TunnelService] Redis URL not configured, tunnel service will use in-memory storage"
            )
            self.redis_client = None
            self._in_memory_tunnels: Dict[str, str] = {}
            self._in_memory_workspace_tunnel_records: Dict[str, str] = {}

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

    def _workspace_tunnel_record_key(self, workspace_id: str) -> str:
        return f"{WORKSPACE_TUNNEL_RECORD_PREFIX}{workspace_id}"

    def get_workspace_tunnel_record(self, workspace_id: str) -> Optional[Dict]:
        """
        Get workspace record (user_id, repo_url, status). Returns None if not found.
        """
        try:
            key = self._workspace_tunnel_record_key(workspace_id)
            raw = None
            if self.redis_client:
                raw = self.redis_client.get(key)
            else:
                raw = self._in_memory_workspace_tunnel_records.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.error(
                f"[TunnelService] Error getting workspace tunnel record: {e}",
                exc_info=True,
            )
            return None

    def set_workspace_tunnel_record(
        self,
        workspace_id: str,
        user_id: str,
        repo_url: str,
        status: str = "active",
    ) -> bool:
        """Store workspace record in Redis (or in-memory fallback) with 24h TTL."""
        try:
            key = self._workspace_tunnel_record_key(workspace_id)
            value = json.dumps({
                "user_id": user_id,
                "repo_url": repo_url,
                "status": status,
            })
            if self.redis_client:
                self.redis_client.setex(key, WORKSPACE_TUNNEL_RECORD_TTL, value)
            else:
                self._in_memory_workspace_tunnel_records[key] = value
            logger.debug(
                f"[TunnelService] Stored workspace tunnel record: workspace_id={workspace_id}"
            )
            return True
        except Exception as e:
            logger.error(
                f"[TunnelService] Error setting workspace tunnel record: {e}",
                exc_info=True,
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
        Get tunnel identifier for a user/conversation/workspace (Socket.IO path).

        Resolves workspace_id from user_id + repository (or conversation stored data).
        Returns socket://{workspace_id} when a socket is connected for that workspace, else None.

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID (used to look up repository from stored data)
            tunnel_url: Ignored (kept for API compat)
            repository: Optional repository (e.g. owner/repo) for workspace resolution
            branch: Optional branch (unused, kept for API compat)

        Returns:
            socket://{workspace_id} if workspace socket is online, None otherwise
        """
        try:
            workspace_id = self.get_workspace_id(user_id, conversation_id, repository, branch)
            if not workspace_id:
                return None
            if not self.get_workspace_socket_status(workspace_id):
                return None
            return SOCKET_TUNNEL_PREFIX + workspace_id
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

        Lookup order matches get_tunnel_url: workspace (when repo+branch) -> conversation (no user-level).

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
                key = self._get_conversation_key(conversation_id)
                tunnel_data = self._get_tunnel_data(key)
                if tunnel_data:
                    return tunnel_data
            return None
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
        Unregister a tunnel connection (workspace/conversation only; no user-level).

        If conversation_id is provided, removes the conversation-level mapping.
        If conversation_id is not provided, no-op (user-level tunnels are not used).

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID

        Returns:
            True if unregistration succeeded or no-op
        """
        try:
            if conversation_id:
                conversation_key = self._get_conversation_key(conversation_id)
                self._delete_tunnel_key(conversation_key)
                logger.info(
                    f"[TunnelService] Unregistered conversation-level tunnel: {conversation_key}"
                )
            else:
                logger.debug("[TunnelService] Unregister with no conversation_id: no-op (workspace-only mode)")

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

    def get_workspace_id(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve workspace_id for socket routing.
        When repository is provided, returns compute_workspace_id(user_id, normalise_repo_url(repository)).
        When only conversation_id is provided, looks up stored tunnel data for repository and computes workspace_id.
        """
        if repository:
            repo_normalised = normalise_repo_url(repository)
            if repo_normalised:
                return compute_workspace_id(user_id, repo_normalised)
        if conversation_id:
            key = self._get_conversation_key(conversation_id)
            tunnel_data = self._get_tunnel_data(key)
            if tunnel_data and tunnel_data.get("repository"):
                repo_normalised = normalise_repo_url(tunnel_data["repository"])
                if repo_normalised:
                    return compute_workspace_id(user_id, repo_normalised)
        return None

    def get_workspace_socket_status(self, workspace_id: str) -> bool:
        """Check if a socket is connected for this workspace_id."""
        from app.modules.tunnel.socket_service import get_socket_service
        return get_socket_service().is_workspace_online(workspace_id)

    def is_tunnel_available(
        self, user_id: str, conversation_id: Optional[str] = None,
        repository: Optional[str] = None, branch: Optional[str] = None,
    ) -> bool:
        """
        Check if a tunnel/socket is available for a user/conversation/workspace.

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID
            repository: Optional repository for workspace resolution
            branch: Optional branch (unused but kept for API compat)

        Returns:
            True if workspace socket is online
        """
        workspace_id = self.get_workspace_id(user_id, conversation_id, repository, branch)
        if not workspace_id:
            return False
        return self.get_workspace_socket_status(workspace_id)

    def list_user_tunnels(self, user_id: str) -> Dict[str, Dict]:
        """
        List all tunnels for a user (for debugging).

        Searches workspace and conversation-level tunnels only (no user-level).

        Args:
            user_id: User ID

        Returns:
            Dictionary mapping tunnel keys to tunnel data
        """
        tunnels = {}
        try:
            if self.redis_client:
                # Search for conversation-level tunnels belonging to this user.
                # Use SCAN instead of KEYS to avoid blocking Redis on large key sets.
                conversation_pattern = f"{TUNNEL_KEY_PREFIX}:conversation:*"
                conversation_keys: List[str] = [
                    k.decode() if isinstance(k, bytes) else k
                    for k in self.redis_client.scan_iter(match=conversation_pattern)
                ]
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
                # Check in-memory storage - conversation-level only
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

    def execute_tool_call(
        self,
        workspace_id: str,
        endpoint: str,
        payload: Dict,
        method: str = "POST",
        max_retries: int = 2,
        timeout: float = 30.0,
        verify_health_on_retry: bool = True,
    ) -> Dict:
        """
        Execute a tool call via Socket.IO for the given workspace_id.
        Delegates to WorkspaceSocketService.

        Args:
            workspace_id: 16-char hex workspace id (from get_workspace_id)
            endpoint: API endpoint (e.g., "/api/files/read-batch")
            payload: Request payload (JSON body)
            method: Ignored (kept for API compat; socket always sends as POST-style)
            max_retries: Ignored (kept for API compat)
            timeout: Request timeout in seconds
            verify_health_on_retry: Ignored (kept for API compat)

        Returns:
            Response dict (from tool_response event)

        Raises:
            TunnelConnectionError: If workspace offline or timeout
        """
        from app.modules.tunnel.socket_service import get_socket_service
        return get_socket_service().execute_tool_call_sync(
            workspace_id=workspace_id,
            endpoint=endpoint,
            payload=payload,
            timeout=timeout,
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
        repository: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Dict:
        """
        Execute a tool call with workspace resolution (Socket.IO).
        Resolves workspace_id from user_id/conversation_id/repository and delegates to socket service.

        Args:
            user_id: User ID
            conversation_id: Optional conversation ID
            endpoint: API endpoint (e.g., "/api/files/write")
            payload: Request payload (JSON body)
            tunnel_url: Optional; if socket://{workspace_id}, workspace_id is extracted
            method: Ignored (kept for API compat)
            max_retries: Ignored (kept for API compat)
            timeout: Request timeout in seconds
            repository: Optional repository for workspace resolution
            branch: Optional branch (unused, kept for API compat)

        Returns:
            Response JSON dict

        Raises:
            TunnelConnectionError: If no workspace socket available or call fails
        """
        workspace_id = None
        if tunnel_url and tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
            workspace_id = tunnel_url[len(SOCKET_TUNNEL_PREFIX) :].strip()
        if not workspace_id:
            workspace_id = self.get_workspace_id(user_id, conversation_id, repository, branch)
        if not workspace_id:
            raise TunnelConnectionError(
                f"No workspace socket for user {user_id}, conversation {conversation_id}. "
                "Ensure the VS Code extension is running and registered (repository in context).",
                last_error="No workspace_id",
            )
        return self.execute_tool_call(
            workspace_id=workspace_id,
            endpoint=endpoint,
            payload=payload,
            timeout=timeout,
        )

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
