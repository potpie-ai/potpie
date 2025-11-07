"""
Repository Manager Implementation

Manages local copies of repositories stored in .repos folder.
Tracks repository metadata in Redis for efficient querying and eviction.
"""

import os
import json
import logging
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

import redis

from app.modules.repo_manager.repo_manager_interface import IRepoManager
from app.core.config_provider import ConfigProvider

logger = logging.getLogger(__name__)


class RepoManager(IRepoManager):
    """
    Implementation of IRepoManager using local filesystem and Redis.

    Repositories are stored in .repos folder and metadata is tracked in Redis.
    """

    def __init__(self, repos_base_path: Optional[str] = None):
        """
        Initialize the repository manager.

        Args:
            repos_base_path: Base path for storing repositories. Defaults to .repos in project root.
        """
        self.config = ConfigProvider()
        self.redis_client = redis.from_url(self.config.get_redis_url())

        # Determine repos base path
        if repos_base_path:
            self.repos_base_path = Path(repos_base_path).resolve()
        else:
            # Default to .repos in project root (parent of app directory)
            project_root = Path(__file__).parent.parent.parent.parent
            self.repos_base_path = project_root / ".repos"

        # Ensure repos directory exists
        self.repos_base_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"RepoManager initialized with base path: {self.repos_base_path}")

    def _get_repo_key(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> str:
        """
        Generate Redis key for a repository.

        Args:
            repo_name: Repository name
            branch: Branch name (optional)
            commit_id: Commit SHA (optional)

        Returns:
            Redis key string
        """
        parts = [repo_name]
        if branch:
            parts.append(f"branch:{branch}")
        if commit_id:
            parts.append(f"commit:{commit_id}")
        return ":".join(parts)

    def _get_redis_key(self, repo_key: str) -> str:
        """Get full Redis key with prefix."""
        return f"repo:info:{repo_key}"

    def _get_index_key(self, index_type: str, value: str = "") -> str:
        """Get Redis key for an index."""
        if value:
            return f"repo:index:{index_type}:{value}"
        return f"repo:index:{index_type}"

    def _get_repo_local_path(self, repo_name: str) -> Path:
        """
        Get local filesystem path for a repository.

        Uses hierarchical structure: .repos/owner/repo
        """
        # Use the full repo name as-is for hierarchical structure
        return self.repos_base_path / repo_name

    def _serialize_datetime(self, dt: datetime) -> str:
        """Serialize datetime to ISO format string."""
        return dt.isoformat()

    def _deserialize_datetime(self, dt_str: str) -> datetime:
        """Deserialize ISO format string to datetime."""
        return datetime.fromisoformat(dt_str)

    def is_repo_available(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Check if a repository is available locally."""
        repo_key = self._get_repo_key(repo_name, branch, commit_id)
        redis_key = self._get_redis_key(repo_key)

        logger.debug(
            f"[REPO_MANAGER] Checking availability for repo_key: {repo_key}, "
            f"redis_key: {redis_key}"
        )

        # Check if metadata exists in Redis
        if not self.redis_client.exists(redis_key):
            logger.debug(f"[REPO_MANAGER] Redis key {redis_key} does not exist")
            return False

        # Check if local path exists
        repo_info = self._get_repo_info_from_redis(redis_key)
        if not repo_info:
            logger.debug(f"[REPO_MANAGER] No repo info found in Redis for {redis_key}")
            return False

        local_path = repo_info.get("local_path")
        if not local_path or not os.path.exists(local_path):
            logger.debug(
                f"[REPO_MANAGER] Local path {local_path} does not exist for {redis_key}"
            )
            return False

        # If user_id specified, check if it matches
        if user_id and repo_info.get("user_id") != user_id:
            logger.debug(
                f"[REPO_MANAGER] User ID mismatch for {redis_key} "
                f"(expected: {user_id}, found: {repo_info.get('user_id')})"
            )
            return False

        logger.debug(f"[REPO_MANAGER] Repo is available: {repo_key} at {local_path}")
        return True

    def register_repo(
        self,
        repo_name: str,
        local_path: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a repository that has been downloaded/parsed."""
        # Validate local path exists
        if not os.path.exists(local_path):
            raise ValueError(f"Local path does not exist: {local_path}")

        repo_key = self._get_repo_key(repo_name, branch, commit_id)
        redis_key = self._get_redis_key(repo_key)

        now = datetime.utcnow()

        # Prepare repo info
        repo_info = {
            "repo_name": repo_name,
            "local_path": local_path,
            "branch": branch,
            "commit_id": commit_id,
            "user_id": user_id,
            "registered_at": self._serialize_datetime(now),
            "last_accessed": self._serialize_datetime(now),
            "metadata": json.dumps(metadata) if metadata else None,
        }

        # Store in Redis as hash
        pipe = self.redis_client.pipeline()
        pipe.hset(redis_key, mapping={k: (v or "") for k, v in repo_info.items()})

        # Add to indexes
        pipe.sadd(self._get_index_key("all"), repo_key)
        pipe.sadd(self._get_index_key("by_name", repo_name), repo_key)
        if user_id:
            pipe.sadd(self._get_index_key("by_user", user_id), repo_key)

        pipe.execute()

        logger.info(f"Registered repo: {repo_key} at {local_path}")
        return repo_key

    def get_repo_path(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get the local filesystem path for a repository."""
        repo_key = self._get_repo_key(repo_name, branch, commit_id)
        redis_key = self._get_redis_key(repo_key)

        logger.debug(
            f"[REPO_MANAGER] Getting repo path for repo_key: {repo_key}, "
            f"redis_key: {redis_key}"
        )

        repo_info = self._get_repo_info_from_redis(redis_key)
        if not repo_info:
            logger.debug(f"[REPO_MANAGER] No repo info found in Redis for {redis_key}")
            return None

        # Check user_id if specified
        if user_id and repo_info.get("user_id") != user_id:
            logger.debug(
                f"[REPO_MANAGER] User ID mismatch for {redis_key} "
                f"(expected: {user_id}, found: {repo_info.get('user_id')})"
            )
            return None

        local_path = repo_info.get("local_path")
        if local_path and os.path.exists(local_path):
            logger.debug(f"[REPO_MANAGER] Found repo path for {repo_key}: {local_path}")
            return local_path

        logger.debug(
            f"[REPO_MANAGER] Local path {local_path} does not exist for {repo_key}"
        )
        return None

    def update_last_accessed(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Update the last accessed timestamp for a repository."""
        repo_key = self._get_repo_key(repo_name, branch, commit_id)
        redis_key = self._get_redis_key(repo_key)

        if not self.redis_client.exists(redis_key):
            logger.debug(f"Repo not found for update: {repo_key}")
            return

        # Check user_id if specified
        if user_id:
            repo_info = self._get_repo_info_from_redis(redis_key)
            if repo_info and repo_info.get("user_id") != user_id:
                return

        now = datetime.utcnow()
        self.redis_client.hset(
            redis_key, "last_accessed", self._serialize_datetime(now)
        )

    def get_repo_info(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get information about a registered repository."""
        repo_key = self._get_repo_key(repo_name, branch, commit_id)
        redis_key = self._get_redis_key(repo_key)

        repo_info = self._get_repo_info_from_redis(redis_key)
        if not repo_info:
            return None

        # Check user_id if specified
        if user_id and repo_info.get("user_id") != user_id:
            return None

        # Deserialize fields
        result = {
            "repo_key": repo_key,
            "repo_name": repo_info.get("repo_name"),
            "local_path": repo_info.get("local_path"),
            "branch": repo_info.get("branch") or None,
            "commit_id": repo_info.get("commit_id") or None,
            "user_id": repo_info.get("user_id") or None,
            "registered_at": self._deserialize_datetime(
                repo_info.get("registered_at", datetime.utcnow().isoformat())
            ),
            "last_accessed": self._deserialize_datetime(
                repo_info.get("last_accessed", datetime.utcnow().isoformat())
            ),
        }

        # Parse metadata
        metadata_str = repo_info.get("metadata")
        if metadata_str:
            try:
                result["metadata"] = json.loads(metadata_str)
            except json.JSONDecodeError:
                result["metadata"] = {}
        else:
            result["metadata"] = {}

        return result

    def list_available_repos(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List all available repositories."""
        # Get repo keys from appropriate index
        if user_id:
            index_key = self._get_index_key("by_user", user_id)
        else:
            index_key = self._get_index_key("all")

        repo_keys_set = self.redis_client.smembers(index_key)
        repo_keys = list(repo_keys_set) if repo_keys_set else []  # type: ignore

        # Decode bytes to strings
        repo_keys = [k.decode() if isinstance(k, bytes) else k for k in repo_keys]

        # Get repo info for each key
        repos = []
        for repo_key in repo_keys:
            redis_key = self._get_redis_key(repo_key)
            repo_info = self._get_repo_info_from_redis(redis_key)

            if not repo_info:
                continue

            # Check if local path still exists
            local_path = repo_info.get("local_path")
            if not local_path or not os.path.exists(local_path):
                continue

            # Deserialize and format
            try:
                info = {
                    "repo_key": repo_key,
                    "repo_name": repo_info.get("repo_name"),
                    "local_path": local_path,
                    "branch": repo_info.get("branch") or None,
                    "commit_id": repo_info.get("commit_id") or None,
                    "user_id": repo_info.get("user_id") or None,
                    "registered_at": self._deserialize_datetime(
                        repo_info.get("registered_at", datetime.utcnow().isoformat())
                    ),
                    "last_accessed": self._deserialize_datetime(
                        repo_info.get("last_accessed", datetime.utcnow().isoformat())
                    ),
                }

                metadata_str = repo_info.get("metadata")
                if metadata_str:
                    try:
                        info["metadata"] = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        info["metadata"] = {}
                else:
                    info["metadata"] = {}

                repos.append(info)
            except Exception as e:
                logger.warning(f"Error processing repo {repo_key}: {e}")
                continue

        # Sort by last_accessed (most recent first)
        repos.sort(key=lambda x: x["last_accessed"], reverse=True)

        # Apply limit
        if limit:
            repos = repos[:limit]

        return repos

    def evict_repo(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Evict a repository from local storage."""
        repo_key = self._get_repo_key(repo_name, branch, commit_id)
        redis_key = self._get_redis_key(repo_key)

        if not self.redis_client.exists(redis_key):
            return False

        # Get repo info before deletion
        repo_info = self._get_repo_info_from_redis(redis_key)
        if not repo_info:
            return False

        # Check user_id if specified
        if user_id and repo_info.get("user_id") != user_id:
            return False

        local_path = repo_info.get("local_path")
        user_id_from_info = repo_info.get("user_id")

        # Remove from Redis
        pipe = self.redis_client.pipeline()
        pipe.delete(redis_key)
        pipe.srem(self._get_index_key("all"), repo_key)
        pipe.srem(self._get_index_key("by_name", repo_name), repo_key)
        if user_id_from_info:
            pipe.srem(self._get_index_key("by_user", user_id_from_info), repo_key)
        pipe.execute()

        # Delete local filesystem copy
        if local_path and os.path.exists(local_path):
            try:
                if os.path.isdir(local_path):
                    shutil.rmtree(local_path)
                else:
                    os.remove(local_path)
                logger.info(f"Deleted local copy: {local_path}")
            except Exception as e:
                logger.error(f"Error deleting local copy {local_path}: {e}")

        logger.info(f"Evicted repo: {repo_key}")
        return True

    def evict_stale_repos(
        self,
        max_age_days: int,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """Evict repositories that haven't been accessed in a while."""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        evicted = []

        # Get all repos (filtered by user if specified)
        repos = self.list_available_repos(user_id=user_id)

        for repo_info in repos:
            last_accessed = repo_info.get("last_accessed")
            if not last_accessed:
                continue

            if last_accessed < cutoff_date:
                repo_name = repo_info.get("repo_name")
                branch = repo_info.get("branch")
                commit_id = repo_info.get("commit_id")
                repo_user_id = repo_info.get("user_id")

                if repo_name and self.evict_repo(
                    repo_name,
                    branch=branch,
                    commit_id=commit_id,
                    user_id=repo_user_id,
                ):
                    evicted.append(repo_info.get("repo_key"))

        logger.info(
            f"Evicted {len(evicted)} stale repos (older than {max_age_days} days)"
        )
        return evicted

    def get_repo_size(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[int]:
        """Get the size of a repository in bytes."""
        local_path = self.get_repo_path(repo_name, branch, commit_id, user_id)
        if not local_path:
            return None

        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(local_path):
                # Skip .git directory
                if ".git" in dirpath:
                    continue

                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        continue

            return total_size
        except Exception as e:
            logger.warning(f"Error calculating repo size for {local_path}: {e}")
            return None

    def _get_repo_info_from_redis(self, redis_key: str) -> Optional[Dict[str, str]]:
        """Get repository info from Redis hash."""
        if not self.redis_client.exists(redis_key):
            return None

        info = self.redis_client.hgetall(redis_key)
        if not info:
            return None

        # Decode bytes to strings
        return {
            k.decode() if isinstance(k, bytes) else k: (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in info.items()
        }  # type: ignore
