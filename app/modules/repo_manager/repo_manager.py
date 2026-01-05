"""
Repository Manager Implementation

Manages local copies of repositories stored in `.repos` (or path specified by
REPOS_BASE_PATH environment variable).
Tracks repository metadata using the filesystem instead of Redis.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.modules.repo_manager.repo_manager_interface import IRepoManager
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class RepoManager(IRepoManager):
    """
    Implementation of IRepoManager backed entirely by the local filesystem.

    Repository checkouts live under `.repos/<owner>/<repo>` and metadata about
    worktrees/refs is persisted alongside them inside `.repos/.meta/...`.
    """

    _METADATA_ROOT_NAME = ".meta"
    _METADATA_EXTENSION = ".json"

    def __init__(self, repos_base_path: Optional[str] = None):
        """
        Initialize the repository manager.

        Args:
            repos_base_path: Base path for storing repositories. If not provided,
                checks REPOS_BASE_PATH environment variable. Defaults to `.repos`
                at the project root (parent of the `app` directory) if neither is set.
        """
        if repos_base_path:
            self.repos_base_path = Path(repos_base_path).resolve()
        else:
            # Check environment variable
            env_path = os.getenv("REPOS_BASE_PATH")
            if env_path:
                self.repos_base_path = Path(env_path).resolve()
            else:
                # Default to .repos at project root
                project_root = Path(__file__).parent.parent.parent.parent
                self.repos_base_path = project_root / ".repos"

        self.metadata_base_path = self.repos_base_path / self._METADATA_ROOT_NAME

        self.repos_base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_base_path.mkdir(parents=True, exist_ok=True)

        # Volume limit configuration (in bytes)
        # Default: 100GB if not specified
        volume_limit_str = os.getenv("REPOS_VOLUME_LIMIT_BYTES")
        if volume_limit_str:
            try:
                self.volume_limit_bytes = int(volume_limit_str)
            except ValueError:
                logger.warning(
                    f"Invalid REPOS_VOLUME_LIMIT_BYTES value '{volume_limit_str}', using default 100GB"
                )
                self.volume_limit_bytes = 100 * 1024 * 1024 * 1024  # 100GB
        else:
            self.volume_limit_bytes = 100 * 1024 * 1024 * 1024  # 100GB default

        logger.info(
            "RepoManager initialized with base path %s, metadata path %s, volume limit %d bytes (%.2f GB)",
            self.repos_base_path,
            self.metadata_base_path,
            self.volume_limit_bytes,
            self.volume_limit_bytes / (1024**3),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize_for_filename(value: str) -> str:
        """Convert arbitrary text into a filesystem-safe token."""
        return "".join(
            c if c.isalnum() or c in ("-", "_", ".", "=") else "_" for c in value
        )

    def _metadata_dir(self, repo_name: str) -> Path:
        """Return the metadata directory for a given repository."""
        # Prevent path traversal
        if ".." in repo_name or Path(repo_name).is_absolute():
            raise ValueError(f"Invalid repo_name: {repo_name}")
        return self.metadata_base_path / Path(repo_name)

    def _metadata_filename(
        self,
        branch: Optional[str],
        commit_id: Optional[str],
    ) -> str:
        """Build a deterministic filename for the metadata entry."""
        parts: List[str] = []
        if branch:
            parts.append(f"branch={branch}")
        if commit_id:
            parts.append(f"commit={commit_id}")
        if not parts:
            parts.append("default")
        filename = "__".join(self._sanitize_for_filename(part) for part in parts)
        return f"{filename}{self._METADATA_EXTENSION}"

    def _metadata_path(
        self,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
    ) -> Path:
        return self._metadata_dir(repo_name) / self._metadata_filename(
            branch, commit_id
        )

    @staticmethod
    def _serialize_datetime(dt: datetime) -> str:
        return dt.isoformat()

    @staticmethod
    def _deserialize_datetime(dt_str: Optional[str]) -> datetime:
        if not dt_str:
            return datetime.utcnow()
        try:
            return datetime.fromisoformat(dt_str)
        except ValueError:
            logger.warning(f"Failed to parse datetime '{dt_str}'; defaulting to now()")
            return datetime.utcnow()

    def _load_metadata_entry(
        self,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Load a single metadata entry from disk."""
        path = self._metadata_path(repo_name, branch, commit_id)
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to read repo metadata at {path}: {exc}")
            return None

        if not isinstance(data, dict):
            logger.warning(f"Metadata at {path} is not a JSON object")
            return None

        data.setdefault("repo_name", repo_name)
        data.setdefault("branch", branch)
        data.setdefault("commit_id", commit_id)
        return data

    def _write_metadata_entry(
        self,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
        data: Dict[str, Any],
    ) -> None:
        """Persist a metadata entry atomically."""
        path = self._metadata_path(repo_name, branch, commit_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(temp_path, path)

    def _delete_metadata_entry(
        self,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
    ) -> None:
        """Remove a metadata entry and clean up any empty directories."""
        path = self._metadata_path(repo_name, branch, commit_id)
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning(f"Failed to delete metadata file {path}: {exc}")

        # Remove empty parents up to metadata root
        current = path.parent
        while current != self.metadata_base_path and current != current.parent:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def _iter_metadata_entries(
        self,
        user_id: Optional[str] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Yield formatted metadata entries, optionally filtered by user."""
        if not self.metadata_base_path.exists():
            return

        for meta_file in self.metadata_base_path.rglob(f"*{self._METADATA_EXTENSION}"):
            repo_relative = meta_file.relative_to(self.metadata_base_path)
            repo_name = "/".join(repo_relative.parts[:-1])

            try:
                with meta_file.open("r", encoding="utf-8") as fh:
                    raw_data = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(f"Skipping corrupt metadata file {meta_file}: {exc}")
                continue

            if not isinstance(raw_data, dict):
                logger.warning(f"Unexpected metadata format in {meta_file}")
                continue

            entry = self._format_repo_info(repo_name, raw_data)
            if user_id and entry.get("user_id") != user_id:
                continue

            yield entry

    def _calculate_volume_bytes(self, path: str) -> Optional[int]:
        """
        Calculate the disk usage of a directory in bytes using 'du' command.

        Args:
            path: Path to the directory

        Returns:
            Size in bytes, or None if calculation fails
        """
        if not os.path.exists(path):
            return None

        try:
            # Use 'du -sb' to get size in bytes (summary, bytes)
            # This is more accurate and faster than walking the directory tree
            result = subprocess.run(
                ["du", "-sb", path],
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout for large directories
            )

            if result.returncode == 0:
                # du output format: "size_bytes\tpath"
                size_str = result.stdout.split()[0]
                return int(size_str)
            else:
                logger.warning(
                    f"Failed to calculate volume for {path}: {result.stderr}"
                )
                return None
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout calculating volume for {path}")
            return None
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing du output for {path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error calculating volume for {path}: {e}")
            return None

    def _format_repo_info(
        self,
        repo_name: str,
        raw_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize a raw metadata dict into the public repo info shape."""
        branch = raw_data.get("branch") or None
        commit_id = raw_data.get("commit_id") or None
        user_id = raw_data.get("user_id") or None
        repo_key = self._get_repo_key(repo_name, branch, commit_id, user_id)

        metadata_raw = raw_data.get("metadata") or {}
        if isinstance(metadata_raw, str):
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {}
        else:
            metadata = metadata_raw

        registered_at = self._deserialize_datetime(raw_data.get("registered_at"))
        last_accessed = self._deserialize_datetime(raw_data.get("last_accessed"))

        return {
            "repo_key": repo_key,
            "repo_name": repo_name,
            "local_path": raw_data.get("local_path"),
            "branch": branch,
            "commit_id": commit_id,
            "user_id": raw_data.get("user_id") or None,
            "registered_at": registered_at,
            "last_accessed": last_accessed,
            "metadata": metadata,
            "volume_bytes": raw_data.get(
                "volume_bytes"
            ),  # Include volume in formatted info
        }

    # ------------------------------------------------------------------ #
    # Public API (IRepoManager)
    # ------------------------------------------------------------------ #
    def _get_repo_key(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        parts = [repo_name]
        if branch:
            parts.append(f"branch:{branch}")
        if commit_id:
            parts.append(f"commit:{commit_id}")
        if user_id:
            parts.append(f"user:{user_id}")
        return ":".join(parts)

    def _get_repo_local_path(self, repo_name: str) -> Path:
        """Expose repository location for callers that already rely on it."""
        # Prevent path traversal
        if ".." in repo_name or Path(repo_name).is_absolute():
            raise ValueError(f"Invalid repo_name: {repo_name}")
        return self.repos_base_path / Path(repo_name)

    def is_repo_available(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        # First, try to find via metadata entry
        entry = self._load_metadata_entry(repo_name, branch, commit_id)
        if entry:
            if user_id and entry.get("user_id") != user_id:
                return False

            local_path = entry.get("local_path")
            if local_path and os.path.exists(local_path):
                return True

        # Fallback: Check if repository exists in expected filesystem location
        # This handles cases where repo exists but wasn't registered in metadata
        if not branch and not commit_id:  # Only for base repo lookups
            expected_path = self._get_repo_local_path(repo_name)
            if expected_path.exists() and expected_path.is_dir():
                # Check if it's a valid git repository
                git_dir = expected_path / ".git"
                if git_dir.exists():
                    return True

        return False

    def register_repo(
        self,
        repo_name: str,
        local_path: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not os.path.exists(local_path):
            raise ValueError(f"Local path does not exist: {local_path}")

        # Calculate volume before registration
        volume_bytes = self._calculate_volume_bytes(local_path)
        if volume_bytes is None:
            logger.warning(
                f"Failed to calculate volume for {local_path}, proceeding without volume tracking"
            )

        # Check if we need to evict repos to make space
        if volume_bytes:
            current_total = self.get_total_volume_bytes()
            if current_total + volume_bytes > self.volume_limit_bytes:
                logger.info(
                    f"Volume limit would be exceeded (current: {current_total:,}, new: {volume_bytes:,}, limit: {self.volume_limit_bytes:,}). Evicting LRU repos..."
                )
                evicted = self._evict_lru_repos_until_space_available(volume_bytes)
                # Check if we freed enough space
                new_total = self.get_total_volume_bytes()
                if new_total + volume_bytes > self.volume_limit_bytes:
                    logger.warning(
                        f"Could not free enough space. Current: {new_total:,}, new repo: {volume_bytes:,}, "
                        f"limit: {self.volume_limit_bytes:,}. Proceeding anyway (will exceed limit by "
                        f"{(new_total + volume_bytes - self.volume_limit_bytes):,} bytes)"
                    )
                else:
                    logger.info(
                        f"Freed enough space. New total: {new_total:,}, after adding repo: {new_total + volume_bytes:,}, "
                        f"limit: {self.volume_limit_bytes:,}"
                    )

        now = datetime.utcnow()
        data = {
            "repo_name": repo_name,
            "local_path": local_path,
            "branch": branch,
            "commit_id": commit_id,
            "user_id": user_id,
            "registered_at": self._serialize_datetime(now),
            "last_accessed": self._serialize_datetime(now),
            "metadata": metadata or {},
            "volume_bytes": volume_bytes,  # Store volume in metadata
        }

        self._write_metadata_entry(repo_name, branch, commit_id, data)
        repo_key = self._get_repo_key(repo_name, branch, commit_id, user_id)
        logger.info(
            "Registered repo %s at %s (volume: %s bytes)",
            repo_key,
            local_path,
            f"{volume_bytes:,}" if volume_bytes else "unknown",
        )
        return repo_key

    def get_repo_path(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        # First, try to find via metadata entry
        entry = self._load_metadata_entry(repo_name, branch, commit_id)
        if entry:
            if user_id and entry.get("user_id") != user_id:
                return None

            local_path = entry.get("local_path")
            if local_path and os.path.exists(local_path):
                return local_path

        # Fallback: Check if repository exists in expected filesystem location
        # This handles cases where repo exists but wasn't registered in metadata
        if not branch and not commit_id:  # Only for base repo lookups
            expected_path = self._get_repo_local_path(repo_name)
            if expected_path.exists() and expected_path.is_dir():
                # Check if it's a valid git repository
                git_dir = expected_path / ".git"
                if git_dir.exists():
                    logger.debug(
                        f"[REPO_MANAGER] Found unregistered repo at filesystem path: {expected_path}"
                    )
                    return str(expected_path)

        return None

    def update_last_accessed(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        entry = self._load_metadata_entry(repo_name, branch, commit_id)
        if not entry:
            logger.debug(
                "[REPO_MANAGER] Cannot update last_accessed; entry missing for %s",
                repo_name,
            )
            return

        if user_id and entry.get("user_id") != user_id:
            return

        entry["last_accessed"] = self._serialize_datetime(datetime.utcnow())
        self._write_metadata_entry(repo_name, branch, commit_id, entry)

    def get_repo_info(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        entry = self._load_metadata_entry(repo_name, branch, commit_id)
        if not entry:
            return None

        formatted = self._format_repo_info(repo_name, entry)
        if user_id and formatted.get("user_id") != user_id:
            return None

        return formatted

    def list_available_repos(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        repos = list(self._iter_metadata_entries(user_id=user_id))
        repos = [
            repo
            for repo in repos
            if repo.get("local_path") and os.path.exists(repo["local_path"])
        ]

        repos.sort(key=lambda item: item["last_accessed"], reverse=True)

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
        entry = self._load_metadata_entry(repo_name, branch, commit_id)
        if not entry:
            return False

        if user_id and entry.get("user_id") != user_id:
            return False

        local_path = entry.get("local_path")
        volume_bytes = entry.get("volume_bytes")
        self._delete_metadata_entry(repo_name, branch, commit_id)

        if local_path and os.path.exists(local_path):
            try:
                if os.path.isdir(local_path):
                    shutil.rmtree(local_path)
                else:
                    os.remove(local_path)
                logger.info(
                    "Deleted local repo copy at %s (volume: %s bytes)",
                    local_path,
                    f"{volume_bytes:,}" if volume_bytes else "unknown",
                )
            except OSError:
                logger.exception(f"Failed to delete local repo copy at {local_path}")

        logger.info(
            "Evicted repo %s (branch=%s, commit=%s, volume=%s bytes)",
            repo_name,
            branch,
            commit_id,
            f"{volume_bytes:,}" if volume_bytes else "unknown",
        )
        return True

    def recalculate_repo_volume(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Recalculate and update the volume for a specific repository.
        Useful if the repository size has changed significantly.

        Args:
            repo_name: Repository name
            branch: Branch name (optional)
            commit_id: Commit ID (optional)
            user_id: User ID (optional)

        Returns:
            Updated volume in bytes, or None if calculation failed
        """
        entry = self._load_metadata_entry(repo_name, branch, commit_id)
        if not entry:
            return None

        if user_id and entry.get("user_id") != user_id:
            return None

        local_path = entry.get("local_path")
        if not local_path or not os.path.exists(local_path):
            return None

        volume_bytes = self._calculate_volume_bytes(local_path)
        if volume_bytes is not None:
            entry["volume_bytes"] = volume_bytes
            self._write_metadata_entry(repo_name, branch, commit_id, entry)
            logger.info(
                f"Recalculated volume for {repo_name}@{branch or commit_id}: {volume_bytes:,} bytes"
            )

        return volume_bytes

    def evict_stale_repos(
        self,
        max_age_days: int,
        user_id: Optional[str] = None,
    ) -> List[str]:
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        evicted: List[str] = []

        for repo_info in self.list_available_repos(user_id=user_id):
            if repo_info["last_accessed"] < cutoff:
                repo_name = repo_info["repo_name"]
                branch = repo_info["branch"]
                commit_id = repo_info["commit_id"]
                repo_user_id = repo_info.get("user_id")

                if self.evict_repo(
                    repo_name,
                    branch=branch,
                    commit_id=commit_id,
                    user_id=repo_user_id,
                ):
                    evicted.append(repo_info["repo_key"])

        if evicted:
            logger.info(
                "Evicted %d stale repos older than %d days",
                len(evicted),
                max_age_days,
            )
        return evicted

    def get_repo_size(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[int]:
        local_path = self.get_repo_path(repo_name, branch, commit_id, user_id)
        if not local_path:
            return None

        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(local_path):
                if ".git" in dirpath:
                    continue

                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except (FileNotFoundError, OSError):
                        continue
        except OSError as exc:
            logger.warning(f"Error calculating repo size for {local_path}: {exc}")
            return None

        return total_size

    def get_total_volume_bytes(self, user_id: Optional[str] = None) -> int:
        """
        Get total volume used by all registered repositories in bytes.

        Args:
            user_id: Optional user ID to filter by user

        Returns:
            Total volume in bytes
        """
        total = 0
        for repo_info in self.list_available_repos(user_id=user_id):
            volume = repo_info.get("volume_bytes")
            if volume:
                total += volume
        return total

    def get_volume_percentage(self, user_id: Optional[str] = None) -> float:
        """
        Get the percentage of volume limit currently used.

        Args:
            user_id: Optional user ID to filter by user

        Returns:
            Percentage used (0.0 to 100.0)
        """
        total_bytes = self.get_total_volume_bytes(user_id=user_id)
        if self.volume_limit_bytes == 0:
            return 0.0
        percentage = (total_bytes / self.volume_limit_bytes) * 100.0
        return min(percentage, 100.0)  # Cap at 100%

    def _evict_lru_repos_until_space_available(
        self, required_bytes: int, user_id: Optional[str] = None
    ) -> List[str]:
        """
        Evict least recently used repos until enough space is available.

        Args:
            required_bytes: Number of bytes that need to be freed
            user_id: Optional user ID to filter evictions by user

        Returns:
            List of repo keys that were evicted
        """
        evicted: List[str] = []
        current_total = self.get_total_volume_bytes(user_id=user_id)
        target_total = current_total + required_bytes - self.volume_limit_bytes

        if target_total <= 0:
            # Already have enough space
            return evicted

        # Get all repos sorted by last_accessed (oldest first)
        repos = self.list_available_repos(user_id=user_id)
        repos.sort(key=lambda r: r.get("last_accessed", datetime.min))

        freed_bytes = 0
        for repo_info in repos:
            if freed_bytes >= target_total:
                break

            repo_name = repo_info["repo_name"]
            branch = repo_info.get("branch")
            commit_id = repo_info.get("commit_id")
            repo_user_id = repo_info.get("user_id")
            volume = repo_info.get("volume_bytes", 0)

            if self.evict_repo(
                repo_name,
                branch=branch,
                commit_id=commit_id,
                user_id=repo_user_id,
            ):
                evicted.append(repo_info["repo_key"])
                freed_bytes += volume
                logger.info(
                    f"Evicted repo {repo_info['repo_key']} to free {volume} bytes "
                    f"(freed: {freed_bytes}/{target_total} bytes)"
                )

        if evicted:
            logger.info(
                f"Evicted {len(evicted)} repos, freed {freed_bytes} bytes "
                f"(target was {target_total} bytes)"
            )
        else:
            logger.warning(
                f"Could not free enough space. Required: {required_bytes} bytes, "
                f"current usage: {current_total} bytes, limit: {self.volume_limit_bytes} bytes"
            )

        return evicted
