"""
Repository Manager Implementation

Manages local copies of repositories stored in `.repos`.
Tracks repository metadata using the filesystem instead of Redis.
"""

import json
import os
import shutil
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
            repos_base_path: Base path for storing repositories. Defaults to `.repos`
                at the project root (parent of the `app` directory).
        """
        if repos_base_path:
            self.repos_base_path = Path(repos_base_path).resolve()
        else:
            project_root = Path(__file__).parent.parent.parent.parent
            self.repos_base_path = project_root / ".repos"

        self.metadata_base_path = self.repos_base_path / self._METADATA_ROOT_NAME

        self.repos_base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_base_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "RepoManager initialized with base path %s and metadata path %s",
            self.repos_base_path,
            self.metadata_base_path,
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
        entry = self._load_metadata_entry(repo_name, branch, commit_id)
        if not entry:
            return False

        if user_id and entry.get("user_id") != user_id:
            return False

        local_path = entry.get("local_path")
        if not local_path or not os.path.exists(local_path):
            return False

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
        if not os.path.exists(local_path):
            raise ValueError(f"Local path does not exist: {local_path}")

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
        }

        self._write_metadata_entry(repo_name, branch, commit_id, data)
        repo_key = self._get_repo_key(repo_name, branch, commit_id, user_id)
        logger.info(f"Registered repo {repo_key} at {local_path}")
        return repo_key

    def get_repo_path(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        entry = self._load_metadata_entry(repo_name, branch, commit_id)
        if not entry:
            return None

        if user_id and entry.get("user_id") != user_id:
            return None

        local_path = entry.get("local_path")
        if local_path and os.path.exists(local_path):
            return local_path

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
        self._delete_metadata_entry(repo_name, branch, commit_id)

        if local_path and os.path.exists(local_path):
            try:
                if os.path.isdir(local_path):
                    shutil.rmtree(local_path)
                else:
                    os.remove(local_path)
                logger.info(f"Deleted local repo copy at {local_path}")
            except OSError:
                logger.exception(f"Failed to delete local repo copy at {local_path}")

        logger.info(
            "Evicted repo %s (branch=%s, commit=%s)",
            repo_name,
            branch,
            commit_id,
        )
        return True

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
