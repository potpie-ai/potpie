"""
Repository Manager - Bare Repo + Worktree Architecture

Manages local copies of Git repositories using bare repositories + worktrees
for efficient storage and fast parsing.

Design Principles:
- Multi-tenant: user_id tracked everywhere for isolation
- Bare repos: Persistent, shared across all parses
- Worktrees: Temporary, created for parsing, removed by background eviction
- Metadata: Centralized in .meta directory for all repos
- Security: Input validation and credential sanitization
- Eviction: Tiered (worktrees first, then repos) with background cleanup
"""

import json
import os
import secrets
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from app.modules.repo_manager.repo_manager_interface import IRepoManager
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class RepoManager(IRepoManager):
    """
    Implementation of IRepoManager backed entirely by the local filesystem.

    Repository checkouts live under `.repos/<owner>/<repo>` and metadata about
    worktrees/refs is persisted alongside them in `.repos/.meta/...`.

    Architecture:
    - Bare repositories: `.repos/<owner>/<repo>/.bare/`
    - Worktrees: `.repos/<owner>/<repo>/worktrees/<ref>/`
    - Metadata: `.repos/.meta/<owner>/<repo>/branch__commit.json`
    """

    # ========== CONSTANTS ==========

    # Metadata
    _METADATA_ROOT_NAME = ".meta"
    _METADATA_EXTENSION = ".json"
    _TYPE_BARE_REPO = "bare_repo"
    _TYPE_WORKTREE = "worktree"

    # Directory structure
    _BARE_REPO_DIR_NAME = ".bare"
    _WORKTREES_DIR_NAME = "worktrees"

    # Timeouts (seconds)
    _FETCH_TIMEOUT = 300
    _CLONE_TIMEOUT = 600
    _WORKTREE_REMOVE_TIMEOUT = 60

    # Eviction thresholds
    _WORKTREE_EVICTION_THRESHOLD_PERCENTAGE = 80.0
    _REPO_EVICTION_TARGET_PERCENTAGE = 90.0
    _STALE_WORKTREE_MAX_AGE_DAYS = 30
    _DEFAULT_VOLUME_LIMIT_GB = 100

    # ========== INITIALIZATION ==========

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
                self.volume_limit_bytes = 100 * 1024 * 1024 * 1024
        else:
            self.volume_limit_bytes = 100 * 1024 * 1024 * 1024

        logger.info(
            "RepoManager initialized with base path %s, metadata path %s, volume limit %d bytes (%.2f GB)",
            self.repos_base_path,
            self.metadata_base_path,
            self.volume_limit_bytes,
            self.volume_limit_bytes / (1024**3),
        )

    # ========== VALIDATION & SECURITY ==========

    @staticmethod
    def _validate_repo_name(repo_name: str) -> None:
        """Validate repo_name to prevent path traversal attacks."""
        if not repo_name:
            raise ValueError("repo_name cannot be empty")
        if ".." in repo_name or repo_name.startswith("/") or repo_name.startswith("\\"):
            raise ValueError(
                f"Invalid repo_name '{repo_name}': path traversal not allowed"
            )
        if "/" not in repo_name:
            raise ValueError(
                f"Invalid repo_name '{repo_name}': must be in format 'owner/repo'"
            )

    @staticmethod
    def _validate_ref(ref: str) -> None:
        """Validate ref to prevent command injection."""
        if not ref:
            raise ValueError("ref cannot be empty")
        if ".." in ref:
            raise ValueError(f"Invalid ref '{ref}': path traversal not allowed")
        if "\n" in ref or "\r" in ref:
            raise ValueError(f"Invalid ref '{ref}': newline characters not allowed")

    @staticmethod
    def _sanitize_error_message(error_msg: str) -> str:
        """Remove potential credentials from error messages."""
        import re

        sanitized = re.sub(r"oauth2:[^@]+@", "oauth2:***@", error_msg)
        sanitized = re.sub(r"ghp_[a-zA-Z0-9]{36}", "***", sanitized)
        return sanitized

    # ========== GIT AUTHENTICATION ==========

    def _derive_github_url(self, repo_name: str) -> str:
        """Derive GitHub URL from repo_name using GITHUB_BASE_URL."""
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        github_base = os.getenv("GITHUB_BASE_URL", "github.com")

        return f"https://{github_base}/{repo_name}.git"

    def _get_github_token(self) -> Optional[str]:
        """Get GitHub token from environment or token pool."""
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        if token:
            return token

        token_list_str = os.getenv("GH_TOKEN_LIST", "")
        if token_list_str:
            token_list = [t.strip() for t in token_list_str.split(",") if t.strip()]
            if token_list:
                token = secrets.choice(token_list)
                logger.info(f"Selected token from pool ({len(token_list)} available)")
                return token

        return None

    def _build_authenticated_url(self, repo_url: str, auth_token: Optional[str]) -> str:
        """Build authenticated URL with oauth2: prefix."""
        if not auth_token:
            return repo_url

        parsed = urlparse(repo_url)

        if parsed.scheme in ("https", "http"):
            netloc = f"oauth2:{auth_token}@{parsed.netloc}"
            return parsed._replace(netloc=netloc).geturl()
        else:
            logger.warning(f"Unsupported URL scheme: {parsed.scheme}")
            return repo_url

    # ========== PATH HELPERS ==========

    def _get_repo_dir(self, repo_name: str) -> Path:
        """Get base directory for a repo."""
        self._validate_repo_name(repo_name)
        return self.repos_base_path / repo_name

    def _get_repo_local_path(self, repo_name: str) -> Path:
        """Get repo checkout path (for direct access, same as _get_repo_dir)."""
        self._validate_repo_name(repo_name)
        return self.repos_base_path / repo_name

    def _get_bare_repo_path(self, repo_name: str) -> Path:
        """Get bare repository directory path."""
        self._validate_repo_name(repo_name)
        return self.repos_base_path / repo_name / self._BARE_REPO_DIR_NAME

    def _get_worktrees_dir(self, repo_name: str) -> Path:
        """Get worktrees directory path."""
        self._validate_repo_name(repo_name)
        return self.repos_base_path / repo_name / self._WORKTREES_DIR_NAME

    def _get_worktree_path(self, repo_name: str, ref: str) -> Path:
        """Get specific worktree path."""
        self._validate_repo_name(repo_name)
        safe_ref = ref.replace("/", "_").replace("\\", "_")
        return self._get_worktrees_dir(repo_name) / safe_ref

    def _get_unique_worktree_path(
        self, repo_name: str, ref: str, user_id: str, unique_id: str
    ) -> Path:
        """
        Get path for a unique worktree with user_id and unique_id prefix.

        Args:
            repo_name: Repository name
            ref: Branch name or commit SHA
            user_id: User ID
            unique_id: Unique identifier for this worktree

        Returns:
            Path to unique worktree directory
        """
        self._validate_repo_name(repo_name)
        safe_ref = ref.replace("/", "_").replace("\\", "_")
        safe_user_id = user_id.replace("/", "_").replace("\\", "_")
        safe_unique_id = unique_id.replace("/", "_").replace("\\", "_")
        return (
            self._get_worktrees_dir(repo_name)
            / f"{safe_user_id}_{safe_unique_id}_{safe_ref}"
        )

    # ========== METADATA HELPERS ==========

    @staticmethod
    def _sanitize_for_filename(value: str) -> str:
        """Convert arbitrary text into a filesystem-safe token."""
        return "".join(
            c if c.isalnum() or c in ("-", "_", ".", "=") else "_" for c in value
        )

    def _metadata_dir(self, repo_name: str) -> Path:
        """Return the metadata directory for a given repository."""
        self._validate_repo_name(repo_name)
        return self.metadata_base_path / repo_name

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

    # ========== METADATA I/O ==========

    def _read_metadata(
        self,
        metadata_path: Path,
    ) -> Optional[Dict[str, Any]]:
        """Read metadata from file."""
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read metadata at {metadata_path}: {e}")
            return None

    def _write_metadata(
        self,
        metadata_path: Path,
        data: Dict[str, Any],
    ) -> None:
        """Write metadata atomically."""
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(temp_path, metadata_path)

    def _delete_metadata(
        self,
        metadata_path: Path,
    ) -> None:
        """Delete metadata and cleanup empty dirs."""
        try:
            if metadata_path.exists():
                metadata_path.unlink()
        except OSError as exc:
            logger.warning(f"Failed to delete metadata file {metadata_path}: {exc}")

        # Remove empty parents up to metadata root
        current = metadata_path.parent
        while current != self.metadata_base_path and current != current.parent:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

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

    def _delete_metadata_by_repo(
        self,
        repo_name: str,
    ) -> int:
        self._validate_repo_name(repo_name)
        repo_metadata_dir = self._metadata_dir(repo_name)

        if not repo_metadata_dir.exists():
            logger.info(f"Metadata directory for repo {repo_name} does not exist")
            return 0

        deleted_count = 0

        for meta_file in repo_metadata_dir.glob(f"*{self._METADATA_EXTENSION}"):
            try:
                if meta_file.is_file():
                    meta_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted metadata file: {meta_file.name}")
            except OSError as exc:
                logger.warning(f"Failed to delete metadata file {meta_file}: {exc}")
                continue

        try:
            if repo_metadata_dir.exists() and not any(repo_metadata_dir.iterdir()):
                repo_metadata_dir.rmdir()
                logger.info(f"Removed empty metadata directory for repo: {repo_name}")
        except OSError as exc:
            logger.warning(
                f"Failed to remove metadata directory for repo {repo_name}: {exc}"
            )

        logger.info(f"Deleted {deleted_count} metadata entries for repo: {repo_name}")
        return deleted_count

    def _iter_metadata_entries(
        self,
        user_id: Optional[str] = None,
    ) -> Any:
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

        # Merge top-level "type" into metadata for list_repos filtering
        repo_type = raw_data.get("type")
        if repo_type and not metadata.get("type"):
            metadata["type"] = repo_type

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

    # ========== VOLUME CALCULATION ==========

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
                ["du", "-s", path],
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

    # ========== METADATA UPDATE HELPERS (v2 additions adapted for v1 structure) ==========

    def _update_bare_repo_metadata(
        self,
        repo_name: str,
        repo_url: Optional[str] = None,
        auth_used: Optional[bool] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update bare repo metadata (v2 style, adapted for v1 centralized structure)."""
        # Load or create entry
        data = self._load_metadata_entry(repo_name, None, None)
        if data is None:
            data = {
                "repo_name": repo_name,
                "type": self._TYPE_BARE_REPO,
                "repo_url": repo_url,
                "auth_used": auth_used,
                "registered_at": self._serialize_datetime(datetime.utcnow()),
                "volume_bytes": 0,
            }

        # Update fields
        data["last_accessed"] = self._serialize_datetime(datetime.utcnow())
        bare_repo_path = self._get_bare_repo_path(repo_name)
        if bare_repo_path.exists():
            data["local_path"] = str(bare_repo_path)
        else:
            data["local_path"] = str(self._get_repo_local_path(repo_name))
        if repo_url is not None:
            data["repo_url"] = repo_url
        if auth_used is not None:
            data["auth_used"] = auth_used
        if user_id is not None:
            data["user_id"] = user_id

        # Recalculate volume if bare repo exists
        bare_repo_path = self._get_bare_repo_path(repo_name)
        if bare_repo_path.exists():
            volume = self._calculate_volume_bytes(str(bare_repo_path))
            if volume is None:
                volume = data.get("volume_bytes", 0)
            data["volume_bytes"] = volume

        # Write metadata
        self._write_metadata_entry(repo_name, None, None, data)

        return data

    # ========== GIT OPERATIONS ==========

    def _fetch_ref(
        self,
        bare_repo_path: Path,
        ref: str,
        auth_token: Optional[str],
        repo_url: Optional[str] = None,
    ) -> None:
        """Fetch updates to bare repository for specific ref."""
        try:
            logger.info(f"Fetching ref '{ref}' for bare repo at {bare_repo_path}")

            fetch_remote = "origin"
            if auth_token and repo_url:
                fetch_remote = self._build_authenticated_url(repo_url, auth_token)

            result = subprocess.run(
                ["git", "-C", str(bare_repo_path), "fetch", fetch_remote, "--", ref],
                capture_output=True,
                text=True,
                timeout=self._FETCH_TIMEOUT,
            )

            if result.returncode != 0:
                sanitized_error = self._sanitize_error_message(result.stderr or "")
                logger.warning(f"Failed to fetch ref '{ref}': {sanitized_error}")
            else:
                logger.info(f"Successfully fetched ref '{ref}'")

        except subprocess.TimeoutExpired:
            logger.warning(f"Fetch timeout for ref '{ref}'")
        except Exception as e:
            logger.warning(f"Error fetching ref '{ref}': {e}")

    def ensure_bare_repo(
        self,
        repo_name: str,
        repo_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        ref: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Path:
        """
        Ensure bare repository exists and is up-to-date.

        Creates bare repository with --filter=blob:none if it doesn't exist.
        Updates/fetches if it exists.

        Args:
            repo_name: Repository name (e.g., 'owner/repo')
            repo_url: Git repository URL (derived if not provided)
            auth_token: Optional authentication token
            ref: Optional branch or commit to fetch
            user_id: Optional user ID for multi-tenant tracking

        Returns:
            Path to bare repository directory
        """
        self._validate_repo_name(repo_name)
        if ref:
            self._validate_ref(ref)

        bare_repo_path = self._get_bare_repo_path(repo_name)

        # Derive URL if not provided
        if repo_url is None:
            repo_url = self._derive_github_url(repo_name)
            logger.info(f"Derived GitHub URL: {repo_url}")

        # Get auth token
        github_token = auth_token or self._get_github_token()
        clone_url = self._build_authenticated_url(repo_url, github_token)

        if bare_repo_path.exists() and (bare_repo_path / "HEAD").exists():
            logger.info(f"Bare repo already exists: {repo_name}")
            if ref:
                self._fetch_ref(bare_repo_path, ref, github_token, repo_url)
            self._update_bare_repo_metadata(
                repo_name, repo_url, bool(github_token), user_id
            )
            return bare_repo_path

        try:
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--bare",
                    "--filter=blob:none",
                    "--",
                    clone_url,
                    str(bare_repo_path),
                ],
                capture_output=True,
                text=True,
                timeout=self._CLONE_TIMEOUT,
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                sanitized_error = self._sanitize_error_message(error_msg)

                if (
                    "Authentication failed" in error_msg
                    or "Permission denied" in error_msg
                ):
                    raise RuntimeError(
                        f"Git authentication failed for '{repo_name}'.\n"
                        f"Please check:\n"
                        f"  1. GitHub token is valid\n"
                        f"  2. Token has 'repo' scope\n"
                        f"  3. You have access to this repository\n"
                        f"Error: {sanitized_error}"
                    )
                else:
                    raise RuntimeError(f"Git clone failed: {sanitized_error}")

            logger.info(f"Successfully cloned bare repo: {repo_name}")

            if ref:
                self._fetch_ref(bare_repo_path, ref, github_token, repo_url)

            self._update_bare_repo_metadata(
                repo_name,
                repo_url=repo_url,
                auth_used=bool(github_token),
                user_id=user_id,
            )

            return bare_repo_path

        except subprocess.TimeoutExpired:
            logger.error("Git clone timeout", repo_name=repo_name)
            raise RuntimeError(
                f"Git clone timed out after {self._CLONE_TIMEOUT // 60} minutes"
            )
        except Exception as e:
            logger.exception(f"Unexpected error cloning {repo_name}")
            raise RuntimeError(f"Failed to clone bare repo: {e}") from e

    def create_worktree(
        self,
        repo_name: str,
        ref: str,
        auth_token: Optional[str] = None,
        is_commit: bool = False,
        user_id: Optional[str] = None,
        unique_id: Optional[str] = None,
        exists_ok: bool = False,
    ) -> Path:
        """
        Create worktree from bare repository for parsing.

        If worktree already exists for this ref, checks if it's at the latest commit.
        If not, removes and recreates the worktree.

        Args:
            repo_name: Repository name
            ref: Branch name or commit SHA
            auth_token: Optional authentication token
            is_commit: Whether ref is a commit SHA
            user_id: Optional user ID for multi-tenant tracking
            unique_id: Optional unique ID to create detached HEAD worktree with prefix

        Returns:
            Path to worktree directory
        """
        self._validate_repo_name(repo_name)
        self._validate_ref(ref)

        bare_repo_path = self._get_bare_repo_path(repo_name)

        if unique_id:
            if not user_id:
                raise ValueError("user_id is required when unique_id is provided")
            worktree_path = self._get_unique_worktree_path(
                repo_name, ref, user_id, unique_id
            )
        else:
            worktree_path = self._get_worktree_path(repo_name, ref)

        self._get_worktrees_dir(repo_name).mkdir(parents=True, exist_ok=True)

        if worktree_path.exists():
            logger.info(
                f"Worktree already exists for {repo_name}@{ref}, checking if update needed"
            )
            if unique_id:
                if exists_ok:
                    logger.info(f"Worktree already exists for {repo_name}@{ref}")
                    return worktree_path
                else:
                    raise FileExistsError(
                        f"Worktree already exists for {repo_name}@{ref}"
                    )

            try:
                result = subprocess.run(
                    ["git", "-C", str(worktree_path), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    current_commit = result.stdout.strip()

                    if is_commit:
                        if current_commit.startswith(ref):
                            logger.info(f"Worktree already at requested commit {ref}")
                            self.update_last_accessed(
                                repo_name,
                                None if is_commit else ref,
                                ref if is_commit else None,
                                user_id,
                            )
                            return worktree_path
                        else:
                            logger.info(
                                f"Worktree at {current_commit}, but requested {ref}. Recreating..."
                            )
                    else:
                        result = subprocess.run(
                            [
                                "git",
                                "-C",
                                str(bare_repo_path),
                                "config",
                                "remote.origin.url",
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                        )

                        repo_url = (
                            result.stdout.strip() if result.returncode == 0 else None
                        )

                        github_token = auth_token or self._get_github_token()
                        self._fetch_ref(bare_repo_path, ref, github_token, repo_url)

                        branch_ref = f"refs/heads/{ref}"
                        result = subprocess.run(
                            ["git", "-C", str(bare_repo_path), "rev-parse", branch_ref],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )

                        if result.returncode == 0:
                            latest_commit = result.stdout.strip()
                            if current_commit == latest_commit:
                                logger.info(
                                    f"Worktree already at latest commit for branch {ref}"
                                )
                                self.update_last_accessed(repo_name, ref, None, user_id)
                                return worktree_path
                            else:
                                logger.info(
                                    f"Worktree at {current_commit}, latest is {latest_commit}. Recreating..."
                                )
                        else:
                            logger.warning(
                                f"Could not get latest commit for branch {ref}, recreating worktree",
                                result=result,
                            )
                else:
                    logger.warning(
                        "Could not get current HEAD for worktree, recreating"
                    )
            except Exception as e:
                logger.warning(f"Error checking worktree state: {e}, recreating")

            logger.info(f"Removing existing worktree at {worktree_path}")
            try:
                result = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(bare_repo_path),
                        "worktree",
                        "remove",
                        "--force",
                        "--",
                        str(worktree_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self._WORKTREE_REMOVE_TIMEOUT,
                )

                if result.returncode != 0:
                    logger.warning(
                        f"Git worktree remove failed, using rmtree: {result.stderr}"
                    )
                    shutil.rmtree(worktree_path, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error removing worktree: {e}, using rmtree")
                shutil.rmtree(worktree_path, ignore_errors=True)

            self._delete_metadata_entry(
                repo_name, None if is_commit else ref, ref if is_commit else None
            )

        if unique_id or is_commit:
            cmd = [
                "git",
                "-C",
                str(bare_repo_path),
                "worktree",
                "add",
                "--detach",
                "--",
                str(worktree_path),
                ref,
            ]
        else:
            cmd = [
                "git",
                "-C",
                str(bare_repo_path),
                "worktree",
                "add",
                "--",
                str(worktree_path),
                ref,
            ]

        try:
            logger.info(f"Creating worktree for {repo_name}@{ref}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._FETCH_TIMEOUT,
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                logger.error(f"Failed to create worktree: {error_msg}")
                raise RuntimeError(f"Git worktree add failed: {error_msg}")

            logger.info(f"Successfully created worktree: {repo_name}@{ref}")

            # Register worktree in metadata
            metadata = {"type": self._TYPE_WORKTREE, "is_commit": is_commit}
            if unique_id:
                metadata["unique_id"] = unique_id

            self.register_repo(
                repo_name=repo_name,
                local_path=str(worktree_path),
                branch=None if is_commit else ref,
                commit_id=ref if is_commit else None,
                user_id=user_id,
                metadata=metadata,
            )

            # Update bare repo metadata
            self._update_bare_repo_metadata(repo_name, user_id=user_id)

            return worktree_path

        except subprocess.TimeoutExpired:
            logger.error(f"Worktree creation timeout for {repo_name}@{ref}")
            raise RuntimeError(
                f"Worktree creation timed out after {self._FETCH_TIMEOUT // 60} minutes"
            )
        except Exception as e:
            logger.exception(
                f"Unexpected error creating worktree for {repo_name}@{ref}"
            )
            raise RuntimeError(f"Failed to create worktree: {e}") from e

    def remove_worktree(
        self,
        repo_name: str,
        ref: str,
    ) -> bool:
        """
        Remove worktree after parsing is complete.

        Keeps bare repository for future use, removes worktree to free disk space.

        Args:
            repo_name: Repository name
            ref: Branch name or commit SHA

        Returns:
            True if removed successfully, False otherwise
        """
        self._validate_repo_name(repo_name)
        self._validate_ref(ref)

        bare_repo_path = self._get_bare_repo_path(repo_name)
        worktree_path = self._get_worktree_path(repo_name, ref)

        logger.info(f"Removing worktree: {repo_name}@{ref}")

        worktree_removed = False

        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(bare_repo_path),
                    "worktree",
                    "remove",
                    "--force",
                    "--",
                    str(worktree_path),
                ],
                capture_output=True,
                text=True,
                timeout=self._WORKTREE_REMOVE_TIMEOUT,
            )

            if result.returncode == 0:
                logger.info(f"Successfully removed worktree via git: {repo_name}@{ref}")
                worktree_removed = True
            else:
                logger.warning(
                    f"Git worktree remove failed, using rmtree: {result.stderr}"
                )
                shutil.rmtree(worktree_path, ignore_errors=True)
                worktree_removed = not os.path.exists(worktree_path)

        except subprocess.TimeoutExpired:
            logger.warning("Timeout removing worktree, using rmtree")
            shutil.rmtree(worktree_path, ignore_errors=True)
            worktree_removed = not os.path.exists(worktree_path)
        except Exception as e:
            logger.warning(f"Failed to remove worktree via git: {e}, using rmtree")
            shutil.rmtree(worktree_path, ignore_errors=True)
            worktree_removed = not os.path.exists(worktree_path)

        if worktree_removed:
            # Remove metadata entry
            # Detect whether ref was stored as branch or commit by checking existing metadata
            metadata_as_branch = self._load_metadata_entry(repo_name, ref, None)
            if metadata_as_branch:
                self._delete_metadata_entry(repo_name, ref, None)
            else:
                metadata_as_commit = self._load_metadata_entry(repo_name, None, ref)
                if metadata_as_commit:
                    self._delete_metadata_entry(repo_name, None, ref)
                else:
                    logger.warning(
                        f"No metadata found for {repo_name}@{ref}, deleting as branch"
                    )
                    self._delete_metadata_entry(repo_name, ref, None)

            return True
        else:
            logger.error(f"Failed to remove worktree: {repo_name}@{ref}")
            return False

    def get_worktree_path(
        self,
        repo_name: str,
        ref: str,
    ) -> Optional[Path]:
        """
        Get path to existing worktree (v2-style method).

        Args:
            repo_name: Repository name
            ref: Branch name or commit SHA

        Returns:
            Path to worktree, or None if doesn't exist
        """
        self._validate_repo_name(repo_name)
        worktree_path = self._get_worktree_path(repo_name, ref)
        if worktree_path.exists():
            return worktree_path
        return None

    def cleanup_unique_worktree(
        self, repo_name: str, user_id: str, unique_id: str
    ) -> bool:
        """
        Remove a unique worktree for a given user_id and unique_id.

        This removes all worktrees that match the user_id and unique_id prefix
        for the specified repository.

        Args:
            repo_name: Repository name
            user_id: User ID
            unique_id: Unique identifier for worktrees to cleanup

        Returns:
            True if at least one worktree was removed, False otherwise
        """
        self._validate_repo_name(repo_name)
        bare_repo_path = self._get_bare_repo_path(repo_name)
        worktrees_dir = self._get_worktrees_dir(repo_name)

        logger.info(
            f"Cleaning up unique worktree for {repo_name} user:{user_id} id:{unique_id}"
        )

        safe_user_id = user_id.replace("/", "_").replace("\\", "_")
        safe_unique_id = unique_id.replace("/", "_").replace("\\", "_")
        prefix = f"{safe_user_id}_{safe_unique_id}_"

        removed = False

        if not worktrees_dir.exists():
            logger.info(f"Worktrees directory does not exist: {worktrees_dir}")
            return False

        try:
            for worktree_path in worktrees_dir.iterdir():
                if not worktree_path.is_dir():
                    continue

                if worktree_path.name.startswith(prefix):
                    logger.info(f"Removing unique worktree: {worktree_path.name}")

                    try:
                        result = subprocess.run(
                            [
                                "git",
                                "-C",
                                str(bare_repo_path),
                                "worktree",
                                "remove",
                                "--force",
                                "--",
                                str(worktree_path),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=self._WORKTREE_REMOVE_TIMEOUT,
                        )

                        if result.returncode == 0:
                            logger.info(
                                f"Successfully removed unique worktree via git: {worktree_path.name}"
                            )
                        else:
                            logger.warning(
                                f"Git worktree remove failed, using rmtree: {result.stderr}"
                            )
                            shutil.rmtree(worktree_path, ignore_errors=True)

                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"Timeout removing worktree {worktree_path.name}, using rmtree"
                        )
                        shutil.rmtree(worktree_path, ignore_errors=True)
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove worktree via git: {e}, using rmtree"
                        )
                        shutil.rmtree(worktree_path, ignore_errors=True)

                    removed = True

            if removed:
                logger.info(
                    f"Successfully cleaned up unique worktrees for user:{user_id} id:{unique_id}"
                )
            else:
                logger.info(
                    f"No unique worktrees found for user:{user_id} id:{unique_id}"
                )

            return removed

        except Exception as e:
            logger.exception(f"Error cleaning up unique worktrees: {e}")
            return False

    # ========== EVICTION LOGIC (v2 tiered eviction) ==========

    def _evict_if_needed(self, user_id: Optional[str] = None) -> None:
        """Auto-evict based on volume thresholds."""
        percentage = self.get_volume_percentage(user_id=user_id)

        if percentage > self._WORKTREE_EVICTION_THRESHOLD_PERCENTAGE:
            logger.info(f"Volume at {percentage:.1f}%, evicting stale worktrees")
            self.evict_stale_worktrees(
                max_age_days=self._STALE_WORKTREE_MAX_AGE_DAYS, user_id=user_id
            )

        percentage = self.get_volume_percentage(user_id=user_id)

        if percentage > self._REPO_EVICTION_TARGET_PERCENTAGE:
            logger.info(f"Volume at {percentage:.1f}%, evicting LRU repos")
            # Use existing evict_stale_repos with very short age when critical
            self.evict_stale_repos(max_age_days=1, user_id=user_id)

    def evict_stale_worktrees(
        self,
        max_age_days: int = 30,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Evict old worktrees to free disk space.

        Worktrees are evicted before bare repos when volume thresholds are reached.
        This is called automatically by _evict_if_needed().

        Args:
            max_age_days: Maximum age in days before eviction
            user_id: Optional user ID for multi-tenant tracking

        Returns:
            List of worktrees that were evicted
        """
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        evicted = []

        for entry in self.list_available_repos(user_id=user_id):
            metadata = entry.get("metadata", {})
            if metadata.get("type") != self._TYPE_WORKTREE:
                continue

            try:
                last_accessed = entry["last_accessed"]
                if isinstance(last_accessed, str):
                    last_accessed = self._deserialize_datetime(last_accessed)
            except (ValueError, TypeError):
                continue

            if last_accessed < cutoff:
                repo_name = entry["repo_name"]
                branch = entry["branch"]
                commit_id = entry["commit_id"]

                unique_id = metadata.get("unique_id")
                if unique_id:
                    entry_user_id = entry.get("user_id")
                    if entry_user_id:
                        self.cleanup_unique_worktree(
                            repo_name, entry_user_id, unique_id
                        )
                        evicted.append(f"{repo_name}@{entry_user_id}_{unique_id}")
                    continue

                if self.evict_repo(
                    repo_name, branch=branch, commit_id=commit_id, user_id=user_id
                ):
                    ref_str = branch or commit_id or "unknown"
                    evicted.append(f"{repo_name}@{ref_str}")

        if evicted:
            logger.info(
                f"Evicted {len(evicted)} stale worktrees (max_age={max_age_days}d)"
            )

        return evicted

    # ========== PUBLIC API (IRepoManager Implementation) ==========

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

    def is_repo_available(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        commit_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Check if a repository is available locally.
        """
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
        """
        Register a repository that has been downloaded/parsed.
        """
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
            current_total = self.get_total_volume_bytes(user_id=user_id)
            if current_total + volume_bytes > self.volume_limit_bytes:
                logger.info(
                    f"Volume limit would be exceeded (current: {current_total:,}, new: {volume_bytes:,}, limit: {self.volume_limit_bytes:,}). Evicting LRU repos..."
                )
                _evicted = self.evict_stale_repos(max_age_days=1, user_id=user_id)
                # Check if we freed enough space
                new_total = self.get_total_volume_bytes(user_id=user_id)
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
        """
        Get local filesystem path for a repository.

        Can return paths to:
        - Worktrees (when branch/commit_id is specified)
        - Bare repo checkout directories (when no branch/commit - fallback only)
        """
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
        """Update last accessed timestamp for a repository."""
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
        """Get information about a registered repository."""
        entry = self._load_metadata_entry(repo_name, branch, commit_id)
        if not entry:
            return None

        formatted = self._format_repo_info(repo_name, entry)
        if user_id and formatted.get("user_id") != user_id:
            return None

        return formatted

    def list_repos(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all bare repositories (v2-style method).

        Unlike list_available_repos() which includes all metadata entries,
        this returns only bare repo registrations.
        """
        repos = []

        for entry in self._iter_metadata_entries(user_id=user_id):
            metadata = entry.get("metadata", {})
            if metadata.get("type") == self._TYPE_BARE_REPO:
                repos.append(entry)

        if limit:
            repos = repos[:limit]

        return repos

    def list_available_repos(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all available repositories including worktrees.

        DEPRECATED: Use list_repos() for v2-style behavior (bare repos only).
        """
        logger.warning(
            "list_available_repos() is deprecated. Use list_repos() for bare repos only."
        )
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
        """
        Evict a repository from local storage.

        If branch/commit specified, removes only that checkout (worktree).
        If not specified, removes entire repo directory (v2 behavior).
        """
        self._validate_repo_name(repo_name)

        # If specific checkout, use v1-style evict
        if branch or commit_id:
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
                    logger.exception(
                        f"Failed to delete local repo copy at {local_path}"
                    )

            logger.info(
                "Evicted repo %s (branch=%s, commit=%s, volume=%s bytes)",
                repo_name,
                branch,
                commit_id,
                f"{volume_bytes:,}" if volume_bytes else "unknown",
            )
            return True

        # If no branch/commit, evict entire repo (v2 behavior)
        repo_dir = self._get_repo_local_path(repo_name)

        logger.info(f"Evicting entire repo: {repo_name}")

        try:
            # First, delete all metadata entries for this repo
            deleted_count = self._delete_metadata_by_repo(repo_name)
            logger.info(
                f"Deleted {deleted_count} metadata entries for repo: {repo_name}"
            )

            # Then, delete the repo directory
            shutil.rmtree(repo_dir)
            logger.info(f"Successfully evicted repo: {repo_name}")

            return True
        except OSError as e:
            logger.error(f"Failed to evict repo {repo_name}: {e}")
            return False

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
        """
        Evict repositories that haven't been accessed in a while.
        """
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
        """
        Get size of a repository in bytes.
        """
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
        """
        total_bytes = self.get_total_volume_bytes(user_id=user_id)
        if self.volume_limit_bytes == 0:
            return 0.0
        percentage = (total_bytes / self.volume_limit_bytes) * 100.0
        return min(percentage, 100.0)

    # ========== HIGH-LEVEL LIFECYCLE (v2 additions) ==========

    def prepare_for_parsing(
        self,
        repo_name: str,
        ref: str,
        repo_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        is_commit: bool = False,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Prepare a repository for parsing.

        Orchestrates: eviction → ensure bare repo → create worktree.
        Worktrees persist until background eviction removes them (no immediate cleanup).

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            ref: Branch name or commit SHA
            repo_url: Optional Git repository URL (derived if not provided)
            auth_token: Optional authentication token
            is_commit: Whether ref is a commit SHA
            user_id: Optional user ID for multi-tenant tracking

        Returns:
            Path to worktree directory ready for parsing
        """
        self._validate_repo_name(repo_name)
        self._validate_ref(ref)

        self._evict_if_needed(user_id=user_id)

        _ = self.ensure_bare_repo(repo_name, repo_url, auth_token, ref, user_id)

        worktree_path = self.create_worktree(
            repo_name, ref, auth_token, is_commit, user_id
        )

        return str(worktree_path)
