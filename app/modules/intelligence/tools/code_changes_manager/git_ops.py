"""Git and worktree operations for code changes."""

import os
from typing import Dict, Optional, Any

from sqlalchemy.orm import Session

from app.modules.utils.logger import setup_logger

from .constants import CODE_CHANGES_TTL_SECONDS
from .models import ChangeType, FileChange
from .context import _get_local_mode, _get_conversation_id, _get_user_id

logger = setup_logger(__name__)


def write_change_to_worktree(
    project_id: str,
    file_path: str,
    change_type: str,
    content: Optional[str],
    db: Any,
) -> tuple[bool, Optional[str]]:
    """
    Write a single file change directly to the edits worktree (non-local mode only).

    Best-effort: never raises. Redis remains the source of truth.
    No-op when REPO_MANAGER_ENABLED=false or local_mode=True.

    Args:
        project_id: Project ID used to locate the edits worktree
        file_path: Repo-relative path to write
        change_type: "add", "update", or "delete"
        content: File content for add/update; None for delete
        db: DB session for project lookup and OAuth token retrieval

    Returns:
        (success, error_message)
    """
    try:
        if _get_local_mode():
            logger.debug(
                f"_write_change_to_worktree: skipped (local_mode=True), file_path={file_path}, project_id={project_id}"
            )
            return False, "local_mode"

        if not project_id:
            return False, "no_project_id"

        conversation_id = _get_conversation_id()
        if not conversation_id:
            logger.warning(
                f"_write_change_to_worktree: skipped (no conversation_id), file_path={file_path}, project_id={project_id}"
            )
            return False, "no_conversation_id"

        user_id = _get_user_id()

        from app.modules.repo_manager import RepoManager
        from app.modules.repo_manager.sync_helper import get_or_create_edits_worktree_path

        repo_manager = RepoManager()
        worktree_path, failure_reason = get_or_create_edits_worktree_path(
            repo_manager,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            sql_db=db,
        )

        if not worktree_path:
            logger.warning(
                f"_write_change_to_worktree: Could not get edits worktree: {failure_reason}, "
                f"project_id={project_id}, conversation_id={conversation_id}"
            )
            return False, f"worktree_unavailable:{failure_reason}"

        worktree_abs = os.path.abspath(worktree_path)
        full_path = os.path.abspath(os.path.join(worktree_path, file_path))

        try:
            common = os.path.commonpath([worktree_abs, full_path])
            if common != worktree_abs:
                logger.error(
                    f"_write_change_to_worktree: Path traversal blocked for '{file_path}', project_id={project_id}"
                )
                return False, "path_traversal"
        except ValueError:
            return False, "path_traversal"

        if change_type == "delete":
            if os.path.exists(full_path):
                os.remove(full_path)
            logger.info(
                f"_write_change_to_worktree: Deleted '{file_path}' from edits worktree, project_id={project_id}"
            )
            return True, None
        else:
            if content is None:
                return False, "no_content"
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(
                f"_write_change_to_worktree: Wrote '{file_path}' to edits worktree, project_id={project_id}"
            )
            return True, None

    except Exception as e:
        logger.warning(
            f"_write_change_to_worktree: Error writing '{file_path}': {e}, "
            f"project_id={project_id if project_id else 'unknown'}"
        )
        return False, str(e)


def commit_file_and_extract_patch(
    changes: Dict[str, FileChange],
    file_path: str,
    commit_message: str,
    project_id: str,
    db: Optional[Session] = None,
    branch_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Write file to worktree, commit it, extract unified diff patch, and store in Redis.

    Returns:
        Dict with success, commit_hash, patch, error
    """
    try:
        if _get_local_mode():
            return {
                "success": False,
                "error": "commit_file_and_extract_patch is not available in local mode",
            }

        if os.getenv("REPO_MANAGER_ENABLED", "false").lower() != "true":
            return {"success": False, "error": "REPO_MANAGER_ENABLED is not set to true"}

        conversation_id = _get_conversation_id()
        if not conversation_id:
            return {"success": False, "error": "No conversation_id set"}

        user_id = _get_user_id()

        from app.modules.repo_manager import RepoManager
        from app.modules.repo_manager.sync_helper import get_or_create_edits_worktree_path
        from app.modules.code_provider.git_safe import safe_git_repo_operation
        import redis
        from app.core.config_provider import ConfigProvider

        repo_manager = RepoManager()
        worktree_path, failure_reason = get_or_create_edits_worktree_path(
            repo_manager,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            sql_db=db,
        )

        if not worktree_path:
            return {"success": False, "error": f"Could not get edits worktree: {failure_reason}"}

        if not branch_name:
            sanitized_conv_id = conversation_id.replace("/", "-").replace("\\", "-").replace(" ", "-")
            branch_name = f"agent/edits-{sanitized_conv_id}"

        if file_path not in changes:
            return {"success": False, "error": f"File '{file_path}' not found in changes"}

        change = changes[file_path]
        if change.change_type == ChangeType.DELETE:
            content = None
        else:
            content = change.content

        worktree_abs = os.path.abspath(worktree_path)
        full_path = os.path.abspath(os.path.join(worktree_path, file_path))

        try:
            common = os.path.commonpath([worktree_abs, full_path])
            if common != worktree_abs:
                return {"success": False, "error": f"Path traversal blocked for '{file_path}'"}
        except ValueError:
            return {"success": False, "error": f"Path traversal blocked for '{file_path}'"}

        if change.change_type == ChangeType.DELETE:
            if os.path.exists(full_path):
                os.remove(full_path)
        else:
            if content is None:
                return {"success": False, "error": f"No content for file '{file_path}'"}
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        def _git_operation(repo):
            existing_branch = next((b for b in repo.branches if b.name == branch_name), None)
            if existing_branch is not None:
                try:
                    if repo.active_branch.name != branch_name:
                        existing_branch.checkout()
                except TypeError:
                    existing_branch.checkout()
            else:
                new_branch = repo.create_head(branch_name)
                new_branch.checkout()

            repo.git.add("-A", "--", file_path)
            diff = repo.git.diff("--cached", "--name-only")
            if not diff.strip():
                return {"success": False, "error": "No changes to commit"}

            commit = repo.index.commit(commit_message)
            return {"success": True, "commit_hash": commit.hexsha, "branch": branch_name}

        git_result = safe_git_repo_operation(
            worktree_path,
            _git_operation,
            operation_name="commit_file",
            timeout=30.0,
        )

        if not git_result.get("success"):
            return git_result

        commit_hash = git_result["commit_hash"]

        def _extract_patch(repo):
            try:
                patch = repo.git.show(
                    "--pretty=format:",
                    "--patch",
                    commit_hash,
                    "--",
                    file_path,
                )
                return patch if patch and patch.strip() else None
            except Exception as e:
                logger.warning(f"Failed to extract patch for {file_path}: {e}")
                return None

        patch = safe_git_repo_operation(
            worktree_path,
            _extract_patch,
            operation_name="extract_patch",
            timeout=10.0,
        )

        if patch:
            try:
                config = ConfigProvider()
                client = redis.from_url(config.get_redis_url())
                safe_file_path = file_path.replace("/", ":")
                key = f"pr_patches:{conversation_id}:{safe_file_path}"
                client.setex(key, CODE_CHANGES_TTL_SECONDS, patch)
            except Exception as e:
                logger.warning(f"Failed to store patch for {file_path}: {e}")

        return {"success": True, "commit_hash": commit_hash, "patch": patch}

    except Exception as e:
        logger.exception("git_ops.commit_file_and_extract_patch: Error")
        return {"success": False, "error": str(e)}


def commit_all_files_and_extract_patches(
    changes: Dict[str, FileChange],
    commit_message: str,
    project_id: str,
    db: Optional[Session] = None,
    branch_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Commit all files in changes to worktree and extract individual patches."""
    try:
        if _get_local_mode():
            return {
                "success": False,
                "error": "commit_all_files_and_extract_patches is not available in local mode",
            }

        if os.getenv("REPO_MANAGER_ENABLED", "false").lower() != "true":
            return {"success": False, "error": "REPO_MANAGER_ENABLED is not set to true"}

        conversation_id = _get_conversation_id()
        if not conversation_id:
            return {"success": False, "error": "No conversation_id set"}

        if not changes:
            return {"success": False, "error": "No changes to commit"}

        user_id = _get_user_id()

        from app.modules.repo_manager import RepoManager
        from app.modules.repo_manager.sync_helper import get_or_create_edits_worktree_path
        from app.modules.code_provider.git_safe import safe_git_repo_operation
        import redis
        from app.core.config_provider import ConfigProvider

        repo_manager = RepoManager()
        worktree_path, failure_reason = get_or_create_edits_worktree_path(
            repo_manager,
            project_id=project_id,
            conversation_id=conversation_id,
            user_id=user_id,
            sql_db=db,
        )

        if not worktree_path:
            return {"success": False, "error": f"Could not get edits worktree: {failure_reason}"}

        if not branch_name:
            sanitized_conv_id = conversation_id.replace("/", "-").replace("\\", "-").replace(" ", "-")
            branch_name = f"agent/edits-{sanitized_conv_id}"

        worktree_abs = os.path.abspath(worktree_path)
        files_to_stage = []

        for fpath, change in changes.items():
            full_path = os.path.abspath(os.path.join(worktree_path, fpath))

            try:
                common = os.path.commonpath([worktree_abs, full_path])
                if common != worktree_abs:
                    logger.warning(f"Path traversal blocked for '{fpath}'")
                    continue
            except ValueError:
                logger.warning(f"Path traversal blocked for '{fpath}'")
                continue

            if change.change_type == ChangeType.DELETE:
                if os.path.exists(full_path):
                    os.remove(full_path)
                files_to_stage.append(fpath)
            else:
                if change.content is not None:
                    parent_dir = os.path.dirname(full_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(change.content)
                    files_to_stage.append(fpath)

        if not files_to_stage:
            return {"success": False, "error": "No files to commit"}

        def _git_operation(repo):
            existing_branch = next((b for b in repo.branches if b.name == branch_name), None)
            if existing_branch is not None:
                try:
                    if repo.active_branch.name != branch_name:
                        existing_branch.checkout()
                except TypeError:
                    existing_branch.checkout()
            else:
                new_branch = repo.create_head(branch_name)
                new_branch.checkout()

            repo.git.add("-A", "--", *files_to_stage)
            diff = repo.git.diff("--cached", "--name-only")
            if not diff.strip():
                return {"success": False, "error": "No changes to commit"}

            commit = repo.index.commit(commit_message)
            return {
                "success": True,
                "commit_hash": commit.hexsha,
                "branch": branch_name,
                "files_committed": files_to_stage,
            }

        git_result = safe_git_repo_operation(
            worktree_path,
            _git_operation,
            operation_name="commit_all_files",
            timeout=60.0,
        )

        if not git_result.get("success"):
            return git_result

        commit_hash = git_result["commit_hash"]
        patches = {}

        for fpath in files_to_stage:
            def _extract_patch(repo, fp=fpath):
                try:
                    patch = repo.git.show(
                        "--pretty=format:",
                        "--patch",
                        commit_hash,
                        "--",
                        fp,
                    )
                    return patch if patch and patch.strip() else None
                except Exception as e:
                    logger.warning(f"Failed to extract patch for {fp}: {e}")
                    return None

            patch = safe_git_repo_operation(
                worktree_path,
                _extract_patch,
                operation_name=f"extract_patch_{fpath}",
                timeout=10.0,
            )

            if patch:
                patches[fpath] = patch
                try:
                    config = ConfigProvider()
                    client = redis.from_url(config.get_redis_url())
                    safe_file_path = fpath.replace("/", ":")
                    key = f"pr_patches:{conversation_id}:{safe_file_path}"
                    client.setex(key, CODE_CHANGES_TTL_SECONDS, patch)
                except Exception as e:
                    logger.warning(f"Failed to store patch for {fpath}: {e}")

        return {
            "success": True,
            "commit_hash": commit_hash,
            "branch": branch_name,
            "patches": patches,
            "files_committed": files_to_stage,
        }

    except Exception as e:
        logger.exception("git_ops.commit_all_files_and_extract_patches: Error")
        return {"success": False, "error": str(e)}
