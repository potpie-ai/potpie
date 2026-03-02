"""
Code Changes Manager - Core class for managing code changes in Redis.

Delegates to: storage, content_resolver, diff, git_ops, context.
"""

import json
import os
import re
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import redis
from sqlalchemy.orm import Session

from app.modules.utils.logger import setup_logger
from app.core.config_provider import ConfigProvider

from .constants import (
    CODE_CHANGES_KEY_PREFIX,
    CODE_CHANGES_TTL_SECONDS,
    MAX_FILE_SIZE_BYTES,
)
from .models import ChangeType, FileChange
from .storage import load_changes_from_redis, save_changes_to_redis
from .content_resolver import get_current_content, read_file_from_codebase
from .diff import (
    create_unified_diff,
    generate_git_diff_patch,
    fetch_repo_file_content_for_diff,
)
from .git_ops import (
    write_change_to_worktree,
    commit_file_and_extract_patch as _commit_file_and_extract_patch,
    commit_all_files_and_extract_patches as _commit_all_files_and_extract_patches,
)
from .context import _get_local_mode

logger = setup_logger(__name__)


class CodeChangesManager:
    """Manages code changes in Redis for a conversation, persisting across messages"""

    def __init__(self, conversation_id: Optional[str] = None):
        self._conversation_id = conversation_id
        self._fallback_id = str(uuid.uuid4()) if not conversation_id else None

        config = ConfigProvider()
        self._redis_client = redis.from_url(config.get_redis_url())

        self._changes_cache: Optional[Dict[str, FileChange]] = None

        logger.info(
            f"CodeChangesManager: Initialized with conversation_id={conversation_id}, "
            f"redis_key={self._redis_key}"
        )

    @property
    def _redis_key(self) -> str:
        if self._conversation_id:
            return f"{CODE_CHANGES_KEY_PREFIX}:{self._conversation_id}"
        return f"{CODE_CHANGES_KEY_PREFIX}:session:{self._fallback_id}"

    @property
    def changes(self) -> Dict[str, FileChange]:
        if self._changes_cache is None:
            self._load_from_redis()
        assert self._changes_cache is not None
        return self._changes_cache

    @changes.setter
    def changes(self, value: Dict[str, FileChange]) -> None:
        self._changes_cache = value
        self._save_to_redis()

    def _load_from_redis(self) -> None:
        self._changes_cache = load_changes_from_redis(
            self._redis_client, self._redis_key
        )

    def _save_to_redis(self) -> None:
        if self._changes_cache is None:
            return
        save_changes_to_redis(
            self._redis_client,
            self._redis_key,
            self._changes_cache,
            self._conversation_id,
            CODE_CHANGES_TTL_SECONDS,
        )

    def _persist_change(self) -> None:
        self._save_to_redis()

    def _get_current_content(
        self,
        file_path: str,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> str:
        return get_current_content(
            file_path, self.changes, project_id=project_id, db=db
        )

    def _read_file_from_codebase(self, file_path: str) -> Optional[str]:
        return read_file_from_codebase(file_path)

    def add_file(
        self,
        file_path: str,
        content: str,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> bool:
        logger.info(
            f"CodeChangesManager.add_file: Adding file '{file_path}' (content length: {len(content)} chars)"
        )
        changes = self.changes
        if file_path in changes and changes[file_path].change_type != ChangeType.DELETE:
            logger.warning(
                f"CodeChangesManager.add_file: File '{file_path}' already exists (not deleted)"
            )
            return False

        change = FileChange(
            file_path=file_path,
            change_type=ChangeType.ADD,
            content=content,
            description=description,
        )
        self._changes_cache[file_path] = change
        if project_id and not _get_local_mode():
            write_change_to_worktree(project_id, file_path, "add", content, db)
        self._persist_change()
        logger.info(
            f"CodeChangesManager.add_file: Successfully added file '{file_path}'"
        )
        return True

    def _apply_update(
        self,
        file_path: str,
        new_content: str,
        description: Optional[str] = None,
        preserve_previous: bool = True,
        preserve_change_type: bool = True,
        original_content: Optional[str] = None,
        override_previous_content: Optional[str] = None,
    ) -> bool:
        logger.debug(
            f"CodeChangesManager._apply_update: Applying update to '{file_path}'"
        )
        previous_content = override_previous_content
        if file_path in self.changes:
            existing = self.changes[file_path]

            if existing.change_type == ChangeType.DELETE:
                logger.warning(
                    f"CodeChangesManager._apply_update: File '{file_path}' is marked for deletion."
                )
                return False

            if (
                previous_content is None
                and preserve_previous
                and existing.previous_content
            ):
                previous_content = existing.previous_content
            elif previous_content is None and preserve_previous:
                previous_content = existing.content

            original_change_type = existing.change_type
            if preserve_change_type and original_change_type == ChangeType.ADD:
                new_change_type = ChangeType.ADD
            else:
                new_change_type = ChangeType.UPDATE

            existing.content = new_content
            existing.change_type = new_change_type
            existing.updated_at = datetime.now().isoformat()
            if description:
                existing.description = description
            if previous_content:
                existing.previous_content = previous_content
            self._persist_change()
        else:
            change = FileChange(
                file_path=file_path,
                change_type=ChangeType.UPDATE,
                content=new_content,
                previous_content=(
                    override_previous_content
                    if override_previous_content is not None
                    else (
                        original_content
                        if original_content is not None
                        else previous_content
                    )
                ),
                description=description,
            )
            self._changes_cache[file_path] = change
            self._persist_change()
        return True

    def update_file(
        self,
        file_path: str,
        content: str,
        description: Optional[str] = None,
        preserve_previous: bool = True,
        previous_content: Optional[str] = None,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> bool:
        logger.info(
            f"CodeChangesManager.update_file: Updating file '{file_path}' with full content (content length: {len(content)} chars)"
        )
        if project_id and not _get_local_mode():
            write_change_to_worktree(project_id, file_path, "update", content, db)
        result = self._apply_update(
            file_path,
            content,
            description,
            preserve_previous,
            override_previous_content=previous_content,
        )
        if not result:
            logger.warning(
                f"CodeChangesManager.update_file: Failed to update file '{file_path}' - file may be marked for deletion"
            )
            return False
        logger.info(f"CodeChangesManager.update_file: Successfully updated file '{file_path}'")
        return result

    def update_file_lines(
        self,
        file_path: str,
        start_line: int,
        end_line: Optional[int] = None,
        new_content: str = "",
        description: Optional[str] = None,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        logger.info(
            f"CodeChangesManager.update_file_lines: Updating lines {start_line}-{end_line or start_line} in '{file_path}'"
        )
        try:
            current_content = self._get_current_content(
                file_path, project_id=project_id, db=db
            )
            lines = current_content.split("\n")

            if len(lines) == 1 and lines[0] == "":
                logger.warning(
                    f"CodeChangesManager.update_file_lines: WARNING - File '{file_path}' appears to be empty or not found."
                )

            if start_line < 1 or start_line > len(lines):
                return {
                    "success": False,
                    "error": (
                        f"Invalid start_line {start_line}. File has {len(lines)} lines. "
                        f"If you've performed insert_lines or delete_lines, line numbers have shifted."
                    ),
                }

            if end_line is None:
                end_line = start_line

            if end_line < start_line:
                return {"success": False, "error": f"end_line ({end_line}) must be >= start_line ({start_line})"}

            if end_line > len(lines):
                return {
                    "success": False,
                    "error": f"Invalid end_line {end_line}. File has {len(lines)} lines.",
                }

            start_idx = start_line - 1
            end_idx = end_line

            replaced_lines = lines[start_idx:end_idx]
            replaced_content = "\n".join(replaced_lines)

            new_lines = new_content.split("\n")
            updated_lines = lines[:start_idx] + new_lines + lines[end_idx:]
            updated_content = "\n".join(updated_lines)

            change_desc = description or f"Updated lines {start_line}-{end_line}"
            original_content = (
                current_content if file_path not in self.changes else None
            )
            worktree_ok, worktree_err = False, None
            if project_id and not _get_local_mode():
                worktree_ok, worktree_err = write_change_to_worktree(
                    project_id, file_path, "update", updated_content, db
                )
            update_success = self._apply_update(
                file_path,
                updated_content,
                change_desc,
                original_content=original_content,
            )
            if not update_success:
                return {
                    "success": False,
                    "error": f"Cannot update file '{file_path}': file is marked for deletion.",
                }

            context_lines_before = 3
            context_lines_after = 3
            context_start_idx = max(0, start_idx - context_lines_before)
            context_lines_before_list = lines[context_start_idx:start_idx]
            context_after_start_idx = start_idx + len(new_lines)
            context_after_end_idx = min(
                len(updated_lines), context_after_start_idx + context_lines_after
            )
            context_lines_after_list = updated_lines[
                context_after_start_idx:context_after_end_idx
            ]
            context_with_updated = (
                context_lines_before_list + new_lines + context_lines_after_list
            )
            context_start_line = context_start_idx + 1
            context_end_line = context_after_end_idx

            out = {
                "success": True,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "lines_replaced": len(replaced_lines),
                "lines_added": len(new_lines),
                "replaced_content": replaced_content,
                "updated_context": "\n".join(context_with_updated),
                "context_start_line": context_start_line,
                "context_end_line": context_end_line,
            }
            if project_id and not _get_local_mode():
                out["worktree_write"] = "ok" if worktree_ok else f"failed: {worktree_err or 'unknown'}"
            return out
        except Exception as e:
            logger.exception("CodeChangesManager.update_file_lines: Error")
            return {"success": False, "error": str(e)}

    def replace_in_file(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        logger.info(f"CodeChangesManager.replace_in_file: str_replace in '{file_path}'")
        try:
            current_content = self._get_current_content(
                file_path, project_id=project_id, db=db
            )

            count = current_content.count(old_str)

            if count == 0:
                preview = old_str[:120].replace("\n", "↵")
                return {
                    "success": False,
                    "error": (
                        f"old_str not found in '{file_path}'.\n"
                        f"Searched for: {preview!r}\n"
                        "Check that indentation, whitespace, and line endings match exactly."
                    ),
                }

            if count > 1:
                preview = old_str[:120].replace("\n", "↵")
                return {
                    "success": False,
                    "error": (
                        f"old_str matches {count} times in '{file_path}' — ambiguous replacement.\n"
                        f"Searched for: {preview!r}\n"
                        "Add more surrounding lines to old_str to make it unique."
                    ),
                }

            new_content = current_content.replace(old_str, new_str, 1)

            match_pos = current_content.index(old_str)
            match_line = current_content[:match_pos].count("\n") + 1

            change_desc = description or f"str_replace in '{file_path}' at line ~{match_line}"
            original_content = current_content if file_path not in self.changes else None
            worktree_ok, worktree_err = False, None
            if project_id and not _get_local_mode():
                worktree_ok, worktree_err = write_change_to_worktree(
                    project_id, file_path, "update", new_content, db
                )
            update_success = self._apply_update(
                file_path, new_content, change_desc, original_content=original_content
            )
            if not update_success:
                return {
                    "success": False,
                    "error": f"Cannot replace in file '{file_path}': file is marked for deletion.",
                }

            out = {
                "success": True,
                "file_path": file_path,
                "match_line": match_line,
                "replacements_made": 1,
            }
            if project_id and not _get_local_mode():
                out["worktree_write"] = "ok" if worktree_ok else f"failed: {worktree_err or 'unknown'}"
            return out
        except Exception as e:
            logger.exception("CodeChangesManager.replace_in_file: Error")
            return {"success": False, "error": str(e)}

    def insert_lines(
        self,
        file_path: str,
        line_number: int,
        content: str,
        description: Optional[str] = None,
        insert_after: bool = True,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        position = "after" if insert_after else "before"
        logger.info(
            f"CodeChangesManager.insert_lines: Inserting {len(content.split(chr(10)))} line(s) {position} line {line_number} in '{file_path}'"
        )
        try:
            current_content = self._get_current_content(
                file_path, project_id=project_id, db=db
            )
            lines = current_content.split("\n")

            if len(lines) == 1 and lines[0] == "":
                logger.warning(
                    f"CodeChangesManager.insert_lines: WARNING - File '{file_path}' appears to be empty!"
                )

            if line_number < 1:
                return {"success": False, "error": "line_number must be >= 1"}

            if insert_after:
                max_valid_line = len(lines) + 1
            else:
                max_valid_line = len(lines)

            if line_number > max_valid_line:
                return {
                    "success": False,
                    "error": f"line_number {line_number} is beyond end of file ({len(lines)} lines).",
                }

            new_lines = content.split("\n")

            if insert_after:
                insert_idx = min(line_number, len(lines))
            else:
                insert_idx = max(0, min(line_number - 1, len(lines) - 1))

            updated_lines = lines[:insert_idx] + new_lines + lines[insert_idx:]
            updated_content = "\n".join(updated_lines)

            change_desc = (
                description
                or f"Inserted {len(new_lines)} lines {position} line {line_number}"
            )
            original_content = (
                current_content if file_path not in self.changes else None
            )
            worktree_ok, worktree_err = False, None
            if project_id and not _get_local_mode():
                worktree_ok, worktree_err = write_change_to_worktree(
                    project_id, file_path, "update", updated_content, db
                )
            update_success = self._apply_update(
                file_path,
                updated_content,
                change_desc,
                original_content=original_content,
            )
            if not update_success:
                return {
                    "success": False,
                    "error": f"Cannot insert lines in file '{file_path}': file is marked for deletion.",
                }

            context_lines_before = 3
            context_lines_after = 3
            context_start_idx = max(0, insert_idx - context_lines_before)
            context_lines_before_list = lines[context_start_idx:insert_idx]
            context_after_start_idx = insert_idx + len(new_lines)
            context_after_end_idx = min(
                len(updated_lines), context_after_start_idx + context_lines_after
            )
            context_lines_after_list = updated_lines[
                context_after_start_idx:context_after_end_idx
            ]
            context_with_inserted = (
                context_lines_before_list + new_lines + context_lines_after_list
            )
            context_start_line = context_start_idx + 1
            context_end_line = context_after_end_idx

            out = {
                "success": True,
                "file_path": file_path,
                "line_number": line_number,
                "position": position,
                "lines_inserted": len(new_lines),
                "inserted_context": "\n".join(context_with_inserted),
                "context_start_line": context_start_line,
                "context_end_line": context_end_line,
            }
            if project_id and not _get_local_mode():
                out["worktree_write"] = "ok" if worktree_ok else f"failed: {worktree_err or 'unknown'}"
            return out
        except Exception as e:
            logger.exception("CodeChangesManager.insert_lines: Error")
            return {"success": False, "error": str(e)}

    def delete_lines(
        self,
        file_path: str,
        start_line: int,
        end_line: Optional[int] = None,
        description: Optional[str] = None,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        logger.info(
            f"CodeChangesManager.delete_lines: Deleting lines {start_line}-{end_line or start_line} from '{file_path}'"
        )
        try:
            current_content = self._get_current_content(
                file_path, project_id=project_id, db=db
            )
            lines = current_content.split("\n")

            if len(lines) == 1 and lines[0] == "":
                logger.warning(
                    f"CodeChangesManager.delete_lines: WARNING - File '{file_path}' appears to be empty or not found."
                )

            if start_line < 1 or start_line > len(lines):
                return {"success": False, "error": f"Invalid start_line {start_line}. File has {len(lines)} lines."}

            if end_line is None:
                end_line = start_line

            if end_line < start_line:
                return {"success": False, "error": f"end_line ({end_line}) must be >= start_line ({start_line})"}

            if end_line > len(lines):
                return {"success": False, "error": f"Invalid end_line {end_line}. File has {len(lines)} lines."}

            start_idx = start_line - 1
            end_idx = end_line

            deleted_lines = lines[start_idx:end_idx]
            deleted_content = "\n".join(deleted_lines)

            updated_lines = lines[:start_idx] + lines[end_idx:]
            updated_content = "\n".join(updated_lines)

            change_desc = description or f"Deleted lines {start_line}-{end_line}"
            original_content = (
                current_content if file_path not in self.changes else None
            )
            worktree_ok, worktree_err = False, None
            if project_id and not _get_local_mode():
                worktree_ok, worktree_err = write_change_to_worktree(
                    project_id, file_path, "update", updated_content, db
                )
            update_success = self._apply_update(
                file_path,
                updated_content,
                change_desc,
                original_content=original_content,
            )
            if not update_success:
                return {
                    "success": False,
                    "error": f"Cannot delete lines from file '{file_path}': file is marked for deletion.",
                }

            out = {
                "success": True,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "lines_deleted": len(deleted_lines),
                "deleted_content": deleted_content,
            }
            if project_id and not _get_local_mode():
                out["worktree_write"] = "ok" if worktree_ok else f"failed: {worktree_err or 'unknown'}"
            return out
        except Exception as e:
            logger.exception("CodeChangesManager.delete_lines: Error")
            return {"success": False, "error": str(e)}

    def delete_file(
        self,
        file_path: str,
        description: Optional[str] = None,
        preserve_content: bool = True,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> bool:
        logger.info(
            f"CodeChangesManager.delete_file: Marking file '{file_path}' for deletion (preserve_content={preserve_content})"
        )
        previous_content = None
        changes = self.changes
        if file_path in changes:
            existing = changes[file_path]
            if preserve_content and existing.content:
                previous_content = existing.content
            existing.change_type = ChangeType.DELETE
            existing.content = None
            existing.updated_at = datetime.now().isoformat()
            if description:
                existing.description = description
            if previous_content:
                existing.previous_content = previous_content
            if project_id and not _get_local_mode():
                write_change_to_worktree(project_id, file_path, "delete", None, db)
            self._persist_change()
        else:
            change = FileChange(
                file_path=file_path,
                change_type=ChangeType.DELETE,
                content=None,
                previous_content=previous_content,
                description=description,
            )
            self._changes_cache[file_path] = change
            if project_id and not _get_local_mode():
                write_change_to_worktree(project_id, file_path, "delete", None, db)
            self._persist_change()
        logger.info(f"CodeChangesManager.delete_file: Successfully marked file '{file_path}' for deletion")
        return True

    def get_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        if file_path not in self.changes:
            return None

        change = self.changes[file_path]
        result = asdict(change)
        result["change_type"] = change.change_type.value
        return result

    def list_files(
        self,
        change_type_filter: Optional[ChangeType] = None,
        path_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        files = list(self.changes.values())

        if change_type_filter:
            files = [f for f in files if f.change_type == change_type_filter]

        if path_pattern:
            try:
                pattern = re.compile(path_pattern, re.IGNORECASE)
                files = [f for f in files if pattern.search(f.file_path)]
            except re.error:
                files = [f for f in files if path_pattern.lower() in f.file_path.lower()]

        files.sort(key=lambda x: x.file_path)

        result = []
        for f in files:
            file_dict = asdict(f)
            file_dict["change_type"] = f.change_type.value
            result.append(file_dict)
        return result

    def search_content(
        self,
        pattern: str,
        file_pattern: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> List[Dict[str, Any]]:
        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            content_regex = re.compile(pattern, flags)
            file_regex = re.compile(file_pattern, re.IGNORECASE) if file_pattern else None
        except re.error as e:
            return [{"error": f"Invalid regex pattern: {str(e)}"}]

        for file_path, change in self.changes.items():
            if change.change_type == ChangeType.DELETE or not change.content:
                continue

            if file_regex and not file_regex.search(file_path):
                continue

            lines = change.content.split("\n")
            for line_num, line in enumerate(lines, start=1):
                if content_regex.search(line):
                    matches.append(
                        {
                            "file_path": file_path,
                            "line_number": line_num,
                            "line": line.strip(),
                            "change_type": change.change_type.value,
                        }
                    )

        return matches

    def clear_file(self, file_path: str) -> bool:
        changes = self.changes
        if file_path in changes:
            del self._changes_cache[file_path]
            self._persist_change()
            return True
        return False

    def clear_all(self) -> int:
        changes = self.changes
        count = len(changes)
        self._changes_cache.clear()
        self._persist_change()
        return count

    def get_summary(self) -> Dict[str, Any]:
        change_counts = {ct.value: 0 for ct in ChangeType}
        for change in self.changes.values():
            change_counts[change.change_type.value] += 1

        return {
            "conversation_id": self._conversation_id,
            "total_files": len(self.changes),
            "change_counts": change_counts,
            "files": [
                {
                    "file_path": change.file_path,
                    "change_type": change.change_type.value,
                    "description": change.description,
                    "updated_at": change.updated_at,
                }
                for change in sorted(self.changes.values(), key=lambda x: x.file_path)
            ],
        }

    def serialize(self) -> str:
        data = {
            "conversation_id": self._conversation_id,
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
                for change in self.changes.values()
            ],
        }
        return json.dumps(data, indent=2)

    def deserialize(self, json_str: str) -> bool:
        try:
            data = json.loads(json_str)
            self._conversation_id = data.get("conversation_id")
            if self._conversation_id:
                self._fallback_id = None
            else:
                old_session_id = data.get("session_id")
                self._fallback_id = old_session_id or str(uuid.uuid4())
            self._changes_cache = {}

            for change_data in data.get("changes", []):
                change = FileChange(
                    file_path=change_data["file_path"],
                    change_type=ChangeType(change_data["change_type"]),
                    content=change_data.get("content"),
                    previous_content=change_data.get("previous_content"),
                    created_at=change_data.get("created_at", datetime.now().isoformat()),
                    updated_at=change_data.get("updated_at", datetime.now().isoformat()),
                    description=change_data.get("description"),
                )
                self._changes_cache[change.file_path] = change
            self._persist_change()
            return True
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.exception("CodeChangesManager.deserialize: Error deserializing JSON")
            return False

    def generate_diff(
        self,
        file_path: Optional[str] = None,
        context_lines: int = 3,
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, str]:
        diffs = {}
        files_to_diff = [file_path] if file_path else list(self.changes.keys())

        for fp in files_to_diff:
            if fp not in self.changes:
                continue

            change = self.changes[fp]

            if change.change_type == ChangeType.DELETE:
                old_content = change.previous_content or ""
                new_content = ""
                diff = create_unified_diff(
                    old_content, new_content, fp, fp, context_lines
                )
                diffs[fp] = diff
                continue

            if change.change_type == ChangeType.ADD:
                old_content = ""
                new_content = change.content or ""
                diff = create_unified_diff(
                    old_content, new_content, "/dev/null", fp, context_lines
                )
                diffs[fp] = diff
                continue

            new_content = change.content or ""

            if change.previous_content is not None:
                old_content = change.previous_content
                diff = create_unified_diff(old_content, new_content, fp, fp, context_lines)
            else:
                old_content = None
                if project_id and db:
                    try:
                        from app.modules.code_provider.code_provider_service import CodeProviderService
                        from app.modules.code_provider.git_safe import safe_git_operation, GitOperationError
                        from app.modules.projects.projects_model import Project

                        project = db.query(Project).filter(Project.id == project_id).first()
                        if project:
                            cp_service = CodeProviderService(db)

                            def _fetch_repo_content():
                                return cp_service.get_file_content(
                                    repo_name=project.repo_name,
                                    file_path=fp,
                                    branch_name=project.branch_name,
                                    start_line=None,
                                    end_line=None,
                                    project_id=project_id,
                                    commit_id=project.commit_id,
                                )

                            try:
                                repo_content = safe_git_operation(
                                    _fetch_repo_content,
                                    max_retries=1,
                                    timeout=20.0,
                                    max_total_timeout=25.0,
                                    operation_name=f"generate_diff_get_content({fp})",
                                )
                            except GitOperationError:
                                repo_content = None

                            if repo_content:
                                old_content = repo_content
                    except Exception:
                        old_content = None

                if old_content is None:
                    old_content = self._read_file_from_codebase(fp)

                if old_content is None or old_content == "":
                    old_content = ""
                    diff = create_unified_diff(old_content, new_content, "/dev/null", fp, context_lines)
                else:
                    diff = create_unified_diff(old_content, new_content, fp, fp, context_lines)

            diffs[fp] = diff

        return diffs

    def export_changes(
        self,
        format: str = "dict",
        project_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Union[Dict[str, str], List[Dict[str, Any]], str]:
        if format == "dict":
            return {
                fp: change.content or ""
                for fp, change in self.changes.items()
                if change.change_type != ChangeType.DELETE and change.content
            }
        elif format == "list":
            return [
                {
                    "file_path": change.file_path,
                    "change_type": change.change_type.value,
                    "content": change.content,
                    "description": change.description,
                }
                for change in self.changes.values()
            ]
        elif format == "json":
            return self.serialize()
        elif format == "diff":
            patches = []
            for file_path, change in sorted(self.changes.items()):
                if change.change_type == ChangeType.DELETE:
                    old_content = change.previous_content or ""
                    new_content = ""
                elif change.change_type == ChangeType.ADD:
                    old_content = ""
                    new_content = change.content or ""
                else:
                    new_content = change.content or ""
                    if change.previous_content is not None:
                        old_content = change.previous_content
                    elif project_id and db:
                        old_content = fetch_repo_file_content_for_diff(
                            project_id, file_path, db
                        )
                        old_content = old_content if old_content else ""
                    else:
                        old_content = read_file_from_codebase(file_path) or ""

                patch = generate_git_diff_patch(file_path, old_content, new_content)
                if patch:
                    patches.append(patch)

            return "\n".join(patches)
        else:
            raise ValueError(f"Unknown format: {format}. Supported: 'dict', 'list', 'json', 'diff'")

    def commit_file_and_extract_patch(
        self,
        file_path: str,
        commit_message: str,
        project_id: str,
        db: Optional[Session] = None,
        branch_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _commit_file_and_extract_patch(
            self.changes,
            file_path,
            commit_message,
            project_id,
            db=db,
            branch_name=branch_name,
        )

    def commit_all_files_and_extract_patches(
        self,
        commit_message: str,
        project_id: str,
        db: Optional[Session] = None,
        branch_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _commit_all_files_and_extract_patches(
            self.changes,
            commit_message,
            project_id,
            db=db,
            branch_name=branch_name,
        )
