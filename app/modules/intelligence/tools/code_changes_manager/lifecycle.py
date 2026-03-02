"""Manager lifecycle: get/init/reset, project lookup, error extraction."""

import json
import re
from typing import Dict, Optional

from app.modules.utils.logger import setup_logger

from .manager import CodeChangesManager
from .context import (
    get_code_changes_manager_ctx,
    _get_conversation_id,
    _get_local_mode,
    _set_conversation_id,
    _set_agent_id,
    _set_user_id,
    _set_tunnel_url,
    _set_repository,
    _set_branch,
    _set_local_mode,
)

logger = setup_logger(__name__)

# Cache for project_id lookup from conversation_id
_project_id_cache: Dict[str, Optional[str]] = {}


def _get_project_id_from_conversation_id(conversation_id: Optional[str]) -> Optional[str]:
    """Fetch project_id from conversation_id via database lookup (non-local mode only)."""
    if _get_local_mode():
        return None

    if not conversation_id:
        return None

    if conversation_id in _project_id_cache:
        return _project_id_cache[conversation_id]

    try:
        from app.modules.conversations.conversation.conversation_model import Conversation
        from app.core.database import get_db

        db = next(get_db())
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()

        if conversation and conversation.project_ids and len(conversation.project_ids) > 0:
            project_id = conversation.project_ids[0]
            _project_id_cache[conversation_id] = project_id
            logger.info(
                f"CodeChangesManager: Resolved project_id={project_id} from conversation_id={conversation_id}"
            )
            return project_id

        _project_id_cache[conversation_id] = None
        return None
    except Exception as e:
        logger.warning(
            f"CodeChangesManager: Failed to resolve project_id from conversation_id={conversation_id}: {e}"
        )
        return None


def _get_code_changes_manager() -> CodeChangesManager:
    """Get the current code changes manager for this execution context, creating a new one if needed."""
    ctx = get_code_changes_manager_ctx()
    manager = ctx.get()
    conversation_id = _get_conversation_id()

    if manager is not None and manager._conversation_id != conversation_id:
        logger.info(
            f"CodeChangesManager: conversation_id changed from {manager._conversation_id} to {conversation_id}, creating new manager"
        )
        manager = None

    if manager is None:
        logger.info(
            f"CodeChangesManager: Creating new manager instance for conversation_id={conversation_id}"
        )
        manager = CodeChangesManager(conversation_id=conversation_id)
        ctx.set(manager)
        logger.info(
            f"CodeChangesManager: Created new manager with conversation_id={manager._conversation_id}, "
            f"redis_key={manager._redis_key}, existing_changes={len(manager.changes)}"
        )
    return manager


def _init_code_changes_manager(
    conversation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tunnel_url: Optional[str] = None,
    local_mode: bool = False,
    repository: Optional[str] = None,
    branch: Optional[str] = None,
) -> None:
    """Initialize the code changes manager for a new agent run."""
    logger.info(
        f"CodeChangesManager: _init_code_changes_manager called with "
        f"conversation_id={conversation_id}, agent_id={agent_id}, "
        f"user_id={user_id}, tunnel_url={tunnel_url}, local_mode={local_mode}, "
        f"repository={repository}, branch={branch}"
    )
    _set_local_mode(local_mode)
    _set_conversation_id(conversation_id)
    _set_agent_id(agent_id)
    _set_user_id(user_id)
    _set_tunnel_url(tunnel_url)
    _set_repository(repository)
    _set_branch(branch)

    ctx = get_code_changes_manager_ctx()
    old_manager = ctx.get()
    old_conversation_id = old_manager._conversation_id if old_manager else None
    old_count = len(old_manager.changes) if old_manager else 0
    logger.info(
        f"CodeChangesManager: Initializing manager for conversation_id={conversation_id} "
        f"(previous conversation_id: {old_conversation_id}, previous file count: {old_count})"
    )

    new_manager = CodeChangesManager(conversation_id=conversation_id)
    ctx.set(new_manager)

    logger.info(
        f"CodeChangesManager: Initialized with conversation_id={new_manager._conversation_id}, "
        f"loaded {len(new_manager.changes)} existing changes from Redis"
    )


def _reset_code_changes_manager() -> None:
    """Reset the code changes manager. DEPRECATED: Use _init_code_changes_manager(conversation_id) instead."""
    conversation_id = _get_conversation_id()
    _init_code_changes_manager(conversation_id)


def _extract_error_message(error_text: str, status_code: int) -> str:
    """Extract a meaningful error message from response text."""
    if not error_text:
        return f"HTTP {status_code} error (no response body)"

    if error_text.strip().startswith("<!DOCTYPE html>") or error_text.strip().startswith("<html"):
        if status_code == 530 or ("tunnel" in error_text.lower() and "error" in error_text.lower()):
            return "Tunnel/connection error: extension connection unavailable. Please ensure the VS Code extension is running and connected."

        title_match = re.search(r"<title>(.*?)</title>", error_text, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1).strip())
            return f"HTTP {status_code}: {title[:200]}"

        error_match = re.search(
            r"<h[12][^>]*>(.*?)</h[12]>", error_text, re.IGNORECASE | re.DOTALL
        )
        if error_match:
            error_msg = re.sub(r"<[^>]+>", "", error_match.group(1).strip())
            error_msg = re.sub(r"\s+", " ", error_msg)
            return f"HTTP {status_code}: {error_msg[:200]}"

        return f"HTTP {status_code} error (HTML response received)"

    try:
        error_json = json.loads(error_text)
        if isinstance(error_json, dict):
            for key in ["error", "message", "detail", "msg"]:
                if key in error_json:
                    return f"HTTP {status_code}: {str(error_json[key])[:200]}"
            return f"HTTP {status_code}: {str(error_json)[:200]}"
    except (json.JSONDecodeError, ValueError):
        pass

    if len(error_text) > 500:
        return f"HTTP {status_code}: {error_text[:200]}... (truncated)"
    return f"HTTP {status_code}: {error_text}"
