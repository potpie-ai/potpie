"""Routing logic for LocalServer tunnel (sync file operations)."""

from typing import Dict, Any, Optional
from urllib.parse import quote as url_quote

from app.modules.utils.logger import setup_logger

from .context import (
    _get_user_id,
    _get_conversation_id,
    _get_repository,
    _get_branch,
    _get_tunnel_url,
    _get_agent_id,
)
from .lifecycle import _get_code_changes_manager, _extract_error_message
from .diff import create_unified_diff
from .models import ChangeType

logger = setup_logger(__name__)


def _append_line_stats(
    msg: str, res: Dict[str, Any], operation: Optional[str] = None
) -> str:
    """Append line-change stats from LocalServer so the agent can verify edits."""
    lines_changed = res.get("lines_changed")
    lines_added = res.get("lines_added")
    lines_deleted = res.get("lines_deleted")
    if (
        lines_changed is not None
        or lines_added is not None
        or lines_deleted is not None
    ):
        parts = []
        if lines_changed is not None:
            parts.append(f"lines_changed={lines_changed}")
        if lines_added is not None:
            parts.append(f"lines_added={lines_added}")
        if lines_deleted is not None:
            parts.append(f"lines_deleted={lines_deleted}")
        msg += "\n\n**Line stats:** " + ", ".join(parts)
        # Skip "Many lines were deleted" warning for add_file (new file creation): lines_deleted
        # is meaningless there (should be 0); extension may return incorrect stats for new files.
        lines_deleted_val = lines_deleted if lines_deleted is not None else 0
        if lines_deleted_val > 15 and operation != "add_file":
            msg += (
                f"\n\n⚠️ **Many lines were deleted ({lines_deleted_val} lines).** "
                "Double-check with get_file_from_changes that the file content is correct. "
                "Do NOT use placeholders like '... rest of file unchanged ...' in content—they are written literally and remove real code. "
                "If this was unintended, use revert_file then re-apply with full content or targeted edits."
            )
        msg += (
            "\n\nIf these numbers don't match what you intended (e.g. you expected to delete lines but lines_deleted=0), "
            "use get_file_from_changes to verify the file and fix with revert_file or a corrected edit."
        )
    return msg


def _append_diff(
    msg: str, res: Dict[str, Any], operation: Optional[str] = None
) -> str:
    """Append tunnel line stats and diff to response so agent can review/fix changes."""
    msg = _append_line_stats(msg, res, operation)
    raw_diff = res.get("diff")
    diff = (
        raw_diff.strip()
        if isinstance(raw_diff, str)
        else (str(raw_diff).strip() if raw_diff is not None else "")
    )
    if diff:
        msg += "\n\n**Diff (uncommitted changes):**\n```diff\n" + diff + "\n```"
    return msg


def _route_to_local_server(
    operation: str,
    data: Dict[str, Any],
) -> Optional[str]:
    """Route file operation to LocalServer via tunnel (sync version).

    Returns:
        Result string if successful, None if should fall back to CodeChangesManager
    """

    try:
        from app.modules.tunnel.tunnel_service import get_tunnel_service
        import httpx

        user_id = _get_user_id()
        conversation_id = _get_conversation_id()
        repository = _get_repository()
        branch = _get_branch()

        if not user_id:
            logger.debug("No user_id in context, skipping tunnel routing")
            return None

        tunnel_service = get_tunnel_service()
        # Resolve tunnel by workspace (repo + branch) when available; else conversation then user
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id,
            conversation_id,
            tunnel_url=_get_tunnel_url(),
            repository=repository,
            branch=branch,
        )

        if not tunnel_url:
            logger.debug(
                f"No tunnel available for user {user_id}, using CodeChangesManager"
            )
            return None

        # Map operation to LocalServer endpoint (must be defined before smart routing)
        endpoint_map = {
            "add_file": "/api/files/create",
            "update_file": "/api/files/update",
            "update_file_lines": "/api/files/update-lines",
            "insert_lines": "/api/files/insert-lines",
            "delete_lines": "/api/files/delete-lines",
            "delete_file": "/api/files/delete",
            "replace_in_file": "/api/files/replace",
            "get_file": "/api/files/read",
            "show_updated_file": "/api/files/read",
            "revert_file": "/api/files/revert",
        }

        endpoint = endpoint_map.get(operation)
        if not endpoint:
            logger.warning(f"Unknown operation for tunnel routing: {operation}")
            return None

        # Prepare request data
        request_data = {
            **data,
            "conversation_id": conversation_id,
        }

        route_result: Optional[Dict[str, Any]] = None
        # Socket path: tunnel_url is socket://{workspace_id}; use Socket.IO RPC instead of HTTP
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
            SOCKET_TUNNEL_PREFIX,
            _execute_via_socket,
        )

        if tunnel_url and tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
            logger.info(
                f"[Tunnel Routing] 🚀 Routing {operation} to LocalServer via Socket.IO (timeout=120s)"
            )
            route_result = _execute_via_socket(
                user_id=user_id,
                conversation_id=conversation_id,
                endpoint=endpoint,
                payload=request_data,
                tunnel_url=tunnel_url,
                repository=repository,
                branch=branch,
                timeout=120.0,
            )
            if route_result is None:
                return None
        else:
            # HTTP path: Smart routing (direct localhost when VSCODE_LOCAL_TUNNEL_SERVER set, else tunnel URL)
            try:
                import os

                force_tunnel = os.getenv("FORCE_TUNNEL", "").lower() in ["true", "1", "yes"]
                from app.modules.tunnel.tunnel_service import (
                    _get_local_tunnel_server_url,
                )

                direct_url = _get_local_tunnel_server_url()
                if not force_tunnel and direct_url:
                    try:
                        test_client = httpx.Client(timeout=2.0)
                        health_check = test_client.get(f"{direct_url}/health")
                        test_client.close()
                        if health_check.status_code == 200:
                            logger.info(
                                f"[Tunnel Routing] 🏠 VSCODE_LOCAL_TUNNEL_SERVER set, using direct connection: {direct_url} "
                                f"(bypassing tunnel {tunnel_url})"
                            )
                            url = f"{direct_url}{endpoint}"
                        else:
                            logger.warning(
                                f"[Tunnel Routing] ⚠️ LocalServer not responding on {direct_url}, falling back to tunnel"
                            )
                            url = f"{tunnel_url}{endpoint}"
                    except Exception as e:
                        logger.warning(
                            f"[Tunnel Routing] ⚠️ Cannot reach LocalServer on {direct_url}: {e}, using tunnel instead"
                        )
                        url = f"{tunnel_url}{endpoint}"
                else:
                    if force_tunnel:
                        logger.info(
                            f"[Tunnel Routing] 🔧 FORCE_TUNNEL enabled, using tunnel URL: {tunnel_url}{endpoint}"
                        )
                    else:
                        logger.info(
                            f"[Tunnel Routing] 🌐 Using tunnel URL: {tunnel_url}{endpoint}"
                        )
                    url = f"{tunnel_url}{endpoint}"
            except Exception as e:
                logger.warning(
                    f"[Tunnel Routing] Error in smart routing, falling back to tunnel URL: {e}"
                )
                url = f"{tunnel_url}{endpoint}"

            logger.info(f"[Tunnel Routing] 🚀 Routing {operation} to LocalServer: {url}")
            logger.debug(f"[Tunnel Routing] Request data: {request_data}")

            is_tunnel_request = url.startswith("https://") or (
                url.startswith("http://") and "localhost" not in url
            )
            request_timeout = (
                120.0 if is_tunnel_request else 30.0
            )  # 2 minutes for tunnel, 30s for localhost
            logger.debug(
                f"[Tunnel Routing] Using timeout: {request_timeout}s (tunnel={is_tunnel_request})"
            )

            with httpx.Client(timeout=request_timeout) as client:
                try:
                    result: Dict[str, Any] = {}
                    # Read operations use GET, write operations use POST
                    if operation in ["get_file", "show_updated_file"]:
                        # GET request with file path as query parameter
                        file_path = data.get("file_path") or data.get("path", "")
                        if not file_path:
                            logger.warning(
                                f"[Tunnel Routing] No file_path provided for {operation}"
                            )
                            return None
                        url_with_params = f"{url}?path={url_quote(file_path)}"
                        response = client.get(url_with_params)
                    else:
                        # POST request for write operations
                        response = client.post(
                            url,
                            json=request_data,
                            headers={"Content-Type": "application/json"},
                        )

                    if response.status_code == 200:
                        route_result = response.json()
                        logger.info(
                            f"[Tunnel Routing] ✅ LocalServer {operation} succeeded: {route_result}"
                        )
                    else:
                        # Non-200: error handling and optional tunnel invalidation
                        error_text = response.text
                        status_code = response.status_code
                        is_tunnel_error = (
                            status_code in [502, 503, 504, 530]
                            or ("tunnel" in error_text.lower() and "error" in error_text.lower())
                        )
                        if is_tunnel_error and conversation_id:
                            try:
                                from app.modules.tunnel.tunnel_service import get_tunnel_service
                                get_tunnel_service().unregister_tunnel(user_id, conversation_id)
                                logger.info(
                                    f"[Tunnel Routing] ✅ Invalidated stale conversation tunnel for user {user_id}"
                                )
                            except Exception as e:
                                logger.error(f"[Tunnel Routing] Failed to invalidate tunnel: {e}")
                            logger.info(f"[Tunnel Routing] ⬇️ Falling back to cloud execution for {operation}")
                            return None
                        logger.warning(
                            f"[Tunnel Routing] ❌ LocalServer {operation} failed ({status_code}): {error_text[:200]}"
                        )
                        if status_code == 409:
                            file_path = data.get("file_path", "file")
                            if operation == "add_file":
                                return (
                                    f"❌ Cannot create file '{file_path}': File already exists locally.\n\n"
                                    f"**Recommendation**: Use `update_file_in_changes` or `update_file_lines` to modify the existing file instead of `add_file_to_changes`.\n\n"
                                    f"**Action**: If you intended to replace the file, use `update_file_in_changes` with the new content."
                                )
                            return f"❌ Operation failed: File '{file_path}' already exists (409). Please use update operation instead."
                        if status_code == 400:
                            try:
                                error_data = response.json()
                                if error_data.get("error") == "pre_validation_failed":
                                    errors = error_data.get("errors", [])
                                    error_count = len(errors)
                                    file_path = data.get("file_path", "file")
                                    error_msg = (
                                        f"❌ Pre-validation failed for '{file_path}': {error_count} syntax error(s) detected.\n\n"
                                        f"Review the generated code for unmatched brackets, quotes, or incomplete snippets."
                                    )
                                    return error_msg
                            except Exception:
                                pass
                        return None

                except Exception as e:
                    # Handle specific httpx exceptions if available
                    error_type = type(e).__name__
                    if "Timeout" in error_type or "timeout" in str(e).lower():
                        logger.error(
                            f"[Tunnel Routing] ⏱️ Timeout routing {operation} to LocalServer after {request_timeout}s: {e}. "
                            f"URL: {url}. This may indicate the tunnel is not connected or LocalServer is not responding."
                        )
                    elif "Connect" in error_type or "connection" in str(e).lower():
                        logger.warning(
                            f"[Tunnel Routing] 🔌 Connection error routing {operation} to LocalServer: {e}"
                        )
                    else:
                        resp = getattr(e, "response", None)
                        if resp is not None:
                            error_message = _extract_error_message(
                                getattr(resp, "text", str(e)),
                                getattr(resp, "status_code", 0),
                            )
                            logger.warning(
                                f"[Tunnel Routing] ❌ HTTP error routing {operation} to LocalServer: {error_message}"
                            )
                        else:
                            logger.warning(
                                f"[Tunnel Routing] ❌ Error routing {operation} to LocalServer: {e}"
                            )
                    return None  # Fall back to CodeChangesManager

        if route_result:
            result = route_result
            file_path = data.get("file_path") or data.get("path", "file")

            if operation == "replace_in_file":
                replacements_made = result.get("replacements_made", 0)
                total_matches = result.get("total_matches", replacements_made)
                pattern = data.get("pattern", "pattern")

                # Enrich result with diff and line stats if LocalServer didn't return them
                if not result.get("diff") or (
                    result.get("lines_changed") is None
                    and result.get("lines_added") is None
                    and result.get("lines_deleted") is None
                ):
                    try:
                        manager = _get_code_changes_manager()
                        file_data = manager.get_file(file_path)
                        before = (
                            (file_data.get("content") or "")
                            if file_data
                            else None
                        )
                        after = _fetch_file_content_from_local_server(file_path)
                        if before is not None and after is not None:
                            if not result.get("diff"):
                                result["diff"] = create_unified_diff(
                                    before,
                                    after,
                                    file_path,
                                    file_path,
                                    3,
                                )
                            if (
                                result.get("lines_changed") is None
                                and result.get("lines_added") is None
                                and result.get("lines_deleted") is None
                            ):
                                old_lines = len(before.splitlines())
                                new_lines = len(after.splitlines())
                                result["lines_added"] = max(
                                    0, new_lines - old_lines
                                )
                                result["lines_deleted"] = max(
                                    0, old_lines - new_lines
                                )
                    except Exception as e:
                        logger.debug(
                            f"[Tunnel Routing] Could not enrich replace_in_file diff: {e}"
                        )

                response_msg = (
                    f"✅ Replaced pattern '{pattern}' in '{file_path}'\n\n"
                    + f"Made {replacements_made} replacement(s) out of {total_matches} match(es)"
                )
                if result.get("auto_fixed"):
                    response_msg += "\n\n✅ Auto-fixed formatting issues"
                if result.get("errors"):
                    response_msg += (
                        f"\n⚠️ Validation errors: {len(result['errors'])}"
                    )
                return _append_diff(response_msg, result, operation)
            elif operation == "update_file_lines":
                start_line = data.get("start_line", 0)
                end_line = data.get("end_line", start_line)
                has_errors = result.get("errors") or result.get(
                    "auto_fix_failed"
                )
                if has_errors:
                    response_msg = (
                        f"⚠️ Updated lines {start_line}-{end_line} in '{file_path}' locally, "
                        f"but linter reported issues (change may be partial or reverted).\n\n"
                    )
                else:
                    response_msg = (
                        f"✅ Updated lines {start_line}-{end_line} in '{file_path}' locally\n\n"
                        + "Changes applied successfully in your IDE."
                    )
                if result.get("auto_fixed"):
                    response_msg += "\n\n✅ Auto-fixed formatting issues"
                if result.get("errors"):
                    response_msg += (
                        f"\n⚠️ Validation errors: {len(result['errors'])}"
                    )
                return _append_diff(response_msg, result, operation)
            elif operation == "add_file":
                response_msg = f"✅ Created file '{file_path}' locally\n\nChanges applied successfully in your IDE."
                if result.get("auto_fixed"):
                    response_msg += "\n\n✅ Auto-fixed formatting issues"
                if result.get("errors"):
                    response_msg += (
                        f"\n⚠️ Validation errors: {len(result['errors'])}"
                    )
                return _append_diff(response_msg, result, operation)
            elif operation == "update_file":
                response_msg = f"✅ Updated file '{file_path}' locally\n\nChanges applied successfully in your IDE."
                if result.get("auto_fixed"):
                    response_msg += "\n\n✅ Auto-fixed formatting issues"
                if result.get("errors"):
                    response_msg += (
                        f"\n⚠️ Validation errors: {len(result['errors'])}"
                    )
                return _append_diff(response_msg, result, operation)
            elif operation == "insert_lines":
                line_number = data.get("line_number", 0)
                position = (
                    "after" if data.get("insert_after", True) else "before"
                )
                has_errors = result.get("errors") or result.get(
                    "auto_fix_failed"
                )
                if has_errors:
                    response_msg = (
                        f"⚠️ Inserted lines {position} line {line_number} in '{file_path}' locally, "
                        f"but linter reported issues (change may be partial or reverted).\n\n"
                    )
                else:
                    response_msg = (
                        f"✅ Inserted lines {position} line {line_number} in '{file_path}' locally\n\n"
                        + "Changes applied successfully in your IDE."
                    )
                if result.get("auto_fixed"):
                    response_msg += "\n\n✅ Auto-fixed formatting issues"
                if result.get("errors"):
                    response_msg += (
                        f"\n⚠️ Validation errors: {len(result['errors'])}"
                    )
                return _append_diff(response_msg, result, operation)
            elif operation == "delete_lines":
                start_line = data.get("start_line", 0)
                end_line = data.get("end_line", start_line)
                response_msg = (
                    f"✅ Deleted lines {start_line}-{end_line} from '{file_path}' locally\n\n"
                    + "Changes applied successfully in your IDE."
                )
                return _append_diff(response_msg, result, operation)
            elif operation == "delete_file":
                response_msg = (
                    f"✅ Deleted file '{file_path}' locally\n\n"
                    + "File removed successfully from your IDE."
                )
                return _append_diff(response_msg, result, operation)
            elif operation == "revert_file":
                target = data.get("target", "saved")
                target_label = (
                    "last saved version" if target == "saved" else "git HEAD"
                )
                response_msg = (
                    f"✅ Reverted file '{file_path}' to {target_label}\n\n"
                    + "Content applied in your IDE."
                )
                if result.get("auto_fixed"):
                    response_msg += "\n\n✅ Auto-fixed formatting issues"
                if result.get("errors"):
                    response_msg += (
                        f"\n⚠️ Validation errors: {len(result['errors'])}"
                    )
                return _append_diff(response_msg, result, operation)
            elif operation in ["get_file", "show_updated_file"]:
                content = result.get("content", "")
                line_count = result.get("line_count", 0)

                if operation == "get_file":
                    result_msg = f"📄 **{file_path}**\n\n"
                    result_msg += f"**Current Lines:** {line_count}\n"
                    result_msg += f"**Current Size:** {len(content)} chars\n"
                    content_preview = content[:500]
                    result_msg += f"\n**Content preview (first 500 chars):**\n```\n{content_preview}\n```\n"
                    if len(content) > 500:
                        result_msg += (
                            f"\n... ({len(content) - 500} more characters)\n"
                        )
                    return result_msg
                else:  # show_updated_file
                    result_msg = (
                        f"\n\n---\n\n## 📝 **Updated File: {file_path}**\n\n"
                    )
                    result_msg += f"```\n{content}\n```\n\n"
                    result_msg += "---\n\n"
                    return result_msg

            else:
                response_msg = f"✅ Applied {operation.replace('_', ' ')} to '{file_path}' locally"
                return _append_diff(response_msg, result, operation)

    except Exception as e:
        # Outer exception handler for non-httpx errors
        logger.warning(
            f"[Tunnel Routing] Unexpected error in _route_to_local_server: {e}"
        )
        return None


def _should_route_to_local_server() -> bool:
    """Check if file operations should be routed to LocalServer.

    Returns True if:
    - Agent ID is "code", "code_generation_agent", or "codebase_qna_agent" (when tunnel is available)
    - Tunnel is available for the user

    Note: "code_generation_agent" is used for the "code" agent type in the extension
    since it has all the file editing tools. We route it to tunnel for local-first execution.
    """
    agent_id = _get_agent_id()
    user_id = _get_user_id()
    conversation_id = _get_conversation_id()
    repository = _get_repository()
    branch = _get_branch()

    logger.info(
        f"[Tunnel Routing] Checking routing: agent_id={agent_id}, user_id={user_id}, conversation_id={conversation_id}"
    )

    # Route these agents to tunnel when available for local-first code changes
    if agent_id not in ["code", "code_generation_agent", "codebase_qna_agent"]:
        logger.debug(
            f"[Tunnel Routing] Agent {agent_id} not eligible for tunnel routing"
        )
        return False

    try:
        from app.modules.tunnel.tunnel_service import get_tunnel_service

        if not user_id:
            logger.debug("[Tunnel Routing] No user_id in context")
            return False

        tunnel_service = get_tunnel_service()

        # Resolve tunnel by workspace (repository) or conversation
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id,
            conversation_id,
            repository=repository,
            branch=branch,
        )

        if tunnel_url:
            logger.info(
                f"[Tunnel Routing] ✅ Routing to LocalServer via tunnel: {tunnel_url}"
            )
        else:
            # Debug: Check what tunnels exist
            logger.warning(
                f"[Tunnel Routing] ❌ No tunnel available for user {user_id}, conversation {conversation_id}. "
                f"Agent: {agent_id}. Check if tunnel was registered."
            )

        return tunnel_url is not None
    except Exception as e:
        logger.exception(f"[Tunnel Routing] Error checking tunnel: {e}")
        return False


def _get_local_server_base_url_for_files() -> Optional[str]:
    """Return the base URL for LocalServer file API (direct or tunnel).
    Used when recording local changes in Redis after a successful local write.
    """
    try:
        from app.modules.tunnel.tunnel_service import get_tunnel_service
        import httpx
        import os

        user_id = _get_user_id()
        conversation_id = _get_conversation_id()
        repository = _get_repository()
        branch = _get_branch()
        if not user_id:
            return None
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id, conversation_id, repository=repository, branch=branch
        )
        if not tunnel_url:
            tunnel_url = tunnel_service.get_tunnel_url(user_id, None, repository=repository, branch=branch)
        if not tunnel_url:
            return None
        # Socket URL cannot be used as HTTP base; callers use this for httpx
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import SOCKET_TUNNEL_PREFIX
        if tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
            return None

        force_tunnel = os.getenv("FORCE_TUNNEL", "").lower() in ["true", "1", "yes"]
        base_url = os.getenv("BASE_URL", "").lower()
        environment = os.getenv("ENVIRONMENT", "").lower()
        is_local_backend = not force_tunnel and (
            "localhost" in base_url
            or "127.0.0.1" in base_url
            or environment in ["local", "dev", "development"]
            or not base_url
        )
        if is_local_backend and not force_tunnel:
            tunnel_data = tunnel_service._get_tunnel_data(
                tunnel_service._get_tunnel_key(user_id, conversation_id)
            )
            local_port = 3001
            if tunnel_data and tunnel_data.get("local_port"):
                local_port = int(tunnel_data["local_port"])
            direct_url = f"http://localhost:{local_port}"
            try:
                test_client = httpx.Client(timeout=2.0)
                health_check = test_client.get(f"{direct_url}/health")
                test_client.close()
                if health_check.status_code == 200:
                    return direct_url
            except Exception:
                pass
        return tunnel_url
    except Exception as e:
        logger.debug(f"Failed to get LocalServer base URL for files: {e}")
        return None


def _fetch_file_content_from_local_server(file_path: str) -> Optional[str]:
    """Fetch current file content from LocalServer via tunnel or Socket.IO. Used to sync Redis after line-based local writes."""
    base = _get_local_server_base_url_for_files()
    if not base:
        # Try socket path when no HTTP base (e.g. Socket.IO tunnel)
        try:
            from app.modules.tunnel.tunnel_service import get_tunnel_service
            from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
                _execute_via_socket,
                SOCKET_TUNNEL_PREFIX,
            )
            user_id = _get_user_id()
            conversation_id = _get_conversation_id()
            repository = _get_repository()
            branch = _get_branch()
            if not user_id:
                return None
            tunnel_service = get_tunnel_service()
            tunnel_url = tunnel_service.get_tunnel_url(
                user_id, conversation_id, repository=repository, branch=branch
            )
            if tunnel_url and tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
                out = _execute_via_socket(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    endpoint="/api/files/read",
                    payload={"path": file_path},
                    tunnel_url=tunnel_url,
                    repository=repository,
                    branch=branch,
                    timeout=15.0,
                )
                if isinstance(out, dict) and out.get("content") is not None:
                    return out.get("content", "")
        except Exception as e:
            logger.debug(f"Failed to fetch file via socket: {e}")
        return None
    try:
        import httpx

        url = f"{base}/api/files/read?path={url_quote(file_path)}"
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            if response.status_code == 200:
                return response.json().get("content", "")
    except Exception as e:
        logger.debug(f"Failed to fetch file content via LocalServer: {e}")
    return None


def _sync_file_from_local_server_to_redis(file_path: str) -> bool:
    """When in local mode, sync a single file's content from LocalServer to Redis so manager state matches local.

    Call this before reading from the manager (get_file, get_file_diff) so diffs and content are accurate.
    Only updates Redis if the file is already tracked in the manager (so we don't add untracked files).
    Returns True if content was synced, False otherwise.
    """
    if not _should_route_to_local_server():
        return False
    content = _fetch_file_content_from_local_server(file_path)
    if content is None:
        return False
    try:
        manager = _get_code_changes_manager()
        if file_path not in manager.changes:
            return False
        change = manager.changes[file_path]
        if change.change_type == ChangeType.DELETE:
            return False
        manager.update_file(
            file_path=file_path,
            content=content,
            description=change.description or "Synced from local",
            preserve_previous=True,
        )
        logger.debug(
            f"CodeChangesManager: Synced file from LocalServer to Redis: {file_path}"
        )
        return True
    except Exception as e:
        logger.debug(
            f"CodeChangesManager: Failed to sync file from LocalServer to Redis: {e}"
        )
        return False


def _record_local_change_in_redis(
    operation: str,
    data: Dict[str, Any],
    previous_content_for_update: Optional[str] = None,
) -> None:
    """After a successful local write via tunnel, record the change in Redis so get_summary/get_file show it."""
    try:
        manager = _get_code_changes_manager()
        file_path = data.get("file_path") or ""
        if not file_path:
            return
        if operation == "add_file":
            manager.add_file(
                file_path=file_path,
                content=data.get("content", ""),
                description=data.get("description"),
            )
        elif operation == "update_file":
            manager.update_file(
                file_path=file_path,
                content=data.get("content", ""),
                description=data.get("description"),
                previous_content=previous_content_for_update,
            )
        elif operation == "delete_file":
            manager.delete_file(
                file_path=file_path,
                description=data.get("description"),
            )
        elif operation in (
            "update_file_lines",
            "insert_lines",
            "delete_lines",
            "replace_in_file",
        ):
            content = _fetch_file_content_from_local_server(file_path)
            if content is not None:
                manager.update_file(
                    file_path=file_path,
                    content=content,
                    description=data.get("description"),
                )
        elif operation == "revert_file":
            # Revert applied in IDE; sync Redis with reverted content
            content = _fetch_file_content_from_local_server(file_path)
            if content is not None:
                manager.update_file(
                    file_path=file_path,
                    content=content,
                    description=data.get("description"),
                )
        logger.info(
            f"CodeChangesManager: Recorded local change in Redis for {operation} '{file_path}'"
        )
    except Exception as e:
        logger.warning(
            f"CodeChangesManager: Failed to record local change in Redis: {e}"
        )


def _execute_local_write(operation: str, data: Dict[str, Any], file_path: str) -> Optional[str]:
    """Execute a write operation locally with local-first semantics.

    For write operations (add, update, delete, replace, insert), we REQUIRE local execution
    when the user has a VS Code extension connected (tunnel available).

    Returns:
        - Success message if local execution succeeded
        - Error message if local execution failed (does NOT fall back to cloud)
        - None if no tunnel available (caller can decide to use cloud or not)
    """
    should_route = _should_route_to_local_server()

    if not should_route:
        # No tunnel available - user is not using VS Code extension
        # Return None to allow cloud fallback (for web UI users)
        logger.info(f"[Local-First] No tunnel for {operation}, allowing cloud fallback")
        return None

    # User has tunnel = using VS Code extension = expects LOCAL changes
    logger.info(f"[Local-First] 🏠 Executing {operation} locally (local-first mode)")

    # Fetch-before-edit for update_file: get current content from local so we can track diffs accurately
    previous_content_for_update: Optional[str] = None
    if operation == "update_file":
        previous_content_for_update = _fetch_file_content_from_local_server(file_path)
        if previous_content_for_update is not None:
            logger.debug(
                f"[Local-First] Fetched current content from local before update_file ({len(previous_content_for_update)} chars) for accurate diff"
            )

    result = _route_to_local_server(operation, data)

    if result:
        # Local execution succeeded - also store change in Redis so get_summary/get_file show it
        _record_local_change_in_redis(
            operation, data, previous_content_for_update=previous_content_for_update
        )
        return result

    # Local execution FAILED but user expected local
    # Do NOT fall back to cloud - return an error so agent can retry or inform user
    logger.warning(f"[Local-First] ❌ Local {operation} failed for '{file_path}'")

    return (
        f"❌ **Local execution failed** for '{file_path}'\n\n"
        f"Your VS Code extension tunnel appears to be disconnected or stale.\n\n"
        f"**What to do:**\n"
        f"1. Check if VS Code extension is running\n"
        f"2. Check the Output panel in VS Code for tunnel status\n"
        f"3. Reload the VS Code window if needed (Cmd/Ctrl+Shift+P → 'Reload Window')\n"
        f"4. Retry this operation\n\n"
        f"**Note:** Changes are made directly in your local IDE, not in the cloud."
    )
