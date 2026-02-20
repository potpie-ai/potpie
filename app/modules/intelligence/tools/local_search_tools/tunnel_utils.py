"""
Tunnel utilities for routing search operations to LocalServer.

Socket path: get_tunnel_url returns socket://{workspace_id}; tool calls go via
WorkspaceSocketService.execute_tool_call_with_fallback (Socket.IO RPC).
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
import httpx
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Prefix for socket-backed tunnel URL (get_tunnel_url returns this when socket is online)
SOCKET_TUNNEL_PREFIX = "socket://"

# Retry settings for transient socket failures (e.g. brief disconnect / Redis blip)
_SOCKET_MAX_RETRIES = int(os.getenv("SOCKET_MAX_RETRIES", "2"))
_SOCKET_RETRY_DELAY_SECS = float(os.getenv("SOCKET_RETRY_DELAY_SECS", "1.0"))


def _execute_via_socket(
    user_id: str,
    conversation_id: Optional[str],
    endpoint: str,
    payload: Dict[str, Any],
    tunnel_url: Optional[str] = None,
    repository: Optional[str] = None,
    branch: Optional[str] = None,
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """Execute a tool call via Socket.IO with simple retry on transient failures.

    Returns the unwrapped result dict on success, or None if all attempts fail.
    """
    from app.modules.tunnel.tunnel_service import get_tunnel_service, TunnelConnectionError

    last_error: Optional[str] = None
    for attempt in range(1, _SOCKET_MAX_RETRIES + 2):  # +2: initial attempt + retries
        try:
            out = get_tunnel_service().execute_tool_call_with_fallback(
                user_id=user_id,
                conversation_id=conversation_id,
                endpoint=endpoint,
                payload=payload,
                tunnel_url=tunnel_url,
                repository=repository,
                branch=branch,
                timeout=timeout,
            )
            # Socket tool_response shape: { success, result?, error? }
            if isinstance(out, dict) and "success" in out:
                if out.get("success") and "result" in out:
                    return out["result"]
                return None
            return out
        except TunnelConnectionError as exc:
            last_error = exc.last_error or str(exc)
            if attempt <= _SOCKET_MAX_RETRIES:
                delay = _SOCKET_RETRY_DELAY_SECS * attempt
                logger.warning(
                    "[_execute_via_socket] Attempt %d/%d failed (%s) — retrying in %.1fs",
                    attempt,
                    _SOCKET_MAX_RETRIES + 1,
                    last_error,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "[_execute_via_socket] All %d attempts failed (last: %s)",
                    _SOCKET_MAX_RETRIES + 1,
                    last_error,
                )
    return None


def _curl_equivalent_terminal_execute(
    url: str, request_data: Dict[str, Any], timeout_sec: float
) -> str:
    """Build a curl command equivalent to what the backend sends for terminal execute (for debugging)."""
    body = json.dumps(request_data)
    # Escape single quotes in JSON for use inside single-quoted shell string
    body_escaped = body.replace("'", "'\"'\"'")
    return (
        f"curl -v -X POST '{url}' "
        f"-H 'Content-Type: application/json' "
        f"-d '{body_escaped}' "
        f"--max-time {int(timeout_sec)}"
    )


def _is_cloudflare_tunnel_error(response_status: int, response_text: str) -> bool:
    """Check if the response indicates a tunnel/upstream error (e.g. 530, legacy HTTP tunnel)."""
    return response_status == 530 or "tunnel" in response_text.lower() and "error" in response_text.lower()


def _is_tunnel_connection_error(response_status: int, response_text: str) -> bool:
    """Check if the response indicates any tunnel/connection error."""
    return (
        response_status in [502, 503, 504, 530]
        or "tunnel" in response_text.lower()
        and "error" in response_text.lower()
    )


def handle_tunnel_error(error: Exception, operation: str = "tool call") -> str:
    """
    Format a tunnel error into a user-friendly message for the agent.

    Args:
        error: The exception that occurred
        operation: Description of the operation that failed

    Returns:
        Formatted error message string
    """
    from app.modules.tunnel.tunnel_service import TunnelConnectionError

    if isinstance(error, TunnelConnectionError):
        return (
            f"❌ **Tunnel Disconnected**\n\n"
            f"The connection to the local VS Code extension was lost while attempting: {operation}\n\n"
            f"**Error:** {error.message}\n\n"
            f"**Suggestion:** Please check that:\n"
            f"1. VS Code is running with the Potpie extension active\n"
            f"2. The extension shows 'Connected' status\n"
            f"3. Try reloading VS Code (Cmd/Ctrl+Shift+P → 'Reload Window')\n\n"
            f"Once reconnected, please retry your request."
        )

    # Generic connection error
    error_str = str(error)
    if "timeout" in error_str.lower():
        return (
            f"❌ **Request Timeout**\n\n"
            f"The {operation} timed out. The local VS Code extension may be:\n"
            f"- Processing a long operation\n"
            f"- Experiencing network issues\n"
            f"- Disconnected\n\n"
            f"Please check VS Code and retry."
        )

    if "connect" in error_str.lower() or "refused" in error_str.lower():
        return (
            f"❌ **Connection Failed**\n\n"
            f"Could not connect to the local VS Code extension for: {operation}\n\n"
            f"Please ensure the Potpie extension is running and try again."
        )

    return (
        f"❌ **{operation.title()} Failed**\n\n"
        f"Error: {error_str}\n\n"
        f"Please check the VS Code extension status and retry."
    )


def format_tunnel_error_for_agent(
    success: bool = False,
    error_code: str = "TUNNEL_DISCONNECTED",
    message: str = "",
    suggestion: str = "",
) -> dict:
    """
    Create a standardized error response dict for tunnel errors.

    Args:
        success: Always False for errors
        error_code: Error code string
        message: Error message
        suggestion: Suggestion for the user

    Returns:
        Dict with error information
    """
    return {
        "success": success,
        "error": error_code,
        "message": message,
        "suggestion": suggestion
        or (
            "The local VS Code extension tunnel appears to be disconnected. "
            "Please check VS Code and ensure the Potpie extension is running."
        ),
    }


def read_files_batch_from_local_server(
    paths: List[str],
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Call LocalServer POST /api/files/read-batch with body { \"paths\": paths }.

    Response format: { "files": [ { "path", "content"?, "line_count"?, "error"? }, ... ] }
    Missing files have an "error" field (e.g. "File not found") instead of content/line_count.

    Returns:
        Response dict with "files" key, or None if tunnel unavailable or request failed.
    """
    if not paths:
        return {"files": []}

    try:
        from app.modules.tunnel.tunnel_service import get_tunnel_service

        if user_id is None or conversation_id is None:
            u, c = get_context_vars()
            user_id = user_id or u
            conversation_id = conversation_id or c

        if not user_id:
            logger.debug("[read_files_batch] No user_id in context")
            return None

        from app.modules.intelligence.tools.code_changes_manager import (
            _get_tunnel_url,
            _get_repository,
            _get_branch,
        )

        context_tunnel_url = _get_tunnel_url()
        repository = _get_repository()
        branch = _get_branch()
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id,
            conversation_id,
            tunnel_url=context_tunnel_url,
            repository=repository,
            branch=branch,
        )

        force_tunnel = os.getenv("FORCE_TUNNEL", "").lower() in ["true", "1", "yes"]
        base_url = os.getenv("BASE_URL", "").lower()
        environment = os.getenv("ENVIRONMENT", "").lower()
        is_backend_local = (
            "localhost" in base_url
            or "127.0.0.1" in base_url
            or environment in ["local", "dev", "development"]
            or not base_url
        )

        base = None
        if is_backend_local and not force_tunnel:
            from app.modules.tunnel.tunnel_service import _get_local_tunnel_server_url

            direct_base = _get_local_tunnel_server_url()
            if not direct_base:
                local_port_env = os.getenv("LOCAL_SERVER_PORT")
                direct_port = int(local_port_env) if local_port_env else 3001
                direct_base = f"http://localhost:{direct_port}"
            try:
                with httpx.Client(timeout=2.0) as health_client:
                    health = health_client.get(f"{direct_base}/health")
                    if health.status_code == 200:
                        base = direct_base
                        logger.info(
                            f"[read_files_batch] Using direct LocalServer: {base}"
                        )
            except Exception:
                pass

        if base is None and tunnel_url:
            if tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
                # Socket path
                result = _execute_via_socket(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    endpoint="/api/files/read-batch",
                    payload={"paths": paths},
                    tunnel_url=tunnel_url,
                    repository=repository,
                    branch=branch,
                    timeout=30.0,
                )
                if result is not None and "files" in result:
                    return result
                return None
            base = tunnel_url
            logger.info(f"[read_files_batch] Using tunnel: {base}")

        if not base:
            logger.debug("[read_files_batch] No LocalServer base (tunnel or direct)")
            return None

        url = f"{base.rstrip('/')}/api/files/read-batch"
        payload = {"paths": paths}
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        if response.status_code != 200:
            logger.debug(
                f"[read_files_batch] HTTP {response.status_code}: {response.text[:200]}"
            )
            return None
        data = response.json()
        if "files" not in data:
            logger.debug("[read_files_batch] Response missing 'files' key")
            return None
        return data
    except Exception as e:
        logger.debug(f"[read_files_batch] Failed: {e}")
        return None


def route_to_local_server(
    operation: str,
    data: Dict[str, Any],
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Optional[str]:
    """Route search operation to LocalServer via tunnel (sync version).

    Args:
        operation: The operation name (e.g., "search_symbols")
        data: Request data to send to LocalServer
        user_id: User ID from context
        conversation_id: Conversation ID from context

    Returns:
        Result string if successful, None if should fall back
    """
    try:
        from app.modules.tunnel.tunnel_service import get_tunnel_service

        if not user_id:
            logger.debug("No user_id in context, skipping tunnel routing")
            return None

        # Get tunnel_url from context if available (takes priority); use repository/branch for workspace-scoped lookup
        from app.modules.intelligence.tools.code_changes_manager import (
            _get_tunnel_url,
            _get_repository,
            _get_branch,
        )

        context_tunnel_url = _get_tunnel_url()
        repository = _get_repository()
        branch = _get_branch()
        if context_tunnel_url:
            logger.info(
                f"[Tunnel Routing] ✅ Using fresh tunnel_url from context: {context_tunnel_url}"
            )
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id,
            conversation_id,
            tunnel_url=context_tunnel_url,
            repository=repository,
            branch=branch,
        )

        if not tunnel_url:
            logger.error(
                "[Tunnel Routing] No tunnel available for user %s (conversation %s) - LocalServer/tunnel not connected or inactive. Tool will return fallback message.",
                user_id,
                conversation_id or "(none)",
            )
            return None

        # Map operation to LocalServer endpoint
        endpoint_map = {
            "search_symbols": "/api/search/symbols",
            "search_workspace_symbols": "/api/search/workspace-symbols",
            "search_references": "/api/search/references",
            "search_definitions": "/api/search/definitions",
            "search_files": "/api/search/files",
            "search_text": "/api/search/text",
            "search_code_structure": "/api/search/code-structure",
            "search_bash": "/api/search/bash",
            "search_semantic": "/api/search/semantic",
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

        # Prefer direct localhost when backend is local (avoid hairpin + cfargotunnel DNS issues)
        force_tunnel = os.getenv("FORCE_TUNNEL", "").lower() in ["true", "1", "yes"]
        base_url = os.getenv("BASE_URL", "").lower()
        environment = os.getenv("ENVIRONMENT", "").lower()
        is_backend_local = (
            "localhost" in base_url
            or "127.0.0.1" in base_url
            or environment in ["local", "dev", "development"]
            or not base_url  # If BASE_URL not set, assume local
        )

        response = None
        if is_backend_local and not force_tunnel:
            # Prefer VSCODE_LOCAL_TUNNEL_SERVER when set (e.g. http://localhost:3001)
            from app.modules.tunnel.tunnel_service import _get_local_tunnel_server_url

            direct_base = _get_local_tunnel_server_url()
            if not direct_base:
                local_port_env = os.getenv("LOCAL_SERVER_PORT")
                direct_port = int(local_port_env) if local_port_env else 3001
                direct_base = f"http://localhost:{direct_port}"
            direct_url = f"{direct_base}{endpoint}"
            try:
                # Quick health check to ensure LocalServer is reachable directly
                with httpx.Client(timeout=2.0) as health_client:
                    health = health_client.get(f"{direct_base}/health")
                    if health.status_code != 200:
                        logger.error(
                            "[Tunnel Routing] LocalServer health check failed: %s returned %s (tunnel/LocalServer may be inactive)",
                            f"{direct_base}/health",
                            health.status_code,
                        )
                        response = None
                    else:
                        logger.info(
                            f"[Tunnel Routing] 🏠 Backend local -> routing {operation} directly: {direct_url}"
                        )
                        with httpx.Client(timeout=30.0) as client:
                            direct_response = client.post(
                                direct_url,
                                json=request_data,
                                headers={"Content-Type": "application/json"},
                            )
                        if direct_response.status_code == 200:
                            result = direct_response.json()
                            return format_search_result(operation, result)
                        logger.error(
                            "[Tunnel Routing] LocalServer %s failed: HTTP %s - %s",
                            operation,
                            direct_response.status_code,
                            direct_response.text[:300],
                        )
                        response = None
            except Exception as e:
                logger.error(
                    "[Tunnel Routing] LocalServer not reachable at %s (tunnel/LocalServer inactive or not running): %s",
                    direct_base,
                    e,
                )
                response = None
        else:
            response = None

        # Socket path: tunnel_url is socket://{workspace_id}
        if response is None and tunnel_url and tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
            logger.info(
                f"[Tunnel Routing] 🚀 Routing {operation} to LocalServer via Socket.IO"
            )
            result = _execute_via_socket(
                user_id=user_id,
                conversation_id=conversation_id,
                endpoint=endpoint,
                payload=request_data,
                tunnel_url=tunnel_url,
                repository=repository,
                branch=branch,
                timeout=120.0,
            )
            if result is not None:
                logger.info(f"[Tunnel Routing] ✅ LocalServer {operation} succeeded")
                return format_search_result(operation, result)
            logger.error(
                "[Tunnel Routing] Socket tool call failed or no response for %s - falling back to cloud. Ensure VS Code extension is connected.",
                operation,
            )
            return None

    except Exception as e:
        logger.exception(f"Error routing to LocalServer for {operation}: {e}")
        return None


def format_search_result(operation: str, result: dict) -> str:
    """Format search result from LocalServer for agent consumption."""
    if not result.get("success"):
        return f"❌ Search failed: {result.get('error', 'Unknown error')}"

    if operation == "search_symbols":
        symbols = result.get("symbols", [])
        if not symbols:
            return f"📋 No symbols found in {result.get('file_path', 'file')}"

        formatted = f"📋 **Found {len(symbols)} symbol(s) in {result.get('file_path', 'file')}:**\n\n"
        for symbol in symbols[:20]:  # Limit to 20 for readability
            kind_icon = {
                "class": "📦",
                "function": "🔧",
                "method": "⚙️",
                "variable": "📝",
                "interface": "🔌",
                "enum": "📊",
                "type": "🏷️",
            }.get(symbol.get("kind", ""), "•")
            formatted += f"{kind_icon} **{symbol.get('name')}** ({symbol.get('kind')}) - Line {symbol.get('range', {}).get('start', {}).get('line', '?')}\n"
        if len(symbols) > 20:
            formatted += f"\n... and {len(symbols) - 20} more symbols"
        return formatted

    elif operation == "search_workspace_symbols":
        symbols = result.get("symbols", [])
        if not symbols:
            return f"📋 No symbols found matching '{result.get('query', 'query')}'"

        formatted = f"📋 **Found {len(symbols)} symbol(s) matching '{result.get('query')}':**\n\n"
        for symbol in symbols[:20]:
            file_path = symbol.get("location", {}).get("file_path", "unknown")
            line = (
                symbol.get("location", {})
                .get("range", {})
                .get("start", {})
                .get("line", "?")
            )
            formatted += f"• **{symbol.get('name')}** ({symbol.get('kind')}) in {file_path}:{line}\n"
        if len(symbols) > 20:
            formatted += f"\n... and {len(symbols) - 20} more symbols"
        return formatted

    elif operation == "search_references":
        references = result.get("references", [])
        if not references:
            return f"📋 No references found for symbol at {result.get('file_path')}:{result.get('position', {}).get('line')}"

        formatted = f"📋 **Found {len(references)} reference(s):**\n\n"
        for ref in references[:20]:
            file_path = ref.get("file_path", "unknown")
            line = ref.get("range", {}).get("start", {}).get("line", "?")
            formatted += f"• {file_path}:{line}\n"
        if len(references) > 20:
            formatted += f"\n... and {len(references) - 20} more references"
        return formatted

    elif operation == "search_definitions":
        definitions = result.get("definitions", [])
        if not definitions:
            return f"📋 No definitions found for symbol at {result.get('file_path')}:{result.get('position', {}).get('line')}"

        formatted = f"📋 **Found {len(definitions)} definition(s):**\n\n"
        for defn in definitions[:10]:
            file_path = defn.get("file_path", "unknown")
            line = defn.get("range", {}).get("start", {}).get("line", "?")
            formatted += f"• {file_path}:{line}\n"
        return formatted

    elif operation == "search_files":
        files = result.get("files", [])
        if not files:
            return f"📋 No files found matching pattern '{result.get('pattern')}'"

        formatted = (
            f"📋 **Found {len(files)} file(s) matching '{result.get('pattern')}':**\n\n"
        )
        for file_info in files[:30]:
            formatted += f"• {file_info.get('relative_path', file_info.get('file_path', 'unknown'))}\n"
        if len(files) > 30:
            formatted += f"\n... and {len(files) - 30} more files"
        return formatted

    elif operation == "search_text":
        results = result.get("results", [])
        total_matches = result.get("total_matches", 0)
        if not results:
            return f"📋 No matches found for '{result.get('query')}'"

        formatted = f"📋 **Found {total_matches} match(es) for '{result.get('query')}' in {len(results)} file(s):**\n\n"
        for file_result in results[:10]:
            file_path = file_result.get("file_path", "unknown")
            matches = file_result.get("matches", [])
            formatted += f"**{file_path}** ({len(matches)} matches):\n"
            for match in matches[:5]:
                line = match.get("line", "?")
                text = match.get("text", "").strip()[:100]  # Truncate long lines
                formatted += f"  Line {line}: {text}\n"
            if len(matches) > 5:
                formatted += f"  ... and {len(matches) - 5} more matches\n"
            formatted += "\n"
        if len(results) > 10:
            formatted += f"... and {len(results) - 10} more files with matches"
        return formatted

    elif operation == "search_code_structure":
        symbols = result.get("symbols", [])
        if not symbols:
            return f"📋 No code structure found"

        formatted = f"📋 **Found {len(symbols)} symbol(s):**\n\n"
        for symbol in symbols[:20]:
            name = symbol.get("name", "unknown")
            kind = symbol.get("kind", "unknown")
            if "location" in symbol:
                file_path = symbol.get("location", {}).get("file_path", "unknown")
                line = (
                    symbol.get("location", {})
                    .get("range", {})
                    .get("start", {})
                    .get("line", "?")
                )
                formatted += f"• **{name}** ({kind}) in {file_path}:{line}\n"
            else:
                line = symbol.get("range", {}).get("start", {}).get("line", "?")
                formatted += f"• **{name}** ({kind}) at line {line}\n"
        if len(symbols) > 20:
            formatted += f"\n... and {len(symbols) - 20} more symbols"
        return formatted

    elif operation == "search_bash":
        # Format bash command output
        output = result.get("output", "")
        error = result.get("error", "")
        exit_code = result.get("exit_code", 0)
        command = result.get("command", "command")

        if exit_code == 0 and output:
            # Success with output
            lines = output.strip().split("\n")
            formatted = f"📋 **Bash command result** (`{command}`):\n\n"
            formatted += f"```\n{output[:5000]}\n```\n"  # Limit to 5k chars
            if len(output) > 5000:
                formatted += f"\n... (output truncated, {len(output)} total characters)"
            return formatted
        elif exit_code != 0:
            # Command failed
            formatted = (
                f"⚠️ **Bash command failed** (`{command}`, exit code: {exit_code}):\n\n"
            )
            if error:
                formatted += f"Error: {error[:1000]}\n\n"
            if output:
                formatted += f"Output: {output[:1000]}"
            return formatted
        else:
            return f"✅ Command executed: `{command}` (no output)"

    elif operation == "search_semantic":
        results = result.get("results", [])
        total_results = result.get("total_results", 0)
        query = result.get("query", "query")

        if not results:
            return (
                f"📋 No semantically similar code found for '{query}'. "
                f"Ensure the project is parsed and knowledge graph is available."
            )

        formatted = f"📋 **Found {total_results} semantically similar result(s) for '{query}':**\n\n"
        for i, r in enumerate(results[:10], 1):
            file_path = r.get("file_path", "unknown")
            start_line = r.get("start_line", 0)
            similarity = r.get("similarity", 0.0)
            docstring = r.get("docstring", "")
            name = r.get("name", "")
            node_type = r.get("type", "")

            formatted += f"{i}. **{file_path}:{start_line}**"
            if name:
                formatted += f" - `{name}`"
            if node_type:
                formatted += f" ({node_type})"
            formatted += f" (similarity: {similarity:.3f})\n"

            if docstring:
                formatted += (
                    f"   {docstring[:200]}{'...' if len(docstring) > 200 else ''}\n"
                )
            formatted += "\n"

        if len(results) > 10:
            formatted += f"... and {len(results) - 10} more results"
        return formatted

    # Fallback for unknown operations
    return f"✅ Search completed: {json.dumps(result, indent=2)}"


def route_terminal_command(
    command: str,
    working_directory: Optional[str] = None,
    timeout: int = 30000,
    mode: str = "sync",
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Route terminal command to LocalServer via tunnel.

    Args:
        command: The shell command to execute
        working_directory: Optional working directory relative to workspace root
        timeout: Timeout in milliseconds (default: 30000)
        mode: Execution mode - "sync" (default) or "async" for long-running commands
        user_id: User ID from context
        conversation_id: Conversation ID from context

    Returns:
        Tuple of (result_dict, error_type):
        - (result_dict, None) if successful
        - (None, "no_tunnel") if no tunnel registered
        - (None, "tunnel_unreachable") if tunnel registered but unreachable
        - (None, "connection_error") for other connection errors
    """
    try:
        from app.modules.tunnel.tunnel_service import get_tunnel_service
        import httpx

        if not user_id:
            logger.debug("No user_id in context, skipping tunnel routing")
            return None, "no_user_id"

        # Get tunnel_url from context if available (takes priority); use repository/branch for workspace-scoped lookup
        from app.modules.intelligence.tools.code_changes_manager import (
            _get_tunnel_url,
            _get_repository,
            _get_branch,
        )

        context_tunnel_url = _get_tunnel_url()
        repository = _get_repository()
        branch = _get_branch()
        if context_tunnel_url:
            logger.info(
                f"[Tunnel Routing] ✅ Using fresh tunnel_url from context: {context_tunnel_url}"
            )
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id,
            conversation_id,
            tunnel_url=context_tunnel_url,
            repository=repository,
            branch=branch,
        )

        if not tunnel_url:
            workspace_id = tunnel_service.get_workspace_id(
                user_id, conversation_id, repository=repository, branch=branch
            )
            if workspace_id is None:
                logger.info(
                    "[route_terminal_command] No workspace_id (repository not in context for this conversation); cannot route to extension"
                )
            else:
                logger.info(
                    f"[route_terminal_command] Workspace socket offline for workspace_id={workspace_id} (extension may not have registered)"
                )
            return None, "no_tunnel"

        # Prepare request data
        request_data = {
            "command": command,
            "working_directory": working_directory,
            "timeout": timeout,
            "mode": mode,
            "conversation_id": conversation_id,
        }

        # Socket path: tunnel_url is socket://{workspace_id}
        request_timeout_sec = float(timeout) / 1000 + 5
        if tunnel_url and tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
            logger.info(
                f"[Tunnel Routing] 🚀 Routing terminal command to LocalServer via Socket.IO (timeout={request_timeout_sec}s)"
            )
            result = _execute_via_socket(
                user_id=user_id,
                conversation_id=conversation_id,
                endpoint="/api/terminal/execute",
                payload=request_data,
                tunnel_url=tunnel_url,
                repository=repository,
                branch=branch,
                timeout=request_timeout_sec,
            )
            if result is not None:
                logger.info(f"[Tunnel Routing] ✅ Terminal command succeeded")
                return result, None
            return None, "tunnel_unreachable"

        # Legacy HTTP path (e.g. direct local URL)
        base = (tunnel_url or "").rstrip("/")
        url = f"{base}/api/terminal/execute"
        logger.info(
            f"[Tunnel Routing] 🚀 Routing terminal command via HTTP: {url} (timeout={request_timeout_sec}s)"
        )
        with httpx.Client(timeout=request_timeout_sec) as client:
            response = client.post(
                url,
                json=request_data,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 404:
                fallback_url = f"{base}/terminal/execute"
                response = client.post(
                    fallback_url,
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                )
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[Tunnel Routing] ✅ Terminal command succeeded")
                return result, None
            logger.warning(
                f"[Tunnel Routing] ❌ Terminal command failed ({response.status_code}): {response.text[:500]}"
            )
            return None, "command_error"

    except httpx.ConnectError as e:  # type: ignore[misc]
        # DNS resolution failure or connection refused
        error_msg = str(e)
        is_dns_error = (
            "nodename nor servname provided" in error_msg
            or "Name or service not known" in error_msg
        )

        # Get tunnel info if available (may not be set if error occurs early)
        tunnel_url_str = "unknown"
        try:
            from app.modules.tunnel.tunnel_service import get_tunnel_service

            if user_id:
                tunnel_service = get_tunnel_service()
                tunnel_url = tunnel_service.get_tunnel_url(user_id, conversation_id)
                if tunnel_url:
                    tunnel_url_str = tunnel_url
        except Exception:
            pass  # Ignore errors when trying to get tunnel info

        if is_dns_error:
            logger.warning(
                f"[Tunnel Routing] ❌ DNS resolution failed for tunnel URL (tunnel expired or invalid): {tunnel_url_str}"
            )
            # Unregister conversation-level tunnel (workspace-only; no user-level)
            if user_id and conversation_id:
                try:
                    from app.modules.tunnel.tunnel_service import get_tunnel_service

                    tunnel_service = get_tunnel_service()
                    tunnel_service.unregister_tunnel(user_id, conversation_id)
                    logger.info(
                        f"[Tunnel Routing] Unregistered expired conversation tunnel for user {user_id}"
                    )
                except Exception as unreg_error:
                    logger.warning(f"Failed to unregister broken tunnel: {unreg_error}")
            return None, "tunnel_expired"
        else:
            logger.warning(
                f"[Tunnel Routing] ❌ Connection error to tunnel: {error_msg}"
            )
        return None, "connection_error"
    except httpx.TimeoutException as e:  # type: ignore[misc]
        logger.warning(
            "[Tunnel Routing] ❌ Timeout connecting to tunnel (tunnel may be unreachable): {}",
            str(e),
        )
        # Log curl equivalent so the same request can be reproduced (e.g. from backend vs local)
        try:
            from app.modules.tunnel.tunnel_service import get_tunnel_service
            from app.modules.intelligence.tools.code_changes_manager import (
                _get_repository,
                _get_branch,
            )
            tunnel_url_str = (
                get_tunnel_service().get_tunnel_url(
                    user_id or "",
                    conversation_id,
                    repository=_get_repository(),
                    branch=_get_branch(),
                )
                or ""
            )
            if tunnel_url_str:
                url = f"{tunnel_url_str}/api/terminal/execute"
                req_data = {
                    "command": command,
                    "working_directory": working_directory,
                    "timeout": timeout,
                    "mode": mode,
                    "conversation_id": conversation_id,
                }
                curl_cmd = _curl_equivalent_terminal_execute(
                    url, req_data, float(timeout) / 1000 + 5
                )
                logger.warning(
                    "[Tunnel Routing] To reproduce from your machine or backend: {}",
                    curl_cmd,
                )
        except Exception:
            pass
        return None, "timeout"
    except Exception as e:
        logger.exception(f"Error routing terminal command to LocalServer: {e}")
        return None, "unknown_error"


def get_terminal_session_output(
    session_id: str,
    offset: int = 0,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Get output from an async terminal session.

    Args:
        session_id: The session ID from async command execution
        offset: Byte offset to read from (default: 0)
        user_id: User ID from context
        conversation_id: Conversation ID from context

    Returns:
        Output dict if successful, None if error
    """
    try:
        from app.modules.tunnel.tunnel_service import get_tunnel_service

        if not user_id:
            logger.debug("No user_id in context, skipping tunnel routing")
            return None

        # Get tunnel_url from context if available; use repository/branch for workspace-scoped lookup
        from app.modules.intelligence.tools.code_changes_manager import (
            _get_tunnel_url,
            _get_repository,
            _get_branch,
        )

        context_tunnel_url = _get_tunnel_url()
        repository = _get_repository()
        branch = _get_branch()
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id,
            conversation_id,
            tunnel_url=context_tunnel_url,
            repository=repository,
            branch=branch,
        )

        if not tunnel_url:
            logger.debug(f"No tunnel available for user {user_id}")
            return None

        if tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
            result = _execute_via_socket(
                user_id=user_id,
                conversation_id=conversation_id,
                endpoint=f"/api/terminal/sessions/{session_id}/output",
                payload={"offset": offset},
                tunnel_url=tunnel_url,
                repository=repository,
                branch=branch,
                timeout=30.0,
            )
            return result if result else None

        base = tunnel_url.rstrip("/")
        urls_to_try = (
            f"{base}/api/terminal/sessions/{session_id}/output",
            f"{base}/terminal/sessions/{session_id}/output",
        )
        with httpx.Client(timeout=30.0) as client:
            response = None
            for url in urls_to_try:
                response = client.get(
                    url,
                    params={"offset": offset},
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code == 200:
                    return response.json()
                if response.status_code != 404:
                    break
            return None

    except Exception as e:
        logger.exception(f"Error getting terminal session output: {e}")
        return None


def send_terminal_session_signal(
    session_id: str,
    signal: str = "SIGINT",
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Send a signal to a terminal session (e.g., SIGINT to stop).

    Args:
        session_id: The session ID
        signal: Signal to send (default: "SIGINT")
        user_id: User ID from context
        conversation_id: Conversation ID from context

    Returns:
        Result dict if successful, None if error
    """
    try:
        from app.modules.tunnel.tunnel_service import get_tunnel_service

        if not user_id:
            logger.debug("No user_id in context, skipping tunnel routing")
            return None

        # Get tunnel_url from context if available; use repository/branch for workspace-scoped lookup
        from app.modules.intelligence.tools.code_changes_manager import (
            _get_tunnel_url,
            _get_repository,
            _get_branch,
        )

        context_tunnel_url = _get_tunnel_url()
        repository = _get_repository()
        branch = _get_branch()
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id,
            conversation_id,
            tunnel_url=context_tunnel_url,
            repository=repository,
            branch=branch,
        )

        if not tunnel_url:
            logger.debug(f"No tunnel available for user {user_id}")
            return None

        if tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
            result = _execute_via_socket(
                user_id=user_id,
                conversation_id=conversation_id,
                endpoint=f"/api/terminal/sessions/{session_id}/signal",
                payload={"signal": signal},
                tunnel_url=tunnel_url,
                repository=repository,
                branch=branch,
                timeout=30.0,
            )
            return result if result else None

        base = tunnel_url.rstrip("/")
        urls_to_try = (
            f"{base}/api/terminal/sessions/{session_id}/signal",
            f"{base}/terminal/sessions/{session_id}/signal",
        )
        logger.info(f"[Tunnel Routing] Sending signal {signal} to session {session_id}")
        with httpx.Client(timeout=30.0) as client:
            for url in urls_to_try:
                response = client.post(
                    url,
                    json={"signal": signal},
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code == 200:
                    return response.json()
                if response.status_code != 404:
                    break
            return None

    except Exception as e:
        logger.exception(f"Error sending terminal session signal: {e}")
        return None


def format_terminal_result(result: Dict[str, Any]) -> str:
    """Format terminal command result for agent consumption."""
    if not result.get("success"):
        error = result.get("error", "Unknown error")
        exit_code = result.get("exit_code", -1)

        if exit_code == -1:
            # Validation failed
            return f"❌ **Command blocked:** {error}\n\nThis command is not allowed for security reasons."
        else:
            return f"❌ **Command failed** (exit code: {exit_code}):\n\n{error}"

    command = result.get("command", "command")
    output = result.get("output", "")
    error = result.get("error", "")
    exit_code = result.get("exit_code", 0)
    duration_ms = result.get("duration_ms", 0)
    warnings = result.get("warnings", [])
    truncated = result.get("truncated", False)

    formatted = f"✅ **Command executed** (`{command}`)"
    if duration_ms:
        formatted += f" (took {duration_ms}ms)"
    formatted += ":\n\n"

    if output:
        # Limit output to 5000 chars for readability
        output_preview = output[:5000]
        formatted += f"```\n{output_preview}\n```\n"
        if len(output) > 5000 or truncated:
            total_chars = len(output) if not truncated else "many"
            formatted += f"\n... (output truncated, {total_chars} total characters)"

    if error:
        formatted += f"\n⚠️ **Error output:**\n```\n{error[:1000]}\n```"
        if len(error) > 1000:
            formatted += f"\n... (error truncated, {len(error)} total characters)"

    if exit_code != 0:
        formatted += f"\n⚠️ **Exit code:** {exit_code}"

    if warnings:
        formatted += f"\n⚠️ **Warnings:**\n"
        for warning in warnings[:5]:
            formatted += f"- {warning}\n"
        if len(warnings) > 5:
            formatted += f"... and {len(warnings) - 5} more warnings"

    return formatted


def get_context_vars():
    """Get user_id and conversation_id from context variables.

    Note: tunnel_url should be fetched separately using _get_tunnel_url()
    from code_changes_manager for tunnel routing.
    """
    try:
        # Import here to avoid circular dependency
        from app.modules.intelligence.tools.code_changes_manager import (
            _get_user_id,
            _get_conversation_id,
        )

        user_id = _get_user_id()
        conversation_id = _get_conversation_id()
        return user_id, conversation_id
    except Exception as e:
        logger.warning(f"Failed to get context vars: {e}")
        return None, None
