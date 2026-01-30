"""
Tunnel utilities for routing search operations to LocalServer.
"""

import json
import os
from typing import Dict, Any, Optional
import httpx
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _is_cloudflare_tunnel_error(response_status: int, response_text: str) -> bool:
    """Check if the response indicates a Cloudflare tunnel error (tunnel not reachable)."""
    return response_status == 530 and "Cloudflare Tunnel error" in response_text


def _is_tunnel_connection_error(response_status: int, response_text: str) -> bool:
    """Check if the response indicates any tunnel connection error."""
    return (
        response_status in [502, 503, 504, 530]
        or "Cloudflare Tunnel error" in response_text
        or "cloudflared" in response_text.lower()
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

        # Get tunnel_url from context if available (takes priority)
        from app.modules.intelligence.tools.code_changes_manager import _get_tunnel_url

        context_tunnel_url = _get_tunnel_url()
        if context_tunnel_url:
            logger.info(
                f"[Tunnel Routing] ✅ Using fresh tunnel_url from context: {context_tunnel_url}"
            )
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id, conversation_id, tunnel_url=context_tunnel_url
        )

        if not tunnel_url:
            logger.debug(f"No tunnel available for user {user_id}, using fallback")
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

        if is_backend_local and not force_tunnel:
            # Try to get local_port from environment or tunnel data, default to 3001
            local_port_env = os.getenv("LOCAL_SERVER_PORT")
            direct_port = int(local_port_env) if local_port_env else 3001
            direct_base = f"http://localhost:{direct_port}"
            direct_url = f"{direct_base}{endpoint}"
            try:
                # Quick health check to ensure LocalServer is reachable directly
                with httpx.Client(timeout=2.0) as health_client:
                    health = health_client.get(f"{direct_base}/health")
                    if health.status_code == 200:
                        logger.info(
                            f"[Tunnel Routing] 🏠 Backend local -> routing {operation} directly: {direct_url}"
                        )
                        with httpx.Client(timeout=30.0) as client:
                            response = client.post(
                                direct_url,
                                json=request_data,
                                headers={"Content-Type": "application/json"},
                            )
                    else:
                        response = None
            except Exception as e:
                logger.info(
                    f"[Tunnel Routing] 🏠 Direct LocalServer not reachable, falling back to tunnel. reason={e}"
                )
                response = None
        else:
            response = None

        # Fall back to tunnel
        if response is None:
            url = f"{tunnel_url}{endpoint}"
            logger.info(
                f"[Tunnel Routing] 🚀 Routing {operation} to LocalServer via tunnel: {url}"
            )
            logger.debug(f"[Tunnel Routing] Request data: {request_data}")

            # Use longer timeout for tunnel requests (production) - can be slower due to network latency
            # 2 minutes for tunnel requests, 30s for direct localhost
            request_timeout = 120.0  # 2 minutes for tunnel requests
            
            logger.debug(f"[Tunnel Routing] Using timeout: {request_timeout}s for tunnel request")

            with httpx.Client(timeout=request_timeout) as client:
                response = client.post(
                    url,
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"[Tunnel Routing] ✅ LocalServer {operation} succeeded")

                # Format search results for agent consumption
                return format_search_result(operation, result)
            else:
                error_text = response.text
                status_code = response.status_code

                # Detect Cloudflare tunnel errors (stale tunnel)
                is_tunnel_error = (
                    status_code in [502, 503, 504, 530]
                    or "Cloudflare Tunnel error" in error_text
                    or "cloudflared" in error_text.lower()
                )

                if is_tunnel_error:
                    logger.warning(
                        f"[Tunnel Routing] ❌ Stale tunnel detected ({status_code}): {tunnel_url}. "
                        f"Invalidating and retrying with fresh URL from context."
                    )

                    # Invalidate the stale tunnel URL
                    try:
                        tunnel_service.unregister_tunnel(user_id, conversation_id)
                        logger.info(
                            f"[Tunnel Routing] ✅ Invalidated stale conversation tunnel for user {user_id}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[Tunnel Routing] Failed to invalidate tunnel: {e}"
                        )

                    # PRIORITY 1: Retry with fresh tunnel_url from context (if available and different)
                    fresh_tunnel_url = _get_tunnel_url()
                    if fresh_tunnel_url and fresh_tunnel_url != tunnel_url:
                        logger.info(
                            f"[Tunnel Routing] 🔄 Retrying with fresh tunnel URL from context: {fresh_tunnel_url}"
                        )
                        retry_url = f"{fresh_tunnel_url}{endpoint}"
                        try:
                            # Use same longer timeout for retry
                            with httpx.Client(timeout=120.0) as retry_client:
                                retry_response = retry_client.post(
                                    retry_url,
                                    json=request_data,
                                    headers={"Content-Type": "application/json"},
                                )
                                if retry_response.status_code == 200:
                                    result = retry_response.json()
                                    logger.info(
                                        f"[Tunnel Routing] ✅ {operation} succeeded with fresh tunnel URL"
                                    )
                                    # Update cache with fresh URL
                                    tunnel_service.register_tunnel(
                                        user_id=user_id,
                                        tunnel_url=fresh_tunnel_url,
                                        conversation_id=conversation_id,
                                    )
                                    return format_search_result(operation, result)
                                else:
                                    logger.warning(
                                        f"[Tunnel Routing] ❌ Fresh tunnel URL also failed: {retry_response.status_code}"
                                    )
                        except Exception as retry_e:
                            logger.warning(
                                f"[Tunnel Routing] ❌ Retry with fresh URL failed: {retry_e}"
                            )

                    # PRIORITY 2: Try user-level tunnel as fallback (if conversation-specific failed)
                    if conversation_id:
                        user_level_tunnel = tunnel_service.get_tunnel_url(
                            user_id, None, tunnel_url=fresh_tunnel_url
                        )
                        if user_level_tunnel and user_level_tunnel != tunnel_url:
                            logger.info(
                                f"[Tunnel Routing] 🔄 Retrying with user-level tunnel: {user_level_tunnel}"
                            )

                            # Retry with user-level tunnel
                            retry_url = f"{user_level_tunnel}{endpoint}"
                            try:
                                # Use same longer timeout for retry
                                with httpx.Client(timeout=120.0) as retry_client:
                                    retry_response = retry_client.post(
                                        retry_url,
                                        json=request_data,
                                        headers={"Content-Type": "application/json"},
                                    )

                                    if retry_response.status_code == 200:
                                        result = retry_response.json()
                                        logger.info(
                                            f"[Tunnel Routing] ✅ User-level fallback succeeded for {operation}"
                                        )
                                        return format_search_result(operation, result)
                                    else:
                                        logger.warning(
                                            f"[Tunnel Routing] ❌ User-level fallback also failed: {retry_response.status_code}"
                                        )
                            except Exception as retry_e:
                                logger.warning(
                                    f"[Tunnel Routing] ❌ User-level fallback error: {retry_e}"
                                )

                    # Return None to allow fallback to cloud execution
                    logger.info(
                        f"[Tunnel Routing] ⬇️ Falling back to cloud execution for {operation}"
                    )
                    return None

                logger.warning(
                    f"[Tunnel Routing] ❌ LocalServer {operation} failed ({status_code}): {error_text[:200]}"
                )

                # For bash commands, provide helpful error message instead of returning None
                if operation == "search_bash" and response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", error_text)
                        return (
                            f"❌ Bash command rejected by LocalServer: {error_msg}\n\n"
                            f"**Command:** `{data.get('command', 'unknown')}`\n\n"
                            f"**Reason:** The command contains patterns that are blocked for security reasons.\n\n"
                            f"**Allowed operations:** Read-only commands like `grep`, `find`, `cat`, `head`, `tail`, `wc`, `sort`, `uniq`.\n"
                            f"**Blocked operations:** Command chaining (`;`, `&&`), file redirection (`>`, `>>`), destructive commands (`rm`, `mv`, `cp`), and git write operations.\n\n"
                            f"**Note:** Pipes (`|`) are allowed for read-only operations (e.g., `grep pattern | head -n 10`).\n\n"
                            f"**Recommendation:** Simplify the command or use the cloud `bash_command` tool if you need more complex operations."
                        )
                    except:
                        return (
                            f"❌ Bash command rejected by LocalServer (400): {error_text}\n\n"
                            f"**Command:** `{data.get('command', 'unknown')}`\n\n"
                            f"Please simplify the command or use the cloud `bash_command` tool for complex operations."
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

        # Get tunnel_url from context if available (takes priority)
        from app.modules.intelligence.tools.code_changes_manager import _get_tunnel_url

        context_tunnel_url = _get_tunnel_url()
        if context_tunnel_url:
            logger.info(
                f"[Tunnel Routing] ✅ Using fresh tunnel_url from context: {context_tunnel_url}"
            )
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id, conversation_id, tunnel_url=context_tunnel_url
        )

        if not tunnel_url:
            logger.debug(f"No tunnel available for user {user_id}, using fallback")
            return None, "no_tunnel"

        # Prepare request data
        request_data = {
            "command": command,
            "working_directory": working_directory,
            "timeout": timeout,
            "mode": mode,
            "conversation_id": conversation_id,
        }

        # Make request to LocalServer via tunnel
        url = f"{tunnel_url}/api/terminal/execute"
        logger.info(
            f"[Tunnel Routing] 🚀 Routing terminal command to LocalServer via tunnel: {url}"
        )
        logger.debug(
            f"[Tunnel Routing] Command: {command}, Working directory: {working_directory}"
        )

        with httpx.Client(timeout=float(timeout) / 1000 + 5) as client:  # Add 5s buffer
            response = client.post(
                url,
                json=request_data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"[Tunnel Routing] ✅ Terminal command succeeded")
                return result, None
            else:
                error_text = response.text

                # Detect Cloudflare tunnel errors (530 status with HTML error page)
                is_cloudflare_error = _is_cloudflare_tunnel_error(
                    response.status_code, error_text
                ) or _is_tunnel_connection_error(response.status_code, error_text)

                if is_cloudflare_error:
                    logger.warning(
                        f"[Tunnel Routing] ❌ Stale tunnel detected ({response.status_code}): {tunnel_url}. "
                        f"Invalidating and retrying with fresh URL from context."
                    )

                    # Invalidate the stale tunnel
                    try:
                        tunnel_service.unregister_tunnel(user_id, conversation_id)
                        logger.info(
                            f"[Tunnel Routing] ✅ Invalidated stale conversation tunnel"
                        )
                    except Exception as e:
                        logger.error(
                            f"[Tunnel Routing] Failed to invalidate tunnel: {e}"
                        )

                    # Retry with fresh tunnel_url from context (if available)
                    fresh_tunnel_url = _get_tunnel_url()
                    if fresh_tunnel_url and fresh_tunnel_url != tunnel_url:
                        logger.info(
                            f"[Tunnel Routing] 🔄 Retrying with fresh tunnel URL from context: {fresh_tunnel_url}"
                        )
                        retry_url = f"{fresh_tunnel_url}/api/terminal/execute"
                        try:
                            with httpx.Client(
                                timeout=float(timeout) / 1000 + 5
                            ) as retry_client:
                                retry_response = retry_client.post(
                                    retry_url,
                                    json=request_data,
                                    headers={"Content-Type": "application/json"},
                                )
                                if retry_response.status_code == 200:
                                    result = retry_response.json()
                                    logger.info(
                                        f"[Tunnel Routing] ✅ Terminal command succeeded with fresh tunnel URL"
                                    )
                                    # Update cache with fresh URL
                                    tunnel_service.register_tunnel(
                                        user_id=user_id,
                                        tunnel_url=fresh_tunnel_url,
                                        conversation_id=conversation_id,
                                    )
                                    return result, None
                                else:
                                    logger.warning(
                                        f"[Tunnel Routing] ❌ Fresh tunnel URL also failed: {retry_response.status_code}"
                                    )
                        except Exception as retry_e:
                            logger.warning(
                                f"[Tunnel Routing] ❌ Retry with fresh URL failed: {retry_e}"
                            )

                    # If no fresh URL or retry failed, try user-level fallback
                    if conversation_id:
                        user_level_tunnel = tunnel_service.get_tunnel_url(user_id, None)
                        if user_level_tunnel and user_level_tunnel != tunnel_url:
                            logger.info(
                                f"[Tunnel Routing] 🔄 Retrying with user-level tunnel: {user_level_tunnel}"
                            )
                            retry_url = f"{user_level_tunnel}/api/terminal/execute"
                            try:
                                with httpx.Client(
                                    timeout=float(timeout) / 1000 + 5
                                ) as retry_client:
                                    retry_response = retry_client.post(
                                        retry_url,
                                        json=request_data,
                                        headers={"Content-Type": "application/json"},
                                    )
                                    if retry_response.status_code == 200:
                                        result = retry_response.json()
                                        logger.info(
                                            f"[Tunnel Routing] ✅ Terminal command succeeded with user-level tunnel"
                                        )
                                        return result, None
                            except Exception as retry_e:
                                logger.warning(
                                    f"[Tunnel Routing] ❌ User-level tunnel retry failed: {retry_e}"
                                )

                    return None, "tunnel_unreachable"
                else:
                    logger.warning(
                        f"[Tunnel Routing] ❌ Terminal command failed ({response.status_code}): {error_text[:500]}"
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
            # Unregister both conversation-specific and user-level tunnels
            # since they may both be expired (Cloudflare tunnels are ephemeral)
            if user_id:
                try:
                    from app.modules.tunnel.tunnel_service import get_tunnel_service

                    tunnel_service = get_tunnel_service()
                    # Unregister conversation-specific tunnel
                    if conversation_id:
                        tunnel_service.unregister_tunnel(user_id, conversation_id)
                        logger.info(
                            f"[Tunnel Routing] Unregistered expired conversation tunnel for user {user_id}"
                        )
                    # Also unregister user-level tunnel (fallback)
                    tunnel_service.unregister_tunnel(user_id, None)
                    logger.info(
                        f"[Tunnel Routing] Unregistered expired user-level tunnel for user {user_id}"
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
            f"[Tunnel Routing] ❌ Timeout connecting to tunnel (tunnel may be unreachable): {e}"
        )
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

        # Get tunnel_url from context if available (takes priority)
        from app.modules.intelligence.tools.code_changes_manager import _get_tunnel_url

        context_tunnel_url = _get_tunnel_url()
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id, conversation_id, tunnel_url=context_tunnel_url
        )

        if not tunnel_url:
            logger.debug(f"No tunnel available for user {user_id}")
            return None

        url = f"{tunnel_url}/api/terminal/sessions/{session_id}/output"
        logger.debug(
            f"[Tunnel Routing] Getting session output: {url}, offset: {offset}"
        )

        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                url,
                params={"offset": offset},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                logger.debug(f"[Tunnel Routing] ✅ Got session output")
                return result
            else:
                error_text = response.text

                # Detect Cloudflare tunnel errors
                is_cloudflare_error = _is_cloudflare_tunnel_error(
                    response.status_code, error_text
                )

                if is_cloudflare_error:
                    logger.warning(
                        f"[Tunnel Routing] ❌ Cloudflare tunnel is not reachable when getting session output"
                    )
                else:
                    logger.warning(
                        f"[Tunnel Routing] ❌ Failed to get session output ({response.status_code}): {error_text[:500]}"
                    )
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

        # Get tunnel_url from context if available (takes priority)
        from app.modules.intelligence.tools.code_changes_manager import _get_tunnel_url

        context_tunnel_url = _get_tunnel_url()
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(
            user_id, conversation_id, tunnel_url=context_tunnel_url
        )

        if not tunnel_url:
            logger.debug(f"No tunnel available for user {user_id}")
            return None

        url = f"{tunnel_url}/api/terminal/sessions/{session_id}/signal"
        logger.info(f"[Tunnel Routing] Sending signal {signal} to session {session_id}")

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                url,
                json={"signal": signal},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"[Tunnel Routing] ✅ Signal sent successfully")
                return result
            else:
                error_text = response.text

                # Detect Cloudflare tunnel errors
                is_cloudflare_error = _is_cloudflare_tunnel_error(
                    response.status_code, error_text
                )

                if is_cloudflare_error:
                    logger.warning(
                        f"[Tunnel Routing] ❌ Cloudflare tunnel is not reachable when sending signal"
                    )
                else:
                    logger.warning(
                        f"[Tunnel Routing] ❌ Failed to send signal ({response.status_code}): {error_text[:500]}"
                    )
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
