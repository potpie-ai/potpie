"""
Tunnel utilities for routing search operations to LocalServer.
"""

import json
from typing import Dict, Any, Optional
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


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
        import httpx
        
        if not user_id:
            logger.debug("No user_id in context, skipping tunnel routing")
            return None
        
        tunnel_service = get_tunnel_service()
        tunnel_url = tunnel_service.get_tunnel_url(user_id, conversation_id)
        
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
        
        # Make request to LocalServer via tunnel (sync)
        url = f"{tunnel_url}{endpoint}"
        logger.info(f"[Tunnel Routing] 🚀 Routing {operation} to LocalServer via tunnel: {url}")
        logger.debug(f"[Tunnel Routing] Request data: {request_data}")
        
        with httpx.Client(timeout=30.0) as client:
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
                logger.warning(f"[Tunnel Routing] ❌ LocalServer {operation} failed ({response.status_code}): {error_text}")
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
                'class': '📦', 'function': '🔧', 'method': '⚙️', 'variable': '📝',
                'interface': '🔌', 'enum': '📊', 'type': '🏷️'
            }.get(symbol.get('kind', ''), '•')
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
            file_path = symbol.get('location', {}).get('file_path', 'unknown')
            line = symbol.get('location', {}).get('range', {}).get('start', {}).get('line', '?')
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
            file_path = ref.get('file_path', 'unknown')
            line = ref.get('range', {}).get('start', {}).get('line', '?')
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
            file_path = defn.get('file_path', 'unknown')
            line = defn.get('range', {}).get('start', {}).get('line', '?')
            formatted += f"• {file_path}:{line}\n"
        return formatted
    
    elif operation == "search_files":
        files = result.get("files", [])
        if not files:
            return f"📋 No files found matching pattern '{result.get('pattern')}'"
        
        formatted = f"📋 **Found {len(files)} file(s) matching '{result.get('pattern')}':**\n\n"
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
                line = symbol.get("location", {}).get("range", {}).get("start", {}).get("line", "?")
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
            formatted = f"⚠️ **Bash command failed** (`{command}`, exit code: {exit_code}):\n\n"
            if error:
                formatted += f"Error: {error[:1000]}\n\n"
            if output:
                formatted += f"Output: {output[:1000]}"
            return formatted
        else:
            return f"✅ Command executed: `{command}` (no output)"
    
    # Fallback for unknown operations
    return f"✅ Search completed: {json.dumps(result, indent=2)}"


def get_context_vars():
    """Get user_id and conversation_id from context variables."""
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
