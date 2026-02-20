"""
User Local Tunnel Provider

Code provider implementation that communicates with LocalServer via tunnel/extension.
This provider routes file operations through the VS Code extension's tunnel connection.
"""

import httpx
from typing import Any, Dict, List, Optional
from urllib.parse import quote as url_quote

from app.modules.code_provider.base.code_provider_interface import (
    AuthMethod,
    ICodeProvider,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class UserLocalTunnelProvider(ICodeProvider):
    """
    Code provider that uses tunnel/extension to access local workspace files.
    
    This provider communicates with LocalServer running in the VS Code extension
    via Cloudflare tunnel. It provides access to the user's local workspace
    without requiring direct filesystem access.
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        tunnel_url: Optional[str] = None,
    ):
        """
        Initialize tunnel provider.

        Args:
            user_id: User ID for tunnel lookup (optional, can be set later)
            conversation_id: Conversation ID for tunnel lookup (optional)
            tunnel_url: Direct tunnel URL (optional, will be looked up if not provided)
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.tunnel_url = tunnel_url
        self.client = None  # No client object for tunnel provider

    def _get_tunnel_url(self) -> Optional[str]:
        """Get tunnel URL from tunnel service."""
        if self.tunnel_url:
            return self.tunnel_url

        if not self.user_id:
            logger.debug("No user_id provided for tunnel lookup")
            return None

        try:
            from app.modules.tunnel.tunnel_service import get_tunnel_service

            tunnel_service = get_tunnel_service()
            url = tunnel_service.get_tunnel_url(self.user_id, self.conversation_id)
            if url:
                self.tunnel_url = url
            return url
        except Exception as e:
            logger.warning(f"Failed to get tunnel URL: {e}")
            return None

    def _make_tunnel_request(
        self, method: str, endpoint: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to LocalServer via tunnel.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., '/api/files/read')
            params: Query parameters
            json_data: JSON body for POST requests

        Returns:
            Response JSON dict or None if request failed
        """
        tunnel_url = self._get_tunnel_url()
        if not tunnel_url:
            logger.debug("No tunnel URL available")
            return None

        url = f"{tunnel_url}{endpoint}"
        
        # Add query parameters if provided
        if params:
            query_string = "&".join([f"{k}={url_quote(str(v))}" for k, v in params.items()])
            url = f"{url}?{query_string}"

        logger.debug(f"[UserLocalTunnelProvider] Making {method} request to {url}")

        try:
            with httpx.Client(timeout=30.0) as client:
                if method.upper() == "GET":
                    response = client.get(url)
                elif method.upper() == "POST":
                    response = client.post(url, json=json_data)
                else:
                    logger.warning(f"Unsupported HTTP method: {method}")
                    return None

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"[UserLocalTunnelProvider] Request failed: {response.status_code} - {response.text[:200]}"
                    )
                    return None
        except Exception as e:
            logger.warning(f"[UserLocalTunnelProvider] Request error: {e}")
            return None

    # ============ Authentication ============

    def authenticate(self, credentials: Dict[str, Any], method: AuthMethod) -> Any:
        """Authentication is handled via tunnel connection - no additional auth needed."""
        logger.debug("UserLocalTunnelProvider.authenticate called; tunnel handles auth")
        return None

    def get_supported_auth_methods(self) -> List[AuthMethod]:
        """Tunnel provider doesn't require traditional authentication."""
        return []

    # ============ Repository Operations ============

    def get_repository(self, repo_name: str) -> Dict[str, Any]:
        """
        Get repository metadata from local workspace.

        Args:
            repo_name: Workspace identifier (not used for tunnel, but kept for interface compatibility)

        Returns:
            Dictionary with repository metadata
        """
        # For tunnel provider, we return minimal metadata
        # The actual workspace is accessed via the extension
        return {
            "id": "local-workspace",
            "name": "local-workspace",
            "full_name": "local-workspace",
            "owner": "user",
            "default_branch": "main",
            "private": True,
            "url": "local://workspace",
            "description": "Local workspace accessed via tunnel",
        }

    def check_repository_access(self, repo_name: str) -> bool:
        """Check if tunnel is available and workspace is accessible."""
        tunnel_url = self._get_tunnel_url()
        if not tunnel_url:
            return False

        # Try a simple request to verify tunnel is working
        result = self._make_tunnel_request("GET", "/api/files/structure", params={"path": ""})
        return result is not None and result.get("success", False)

    # ============ Content Operations ============

    def get_file_content(
        self,
        repo_name: str,
        file_path: str,
        ref: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """
        Get file content from local workspace via tunnel.

        Args:
            repo_name: Not used (kept for interface compatibility)
            file_path: Path to file relative to workspace root
            ref: Git reference (not used for tunnel provider)
            start_line: Optional start line (1-based, inclusive)
            end_line: Optional end line (inclusive)

        Returns:
            File content as string
        """
        result = self._make_tunnel_request(
            "GET",
            "/api/files/read",
            params={"path": file_path},
        )

        if not result or not result.get("success"):
            error = result.get("error", "Unknown error") if result else "No response from tunnel"
            raise FileNotFoundError(f"Failed to read file '{file_path}': {error}")

        content = result.get("content", "")

        # Apply line filtering if specified
        if start_line is not None or end_line is not None:
            lines = content.split("\n")
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else len(lines)
            content = "\n".join(lines[start_idx:end_idx])

        return content

    def get_repository_structure(
        self,
        repo_name: str,
        path: str = "",
        ref: Optional[str] = None,
        max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Get repository directory structure from local workspace via tunnel.

        Args:
            repo_name: Not used (kept for interface compatibility)
            path: Directory path relative to workspace root (empty for root)
            ref: Git reference (not used for tunnel provider)
            max_depth: Maximum depth to traverse (not fully enforced by tunnel API)

        Returns:
            List of file/directory dictionaries
        """
        # Build params for structure request
        params = {}
        if path:
            params["path"] = path
        if max_depth:
            params["max_depth"] = str(max_depth)
        
        result = self._make_tunnel_request(
            "GET",
            "/api/files/structure",
            params=params,
        )

        if not result or not result.get("success"):
            error = result.get("error", "Unknown error") if result else "No response from tunnel"
            logger.warning(f"Failed to get structure for path '{path}': {error}")
            return []

        # The structure endpoint returns a nested object structure
        # Format it to match the expected string format (like LocalRepoService does)
        structure_obj = result.get("structure", {})
        
        if not structure_obj:
            logger.warning(f"Empty structure returned for path '{path}'")
            return []
        
        # Format the structure object to a string
        structure_str = self._format_tree_structure(structure_obj)
        
        # Return as a single formatted string entry
        # This matches what the code_provider_service expects (string format)
        return [{"path": path, "structure": structure_str}]
    
    def _format_tree_structure(self, structure: Dict[str, Any]) -> str:
        """
        Format nested structure object to indented string format.
        
        Matches the format used by LocalRepoService and GithubService.
        
        Args:
            structure: Dictionary with 'name' and 'children' keys
            
        Returns:
            Formatted string with indented hierarchy
        """
        def _format_node(node: Dict[str, Any], depth: int = 0) -> List[str]:
            output = []
            indent = "  " * depth
            
            # Skip root name if it's the workspace root
            if depth > 0:
                output.append(f"{indent}{node.get('name', '')}")
            
            # Process children if present
            children = node.get("children", [])
            if children:
                # Sort: directories first, then files, both alphabetically
                sorted_children = sorted(
                    children,
                    key=lambda x: (x.get("type") != "directory", x.get("name", "").lower())
                )
                for child in sorted_children:
                    output.extend(_format_node(child, depth + 1))
            
            return output
        
        return "\n".join(_format_node(structure))

    # ============ Branch Operations ============

    def list_branches(self, repo_name: str) -> List[str]:
        """List branches - not supported for tunnel provider."""
        logger.warning("list_branches not supported for UserLocalTunnelProvider")
        return ["main"]  # Return default branch

    def get_branch(self, repo_name: str, branch_name: str) -> Dict[str, Any]:
        """Get branch details - not supported for tunnel provider."""
        logger.warning("get_branch not supported for UserLocalTunnelProvider")
        return {
            "name": branch_name or "main",
            "commit_sha": "local",
            "protected": False,
        }

    def create_branch(
        self, repo_name: str, branch_name: str, base_branch: str
    ) -> Dict[str, Any]:
        """Create branch - not supported for tunnel provider."""
        raise NotImplementedError("Branch operations not supported for tunnel provider")

    def compare_branches(
        self, repo_name: str, base_branch: str, head_branch: str
    ) -> Dict[str, Any]:
        """Compare branches - not supported for tunnel provider."""
        raise NotImplementedError("Branch operations not supported for tunnel provider")

    # ============ Pull Request Operations ============

    def list_pull_requests(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List PRs - not supported for tunnel provider."""
        raise NotImplementedError("Pull request operations not supported for tunnel provider")

    def get_pull_request(
        self, repo_name: str, pr_number: int, include_diff: bool = False
    ) -> Dict[str, Any]:
        """Get PR - not supported for tunnel provider."""
        raise NotImplementedError("Pull request operations not supported for tunnel provider")

    def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
        reviewers: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create PR - not supported for tunnel provider."""
        raise NotImplementedError("Pull request operations not supported for tunnel provider")

    def add_pull_request_comment(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        commit_id: Optional[str] = None,
        path: Optional[str] = None,
        line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Add PR comment - not supported for tunnel provider."""
        raise NotImplementedError("Pull request operations not supported for tunnel provider")

    def create_pull_request_review(
        self,
        repo_name: str,
        pr_number: int,
        body: str,
        event: str,
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create PR review - not supported for tunnel provider."""
        raise NotImplementedError("Pull request operations not supported for tunnel provider")

    # ============ Issue Operations ============

    def list_issues(
        self, repo_name: str, state: str = "open", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List issues - not supported for tunnel provider."""
        raise NotImplementedError("Issue operations not supported for tunnel provider")

    def get_issue(self, repo_name: str, issue_number: int) -> Dict[str, Any]:
        """Get issue - not supported for tunnel provider."""
        raise NotImplementedError("Issue operations not supported for tunnel provider")

    def create_issue(
        self, repo_name: str, title: str, body: str, labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create issue - not supported for tunnel provider."""
        raise NotImplementedError("Issue operations not supported for tunnel provider")

    # ============ File Modification Operations ============

    def create_or_update_file(
        self,
        repo_name: str,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update file - not supported for tunnel provider (read-only)."""
        raise NotImplementedError(
            "File modification operations not supported for tunnel provider (read-only)"
        )

    # ============ User/Organization Operations ============

    def list_user_repositories(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List user repos - not supported for tunnel provider."""
        logger.warning("list_user_repositories not supported for UserLocalTunnelProvider")
        return []

    def get_user_organizations(self) -> List[Dict[str, Any]]:
        """Get user orgs - not supported for tunnel provider."""
        logger.warning("get_user_organizations not supported for UserLocalTunnelProvider")
        return []

    # ============ Provider Metadata ============

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "user_local_tunnel"

    def get_api_base_url(self) -> str:
        """Return tunnel URL as base URL."""
        tunnel_url = self._get_tunnel_url()
        return tunnel_url or "tunnel://local"

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit info - not applicable for tunnel provider."""
        return {
            "limit": float("inf"),
            "remaining": float("inf"),
            "reset_at": None,
        }
