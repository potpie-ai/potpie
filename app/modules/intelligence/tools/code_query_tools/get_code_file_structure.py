import asyncio
import logging
from typing import Optional
from urllib.parse import quote as url_quote

import httpx
from app.modules.intelligence.tools.tool_schema import OnyxTool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.intelligence.tools.tool_utils import truncate_response

logger = logging.getLogger(__name__)

# Character limit for tool responses to prevent sending insanely large content to LLM
MAX_RESPONSE_LENGTH = 80000  # 80k characters


class RepoStructureRequest(BaseModel):
    project_id: str
    path: Optional[str] = None


class GetCodeFileStructureTool:
    name = "get_code_file_structure"
    description = """Retrieve the hierarchical file structure of a specified repository.
        :param project_id: string, the repository ID (UUID) to get the file structure for.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000"
            }

        Returns string containing the hierarchical file structure.

        ⚠️ IMPORTANT: Large repositories may result in truncated responses (max 80,000 characters).
        If the response is truncated, a notice will be included indicating the truncation occurred.
        """

    def __init__(self, db: Session):
        self.cp_service = CodeProviderService(db)

    def _format_tree_structure(self, structure: dict) -> str:
        """
        Format nested structure object to indented string format.

        Matches the format used by LocalRepoService and GithubService.

        Args:
            structure: Dictionary with 'name' and 'children' keys

        Returns:
            Formatted string with indented hierarchy
        """

        def _format_node(node: dict, depth: int = 0) -> list:
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
                    key=lambda x: (
                        x.get("type") != "directory",
                        x.get("name", "").lower(),
                    ),
                )
                for child in sorted_children:
                    output.extend(_format_node(child, depth + 1))

            return output

        return "\n".join(_format_node(structure))

    def _try_local_server(self, path: Optional[str] = None) -> Optional[str]:
        """Try to fetch directory structure from LocalServer via tunnel or Socket.IO (local-first approach)"""
        try:
            from app.modules.tunnel.tunnel_service import get_tunnel_service
            from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
                get_context_vars,
                _execute_via_socket,
                SOCKET_TUNNEL_PREFIX,
            )
            from app.modules.intelligence.tools.code_changes_manager import (
                _get_repository,
                _get_branch,
            )

            user_id, conversation_id = get_context_vars()
            repository = _get_repository()
            branch = _get_branch()

            if not user_id or not conversation_id:
                logger.debug(
                    "[get_code_file_structure] No user/conversation context for local routing"
                )
                return None

            tunnel_service = get_tunnel_service()
            tunnel_url = tunnel_service.get_tunnel_url(
                user_id, conversation_id, repository=repository, branch=branch
            )

            if not tunnel_url:
                logger.debug(
                    f"[get_code_file_structure] No tunnel available for user {user_id}"
                )
                return None

            # Socket path: use Socket.IO RPC
            if tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
                logger.info("[get_code_file_structure] 🚀 Routing to LocalServer via Socket.IO")
                payload = {"path": path} if path else {}
                out = _execute_via_socket(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    endpoint="/api/files/structure",
                    payload=payload,
                    tunnel_url=tunnel_url,
                    repository=repository,
                    branch=branch,
                    timeout=30.0,
                )
                if not isinstance(out, dict) or not out.get("success"):
                    return None
                structure_obj = out.get("structure", {})
                if structure_obj:
                    structure = self._format_tree_structure(structure_obj)
                    logger.info("[get_code_file_structure] ✅ LocalServer returned directory structure via Socket.IO")
                    return structure
                return None

            # HTTP path
            url = f"{tunnel_url}/api/files/structure"
            if path:
                url_with_params = f"{url}?path={url_quote(path)}"
            else:
                url_with_params = url

            logger.info(
                f"[get_code_file_structure] 🚀 Routing to LocalServer: {url_with_params}"
            )

            with httpx.Client(timeout=30.0) as client:
                response = client.get(url_with_params)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        # The structure endpoint returns a nested object, format it to string
                        structure_obj = result.get("structure", {})
                        if structure_obj:
                            structure = self._format_tree_structure(structure_obj)
                            logger.info(
                                "[get_code_file_structure] ✅ LocalServer returned directory structure"
                            )
                            return structure
                        else:
                            logger.warning(
                                "[get_code_file_structure] Empty structure returned"
                            )
                            return None
                    else:
                        logger.debug(
                            f"[get_code_file_structure] LocalServer returned error: {result.get('error')}"
                        )
                        return None
                else:
                    status_code = response.status_code
                    error_text = response.text

                    # Detect tunnel/connection errors
                    is_tunnel_error = (
                        status_code in [502, 503, 504, 530]
                        or ("tunnel" in error_text.lower() and "error" in error_text.lower())
                    )

                    if is_tunnel_error:
                        logger.warning(
                            f"[get_code_file_structure] ❌ Stale tunnel detected ({status_code}): {tunnel_url}. "
                            f"Invalidating tunnel URL and falling back to remote."
                        )

                        # Invalidate the stale tunnel URL
                        try:
                            tunnel_service.unregister_tunnel(user_id, conversation_id)
                            logger.info(
                                f"[get_code_file_structure] ✅ Invalidated stale tunnel for user {user_id}"
                            )
                        except Exception as e:
                            logger.error(
                                f"[get_code_file_structure] Failed to invalidate tunnel: {e}"
                            )

                    logger.debug(
                        f"[get_code_file_structure] LocalServer returned {status_code}"
                    )
                    return None

        except Exception as e:
            logger.debug(f"[get_code_file_structure] Local routing failed: {e}")
            return None

    async def fetch_repo_structure(
        self, project_id: str, path: Optional[str] = None
    ) -> str:
        # LOCAL-FIRST: Try LocalServer via tunnel first
        local_result = self._try_local_server(path)
        if local_result:
            return local_result

        # Fall back to remote/CodeProviderService
        logger.debug(
            f"[get_code_file_structure] Falling back to remote for project {project_id}, path={path}"
        )
        return await self.cp_service.get_project_structure_async(project_id, path)

    async def arun(self, project_id: str, path: Optional[str] = None) -> str:
        try:
            result = await self.fetch_repo_structure(project_id, path)
            truncated_result = truncate_response(result)
            if len(result) > 80000:
                logger.warning(
                    f"get_code_file_structure output truncated from {len(result)} to 80000 characters "
                    f"for project {project_id}, path={path}"
                )
            return truncated_result
        except Exception as e:
            logger.error(f"Error fetching file structure: {e}")
            return "error fetching data"

    def run(self, project_id: str, path: Optional[str] = None) -> str:
        try:
            result = asyncio.run(self.fetch_repo_structure(project_id, path))
            truncated_result = truncate_response(result)
            if len(result) > 80000:
                logger.warning(
                    f"get_code_file_structure output truncated from {len(result)} to 80000 characters "
                    f"for project {project_id}, path={path}"
                )
            return truncated_result
        except Exception as e:
            logger.error(f"Error fetching file structure: {e}")
            return "error fetching data"


def get_code_file_structure_tool(db: Session) -> OnyxTool:
    return OnyxTool(
        name="get_code_file_structure",
        description="""Retrieve the hierarchical file structure of a specified repository or subdirectory in a repository. Expecting 'project_id' as a required input and an optional 'path' to specify a subdirectory. If no path is provided, it will assume the root by default.
        For input :
        ```
            dir_name
                subdir_name
                    ...
                filename.extension
        ```
        the path for the subdir_name should be dir_name/subdir_name

        ⚠️ IMPORTANT: Large repositories may result in truncated responses (max 80,000 characters).
        If the response is truncated, a notice will be included indicating the truncation occurred.""",
        coroutine=GetCodeFileStructureTool(db).arun,
        func=GetCodeFileStructureTool(db).run,
        args_schema=RepoStructureRequest,
    )
