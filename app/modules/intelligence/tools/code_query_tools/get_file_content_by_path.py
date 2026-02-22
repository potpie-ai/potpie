from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
from typing import List, Optional, Type, Dict, Any
from urllib.parse import quote as url_quote
from pydantic import BaseModel, Field
from redis import Redis
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_service import ProjectService
from app.core.config_provider import config_provider
from app.modules.intelligence.tools.tool_utils import truncate_response
import httpx


class FetchFileToolInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    file_path: str = Field(..., description="Path to the file within the repo")
    start_line: Optional[int] = Field(
        None, description="First line to fetch (1-based, inclusive)"
    )
    end_line: Optional[int] = Field(None, description="Last line to fetch (inclusive)")


class FetchFilesBatchToolInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    paths: List[str] = Field(
        ...,
        description="List of file paths to read (e.g. ['src/App.tsx', 'src/utils.ts']). Use 2–20 paths per call for efficiency.",
        min_length=1,
        max_length=20,
    )
    with_line_numbers: bool = Field(
        False,
        description="Whether to prefix each line with its line number (e.g. '1:...', '2:...')",
    )


class FetchFileTool:
    name: str = "fetch_file"
    description: str = """Fetch a single file's content from a repository using project_id and file path.
        For 2+ files at once, prefer fetch_files_batch (single round-trip; local mode uses POST /api/files/read-batch).
        Returns the content between optional start_line and end_line.
        Make sure the file exists before querying for it, confirm it by checking the file structure.
        File content is hashed for caching purposes. Cache won't be used if start_line or end_line are different.
        Use with_line_numbers to include line numbers in the response to better understand the context and location of the code.

        IMPORTANT LIMITS:
        - Maximum 1200 lines can be fetched at once
        - If the entire file is requested and it has more than 1200 lines, an error will be returned
        - If start_line and end_line span more than 1200 lines, an error will be returned
        - Always use start_line and end_line to fetch specific sections of large files
        - Maximum 80,000 characters per response (content will be truncated with a notice if exceeded)

        param project_id: string, the repository ID (UUID) to get the file content for.
        param file_path: string, the path to the file in the repository.
        param with_line_numbers: bool, whether to include line numbers in the response.
        param start_line: int, the first line to fetch (1-based, inclusive).
        param end_line: int, the last line to fetch (inclusive).

        IMPORTANT: Use line numbers as much as possible, some files are large. Use other tools to access small parts of the file.
        You can use knowledge graph tools and node_ids to fetch code snippets

        example:
        {
            "project_id": "550e8400-e29b-41d4-a716-446655440000",
            "file_path": "src/main.py",
            "start_line": 1,
            "end_line": 10
        }
        Returns string containing the content of the file.

        If with_line_numbers is true, the content will be formatted with line numbers, starting from 1.

        format:
        line_number:line

        no extra spaces or tabs in between.

        Example:
        1:def hello_world():
        2:    print("Hello, world!")
        3:
        4:hello_world()

        """
    args_schema: Type[BaseModel] = FetchFileToolInput

    def __init__(self, sql_db: Session, user_id: str, internal_call: bool = False):
        self.sql_db = sql_db
        self.user_id = user_id
        self.cp_service = CodeProviderService(self.sql_db)
        self.project_service = ProjectService(self.sql_db)
        self.redis = Redis.from_url(config_provider.get_redis_url())
        self.internal_call = internal_call

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        details = self.project_service.get_project_from_db_by_id_sync(project_id)
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def with_line_numbers(
        self, content: str, include_line_number: bool, starting_line: int
    ) -> str:
        if include_line_number:
            lines = content.splitlines()
            numbered_lines = [
                f"{starting_line + i}:{line}" for i, line in enumerate(lines)
            ]
            return "\n".join(numbered_lines)
        return content

    def _check_line_limits(
        self, content: str, start_line: Optional[int], end_line: Optional[int]
    ) -> Dict[str, Any]:
        """Check if the content or requested range exceeds 1200 lines"""
        lines = content.splitlines()
        total_lines = len(lines)

        # If specific lines are requested, check the range
        if start_line is not None and end_line is not None:
            requested_lines = end_line - start_line + 1
            if requested_lines > 1200 and not self.internal_call:
                return {
                    "success": False,
                    "error": f"Too much content requested. You asked for {requested_lines} lines (lines {start_line}-{end_line}), but maximum allowed is 1200 lines. Please make your request more targeted by specifying a smaller line range.",
                    "content": None,
                    "total_lines": total_lines,
                }
        # If no specific lines requested and total file exceeds 1200 lines
        elif (
            start_line is None
            and end_line is None
            and total_lines > 1200
            and not self.internal_call
        ):
            return {
                "success": False,
                "error": f"Too much content requested. The file has {total_lines} lines, but maximum allowed is 1200 lines. Please specify start_line and end_line parameters to fetch a smaller section of the file.",
                "content": None,
                "total_lines": total_lines,
            }

        return {"success": True}

    def _try_local_server(
        self,
        file_path: str,
        with_line_numbers: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Try to fetch file from LocalServer via tunnel or Socket.IO (local-first approach)"""
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
                    "[fetch_file] No user/conversation context for local routing"
                )
                return None

            tunnel_service = get_tunnel_service()
            tunnel_url = tunnel_service.get_tunnel_url(
                user_id, conversation_id, repository=repository, branch=branch
            )

            logger.debug(
                f"[fetch_file] 🔍 Tunnel lookup result: tunnel_url={tunnel_url}"
            )

            if not tunnel_url:
                logger.debug(
                    f"[fetch_file] ❌ No tunnel available for user {user_id}, conversation {conversation_id} - falling back to GitHub"
                )
                return None

            # Socket path: use Socket.IO RPC
            if tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
                logger.info(f"[fetch_file] 🚀 Routing to LocalServer via Socket.IO")
                out = _execute_via_socket(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    endpoint="/api/files/read",
                    payload={"path": file_path},
                    tunnel_url=tunnel_url,
                    repository=repository,
                    branch=branch,
                    timeout=30.0,
                )
                if not isinstance(out, dict) or not out.get("success"):
                    return None
                content = out.get("content", "")
                if start_line is not None or end_line is not None:
                    lines = content.split("\n")
                    start_idx = (start_line - 1) if start_line else 0
                    end_idx = end_line if end_line else len(lines)
                    content = "\n".join(lines[start_idx:end_idx])
                if with_line_numbers:
                    starting_line = start_line or 1
                    content = self.with_line_numbers(content, True, starting_line)
                content = truncate_response(content)
                logger.info(
                    f"[fetch_file] ✅ LocalServer returned file content via Socket.IO"
                )
                return {"success": True, "content": content, "source": "local"}

            # HTTP path
            url = f"{tunnel_url}/api/files/read"
            url_with_params = f"{url}?path={url_quote(file_path)}"

            logger.info(f"[fetch_file] 🚀 Routing to LocalServer: {url_with_params}")

            with httpx.Client(timeout=30.0) as client:
                response = client.get(url_with_params)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        content = result.get("content", "")

                        # Apply line filtering if needed
                        if start_line is not None or end_line is not None:
                            lines = content.split("\n")
                            start_idx = (start_line - 1) if start_line else 0
                            end_idx = end_line if end_line else len(lines)
                            content = "\n".join(lines[start_idx:end_idx])

                        # Apply line numbers
                        if with_line_numbers:
                            starting_line = start_line or 1
                            content = self.with_line_numbers(
                                content, True, starting_line
                            )

                        # Truncate if needed
                        content = truncate_response(content)

                        logger.info(
                            f"[fetch_file] ✅ LocalServer returned file content"
                        )
                        return {
                            "success": True,
                            "content": content,
                            "source": "local",
                        }
                    else:
                        logger.debug(
                            f"[fetch_file] LocalServer returned error: {result.get('error')}"
                        )
                        return None
                else:
                    status_code = response.status_code
                    error_text = response.text

                    # Detect tunnel/connection errors
                    is_tunnel_error = status_code in [502, 503, 504, 530] or (
                        "tunnel" in error_text.lower() and "error" in error_text.lower()
                    )

                    if is_tunnel_error:
                        logger.warning(
                            f"[fetch_file] ❌ Stale tunnel detected ({status_code}): {tunnel_url}. "
                            f"Invalidating conversation tunnel (workspace-only)."
                        )

                        # Invalidate the stale tunnel URL (workspace-only; no user-level)
                        try:
                            tunnel_service.unregister_tunnel(user_id, conversation_id)
                            logger.info(
                                f"[fetch_file] ✅ Invalidated stale conversation tunnel for user {user_id}"
                            )
                        except Exception as e:
                            logger.error(
                                f"[fetch_file] Failed to invalidate tunnel: {e}"
                            )

                    logger.debug(f"[fetch_file] LocalServer returned {status_code}")
                    return None

        except Exception as e:
            logger.debug(f"[fetch_file] Local routing failed: {e}")
            return None

    def _run(
        self,
        project_id: str,
        file_path: str,
        with_line_numbers: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            # LOCAL-FIRST: Try LocalServer via tunnel first
            local_result = self._try_local_server(
                file_path, with_line_numbers, start_line, end_line
            )
            if local_result:
                return local_result

            # Fall back to GitHub/remote
            logger.debug(f"[fetch_file] Falling back to remote for {file_path}")

            details = self._get_project_details(project_id)
            # Modify cache key to reflect that we're only caching the specific path
            cache_key = f"file_content:{project_id}:exact_path_{file_path}:start_line_{start_line}:end_line_{end_line}"
            result = self.redis.get(cache_key)

            if result:
                cached_content = result.decode("utf-8")
                # Check line limits for cached content
                limit_check = self._check_line_limits(
                    cached_content, start_line, end_line
                )
                if not limit_check["success"]:
                    return limit_check

                content = self.with_line_numbers(
                    cached_content,
                    with_line_numbers,
                    starting_line=start_line or 1,
                )

                # Truncate content if it exceeds character limits
                original_length = len(content)
                content = truncate_response(content)
                if len(content) > 80000:
                    logger.warning(
                        f"fetch_file (cached) output truncated from {original_length} to 80000 characters "
                        f"for file {file_path}, project_id={project_id}"
                    )

                return {
                    "success": True,
                    "content": content,
                }

            # Use repo_path for local repos, project_name for remote repos
            repo_identifier = details.get("repo_path") or details["project_name"]

            try:
                content = self.cp_service.get_file_content(
                    repo_name=repo_identifier,
                    file_path=file_path,
                    branch_name=details["branch_name"],
                    start_line=start_line if start_line is not None else None,
                    end_line=end_line if end_line is not None else None,
                    project_id=project_id,
                    commit_id=details["commit_id"],
                )
            except Exception as github_error:
                # GitHub failed (likely 404) - try local server as fallback
                # This handles the common case where file exists locally but isn't pushed yet
                error_str = str(github_error).lower()
                is_not_found = "404" in error_str or "not found" in error_str

                if is_not_found:
                    logger.info(
                        f"[fetch_file] GitHub 404 for {file_path}, trying LocalServer fallback"
                    )
                    local_result = self._try_local_server(
                        file_path, with_line_numbers, start_line, end_line
                    )
                    if local_result and local_result.get("success"):
                        logger.info(
                            f"[fetch_file] ✅ LocalServer fallback succeeded for {file_path}"
                        )
                        return local_result

                    # Both GitHub and local failed - provide helpful error
                    return {
                        "success": False,
                        "error": (
                            f"File '{file_path}' not found.\n\n"
                            f"Checked:\n"
                            f"1. GitHub repository: File not found (404) - may not be pushed yet\n"
                            f"2. Local workspace: {'Not available (no tunnel connection)' if not local_result else 'File not found locally'}\n\n"
                            f"Possible solutions:\n"
                            f"- If working locally: Ensure the VS Code extension is connected and the file exists in your workspace\n"
                            f"- If expecting from GitHub: Push your changes or verify the file path is correct\n"
                        ),
                        "content": None,
                    }
                else:
                    # Re-raise non-404 errors
                    raise github_error

            # Check line limits for new content
            limit_check = self._check_line_limits(content, start_line, end_line)
            if not limit_check["success"]:
                return limit_check

            self.redis.setex(cache_key, 600, content)  # Cache for 10 minutes
            content = self.with_line_numbers(
                content, with_line_numbers, starting_line=start_line or 1
            )

            # Truncate content if it exceeds character limits
            original_length = len(content)
            content = truncate_response(content)
            if len(content) > 80000:
                logger.warning(
                    f"fetch_file output truncated from {original_length} to 80000 characters "
                    f"for file {file_path}, project_id={project_id}"
                )

            return {
                "success": True,
                "content": content,
            }
        except FileNotFoundError as e:
            # File not found - try local as last resort
            logger.warning(
                f"File not found in remote: {file_path} - trying local fallback"
            )
            local_result = self._try_local_server(
                file_path, with_line_numbers, start_line, end_line
            )
            if local_result and local_result.get("success"):
                return local_result

            return {
                "success": False,
                "error": (
                    f"File '{file_path}' not found in repository or locally.\n"
                    f"Please verify the file path is correct or check if the file has been created yet."
                ),
                "content": None,
            }
        except Exception as e:
            logger.exception(f"Failed to fetch file content for {file_path}")
            return {"success": False, "error": str(e), "content": None}

    async def _arun(
        self,
        project_id: str,
        file_path: str,
        with_line_numbers: bool = False,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Synchronously for compatibility, like GithubTool
        return self._run(project_id, file_path, with_line_numbers, start_line, end_line)


def fetch_file_tool(
    sql_db: Session, user_id: str, internal_call: bool = False
) -> StructuredTool:
    tool_instance = FetchFileTool(sql_db, user_id, internal_call=internal_call)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="fetch_file",
        description=tool_instance.description,
        args_schema=FetchFileToolInput,
    )


# --------------------------------------------------------------------------- #
# fetch_files_batch: read multiple files in one call (local mode: POST /api/files/read-batch)
# --------------------------------------------------------------------------- #

FETCH_FILES_BATCH_DESCRIPTION = """Read multiple files in one call. Use this when you need 2–20 files at once (e.g. several source files or configs) instead of calling fetch_file repeatedly.

**When connected via the VS Code extension (local mode):** Uses POST /api/files/read-batch for a single round-trip. Response includes one entry per path with `path`, `content`, and `line_count`, or `error` for missing files.

**When not in local mode:** Falls back to fetching each file individually (slower).

**Response format:** `{ "success": true, "files": [ { "path": "src/App.tsx", "content": "...", "line_count": 100 }, { "path": "missing.ts", "error": "File not found" } ] }`

- param project_id: Repository ID (UUID).
- param paths: List of file paths (e.g. ["src/App.tsx", "src/utils.ts"]). Prefer 2–20 paths per call.
- param with_line_numbers: If true, content is prefixed with line numbers (e.g. "1:...", "2:...").

Use fetch_file for a single file; use fetch_files_batch when you need multiple files at once."""


class FetchFilesBatchTool:
    name: str = "fetch_files_batch"
    description: str = FETCH_FILES_BATCH_DESCRIPTION
    args_schema: Type[BaseModel] = FetchFilesBatchToolInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.fetch_file_tool_instance = FetchFileTool(
            sql_db, user_id, internal_call=True
        )

    def _run(
        self,
        project_id: str,
        paths: List[str],
        with_line_numbers: bool = False,
    ) -> Dict[str, Any]:
        from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
            read_files_batch_from_local_server,
        )

        # Local mode: try read-batch first
        batch = read_files_batch_from_local_server(paths)
        if batch is not None and "files" in batch:
            out_files: List[Dict[str, Any]] = []
            for entry in batch["files"]:
                path = entry.get("path", "")
                error = entry.get("error")
                if error is not None:
                    out_files.append({"path": path, "error": error})
                    continue
                content = entry.get("content", "")
                line_count = entry.get("line_count")
                if with_line_numbers and content:
                    content = self.fetch_file_tool_instance.with_line_numbers(
                        content, True, 1
                    )
                content = truncate_response(content)
                item: Dict[str, Any] = {"path": path, "content": content}
                if line_count is not None:
                    item["line_count"] = line_count
                out_files.append(item)
            return {"success": True, "files": out_files}

        # Fallback: no tunnel or read-batch failed — fetch each file
        out_files = []
        for file_path in paths:
            result = self.fetch_file_tool_instance._run(
                project_id=project_id,
                file_path=file_path,
                with_line_numbers=with_line_numbers,
            )
            if result.get("success"):
                content = result.get("content", "")
                line_count = len(content.splitlines()) if content else 0
                out_files.append(
                    {"path": file_path, "content": content, "line_count": line_count}
                )
            else:
                out_files.append(
                    {"path": file_path, "error": result.get("error", "Unknown error")}
                )
        return {"success": True, "files": out_files}

    async def _arun(
        self,
        project_id: str,
        paths: List[str],
        with_line_numbers: bool = False,
    ) -> Dict[str, Any]:
        return self._run(project_id, paths, with_line_numbers)


def fetch_files_batch_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool_instance = FetchFilesBatchTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="fetch_files_batch",
        description=FETCH_FILES_BATCH_DESCRIPTION,
        args_schema=FetchFilesBatchToolInput,
    )
