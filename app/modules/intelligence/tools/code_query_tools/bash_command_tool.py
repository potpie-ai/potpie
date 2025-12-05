"""
Bash Command Tool

Allows agents to run bash commands (grep, awk, find, etc.) on the codebase.
Only works if the project's worktree exists in the repo manager.
If the worktree doesn't exist, it will attempt to clone the repository using
the same logic as the parsing flow (supports GitBucket private repos).
"""

import asyncio
import logging
import os
import shlex
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager
from app.modules.utils.gvisor_runner import run_command_isolated, CommandResult
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails

logger = logging.getLogger(__name__)

# SECURITY: Commands that are ALWAYS blocked (write/modify operations)
ALWAYS_BLOCKED_COMMANDS = {
    "rm",
    "rmdir",
    "touch",
    "mkdir",
    "mv",
    "cp",
    "chmod",
    "chown",
    "git",  # Blocked to prevent repository modifications
    "npm",
    "pip",
    "yarn",
    "pnpm",
    "docker",
    "kubectl",
    "curl",
    "wget",
    "nc",
    "netcat",
    "ssh",
    "scp",
    "rsync",
    "sudo",
    "su",
}

# SECURITY: Commands that are blocked when used with write operations
WRITE_BLOCKED_COMMANDS = {
    "sed": ["-i"],  # sed -i modifies files
    "awk": ["-i"],  # awk -i modifies files
}

# SECURITY: Dangerous patterns that indicate write operations
DANGEROUS_PATTERNS = [
    ">",  # Output redirection (write)
    ">>",  # Append redirection (write)
    "sed -i",  # sed in-place editing
    "awk -i",  # awk in-place editing
]

# SECURITY: Command injection patterns (pipes are allowed for read-only filtering)
INJECTION_PATTERNS = [
    "<",  # Input redirection (can be used for injection)
    ";",  # Command separator
    "&&",  # Command chaining
    "||",  # Command chaining
    "`",  # Command substitution
    "$(",  # Command substitution
]


def _validate_command_safety(command: str) -> tuple[bool, Optional[str]]:
    """
    Validate that a command is safe (read-only) and doesn't attempt write operations.

    Args:
        command: The command string to validate

    Returns:
        Tuple of (is_safe, error_message)
        is_safe: True if command is safe, False otherwise
        error_message: Error message if command is unsafe, None if safe
    """
    command_lower = command.lower().strip()

    # Check for write operation patterns (redirection, in-place editing)
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command_lower:
            return (
                False,
                f"Command contains write operation pattern '{pattern}'. Write operations are not allowed.",
            )

    # Check for command injection patterns
    for pattern in INJECTION_PATTERNS:
        if pattern in command:
            return (
                False,
                f"Command contains injection pattern '{pattern}'. Command chaining/substitution is not allowed.",
            )

    # Check if command starts with an always-blocked command
    first_word = command_lower.split()[0] if command_lower.split() else ""
    if first_word in ALWAYS_BLOCKED_COMMANDS:
        return (
            False,
            f"Command '{first_word}' is not allowed. This tool only supports read-only operations.",
        )

    # Check for write-blocked commands with dangerous flags
    for cmd, dangerous_flags in WRITE_BLOCKED_COMMANDS.items():
        if command_lower.startswith(cmd):
            for flag in dangerous_flags:
                if flag in command_lower:
                    return (
                        False,
                        f"Command '{cmd}' with flag '{flag}' is not allowed. This would modify files.",
                    )

    # Block environment variable access that might expose secrets
    if command_lower.strip() == "env":
        return (
            False,
            "The 'env' command is blocked to prevent exposure of sensitive environment variables.",
        )

    # Block commands that try to access parent directories
    if "../" in command or ".." in command.split():
        return (
            False,
            "Accessing parent directories is not allowed for security reasons.",
        )

    return (True, None)


class BashCommandToolInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    command: str = Field(
        ...,
        description="Bash command to execute (e.g., 'grep -r \"function\" .', 'find . -name \"*.py\"', 'awk '/pattern/ {print}' file.txt')",
    )
    working_directory: Optional[str] = Field(
        None,
        description="Optional subdirectory within the repo to run the command. If not specified, runs from repo root.",
    )


class BashCommandTool:
    name: str = "bash_command"
    description: str = (
        """Run bash commands (grep, awk, find, sed, etc.) on the codebase.

        This tool allows you to execute common Unix/bash commands directly on the repository files.
        The command will be executed in the repository's worktree directory using gVisor sandbox isolation
        for enhanced security. Commands run in an isolated environment that prevents filesystem modifications.

        🔒 Security: Commands are executed in a gVisor sandbox, providing strong isolation and preventing
        unauthorized access or modifications to the host system.

        ⚠️ CRITICAL RESTRICTION: ONLY USE READ-ONLY COMMANDS ⚠️
        This tool is designed for read-only operations only. Commands that modify, delete, or write files are NOT supported and may fail or cause unexpected behavior. The gVisor sandbox provides additional protection against accidental modifications.

        IMPORTANT: This tool only works if the repository has been parsed and is available in the repo manager.
        If the worktree doesn't exist, the tool will return an error.

        ✅ ALLOWED (Read-only commands):
        - Search for patterns: grep -r "pattern" .
        - Find files: find . -name "*.py" -type f
        - Process text: awk '/pattern/ {print $1}' file.txt (read-only)
        - Count occurrences: grep -c "pattern" file.txt
        - List files: ls -la directory/
        - Filter output: grep "error" log.txt | head -20 (pipes allowed for filtering)
        - View file contents: cat file.txt, head file.txt, tail file.txt
        - Check file info: stat file.txt, file file.txt
        - Search in files: grep, ag, rg (ripgrep)

        ❌ NOT ALLOWED (Write/modify commands):
        - File modification: echo > file, sed -i, awk -i
        - File creation: touch, mkdir, > file, >> file
        - File deletion: rm, rmdir
        - Git operations: git (all git commands blocked)
        - Package installation: npm, pip, yarn, pnpm
        - Network commands: curl, wget, ssh, scp
        - Command chaining: ; && || (use pipes | for filtering instead)
        - Command substitution: `command` or $(command)
        - Environment access: env (blocked to prevent secret exposure)
        - Any command that modifies the filesystem

        🔒 Security Features:
        - Commands run in gVisor sandbox with read-only filesystem mounts
        - Write operations are blocked at both command validation and filesystem level
        - Environment variables are filtered to prevent secret exposure
        - Network access is disabled in the sandbox
        - Only the specific project's repository is accessible

        Args:
            project_id: The repository ID (UUID) to run the command on
            command: The bash command to execute (MUST be read-only)
            working_directory: Optional subdirectory within the repo (relative path from repo root)

        Returns:
            Dictionary with:
            - success: bool indicating if command succeeded
            - output: Command stdout output
            - error: Command stderr output (if any)
            - exit_code: Command exit code

        Example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000",
                "command": "grep -r \"def main\" . --include=\"*.py\"",
                "working_directory": "src"
            }
        """
    )
    args_schema: type[BaseModel] = BashCommandToolInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.project_service = ProjectService(sql_db)

        # Initialize repo manager if enabled
        self.repo_manager = None
        try:
            repo_manager_enabled = (
                os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
            )
            if repo_manager_enabled:
                self.repo_manager = RepoManager()
                logger.info("BashCommandTool: RepoManager initialized")
        except Exception as e:
            logger.warning(f"BashCommandTool: Failed to initialize RepoManager: {e}")

        # Initialize code provider service and parse helper for cloning repos
        self.code_provider_service = CodeProviderService(sql_db)
        self.parse_helper = ParseHelper(sql_db)

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        """Get project details and validate user access."""
        details = self.project_service.get_project_from_db_by_id_sync(project_id)
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def _get_worktree_path(
        self, repo_name: str, branch: Optional[str], commit_id: Optional[str]
    ) -> Optional[str]:
        """Get the worktree path for the project."""
        if not self.repo_manager:
            return None

        # Try to get worktree path
        worktree_path = self.repo_manager.get_repo_path(
            repo_name, branch=branch, commit_id=commit_id
        )
        if worktree_path and os.path.exists(worktree_path):
            return worktree_path

        # Try with just commit_id
        if commit_id:
            worktree_path = self.repo_manager.get_repo_path(
                repo_name, commit_id=commit_id
            )
            if worktree_path and os.path.exists(worktree_path):
                return worktree_path

        # Try with just branch
        if branch:
            worktree_path = self.repo_manager.get_repo_path(repo_name, branch=branch)
            if worktree_path and os.path.exists(worktree_path):
                return worktree_path

        return None

    async def _clone_repository(
        self,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
        user_id: str,
    ) -> Optional[str]:
        """
        Clone repository using the same logic as the parsing flow.
        Supports GitBucket private repos with authentication.

        Args:
            repo_name: Full repository name (e.g., 'owner/repo')
            branch: Branch name
            commit_id: Commit SHA
            user_id: User ID

        Returns:
            Path to the cloned repository, or None if cloning failed
        """
        logger.info(
            f"[BASH_COMMAND] Attempting to clone repository {repo_name} "
            f"(branch={branch}, commit={commit_id})"
        )

        try:
            # Get the repo object from code provider service
            github, repo = self.code_provider_service.get_repo(repo_name)

            # Extract auth from the Github client
            auth = None
            if hasattr(github, "_Github__requester") and hasattr(
                github._Github__requester, "auth"
            ):
                auth = github._Github__requester.auth

            # Determine target directory for cloning
            target_dir = os.getenv("PROJECT_PATH", "/tmp/repos")
            os.makedirs(target_dir, exist_ok=True)

            # Use the ref (commit_id or branch)
            ref = commit_id if commit_id else branch
            if not ref:
                logger.error(
                    f"[BASH_COMMAND] No branch or commit_id provided for {repo_name}"
                )
                return None

            # Create repo_details for the parse helper
            # Use "HEAD" as placeholder when no branch (commit-only checkout)
            # Explicitly set commit_id in its proper field
            repo_details = RepoDetails(
                repo_name=repo_name,
                branch_name=branch or "HEAD",
                commit_id=commit_id,
            )

            # Download and extract the repository using the same logic as parsing
            extracted_dir = await self.parse_helper.download_and_extract_tarball(
                repo, ref, target_dir, auth, repo_details, user_id
            )

            if not extracted_dir or not os.path.exists(extracted_dir):
                logger.error(
                    f"[BASH_COMMAND] Failed to clone repository {repo_name}: "
                    "extracted directory does not exist"
                )
                return None

            logger.info(
                f"[BASH_COMMAND] Successfully cloned repository {repo_name} to {extracted_dir}"
            )

            # Register the cloned repo with the repo manager
            if self.repo_manager:
                try:
                    # Get repository metadata
                    metadata = {}
                    try:
                        metadata = ParseHelper.extract_repository_metadata(repo)
                    except Exception as meta_error:
                        logger.warning(
                            f"[BASH_COMMAND] Failed to extract repository metadata: {meta_error}"
                        )

                    # Copy to repo manager location using the same logic as parsing
                    await self.parse_helper._copy_repo_to_repo_manager(
                        repo_name,
                        extracted_dir,
                        branch,
                        commit_id,
                        user_id,
                        metadata,
                    )

                    # Get the registered worktree path
                    worktree_path = self._get_worktree_path(repo_name, branch, commit_id)
                    if worktree_path:
                        logger.info(
                            f"[BASH_COMMAND] Registered repo with repo manager at {worktree_path}"
                        )
                        return worktree_path
                except Exception as register_error:
                    logger.warning(
                        f"[BASH_COMMAND] Failed to register repo with repo manager: {register_error}. "
                        "Using extracted directory directly."
                    )

            # If repo manager registration failed or wasn't available, use extracted_dir
            return extracted_dir

        except Exception as e:
            logger.error(f"[BASH_COMMAND] Failed to clone repository {repo_name}: {e}")
            return None

    def _execute_command(
        self,
        project_id: str,
        command: str,
        worktree_path: str,
        working_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute bash command in the given worktree path.
        This is the core execution logic shared by both _run and _arun.

        Args:
            project_id: Project ID for logging
            command: The bash command to execute
            worktree_path: Path to the repository worktree
            working_directory: Optional subdirectory within the repo

        Returns:
            Dictionary with success, output, error, and exit_code
        """
        try:
            # Get project details for logging
            details = self._get_project_details(project_id)
            repo_name = details["project_name"]
            branch = details.get("branch_name")
            commit_id = details.get("commit_id")

            # SECURITY: Normalize paths to prevent directory traversal
            worktree_path = os.path.abspath(worktree_path)

            # SECURITY: Determine working directory and validate it's within the worktree
            if working_directory:
                requested_dir = os.path.normpath(working_directory)
                if os.path.isabs(requested_dir) or ".." in requested_dir:
                    return {
                        "success": False,
                        "error": f"Invalid working directory: '{working_directory}'. Directory traversal is not allowed.",
                        "output": "",
                        "exit_code": -1,
                    }

                cmd_dir = os.path.join(worktree_path, requested_dir)
                cmd_dir = os.path.abspath(cmd_dir)

                if (
                    not cmd_dir.startswith(worktree_path + os.sep)
                    and cmd_dir != worktree_path
                ):
                    return {
                        "success": False,
                        "error": f"Working directory '{working_directory}' is outside the repository boundaries",
                        "output": "",
                        "exit_code": -1,
                    }

                if not os.path.exists(cmd_dir):
                    return {
                        "success": False,
                        "error": f"Working directory '{working_directory}' does not exist in the repository",
                        "output": "",
                        "exit_code": -1,
                    }
            else:
                cmd_dir = worktree_path

            relative_working_dir = os.path.relpath(cmd_dir, worktree_path)
            if relative_working_dir == ".":
                relative_working_dir = None

            # SECURITY: Validate command before execution
            is_safe, safety_error = _validate_command_safety(command)
            if not is_safe:
                logger.warning(
                    f"[BASH_COMMAND] Blocked unsafe command for project {project_id}: {command}"
                )
                return {
                    "success": False,
                    "error": safety_error or "Command is not allowed for security reasons",
                    "output": "",
                    "exit_code": -1,
                }

            logger.info(
                f"[BASH_COMMAND] Executing command in {cmd_dir} (relative: {relative_working_dir or '.'}): {command} "
                f"(project: {repo_name}@{commit_id or branch})"
            )

            # Parse command
            try:
                command_parts = shlex.split(command)
            except ValueError as e:
                logger.warning(
                    f"[BASH_COMMAND] Failed to parse command with shlex: {e}, using simple split"
                )
                command_parts = command.split()

            # If we need to run in a subdirectory, prepend cd command
            if relative_working_dir and relative_working_dir != ".":
                cd_command = f"cd {shlex.quote(relative_working_dir)} && {' '.join(shlex.quote(arg) for arg in command_parts)}"
                final_command = ["sh", "-c", cd_command]
            else:
                final_command = command_parts

            # Safe environment variables
            safe_env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "USER": "sandbox",
                "SHELL": "/bin/sh",
                "LANG": "C",
                "TERM": "dumb",
            }

            # Execute command with gVisor isolation
            try:
                result: CommandResult = run_command_isolated(
                    command=final_command,
                    working_dir=worktree_path,
                    repo_path=None,
                    env=safe_env,
                    timeout=30,
                    use_gvisor=True,
                )

                logger.info(
                    f"[BASH_COMMAND] Command completed with exit code {result.returncode} "
                    f"(success: {result.success}) for project {project_id}"
                )

                return {
                    "success": result.success,
                    "output": result.stdout,
                    "error": result.stderr,
                    "exit_code": result.returncode,
                }
            except Exception as e:
                logger.error(f"[BASH_COMMAND] Error executing command with gVisor: {e}")
                return {
                    "success": False,
                    "error": f"Error executing command: {str(e)}",
                    "output": "",
                    "exit_code": -1,
                }

        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "exit_code": -1,
            }
        except Exception as e:
            logger.exception(f"[BASH_COMMAND] Unexpected error in execute_command: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "output": "",
                "exit_code": -1,
            }

    def _run(
        self,
        project_id: str,
        command: str,
        working_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute bash command in the repository worktree."""
        try:
            # Check if repo manager is available
            if not self.repo_manager:
                return {
                    "success": False,
                    "error": "Repo manager is not enabled. Bash commands require a local worktree.",
                    "output": "",
                    "exit_code": -1,
                }

            # Get project details
            details = self._get_project_details(project_id)
            repo_name = details["project_name"]
            branch = details.get("branch_name")
            commit_id = details.get("commit_id")

            # Get worktree path
            worktree_path = self._get_worktree_path(repo_name, branch, commit_id)
            if not worktree_path:
                logger.info(
                    f"[BASH_COMMAND] Worktree not found for project {project_id}, "
                    f"attempting to clone repository {repo_name}"
                )
                # Try to clone the repository using the same logic as parsing flow
                try:
                    # Run async clone in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        worktree_path = loop.run_until_complete(
                            self._clone_repository(repo_name, branch, commit_id, self.user_id)
                        )
                    finally:
                        loop.close()

                    if not worktree_path:
                        return {
                            "success": False,
                            "error": f"Worktree not found for project {project_id} and failed to clone repository. "
                            "Please ensure the repository is accessible and try again.",
                            "output": "",
                            "exit_code": -1,
                        }
                    logger.info(
                        f"[BASH_COMMAND] Successfully cloned repository to {worktree_path}"
                    )
                except Exception as clone_error:
                    logger.error(
                        f"[BASH_COMMAND] Failed to clone repository {repo_name}: {clone_error}"
                    )
                    return {
                        "success": False,
                        "error": f"Worktree not found for project {project_id} and failed to clone repository: {clone_error}",
                        "output": "",
                        "exit_code": -1,
                    }

            # Execute the command using shared execution logic
            return self._execute_command(project_id, command, worktree_path, working_directory)

        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "exit_code": -1,
            }
        except Exception as e:
            logger.exception(f"[BASH_COMMAND] Unexpected error: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "output": "",
                "exit_code": -1,
            }

    async def _arun(
        self,
        project_id: str,
        command: str,
        working_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async implementation that handles cloning natively."""
        try:
            # Check if repo manager is available
            if not self.repo_manager:
                return {
                    "success": False,
                    "error": "Repo manager is not enabled. Bash commands require a local worktree.",
                    "output": "",
                    "exit_code": -1,
                }

            # Get project details
            details = self._get_project_details(project_id)
            repo_name = details["project_name"]
            branch = details.get("branch_name")
            commit_id = details.get("commit_id")

            # Get worktree path
            worktree_path = self._get_worktree_path(repo_name, branch, commit_id)
            if not worktree_path:
                logger.info(
                    f"[BASH_COMMAND] Worktree not found for project {project_id}, "
                    f"attempting to clone repository {repo_name}"
                )
                # Clone repository asynchronously
                try:
                    worktree_path = await self._clone_repository(
                        repo_name, branch, commit_id, self.user_id
                    )
                    if not worktree_path:
                        return {
                            "success": False,
                            "error": f"Worktree not found for project {project_id} and failed to clone repository. "
                            "Please ensure the repository is accessible and try again.",
                            "output": "",
                            "exit_code": -1,
                        }
                    logger.info(
                        f"[BASH_COMMAND] Successfully cloned repository to {worktree_path}"
                    )
                except Exception as clone_error:
                    logger.error(
                        f"[BASH_COMMAND] Failed to clone repository {repo_name}: {clone_error}"
                    )
                    return {
                        "success": False,
                        "error": f"Worktree not found for project {project_id} and failed to clone repository: {clone_error}",
                        "output": "",
                        "exit_code": -1,
                    }

            # Run the rest of the command execution in a thread to not block
            # Pass the already-resolved worktree_path to avoid re-checking
            return await asyncio.to_thread(
                self._execute_command, project_id, command, worktree_path, working_directory
            )

        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "exit_code": -1,
            }
        except Exception as e:
            logger.exception(f"[BASH_COMMAND] Unexpected error in async run: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "output": "",
                "exit_code": -1,
            }


def bash_command_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    """
    Create bash command tool if repo manager is enabled.

    Returns None if repo manager is not enabled.
    """
    repo_manager_enabled = os.getenv("REPO_MANAGER_ENABLED", "false").lower() == "true"
    if not repo_manager_enabled:
        logger.debug("BashCommandTool not created: REPO_MANAGER_ENABLED is false")
        return None

    tool_instance = BashCommandTool(sql_db, user_id)
    if not tool_instance.repo_manager:
        logger.debug("BashCommandTool not created: RepoManager initialization failed")
        return None

    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="bash_command",
        description=tool_instance.description,
        args_schema=BashCommandToolInput,
    )
