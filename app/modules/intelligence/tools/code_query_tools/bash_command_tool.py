"""
Bash Command Tool

Allows agents to run bash commands (grep, awk, find, etc.) on the codebase.
Only works if the project's worktree exists in the repo manager.
"""

import os
import re
import shlex
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.projects.projects_service import ProjectService
from app.modules.repo_manager import RepoManager
from app.modules.utils.gvisor_runner import run_command_isolated, CommandResult
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Character limits for command output to prevent sending insanely large content to LLM
MAX_OUTPUT_LENGTH = 80000  # 80k characters for stdout
MAX_ERROR_LENGTH = 10000  # 10k characters for stderr

# SECURITY: Whitelist of allowed read-only commands
# Only these commands are permitted to execute
ALLOWED_COMMANDS = {
    # File search and listing
    "grep",
    "find",
    "locate",
    "which",
    "whereis",
    "ls",
    "dir",  # Windows equivalent
    # File content viewing (read-only)
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",  # Word count
    "nl",  # Number lines
    # Text processing (read-only operations)
    "awk",  # Only read-only, -i flag blocked separately
    "sed",  # Only read-only, -i flag blocked separately
    "cut",
    "sort",
    "uniq",
    "tr",
    "grep",
    "egrep",
    "fgrep",
    # File information
    "file",
    "stat",
    "readlink",
    "realpath",
    # Directory operations (read-only)
    "pwd",
    "dirname",
    "basename",
    # Text utilities
    "diff",  # Read-only comparison
    "cmp",  # Read-only comparison
    # Archive viewing (read-only)
    "tar",  # Only with -t (list) flag, extraction blocked
    "zipinfo",  # List zip contents
    "unzip",  # Only with -l (list) flag, extraction blocked
    # Process info (read-only)
    "ps",
    # System info (read-only)
    "uname",
    "date",
    "id",
    # String utilities
    "strings",
    "od",  # Octal dump
    "hexdump",
    # Search tools
    "ag",  # The Silver Searcher
    "rg",  # ripgrep
    "ack",  # ack-grep
}

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


def _split_pipes_respecting_quotes(command: str) -> list[str]:
    """
    Split a command on pipes (|) while respecting quoted strings.

    This handles cases like: grep "pattern|with|pipes" | head
    where the pipe inside quotes should not be treated as a separator.

    Args:
        command: The command string that may contain pipes

    Returns:
        List of command parts split on pipes (outside of quotes)
    """
    parts = []
    current_part = []
    in_single_quote = False
    in_double_quote = False
    i = 0

    while i < len(command):
        char = command[i]

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current_part.append(char)
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current_part.append(char)
        elif char == "|" and not in_single_quote and not in_double_quote:
            # This is a real pipe separator
            parts.append("".join(current_part).strip())
            current_part = []
        else:
            current_part.append(char)
        i += 1

    # Add the last part
    if current_part:
        parts.append("".join(current_part).strip())

    return parts if parts else [command.strip()]


def _extract_command_name(command: str) -> str:
    """
    Extract the base command name from a command string.
    Handles commands with paths, pipes, and arguments.

    Args:
        command: The command string

    Returns:
        The base command name (e.g., 'grep' from '/usr/bin/grep -r pattern .')
    """
    # Remove leading/trailing whitespace
    command = command.strip()

    # Handle pipes - extract first command
    if "|" in command:
        # Use proper quote-aware splitting
        pipe_parts = _split_pipes_respecting_quotes(command)
        if pipe_parts:
            command = pipe_parts[0].strip()

    # Split by whitespace to get first token
    parts = command.split()
    if not parts:
        return ""

    # Get the first part (command name or path)
    first_part = parts[0]

    # Extract just the command name (handle paths like /usr/bin/grep or ./script.sh)
    # Remove any path components
    if "/" in first_part:
        first_part = first_part.split("/")[-1]

    return first_part.lower()


def _is_semicolon_in_find_exec(command: str) -> bool:
    """
    Check if a semicolon is part of a valid find -exec syntax.

    The find -exec command requires a semicolon (often escaped as \\;)
    to terminate the exec command. This is a legitimate use case.

    Args:
        command: The command string to check

    Returns:
        True if the semicolon appears to be part of find -exec syntax
    """
    # Check if command starts with 'find' (case insensitive)
    command_lower = command.lower().strip()
    if not command_lower.startswith("find "):
        return False

    # Look for -exec flag followed by a semicolon (escaped or not)
    # Pattern: -exec ... ; or -exec ... \;
    # The semicolon can be at the end or followed by other find options like -o, -and, etc.
    # We match: -exec followed by whitespace, then any characters (non-greedy), then ; or \;
    exec_pattern = r"-exec\s+[^;]+[\\]?;"
    if re.search(exec_pattern, command):
        return True

    return False


def _is_semicolon_safe_in_single_command(command: str) -> bool:
    """
    Check if a semicolon is safe because it's part of a single command syntax,
    not used for command chaining.

    We allow semicolons when:
    1. They're part of find -exec syntax
    2. They're inside quoted strings (not actual command separators)
    3. There's only one unquoted semicolon and it appears to be part of command syntax

    Args:
        command: The command string to check

    Returns:
        True if the semicolon appears safe (part of single command syntax)
    """
    # First check if it's find -exec (already handled, but check anyway)
    if _is_semicolon_in_find_exec(command):
        return True

    # Count unquoted, unescaped semicolons (actual command separators)
    # and check if they appear to separate distinct commands
    semicolon_positions = []
    in_single_quote = False
    in_double_quote = False
    i = 0

    while i < len(command):
        char = command[i]

        # Handle escaping (skip escaped characters)
        if char == "\\" and i + 1 < len(command):
            # Skip the escaped character
            i += 2
            continue

        # Track quote state
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == ";" and not in_single_quote and not in_double_quote:
            # This is an unquoted, unescaped semicolon - potential command separator
            semicolon_positions.append(i)
        i += 1

    # If no unquoted semicolons, it's safe (all semicolons were in quotes)
    if not semicolon_positions:
        return True

    # If there's only one unquoted semicolon, check if it looks like command chaining
    # Command chaining typically has whitespace before and after the semicolon,
    # and the parts before/after look like separate commands
    if len(semicolon_positions) == 1:
        semicolon_pos = semicolon_positions[0]
        before = command[:semicolon_pos].strip()
        after = command[semicolon_pos + 1 :].strip()

        # If there's substantial content before and after, it's likely command chaining
        # Check if both parts look like they could be standalone commands
        # (have at least a few characters and don't look like part of find -exec)
        if len(before) > 3 and len(after) > 3:
            # Check if the part before looks like a complete command
            # (starts with a command name, not part of find -exec syntax)
            before_lower = before.lower()
            if not before_lower.startswith("find ") or "-exec" not in before_lower:
                # This looks like command chaining - block it
                return False

        # If we get here, it's likely part of command syntax (like find -exec)
        return True

    # Multiple unquoted semicolons definitely indicate command chaining - block it
    return False


def _validate_command_safety(command: str) -> tuple[bool, Optional[str]]:
    """
    Validate that a command is safe (read-only) and doesn't attempt write operations.
    Uses a whitelist approach - only explicitly allowed commands are permitted.

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

    # Check for command injection patterns (except pipes which are handled separately)
    # Allow semicolons if they're part of a single command (not command chaining)
    for pattern in INJECTION_PATTERNS:
        if pattern in command:
            # Special case: allow semicolons when they're part of a single command syntax
            # (e.g., find -exec) rather than used for command chaining
            if pattern == ";" and _is_semicolon_safe_in_single_command(command):
                continue  # Skip this check, it's a valid semicolon in single command
            return (
                False,
                f"Command contains injection pattern '{pattern}'. Command chaining/substitution is not allowed.",
            )

    # SECURITY: Whitelist check - only allowed commands can execute
    # Handle pipes - validate each command in the pipe chain
    # Use quote-aware splitting to handle pipes inside quoted strings
    if "|" in command:
        pipe_commands = _split_pipes_respecting_quotes(command)
        for pipe_cmd in pipe_commands:
            if not pipe_cmd:  # Skip empty parts
                continue
            cmd_name = _extract_command_name(pipe_cmd)
            if not cmd_name:
                return (
                    False,
                    "Invalid command in pipe chain.",
                )
            if cmd_name not in ALLOWED_COMMANDS:
                return (
                    False,
                    f"Command '{cmd_name}' is not in the whitelist of allowed read-only commands. Only safe read-only commands like grep, find, cat, head, tail, etc. are permitted.",
                )
    else:
        # Single command - check if it's whitelisted
        cmd_name = _extract_command_name(command)
        if not cmd_name:
            return (
                False,
                "Invalid command.",
            )
        if cmd_name not in ALLOWED_COMMANDS:
            return (
                False,
                f"Command '{cmd_name}' is not in the whitelist of allowed read-only commands. Only safe read-only commands like grep, find, cat, head, tail, etc. are permitted.",
            )

    # Additional check: block commands that are in the always-blocked list
    # (redundant with whitelist, but provides clearer error messages)
    first_word = command_lower.split()[0] if command_lower.split() else ""
    if first_word in ALWAYS_BLOCKED_COMMANDS:
        return (
            False,
            f"Command '{first_word}' is explicitly blocked. This tool only supports read-only operations.",
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
    description: str = """Run bash commands (grep, awk, find, sed, etc.) on the codebase.

        This tool allows you to execute common Unix/bash commands directly on the repository files.
        The command will be executed in the repository's worktree directory using gVisor sandbox isolation
        for enhanced security. Commands run in an isolated environment that prevents filesystem modifications.

        🔒 Security: Commands are executed in a gVisor sandbox, providing strong isolation and preventing
        unauthorized access or modifications to the host system.

        ⚠️ CRITICAL RESTRICTION: ONLY USE READ-ONLY COMMANDS ⚠️
        This tool is designed for read-only operations only. Commands that modify, delete, or write files are NOT supported and may fail or cause unexpected behavior. The gVisor sandbox provides additional protection against accidental modifications.

        IMPORTANT: This tool only works if the repository has been parsed and is available in the repo manager.
        If the worktree doesn't exist, the tool will return an error.

        ✅ ALLOWED (Whitelisted read-only commands only):
        - Search for patterns: grep -r "pattern" .
        - Find files: find . -name "*.py" -type f
        - Process text: awk '/pattern/ {print $1}' file.txt (read-only, no -i flag)
        - Count occurrences: grep -c "pattern" file.txt
        - List files: ls -la directory/
        - Filter output: grep "error" log.txt | head -20 (pipes allowed for filtering)
        - View file contents: cat file.txt, head file.txt, tail file.txt
        - Check file info: stat file.txt, file file.txt
        - Search in files: grep, ag (Silver Searcher), rg (ripgrep)
        - Text utilities: sort, uniq, cut, wc, diff, cmp
        - File information: stat, file, readlink, realpath

        ⚠️ SECURITY: Only whitelisted commands are allowed. Commands like python, python3,
        node, bash, sh, and other interpreters are BLOCKED for security reasons.

        ❌ NOT ALLOWED:
        - Interpreters/executables: python, python3, node, bash, sh, perl, ruby, etc.
        - File modification: echo > file, sed -i, awk -i
        - File creation: touch, mkdir, > file, >> file
        - File deletion: rm, rmdir
        - Git operations: git (all git commands blocked)
        - Package installation: npm, pip, yarn, pnpm
        - Network commands: curl, wget, ssh, scp
        - Command chaining: ; && || (use pipes | for filtering instead)
        - Command substitution: `command` or $(command)
        - Environment access: env (blocked to prevent secret exposure)
        - Any command not in the whitelist
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

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        """Get project details and validate user access."""
        # Note: get_project_from_db_by_id_sync type hint says int, but Project.id is Text (string)
        details = self.project_service.get_project_from_db_by_id_sync(project_id)  # type: ignore[arg-type]
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def _get_worktree_path(
        self,
        repo_name: str,
        branch: Optional[str],
        commit_id: Optional[str],
        user_id: Optional[str],
    ) -> Optional[str]:
        """Get the worktree path for the project."""
        if not self.repo_manager:
            return None

        # Try to get worktree path with user_id for security
        worktree_path = self.repo_manager.get_repo_path(
            repo_name, branch=branch, commit_id=commit_id, user_id=user_id
        )
        if worktree_path and os.path.exists(worktree_path):
            return worktree_path

        # Try with just commit_id (with user_id)
        if commit_id:
            worktree_path = self.repo_manager.get_repo_path(
                repo_name, commit_id=commit_id, user_id=user_id
            )
            if worktree_path and os.path.exists(worktree_path):
                return worktree_path

        # Try with just branch (with user_id)
        if branch:
            worktree_path = self.repo_manager.get_repo_path(
                repo_name, branch=branch, user_id=user_id
            )
            if worktree_path and os.path.exists(worktree_path):
                return worktree_path

        return None

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
            user_id = details.get("user_id")

            # Get worktree path (with user_id for security)
            worktree_path = self._get_worktree_path(
                repo_name, branch, commit_id, user_id
            )
            if not worktree_path:
                return {
                    "success": False,
                    "error": f"Worktree not found for project {project_id}. The repository must be parsed and available in the repo manager.",
                    "output": "",
                    "exit_code": -1,
                }

            # Update last accessed time for usage tracking
            try:
                self.repo_manager.update_last_accessed(
                    repo_name, branch=branch, commit_id=commit_id, user_id=user_id
                )
            except Exception as e:
                logger.warning(
                    f"[BASH_COMMAND] Failed to update last_accessed for {repo_name}: {e}"
                )

            # SECURITY: Normalize paths to prevent directory traversal
            # Only the specific project's worktree will be accessible
            worktree_path = os.path.abspath(worktree_path)

            # SECURITY: Determine working directory and validate it's within the worktree
            # This ensures commands can only access files within this specific repository
            if working_directory:
                # Resolve the working directory path
                requested_dir = os.path.normpath(working_directory)
                # Prevent directory traversal attacks
                if os.path.isabs(requested_dir) or ".." in requested_dir:
                    return {
                        "success": False,
                        "error": f"Invalid working directory: '{working_directory}'. Directory traversal is not allowed.",
                        "output": "",
                        "exit_code": -1,
                    }

                cmd_dir = os.path.join(worktree_path, requested_dir)
                # Resolve to absolute path and ensure it's within worktree
                cmd_dir = os.path.abspath(cmd_dir)

                # Security check: ensure cmd_dir is within worktree_path
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

            # Calculate relative path from worktree root for use in sandbox
            # The gVisor runner will mount worktree_path, so we need relative path
            relative_working_dir = os.path.relpath(cmd_dir, worktree_path)
            if relative_working_dir == ".":
                relative_working_dir = None  # Use root of mounted directory

            # SECURITY: Validate command before execution
            is_safe, safety_error = _validate_command_safety(command)
            if not is_safe:
                logger.warning(
                    f"[BASH_COMMAND] Blocked unsafe command for project {project_id}: {command}"
                )
                return {
                    "success": False,
                    "error": safety_error
                    or "Command is not allowed for security reasons",
                    "output": "",
                    "exit_code": -1,
                }

            logger.info(
                f"[BASH_COMMAND] Executing command in {cmd_dir} (relative: {relative_working_dir or '.'}): {command} "
                f"(project: {repo_name}@{commit_id or branch})"
            )

            # Check if command contains pipes - pipes require shell execution
            has_pipe = "|" in command

            # If command has pipes or we need to run in a subdirectory, use shell execution
            if has_pipe or (relative_working_dir and relative_working_dir != "."):
                # Execute via shell to support pipes and working directory changes
                if relative_working_dir and relative_working_dir != ".":
                    # Change to subdirectory and run command
                    shell_command = (
                        f"cd {shlex.quote(relative_working_dir)} && {command}"
                    )
                else:
                    # Just run the command (pipes require shell)
                    shell_command = command
                final_command = ["sh", "-c", shell_command]
            else:
                # No pipes and no subdirectory - can use list-based execution for better security
                # Parse command into list for gVisor runner
                try:
                    # Use shlex to properly parse the command while preserving quotes
                    command_parts = shlex.split(command)
                except ValueError as e:
                    # If parsing fails, try a simpler approach
                    logger.warning(
                        f"[BASH_COMMAND] Failed to parse command with shlex: {e}, using simple split"
                    )
                    command_parts = command.split()
                final_command = command_parts

            # SECURITY: Don't pass environment variables - they will be filtered by gVisor runner
            # but we don't want to pass them at all to prevent any exposure
            safe_env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "USER": "sandbox",
                "SHELL": "/bin/sh",
                "LANG": "C",
                "TERM": "dumb",
            }

            # Execute command with gVisor isolation
            # Only mount the worktree_path as READ-ONLY - this ensures commands can only access this specific repo
            try:
                result: CommandResult = run_command_isolated(
                    command=final_command,
                    working_dir=worktree_path,  # Mount only the worktree root (as read-only)
                    repo_path=None,  # Don't mount separately since working_dir is the repo
                    env=safe_env,  # Use minimal safe environment, not os.environ
                    timeout=30,  # 30 second timeout
                    use_gvisor=True,  # Enable gVisor isolation
                )

                logger.info(
                    f"[BASH_COMMAND] Command completed with exit code {result.returncode} "
                    f"(success: {result.success}) for project {project_id}"
                )

                # Truncate output if it exceeds character limits
                output = result.stdout
                error = result.stderr
                output_truncated = False
                error_truncated = False

                if len(output) > MAX_OUTPUT_LENGTH:
                    output = output[:MAX_OUTPUT_LENGTH]
                    output_truncated = True
                    logger.warning(
                        f"[BASH_COMMAND] Output truncated from {len(result.stdout)} to {MAX_OUTPUT_LENGTH} characters "
                        f"for project {project_id}"
                    )

                if len(error) > MAX_ERROR_LENGTH:
                    error = error[:MAX_ERROR_LENGTH]
                    error_truncated = True
                    logger.warning(
                        f"[BASH_COMMAND] Error output truncated from {len(result.stderr)} to {MAX_ERROR_LENGTH} characters "
                        f"for project {project_id}"
                    )

                # Build response with truncation notices
                response = {
                    "success": result.success,
                    "output": output,
                    "error": error,
                    "exit_code": result.returncode,
                }

                # Add truncation notices if applicable
                truncation_notices = []
                if output_truncated:
                    truncation_notices.append(
                        f"⚠️ Output truncated: showing first {MAX_OUTPUT_LENGTH:,} characters of {len(result.stdout):,} total"
                    )
                if error_truncated:
                    truncation_notices.append(
                        f"⚠️ Error output truncated: showing first {MAX_ERROR_LENGTH:,} characters of {len(result.stderr):,} total"
                    )

                if truncation_notices:
                    # Prepend truncation notices to output
                    response["output"] = (
                        "\n".join(truncation_notices) + "\n\n" + response["output"]
                    )

                return response
            except Exception:
                logger.exception(
                    "[BASH_COMMAND] Error executing command with gVisor",
                    project_id=project_id,
                    command=command,
                    working_directory=working_directory,
                )
                return {
                    "success": False,
                    "error": "Error executing command",
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
        except Exception:
            logger.exception(
                "[BASH_COMMAND] Unexpected error",
                project_id=project_id,
                command=command,
                working_directory=working_directory,
            )
            return {
                "success": False,
                "error": "Unexpected error",
                "output": "",
                "exit_code": -1,
            }

    async def _arun(
        self,
        project_id: str,
        command: str,
        working_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for _run."""
        import asyncio

        return await asyncio.to_thread(
            self._run, project_id, command, working_directory
        )


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
