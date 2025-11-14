"""
CLI tool execution module.

This module provides a unified interface for running various CLI tools (Codex, Claude Code, Cursor, etc.)
with their default configurations. All tool configs, command building, and execution logic is in this file.
"""

import asyncio
import shlex
import shutil
import subprocess
from typing import Optional, List, Dict, Tuple


# CLI tool configurations
CLI_CONFIGS: Dict[str, Dict] = {
    "codex": {
        "base_command": "codex",
        "subcommand": "exec",
        "default_flags": ["--model", "o3"],
        "description": "Codex CLI tool",
    },
    "claude_code": {
        "base_command": "claude",
        "subcommand": None,
        "default_flags": ["--print"],
        "description": "Claude Code CLI tool",
    },
    "cursor": {
        "base_command": "cursor-agent",
        "subcommand": None,
        "default_flags": ["-p", "--output-format", "text"],
        "description": "Cursor CLI tool (non-interactive mode with text output)",
    },
    "factory_ai": {
        "base_command": "droids",
        "subcommand": None,
        "default_flags": [],
        "description": "Factory AI CLI tool (Droids)",
    },
}


def build_command_parts(
    tool_name: str,
    prompt: str,
    flags: Optional[List[str]] = None,
    use_default_flags: bool = True,
) -> List[str]:
    """
    Build command parts for a CLI tool based on its configuration.
    
    Args:
        tool_name: Name of the CLI tool (must exist in CLI_CONFIGS)
        prompt: The prompt/question to send to the CLI tool
        flags: Optional additional flags to override or append to defaults
        use_default_flags: If True, includes default flags from config
    
    Returns:
        List of command parts ready to pass to run_cli_command()
    
    Example:
        >>> build_command_parts("codex", "What is Python?")
        ["codex", "exec", "--model", "o3", "What is Python?"]
        
        >>> build_command_parts("codex", "What is Python?", flags=["--model", "o1"])
        ["codex", "exec", "--model", "o1", "What is Python?"]
    """
    if tool_name not in CLI_CONFIGS:
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available tools: {list(CLI_CONFIGS.keys())}"
        )
    
    config = CLI_CONFIGS[tool_name]
    command_parts = [config["base_command"]]
    
    if config["subcommand"]:
        command_parts.append(config["subcommand"])
    
    if use_default_flags and config["default_flags"]:
        command_parts.extend(config["default_flags"])
    
    if flags:
        command_parts.extend(flags)
    
    command_parts.append(prompt)
    
    return command_parts


def run_cli_command(
    command_parts: List[str],
    workspace_path: Optional[str] = None,
) -> str:
    """
    Execute a CLI command and capture its output.
    
    Args:
        command_parts: List of command parts (e.g., ["codex", "exec", "--model", "o3", "prompt"])
        workspace_path: Optional path to workspace directory. If provided, command runs from this directory.
    
    Returns:
        The captured output from the command as a string.
        Returns an error message if the command fails.
    """
    # Construct the final command string with proper quoting
    quoted_parts = []
    for part in command_parts:
        if " " in part and not part.startswith("--"):
            quoted_parts.append(shlex.quote(part))
        else:
            quoted_parts.append(part)
    command = " ".join(quoted_parts)
    
    if workspace_path:
        print(f"Using workspace: {workspace_path}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=workspace_path if workspace_path else None,
        )
        
        return result.stdout.strip()
        
    except FileNotFoundError:
        return f"Error: The '{command_parts[0]}' command was not found. Please ensure it is installed and in your system's PATH."
        
    except subprocess.CalledProcessError as e:
        error_details = e.stderr.strip()
        return f"Error: The '{command_parts[0]}' command failed with exit code {e.returncode}.\nDetails: {error_details}"


def run_cli_tool(
    tool_name: str,
    prompt: str,
    workspace_path: Optional[str] = None,
    flags: Optional[List[str]] = None,
    use_default_flags: bool = True,
) -> str:
    """
    Run a CLI tool using its configuration.
    
    This is the main function to use - it builds the command from config and executes it.
    
    Args:
        tool_name: Name of the CLI tool (e.g., "codex", "claude_code")
        prompt: The prompt/question to send to the CLI tool
        workspace_path: Optional path to workspace directory
        flags: Optional flags to override or append to defaults
        use_default_flags: If True, includes default flags from config
    
    Returns:
        The captured output from the command as a string.
    
    Example:
        >>> run_cli_tool("codex", "What is Python?", workspace_path="/path/to/repo")
        # Runs: codex exec --model o3 "What is Python?" from /path/to/repo
    """
    command_parts = build_command_parts(
        tool_name=tool_name,
        prompt=prompt,
        flags=flags,
        use_default_flags=use_default_flags,
    )
    return run_cli_command(command_parts, workspace_path)


# Helper functions for convenience
def get_available_tools() -> List[str]:
    """Return list of available CLI tool names."""
    return list(CLI_CONFIGS.keys())


def get_tool_config(tool_name: str) -> Dict:
    """Get configuration for a specific tool."""
    if tool_name not in CLI_CONFIGS:
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available tools: {list(CLI_CONFIGS.keys())}"
        )
    return CLI_CONFIGS[tool_name].copy()


def verify_cli_tool_available(tool_name: str) -> Tuple[bool, Optional[str]]:
    """
    Verify if a CLI tool is properly configured and available.
    
    Args:
        tool_name: Name of the CLI tool to verify
        
    Returns:
        Tuple of (is_available: bool, error_message: Optional[str])
        - If available, returns (True, None)
        - If not available, returns (False, error_message)
    """
    if tool_name not in CLI_CONFIGS:
        return False, f"Unknown tool '{tool_name}'. Available tools: {list(CLI_CONFIGS.keys())}"
    
    config = CLI_CONFIGS[tool_name]
    base_command = config["base_command"]
    
    # Check if command exists in PATH
    command_path = shutil.which(base_command)
    
    if command_path is None:
        # Provide helpful installation instructions for known tools
        install_hint = ""
        if tool_name == "cursor":
            install_hint = " Install with: curl https://cursor.com/install -fsS | bash"
        elif tool_name == "codex":
            install_hint = " Install Codex CLI and ensure it's in your PATH."
        elif tool_name == "claude_code":
            install_hint = " Install Claude Code CLI and ensure it's in your PATH."
        elif tool_name == "factory_ai":
            install_hint = " Install Factory AI CLI and ensure it's in your PATH."
        
        return False, (
            f"Command '{base_command}' not found in PATH. "
            f"Please ensure {tool_name} is installed and available in your system PATH.{install_hint}"
        )
    
    # Command found in PATH - consider it available
    # We don't test execution here to avoid issues with tools that require
    # interactive mode or specific arguments. The actual execution will
    # catch any runtime errors.
    return True, None


async def run_cli_tool_async(
    tool_name: str,
    prompt: str,
    workspace_path: Optional[str] = None,
    flags: Optional[List[str]] = None,
    use_default_flags: bool = True,
) -> str:
    """
    Async wrapper for run_cli_tool.
    
    Runs the blocking CLI command in a thread pool to avoid blocking the event loop.
    
    Args:
        tool_name: Name of the CLI tool (e.g., "codex", "claude_code")
        prompt: The prompt/question to send to the CLI tool
        workspace_path: Optional path to workspace directory
        flags: Optional flags to override or append to defaults
        use_default_flags: If True, includes default flags from config
    
    Returns:
        The captured output from the command as a string.
    """
    return await asyncio.to_thread(
        run_cli_tool,
        tool_name=tool_name,
        prompt=prompt,
        workspace_path=workspace_path,
        flags=flags,
        use_default_flags=use_default_flags,
    )
