"""bash_command — subprocess in worktree with policy enforcement."""

from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path

from pydantic_ai import RunContext

from poc.managers.deps import PoCDeepDeps
from poc.tools.shell_policy import ShellPolicy


# Allowlisted commands for read-only shell access
READ_ONLY_COMMANDS = frozenset(
    [
        # Search
        "rg",
        "grep",
        "find",
        "fd",
        # File inspection
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "file",
        "tree",
        # Git inspection (read-only)
        "git",
        # General
        "echo",
        "pwd",
        "which",
        "dirname",
        "basename",
    ]
)

# Git subcommands that are read-only
READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    [
        "status",
        "diff",
        "log",
        "show",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "branch",
        "remote",
        "config",
        "describe",
        "tag",
        "blame",
    ]
)

# Validation-specific commands (in addition to read-only)
VALIDATION_COMMANDS = frozenset(
    [
        "python",
        "python3",
        "pytest",
        "py.test",
        "mypy",
        "ruff",
        "black",
        "isort",
        "pylint",
        "flake8",
        "bandit",
    ]
)

# Dangerous patterns that are always rejected
DANGEROUS_PATTERNS = [
    # In-place editing
    r"sed\s+.*-i",
    r"perl\s+.*-pi",
    # File modification
    r"\bcp\s+",
    r"\bmv\s+",
    r"\brm\s+",
    r"\bchmod\s+",
    r"\bchown\s+",
    # Redirections (writing to files)
    r">[^>]",
    r">>",
    # Heredocs
    r"<<",
    # Git mutation
    r"git\s+checkout\s+(?!--\s|HEAD\s|@\{|--track)",
    r"git\s+add\s+",
    r"git\s+commit\s+",
    r"git\s+push\s+",
    r"git\s+reset\s+",
    r"git\s+revert\s+",
    r"git\s+rebase\s+",
    r"git\s+merge\s+",
    r"git\s+worktree\s+",
    r"git\s+branch\s+-[dD]",
    r"git\s+branch\s+-[mM]",
    # Branch creation (harness-only)
    r"git\s+checkout\s+-b",
    r"git\s+switch\s+-c",
    r"git\s+branch\s+[^-]",
]

SAFE_PYTHON_VALIDATE_PATTERNS = (
    r"^python3?\s+-m\s+py_compile(\s+\S+)+\s*$",
    r"^python3?\s+-m\s+compileall(\s+\S+)+\s*$",
)


def _extract_base_command(command: str) -> str | None:
    """Extract the base command from a shell command string."""
    # Remove leading whitespace and common prefixes
    cmd = command.strip()
    if cmd.startswith("cd "):
        # Handle 'cd dir && command' or 'cd dir; command'
        parts = re.split(r"[;&]|&&|\|\|", cmd, maxsplit=1)
        if len(parts) > 1:
            cmd = parts[1].strip()
        else:
            return "cd"  # Just a cd command

    # Try to parse with shlex
    try:
        tokens = shlex.split(cmd)
        if not tokens:
            return None
        return tokens[0]
    except ValueError:
        # Fallback: simple split
        parts = cmd.split()
        return parts[0] if parts else None


def _is_git_read_only(command: str) -> bool:
    """Check if a git command is read-only."""
    # Check for dangerous git patterns first
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return False

    # Extract git subcommand
    match = re.search(r"git\s+(\w+)", command)
    if not match:
        return False  # Can't determine, reject

    subcommand = match.group(1)
    return subcommand in READ_ONLY_GIT_SUBCOMMANDS


def validate_shell_command(command: str, policy: ShellPolicy) -> tuple[bool, str]:
    """
    Validate a shell command against a policy.

    Returns (is_valid, error_message).
    """
    if policy == ShellPolicy.FORBIDDEN:
        return False, "Shell commands are forbidden for this role."

    if policy == ShellPolicy.UNRESTRICTED:
        # Still check for obviously dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return (
                    False,
                    f"Command contains dangerous pattern matching: {pattern}. "
                    "This operation is not allowed.",
                )
        return True, ""

    normalized = command.strip()
    if not normalized:
        return False, "Error: empty shell command rejected"

    if normalized == ":":
        return (
            False,
            "Error: no-op shell command ':' rejected. "
            "Do not retry it. Use a real search/validation command or switch tools.",
        )

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return (
                False,
                f"Command rejected: contains dangerous pattern '{pattern}'. "
                f"Policy '{policy.value}' does not allow file modifications, "
                "git mutations, or output redirections.",
            )

    # Check for command chaining (suspicious)
    if re.search(r"[;&]|&&|\|\|", command) and not command.strip().startswith("cd "):
        # Allow simple 'cd X && command' but flag other chaining
        if not re.match(r"^cd\s+\S+\s+&&\s+\S", command):
            return (
                False,
                "Command chaining detected. Use simple, single-purpose commands only.",
            )

    # Extract base command
    base_cmd = _extract_base_command(command)
    if not base_cmd:
        return False, "Could not parse command"

    # Handle git specially
    if base_cmd == "git":
        if not _is_git_read_only(command):
            return (
                False,
                f"Git command '{command}' is not read-only. "
                "Only status, diff, log, branch (list), ls-files, etc. are allowed.",
            )
        return True, ""

    # Check against allowlist for read-only
    if policy == ShellPolicy.READ_ONLY:
        if base_cmd not in READ_ONLY_COMMANDS:
            return (
                False,
                f"Command '{base_cmd}' is not in the read-only allowlist. "
                f"Allowed: {', '.join(sorted(READ_ONLY_COMMANDS))}.",
            )
        return True, ""

    # Check against allowlist for validate-only
    if policy == ShellPolicy.VALIDATE_ONLY:
        allowed = READ_ONLY_COMMANDS | VALIDATION_COMMANDS
        if base_cmd not in allowed:
            return (
                False,
                f"Command '{base_cmd}' is not allowed. "
                f"Allowed: search/inspect ({', '.join(sorted(READ_ONLY_COMMANDS))}) "
                f"and validation ({', '.join(sorted(VALIDATION_COMMANDS))}).",
            )

        if base_cmd in ("python", "python3"):
            if not any(re.match(pattern, command.strip()) for pattern in SAFE_PYTHON_VALIDATE_PATTERNS):
                return (
                    False,
                    "Python shell usage is restricted to 'python -m py_compile ...' or "
                    "'python -m compileall ...' in validate-only mode.",
                )

        # Additional validation: pytest must target specific files
        if base_cmd in ("pytest", "py.test"):
            # Check if it's a general run (no specific file)
            if not re.search(r"pytest\s+\S+\.(py|yaml|yml|toml)", command):
                # Allow if it has a path or test file pattern
                if not re.search(r"pytest\s+[^\s-]", command):
                    return (
                        False,
                        "pytest must target specific files or directories. "
                        "Use: 'pytest path/to/test_file.py' or 'pytest tests/unit/'. "
                        "Avoid: 'pytest' (runs all tests).",
                    )

        return True, ""

    return True, ""


def _normalize_command(command: str) -> str:
    return " ".join(command.strip().split())


async def bash_command(
    ctx: RunContext[PoCDeepDeps],
    command: str,
    timeout_seconds: float = 120.0,
) -> str:
    """Execute a shell command with policy enforcement."""
    # Get policy from context (default to unrestricted for backwards compatibility)
    policy = getattr(ctx.deps, "shell_policy", ShellPolicy.UNRESTRICTED)
    normalized = _normalize_command(command)

    if normalized and ctx.deps.poc_run.shell_failures.get(normalized, 0) >= 1:
        return (
            "Error: This exact failing shell command was already attempted. "
            "Do not retry it. Use a different command, switch tools, or return BLOCKED."
        )

    is_valid, error_msg = validate_shell_command(command, policy)
    if not is_valid:
        if normalized:
            ctx.deps.poc_run.shell_failures[normalized] = (
                ctx.deps.poc_run.shell_failures.get(normalized, 0) + 1
            )
        return f"Error: {error_msg}"

    cwd = Path(ctx.deps.poc_run.worktree_path or ctx.deps.poc_run.project_root)
    try:
        proc = await asyncio.create_subprocess_shell(
            command.strip(),
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        proc.kill()
        return "Error: timeout"
    except Exception as e:
        return f"Error: {e}"
    out = out_b.decode(errors="replace")[:100_000]
    err = err_b.decode(errors="replace")[:20_000]
    code = proc.returncode
    if normalized:
        if code == 0:
            ctx.deps.poc_run.shell_failures.pop(normalized, None)
        else:
            ctx.deps.poc_run.shell_failures[normalized] = (
                ctx.deps.poc_run.shell_failures.get(normalized, 0) + 1
            )
    return f"exit={code}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"


async def read_only_bash(
    ctx: RunContext[PoCDeepDeps],
    command: str,
    timeout_seconds: float = 120.0,
) -> str:
    """Execute a read-only shell command (for discovery agents)."""
    # Force read-only policy regardless of context
    policy = ShellPolicy.READ_ONLY

    is_valid, error_msg = validate_shell_command(command, policy)
    if not is_valid:
        return f"Error: {error_msg}"

    # Temporarily override policy for execution
    original_policy = getattr(ctx.deps, "shell_policy", ShellPolicy.UNRESTRICTED)
    ctx.deps.shell_policy = policy
    try:
        result = await bash_command(ctx, command, timeout_seconds)
    finally:
        ctx.deps.shell_policy = original_policy
    return result


async def validate_only_bash(
    ctx: RunContext[PoCDeepDeps],
    command: str,
    timeout_seconds: float = 120.0,
) -> str:
    """Execute a validation shell command (for implement/verify agents)."""
    # Force validate-only policy
    policy = ShellPolicy.VALIDATE_ONLY

    is_valid, error_msg = validate_shell_command(command, policy)
    if not is_valid:
        return f"Error: {error_msg}"

    original_policy = getattr(ctx.deps, "shell_policy", ShellPolicy.UNRESTRICTED)
    ctx.deps.shell_policy = policy
    try:
        result = await bash_command(ctx, command, timeout_seconds)
    finally:
        ctx.deps.shell_policy = original_policy
    return result
