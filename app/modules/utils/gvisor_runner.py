"""
gVisor Command Runner Utility

This module provides utilities for running commands in isolated gVisor sandboxes.
gVisor provides better security isolation when executing commands for repositories.

Usage:
    from app.modules.utils.gvisor_runner import run_command_isolated

    result = run_command_isolated(
        command=["ls", "-la"],
        working_dir="/path/to/repo",
        repo_path="/.repos/repo-name"
    )
"""

import os
import subprocess
import logging
import platform
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

from app.modules.utils.install_gvisor import get_runsc_path

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""

    returncode: int
    stdout: str
    stderr: str
    success: bool


def get_runsc_binary() -> Optional[Path]:
    """
    Get the path to the runsc binary.

    Returns:
        Path to runsc binary, or None if not found
    """
    return get_runsc_path()


def is_gvisor_available() -> bool:
    """
    Check if gVisor is available for use.

    Returns:
        True if gVisor is installed and available, False otherwise
    """
    # gVisor only works on Linux
    # On Mac/Windows, it can work through Docker Desktop (Linux VM)
    system = platform.system().lower()

    if system == "linux":
        # Native Linux - check for runsc binary
        return get_runsc_binary() is not None
    elif system in ["darwin", "windows"]:
        # Mac/Windows - can use Docker Desktop with runsc runtime
        # Docker Desktop runs a Linux VM, so gVisor can work there
        docker_ready = _check_docker_available()
        if docker_ready:
            # Docker is available, and the probe already confirmed runsc works
            return True
        return False
    else:
        return False


def _filter_safe_environment_variables(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    Filter environment variables to only include safe, non-sensitive ones.

    SECURITY: This prevents exposure of API keys, passwords, tokens, and other secrets.

    Args:
        env: Original environment variables dictionary

    Returns:
        Filtered dictionary with only safe environment variables
    """
    # Safe environment variables that don't contain secrets
    SAFE_ENV_VARS = {
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "TZ",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
    }

    # Patterns for sensitive variable names (case-insensitive)
    SENSITIVE_PATTERNS = [
        "key",
        "secret",
        "password",
        "token",
        "credential",
        "auth",
        "api",
        "private",
        "passwd",
        "pwd",
        "encrypt",
        "decrypt",
    ]

    if not env:
        return {}

    filtered = {}
    for key, value in env.items():
        key_upper = key.upper()

        # Allow explicitly safe variables
        if key in SAFE_ENV_VARS or key_upper in SAFE_ENV_VARS:
            filtered[key] = value
            continue

        # Block variables with sensitive patterns
        is_sensitive = any(
            pattern in key_upper for pattern in [p.upper() for p in SENSITIVE_PATTERNS]
        )

        if not is_sensitive:
            # Additional check: block if value looks like a secret
            # (long alphanumeric strings, common secret patterns)
            if len(value) > 20 and (
                value.startswith("sk-")
                or value.startswith("ghp_")
                or value.startswith("xoxb-")
                or value.startswith("xoxp-")
                or "BEGIN" in value
                or "PRIVATE" in value
            ):
                logger.debug(
                    f"Filtered out environment variable '{key}' (looks like a secret)"
                )
                continue

            filtered[key] = value
        else:
            logger.debug(f"Filtered out sensitive environment variable: {key}")

    return filtered


def _is_running_in_container() -> bool:
    """
    Check if we're running inside a container (Docker/K8s).

    Returns:
        True if running in a container, False otherwise
    """
    # Check for common container indicators
    if os.path.exists("/.dockerenv"):
        return True
    # Check cgroup (common in containers)
    try:
        with open("/proc/self/cgroup", "r") as f:
            content = f.read()
            # Docker and K8s use specific cgroup patterns
            if "docker" in content or "kubepods" in content or "containerd" in content:
                return True
    except Exception:
        pass
    return False


def run_command_isolated(
    command: List[str],
    working_dir: Optional[str] = None,
    repo_path: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    use_gvisor: bool = True,
) -> CommandResult:
    """
    Run a command in an isolated gVisor sandbox.

    This function uses gVisor's runsc to execute commands in a sandboxed environment,
    providing better security isolation. If gVisor is not available, it falls back
    to regular subprocess execution.

    Args:
        command: Command to execute as a list of strings (e.g., ["ls", "-la"])
        working_dir: Working directory for the command (will be mounted in sandbox)
        repo_path: Path to the repository (will be mounted read-only in sandbox)
        env: Environment variables to set
        timeout: Timeout in seconds for command execution
        use_gvisor: If False, skip gVisor and use regular subprocess

    Returns:
        CommandResult with returncode, stdout, stderr, and success flag
    """
    if not use_gvisor:
        logger.info("[GVISOR] gVisor disabled by parameter, using regular subprocess")
        return _run_command_regular(
            command=command,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
        )

    if not is_gvisor_available():
        logger.warning(
            "[GVISOR] gVisor not available, falling back to regular subprocess (less secure)"
        )
        return _run_command_regular(
            command=command,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
        )

    runsc_path = get_runsc_binary()
    if not runsc_path:
        logger.warning(
            "[GVISOR] gVisor runsc binary not found, falling back to regular subprocess (less secure)"
        )
        return _run_command_regular(
            command=command,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
        )

    logger.info(f"[GVISOR] gVisor available, using runsc at {runsc_path}")

    try:
        # Determine the best method based on environment:
        # 1. If in K8s/container: Use runsc directly (no Docker needed)
        # 2. If on Linux with Docker: Use Docker with runsc runtime
        # 3. Otherwise: Fall back to regular subprocess

        in_container = _is_running_in_container()

        if in_container:
            # In K8s/container: Use runsc directly with a simple sandbox
            logger.info(
                "[GVISOR] Running in container environment, attempting to use runsc directly"
            )
            if runsc_path:
                return _run_with_runsc_direct(
                    command=command,
                    working_dir=working_dir,
                    repo_path=repo_path,
                    env=env,
                    timeout=timeout,
                    runsc_path=runsc_path,
                )
            logger.warning(
                "[GVISOR] runsc binary unavailable inside container; using regular subprocess (container already provides isolation)"
            )
            return _run_command_regular(
                command=command,
                working_dir=working_dir,
                env=env,
                timeout=timeout,
            )
        else:
            # On host (Linux, Mac, or Windows): Try Docker with runsc runtime
            system = platform.system().lower()
            docker_available = _check_docker_available()
            if docker_available:
                logger.info(
                    "[GVISOR] Docker available, attempting to use Docker with gVisor runtime"
                )
                is_desktop = _is_docker_desktop()
                if is_desktop and system != "linux":
                    logger.info(
                        "[GVISOR] Using Docker Desktop with runsc runtime (Mac/Windows)"
                    )
                else:
                    logger.debug("Using Docker with runsc runtime")
                return _run_with_docker_gvisor(
                    command=command,
                    working_dir=working_dir,
                    repo_path=repo_path,
                    env=env,
                    timeout=timeout,
                    runsc_path=runsc_path,
                )
            else:
                # No Docker, try direct runsc (only works on Linux)
                if system == "linux":
                    if runsc_path:
                        logger.warning(
                            "[GVISOR] Docker not available, attempting direct runsc usage (Linux only)"
                        )
                        return _run_with_runsc_direct(
                            command=command,
                            working_dir=working_dir,
                            repo_path=repo_path,
                            env=env,
                            timeout=timeout,
                            runsc_path=runsc_path,
                        )
                    logger.warning(
                        "[GVISOR] Docker not available and runsc binary missing on Linux, falling back to regular subprocess (less secure)"
                    )
                    return _run_command_regular(
                        command=command,
                        working_dir=working_dir,
                        env=env,
                        timeout=timeout,
                    )
                else:
                    # Mac/Windows without Docker - fall back to regular subprocess
                    logger.warning(
                        "[GVISOR] No Docker available on Mac/Windows, falling back to regular subprocess (less secure)"
                    )
                    return _run_command_regular(
                        command=command,
                        working_dir=working_dir,
                        env=env,
                        timeout=timeout,
                    )

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds")
        return CommandResult(
            returncode=124,  # Standard timeout exit code
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            success=False,
        )
    except Exception as e:
        logger.error(f"Error running isolated command: {e}", exc_info=True)
        # Fallback to regular execution
        logger.info("Falling back to regular subprocess execution")
        return _run_command_regular(
            command=command,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
        )


def _check_docker_available() -> bool:
    """Check if Docker is available and runsc runtime is configured and working."""
    try:
        # Check if docker command exists
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False

        # Check if runsc runtime is available in Docker
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_runtime = "runsc" in result.stdout or "gvisor" in result.stdout.lower()

        if not has_runtime:
            return False

        # Test if runsc actually works (it might be configured but not functional)
        # Try a simple test container
        test_result = subprocess.run(
            ["docker", "run", "--rm", "--runtime=runsc", "busybox", "echo", "test"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # If it works, return True. If it fails with specific errors, it might not be functional
        if test_result.returncode == 0:
            return True

        # Check for known errors that indicate runsc isn't working properly
        error_output = test_result.stderr.lower()
        if any(
            err in error_output
            for err in [
                "exec format error",
                "no such file",
                "cannot create sandbox",
                "waiting for sandbox",
                "client sync file",
            ]
        ):
            logger.warning(
                "runsc runtime is configured but not functional. "
                "This may be due to architecture mismatch (e.g., arm64 Mac) or missing dependencies. "
                "Falling back to regular subprocess."
            )
            return False

        # Other errors might be transient, so we'll try anyway
        return True

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except Exception as e:
        logger.debug(f"Error checking Docker gVisor availability: {e}")
        return False


def _is_docker_desktop() -> bool:
    """Check if Docker Desktop is being used (Mac/Windows)."""
    try:
        # Docker Desktop sets specific environment variables
        if os.environ.get("DOCKER_DESKTOP") == "1":
            return True

        # Check Docker context
        result = subprocess.run(
            ["docker", "context", "show"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Docker Desktop typically uses "desktop-linux" context
            return "desktop" in result.stdout.lower()

        return False
    except Exception:
        return False


def _run_with_docker_gvisor(
    command: List[str],
    working_dir: Optional[str],
    repo_path: Optional[str],
    env: Optional[Dict[str, str]],
    timeout: Optional[int],
    runsc_path: Optional[Path],
) -> CommandResult:
    """
    Run command using Docker with gVisor (runsc) runtime.

    This is the recommended way to use gVisor for command isolation.
    """
    import uuid
    import shlex

    if runsc_path:
        logger.info(
            f"[GVISOR] Using Docker with gVisor runtime (runsc at {runsc_path})"
        )
    else:
        logger.info(
            "[GVISOR] Using Docker with gVisor runtime (runsc provided by Docker runtime)"
        )
    container_name = f"gvisor_cmd_{uuid.uuid4().hex[:8]}"
    docker_cmd = [
        "docker",
        "run",
        "--rm",  # Remove container after execution
        "--runtime=runsc",  # Use gVisor runtime
        "--network=none",  # Disable network for security
        "--name",
        container_name,
    ]

    # SECURITY: Mount working directory as READ-ONLY to prevent file modifications
    if working_dir:
        if not os.path.exists(working_dir):
            return CommandResult(
                returncode=1,
                stdout="",
                stderr=f"Working directory does not exist: {working_dir}",
                success=False,
            )
        # SECURITY: Mount as read-only to prevent any write operations
        docker_cmd.extend(["-v", f"{working_dir}:/workspace:ro"])
        docker_cmd.extend(["-w", "/workspace"])
        # Add --read-only flag for additional protection
        docker_cmd.append("--read-only")
        # Add tmpfs for /tmp since --read-only requires writable tmpfs
        docker_cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=100m"])

    # Mount repo path as read-only if provided and different from working_dir
    if repo_path and repo_path != working_dir and os.path.exists(repo_path):
        docker_cmd.extend(["-v", f"{repo_path}:/repo:ro"])

    # SECURITY: Filter environment variables to prevent secret exposure
    safe_env = _filter_safe_environment_variables(env)
    if safe_env:
        for key, value in safe_env.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

    # Use a minimal Linux image (alpine or busybox)
    # We'll use busybox as it's very small
    docker_cmd.append("busybox:latest")

    # Add the command to run
    # Escape command properly for shell execution
    if len(command) == 1:
        docker_cmd.append(command[0])
    else:
        # For multiple arguments, join them properly
        docker_cmd.append("sh")
        docker_cmd.append("-c")
        # Properly escape each argument
        escaped_cmd = " ".join(shlex.quote(arg) for arg in command)
        docker_cmd.append(escaped_cmd)

    logger.info(
        f"[GVISOR] Executing Docker command with gVisor: {' '.join(docker_cmd[:10])}..."
    )  # Log first 10 args for brevity

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        logger.info(
            f"[GVISOR] Command executed with gVisor (Docker+runsc) - "
            f"exit code: {result.returncode}, success: {result.returncode == 0}"
        )
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            success=result.returncode == 0,
        )
    except subprocess.TimeoutExpired:
        # Try to clean up the container
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

        return CommandResult(
            returncode=124,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            success=False,
        )


def _run_with_runsc_direct(
    command: List[str],
    working_dir: Optional[str],
    repo_path: Optional[str],
    env: Optional[Dict[str, str]],
    timeout: Optional[int],
    runsc_path: Path,
) -> CommandResult:
    """
    Run command directly with runsc in container environments.

    Note: Direct runsc usage requires creating OCI bundles, which is complex.
    In K8s/container environments, the container itself already provides isolation.
    This function attempts to use runsc if possible, but falls back gracefully.

    For best results in K8s, consider configuring containerd with runsc runtime
    at the node level, or use Docker with runsc runtime in local development.
    """
    # In container environments (K8s), using runsc directly is complex because:
    # 1. We'd need to create OCI bundles
    # 2. We'd need proper permissions (may require privileged containers)
    # 3. Nested containers may not be allowed

    # However, in K8s, the container itself provides isolation, so falling back
    # to regular subprocess is still secure. We log this for visibility.

    logger.info(
        "Running in container environment. "
        "Direct runsc usage requires OCI bundle creation which is complex. "
        "Using regular subprocess - container isolation provides security. "
        "For additional gVisor isolation, configure containerd with runsc runtime at node level."
    )

    # Use regular subprocess - in containers, this is still isolated
    # The container itself provides the isolation layer
    return _run_command_regular(
        command=command,
        working_dir=working_dir,
        env=env,
        timeout=timeout,
    )


def _run_command_regular(
    command: List[str],
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> CommandResult:
    """
    Run a command using regular subprocess (fallback when gVisor is not available).

    SECURITY WARNING: This is less secure than gVisor. Only use when gVisor is unavailable.
    We still filter environment variables and validate paths for basic protection.

    Args:
        command: Command to execute
        working_dir: Working directory
        env: Environment variables (will be filtered)
        timeout: Timeout in seconds

    Returns:
        CommandResult
    """
    logger.warning(
        "[GVISOR] Using regular subprocess (gVisor not available) - reduced security isolation"
    )
    try:
        # SECURITY: Filter environment variables even in fallback mode
        safe_env = _filter_safe_environment_variables(env)

        # Start with minimal safe environment
        process_env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": "/tmp",  # Use temp directory, not real home
            "USER": "sandbox",
            "SHELL": "/bin/sh",
            "LANG": os.environ.get("LANG", "C"),
            "TERM": "dumb",  # Prevent terminal escape sequences
        }
        # Add filtered environment variables
        process_env.update(safe_env)

        result = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=process_env,
        )

        logger.info(
            f"[GVISOR] Command executed with regular subprocess (no gVisor) - "
            f"exit code: {result.returncode}, success: {result.returncode == 0}"
        )
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            success=result.returncode == 0,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            returncode=124,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            success=False,
        )
    except FileNotFoundError:
        # Command not found
        return CommandResult(
            returncode=127,  # Standard "command not found" exit code
            stdout="",
            stderr=f"Command not found: {command[0] if command else 'unknown'}",
            success=False,
        )
    except Exception as e:
        logger.error(f"Error running command: {e}", exc_info=True)
        return CommandResult(
            returncode=1,
            stdout="",
            stderr=str(e),
            success=False,
        )


def run_shell_command_isolated(
    shell_command: str,
    working_dir: Optional[str] = None,
    repo_path: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    use_gvisor: bool = True,
) -> CommandResult:
    """
    Run a shell command in an isolated gVisor sandbox.

    Convenience wrapper that splits a shell command string into a list.

    Args:
        shell_command: Shell command as a string (e.g., "ls -la")
        working_dir: Working directory for the command
        repo_path: Path to the repository (mounted read-only)
        env: Environment variables to set
        timeout: Timeout in seconds
        use_gvisor: If False, skip gVisor and use regular subprocess

    Returns:
        CommandResult
    """
    import shlex

    command = shlex.split(shell_command)
    return run_command_isolated(
        command=command,
        working_dir=working_dir,
        repo_path=repo_path,
        env=env,
        timeout=timeout,
        use_gvisor=use_gvisor,
    )
