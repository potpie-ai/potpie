#!/usr/bin/env python3
"""
PotPie CLI — Local Development Interface
========================================
Command-line interface for local development interactions with PotPie.

Usage:
    potpie start                  # Start the local PotPie server
    potpie stop                   # Stop the local PotPie server
    potpie parse <repo-path>      # Parse a repository
    potpie chat <project-id>      # Interactive chat with an agent
    potpie status                 # Check server status
    potpie projects               # List parsed projects
    potpie agents                 # List available agents
"""

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8001"
DEFAULT_API_KEY = os.environ.get("POTPIE_API_KEY", "")
DEFAULT_COMPOSE_FILE = "compose.yaml"
CONFIG_DIR = Path.home() / ".potpie"
CONFIG_FILE = CONFIG_DIR / "config.json"


class PotPieError(Exception):
    """CLI error with exit code."""

    def __init__(self, message: str, exit_code: int = 1):
        """Initialize error with message and exit code."""
        super().__init__(message)
        self.exit_code = exit_code


def load_config() -> dict:
    """Load CLI configuration."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, IOError) as e:
            print(f"⚠️  Config file is invalid/unreadable: {e}")
            print("   Using empty configuration.")
            return {}
    return {}


def save_config(config: dict):
    """Save CLI configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Set secure permissions for config directory
    os.chmod(CONFIG_DIR, 0o700)

    # Write config file with secure permissions
    fd = os.open(CONFIG_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(config, f, indent=2)
    # Force permissions in case file already existed with looser mode
    os.chmod(CONFIG_FILE, 0o600)


def get_base_url() -> str:
    """Get the PotPie server URL."""
    config = load_config()
    return config.get("base_url", DEFAULT_BASE_URL)


def get_api_key() -> str:
    """Get the API key."""
    config = load_config()
    return config.get("api_key", DEFAULT_API_KEY)


def get_headers() -> dict:
    """Get request headers with API key."""
    api_key = get_api_key()
    headers = {"Content-Type": "application/json", "User-Agent": "PotPie-CLI/1.0"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def resolve_compose_file(compose_arg: Optional[str]) -> Path:
    """Resolve compose.yaml path from argument or default locations."""
    if compose_arg:
        compose_file = Path(compose_arg)
        if not compose_file.exists():
            raise PotPieError(f"Compose file not found: {compose_file}")
        return compose_file
    compose_file = Path(DEFAULT_COMPOSE_FILE)
    if not compose_file.exists():
        cli_dir = Path(__file__).parent.parent
        compose_file = cli_dir / DEFAULT_COMPOSE_FILE
    return compose_file


# ─── HTTP Client ─────────────────────────────────────────────────────────────

class PotPieClient:
    """HTTP client wrapper for PotPie API communication."""

    def __init__(self, base_url: Optional[str] = None):
        """Initialize client with optional base URL override."""
        self.base_url = (base_url or get_base_url()).rstrip("/")
        if httpx is None:
            raise ImportError("httpx is required: pip install httpx")
        self.client = httpx.Client(headers=get_headers())

    def request(self, method: str, path: str, **kwargs) -> dict:
        """Make an API request."""
        url = f"{self.base_url}{path}"
        try:
            resp = self.client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            error_body = e.response.text[:500] if e.response else str(e)
            raise PotPieError(f"HTTP {e.response.status_code}: {error_body}")
        except httpx.ConnectError:
            raise PotPieError(
                f"Cannot connect to {self.base_url}\n"
                "   Is the server running? Try: potpie start"
            )
        except Exception as e:
            raise PotPieError(f"Error: {e}")

    def close(self):
        """Close the underlying HTTP client connection."""
        self.client.close()


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_start(args):
    """Start the local PotPie server using Docker Compose."""
    print("🚀 Starting PotPie server...")

    compose_file = resolve_compose_file(args.compose)

    if not compose_file.exists():
        print("❌ compose.yaml not found. Run this from the potpie directory.")
        raise PotPieError("compose.yaml not found")

    cmd = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
    if args.build:
        cmd.append("--build")

    print(f"   Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=False, check=False)
    except FileNotFoundError:
        raise PotPieError("docker not found. Please install Docker.")

    if result.returncode == 0:
        print("\n✅ PotPie server started!")
        print(f"   API: {get_base_url()}")
        print("   UI:  http://localhost:3000")
        print("\n   Run 'potpie status' to check readiness.")
    else:
        raise PotPieError("Failed to start. Check Docker is running.")


def cmd_stop(args):
    """Stop the local PotPie server."""
    print("⏹️  Stopping PotPie server...")

    compose_file = resolve_compose_file(args.compose)

    cmd = ["docker", "compose", "-f", str(compose_file), "down"]
    if args.volumes:
        cmd.append("-v")

    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        raise PotPieError("docker not found. Please install Docker.")
    if result.returncode == 0:
        print("✅ PotPie server stopped.")
    else:
        raise PotPieError("Failed to stop.")


def cmd_status(args):
    """Check server status and project details."""
    client = PotPieClient()
    try:
        # Check server health
        result = client.request("GET", "/projects/list")
        print(f"✅ Server is running at {client.base_url}")

        if isinstance(result, list):
            print(f"📊 Projects: {len(result)}")

            # Show project details if requested
            if args.project_id:
                for project in result:
                    if project.get("project_id") == args.project_id or project.get("name") == args.project_id:
                        print(f"\n📋 Project: {project.get('name', args.project_id)}")
                        print(f"   ID: {project.get('project_id')}")
                        print(f"   Status: {project.get('status', 'unknown')}")
                        print(f"   Created: {project.get('created_at', 'N/A')}")
                        print(f"   Updated: {project.get('updated_at', 'N/A')}")
                        break
                else:
                    print(f"\n❌ Project not found: {args.project_id}")
            else:
                # Show summary
                status_counts = {}
                for project in result:
                    status = project.get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1

                print(f"   Status summary: {', '.join([f'{k}: {v}' for k, v in status_counts.items()])}")
        else:
            print("   Projects: N/A")
    except PotPieError:
        raise
    finally:
        client.close()


def cmd_parse(args):
    """Parse a repository."""
    client = PotPieClient()

    repo_path = args.repo_path
    branch = args.branch

    # Validate repo path
    if not repo_path.startswith(("http://", "https://", "git@")):
        # Local path
        local_path = Path(repo_path).resolve()
        if not local_path.exists():
            raise PotPieError(f"Path not found: {local_path}")
        if not local_path.is_dir():
            raise PotPieError(f"Path is not a directory: {local_path}")
        # Check if it's a git repository
        if not (local_path / ".git").exists():
            raise PotPieError(f"Not a git repository: {local_path}")
        # Validate branch if specified
        if branch:
            try:
                result = subprocess.run(
                    ["git", "-C", str(local_path), "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
                    capture_output=True,
                )
            except FileNotFoundError:
                raise PotPieError("git not found. Please install git.")
            if result.returncode != 0:
                raise PotPieError(f"Branch not found: {branch}")
        repo_path = str(local_path)
    else:
        # Remote URL validation
        try:
            if branch:
                result = subprocess.run(
                    ["git", "ls-remote", "--heads", repo_path, branch],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode != 0:
                    raise PotPieError(f"git ls-remote failed: {result.stderr.strip() or result.stdout.strip()}")
                if branch not in result.stdout:
                    raise PotPieError(f"Branch not found: {branch}")
            else:
                result = subprocess.run(
                    ["git", "ls-remote", repo_path],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode != 0:
                    raise PotPieError(f"git ls-remote failed: {result.stderr.strip() or result.stdout.strip()}")
        except FileNotFoundError:
            raise PotPieError("git not found. Please install git.")
        except subprocess.TimeoutExpired:
            raise PotPieError("git ls-remote timed out. Check your network or the repository URL.")

    # Get repo name from path
    repo_name = Path(repo_path).name if not repo_path.startswith(("http", "git@")) else repo_path.split("/")[-1].replace(".git", "")

    print(f"📂 Parsing repository: {repo_name}")
    if branch:
        print(f"   Branch: {branch}")

    # Submit parsing request
    payload = {"repo_path": repo_path, "repo_name": repo_name}
    if branch:
        payload["branch_name"] = branch

    try:
        result = client.request("POST", "/parse", json=payload)
        project_id = result.get("project_id")
        status = result.get("status", "unknown")

        print(f"   Project ID: {project_id}")
        print(f"   Status: {status}")

        if status == "ready":
            print("✅ Parsing complete!")
            return

        # Poll for status
        print("\n⏳ Waiting for parsing to complete...")
        max_wait = args.timeout
        start = time.time()
        consecutive_failures = 0

        while time.time() - start < max_wait:
            time.sleep(5)
            elapsed = int(time.time() - start)

            try:
                status_result = client.request("GET", f"/parsing-status/{project_id}")
                current_status = status_result.get("status", "unknown")
                print(f"   [{elapsed}s] Status: {current_status}")
                consecutive_failures = 0

                if current_status in ("ready", "completed", "success"):
                    print(f"\n✅ Parsing complete! ({elapsed}s)")
                    print(f"   Project ID: {project_id}")
                    print(f"\n   Start chatting: potpie chat {project_id}")
                    return
                elif current_status in ("failed", "error"):
                    print("\n❌ Parsing failed.")
                    if "error" in status_result:
                        print(f"   Error: {status_result['error']}")
                    raise PotPieError("Parsing failed")
            except PotPieError:
                raise
            except Exception as poll_err:
                consecutive_failures += 1
                print(f"   [{elapsed}s] Polling error (retry {consecutive_failures}): {poll_err}")
                if consecutive_failures >= 5:
                    raise PotPieError(f"Too many consecutive polling failures ({consecutive_failures})")

        print(f"\n⏱️  Timeout after {max_wait}s. Parsing may still be running.")
        print("   Check later: potpie status")

    except PotPieError:
        raise
    finally:
        client.close()


def cmd_chat(args):
    """Interactive chat with an agent."""
    client = PotPieClient()
    project_id = args.project_id
    agent_id = args.agent or "codebase_qna_agent"
    branch = getattr(args, "branch", None)

    try:
        print("💬 Starting chat session")
        print(f"   Project: {project_id}")
        print(f"   Agent: {agent_id}")

        # Preflight check: verify project exists and is ready
        try:
            project_result = client.request("GET", f"/projects/{project_id}")
            project_status = project_result.get("status", "unknown")

            if project_status not in ("ready", "completed", "success"):
                raise PotPieError(
                    f"Project is not ready for chatting (status: {project_status}).\n"
                    "   Please wait for parsing to complete or check with: potpie status"
                )
        except PotPieError:
            raise
        except Exception as e:
            raise PotPieError(
                f"Failed to verify project: {e}\n"
                "   Project ID may be invalid or server unavailable."
            )

        print("   Type 'quit' or Ctrl+C to exit.\n")

        # Create conversation
        conv_payload = {
            "project_ids": [project_id],
            "agent_ids": [agent_id],
        }
        if branch:
            conv_payload["branch"] = branch
        conv_result = client.request("POST", "/conversations/", json=conv_payload)
        conversation_id = conv_result.get("conversation_id")
        print(f"   Session ID: {conversation_id}\n")

        # Interactive loop
        while True:
            try:
                user_input = input("You: ").strip()[:4000]  # Limit input length
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break

            print("🤔 Thinking...", end="", flush=True)

            try:
                # Send message
                msg_result = client.request(
                    "POST",
                    f"/conversations/{conversation_id}/message/",
                    json={"content": user_input},
                    params={"stream": False},
                )

                # Clear "Thinking..."
                print("\r" + " " * 20 + "\r", end="")

                # Display response
                if isinstance(msg_result, dict):
                    content = msg_result.get("content", str(msg_result))
                    print(f"PotPie: {content}\n")
                else:
                    print(f"PotPie: {msg_result}\n")

            except Exception as e:
                print(f"\r❌ Error: {e}\n")

    except PotPieError:
        raise
    finally:
        client.close()


def cmd_projects(args):
    """List parsed projects."""
    client = PotPieClient()
    try:
        result = client.request("GET", "/projects/list")
        if isinstance(result, list) and result:
            print(f"📋 Projects ({len(result)}):\n")
            for p in result:
                name = p.get("repo_name", p.get("name", "Unknown"))
                pid = p.get("id", p.get("project_id", ""))
                branch = p.get("branch_name", "")
                print(f"  • {name} ({branch})")
                print(f"    ID: {pid}")
                print()
        else:
            print("📋 No projects found.")
            print("   Parse one: potpie parse <repo-path>")
    except PotPieError:
        raise
    finally:
        client.close()


def cmd_agents(args):
    """List available agents."""
    client = PotPieClient()
    try:
        result = client.request("GET", "/list-available-agents")
        if isinstance(result, list):
            print(f"🤖 Available Agents ({len(result)}):\n")
            for a in result:
                aid = a.get("id", a.get("agent_id", ""))
                name = a.get("name", aid)
                desc = a.get("description", "")[:80]
                print(f"  • {name} ({aid})")
                if desc:
                    print(f"    {desc}")
                print()
        else:
            print(result)
    except PotPieError:
        raise
    finally:
        client.close()


def cmd_config(args):
    """View or set configuration."""
    if args.set_url:
        config = load_config()
        config["base_url"] = args.set_url
        save_config(config)
        print(f"✅ Server URL set to: {args.set_url}")

    if args.set_key:
        # Secure interactive input for API key
        api_key = getpass.getpass("Enter API key: ")
        config = load_config()
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        config["api_key"] = api_key
        save_config(config)
        print(f"✅ API key saved ({masked_key}).")
        print("   Note: API keys are stored with file permissions 600.")

    if not args.set_url and not args.set_key:
        config = load_config()
        print("📋 Current configuration:\n")
        print(f"  Server URL: {config.get('base_url', DEFAULT_BASE_URL)}")
        key = config.get("api_key", DEFAULT_API_KEY)
        if key:
            # Show masked API key for security
            masked_key = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
            print(f"  API Key: {masked_key}")
        else:
            print("  API Key: (not set)")
        print(f"\n  Config file: {CONFIG_FILE}")
        print(f"  Config permissions: {oct(CONFIG_FILE.stat().st_mode)[-3:] if CONFIG_FILE.exists() else 'N/A'}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    """Entry point for the PotPie CLI."""
    parser = argparse.ArgumentParser(
        prog="potpie",
        description="PotPie CLI — Local Development Interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start
    p_start = subparsers.add_parser("start", help="Start the local PotPie server")
    p_start.add_argument("--compose", help="Path to compose.yaml")
    p_start.add_argument("--build", action="store_true", help="Build images before starting")
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop the local PotPie server")
    p_stop.add_argument("--compose", help="Path to compose.yaml")
    p_stop.add_argument("-v", "--volumes", action="store_true", help="Also remove volumes")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = subparsers.add_parser("status", help="Check server status")
    p_status.add_argument("project_id", nargs="?", help="Optional project ID to show details")
    p_status.set_defaults(func=cmd_status)

    # parse
    p_parse = subparsers.add_parser("parse", help="Parse a repository")
    p_parse.add_argument("repo_path", help="Repository path or URL")
    p_parse.add_argument("--branch", "-b", help="Branch name")
    p_parse.add_argument("--timeout", "-t", type=int, default=300, help="Max wait time (seconds)")
    p_parse.set_defaults(func=cmd_parse)

    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive chat with an agent")
    p_chat.add_argument("project_id", help="Project ID to chat about")
    p_chat.add_argument("--agent", "-a", help="Agent ID (default: codebase_qna_agent)")
    p_chat.add_argument("--branch", "-b", help="Branch name")
    p_chat.set_defaults(func=cmd_chat)

    # projects
    p_projects = subparsers.add_parser("projects", help="List parsed projects")
    p_projects.set_defaults(func=cmd_projects)

    # agents
    p_agents = subparsers.add_parser("agents", help="List available agents")
    p_agents.set_defaults(func=cmd_agents)

    # config
    p_config = subparsers.add_parser("config", help="View or set configuration")
    p_config.add_argument("--set-url", help="Set server URL")
    p_config.add_argument("--set-key", action="store_true", help="Set API key (interactive prompt)")
    p_config.set_defaults(func=cmd_config)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except PotPieError as e:
        print(f"❌ {e}")
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)


if __name__ == "__main__":
    main()
