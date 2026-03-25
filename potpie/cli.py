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
CONFIG_DIR = Path.home() / ".potpie"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    """Load CLI configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(config: dict):
    """Save CLI configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


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


# ─── HTTP Client ─────────────────────────────────────────────────────────────

class PotPieClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or get_base_url()).rstrip("/")
        if httpx is None:
            raise ImportError("httpx is required: pip install httpx")
        self.client = httpx.Client(timeout=60.0, headers=get_headers())

    def request(self, method: str, path: str, **kwargs) -> dict:
        """Make an API request."""
        url = f"{self.base_url}{path}"
        try:
            resp = self.client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            error_body = e.response.text[:500] if e.response else str(e)
            print(f"❌ HTTP {e.response.status_code}: {error_body}")
            sys.exit(1)
        except httpx.ConnectError:
            print(f"❌ Cannot connect to {self.base_url}")
            print("   Is the server running? Try: potpie start")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    def close(self):
        self.client.close()


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_start(args):
    """Start the local PotPie server using Docker Compose."""
    print("🚀 Starting PotPie server...")

    compose_file = Path(args.compose or "compose.yaml")
    if not compose_file.exists():
        # Try to find it relative to the CLI location
        cli_dir = Path(__file__).parent.parent
        compose_file = cli_dir / "compose.yaml"

    if not compose_file.exists():
        print("❌ compose.yaml not found. Run this from the potpie directory.")
        sys.exit(1)

    cmd = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
    if args.build:
        cmd.append("--build")

    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("\n✅ PotPie server started!")
        print(f"   API: {get_base_url()}")
        print(f"   UI:  http://localhost:3000")
        print("\n   Run 'potpie status' to check readiness.")
    else:
        print("\n❌ Failed to start. Check Docker is running.")
        sys.exit(1)


def cmd_stop(args):
    """Stop the local PotPie server."""
    print("⏹️  Stopping PotPie server...")

    compose_file = Path(args.compose or "compose.yaml")
    if not compose_file.exists():
        cli_dir = Path(__file__).parent.parent
        compose_file = cli_dir / "compose.yaml"

    cmd = ["docker", "compose", "-f", str(compose_file), "down"]
    if args.volumes:
        cmd.append("-v")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("✅ PotPie server stopped.")
    else:
        print("❌ Failed to stop.")
        sys.exit(1)


def cmd_status(args):
    """Check server status."""
    client = PotPieClient()
    try:
        # Try to list projects as a health check
        result = client.request("GET", "/projects/list")
        print(f"✅ Server is running at {client.base_url}")
        print(f"   Projects: {len(result) if isinstance(result, list) else 'N/A'}")
    except SystemExit:
        pass
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
            print(f"❌ Path not found: {local_path}")
            sys.exit(1)
        repo_path = str(local_path)

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

        while time.time() - start < max_wait:
            time.sleep(5)
            elapsed = int(time.time() - start)

            try:
                status_result = client.request("GET", f"/parsing-status/{project_id}")
                current_status = status_result.get("status", "unknown")
                print(f"   [{elapsed}s] Status: {current_status}")

                if current_status in ("ready", "completed", "success"):
                    print(f"\n✅ Parsing complete! ({elapsed}s)")
                    print(f"   Project ID: {project_id}")
                    print(f"\n   Start chatting: potpie chat {project_id}")
                    return
                elif current_status in ("failed", "error"):
                    print(f"\n❌ Parsing failed.")
                    if "error" in status_result:
                        print(f"   Error: {status_result['error']}")
                    sys.exit(1)
            except Exception:
                pass  # Continue polling

        print(f"\n⏱️  Timeout after {max_wait}s. Parsing may still be running.")
        print(f"   Check later: potpie status")

    except SystemExit:
        pass
    finally:
        client.close()


def cmd_chat(args):
    """Interactive chat with an agent."""
    client = PotPieClient()
    project_id = args.project_id
    agent_id = args.agent or "codebase_qna_agent"

    print(f"💬 Starting chat session")
    print(f"   Project: {project_id}")
    print(f"   Agent: {agent_id}")
    print(f"   Type 'quit' or Ctrl+C to exit.\n")

    # Create conversation
    try:
        conv_result = client.request("POST", "/conversations/", json={
            "project_ids": [project_id],
            "agent_ids": [agent_id],
        })
        conversation_id = conv_result.get("conversation_id")
        print(f"   Session ID: {conversation_id}\n")

        # Interactive loop
        while True:
            try:
                user_input = input("You: ").strip()
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

    except SystemExit:
        pass
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
    except SystemExit:
        pass
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
    except SystemExit:
        pass
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
        config = load_config()
        config["api_key"] = args.set_key
        save_config(config)
        print("✅ API key saved.")

    if not args.set_url and not args.set_key:
        config = load_config()
        print("📋 Current configuration:\n")
        print(f"  Server URL: {config.get('base_url', DEFAULT_BASE_URL)}")
        key = config.get("api_key", DEFAULT_API_KEY)
        if key:
            print(f"  API Key: {key[:8]}...")
        else:
            print(f"  API Key: (not set)")
        print(f"\n  Config file: {CONFIG_FILE}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
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
    p_config.add_argument("--set-key", help="Set API key")
    p_config.set_defaults(func=cmd_config)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
