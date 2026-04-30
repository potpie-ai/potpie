#!/usr/bin/env python3
"""
Potpie CLI - Command-line interface for local Potpie usage

This CLI provides commands to:
- Start and stop local Potpie services
- Parse repositories
- Chat with agents

Usage:
    potpie start
    potpie stop
    potpie parse <repo-path> [--branch <branch-name>]
    potpie chat <project-id> --agent <agent-name> [--branch <branch-name>]
"""

import argparse
import os
import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import Optional

# Constants
POTPIE_DIR = Path.home() / ".potpie"
PID_FILE = POTPIE_DIR / "potpie.pid"
API_BASE_URL = "http://localhost:8001"


def ensure_potpie_dir():
    """Ensure the potpie config directory exists."""
    POTPIE_DIR.mkdir(parents=True, exist_ok=True)


def get_potpiedir() -> Path:
    """Get the potpie directory from environment or default."""
    potpie_dir = os.environ.get("POTPIE_DIR", str(Path.cwd()))
    return Path(potpie_dir)


def is_server_running() -> bool:
    """Check if the potpie server is running by checking the pid file and port."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_command(args):
    """Start the Potpie server."""
    ensure_potpie_dir()
    
    if is_server_running():
        print("✓ Potpie server is already running")
        return 0
    
    potpie_dir = get_potpiedir()
    
    if not (potpie_dir / "scripts" / "start.sh").exists():
        print(f"Error: Cannot find Potpie installation at {potpie_dir}")
        print("Please set POTPIE_DIR environment variable to your potpie installation path")
        return 1
    
    print("Starting Potpie server...")
    
    # Run start script in background
    try:
        process = subprocess.Popen(
            ["bash", str(potpiedir / "scripts" / "start.sh")],
            cwd=str(potpiedir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        
        # Save PID
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))
        
        print(f"Server starting with PID {process.pid}")
        print("Waiting for server to be ready...")
        
        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            if is_server_running():
                print("✓ Potpie server is running on http://localhost:8001")
                return 0
            time.sleep(1)
        
        print("⚠ Server started but not responding yet. Check logs for details.")
        return 0
        
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1


def stop_command(args):
    """Stop the Potpie server."""
    if not is_server_running():
        print("Potpie server is not running")
        return 0
    
    potpie_dir = get_potpiedir()
    
    if not (potpie_dir / "scripts" / "stop.sh").exists():
        print(f"Error: Cannot find Potpie installation at {potpie_dir}")
        return 1
    
    print("Stopping Potpie server...")
    
    try:
        result = subprocess.run(
            ["bash", str(potpiedir / "scripts" / "stop.sh")],
            cwd=str(potpiedir),
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print("✓ Potpie server stopped successfully")
            if PID_FILE.exists():
                PID_FILE.unlink()
            return 0
        else:
            print(f"Error stopping server: {result.stderr}")
            return 1
            
    except Exception as e:
        print(f"Error stopping server: {e}")
        return 1


def validate_repo_path(repo_path: str) -> bool:
    """Validate that the repo path exists and is a git repository."""
    path = Path(repo_path).expanduser().resolve()
    
    if not path.exists():
        print(f"Error: Path '{repo_path}' does not exist")
        return False
    
    if not path.is_dir():
        print(f"Error: Path '{repo_path}' is not a directory")
        return False
    
    if not (path / ".git").exists():
        print(f"Warning: Path '{repo_path}' does not appear to be a git repository")
    
    return True


def get_current_branch(repo_path: str) -> str:
    """Get the current git branch of the repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "main"


def parse_command(args):
    """Parse a repository using the Potpie API."""
    if not is_server_running():
        print("Error: Potpie server is not running. Run 'potpie start' first.")
        return 1
    
    repo_path = Path(args.repo_path).expanduser().resolve()
    
    if not validate_repo_path(str(repo_path)):
        return 1
    
    branch = args.branch or get_current_branch(str(repo_path))
    
    print(f"Parsing repository: {repo_path}")
    print(f"Branch: {branch}")
    
    try:
        # Submit parse request
        response = requests.post(
            f"{API_BASE_URL}/parse",
            json={
                "repo_path": str(repo_path),
                "branch_name": branch,
            },
            timeout=30,
        )
        
        if response.status_code != 200:
            print(f"Error: Failed to submit parse request: {response.text}")
            return 1
        
        data = response.json()
        project_id = data.get("project_id")
        
        if not project_id:
            print("Error: No project ID returned from server")
            return 1
        
        print(f"Project ID: {project_id}")
        print("Parsing in progress...")
        
        # Poll for status
        max_retries = 300  # 5 minutes with 1-second intervals
        for i in range(max_retries):
            status_response = requests.get(
                f"{API_BASE_URL}/parsing-status/{project_id}",
                timeout=10,
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get("status", "unknown")
                
                if status == "ready":
                    print(f"✓ Repository parsed successfully!")
                    print(f"Project ID: {project_id}")
                    return 0
                elif status == "error":
                    print(f"✗ Parsing failed: {status_data.get('error', 'Unknown error')}")
                    return 1
                else:
                    # Show progress
                    progress = status_data.get("progress", 0)
                    print(f"  Status: {status} ({progress}%)", end="\r", flush=True)
            
            time.sleep(1)
        
        print("\n⚠ Parsing is taking longer than expected. Check status with:")
        print(f"  potpie status {project_id}")
        return 0
        
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
        return 1


def chat_command(args):
    """Chat with a Potpie agent."""
    if not is_server_running():
        print("Error: Potpie server is not running. Run 'potpie start' first.")
        return 1
    
    project_id = args.project_id
    agent_name = args.agent
    branch = args.branch
    
    # Validate project status
    try:
        status_response = requests.get(
            f"{API_BASE_URL}/parsing-status/{project_id}",
            timeout=10,
        )
        
        if status_response.status_code != 200:
            print(f"Error: Could not check project status: {status_response.text}")
            return 1
        
        status_data = status_response.json()
        status = status_data.get("status")
        
        if status != "ready":
            print(f"Error: Project is not ready (status: {status})")
            print("Please wait for parsing to complete or check for errors.")
            return 1
        
        print(f"Project ready: {project_id}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error checking project status: {e}")
        return 1
    
    # Create conversation
    try:
        print(f"Creating conversation with agent '{agent_name}'...")
        
        conv_response = requests.post(
            f"{API_BASE_URL}/conversations/",
            json={
                "project_ids": [project_id],
                "agent_ids": [agent_name],
            },
            timeout=10,
        )
        
        if conv_response.status_code != 200:
            print(f"Error: Failed to create conversation: {conv_response.text}")
            return 1
        
        conv_data = conv_response.json()
        conversation_id = conv_data.get("conversation_id")
        
        if not conversation_id:
            print("Error: No conversation ID returned")
            return 1
        
        print(f"Conversation ID: {conversation_id}")
        print(f"Agent: {agent_name}")
        print("\nType your message (or 'quit' to exit):")
        
        # Interactive chat loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                print("\nExiting...")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                print("Exiting...")
                break
            
            # Send message
            try:
                print("Agent: ", end="", flush=True)
                
                msg_response = requests.post(
                    f"{API_BASE_URL}/conversations/{conversation_id}/message/",
                    json={"content": user_input},
                    timeout=120,
                )
                
                if msg_response.status_code == 200:
                    response_data = msg_response.json()
                    if isinstance(response_data, list) and len(response_data) > 0:
                        print(response_data[-1].get("content", "No response"))
                    else:
                        print(response_data.get("content", "No response"))
                else:
                    print(f"Error: {msg_response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Error sending message: {e}")
        
        return 0
        
    except requests.exceptions.RequestException as e:
        print(f"Error in conversation: {e}")
        return 1


def status_command(args):
    """Check the status of a parsing project."""
    if not is_server_running():
        print("Error: Potpie server is not running. Run 'potpie start' first.")
        return 1
    
    project_id = args.project_id
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/parsing-status/{project_id}",
            timeout=10,
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Project ID: {project_id}")
            print(f"Status: {data.get('status', 'unknown')}")
            print(f"Progress: {data.get('progress', 0)}%")
            if data.get('error'):
                print(f"Error: {data['error']}")
            return 0
        else:
            print(f"Error: {response.text}")
            return 1
            
    except requests.exceptions.RequestException as e:
        print(f"Error checking status: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        prog="potpie",
        description="Potpie CLI - Command-line interface for local Potpie usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  potpie start                                   Start the Potpie server
  potpie stop                                    Stop the Potpie server
  potpie parse /path/to/repo --branch main       Parse a repository
  potpie chat <project-id> --agent <agent>       Chat with an agent
  potpie status <project-id>                     Check parsing status
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the Potpie server")
    start_parser.set_defaults(func=start_command)
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the Potpie server")
    stop_parser.set_defaults(func=stop_command)
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse a repository")
    parse_parser.add_argument("repo_path", help="Path to the repository")
    parse_parser.add_argument(
        "--branch", "-b",
        help="Branch name (defaults to current branch)",
    )
    parse_parser.set_defaults(func=parse_command)
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with a Potpie agent")
    chat_parser.add_argument("project_id", help="Project ID from parsing")
    chat_parser.add_argument(
        "--agent", "-a",
        required=True,
        help="Agent name to use for the conversation",
    )
    chat_parser.add_argument(
        "--branch", "-b",
        help="Branch name (optional)",
    )
    chat_parser.set_defaults(func=chat_command)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check parsing status")
    status_parser.add_argument("project_id", help="Project ID to check")
    status_parser.set_defaults(func=status_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
