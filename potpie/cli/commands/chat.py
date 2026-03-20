"""potpie chat – start an interactive chat session with a Potpie agent."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

from potpie.cli.client import PotpieClient, PotpieClientError

_QUIT_COMMANDS = {"exit", "quit", "q", ":q"}


def _format_response(data: Any) -> str:
    """Extract a human-readable reply from a server response."""
    if isinstance(data, dict):
        # Try common response fields in order of preference
        for key in ("message", "response", "content", "text"):
            value = data.get(key)
            if value:
                return str(value)
        # Fallback: pretty-print the dict
        return json.dumps(data, indent=2)
    if isinstance(data, str):
        return data
    return repr(data)


def start_chat(
    project_id: str,
    agent_id: str,
    base_url: str | None = None,
) -> None:
    """Start an interactive chat session with the specified agent.

    Args:
        project_id: The project ID to chat about.
        agent_id: The agent ID (or name) to use for the conversation.
        base_url: Override the server URL (default: ``http://localhost:8001``).
    """
    if not project_id or not project_id.strip():
        print("Error: project_id cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if not agent_id or not agent_id.strip():
        print("Error: agent must be specified.", file=sys.stderr)
        sys.exit(1)

    client = PotpieClient(base_url) if base_url else PotpieClient()

    print(f"Creating conversation for project '{project_id}' with agent '{agent_id}'…")

    try:
        conv = client.create_conversation(project_id=project_id, agent_id=agent_id)
    except PotpieClientError as exc:
        print(f"Error creating conversation: {exc}", file=sys.stderr)
        sys.exit(1)

    conversation_id = conv.get("conversation_id")
    if not conversation_id:
        print(f"Error: server did not return a conversation_id: {conv}", file=sys.stderr)
        sys.exit(1)

    print(f"Conversation started (ID: {conversation_id})")
    print(f"Type your message and press Enter. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding chat session.")
            break

        if not user_input:
            continue

        if user_input.lower() in _QUIT_COMMANDS:
            print("Ending chat session.")
            break

        try:
            response = client.send_message(conversation_id, user_input)
        except PotpieClientError as exc:
            print(f"Error sending message: {exc}", file=sys.stderr)
            continue

        reply = _format_response(response)
        print(f"\nAgent: {reply}\n")
