"""potpie list-agents – display all available agents."""

from __future__ import annotations

import sys

from potpie.cli.client import PotpieClient, PotpieClientError


def list_agents(base_url: str | None = None) -> None:
    """Fetch and display available agents from the local Potpie server.

    Args:
        base_url: Override the server URL (default: ``http://localhost:8001``).
    """
    client = PotpieClient(base_url) if base_url else PotpieClient()

    try:
        agents = client.list_agents(list_system_agents=True)
    except PotpieClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not agents:
        print("No agents found.")
        return

    print(f"{'ID':<40}  {'NAME':<35}  STATUS")
    print("-" * 85)
    for agent in agents:
        agent_id = agent.get("id", "")
        name = agent.get("name", "")
        status = agent.get("status", "")
        description = agent.get("description", "")
        print(f"{agent_id:<40}  {name:<35}  {status}")
        if description:
            # Indent and wrap description
            wrapped = description[:80] + ("…" if len(description) > 80 else "")
            print(f"  {wrapped}")
