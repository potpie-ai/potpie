"""Minimal OpenRouter smoke test (optional)."""

from __future__ import annotations

import asyncio
import os
import sys

from pydantic_ai import Agent

from poc.config.provider import get_model


async def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)
    agent = Agent(get_model(), instructions="Reply with OK.")
    r = await agent.run("ping")
    print(r.output)


if __name__ == "__main__":
    asyncio.run(main())
