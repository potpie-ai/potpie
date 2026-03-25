"""web_search_tool, webpage_extractor."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from pydantic_ai import RunContext

from poc.managers.deps import PoCDeepDeps


async def web_search_tool(
    ctx: RunContext[PoCDeepDeps], query: str, max_results: int = 5
) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"search error: {e}"
    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}: {r.get('href', '')}")
    return "\n".join(lines) if lines else "(no results)"


async def webpage_extractor(ctx: RunContext[PoCDeepDeps], url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
    except Exception as e:
        return f"fetch error: {e}"
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return text[:50_000]
