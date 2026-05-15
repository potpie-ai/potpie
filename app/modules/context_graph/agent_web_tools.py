"""Web agent tools for the context-graph reconciliation agent.

Wraps Potpie's existing user-scoped web tooling — ``WebSearchTool``
(Perplexity via the user's OpenRouter/provider config) and
``WebpageExtractorTool`` (Firecrawl) — as pydantic-deep tool callables so
the ingestion agent can ground external references (changelogs, vendor
docs, status pages, RFC links) instead of guessing from a webhook summary.

This module is host-coupled by design: it lives on the Potpie side rather
than inside the context-engine hexagon because it depends on
``app.modules.intelligence`` infrastructure. The builder is wired via
``PydanticDeepReconciliationAgent.add_extra_tools`` in
:mod:`app.modules.context_graph.wiring`.

Both tools resolve the *pot owner's* account so web search bills/keys and
provider routing match "the account configured for that user". Each tool
soft-skips (the whole builder returns ``[]``) when its API key is absent,
mirroring the standalone factory functions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

PotUserResolver = Callable[[Optional[str]], Optional[str]]
"""``(pot_id) -> user_id | None`` — the account web reads run as."""


def build_web_tools(
    *,
    db: Session,
    user_resolver: PotUserResolver,
) -> Callable[[Any], list[Any]]:
    """Return a per-batch builder exposing web search + page extraction.

    Args:
        db: Request/worker-scoped session reused by the underlying tools
            (they only read provider config through it).
        user_resolver: Resolves ``state.pot_id`` to the user id whose
            provider keys/quota the web calls run against. Returning
            ``None`` disables web tools for the batch.
    """

    def _builder(state: Any) -> list[Any]:
        try:
            from pydantic_ai import Tool  # type: ignore[import-not-found]
        except Exception:
            try:
                from pydantic_deep import Tool  # type: ignore[import-not-found, no-redef]
            except Exception:
                logger.warning(
                    "pydantic-ai/pydantic-deep Tool not importable; skipping web tools"
                )
                return []

        pot_id = getattr(state, "pot_id", None)
        try:
            user_id = user_resolver(pot_id)
        except Exception:
            logger.exception("web tools: pot→user resolution failed for %s", pot_id)
            user_id = None
        if not user_id:
            logger.info(
                "web tools disabled for pot %s (no owning user resolved)", pot_id
            )
            return []

        tools: list[Any] = []

        if (os.getenv("OPENROUTER_API_KEY") or "").strip():
            from app.modules.intelligence.tools.web_tools.web_search_tool import (
                WebSearchTool,
            )

            search = WebSearchTool(db, user_id)

            async def web_search(query: str) -> dict[str, Any]:
                """Search the web and return a cited answer for an external question."""
                try:
                    return await search.arun(query)
                except Exception as exc:
                    logger.exception("web_search failed")
                    return {"success": False, "content": str(exc), "citations": []}

            tools.append(
                Tool(
                    web_search,
                    name="web_search",
                    description=(
                        "Search the public web and get a synthesized, cited "
                        "answer. Use for things not in the graph or repo: "
                        "vendor changelogs, dependency advisories, API/SDK "
                        "behaviour, incident status pages. Pass a full "
                        "question, not keywords. Returns {success, content, "
                        "citations}. Ground claims in the result; never invent."
                    ),
                )
            )
        else:
            logger.info("web_search tool disabled (OPENROUTER_API_KEY unset)")

        if (os.getenv("FIRECRAWL_API_KEY") or "").strip():
            from app.modules.intelligence.tools.web_tools.webpage_extractor_tool import (
                WebpageExtractorTool,
            )

            extractor = WebpageExtractorTool(db, user_id)

            async def web_extract_page(url: str) -> dict[str, Any]:
                """Extract the readable text/markdown content of one web page by URL."""
                try:
                    return await extractor.arun(url)
                except Exception as exc:
                    logger.exception("web_extract_page failed for %s", url)
                    return {"success": False, "error": str(exc), "content": None}

            tools.append(
                Tool(
                    web_extract_page,
                    name="web_extract_page",
                    description=(
                        "Fetch one web page by URL and return its main content "
                        "as markdown plus metadata. Use to read a specific link "
                        "referenced by an event (a doc, RFC, postmortem, release "
                        "page). Returns {success, content, metadata}."
                    ),
                )
            )
        else:
            logger.info("web_extract_page tool disabled (FIRECRAWL_API_KEY unset)")

        return tools

    return _builder


__all__ = ["build_web_tools", "PotUserResolver"]
