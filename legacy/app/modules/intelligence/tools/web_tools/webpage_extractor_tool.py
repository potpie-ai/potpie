import asyncio
import os
from typing import Any, Dict, Optional

from firecrawl import FirecrawlApp
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.tool_utils import truncate_dict_response
from observability import get_logger

logger = get_logger(__name__)


class WebpageExtractorInput(BaseModel):
    url: str = Field(description="The URL of the webpage to extract content from")


class WebpageExtractorTool:
    name = "Webpage Extractor"
    description = """Extracts the full text content from a webpage.
        :param url: string, the URL of the webpage to extract.

            example:
            {
                "url": "https://example.com/article"
            }

        Returns dictionary containing the webpage content, metadata, and success status.

        ⚠️ IMPORTANT: Large webpages may result in truncated responses (max 80,000 characters).
        If the response is truncated, a notice will be included indicating the truncation occurred.
        """

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        # fastCRW is a Firecrawl-API-compatible web scraper (single binary; self-host
        # or cloud). Reuse the same FirecrawlApp client and point it at the fastCRW
        # base URL. Falls back to Firecrawl when CRW_API_KEY is not set.
        self.api_key = os.getenv("CRW_API_KEY") or os.getenv("FIRECRAWL_API_KEY")
        self.api_url = None
        if os.getenv("CRW_API_KEY"):
            # Default to fastCRW cloud; allow self-host override via CRW_API_URL.
            self.api_url = os.getenv("CRW_API_URL", "https://fastcrw.com/api")
        self.firecrawl = None
        if self.api_key:
            if self.api_url:
                self.firecrawl = FirecrawlApp(api_key=self.api_key, api_url=self.api_url)
            else:
                self.firecrawl = FirecrawlApp(api_key=self.api_key)

    async def arun(self, url: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.run, url)

    def run(self, url: str) -> Dict[str, Any]:
        try:
            content = self._extract_content(url)
            if not content:
                return {
                    "success": False,
                    "error": "Failed to extract content",
                    "content": None,
                }
            return content
        except Exception as e:
            logger.exception("An unexpected error occurred")
            return {
                "success": False,
                "error": f"An unexpected error occurred: {str(e)}",
                "content": None,
            }

    def _extract_content(self, url: str) -> Optional[Dict[str, Any]]:
        if not self.api_key or not self.firecrawl:
            return None

        if not url:
            return None

        response = self.firecrawl.scrape_url(
            url=url,
            params={
                "formats": ["markdown"],
            },
        )
        if not response.get("markdown"):
            return None

        data = response.get("markdown", {})
        metadata = response.get("metadata", {})

        result = {
            "success": True,
            "content": data,
            "metadata": {
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "language": metadata.get("language"),
                "url": metadata.get("sourceURL", url),
            },
        }

        # Truncate response if it exceeds character limits
        truncated_result = truncate_dict_response(result)
        if len(str(result)) > 80000:
            logger.warning(
                f"webpage_extractor_tool output truncated for URL: {url}", url=url
            )
        return truncated_result


def webpage_extractor_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    if not os.getenv("CRW_API_KEY") and not os.getenv("FIRECRAWL_API_KEY"):
        logger.warning(
            "Neither CRW_API_KEY nor FIRECRAWL_API_KEY set, webpage extractor tool "
            "will not be initialized"
        )
        return None

    tool_instance = WebpageExtractorTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Webpage Content Extractor",
        description="""Extracts the full text content from a webpage.
        :param url: string, the URL of the webpage to extract.

            example:
            {
                "url": "https://example.com/article"
            }

        Returns dictionary containing the webpage content, metadata, and success status.""",
        args_schema=WebpageExtractorInput,
    )
