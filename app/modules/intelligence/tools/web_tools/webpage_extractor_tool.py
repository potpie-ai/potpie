import asyncio
import logging
import os
from typing import Any, Dict, Optional

from firecrawl import FirecrawlApp
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session


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
        """

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        self.firecrawl = None
        if self.api_key:
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
            logging.exception(f"An unexpected error occurred: {str(e)}")
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

        return {
            "success": True,
            "content": data,
            "metadata": {
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "language": metadata.get("language"),
                "url": metadata.get("sourceURL", url),
            },
        }


def webpage_extractor_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    if not os.getenv("FIRECRAWL_API_KEY"):
        logging.warning(
            "FIRECRAWL_API_KEY not set, webpage extractor tool will not be initialized"
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
