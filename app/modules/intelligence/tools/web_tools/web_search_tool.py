import os
import asyncio
import logging
from typing import Any, Dict, Optional

import instructor
import litellm
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.provider.provider_service import ProviderService


class WebSearchToolInput(BaseModel):
    query: str = Field(
        description="Query that needs online searching and reasoning. This needs to be a proper formed question and paragraph, not just the exact term. Make the query as informative as possible."
    )


class WebSearchToolOutput(BaseModel):
    success: bool = Field(
        description="Boolean True | False on whether answer generation was sucessful"
    )
    content: str = Field(description="Answer by the LLM Model")
    citations: list[str] = Field(
        description="List of website and citations reference for in the answer"
    )


class WebSearchTool:
    name = """Web Search Tool"""
    description = (
        """ Searches the web for any information and then gives you the answer"""
    )

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.api_key = os.getenv("OPENROUTER_API_KEY", "None")
        if self.api_key == "None":
            logging.warning("OPENROUTER_API_KEY environment variable is not set")
        self.temperature = 0.3
        self.max_tokens = 12000
        self.output_schema = WebSearchToolOutput

    async def arun(self, query: str):
        return await asyncio.to_thread(self.run, query)

    def run(self, query: str):
        try:
            response = self._make_llm_call(query)
            if not response:
                response = {
                    "success": False,
                    "content": "Tool Call Failed",
                    "citations": [],
                }
            return response
        except Exception as e:
            logging.exception(f"Error {str(e)}")
            response = {
                "success": False,
                "content": "Tool Call Error",
                "citations": [],
            }
            return response

    def _make_llm_call(self, query: str) -> Dict[str, Any]:
        try:
            messages = [{"role": "user", "content": query}]
            provider_service = ProviderService(self.sql_db, self.user_id)
            extra_params, _ = provider_service.get_extra_params_and_headers(
                "openrouter"
            )
            client = instructor.from_litellm(
                litellm.completion, mode=instructor.Mode.JSON
            )
            response = client.chat.completions.create(
                model="openrouter/perplexity/sonar",
                messages=messages,
                response_model=self.output_schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                **extra_params,
            )

            return {
                "success": response.success,
                "content": response.content,
                "citations": response.citations,
            }
        except Exception as e:
            logging.exception(f"Error in _make_llm_call: {str(e)}")
            return {
                "success": False,
                "content": f"LLM call failed: {str(e)}",
                "citations": [],
            }


def web_search_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    tool_instance = WebSearchTool(sql_db, user_id)
    if tool_instance.api_key == "None":
        # DO NOT USE THIS TOOL IF THE API KEY IS NOT SET
        return None
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Web Search Tool",
        description="""Searches the internet for current and relevant information with citations. Designed for agent workflows that require external or recent data to supplement internal codebase knowledge.

  📌 Use cases:
  - When user queries reference APIs, frameworks, or libraries not found in the codebase.
  - For understanding unfamiliar technologies, services, or vendors mentioned in the codebase.
  - To research public company documentation, changelogs, or ecosystem developments.
  - To supplement an agent's lack of training on new LLMs, SDKs, or APIs introduced after 2023.

  🧠 Tips for forming good queries:
  - Be descriptive and specific: include context, technologies, dates, and desired insights.
  - Frame queries as complete questions or short paragraphs.
  - Include what you're trying to understand, not just keywords.

  ✅ Example of an effective query:
  {
      "query": "What is Potpie.ai? Which websites have reviewed it recently and what features are highlighted by partners or reviewers?"
  }

  📦 Returns:
  - 'content': Rich, paragraph-style answer based on recent web results.
  - 'citations': List of source URLs referenced.
  - 'success': Whether the search and response were successful.
  """,
        args_schema=WebSearchToolInput,
    )
