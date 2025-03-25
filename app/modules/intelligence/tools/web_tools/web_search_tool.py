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
    query: str = Field(description="Query that needs online searching and reasoning. This needs to be a proper formed question and paragraph, not just the exact term. Make the query as informative as possible.")

class WebSearchToolOutput(BaseModel):
    success: bool = Field(description="Boolean True | False on whether answer generation was sucessful")
    content : str = Field(description="Answer by the LLM Model")
    citations : list[str] = Field(description="List of website and citations reference for in the answer")

class WebSearchTool:
    name = """Web Seach Tool"""
    description = """ Searches the web for any information and then gives you the answer"""

    def __init__(self, sql_db: Session, user_id:str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.temperature = 0.3
        self.max_tokens = 12000
        self.output_schema = WebSearchToolOutput

    async def arun(self, query:str):
        return await asyncio.to_thread(self.run, query)
    
    def run (self, query:str):
        try:
            response =  self._make_llm_call(query)
            if not response:
                response = {
                    "success"  : False,
                    "response" : "Tool Call Failed",
                    "citations": [],
                }
            return response
        except Exception as e:
            logging.exception(f"Error {str(e)}")
            response = {
                    "success"  : False,
                    "response" : "Tool Call Error",
                    "citations": [],
                }

    def _make_llm_call(self, query: str) -> Dict[str, Any]:
        try:
            messages = [{"role": "user", "content": query}]
            provider_service = ProviderService(self.sql_db, self.user_id)
            extra_params , _ = provider_service.get_extra_params_and_headers("openrouter")
            client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)
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
                "citations": response.citations
            }
        except Exception as e:
            logging.exception(f"Error in _make_llm_call: {str(e)}")
            return {
                "success": False,
                "content": f"LLM call failed: {str(e)}",
                "citations": []
            }

def web_search_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    
    tool_instance = WebSearchTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Web Search Tool",
        description="""Searches the internet for current information and provides comprehensive results with reliable citations.

        Given detailed, specific queries rather than broad topics. Include key context, 
        specific aspects you're interested in, and any time-sensitive information needs.

        Examples of effective queries:
        {
            "query": "Tell me what is Potpie.ai about? Which websites have covered its functionality and who are its partners?"
        }

        Returns a dictionary containing:
        - 'content': A comprehensive answer based on current web information
        - 'citations': List of specific sources used in the answer for verification
        - 'success': Boolean indicating whether the search was successful
        """,
        args_schema=WebSearchToolInput,
    )
