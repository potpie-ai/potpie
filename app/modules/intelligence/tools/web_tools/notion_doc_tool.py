import os
import asyncio
import logging
import re
from typing import Optional

from notion_client import Client
from notion_to_md import NotionToMarkdown
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session
from app.modules.key_management.secret_manager import SecretStorageHandler


class NotionDocToolInput(BaseModel):
    url: str = Field(
        description="The URL of the notion document to extract content from"
    )


class NotionDocToolOutput(BaseModel):
    success: bool = Field(
        description="Boolean True | False on whether answer generation was sucessful"
    )
    content: str = Field(description="Content of the Notion document in md format")


class NotionDocTool:
    name = """Notion Doc Tool"""
    description = """ Extracts the content of a Notion document and returns it in a structured format"""

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.output_schema = NotionDocToolOutput

    async def arun(self, url: str):
        try:
            page_id = self._extract_page_id(url)
            notion_key = None

            has_key = await SecretStorageHandler.check_secret_exists(
                service="notion",
                customer_id=self.user_id,
                service_type="integration",
                db=self.sql_db,
            )
            
            if not has_key:
                notion_key = os.getenv("NOTION_API_KEY")
            else:
                notion_key = await SecretStorageHandler.get_secret(
                    service="notion",
                    customer_id=self.user_id,
                    service_type="integration",
                    db=self.sql_db,
                )

            if not notion_key:
                return NotionDocToolOutput(
                    success=False,
                    content="Ask user to head to the Key Management screen and add their Notion API Key in order to use Notion operations",
                )

            notion = Client(auth=notion_key)
            n2m = NotionToMarkdown(notion)
            md_blocks = n2m.page_to_markdown(page_id)
            md_str = n2m.to_markdown_string(md_blocks).get("parent")

            return NotionDocToolOutput(success=True, content=md_str)
        except Exception as e:
            logging.exception(f"Error extracting notion doc: {str(e)}")
            return NotionDocToolOutput(success=False, content="Tool Call Error")

    def run(self, url: str):
        return asyncio.run(self.arun(url))

    def _extract_page_id(self, url: str) -> str:

        if "notion.so/" in url:
            match = re.search(
                r"([a-f0-9]{32}|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
                url,
            )
            if match:
                return match.group(1)
            else:
                raise ValueError("Error Parsing Notion URL")
        else:
            raise ValueError("Invalid Notion URL")


def notion_doc_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    tool_instance = NotionDocTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Notion Doc Tool",
        description="""Extracts content from Notion documents and converts it to Markdown format.

        ## Purpose
        This tool helps you access and understand content from a user's Notion documents to provide more informed and contextually relevant responses.

        ## When to Use
        - When the user references specific project information you need to understand
        - To learn about unfamiliar technologies, terminology, or methods mentioned in queries
        - To understand user-specific deadlines, progress metrics, or quality standards
        - To gain deeper context about the user's situation for more relevant answers

        ## How to Process the Content
        1. Analyze the document holistically, not just individual keywords
        2. Identify relationships between different sections, elements, and data points
        3. Process tables, lists, and structured content appropriately
        4. Form a comprehensive understanding of the document's purpose and meaning

        ## Critical Guidelines
        - Only use factual information present in the document
        - Do not hallucinate or infer information not explicitly stated
        - Maintain accuracy in your interpretation of the content
        - Continue processing until you have sufficient understanding to respond confidently
        - If content is ambiguous, acknowledge limitations rather than making assumptions

        ## Example Input
        ```json
        {
        "url": "https://www.notion.so/SmartO-Questions-19e2d57d0874803e957ddca89a8702c5"
        }
        """,
        args_schema=NotionDocToolInput,
    )
