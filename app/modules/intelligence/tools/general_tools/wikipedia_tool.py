from typing import Type

from langchain.tools import BaseTool as LangchainToolBaseModel
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel


class WikipediaInput(BaseModel):
    query: str


class WikipediaTool(LangchainToolBaseModel):
    name = "Wikipedia"
    description = "Fetch factual information from Wikipedia on various topics."
    args_schema: Type[BaseModel] = WikipediaInput

    def _run(self, query: str) -> str:
        try:
            query_run = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            result = query_run.run(query)
            return result.strip()  # Clean up any leading/trailing whitespace
        except Exception as e:
            return f"Error fetching Wikipedia information: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
