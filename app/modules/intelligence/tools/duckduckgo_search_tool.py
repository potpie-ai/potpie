from typing import Type
from langchain.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel

class DuckDuckGoInput(BaseModel):
    query: str

class DuckDuckGoTool(BaseTool):
    name = "DuckDuckGoSearch"
    description = "Searches for information using DuckDuckGo."
    args_schema: Type[BaseModel] = DuckDuckGoInput

    def _run(self, query: str) -> str:
        try:
            search_run = DuckDuckGoSearchRun()
            result = search_run.run(query)
            # Return only the first few lines or relevant section to avoid irrelevant results
            return result.split('\n')[0]  # You can refine this further based on your needs
        except Exception as e:
            return f"An error occurred while searching DuckDuckGo: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
