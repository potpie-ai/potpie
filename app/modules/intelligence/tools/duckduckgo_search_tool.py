from typing import Type
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool as LangchainToolBaseModel
from pydantic import BaseModel
class DuckDuckGoInput(BaseModel):
    query: str

class DuckDuckGoTool(LangchainToolBaseModel):
    name = "DuckDuckGoSearch"
    description = "Searches for information using DuckDuckGo."
    args_schema: Type[BaseModel] = DuckDuckGoInput

    def _run(self, query: str) -> str:
        try:
            search_run = DuckDuckGoSearchRun()
            result = search_run.run(query)
            return str(result.split('\n')[0])  # Ensure the result is a string
        except Exception as e:
            return f"An error occurred while searching DuckDuckGo: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
