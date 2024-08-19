from typing import Type
from langchain.tools import BaseTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel

class WikipediaInput(BaseModel):
    query: str

class WikipediaTool(BaseTool):
    name = "Wikipedia"
    description = "Fetch information from Wikipedia. Use this for factual queries about various topics."
    args_schema: Type[BaseModel] = WikipediaInput

    def _run(self, query: str) -> str:
        try:
            query_run = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            result = query_run.run(query)
            # Ensure the result is relevant to the query
            if query.lower() not in result.lower():
                return f"Relevant information not found directly related to '{query}'."
            return result
        except Exception as e:
            return f"An error occurred while fetching information from Wikipedia: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
