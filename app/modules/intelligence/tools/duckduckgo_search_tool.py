from langchain_community.tools import DuckDuckGoSearchRun
from app.modules.intelligence.tools.tool_base import ToolBase

class DuckDuckGoTool(ToolBase):
    def __init__(self):
        self.name = "DuckDuckGoSearch"
        self.description = "Searches for information using DuckDuckGo."
        self.tool = DuckDuckGoSearchRun()

    def run(self, query: str) -> str:
        try:
            return self.tool.run(query)
        except Exception as e:
            return f"An error occurred while searching DuckDuckGo: {str(e)}"