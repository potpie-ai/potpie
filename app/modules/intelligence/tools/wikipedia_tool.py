from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from app.modules.intelligence.tools.tool_base import ToolBase

class WikipediaTool(ToolBase):
    def __init__(self):
        self.name = "Wikipedia"
        self.description = "Fetch information from Wikipedia. Use this for factual queries about various topics."
        self.tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    def run(self, query: str) -> str:
        return self.tool.run(query)