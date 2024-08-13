from langchain_community.tools import DuckDuckGoSearchResults

class DuckDuckGoSearchTool:
    def __init__(self):
        self.tool = DuckDuckGoSearchResults()

    def execute(self, query: str):
        return self.tool.run(query)
