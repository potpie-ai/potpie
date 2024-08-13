from abc import ABC, abstractmethod

class BaseTool(ABC):

    @abstractmethod
    def execute(self, query: str):
        pass

    @abstractmethod
    def to_langchain_tool(self):
        pass