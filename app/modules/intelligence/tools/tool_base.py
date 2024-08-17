from abc import ABC, abstractmethod

class ToolBase(ABC):
    @abstractmethod
    def __init__(self, name: str, description: str):
        # self.name = ""
        # self.description = ""
        # self.tool = 
        pass

    @abstractmethod
    def run(self, input_data: str) -> str:
        pass