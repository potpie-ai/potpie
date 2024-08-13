from abc import ABC, abstractmethod

class BaseAgent(ABC):
    
    @abstractmethod
    def load_tools(self):
        pass
    
    @abstractmethod
    def run(self, input_data: str):
        pass
