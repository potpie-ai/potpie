from abc import ABC, abstractmethod

class ProviderBase(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def get_provider_info(self):
        pass

