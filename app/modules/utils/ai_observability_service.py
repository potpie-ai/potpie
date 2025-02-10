import logging
import agentops
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AiObservabilityService(ABC):
    """Interface for setting up AI observability"""

    @abstractmethod
    def start_session(self):
        """
        Start a new session in the observability platform.
        """
        pass

    @abstractmethod
    def end_session(self, status: str = "success"):
        """
        End the session in the observability platform.
        """
        pass


class MockAiObservabilityService(AiObservabilityService):
    """Mock observability client for testing purposes and development"""

    def __init__(self):
        pass

    def start_session(self):
        logger.info("Starting new session for AI observability")

    def end_session(self, status: str = "success"):
        logger.info(f"Ending session for AI observability with status = {status}")


class AgentopsAiObservabilityService(AiObservabilityService):
    """Agentops observability client for capturing observability events"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def start_session(
        self,
    ):
        agentops.init(api_key=self.api_key)

    def end_session(self, status: str = "success"):
        agentops.end_session(status)
