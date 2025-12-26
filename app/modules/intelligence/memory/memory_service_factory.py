import os
import logging
from typing import Optional
from app.modules.intelligence.memory.memory_interface import MemoryInterface
from app.modules.intelligence.memory.letta_service import LettaService

logger = logging.getLogger(__name__)


class MemoryServiceFactory:
    """Factory for creating memory service instances"""

    @staticmethod
    def create(llm_config: Optional[dict] = None) -> MemoryInterface:
        """
        Create a memory service instance

        Args:
            llm_config: Optional LLM configuration for Letta

        Returns:
            MemoryInterface instance (uses Letta)
        """
        if llm_config is None:
            llm_config = MemoryServiceFactory.get_default_llm_config()
            logger.info(
                f"Using default LLM config for Letta: model={llm_config.get('model')}"
            )

        return LettaService(llm_config=llm_config)

    @staticmethod
    def get_default_llm_config() -> dict:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning(
                "OPENAI_API_KEY not found in environment - Letta may not work"
            )

        return {
            "model": os.getenv("LETTA_MODEL", "openai/gpt-4o-mini"),
            "embedding": os.getenv("LETTA_EMBEDDING", "openai/text-embedding-ada-002"),
        }
