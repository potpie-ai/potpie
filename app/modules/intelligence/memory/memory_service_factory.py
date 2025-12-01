import os
import logging
from typing import Optional
from app.modules.intelligence.memory.memory_interface import MemoryInterface
from app.modules.intelligence.memory.mem0_service import Mem0Service

logger = logging.getLogger(__name__)


class MemoryServiceFactory:
    """Factory for creating memory service instances"""
    
    @staticmethod
    def create(
        provider: Optional[str] = None,
        llm_config: Optional[dict] = None
    ) -> MemoryInterface:
        """
        Create a memory service instance
        
        Args:
            provider: "mem0_neo4j", "mem0_chromadb", or None (auto-detect from env)
            llm_config: Optional LLM configuration for mem0
        
        Returns:
            MemoryInterface instance
        """
        # Get provider from env or parameter
        if not provider:
            provider = os.getenv("MEMORY_PROVIDER", "mem0_neo4j")
        
        # Use default LLM config if none provided (required for categorization)
        if llm_config is None:
            llm_config = MemoryServiceFactory.get_default_llm_config()
            logger.info(f"Using default LLM config for mem0 categorization: model={llm_config['config']['model']}")
        
        if provider == "mem0_neo4j":
            return Mem0Service(vector_store="neo4j", llm_config=llm_config)
        elif provider == "mem0_chromadb":
            return Mem0Service(vector_store="chromadb", llm_config=llm_config)
        else:
            raise ValueError(f"Unsupported memory provider: {provider}")
    
    @staticmethod
    def get_default_llm_config() -> dict:
        """Get default LLM configuration for mem0"""
        # Use the same LLM provider as the main system
        # This can be customized based on your needs
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment - mem0 categorization may not work")
        
        return {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",  # Lightweight model for preference extraction
                "api_key": openai_api_key,
            }
        }

