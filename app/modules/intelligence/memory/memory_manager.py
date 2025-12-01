import logging
from typing import Optional, List
from app.modules.intelligence.memory.memory_service_factory import MemoryServiceFactory
from app.modules.intelligence.memory.memory_interface import MemoryInterface

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manager for user preference memory operations"""
    
    def __init__(self, memory_service: Optional[MemoryInterface] = None):
        """
        Initialize memory manager
        
        Args:
            memory_service: Optional memory service instance (creates default if None)
        """
        self.memory_service = memory_service or MemoryServiceFactory.create()
        logger.info("Initialized MemoryManager")
    
    async def get_user_preferences_block(
        self,
        user_id: str,
        query: str,
        project_id: Optional[str] = None,
        limit: int = 5
    ) -> str:
        """
        Retrieve user preferences and format as a block for system prompt injection
        
        Args:
            user_id: User identifier
            query: Current user query (for semantic search)
            project_id: Optional project identifier for project-scoped search
            limit: Maximum number of preferences to retrieve
        
        Returns:
            Formatted preferences block string (empty if no preferences found)
        """
        try:
            logger.info(
                f"[memory manager ] Searching for preferences before injection: "
                f"user_id={user_id}, project_id={project_id}, query='{query[:100]}...', limit={limit}"
            )
            
            search_response = await self.memory_service.search(
                query=query,
                user_id=user_id,
                project_id=project_id,
                limit=limit
            )
            
            if not search_response.results:
                logger.info(
                    f"[memory manager ] No preferences found for injection: "
                    f"user_id={user_id}, project_id={project_id}, query='{query[:100]}...'"
                )
                return ""
            
            # Format preferences as a block
            preferences = [
                f"- {result.memory}"
                for result in search_response.results
            ]
            
            # Log what we're injecting
            preferences_list = [result.memory for result in search_response.results]
            logger.info(
                f"[memory manager ] Injecting preferences from mem0: "
                f"user_id={user_id}, project_id={project_id}, found={len(preferences_list)} preferences, "
                f"preferences={preferences_list}"
            )
            
            preferences_block = f"""
USER PREFERENCES:
{chr(10).join(preferences)}

Please consider these preferences when responding to the user.
"""
            return preferences_block.strip()
        except Exception as e:
            logger.error(
                f"[memory manager ] Error retrieving preferences for injection: "
                f"user_id={user_id}, project_id={project_id}, query='{query[:100]}...', error={e}",
                exc_info=True
            )
            return ""  # Return empty on error to not break agent execution
    
    async def extract_preferences_async(
        self,
        messages: List[dict],
        user_id: str,
        project_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """
        Extract preferences from conversation messages (async wrapper)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            user_id: User identifier
            project_id: Optional project identifier for project-scoped storage
            metadata: Optional metadata
        
        Returns:
            Result from memory service
        """
        logger.info(
            f"[memory manager ] Starting memory extraction: "
            f"user_id={user_id}, project_id={project_id}, messages_count={len(messages)}, metadata={metadata}"
        )
        
        try:
            result = await self.memory_service.add(
                messages=messages,
                user_id=user_id,
                project_id=project_id,
                metadata=metadata,
                auto_scope=True  # Let Mem0's categories determine scope
            )
            
            # Check if extraction was successful and log what was extracted
            # mem0 returns results in "results" key, not "memories"
            # Categories are assigned by our custom logic
            from app.modules.intelligence.memory.categorization import categorize_memory
            
            extracted_memories = result.get("results", []) if isinstance(result, dict) else []
            extracted_count = len(extracted_memories) if isinstance(extracted_memories, list) else 0
            extracted_content = []
            categories_assigned = []
            if isinstance(extracted_memories, list):
                for mem in extracted_memories:
                    if isinstance(mem, dict):
                        memory_text = mem.get("memory", mem.get("text", str(mem)))
                        # Use our categorization logic
                        category = categorize_memory(memory_text)
                        extracted_content.append(memory_text)
                        categories_assigned.append(category)
                    else:
                        extracted_content.append(str(mem))
                        categories_assigned.append("uncategorized")
            
            if extracted_count > 0:
                logger.info(
                    f"[memory manager ] Memory extraction SUCCESSFUL: "
                    f"user_id={user_id}, project_id={project_id}, extracted_count={extracted_count}, "
                    f"categories={categories_assigned}, "
                    f"extracted_memories={extracted_content}"
                )
            else:
                logger.info(
                    f"[memory manager ] Memory extraction completed but no memories extracted: "
                    f"user_id={user_id}, project_id={project_id}, result={result}"
                )
            
            return result
        except Exception as e:
            logger.error(
                f"[memory manager ] Memory extraction FAILED: "
                f"user_id={user_id}, project_id={project_id}, error={e}",
                exc_info=True
            )
            raise
    
    async def get_all_memories(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> dict:
        """
        Get all memories for a user
        
        Args:
            user_id: User identifier
            project_id: Optional project identifier for project-scoped retrieval
            limit: Optional limit on number of memories to return
        
        Returns:
            MemorySearchResponse with all memories
        """
        logger.info(
            f"[memory manager ] Getting all memories: "
            f"user_id={user_id}, project_id={project_id}, limit={limit}"
        )
        
        try:
            search_response = await self.memory_service.get_all_for_user(
                user_id=user_id,
                project_id=project_id,
                limit=limit
            )
            
            logger.info(
                f"[memory manager ] Retrieved {len(search_response.results)} memories for user_id={user_id}, project_id={project_id}"
            )
            
            return {
                "memories": [
                    {
                        "memory": result.memory,
                        "metadata": result.metadata,
                        "score": result.score
                    }
                    for result in search_response.results
                ],
                "total": search_response.total
            }
        except Exception as e:
            logger.error(
                f"[memory manager ] Error getting all memories: "
                f"user_id={user_id}, project_id={project_id}, error={e}",
                exc_info=True
            )
            raise
    
    def close(self) -> None:
        """Close memory service connections"""
        self.memory_service.close()

