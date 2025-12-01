import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from mem0 import Memory
from app.modules.intelligence.memory.memory_interface import (
    MemoryInterface,
    MemorySearchResponse,
    MemorySearchResult,
)
from app.modules.intelligence.memory.categorization import (
    categorize_memory,
    USER_LEVEL_CATEGORIES,
    PROJECT_LEVEL_CATEGORIES,
)

logger = logging.getLogger(__name__)


class Mem0Service(MemoryInterface):
    """Mem0-based implementation with automatic project/user level classification"""
    
    def __init__(
        self,
        vector_store: str = "neo4j",
        llm_config: Optional[Dict[str, Any]] = None,
        enable_categories: bool = True
    ):
        """
        Initialize Mem0 service with automatic categorization
        
        Args:
            vector_store: "neo4j" or "chromadb"
            llm_config: Optional LLM configuration for mem0
            enable_categories: Whether to enable automatic categorization
        """
        self.vector_store = vector_store
        self.enable_categories = enable_categories
        self.memory = self._create_memory_instance(llm_config)
        logger.info(f"Initialized Mem0Service with {vector_store} backend and categories={enable_categories}")
    
    def _create_memory_instance(self, llm_config: Optional[Dict[str, Any]]) -> Memory:
        """Create and configure mem0 Memory instance with categories"""
        config = {
            "version": "v1.1",
        }
        
        # Configure vector store
        if self.vector_store in ["neo4j", "chromadb"]:
            config["vector_store"] = {
                "provider": "chroma",
                "config": {
                    "collection_name": "mem0_memories",
                    "path": str(Path(__file__).parent / "chroma_db"),
                },
            }
            logger.info("Configured mem0 with ChromaDB")
        else:
            raise ValueError(f"Unsupported vector store: {self.vector_store}")
        
        # Add LLM config if provided
        if llm_config:
            config["llm"] = llm_config
        
        # Note: custom_categories is not supported in OSS mem0ai SDK
        # It's only available in Mem0 Cloud Platform with MemoryClient
        # We'll implement our own categorization logic instead
        logger.info(
            "📝 Mem0 OSS SDK initialized. Custom categorization will be done post-processing."
        )
        
        return Memory.from_config(config_dict=config)
    
    def _get_agent_id(self, project_id: Optional[str] = None) -> str:
        """
        Generate agent_id based on scope
        - If project_id provided: returns "project_{project_id}"
        - If no project_id: returns "user_global" for user-level preferences
        """
        if project_id:
            return f"project_{project_id}"
        return "user_global"
    
    async def search(
        self,
        query: str,
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 5,
        include_user_preferences: bool = True
    ) -> MemorySearchResponse:
        """
        Search for relevant memories with automatic scope detection
        
        Args:
            query: Search query
            user_id: User identifier
            project_id: Optional project identifier for project-scoped search
            limit: Max results to return
            include_user_preferences: If True and project_id provided, also searches user-level prefs
        
        Returns:
            MemorySearchResponse with results from appropriate scope(s)
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            
            results_combined = []
            
            # If project_id provided, search project-specific memories first
            if project_id:
                agent_id = self._get_agent_id(project_id)
                logger.info(
                    f"[mem0 search] Project-scoped search: query='{query[:50]}...', "
                    f"user_id={user_id}, agent_id={agent_id}"
                )
                
                project_result = await loop.run_in_executor(
                    None,
                    lambda: self.memory.search(
                        query, 
                        user_id=user_id, 
                        agent_id=agent_id,
                        limit=limit
                    )
                )
                
                project_memories = project_result.get("results", []) if isinstance(project_result, dict) else []
                logger.debug(f"Found {len(project_memories)} project-specific memories")
                
                # Add scope metadata to results
                for mem in project_memories:
                    mem["scope"] = "project"
                    mem["project_id"] = project_id
                
                results_combined.extend(project_memories)
            
            # Search user-level preferences (always, or when no project_id)
            if not project_id or include_user_preferences:
                agent_id = self._get_agent_id(None)  # "user_global"
                logger.info(
                    f"[mem0 search] User-scoped search: query='{query[:50]}...', "
                    f"user_id={user_id}, agent_id={agent_id}"
                )
                
                # Adjust limit if we already have project results
                user_limit = limit if not project_id else max(3, limit - len(results_combined))
                
                user_result = await loop.run_in_executor(
                    None,
                    lambda: self.memory.search(
                        query,
                        user_id=user_id,
                        agent_id=agent_id,
                        limit=user_limit
                    )
                )
                
                user_memories = user_result.get("results", []) if isinstance(user_result, dict) else []
                logger.debug(f"Found {len(user_memories)} user-level preferences")
                
                # Add scope metadata
                for mem in user_memories:
                    mem["scope"] = "user"
                
                results_combined.extend(user_memories)
            
            # Transform to our interface
            memories = [
                MemorySearchResult(
                    memory=entry.get("memory", ""),
                    metadata={
                        **entry.get("metadata", {}),
                        "scope": entry.get("scope"),
                        "project_id": entry.get("project_id"),
                        "category": entry.get("category"),  # Auto-assigned by Mem0
                    },
                    score=entry.get("score")
                )
                for entry in results_combined[:limit]  # Apply final limit
            ]
            
            logger.info(
                f"[mem0 search] Returning {len(memories)} total memories "
                f"(project={sum(1 for m in memories if m.metadata.get('scope') == 'project')}, "
                f"user={sum(1 for m in memories if m.metadata.get('scope') == 'user')})"
            )
            
            return MemorySearchResponse(
                results=memories,
                total=len(memories)
            )
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}", exc_info=True)
            return MemorySearchResponse(results=[], total=0)
    
    async def get_all_for_user(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> MemorySearchResponse:
        """
        Get all memories for a user, optionally filtered by project
        
        Args:
            user_id: User identifier
            project_id: Optional project filter
            limit: Optional limit on results
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            
            agent_id = self._get_agent_id(project_id) if project_id else None
            
            logger.info(
                f"[mem0 get_all] user_id={user_id}, project_id={project_id}, "
                f"agent_id={agent_id}, limit={limit}"
            )
            
            # Build kwargs for get_all
            kwargs = {"user_id": user_id}
            if agent_id:
                kwargs["agent_id"] = agent_id
            
            result = await loop.run_in_executor(
                None,
                lambda: self.memory.get_all(**kwargs)
            )
            
            # Extract results
            if isinstance(result, dict) and "results" in result:
                raw_results = result["results"]
            elif isinstance(result, list):
                raw_results = result
            else:
                raw_results = []
            
            # Apply limit if specified
            if limit and len(raw_results) > limit:
                raw_results = raw_results[:limit]
            
            # Transform to our interface
            memories = [
                MemorySearchResult(
                    memory=entry.get("memory", entry.get("text", "")),
                    metadata={
                        **entry.get("metadata", {}),
                        "category": entry.get("category"),  # Auto-assigned category
                    },
                    score=entry.get("score")
                )
                for entry in raw_results
            ]
            
            logger.info(f"[mem0 get_all] Returning {len(memories)} memories")
            
            return MemorySearchResponse(
                results=memories,
                total=len(memories)
            )
            
        except Exception as e:
            logger.error(f"Error getting all memories: {e}", exc_info=True)
            return MemorySearchResponse(results=[], total=0)
    
    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_scope: bool = True
    ) -> Dict[str, Any]:
        """
        Add conversation to memory with automatic categorization and scope detection
        
        Args:
            messages: Conversation messages
            user_id: User identifier
            project_id: Optional project identifier for context
            metadata: Optional additional metadata
            auto_scope: If True, Mem0's category determines scope (user vs project)
                       If False, project_id presence determines scope
        
        Note: With auto_scope=True, Mem0 analyzes content and decides:
              - User-level categories (coding_style, tool_preferences, etc.) → stored globally
              - Project-level categories (tech_stack, database, etc.) → stored per-project
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            
            # First, let Mem0 extract and categorize memories
            # We do a temporary add to get the categories, then store appropriately
            if auto_scope and project_id:
                # Step 1: Add with project context to get categorization
                temp_kwargs = {
                    "user_id": user_id,
                    "agent_id": self._get_agent_id(project_id),
                }
                
                if metadata:
                    temp_kwargs["metadata"] = {**metadata, "project_id": project_id}
                
                result = await loop.run_in_executor(
                    None,
                    lambda: self.memory.add(messages, **temp_kwargs)
                )
                
                # Step 2: Check categories and re-store user-level ones globally
                stored_memories = result.get("results", []) if isinstance(result, dict) else []
                
                user_level_categories = USER_LEVEL_CATEGORIES
                project_level_categories = PROJECT_LEVEL_CATEGORIES
                
                # Categorize memories using our own logic (OSS mem0 doesn't support custom categories)
                memories_to_relocate = []
                for mem in stored_memories:
                    if isinstance(mem, dict):
                        memory_text = mem.get("memory", mem.get("text", ""))
                        memory_id = mem.get("id")
                        
                        # Apply our own categorization
                        category = categorize_memory(memory_text)
                        
                        # Store category in metadata (update via mem0's update method would be complex,
                        # so we'll just log it and rely on metadata passed during search)
                        logger.info(
                            f"[mem0 add] Memory categorized: category='{category}', "
                            f"memory='{memory_text[:60]}...', memory_id={memory_id}"
                        )
                        
                        # If it's a user-level category, we need to also store it globally
                        if category in user_level_categories and memory_id:
                            memories_to_relocate.append((mem, category))
                            logger.info(
                                f"[mem0 add] Detected user-level preference (category={category}), "
                                f"will also store globally"
                            )
                
                # Step 3: Also store user-level preferences globally
                for mem, category in memories_to_relocate:
                    memory_text = mem.get("memory", "")
                    if memory_text:
                        # Store in user-global scope as well
                        user_global_kwargs = {
                            "user_id": user_id,
                            "agent_id": "user_global",
                        }
                        
                        user_metadata = {
                            "scope": "user",
                            "category": category,  # Add our categorization
                            "also_in_project": project_id,
                            **(metadata or {})
                        }
                        
                        if user_metadata:
                            user_global_kwargs["metadata"] = user_metadata
                        
                        await loop.run_in_executor(
                            None,
                            lambda: self.memory.add(
                                [{"role": "assistant", "content": memory_text}],
                                **user_global_kwargs
                            )
                        )
                        logger.info(
                            f"[mem0 add] Stored user-level preference globally: {memory_text[:60]}..."
                        )
                
                # Log what was stored with our categorization
                categories_assigned = []
                for mem in stored_memories:
                    if isinstance(mem, dict):
                        memory_text = mem.get("memory", mem.get("text", ""))
                        cat = categorize_memory(memory_text)
                        categories_assigned.append(cat)
                
                logger.info(
                    f"[mem0 add] ✅ CATEGORIZATION COMPLETE: Stored {len(stored_memories)} memories: "
                    f"project_scope={sum(1 for c in categories_assigned if c in project_level_categories)}, "
                    f"user_scope={sum(1 for c in categories_assigned if c in user_level_categories)}, "
                    f"uncategorized={sum(1 for c in categories_assigned if c == 'uncategorized')}, "
                    f"categories={categories_assigned}"
                )
                
                return result
            
            else:
                # Original behavior: scope determined by project_id presence
                agent_id = self._get_agent_id(project_id)
                scope = "project" if project_id else "user"
                
                logger.info(
                    f"[mem0 add] Adding memories (manual scope): user_id={user_id}, "
                    f"project_id={project_id}, agent_id={agent_id}, scope={scope}, "
                    f"messages_count={len(messages)}"
                )
                
                # Build metadata with scope information
                combined_metadata = {
                    "scope": scope,
                    **({"project_id": project_id} if project_id else {}),
                    **(metadata or {})
                }
                
                # Build kwargs for add
                add_kwargs = {
                    "user_id": user_id,
                    "agent_id": agent_id,
                }
                
                # Only add metadata if it has content
                if combined_metadata:
                    add_kwargs["metadata"] = combined_metadata
                
                result = await loop.run_in_executor(
                    None,
                    lambda: self.memory.add(messages, **add_kwargs)
                )
                
                # Log what was stored with our categorization
                stored_memories = result.get("results", []) if isinstance(result, dict) else []
                categories_assigned = []
                for mem in stored_memories:
                    if isinstance(mem, dict):
                        memory_text = mem.get("memory", mem.get("text", ""))
                        cat = categorize_memory(memory_text)
                        categories_assigned.append(cat)
                        logger.debug(
                            f"[mem0 add] Memory categorized: category='{cat}', "
                            f"memory='{memory_text[:60]}...'"
                        )
                
                logger.info(
                    f"[mem0 add] ✅ CATEGORIZATION COMPLETE: Stored {len(stored_memories)} memories with scope={scope}, "
                    f"categories={categories_assigned}"
                )
                
                return result
            
        except Exception as e:
            logger.error(f"Error adding messages to memory: {e}", exc_info=True)
            raise
    
    async def delete(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Delete memories for a user, optionally scoped to a project
        
        Args:
            user_id: User identifier
            project_id: Optional project scope
            memory_ids: Optional specific memory IDs to delete
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            
            if memory_ids:
                # Delete specific memories
                for memory_id in memory_ids:
                    await loop.run_in_executor(
                        None,
                        self.memory.delete,
                        memory_id
                    )
                logger.info(f"Deleted {len(memory_ids)} specific memories")
            else:
                # Delete all memories for user (and optionally project)
                agent_id = self._get_agent_id(project_id) if project_id else None
                
                kwargs = {"user_id": user_id}
                if agent_id:
                    kwargs["agent_id"] = agent_id
                
                await loop.run_in_executor(
                    None,
                    lambda: self.memory.delete_all(**kwargs)
                )
                logger.info(
                    f"Deleted all memories for user_id={user_id}, "
                    f"project_id={project_id or 'all'}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memories: {e}", exc_info=True)
            return False
    
    async def get_categories_distribution(
        self,
        user_id: str,
        project_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Get distribution of memories across categories
        Useful for debugging and understanding what's being stored where
        """
        try:
            all_memories = await self.get_all_for_user(user_id, project_id)
            
            category_counts = {}
            for memory in all_memories.results:
                category = memory.metadata.get("category", "uncategorized")
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return category_counts
            
        except Exception as e:
            logger.error(f"Error getting category distribution: {e}", exc_info=True)
            return {}
    
    def close(self) -> None:
        """Close mem0 connections"""
        logger.info("Closed Mem0Service connections")