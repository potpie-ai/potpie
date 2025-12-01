from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class MemorySearchResult(BaseModel):
    """Result from memory search"""
    memory: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class MemorySearchResponse(BaseModel):
    """Response from memory search"""
    results: List[MemorySearchResult]
    total: int


class MemoryInterface(ABC):
    """Abstract interface for long-term memory operations"""
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 5,
        include_user_preferences: bool = True
    ) -> MemorySearchResponse:
        """Search for relevant memories/preferences for a user"""
        pass
    
    @abstractmethod
    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_scope: bool = True
    ) -> Dict[str, Any]:
        """Add conversation messages to memory for preference extraction"""
        pass
    
    @abstractmethod
    async def get_all_for_user(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> MemorySearchResponse:
        """Get all memories for a specific user"""
        pass
    
    @abstractmethod
    async def delete(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None
    ) -> bool:
        """Delete memories for a user (all if memory_ids is None)"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close connections and cleanup resources"""
        pass

