from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel


class TriggerHash(BaseModel):
    hash: str
    user_id: str
    node_type: str
    created_at: str
    is_active: bool = True
    deactivated_at: Optional[str] = None
    workflow_id: Optional[str] = None


class TriggerHashStore(ABC):
    """Abstract adapter for trigger hash store."""

    @abstractmethod
    async def create_trigger_hash(self, user_id: str, node_type: str) -> TriggerHash:
        """Create a new trigger hash for a user and node type."""
        pass

    @abstractmethod
    async def create_trigger_hash_with_hash(
        self, user_id: str, node_type: str, hash_value: str
    ) -> TriggerHash:
        """Create a new trigger hash with a specific hash value for a user and node type."""
        pass

    @abstractmethod
    async def get_trigger_hash(
        self, user_id: str, node_type: str
    ) -> Optional[TriggerHash]:
        """Get existing trigger hash for a user and node type."""
        pass

    @abstractmethod
    async def get_trigger_info(self, trigger_hash: str) -> Optional[TriggerHash]:
        """Get trigger information by hash."""
        pass

    @abstractmethod
    async def deactivate_trigger_hash(self, trigger_hash: str) -> bool:
        """Deactivate a trigger hash."""
        pass

    @abstractmethod
    async def list_user_trigger_hashes(self, user_id: str) -> List[TriggerHash]:
        """List all trigger hashes for a user."""
        pass

    @abstractmethod
    async def update_workflow_id(self, trigger_hash: str, workflow_id: str) -> bool:
        """Update the workflow_id for a given trigger hash."""
        pass
