"""Core data models for events in executions"""

from abc import ABC, abstractmethod
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from app.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


class Event(BaseModel):
    id: str
    time: datetime
    source: str
    source_type: str
    payload: Dict[str, Any]
    headers: Optional[Dict[str, str]] = None


class EventStore(ABC):
    """Abstract adapter for Event store."""

    @abstractmethod
    async def save_event(
        self,
        source: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Event:
        """Save an incomming event"""
        pass

    @abstractmethod
    async def get_event_by_id(self, id: str) -> Event:
        """Save an incomming event"""
        pass
