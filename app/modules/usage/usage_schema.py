from typing import Dict

from pydantic import BaseModel


class UsageResponse(BaseModel):
    total_human_messages: int
    agent_message_counts: Dict[str, int]
