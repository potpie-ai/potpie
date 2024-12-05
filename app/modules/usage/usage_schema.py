from pydantic import BaseModel
from typing import Dict

class UsageResponse(BaseModel):
    total_human_messages: int
    agent_message_counts: Dict[str, int] 