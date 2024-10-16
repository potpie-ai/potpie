from typing import Any, List

from pydantic import BaseModel, Field, field_validator


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str